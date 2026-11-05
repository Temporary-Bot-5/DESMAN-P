"""Process locks must serialize writes on Windows as well as POSIX."""

from __future__ import annotations

import errno
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeClock
from desman.config import ConfigManager
from desman.core import filelock

CHECK = """
import sys
from desman.core import filelock
with open(sys.argv[1], 'a+b') as handle:
    try:
        filelock.lock(handle.fileno(), blocking=False)
    except BlockingIOError:
        sys.exit(23)
    filelock.unlock(handle.fileno())
"""


def test_contention_and_explicit_release(tmp_path: Path) -> None:
    path = tmp_path / "slot"
    with path.open("a+b") as handle:
        filelock.lock(handle.fileno())
        busy = subprocess.run([sys.executable, "-c", CHECK, str(path)], timeout=10)
        assert busy.returncode == 23
        filelock.unlock(handle.fileno())
    free = subprocess.run([sys.executable, "-c", CHECK, str(path)], timeout=10)
    assert free.returncode == 0


def test_blocking_waits_until_unlock(tmp_path: Path) -> None:
    path = tmp_path / "slot"
    script = CHECK.replace("blocking=False", "blocking=True")
    with path.open("a+b") as handle:
        filelock.lock(handle.fileno())
        child = subprocess.Popen([sys.executable, "-c", script, str(path)])
        try:
            try:
                child.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise AssertionError("a blocking lock bypassed its holder")
            filelock.unlock(handle.fileno())
            assert child.wait(timeout=10) == 0
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


def test_windows_blocking_retries_and_nonblocking_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    attempts: list[tuple[int, int, int]] = []
    remaining = 250  # more than Windows LK_LOCK's ten-second retry window

    def locking(fd: int, mode: int, size: int) -> None:
        nonlocal remaining
        attempts.append((mode, size, os.lseek(fd, 0, os.SEEK_CUR)))
        if mode == 1 and remaining:
            remaining -= 1
            raise OSError(errno.EACCES, "locked")

    monkeypatch.setattr(filelock, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        filelock,
        "msvcrt",
        SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2),
        raising=False,
    )
    monkeypatch.setattr(filelock, "time", SimpleNamespace(sleep=clock.sleep))
    with (tmp_path / "slot").open("a+b") as handle:
        handle.write(b"data")
        handle.flush()
        with pytest.raises(BlockingIOError):
            filelock.lock(handle.fileno(), blocking=False)
        before = clock()
        filelock.lock(handle.fileno())
        assert (clock() - before).total_seconds() > 10
        filelock.unlock(handle.fileno())
    assert attempts == [(1, 1, 0)] * 251 + [(2, 1, 0)]


def test_config_lock_can_be_taken_again(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    with manager.locked():
        pass
    with (tmp_path / ".config.json.lock").open("a+b") as handle:
        filelock.lock(handle.fileno(), blocking=False)
        try:
            pass
        finally:
            filelock.unlock(handle.fileno())
