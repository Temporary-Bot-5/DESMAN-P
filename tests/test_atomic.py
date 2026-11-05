"""Windows sharing violations must neither lose a write nor retry forever."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import FakeClock
from desman.core import atomic


@pytest.mark.parametrize("platform,failures", [("win32", 3), ("win32", 41), ("linux", 1)])
def test_replace_retries_only_windows_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    platform: str,
    failures: int,
) -> None:
    source, target = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new")
    target.write_bytes(b"old")
    real_replace = os.replace
    attempts = 0

    def replace(src: str | Path, dst: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= failures:
            raise PermissionError("reader holds target")
        real_replace(src, dst)

    monkeypatch.setattr(atomic, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(atomic, "time", SimpleNamespace(sleep=clock.sleep))
    monkeypatch.setattr(os, "replace", replace)
    before = clock()
    if platform == "win32" and failures < 41:
        atomic.replace_file(source, target)
        assert target.read_bytes() == b"new"
        assert attempts == 4
    else:
        with pytest.raises(PermissionError):
            atomic.replace_file(source, target)
        assert target.read_bytes() == b"old"
        assert source.read_bytes() == b"new"
        assert attempts == (41 if platform == "win32" else 1)
    assert (clock() - before).total_seconds() <= 2
