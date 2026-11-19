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
            error = PermissionError("reader holds target")
            monkeypatch.setattr(error, "winerror", 32, raising=False)
            raise error
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


@pytest.mark.parametrize("reader", ["load", "snapshot"])
@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("failure", ["transient", "persistent", "missing"])
def test_config_reads_retry_only_windows_permission_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
    reader: str,
    platform: str,
    failure: str,
) -> None:
    from desman.config import Config, ConfigError, ConfigManager

    manager = ConfigManager(tmp_path / "config.json")
    config = Config(language="de")
    manager.save(config)
    text = manager.config_path.read_text(encoding="utf-8")
    attempts = 0
    real_read = Path.read_text

    def read(path: Path, **kwargs: str) -> str:
        nonlocal attempts
        if path != manager.config_path:
            return real_read(path, **kwargs)
        attempts += 1
        if failure == "missing":
            raise FileNotFoundError(path)
        if failure == "persistent" or attempts <= 3:
            error = PermissionError("replacement in progress")
            monkeypatch.setattr(error, "winerror", 33, raising=False)
            raise error
        return real_read(path, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(atomic, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(atomic, "time", SimpleNamespace(sleep=clock.sleep))
    read_config = manager.load if reader == "load" else manager.snapshot

    if failure == "missing":
        assert read_config() == (Config() if reader == "load" else None)
        assert attempts == 1
    elif platform == "win32" and failure == "transient":
        assert read_config() == (config if reader == "load" else text)
        assert attempts == 4
    else:
        error = ConfigError if reader == "load" else PermissionError
        with pytest.raises(error, match="replacement in progress"):
            read_config()
        assert attempts == (41 if platform == "win32" else 1)
    assert clock.slept == pytest.approx((attempts - 1) * 0.05)


def test_adopting_config_retries_windows_sharing_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    from desman.config import adopt_config

    old, new = tmp_path / "old.json", tmp_path / "new.json"
    payload = b'{\r\n"language": "de"\r\n}'
    old.write_bytes(payload)
    real_read = Path.read_bytes
    attempts = 0

    def read(path: Path) -> bytes:
        nonlocal attempts
        if path == old:
            attempts += 1
            if attempts == 1:
                error = PermissionError("replacement in progress")
                monkeypatch.setattr(error, "winerror", 33, raising=False)
                raise error
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(atomic, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(atomic, "time", SimpleNamespace(sleep=clock.sleep))
    adopt_config(old, new)
    assert attempts == 2
    assert clock.slept == pytest.approx(0.05)
    assert new.read_bytes() == payload
    assert not old.exists()


@pytest.mark.parametrize("winerror", [None, 5, 32, 33, 65])
def test_windows_retries_only_sharing_codes(
    monkeypatch: pytest.MonkeyPatch, clock: FakeClock, winerror: int | None
) -> None:
    error = PermissionError("operation denied")
    if winerror is not None:
        monkeypatch.setattr(error, "winerror", winerror, raising=False)
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise error
        return "ready"

    monkeypatch.setattr(atomic, "sys", SimpleNamespace(platform="win32"))
    if winerror in (5, 32, 33):
        assert atomic.retry_sharing(operation, sleep=clock.sleep) == "ready"
        assert attempts == 2
        assert clock.slept == pytest.approx(0.05)
    else:
        with pytest.raises(PermissionError) as caught:
            atomic.retry_sharing(operation, sleep=clock.sleep)
        assert caught.value is error
        assert attempts == 1
        assert clock.slept == 0
