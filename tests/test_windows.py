"""Windows compatibility, with WinAPI responses simulated on other platforms."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from desman.core import owner
from desman.core.owner import current_owner, owner_is_gone
from desman.llm import hub


class Function:
    def __init__(self, call: Any) -> None:
        self.call = call

    def __call__(self, *args: Any) -> Any:
        return self.call(*args)


@pytest.mark.parametrize(
    ("handle", "error", "success", "code", "gone"),
    [
        (0, 87, False, 0, True),
        (0, 5, False, 0, False),
        (123, 0, True, 259, False),
        (123, 0, True, 0, True),
        (123, 0, False, 0, False),
    ],
)
def test_windows_process_check(
    monkeypatch: pytest.MonkeyPatch,
    handle: int,
    error: int,
    success: bool,
    code: int,
    gone: bool,
) -> None:
    monkeypatch.setattr(owner, "sys", SimpleNamespace(platform="win32"))
    closed: list[int] = []

    def exit_code(h: int, result: Any) -> bool:
        assert h == handle
        result._obj.value = code
        return success

    class Kernel:
        OpenProcess = Function(lambda access, inherit, pid: handle)
        GetExitCodeProcess = Function(exit_code)
        CloseHandle = Function(lambda h: closed.append(h))

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: Kernel(), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)
    assert owner._windows_gone(456) is gone
    assert closed == ([handle] if handle else [])


@pytest.mark.parametrize("success", [True, False])
def test_windows_available_memory(monkeypatch: pytest.MonkeyPatch, success: bool) -> None:
    monkeypatch.setattr(hub, "sys", SimpleNamespace(platform="win32"))

    def memory(status: Any) -> bool:
        assert status._obj.length == ctypes.sizeof(status._obj)
        status._obj.avail_phys = 12_345_678_901
        return success

    class Kernel:
        GlobalMemoryStatusEx = Function(memory)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: Kernel(), raising=False)
    assert hub._windows_ram() == (12_345_678_901 if success else None)


def test_owner_is_gone_only_when_certain() -> None:
    host = current_owner().rpartition(":")[0]
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert owner_is_gone(f"{host}:{child.pid}")  # exited and reaped
    assert not owner_is_gone(current_owner())  # ourselves
    assert not owner_is_gone(f"{host}:{os.getppid()}")  # alive
    if sys.platform != "win32":
        assert not owner_is_gone(f"{host}:1")  # Windows does not reserve PID 1
    assert not owner_is_gone(f"some-other-host:{child.pid}")  # cannot know
    for garbage in ("", "coordinator", f"{host}:", f"{host}:-5", f"{host}:0", f"{host}:x1", "w"):
        assert not owner_is_gone(garbage), garbage
