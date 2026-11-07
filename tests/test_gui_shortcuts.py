"""Shortcut creation stays non-interactive and uses only test home directories."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from desman.cli import cli
from desman.gui import shortcuts


@pytest.mark.skipif(os.name == "nt", reason="Linux desktop entries and POSIX file modes")
@pytest.mark.parametrize("desktop", [True, False])
def test_linux_shortcuts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desktop: bool) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("XDG_DESKTOP_DIR", raising=False)
    script = tmp_path / "bin space" / "desman"
    script.parent.mkdir()
    script.touch()
    monkeypatch.setattr(sys, "executable", str(script.parent / "python"))
    # Another installation first on PATH is never what the shortcut opens (Devin, PR #101).
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/desman")
    if desktop:
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "user-dirs.dirs").write_text('XDG_DESKTOP_DIR="$HOME/My Desktop"\n')
        (tmp_path / "My Desktop").mkdir()
    paths = shortcuts.create()
    assert len(paths) == (2 if desktop else 1)
    content = paths[0].read_text()
    assert f'Exec="{script}" gui\n' in content
    assert "Terminal=false\nCategories=Utility;\n" in content
    assert "logo.png\n" in content
    assert paths[0].stat().st_mode & 0o777 == 0o644
    if desktop:
        assert paths[1].read_text() == content
        assert paths[1].stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("launcher", ["pythonw", "python", "script", "scripts"])
def test_windows_fixed_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher: str) -> None:
    python = tmp_path / "python.exe"
    python.touch()
    target = python
    if launcher == "pythonw":
        target = tmp_path / "pythonw.exe"
    elif launcher == "script":
        target = tmp_path / "desman.exe"
    elif launcher == "scripts":
        (tmp_path / "Scripts").mkdir()
        target = tmp_path / "Scripts" / "desman.exe"
    target.touch()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", str(python))

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv == [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            shortcuts.WINDOWS_SCRIPT,
        ]
        assert kwargs["shell"] is False and kwargs["check"] is True
        payload = json.loads(kwargs["input"])
        assert payload["target"] == str(target)
        assert payload["arguments"] == (
            "gui" if launcher in ("script", "scripts") else "-m desman gui"
        )
        assert payload["icon"].endswith("logo.ico")
        assert str(tmp_path) not in argv[-1]
        assert "GetFolderPath('Desktop')" in argv[-1]
        assert "GetFolderPath('Programs')" in argv[-1]
        return subprocess.CompletedProcess(argv, 0, "desktop/DESMAN.lnk\nprograms/DESMAN.lnk\n")

    assert len(shortcuts.create(runner=runner)) == 2


def test_shortcut_cli_never_starts_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shortcuts, "create", lambda: [tmp_path / "desman.desktop"])
    result = CliRunner().invoke(cli, ["gui", "--shortcut"])
    assert result.exit_code == 0, result.output
    assert str(tmp_path / "desman.desktop") in result.output


def test_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    result = CliRunner().invoke(cli, ["gui", "--shortcut"])
    assert result.exit_code == 1
    assert "not supported" in result.output


@pytest.mark.skipif(os.name == "nt", reason="Linux desktop entries")
def test_linux_shortcut_without_a_script_runs_this_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("XDG_DESKTOP_DIR", raising=False)
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/desman")
    content = shortcuts.create()[0].read_text()
    assert f'Exec="{python}" -m desman gui\n' in content
