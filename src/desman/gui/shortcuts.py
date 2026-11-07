"""Non-interactive launch shortcuts; no server or configuration access."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import click

from desman.i18n import t

# Paths arrive as JSON on stdin, never as PowerShell source or shell arguments.
WINDOWS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$p = [Console]::In.ReadToEnd() | ConvertFrom-Json
$ws = New-Object -ComObject WScript.Shell
$folders = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
foreach ($folder in $folders) {
    $path = Join-Path $folder 'DESMAN.lnk'
    $link = $ws.CreateShortcut($path)
    $link.TargetPath = $p.target
    $link.Arguments = $p.arguments
    $link.IconLocation = $p.icon
    $link.Save()
    [Console]::WriteLine($path)
}
"""


def _desktop_quote(value: str) -> str:
    # Desktop Entry Exec has its own quoting and percent field-code syntax.
    if any(c in value for c in "\n\r\0"):
        raise ValueError(t("Invalid launcher path."))
    for char in ("\\", '"', "`", "$"):
        value = value.replace(char, "\\" + char)
    return '"' + value.replace("%", "%%").replace("\\", "\\\\") + '"'


def create(
    *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> list[Path]:
    try:
        return _create(runner)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise click.ClickException(t("Could not create shortcut: {error}", error=str(exc))) from exc


def _create(runner: Callable[..., subprocess.CompletedProcess[str]]) -> list[Path]:
    images = Path(__file__).resolve().parent / "static" / "img"
    executable = Path(sys.executable).absolute()
    if sys.platform == "win32":
        script = executable.parent / "Scripts" / "desman.exe"
        if not script.is_file():
            script = executable.parent / "desman.exe"
        pythonw = executable.with_name("pythonw.exe")
        target = script if script.is_file() else pythonw if pythonw.is_file() else executable
        arguments = "gui" if script.is_file() else "-m desman gui"
        result = runner(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_SCRIPT],
            input=json.dumps(
                {"target": str(target), "arguments": arguments, "icon": str(images / "logo.ico")}
            ),
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=True,
            shell=False,
        )
        return [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not sys.platform.startswith("linux"):
        raise click.ClickException(t("Shortcuts are not supported on this system yet."))
    script_name = shutil.which("desman")
    script_path = Path(script_name).absolute() if script_name else executable.parent / "desman"
    if not script_path.is_file():
        raise click.ClickException(t("The installed desman script was not found."))
    home = Path.home()
    data = Path(os.environ.get("XDG_DATA_HOME", str(home / ".local" / "share")))
    application = data / "applications" / "desman.desktop"
    content = (
        "[Desktop Entry]\nType=Application\nName=DESMAN\n"
        f"Exec={_desktop_quote(str(script_path))} gui\n"
        f"Icon={str(images / 'logo.png')}\n"
        "Terminal=false\nCategories=Utility;\n"
    )
    application.parent.mkdir(parents=True, exist_ok=True)
    application.write_text(content, encoding="utf-8")
    application.chmod(0o644)
    paths = [application]
    desktop = os.environ.get("XDG_DESKTOP_DIR")
    if desktop is None:
        config = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config"))) / "user-dirs.dirs"
        if config.is_file():
            for line in config.read_text(encoding="utf-8").splitlines():
                if line.startswith("XDG_DESKTOP_DIR="):
                    desktop = line.partition("=")[2].strip().strip('"').replace("$HOME", str(home))
                    break
    directory = Path(desktop) if desktop else home / "Desktop"
    if directory.is_dir() and directory != home:
        copy = directory / "desman.desktop"
        copy.write_text(content, encoding="utf-8")
        copy.chmod(0o755)
        paths.append(copy)
    return paths
