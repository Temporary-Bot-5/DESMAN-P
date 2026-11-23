"""Hand an installer-owned installation back to its uninstaller after exit."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import click
from rich.text import Text

import desman
from desman import sandbox_cli, searxng, searxng_native, ui
from desman.config import CONFIG_DIR_ENV, default_config_path, home_config_dir
from desman.core import db
from desman.core.sandbox import Sandbox, SandboxUnavailable, find_docker
from desman.i18n import t


def _installer(app: Path) -> Path:
    return app / "install" / ("install.ps1" if sys.platform == "win32" else "install.sh")


def installed_app() -> Path | None:
    app = Path(desman.__file__).resolve().parents[2]
    if (
        (app / ".desman-install").is_file()
        and (app / ".git").is_dir()
        and (app / "src/desman/cli.py").is_file()
        and _installer(app).is_file()
    ):
        return app
    return None


def require_app() -> Path:
    app = installed_app()
    if app is None:
        raise click.ClickException(
            t("DESMAN was not installed by the installer. Remove it the way it was installed.")
        )
    return app


def _wsl_sandbox(*, removing: bool = False) -> Sandbox | None:
    box = Sandbox()
    try:
        return box if box.wsl_distribution_present() else None
    except SandboxUnavailable as exc:
        # WSL can be disabled on a machine that has never set up DESMAN's distribution.
        if removing and box.wsl_dir.exists():
            raise click.ClickException(
                t("WSL sandbox removal failed: {error}", error=str(exc))
            ) from exc
        return None


def _config_folder() -> Path:
    return default_config_path().parent if os.environ.get(CONFIG_DIR_ENV) else home_config_dir()


def plan(app: Path, purge: bool) -> list[str]:
    lines = [
        t("Remove the DESMAN app folder: {path}", path=app),
        t("Remove the desman launcher and DESMAN shortcuts."),
    ]
    if purge:
        lines.extend(
            [
                t("Delete the data folder: {path}", path=db.default_db_path().parent.absolute()),
                t("Delete the configuration folder: {path}", path=_config_folder().absolute()),
                t("Remove managed SearXNG if present."),
            ]
        )
        if sys.platform == "win32" and _wsl_sandbox() is not None:
            lines.append(t("Unregister the desman-sandbox WSL distribution."))
    else:
        lines.append(t("Your data, keys and history are kept."))
    lines.append(t("Close other DESMAN windows, chats and workers first."))
    return lines


def _powershell_quote(path: Path) -> str:
    value = str(path)
    for quote in ("'", "‘", "’", "‚", "‛"):
        value = value.replace(quote, quote * 2)
    return "'" + value + "'"


def _purge_services() -> None:
    if sys.platform == "win32" and (box := _wsl_sandbox(removing=True)) is not None:
        sandbox_cli.remove_wsl_sandbox(box)
    # Docker keeps the container and its volumes outside the app, config and data folders.
    if find_docker() is not None:
        try:
            container = searxng.inspect()
        except click.ClickException:
            config = default_config_path().parent
            if any(
                (config / name / "settings.yml").is_file() for name in ("searxng", "searxng-docker")
            ):
                raise
        else:
            if container is not None:
                searxng.docker(["rm", "-f", "-v", searxng.CONTAINER])
    # The installer deletes the native files. Stop the service to release its handles.
    folder = searxng_native.directory()
    if folder.is_dir():
        with searxng_native.locked(folder):
            searxng_native.stop(folder)


def start(
    app: Path,
    purge: bool,
    *,
    spawn: Callable[..., object] | None = None,
) -> Path:
    """Schedule uninstall and return the log path, which outlives the app."""
    if spawn is None:
        spawn = subprocess.Popen
    folder: Path | None = None
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
        removed = [app]
        if purge:
            removed.extend([db.default_db_path().parent, _config_folder()])
        if any(temp_root.is_relative_to(path.resolve()) for path in removed):
            raise click.ClickException(
                t("The temporary folder is inside a directory being removed.")
            )
        folder = Path(tempfile.mkdtemp(prefix="desman-uninstall-", dir=temp_root))
        script = folder / _installer(app).name
        shutil.copyfile(_installer(app), script)
        log = folder / "uninstall.log"
        env = dict(os.environ, DESMAN_HOME=str(app))
        # The detached process must not hold the app (or a purge directory) as its cwd.
        for name in (db.DATA_DIR_ENV, CONFIG_DIR_ENV):
            if env.get(name):
                env[name] = str(Path(env[name]).absolute())
        pid = os.getpid()
        flags = 0
        windows = sys.platform == "win32"
        if windows:
            extra = " -Purge" if purge else ""
            command = (
                f"Wait-Process -Id {pid} -ErrorAction SilentlyContinue; "
                "Start-Sleep -Seconds 2; "
                f"& {_powershell_quote(script)} -Uninstall -Yes{extra} *> {_powershell_quote(log)}"
            )
            argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
            # A hidden console of its own: Windows PowerShell 5.1 exits at once without any
            # console (DETACHED_PROCESS), and the person's window may close before it runs.
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000
            )
        else:
            extra = " --purge" if purge else ""
            command = (
                f"while kill -0 {pid} 2>/dev/null; do sleep 1; done; sleep 2; "
                f"exec sh {shlex.quote(str(script))} --uninstall --yes{extra} "
                f">{shlex.quote(str(log))} 2>&1"
            )
            argv = ["sh", "-c", command]
        if purge:
            _purge_services()
        spawn(
            argv,
            env=env,
            cwd=folder,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=not windows,
            creationflags=flags,
        )
    except (OSError, click.ClickException) as exc:
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        if isinstance(exc, click.ClickException):
            raise
        raise click.ClickException(t("Could not start uninstall: {error}", error=str(exc))) from exc
    return log


def started_message(log: Path) -> str:
    return t("DESMAN will be removed as soon as this program ends. Log: {path}", path=log)


@click.command(help="Remove DESMAN installed by the installer.")
@click.option("--purge", is_flag=True, help="Also delete my data, keys and chat history")
@click.option("--yes", is_flag=True, help="Confirm removal.")
def uninstall(purge: bool, yes: bool) -> None:
    app = require_app()
    if yes:
        ui.out(Text(started_message(start(app, purge))))
        return
    for line in plan(app, purge):
        ui.out(Text(line))
    command = "desman uninstall --purge --yes" if purge else "desman uninstall --yes"
    ui.out(t("Nothing changed. Confirm with: {command}", command=command))
