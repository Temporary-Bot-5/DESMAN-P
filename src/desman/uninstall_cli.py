"""Hand an installer-owned installation back to its uninstaller after exit."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import click
from rich.text import Text

import desman
from desman import sandbox_cli, searxng, searxng_native, ui
from desman.config import CONFIG_DIR_ENV, default_config_path, home_config_dir
from desman.core import db
from desman.core.sandbox import Sandbox, SandboxUnavailable, find_docker
from desman.i18n import language, set_language, t
from desman.stdio import configure_output

FAILURE_MARKER = "uninstall-failed.txt"
_PREVIEW_DOCKER_TIMEOUT = 5
_CLEANUP_DOCKER_TIMEOUT = 30
_DOCKER_PROBE_TTL = 30
_preview_probe: tuple[str, float, click.ClickException | None] | None = None
_monotonic = time.monotonic


def warn_last_failure() -> None:
    marker = db.default_db_path().parent / FAILURE_MARKER
    with suppress(OSError, UnicodeError):
        log = marker.read_text(encoding="utf-8").strip()
        marker.unlink()
        ui.warn(t("The last uninstall did not finish. Details: {log}", log=log))


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


def check_paths(app: Path, purge: bool) -> None:
    """Check the installer's path restrictions before scheduling removal."""
    try:
        _check_paths(app, purge)
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(t("Could not start uninstall: {error}", error=str(exc))) from exc


def _check_paths(app: Path, purge: bool) -> None:
    home = Path.home().resolve()
    if home.is_relative_to(app) or app == Path(app.anchor):
        raise click.ClickException(t("Unsafe app folder: {path}", path=app))
    home_config = home_config_dir()
    data, config = db.default_db_path().parent, _config_folder()
    for path in (data, config, home_config):
        if path.absolute().is_relative_to(app) or path.resolve().is_relative_to(app):
            raise click.ClickException(t("Data and configuration must be outside the app folder."))

    if purge:
        for path in (data.absolute(), config.absolute()):
            try:
                info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise click.ClickException(t("Purge folder is a link: {path}", path=path))
            if (
                ".." in path.parts
                or home.is_relative_to(path.resolve())
                or path == Path(path.anchor)
            ):
                raise click.ClickException(t("Unsafe purge folder: {path}", path=path))
    else:
        source = app / "config.json"
        backup = home_config / "config.json"
        source_exists = source.exists() if sys.platform == "win32" else source.is_file()
        if source_exists and (backup.exists() or backup.is_symlink()):
            raise click.ClickException(
                t(
                    "A config backup already exists at {path}. Preserve app/config.json before uninstalling.",
                    path=backup,
                )
            )


def _docker_settings_exist(app: Path) -> bool:
    roots = {app, default_config_path().parent, home_config_dir()}
    return any(
        (root / name / "settings.yml").is_file()
        for root in roots
        for name in ("searxng", "searxng-docker")
    )


def _daemon_unreachable(exc: click.ClickException) -> bool:
    if isinstance(exc.__cause__, subprocess.TimeoutExpired):
        return True
    error = exc.message.casefold()
    pipe_error = "error during connect" in error and any(
        detail in error
        for detail in ("//./pipe/", "npipe:", "the system cannot find the file specified")
    )
    if pipe_error and sys.platform == "win32":
        pipe = re.search(r"//\./pipe/([^/\s:\"']+)", error)
        if pipe is not None:
            with suppress(OSError):
                return all(name.casefold() != pipe[1] for name in os.listdir("\\\\.\\pipe\\"))
    if "permission denied" in error or "access is denied" in error:
        return False
    return pipe_error or any(
        phrase in error
        for phrase in (
            "cannot connect to the docker daemon",
            "failed to connect to the docker api",
            "is the docker daemon running",
            "connection refused",
        )
    )


def _probe_docker(timeout: int) -> click.ClickException | None:
    try:
        searxng.docker(["info"], timeout=timeout)
    except click.ClickException as exc:
        return exc
    return None


def _preview_docker(executable: str) -> click.ClickException | None:
    global _preview_probe
    if _preview_probe is not None:
        cached_executable, expires, error = _preview_probe
        if executable == cached_executable and _monotonic() < expires:
            return error
    error = _probe_docker(_PREVIEW_DOCKER_TIMEOUT)
    _preview_probe = (executable, _monotonic() + _DOCKER_PROBE_TTL, error)
    return error


def _docker_unreachable_now() -> bool:
    error = _probe_docker(_CLEANUP_DOCKER_TIMEOUT)
    return error is not None and _daemon_unreachable(error)


def docker_search_unavailable(app: Path) -> bool:
    executable = find_docker()
    if executable is None or not _docker_settings_exist(app):
        return False
    error = _preview_docker(executable)
    return error is not None and _daemon_unreachable(error)


def _docker_skip_note() -> str:
    return t(
        "Docker is not running or not answering, so the SearXNG container will stay. Start "
        "Docker before uninstalling to remove it, or run `docker rm -f desman-searxng` later."
    )


def plan(app: Path, purge: bool, *, docker_unavailable: bool | None = None) -> list[str]:
    if docker_unavailable is None:
        docker_unavailable = docker_search_unavailable(app)
    lines = [
        t("Remove the DESMAN app folder: {path}", path=app),
        t("Remove the desman launcher and DESMAN shortcuts."),
        t("Remove managed SearXNG if present (desman search setup brings it back)."),
    ]
    if docker_unavailable:
        lines.append(_docker_skip_note())
    if purge:
        lines.extend(
            [
                t("Delete the data folder: {path}", path=db.default_db_path().parent.absolute()),
                t("Delete the configuration folder: {path}", path=_config_folder().absolute()),
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


def _docker_search(app: Path, *, removing: bool) -> dict[str, Any] | None:
    executable = find_docker()
    if executable is None:
        return None
    managed = _docker_settings_exist(app)
    error = _probe_docker(_CLEANUP_DOCKER_TIMEOUT) if removing else _preview_docker(executable)
    if error is not None:
        if managed and not _daemon_unreachable(error):
            raise error
    else:
        try:
            return searxng.inspect()
        except click.ClickException as exc:
            if (
                managed
                and not _daemon_unreachable(exc)
                and not (removing and _docker_unreachable_now())
            ):
                raise
    if managed and removing:
        ui.out(Text(_docker_skip_note()))
    return None


def _remove_docker_search(
    container: dict[str, Any], *, sleep: Callable[[float], None] = time.sleep
) -> None:
    # The image can have a configuration volume as well as a cache volume.
    volumes = [m for m in container.get("Mounts", []) if m.get("Type") == "volume"]
    cache_only = all(m.get("Destination") == "/var/cache/searxng" for m in volumes)
    flags = ["-v"] if cache_only else []
    try:
        searxng.docker(["rm", "-f", *flags, searxng.CONTAINER])
    except click.ClickException as exc:
        error = exc.message.casefold()
        if "no such container" in error:
            return
        if "removal of container" in error and "is already in progress" in error:
            for _ in range(2):
                sleep(0.5)
                try:
                    if searxng.inspect(timeout=2) is None:
                        return
                except click.ClickException as check:
                    if "no such container" in check.message.casefold():
                        return
                    if not _daemon_unreachable(check):
                        raise
            ui.out(
                Text(
                    t(
                        "The SearXNG container may still be present. "
                        "To remove it later, run `docker rm -f desman-searxng`."
                    )
                )
            )
            return
        if not _docker_unreachable_now():
            raise
        ui.out(Text(_docker_skip_note()))


def _services(app: Path, purge: bool, *, removing: bool) -> None:
    box = _wsl_sandbox(removing=True) if purge and sys.platform == "win32" else None
    container = _docker_search(app, removing=removing)
    if purge:
        # The installer deletes the native files. Stop the service to release its handles.
        folder = searxng_native.directory()
        if folder.is_dir():
            with searxng_native.locked(folder):
                if removing:
                    searxng_native.stop(folder)
    if removing:
        if box is not None:
            sandbox_cli.remove_wsl_sandbox(box)
        if container is not None:
            _remove_docker_search(container)


def _finish_cleanup(purge: bool) -> None:
    configure_output()
    set_language(os.environ.get("DESMAN_UNINSTALL_LANGUAGE", language()))
    app = require_app()
    check_paths(app, purge)
    _services(app, purge, removing=True)


def _cleanup_command(purge: bool) -> str:
    return f"from desman.uninstall_cli import _finish_cleanup; _finish_cleanup({purge})"


def _windows_waiter(pid: int, script: Path, log: Path, purge: bool) -> list[str]:
    extra = " -Purge" if purge else ""
    command = (
        "function Report-Failure { "
        "[IO.Directory]::CreateDirectory($env:DESMAN_UNINSTALL_DATA) | Out-Null; "
        f"[IO.File]::WriteAllText((Join-Path $env:DESMAN_UNINSTALL_DATA '{FAILURE_MARKER}'), "
        "$env:DESMAN_UNINSTALL_LOG) }; "
        "$prefix = $env:DESMAN_HOME + '\\'; "
        "$timer = [Diagnostics.Stopwatch]::StartNew(); "
        f"while (Get-Process | Where-Object {{ $_.Id -eq {pid} -or "
        "($_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) }) { "
        "if ($timer.Elapsed.TotalSeconds -ge 600) { "
        f"Set-Content -LiteralPath {_powershell_quote(log)} -Encoding UTF8 "
        "-Value $env:DESMAN_UNINSTALL_TIMEOUT; Report-Failure; exit 1 }; "
        "Start-Sleep -Seconds 1 }; "
        "Start-Sleep -Seconds 2; "
        f"& $env:DESMAN_UNINSTALL_PYTHON -c '{_cleanup_command(purge)}' "
        f"*> {_powershell_quote(log)}; "
        "if ($LASTEXITCODE -eq 0) { "
        "& powershell -NoProfile -ExecutionPolicy Bypass "
        f"-File {_powershell_quote(script)} -Uninstall -Yes{extra} *>> {_powershell_quote(log)} }}; "
        "$code = $LASTEXITCODE; if ($code -ne 0) { Report-Failure }; exit $code"
    )
    return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]


def _posix_waiter(pid: int, script: Path, log: Path, purge: bool) -> list[str]:
    # A virtualenv may symlink Python outside the app. Check argv[0] as well as the
    # kernel's executable path so a chat or worker using that virtualenv also counts.
    if sys.platform == "linux":
        processes = """
    for entry in /proc/[0-9]*; do
        exe=$(readlink "$entry/exe" 2>/dev/null) || continue
        case "$exe" in "$DESMAN_HOME/"*) return 0 ;; esac
        argv0=$(head -z -n 1 "$entry/cmdline" 2>/dev/null | tr -d '\\000')
        case "$argv0" in "$DESMAN_HOME/"*) return 0 ;; esac
    done
"""
    else:
        processes = """
    ps -axo pid=,comm= | awk '{
        sub(/^[[:space:]]*[0-9]+[[:space:]]+/, "")
        if (index($0, ENVIRON["DESMAN_HOME"] "/") == 1) found = 1
    } END { exit !found }' && return 0
"""
    extra = " --purge" if purge else ""
    command = f"""exec >{shlex.quote(str(log))} 2>&1
failed() {{
    mkdir -p "$DESMAN_UNINSTALL_DATA" &&
        printf '%s\\n' "$DESMAN_UNINSTALL_LOG" > "$DESMAN_UNINSTALL_DATA/{FAILURE_MARKER}"
    exit "$1"
}}
running() {{
    if kill -0 {pid} 2>/dev/null; then return 0; fi
{processes}    return 1
}}
deadline=$(($(date +%s) + 600))
while running; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        printf '%s\\n' "$DESMAN_UNINSTALL_TIMEOUT"
        failed 1
    fi
    sleep 1
done
sleep 2
"$DESMAN_UNINSTALL_PYTHON" -c {shlex.quote(_cleanup_command(purge))} || failed $?
sh {shlex.quote(str(script))} --uninstall --yes{extra} || failed $?
"""
    return ["sh", "-c", command]


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
        check_paths(app, purge)
        temp_root = Path(tempfile.gettempdir()).resolve()
        removed = [app]
        if purge:
            removed.extend([db.default_db_path().parent, _config_folder()])
        if any(temp_root.is_relative_to(path.resolve()) for path in removed):
            raise click.ClickException(
                t("The temporary folder is inside a directory being removed.")
            )
        _services(app, purge, removing=False)
        folder = Path(tempfile.mkdtemp(prefix="desman-uninstall-", dir=temp_root))
        script = folder / _installer(app).name
        shutil.copyfile(_installer(app), script)
        log = folder / "uninstall.log"
        env = dict(
            os.environ,
            DESMAN_HOME=str(app),
            DESMAN_UNINSTALL_PYTHON=sys.executable,
            DESMAN_UNINSTALL_DATA=str(db.default_db_path().parent.absolute()),
            DESMAN_UNINSTALL_LOG=str(log),
            DESMAN_UNINSTALL_LANGUAGE=language(),
            DESMAN_UNINSTALL_TIMEOUT=t(
                "Uninstall cancelled after waiting 10 minutes for DESMAN processes to close."
            ),
        )
        # The detached process must not hold the app (or a purge directory) as its cwd.
        for name in (db.DATA_DIR_ENV, CONFIG_DIR_ENV):
            if env.get(name):
                env[name] = str(Path(env[name]).absolute())
        pid = os.getpid()
        flags = 0
        windows = sys.platform == "win32"
        if windows:
            argv = _windows_waiter(pid, script, log, purge)
            # A hidden console of its own: Windows PowerShell 5.1 exits at once without any
            # console (DETACHED_PROCESS), and the person's window may close before it runs.
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0x08000000
            )
        else:
            argv = _posix_waiter(pid, script, log, purge)
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
    return t(
        "DESMAN will be removed after all DESMAN windows and workers close. "
        "Removal is cancelled if any stay open for more than 10 minutes. Log: {path}",
        path=log,
    )


@click.command(help="Remove DESMAN installed by the installer.")
@click.option("--purge", is_flag=True, help="Also delete my data, keys and chat history")
@click.option("--yes", is_flag=True, help="Confirm removal.")
def uninstall(purge: bool, yes: bool) -> None:
    app = require_app()
    check_paths(app, purge)
    for line in plan(app, purge):
        ui.out(Text(line))
    if yes:
        ui.out(Text(started_message(start(app, purge))))
        return
    command = "desman uninstall --purge --yes" if purge else "desman uninstall --yes"
    ui.out(t("Nothing changed. Confirm with: {command}", command=command))
