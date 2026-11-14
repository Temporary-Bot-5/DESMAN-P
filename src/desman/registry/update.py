"""Release-tag updates for the owner's registry, with schema consent and rollback.

The updater runs separately from the service it restarts. External commands and waits
are injectable; importing this module never contacts git, systemd or the live registry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from desman.core import filelock
from desman.i18n import t
from desman.registry import admin, store

TIMERS = ["desman-registry-daily.timer", "desman-registry-health.timer"]


class Runner(Protocol):
    def __call__(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]: ...


def run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )


def _command(runner: Runner, app: Path, args: list[str]) -> str:
    result = runner(args, cwd=app)
    if result.returncode:
        # Outputs may contain credentials (remote URLs, pip indexes). Keep diagnostics to
        # the executable and status; never journal arbitrary output from a subprocess.
        raise RuntimeError(f"{Path(args[0]).name}: exit {result.returncode}")
    return result.stdout.strip()


def _schema(path: Path) -> int:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])
    finally:
        conn.close()


def _changed_schema(path: Path, current: int) -> bool:
    try:
        return _schema(path) != current
    except (OSError, sqlite3.Error):
        return True  # an unreadable migrated DB also needs restoration


def _log(directory: Path, action: str, details: dict[str, Any]) -> None:
    # No migrations: the service owns schema upgrades, never the updater's old code.
    conn = sqlite3.connect(directory / store.DB_NAME, isolation_level=None)
    try:
        store.log(conn, action, details=details)
    finally:
        conn.close()


def _probe(app: Path, commit: str, runner: Runner) -> int | None:
    with tempfile.TemporaryDirectory(prefix="desman-registry-update-") as temp:
        tree = Path(temp) / "candidate"
        value = None
        try:
            _command(runner, app, ["git", "worktree", "add", "--detach", str(tree), commit])
            script = (
                f"import sys; sys.path.insert(0, {str(tree / 'src')!r}); "
                "from desman.registry.store import MIGRATIONS; print(len(MIGRATIONS))"
            )
            parsed = int(_command(runner, app, [sys.executable, "-I", "-c", script]))
            value = parsed if parsed >= 0 else None
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError):
            pass
        finally:
            try:
                _command(runner, app, ["git", "worktree", "remove", "--force", str(tree)])
            except (OSError, subprocess.SubprocessError, RuntimeError):
                value = None
        return value


def _backup(directory: Path, now: datetime) -> Path:
    folder = directory / "backups"
    folder.mkdir(mode=0o700, exist_ok=True)
    dest = folder / now.astimezone(UTC).strftime("registry-preupdate-%Y%m%d-%H%M%SZ.db")
    admin.backup(directory / store.DB_NAME, dest)
    pattern = re.compile(r"registry-preupdate-(\d{8}-\d{6}Z)\.db", re.ASCII)
    files = []
    for path in folder.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            try:
                stamp = datetime.strptime(match[1], "%Y%m%d-%H%M%SZ").replace(tzinfo=UTC)
            except ValueError:
                continue
            if stamp <= now + timedelta(minutes=5):
                files.append(path)
    for path in sorted(files)[:-14]:
        path.unlink()
    return dest


def _restore(directory: Path, backup: Path) -> None:
    path = directory / store.DB_NAME
    # Service must be stopped before replacing the DB and discarding its old WAL.
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as target:
        temp = Path(target.name)
        try:
            with backup.open("rb") as source:
                shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.chmod(0o600)
        os.replace(temp, path)
        Path(str(path) + "-wal").unlink(missing_ok=True)
        Path(str(path) + "-shm").unlink(missing_ok=True)
    finally:
        temp.unlink(missing_ok=True)


def update(
    directory: Path,
    *,
    app: Path | None = None,
    check: bool = False,
    allow_schema: bool = False,
    to: str | None = None,
    runner: Runner = run,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> tuple[str, bool]:
    """Return a translated message and success flag. Never create or migrate a database."""
    directory = directory.resolve()
    if not (directory / store.DB_NAME).is_file():
        raise admin.AdminError(
            t(
                "No database at {path}: the registry has not run yet.",
                path=str(directory / store.DB_NAME),
            )
        )
    with (directory / "update.lock").open("a+b") as lock:
        try:
            filelock.lock(lock.fileno(), blocking=False)
        except BlockingIOError:
            return t("Another registry update is running."), True
        try:
            return _update(
                directory,
                app,
                check,
                allow_schema,
                to,
                runner,
                sleep,
                now or datetime.now(UTC),
                which,
            )
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, sqlite3.Error) as e:
            raise admin.AdminError(t("Registry update refused: {error}", error=str(e)[:500])) from e
        finally:
            filelock.unlock(lock.fileno())


def _update(
    directory: Path,
    app: Path | None,
    check: bool,
    allow_schema: bool,
    to: str | None,
    runner: Runner,
    sleep: Callable[[float], None],
    now: datetime,
    which: Callable[[str], str | None],
) -> tuple[str, bool]:
    source = Path(__file__).resolve().parent if app is None else app.resolve()
    app = Path(_command(runner, source, ["git", "rev-parse", "--show-toplevel"]))

    def git(*args: str) -> str:
        return _command(runner, app, ["git", *args])

    git("fetch", "--tags", "origin")
    tags = git(
        "-c", "versionsort.suffix=-", "tag", "--merged", "origin/master", "--sort=-v:refname", "v*"
    ).splitlines()
    if to is not None and to not in tags:
        raise RuntimeError(t("The release tag must be on origin/master."))
    if not tags:
        return t("Registry is up to date."), True
    tag = to or tags[0]
    old = git("rev-parse", "HEAD")
    commit = git("rev-parse", f"refs/tags/{tag}^{{commit}}")
    ancestor = runner(["git", "merge-base", "--is-ancestor", old, commit], cwd=app)
    if ancestor.returncode not in (0, 1):
        raise RuntimeError("git merge-base failed")
    if old == commit or ancestor.returncode == 1:
        if to is not None and old != commit:
            return t(
                "Release {tag} is older than the installed version or on another branch: refused.",
                tag=tag,
            ), False
        if not check:
            row = _latest(directory)
            if row is not None and row[0] in ("update_waiting", "update_failed"):
                _log(directory, "update_current", {"tag": tag, "commit": old})
        if to is not None:
            return t("Registry is already at release {tag}.", tag=tag), True
        return t("Registry is up to date."), True
    if git("status", "--porcelain"):
        raise RuntimeError(t("The registry checkout has uncommitted changes."))
    current = _schema(directory / store.DB_NAME)
    try:
        target = _probe(app, commit, runner)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        target = None
    if target is not None and target < current:
        raise RuntimeError(t("The database is newer than the release schema."))
    waiting = target is None or target > current
    if check:
        if waiting and not allow_schema:
            if target is None:
                return t(
                    "Release {tag} would wait (schema unknown); needs --allow-schema.", tag=tag
                ), True
            return t(
                "Release {tag} would wait (schema {before} → {after}); needs --allow-schema.",
                tag=tag,
                before=current,
                after=target,
            ), True
        return t("Release {tag} would apply.", tag=tag), True
    if waiting and not allow_schema:
        row = _latest(directory)
        details = _details(row[1]) if row is not None else {}
        if not (
            row is not None
            and row[0] == "update_waiting"
            and details.get("tag") == tag
            and details.get("to_schema") == target
        ):
            _log(
                directory,
                "update_waiting",
                {"tag": tag, "from_schema": current, "to_schema": target},
            )
        return waiting_message(tag, unknown=target is None), True
    python = str(app / ".venv/bin/python")
    try:
        has_pip = runner([python, "-m", "pip", "--version"], cwd=app).returncode == 0
    except (OSError, subprocess.SubprocessError):
        has_pip = False
    if has_pip:
        installer = [python, "-m", "pip", "install", "-e", "."]
    elif uv := which("uv"):
        installer = [uv, "pip", "install", "--python", python, "-e", "."]
    else:
        raise RuntimeError(
            t("No installer available: install pip in the registry venv or put uv on PATH.")
        )
    step = "stop timers"
    failure: Exception | None = None
    rolled_back = False
    backup: Path | None = None
    changed = False
    timers_started = False
    try:
        _command(runner, app, ["systemctl", "--user", "stop", *TIMERS])
        step = "backup"
        backup = _backup(directory, now)
        step = "checkout"
        changed = True  # checkout may partially succeed before reporting failure
        git("checkout", "--detach", commit)
        step = "pip"
        _command(runner, app, installer)
        step = "restart"
        _command(runner, app, ["systemctl", "--user", "restart", "desman-registry.service"])
        step = "health"
        for attempt in range(30):
            health = runner(
                [str(app / ".venv/bin/desman-registry"), "--dir", str(directory), "health"], cwd=app
            )
            if health.returncode == 0:
                break
            if attempt == 29:
                raise RuntimeError("health failed")
            sleep(2)
        step = "start timers"
        _command(runner, app, ["systemctl", "--user", "start", *TIMERS])
        timers_started = True
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, sqlite3.Error) as e:
        failure = e
        rolled_back = True
        if changed:
            try:
                git("checkout", "--detach", old)
                _command(runner, app, installer)
            except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, sqlite3.Error):
                rolled_back = False
        try:
            if (
                allow_schema
                and backup is not None
                and _changed_schema(directory / store.DB_NAME, current)
            ):
                _command(runner, app, ["systemctl", "--user", "stop", "desman-registry.service"])
                _restore(directory, backup)
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError, sqlite3.Error):
            rolled_back = False
        try:
            _command(runner, app, ["systemctl", "--user", "restart", "desman-registry.service"])
        except (OSError, subprocess.SubprocessError, RuntimeError):
            rolled_back = False
    finally:
        try:
            if not timers_started:
                _command(runner, app, ["systemctl", "--user", "start", *TIMERS])
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            rolled_back = False
            if failure is None:
                failure = e
                step = "start timers"
    if failure is not None:
        _log(
            directory,
            "update_failed",
            {"step": step, "reason": str(failure)[:500], "rolled_back": rolled_back},
        )
        return t(
            "Registry update failed at {step}; rolled back: {rollback}.",
            step=step_message(step),
            rollback=t("yes") if rolled_back else t("no"),
        ), False
    _log(
        directory,
        "update_applied",
        {"from": old, "to": tag, "backup": backup.name if backup else None},
    )
    return t("Registry updated to {tag}.", tag=tag), True


def _details(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _latest(directory: Path) -> tuple[str, object] | None:
    conn = sqlite3.connect((directory / store.DB_NAME).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT action, details_json FROM registry_log WHERE action GLOB 'update_*' "
            "ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return None if row is None else (str(row[0]), row[1])
    finally:
        conn.close()


def waiting_message(tag: str, *, unknown: bool) -> str:
    if unknown:
        return t(
            "Release {tag} waits: its database schema could not be checked; run desman-registry update --allow-schema after checking it.",
            tag=tag,
        )
    return t(
        "Release {tag} waits: it changes the database; run desman-registry update --allow-schema.",
        tag=tag,
    )


def step_message(step: str) -> str:
    return {
        "stop timers": t("stop timers"),
        "backup": t("backup"),
        "checkout": t("checkout"),
        "pip": t("install packages"),
        "restart": t("restart"),
        "health": t("health"),
        "start timers": t("start timers"),
    }.get(step, "?")


def health_message(code: str) -> str:
    if code == "current":
        return t("Current.")
    if code.startswith("waiting unknown schema "):
        return waiting_message(code.removeprefix("waiting unknown schema "), unknown=True)
    if code.startswith("waiting "):
        return waiting_message(code.removeprefix("waiting "), unknown=False)
    return t("Registry update failed at {step}.", step=step_message(code.removeprefix("failed ")))


def update_health(conn: sqlite3.Connection) -> admin.HealthCheck:
    row = conn.execute(
        "SELECT action, details_json FROM registry_log WHERE action GLOB 'update_*' "
        "ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    if row is None or row[0] not in ("update_waiting", "update_failed"):
        return admin.HealthCheck("update", "ok", "current")
    details = _details(row[1])
    if row[0] == "update_waiting":
        prefix = "waiting unknown schema" if details.get("to_schema") is None else "waiting"
        message = f"{prefix} {details.get('tag') or '?'}"
    else:
        message = f"failed {details.get('step') or '?'}"
    return admin.HealthCheck("update", "warn", message)
