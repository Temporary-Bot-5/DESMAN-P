"""Installer detection, preview, and detached handoff without running an installer."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
from click.testing import CliRunner

import desman
from desman import commands, runs, sandbox_cli, searxng, searxng_native, ui, uninstall_cli
from desman.chat import Chat
from desman.cli import cli
from desman.config import CONFIG_DIR_ENV, open_config
from desman.core import db
from desman.core.sandbox import Sandbox, SandboxUnavailable
from desman.core.sessions import Sessions
from desman.gui.front import WebFront
from desman.info import print_project_info
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from helpers import make_config


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    app = tmp_path / "DESMAN's app"
    source = app / "src" / "desman"
    source.mkdir(parents=True)
    (source / "cli.py").touch()
    (app / ".git").mkdir()
    (app / ".desman-install").touch()
    (app / "install").mkdir()
    for script in ("install.sh", "install.ps1"):
        (app / "install" / script).write_text(script, encoding="utf-8")
    monkeypatch.setattr(desman, "__file__", str(source / "__init__.py"))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    temp = tmp_path / "DESMAN's temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    monkeypatch.setattr("desman.uninstall_cli.find_docker", lambda: None)
    monkeypatch.setattr(Sandbox, "wsl_distribution_present", lambda _: False)
    return app


def test_installed_app_and_refusal(app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert uninstall_cli.installed_app() == app
    (app / ".desman-install").unlink()
    assert uninstall_cli.installed_app() is None
    monkeypatch.setattr(uninstall_cli, "start", lambda *_: pytest.fail("spawned"))
    result = CliRunner().invoke(cli, ["uninstall", "--yes", "--purge"])
    assert result.exit_code == 1
    assert "Remove it the way it was installed." in result.output
    assert (app / "src/desman/cli.py").is_file()


@pytest.mark.parametrize("purge", [False, True])
def test_uninstall_previews_without_yes(
    app: Path, purge: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uninstall_cli, "start", lambda *_: pytest.fail("spawned"))
    args = ["uninstall", *(["--purge"] if purge else [])]
    result = CliRunner().invoke(cli, args, terminal_width=250)
    assert result.exit_code == 0, result.output
    assert str(app) in result.output
    assert "Close other DESMAN windows" in result.output
    assert (
        "Remove managed SearXNG if present (desman search setup brings it back)." in result.output
    )
    assert f"Nothing changed. Confirm with: desman {' '.join(args)} --yes" in result.output
    if purge:
        assert str(db.default_db_path().parent) in result.output
        assert os.environ[CONFIG_DIR_ENV] in result.output
    else:
        assert "Your data, keys and history are kept." in result.output


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("purge", [False, True])
def test_uninstall_spawns_detached_copy(
    app: Path, platform: str, purge: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def spawn(argv: list[str], **kwargs: Any) -> None:
        calls.append((argv, kwargs))

    monkeypatch.setattr(subprocess, "Popen", spawn)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        patch.setattr(shutil, "which", lambda *_: None)
        result = CliRunner().invoke(
            cli,
            ["uninstall", "--yes", *(["--purge"] if purge else [])],
            terminal_width=250,
        )
    assert result.exit_code == 0, result.output
    [(argv, kwargs)] = calls
    folder = kwargs["cwd"]
    script = folder / ("install.ps1" if platform == "win32" else "install.sh")
    log = folder / "uninstall.log"
    assert not folder.is_relative_to(app)
    assert script.read_bytes() == (app / "install" / script.name).read_bytes()
    for key in (db.DATA_DIR_ENV, CONFIG_DIR_ENV):
        assert kwargs["env"][key] == os.environ[key]
    assert kwargs["env"]["DESMAN_HOME"] == str(app)
    assert kwargs["env"]["DESMAN_UNINSTALL_DATA"] == str(db.default_db_path().parent)
    assert kwargs["env"]["DESMAN_UNINSTALL_LOG"] == str(log)
    assert "Removal is cancelled if any stay open for more than 10 minutes." in result.output
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert str(log) in result.output
    assert result.output.index("Remove the DESMAN app folder") < result.output.index(
        "DESMAN will be removed after all DESMAN windows and workers close"
    )
    command = argv[-1]
    if platform == "win32":
        assert argv[:-1] == ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]
        assert f"Get-Process | Where-Object {{ $_.Id -eq {os.getpid()}" in command
        assert "$_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)" in command
        assert "Start-Sleep -Seconds 2;" in command
        script_literal, log_literal = (str(path).replace("'", "''") for path in (script, log))
        extra = " -Purge" if purge else ""
        assert f"-File '{script_literal}' -Uninstall -Yes{extra} *>> '{log_literal}'" in command
        assert "Report-Failure; exit 1" in command
        assert "$code -ne 0) { Report-Failure }" in command
        assert kwargs["creationflags"] == 0x200 | 0x08000000
        assert not kwargs["start_new_session"]
    else:
        assert argv[:2] == ["sh", "-c"]
        assert f"kill -0 {os.getpid()}" in command
        assert ('readlink "$entry/exe"' in command) is (platform == "linux")
        assert "\nsleep 2\n" in command
        assert ("ps -axo pid=,comm=" in command) is (platform == "darwin")
        assert command.endswith(f" --uninstall --yes{' --purge' if purge else ''} || failed $?\n")
        assert kwargs["start_new_session"]
        assert kwargs["creationflags"] == 0
    assert f"_finish_cleanup({purge})" in command
    assert "uninstall-failed.txt" in command


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("yes", [False, True])
def test_refuses_existing_config_backup(
    app: Path, platform: str, yes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    (app / "config.json").touch()
    backup = Path.home() / ".config" / "desman" / "config.json"
    backup.parent.mkdir(parents=True)
    backup.touch()
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: pytest.fail("spawned"))
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", platform)
        result = CliRunner().invoke(
            cli, ["uninstall", *(["--yes"] if yes else [])], terminal_width=250
        )
    assert result.exit_code == 1
    assert f"A config backup already exists at {backup}." in result.output
    assert "will be removed" not in result.output


@pytest.mark.parametrize("env", [CONFIG_DIR_ENV, db.DATA_DIR_ENV])
@pytest.mark.parametrize("purge", [False, True])
def test_refuses_folders_inside_app(
    app: Path, env: str, purge: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, str(app / "nested"))
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: pytest.fail("spawned"))
    result = CliRunner().invoke(cli, ["uninstall", "--yes", *(["--purge"] if purge else [])])
    assert result.exit_code == 1
    assert "Data and configuration must be outside the app folder." in result.output


@pytest.mark.parametrize("location", ["home", "root", "parent-traversal", "link", "reparse"])
def test_refuses_unsafe_purge(app: Path, location: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = app.parent / "purge"
    if location == "home":
        path = Path.home()
    elif location == "root":
        path = Path(app.anchor)
    elif location == "parent-traversal":
        path = path / ".." / "other"
    elif location == "link":
        try:
            path.symlink_to(app.parent / "missing", target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Cannot create a symlink: {exc}")
    else:
        lstat = Path.lstat

        def with_reparse(candidate: Path) -> Any:
            if candidate == path:
                return SimpleNamespace(
                    st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT
                )
            return lstat(candidate)

        monkeypatch.setattr(Path, "lstat", with_reparse)
    monkeypatch.setenv(db.DATA_DIR_ENV, str(path))
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: pytest.fail("spawned"))
    monkeypatch.setattr(uninstall_cli, "_services", lambda *_, **__: pytest.fail("purged services"))
    result = CliRunner().invoke(cli, ["uninstall", "--yes", "--purge"], terminal_width=250)
    assert result.exit_code == 1
    expected = (
        "Purge folder is a link" if location in {"link", "reparse"} else "Unsafe purge folder"
    )
    assert f"{expected}: {path}" in result.output


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process discovery uses /proc")
@pytest.mark.parametrize("case", ["executable", "symlink", "timeout"])
def test_linux_waiter_checks_other_processes(app: Path, case: str) -> None:
    sleep = shutil.which("sleep")
    shell = shutil.which("sh")
    assert sleep is not None and shell is not None
    # Not a copy of `sleep`: multi-call coreutils (uutils on newer Ubuntu) dispatch on the
    # program's name and exit at once under another one. A shell runs under any name.
    shell = os.path.realpath(shell)
    worker = app / "worker"
    if case == "symlink":
        worker.symlink_to(shell)
    else:
        shutil.copy(shell, worker)
    folder = app.parent
    script = folder / "remove.sh"
    script.write_text("echo installer-ran\n", encoding="utf-8")
    log = folder / "log"
    waiting = folder / "waiting"
    programs = folder / "waiter-tools"
    programs.mkdir()
    pause = programs / "sleep"
    pause.write_text(
        '#!/bin/sh\nif [ "$1" = 1 ]; then touch "$WAITING"; "$REAL_SLEEP" .05; fi\n',
        encoding="utf-8",
    )
    pause.chmod(0o755)
    if case == "timeout":
        date = programs / "date"
        date.write_text(
            '#!/bin/sh\nif [ -e "$WAITING" ]; then echo 601; else touch "$WAITING"; echo 0; fi\n',
            encoding="utf-8",
        )
        date.chmod(0o755)
    purge_marker = folder / "purge-ran"
    purge_helper = programs / "purge-helper"
    purge_helper.write_text('#!/bin/sh\ntouch "$PURGE_MARKER"\n', encoding="utf-8")
    purge_helper.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{programs}:{os.environ['PATH']}",
        DESMAN_HOME=str(app),
        DESMAN_UNINSTALL_TIMEOUT="Timed out after 10 minutes",
        DESMAN_UNINSTALL_DATA=str(folder / "data"),
        DESMAN_UNINSTALL_LOG=str(log),
        DESMAN_UNINSTALL_PYTHON=str(purge_helper),
        PURGE_MARKER=str(purge_marker),
        WAITING=str(waiting),
        REAL_SLEEP=sleep,
    )
    with subprocess.Popen([worker, "-c", '"$0" 60; :', sleep]) as other:
        try:
            with subprocess.Popen(
                uninstall_cli._posix_waiter(2**30, script, log, True), env=env
            ) as waiter:
                try:
                    if case != "timeout":
                        deadline = time.monotonic() + 5
                        while not waiting.exists() and waiter.poll() is None:
                            assert time.monotonic() < deadline
                            time.sleep(0.01)
                        assert waiting.exists()
                        assert log.read_text() == ""
                        other.terminate()
                        other.wait(timeout=5)
                    assert waiter.wait(timeout=5) == (1 if case == "timeout" else 0)
                finally:
                    if waiter.poll() is None:
                        waiter.kill()
        finally:
            if other.poll() is None:
                other.kill()
    assert log.read_text().strip() == (
        "Timed out after 10 minutes" if case == "timeout" else "installer-ran"
    )
    assert purge_marker.exists() is (case != "timeout")
    failure = folder / "data" / uninstall_cli.FAILURE_MARKER
    if case == "timeout":
        assert failure.read_text().strip() == str(log)
    else:
        assert not failure.exists()


def test_windows_purge_unregisters_sandbox(app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def remove(box: Sandbox) -> bool:
        assert box.platform == "win32"
        calls.append("remove")
        return True

    monkeypatch.setattr(Sandbox, "wsl_distribution_present", lambda _: True)
    monkeypatch.setattr(sandbox_cli, "remove_wsl_sandbox", remove)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        patch.setattr(shutil, "which", lambda *_: None)
        assert any("desman-sandbox" in line for line in uninstall_cli.plan(app, True))
        uninstall_cli._services(app, True, removing=True)
    assert calls == ["remove"]


@pytest.mark.parametrize("purge", [False, True])
@pytest.mark.parametrize("config_volume", [False, True])
def test_cleanup_removes_managed_search(
    app: Path, purge: bool, config_volume: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker: list[list[str]] = []
    stopped: list[Path] = []
    native = searxng_native.directory()
    native.mkdir(parents=True)
    monkeypatch.setattr("desman.uninstall_cli.find_docker", lambda: "docker")
    mounts = [{"Type": "volume", "Destination": "/var/cache/searxng"}]
    if config_volume:
        mounts.append({"Type": "volume", "Destination": "/etc/searxng"})
    monkeypatch.setattr(searxng, "inspect", lambda: {"Mounts": mounts})
    monkeypatch.setattr(searxng, "docker", docker.append)
    monkeypatch.setattr(searxng_native, "stop", stopped.append)
    uninstall_cli._services(app, purge, removing=False)
    assert not docker and not stopped
    uninstall_cli._services(app, purge, removing=True)
    assert docker == [["rm", "-f", *([] if config_volume else ["-v"]), "desman-searxng"]]
    assert stopped == ([native] if purge else [])
    assert native.is_dir()  # the installer purges this directory


def test_catalog_previews_uninstall(app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uninstall_cli, "start", lambda *_: pytest.fail("spawned without approval"))
    entry = commands.find("uninstall")
    assert entry is not None and entry.typed_confirm
    args = commands.UninstallArgs()
    assert entry.kind_of(args) is commands.Kind.CHANGE
    conn = db.open_db()
    try:
        chat = Chat(Sessions(conn))
        ctx = commands.Context(
            chat.store, open_config(), lambda *_: None, lambda _: False, lambda: None
        )
        proposal = entry.handler(ctx, args)
        assert isinstance(proposal, commands.Proposal)
        assert proposal.lines == uninstall_cli.plan(app, False)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "line",
    ["/uninstall --purge", "Remove DESMAN", "/uninstall --purge --yes", "!uninstall --purge --yes"],
)
def test_confirmed_uninstall_ends_terminal_chat(
    app: Path, line: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = app / "searxng-docker" / "settings.yml"
    settings.parent.mkdir()
    settings.touch()
    monkeypatch.setattr(uninstall_cli, "find_docker", lambda: "docker")

    def docker(args: list[str], timeout: float = 60) -> str:
        assert args == ["info"]
        raise click.ClickException("Cannot connect to the Docker daemon")

    monkeypatch.setattr(searxng, "docker", docker)
    open_config().save(make_config(configured=True))
    llm = FakeLLM().on(CHAT, '{"type":"command","name":"uninstall","args":{"purge":true}}')
    monkeypatch.setattr(runs, "models", lambda *_, **__: runs.Models(llm, None))
    confirmations: list[bool] = []

    def confirm(question: str, *, default: bool) -> bool:
        assert (
            "Docker is not running" in question
            or "Docker is not running" in capsys.readouterr().out
        )
        confirmations.append(True)
        return True

    monkeypatch.setattr(ui, "confirm", confirm)
    calls: list[tuple[Path, bool]] = []

    def start(path: Path, purge: bool) -> Path:
        calls.append((path, purge))
        return path / "uninstall.log"

    monkeypatch.setattr(uninstall_cli, "start", start)
    lines = iter([line])
    conn = db.open_db()
    try:
        Chat(Sessions(conn), read=lambda: next(lines)).run()
    finally:
        conn.close()
    assert calls == [(app, True)]
    assert len(confirmations) == (0 if "--yes" in line else 1)


@pytest.mark.parametrize("recovers", [False, True])
def test_stopped_docker_allows_uninstall(
    app: Path, recovers: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = app / "searxng-docker" / "settings.yml"
    settings.parent.mkdir()
    settings.touch()
    error = "failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine"
    commands_run: list[list[str]] = []
    running = False

    def docker(args: list[str], timeout: float = 60) -> str:
        commands_run.append(args)
        if args == ["info"]:
            assert timeout <= 5
        if not running:
            raise click.ClickException(error)
        if args[0] == "ps":
            return searxng.CONTAINER
        if args[0] == "inspect":
            return '[{"Mounts": []}]'
        return ""

    monkeypatch.setattr(uninstall_cli, "find_docker", lambda: "docker")
    monkeypatch.setattr(searxng, "docker", docker)
    lines = uninstall_cli.plan(app, False)
    assert any(
        "Docker is not running" in line and "docker rm -f desman-searxng" in line for line in lines
    )
    assert commands_run == [["info"]]
    spawned: list[object] = []
    uninstall_cli.start(app, False, spawn=lambda argv, **_: spawned.append(argv))
    assert len(spawned) == 1
    assert not capsys.readouterr().out
    running = recovers
    uninstall_cli._finish_cleanup(False)
    if recovers:
        assert commands_run[-1] == ["rm", "-f", "-v", searxng.CONTAINER]
        assert not capsys.readouterr().out
    else:
        assert commands_run[-1] == ["info"]
        [note] = capsys.readouterr().out.splitlines()
        assert "Docker is not running" in note


@pytest.mark.parametrize(
    "error",
    [
        "Error response from daemon: NO SUCH CONTAINER: desman-searxng",
        "Error response from daemon: Removal of container desman-searxng is already in progress",
        "Error response from daemon: permission denied",
    ],
)
def test_container_removal_race(app: Path, error: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uninstall_cli, "find_docker", lambda: "docker")
    monkeypatch.setattr(searxng, "inspect", lambda: {"Mounts": []})
    calls: list[list[str]] = []

    def docker(args: list[str], timeout: float = 60) -> str:
        calls.append(args)
        if args[0] == "rm":
            raise click.ClickException(error)
        return ""

    monkeypatch.setattr(searxng, "docker", docker)
    if "permission denied" in error:
        with pytest.raises(click.ClickException, match="permission denied"):
            uninstall_cli._finish_cleanup(False)
        assert calls[1:] == [["info"]]
    else:
        uninstall_cli._finish_cleanup(False)
        assert len(calls) == 1
    assert calls[0] == ["rm", "-f", "-v", searxng.CONTAINER]


@pytest.mark.parametrize("problem", ["docker", "native", "wsl"])
def test_service_problem_refuses_to_spawn(
    app: Path, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object) -> None:
        raise click.ClickException(f"{problem} unavailable")

    if problem == "docker":
        settings = app / "searxng" / "settings.yml"
        settings.parent.mkdir()
        settings.touch()
        monkeypatch.setattr(uninstall_cli, "find_docker", lambda: "docker")
        monkeypatch.setattr(searxng, "inspect", fail)
        monkeypatch.setattr(searxng, "docker", lambda *_, **__: "")
    elif problem == "native":
        searxng_native.directory().mkdir(parents=True)
        monkeypatch.setattr(searxng_native, "locked", fail)
    else:

        def fail_wsl(_: Sandbox) -> bool:
            raise SandboxUnavailable("wsl unavailable")

        Sandbox().wsl_dir.mkdir(parents=True)
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(shutil, "which", lambda *_: None)
        monkeypatch.setattr(Sandbox, "wsl_distribution_present", fail_wsl)
    with pytest.raises(click.ClickException, match=f"{problem} unavailable"):
        uninstall_cli.start(app, True, spawn=lambda *_, **__: pytest.fail("spawned"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX waiter uses sh")
@pytest.mark.parametrize(("cleanup_code", "installer_code"), [(7, 0), (0, 8), (0, 0)])
def test_waiter_reports_failed_commands(app: Path, cleanup_code: int, installer_code: int) -> None:
    folder = app.parent
    cleanup = folder / "cleanup"
    cleanup.write_text(f"#!/bin/sh\necho cleanup\nexit {cleanup_code}\n", encoding="utf-8")
    cleanup.chmod(0o755)
    script, log = folder / "installer.sh", folder / "uninstall.log"
    script.write_text(f"echo installer\nexit {installer_code}\n", encoding="utf-8")
    pause = folder / "sleep"
    pause.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pause.chmod(0o755)
    data = folder / "missing data"
    env = dict(
        os.environ,
        PATH=f"{folder}:{os.environ['PATH']}",
        DESMAN_HOME=str(app),
        DESMAN_UNINSTALL_DATA=str(data),
        DESMAN_UNINSTALL_LOG=str(log),
        DESMAN_UNINSTALL_PYTHON=str(cleanup),
    )
    argv = uninstall_cli._posix_waiter(2**30, script, log, True)
    result = subprocess.run(argv, env=env, timeout=5, check=False)
    assert result.returncode == (cleanup_code or installer_code)
    assert log.read_text().splitlines() == (
        ["cleanup"] if cleanup_code else ["cleanup", "installer"]
    )
    marker = data / uninstall_cli.FAILURE_MARKER
    if result.returncode:
        assert marker.read_text().strip() == str(log)
    else:
        assert not marker.exists()


def test_bare_desman_warns_once_after_failed_uninstall(
    app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_config().save(make_config(configured=True))
    monkeypatch.setattr(Chat, "run", lambda _: None)
    marker = db.default_db_path().parent / uninstall_cli.FAILURE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    log = app.parent / "logs" / "uninstall.log"
    marker.write_text(str(log), encoding="utf-8")
    runner = CliRunner()
    first = runner.invoke(cli, [], terminal_width=250)
    assert first.exit_code == 0, first.output
    assert f"The last uninstall did not finish. Details: {log}" in first.output
    assert not marker.exists()
    second = runner.invoke(cli, [], terminal_width=250)
    assert second.exit_code == 0, second.output
    assert "The last uninstall did not finish." not in second.output


def test_gui_greeting_warns_once_after_failed_uninstall(app: Path) -> None:
    manager = open_config()
    manager.save(make_config(configured=True))
    marker = db.default_db_path().parent / uninstall_cli.FAILURE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    log = app.parent / "uninstall.log"
    marker.write_text(str(log), encoding="utf-8")
    front = WebFront()
    with ui.use(front), ui.greeting():
        print_project_info(manager)
        print_project_info(manager)
    [warning] = [event for event in front.events if event.get("level") == "warn"]
    assert "The last uninstall did not finish." in warning["html"]
    assert "uninstall.log" in warning["html"]
    assert not marker.exists()
