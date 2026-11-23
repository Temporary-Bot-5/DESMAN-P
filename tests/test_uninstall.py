"""Installer detection, preview, and detached handoff without running an installer."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import desman
from desman import commands, runs, sandbox_cli, searxng, searxng_native, ui, uninstall_cli
from desman.chat import Chat
from desman.cli import cli
from desman.config import CONFIG_DIR_ENV, open_config
from desman.core import db
from desman.core.sandbox import Sandbox
from desman.core.sessions import Sessions
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
    assert f"Nothing changed. Confirm with: desman {' '.join(args)} --yes" in result.output
    if purge:
        assert str(db.default_db_path().parent) in result.output
        assert os.environ[CONFIG_DIR_ENV] in result.output
    else:
        assert "Your data, keys and history are kept." in result.output


@pytest.mark.parametrize("platform", ["linux", "win32"])
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
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert str(log) in result.output
    if platform == "win32":
        assert argv[:-1] == ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]
        assert f"Wait-Process -Id {os.getpid()}" in argv[-1]
        assert "Start-Sleep -Seconds 2;" in argv[-1]
        command = argv[-1].split("; ")[-1]
        script_literal, log_literal = (str(path).replace("'", "''") for path in (script, log))
        extra = " -Purge" if purge else ""
        assert command == f"& '{script_literal}' -Uninstall -Yes{extra} *> '{log_literal}'"
        assert kwargs["creationflags"] == 0x8 | 0x200 | 0x08000000
        assert not kwargs["start_new_session"]
    else:
        assert argv[:2] == ["sh", "-c"]
        assert f"kill -0 {os.getpid()}" in argv[-1]
        assert "sleep 2;" in argv[-1]
        args = ["exec", "sh", str(script), "--uninstall", "--yes"]
        if purge:
            args.append("--purge")
        command = argv[-1].split("; ")[-1]
        assert shlex.split(command) == [*args, f">{log}", "2>&1"]
        assert kwargs["start_new_session"]
        assert kwargs["creationflags"] == 0


def test_windows_purge_unregisters_sandbox_before_spawn(
    app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        uninstall_cli.start(app, True, spawn=lambda *_, **__: calls.append("spawn"))
    assert calls == ["remove", "spawn"]


def test_purge_cleans_managed_search(app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docker: list[list[str]] = []
    stopped: list[Path] = []
    native = searxng_native.directory()
    native.mkdir(parents=True)
    monkeypatch.setattr("desman.uninstall_cli.find_docker", lambda: "docker")
    monkeypatch.setattr(searxng, "inspect", lambda: {"Name": "/desman-searxng"})
    monkeypatch.setattr(searxng, "docker", docker.append)
    monkeypatch.setattr(searxng_native, "stop", stopped.append)
    uninstall_cli.start(app, True, spawn=lambda *_, **__: None)
    assert docker == [["rm", "-f", "-v", "desman-searxng"]]
    assert stopped == [native]
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


@pytest.mark.parametrize("line", ["/uninstall --purge", "Remove DESMAN"])
def test_confirmed_uninstall_ends_terminal_chat(
    app: Path, line: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_config().save(make_config(configured=True))
    llm = FakeLLM().on(CHAT, '{"type":"command","name":"uninstall","args":{"purge":true}}')
    monkeypatch.setattr(runs, "models", lambda *_, **__: runs.Models(llm, None))
    monkeypatch.setattr(ui, "confirm", lambda *_, **__: True)
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
