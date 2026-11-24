"""Tests for the CLI shell and configuration storage."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import subprocess
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from desman import __version__, config
from desman.cli import cli
from desman.config import (
    CONFIG_DIR_ENV,
    ConfigError,
    ConfigManager,
    Permissions,
    default_config_path,
    open_config,
    project_root,
)
from desman.core import db
from helpers import flat_keys, make_config


@pytest.mark.parametrize(
    ("args", "language", "custom_path"),
    [
        (["usage"], "en", False),
        (["status", "task"], "en", False),
        (["list"], "en", False),
        (["run", "Say hello"], "en", False),
        (["worker"], "en", False),
        ([], "en", False),
        (["gui", "--no-browser"], "en", False),
        (["init-db"], "en", False),
        (["usage"], "ru", True),
    ],
)
def test_newer_database_reports_one_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    language: str,
    custom_path: bool,
) -> None:
    monkeypatch.setattr("desman.gui.app.http_server", Mock())
    open_config().save(
        make_config(
            configured=True,
            interface="terminal",
            language=language,
            api_keys={"GROQ_API_KEY": "gsk_test"},
        )
    )
    path = tmp_path / "custom.db" if custom_path else db.default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 5}")

    result = CliRunner().invoke(cli, [*args, *(["--db", str(path)] if custom_path else [])])
    assert result.exit_code == 1, (result.output, result.exception)
    assert isinstance(result.exception, SystemExit)
    assert result.output.count(str(path)) == 1
    assert "Traceback" not in result.output
    assert "DESMAN_DATA_DIR" in result.output
    assert ("Update DESMAN" if language == "en" else "Обновите DESMAN") in result.output


def test_default_config_path_honours_env(isolated_config: Path) -> None:
    assert default_config_path() == isolated_config / "config.json"


def test_unconfigured_by_default() -> None:
    manager = ConfigManager(default_config_path())
    assert not manager.is_configured()
    assert manager.load().language == "en"
    assert flat_keys(manager.load()) == {}


def test_save_and_load_roundtrip() -> None:
    manager = ConfigManager(default_config_path())
    manager.save(
        make_config(
            **{
                "configured": True,
                "language": "ru",
                "api_keys": {"GROQ_API_KEY": "k"},
                "permissions": {"shell_execution": False},
                "settings": {"OLLAMA_HOST": "http://localhost:11434"},
            }
        )
    )
    reloaded = ConfigManager(default_config_path())
    assert reloaded.is_configured()
    assert reloaded.load().language == "ru"
    assert flat_keys(reloaded.load()) == {"GROQ_API_KEY": "k"}
    assert reloaded.load().permissions == Permissions()
    assert reloaded.load().settings == {"OLLAMA_HOST": "http://localhost:11434"}


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("{not json", "not valid JSON"),
        ("[]", "not a JSON object"),
        (json.dumps({"schema_version": config.SCHEMA_VERSION + 1}), "newer than this DESMAN"),
        ('{"schema_version": "1"}', "whole number"),
        ('{"schema_version": 1, "permissions": {"admin_rights": "yes"}}', "admin_rights"),
        ('{"schema_version": 1, "api_keys": {"GROQ_API_KEY": 5}}', "GROQ_API_KEY"),
        ('{"schema_version": 1, "colour": "red"}', "colour"),
    ],
)
def test_broken_config_is_an_error_not_unconfigured(text: str, reason: str) -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=reason):
        ConfigManager(path).load()
    for args in (["config"], ["config", "show"], []):
        result = CliRunner().invoke(cli, args, input="1\n")
        assert result.exit_code == 1 and "Fix the file" in result.output, args
    assert path.read_text(encoding="utf-8") == text  # the wizard never wrote over it


def test_help_is_non_interactive() -> None:
    # Subcommands must never trigger the setup wizard, even when unconfigured.
    result = CliRunner().invoke(cli, ["help"], input="")
    assert result.exit_code == 0
    assert "config" in result.output
    assert not default_config_path().exists()


def test_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_bare_command_shows_banner_when_configured() -> None:
    ConfigManager(default_config_path()).save(make_config(**{"language": "en", "configured": True}))
    result = CliRunner().invoke(cli, [], input="\n")
    assert result.exit_code == 0
    assert "Decentralized Self-Evolving Modular Agents Network" in result.output


# ── Secrets on disk ──────────────────────────────────────────────────────────


def test_config_file_is_private_and_written_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = default_config_path()
    manager = ConfigManager(path)
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o644)  # e.g. created by an older version
    manager.save(make_config(**{"api_keys": {"GROQ_API_KEY": "gsk_secret"}}))
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700  # created by us
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]  # no temp file left

    def crash(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        manager.save(make_config(api_keys={"GROQ_API_KEY": "gsk_other"}))
    assert flat_keys(ConfigManager(path).load()) == {
        "GROQ_API_KEY": "gsk_secret"
    }  # old file intact
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]


# ── Review regressions (PR #13) ──────────────────────────────────────────────


@pytest.mark.skipif(os.name != "posix", reason="symlinks and modes are POSIX here")
def test_symlinked_config_is_updated_at_its_target(tmp_path: Path) -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    target = dotfiles / "desman.json"
    target.write_text("{}", encoding="utf-8")
    path.symlink_to(target)
    ConfigManager(path).save(make_config(**{"api_keys": {"GROQ_API_KEY": "gsk_new"}}))
    assert path.is_symlink() and path.resolve() == target
    assert json.loads(target.read_text(encoding="utf-8"))["api_keys"] == {
        "GROQ_API_KEY": [{"value": "gsk_new", "label": ""}]
    }
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert [p.name for p in dotfiles.iterdir()] == ["desman.json"]


@pytest.mark.skipif(os.name != "posix", reason="hard links and modes are POSIX here")
def test_old_hard_link_does_not_stay_world_readable(tmp_path: Path) -> None:
    path = default_config_path()
    manager = ConfigManager(path)
    path.write_text('{"api_keys": {"GROQ_API_KEY": "gsk_old"}}', encoding="utf-8")
    os.chmod(path, 0o644)
    backup = tmp_path / "backup.json"
    os.link(path, backup)
    manager.save(make_config(**{"api_keys": {"GROQ_API_KEY": "gsk_new"}}))
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


# ── Config location (hotfix/config-in-project) ───────────────────────────────


def test_config_lives_in_the_project_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV)
    root = project_root()
    assert root is not None and (root / "pyproject.toml").is_file()  # tests run from a checkout
    assert default_config_path() == root / "config.json"


def test_installed_package_falls_back_to_the_home_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(CONFIG_DIR_ENV)
    monkeypatch.setattr(config, "project_root", lambda: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert default_config_path() == tmp_path / ".config" / "desman" / "config.json"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_ignores_the_config_and_its_temp_files() -> None:
    root = project_root()
    assert root is not None
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    names = [
        "config.json",
        ".config.json.0123456789ab.tmp",
        ".config.json.lock",
        "searxng/settings.yml",
    ]
    done = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", *names],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.stdout.split() == names, done.stderr


def checkout_with_home_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """A checkout at tmp_path/project and a config saved under the old home location."""
    monkeypatch.delenv(CONFIG_DIR_ENV)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(config, "project_root", lambda: project)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    old = tmp_path / "home" / ".config" / "desman" / "config.json"
    old.parent.mkdir(parents=True)
    ConfigManager(old).save(
        make_config(
            **{
                "configured": True,
                "api_keys": {"GROQ_API_KEY": "gsk_old"},
                "permissions": {"shell_execution": True},
            }
        )
    )
    return old, project / "config.json"


def test_home_config_moves_into_the_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, new = checkout_with_home_config(monkeypatch, tmp_path)
    manager = open_config()
    assert manager.config_path == new and not old.exists()
    assert manager.is_configured()
    assert flat_keys(manager.load()) == {"GROQ_API_KEY": "gsk_old"}
    assert manager.load().permissions.shell_execution
    if os.name == "posix":
        assert stat.S_IMODE(new.stat().st_mode) == 0o600
    from test_wizard import DONE, KEEP_PERMISSIONS, KEEP_PERSONAL, wizard

    result = wizard("1", DONE, *KEEP_PERMISSIONS, *KEEP_PERSONAL, "")
    assert result.exit_code == 0, result.output
    assert flat_keys(open_config().load()) == {"GROQ_API_KEY": "gsk_old"}


def test_checkout_config_wins_over_the_home_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old, new = checkout_with_home_config(monkeypatch, tmp_path)
    ConfigManager(new).save(make_config(**{"api_keys": {"GROQ_API_KEY": "gsk_new"}}))
    assert flat_keys(open_config().load()) == {"GROQ_API_KEY": "gsk_new"}
    assert old.exists()  # left for the user, never merged or overwritten
    config.adopt_config(old, new)  # even if the new file appears after the check
    assert flat_keys(ConfigManager(new).load()) == {"GROQ_API_KEY": "gsk_new"} and old.exists()


def test_windows_streams_write_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    from desman.stdio import configure_output, utf8_stream

    output = io.BytesIO()
    errors = io.BytesIO()
    with (
        monkeypatch.context() as patch,
        io.TextIOWrapper(output, encoding="cp1252") as stdout,
        io.TextIOWrapper(errors, encoding="ascii") as stderr,
        io.TextIOWrapper(io.BytesIO("ключ\n".encode()), encoding="cp1252") as stdin,
    ):
        patch.setattr(sys, "platform", "win32")
        patch.setattr(sys, "stdout", stdout)
        patch.setattr(sys, "stderr", stderr)
        configure_output()
        stdout.write("✓")
        stderr.write("✓")
        stdout.flush()
        stderr.flush()
        assert output.getvalue() == "✓".encode()
        assert errors.getvalue() == "✓".encode()
        utf8_stream(stdin)
        assert stdin.readline() == "ключ\n"


@pytest.mark.parametrize("mode", ["terminal", "gui"])
def test_bare_desman_uses_configured_interface(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from unittest.mock import Mock

    from desman.chat import Chat
    from desman.gui import app

    open_config().save(make_config(configured=True, interface=mode))
    gui_run, chat_run = Mock(), Mock()
    monkeypatch.setattr(app, "run", gui_run)
    monkeypatch.setattr(Chat, "run", chat_run)
    result = CliRunner().invoke(cli, ["-c"])
    assert result.exit_code == 0, result.output
    if mode == "gui":
        gui_run.assert_called_once_with(True, False, 0, True, session_id=None)
        chat_run.assert_not_called()
    else:
        chat_run.assert_called_once_with()
        gui_run.assert_not_called()


@pytest.mark.parametrize("startup_failure", [True, False])
def test_bare_desman_falls_back_only_when_page_server_cannot_start(
    monkeypatch: pytest.MonkeyPatch, startup_failure: bool
) -> None:
    import errno
    from unittest.mock import Mock

    import click

    from desman.chat import Chat
    from desman.gui import app

    open_config().save(make_config(configured=True, interface="gui"))
    chat_run = Mock()
    monkeypatch.setattr(Chat, "run", chat_run)
    if startup_failure:
        monkeypatch.setattr(
            app, "http_server", Mock(side_effect=OSError(errno.EADDRINUSE, "Address in use"))
        )
    else:
        monkeypatch.setattr(app, "run", Mock(side_effect=click.ClickException("Page failed")))

    result = CliRunner().invoke(cli, [])
    if startup_failure:
        assert result.exit_code == 0, result.output
        assert "The page could not start: Address in use." in result.output
        assert "--port" not in result.output
        chat_run.assert_called_once_with()
    else:
        assert result.exit_code == 1
        assert "Page failed" in result.output
        chat_run.assert_not_called()


@pytest.mark.parametrize("launcher,tty", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("unexpected", [False, True])
def test_launcher_pauses_on_errors_only_in_a_terminal(
    monkeypatch: pytest.MonkeyPatch, launcher: bool, tty: bool, unexpected: bool
) -> None:
    import sys
    from unittest.mock import Mock

    import click

    from desman import ui
    from desman.chat import Chat

    open_config().save(make_config(configured=True))
    error = RuntimeError("launch failed") if unexpected else click.ClickException("launch failed")
    monkeypatch.setattr(Chat, "run", Mock(side_effect=error))
    wait = Mock()
    monkeypatch.setattr(ui, "text", wait)
    runner = CliRunner()
    with runner.isolation() as (_, stderr, _):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: tty)
        # Calling Click directly here keeps the test's stdin in place.
        if unexpected and not (launcher and tty):
            with pytest.raises(RuntimeError, match="launch failed"):
                cli.main(["--launcher"] if launcher else [])
        else:
            with pytest.raises(SystemExit, match="1"):
                cli.main(["--launcher"] if launcher else [])
            assert b"launch failed" in stderr.getvalue()
    if launcher and tty:
        wait.assert_called_once_with("Press Enter to close.")
    else:
        wait.assert_not_called()


@pytest.mark.parametrize("exit_kind", ["normal", "cancelled", "interrupt"])
def test_launcher_does_not_pause_on_normal_exit(
    monkeypatch: pytest.MonkeyPatch, exit_kind: str
) -> None:
    import sys
    from unittest.mock import Mock

    from desman import ui
    from desman.chat import Chat

    open_config().save(make_config(configured=True))
    exits = {"normal": None, "cancelled": ui.Cancelled, "interrupt": KeyboardInterrupt}
    monkeypatch.setattr(Chat, "run", Mock(side_effect=exits[exit_kind]))
    wait = Mock()
    monkeypatch.setattr(ui, "text", wait)
    with CliRunner().isolation():
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        with pytest.raises(SystemExit, match="1" if exit_kind == "interrupt" else "0"):
            cli.main(["--launcher"])
    wait.assert_not_called()


def test_interface_command_and_catalog() -> None:
    from desman import commands
    from desman.core import db

    runner = CliRunner()
    assert "Default interface: Terminal" in runner.invoke(cli, ["interface"]).output
    for mode, message in (
        ("gui", "Default interface: Experimental GUI"),
        ("gui", "Nothing changed."),
        ("terminal", "Default interface: Terminal"),
    ):
        result = runner.invoke(cli, ["interface", mode])
        assert result.exit_code == 0, result.output
        assert message in result.output
        assert open_config().load().interface == mode
    assert not db.default_db_path().exists()
    parsed = commands.parse_line("interface gui", open_config().load())
    assert parsed is not None
    entry, args = parsed
    assert entry.name == "interface"
    assert entry.kind_of(args) == commands.Kind.CHANGE
