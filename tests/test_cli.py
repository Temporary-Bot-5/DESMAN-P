"""Tests for the CLI shell and configuration storage."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

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
from helpers import flat_keys, make_config


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
        ('{"schema_version": 7}', "newer than this DESMAN"),
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
    assert json.loads(target.read_text())["api_keys"] == {
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
    names = ["config.json", ".config.json.0123456789ab.tmp"]
    done = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", *names],
        capture_output=True,
        text=True,
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
    result = CliRunner().invoke(cli, ["config"], input="1\n12\n\n\n\n\n")  # en, Done, keep
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
