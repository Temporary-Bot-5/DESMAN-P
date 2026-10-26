"""Tests for the CLI shell and configuration storage."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from desman import __version__
from desman.cli import cli
from desman.config import ConfigManager, default_config_path


def test_default_config_path_honours_env(isolated_config: Path) -> None:
    assert default_config_path() == isolated_config / "config.json"


def test_unconfigured_by_default() -> None:
    manager = ConfigManager(default_config_path())
    assert not manager.is_configured()
    assert manager.get_language() == "en"
    assert manager.get_api_keys() == {}


def test_save_and_load_roundtrip() -> None:
    manager = ConfigManager(default_config_path())
    manager.save_config(
        {
            "language": "ru",
            "api_keys": {"GROQ_API_KEY": "k"},
            "permissions": {"shell_execution": False},
            "local_settings": {"OLLAMA_HOST": "http://localhost:11434"},
        }
    )
    reloaded = ConfigManager(default_config_path())
    assert reloaded.is_configured()
    assert reloaded.get_language() == "ru"
    assert reloaded.get_api_keys() == {"GROQ_API_KEY": "k"}
    assert reloaded.get_permissions() == {"shell_execution": False}
    assert reloaded.get_local_settings() == {"OLLAMA_HOST": "http://localhost:11434"}


def test_corrupt_config_is_treated_as_unconfigured() -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert not ConfigManager(path).is_configured()


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
    ConfigManager(default_config_path()).save_config({"language": "en"})
    result = CliRunner().invoke(cli, [], input="\n")
    assert result.exit_code == 0
    assert "Decentralized Self-Evolving Modular Agents Network" in result.output


# ── Secrets on disk ──────────────────────────────────────────────────────────


def test_config_file_is_private_and_written_atomically() -> None:
    path = default_config_path()
    manager = ConfigManager(path)
    path.write_text("{}", encoding="utf-8")
    os.chmod(path, 0o644)  # e.g. created by an older version
    manager.save_config({"api_keys": {"GROQ_API_KEY": "gsk_secret"}})
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700  # created by us
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]  # no temp file left

    with pytest.raises(TypeError):
        manager.save_config({"api_keys": {"X": object()}})  # cannot be serialised
    assert ConfigManager(path).get_api_keys() == {"GROQ_API_KEY": "gsk_secret"}  # old file intact
    assert [p.name for p in path.parent.iterdir()] == ["config.json"]


def wizard(*answers: str) -> str:
    return "\n".join(answers) + "\n"


NO_PERMISSIONS = ("N", "N", "N", "N")


def test_reconfigure_keeps_keys_that_are_not_reentered() -> None:
    manager = ConfigManager(default_config_path())
    manager.save_config(
        {
            "language": "en",
            "api_keys": {"GROQ_API_KEY": "gsk_old", "GOOGLE_API_KEY": "AIza_old"},
            "local_settings": {"OLLAMA_HOST": "http://box:11434"},
        }
    )
    # English, cloud only, accept privacy note, pick GROQ (3), press Enter at its key prompt.
    result = CliRunner().invoke(
        cli, ["config"], input=wizard("1", "1", "Y", "3", "", *NO_PERMISSIONS)
    )
    assert result.exit_code == 0, result.output
    assert "Enter keeps it" in result.output
    assert "gsk_old" not in result.output  # never echoed
    after = ConfigManager(default_config_path())
    assert after.get_api_keys() == {"GROQ_API_KEY": "gsk_old", "GOOGLE_API_KEY": "AIza_old"}
    assert after.get_local_settings() == {"OLLAMA_HOST": "http://box:11434"}


def test_reconfigure_replaces_or_deletes_a_key() -> None:
    manager = ConfigManager(default_config_path())
    manager.save_config(
        {"language": "en", "api_keys": {"GROQ_API_KEY": "gsk_old", "GOOGLE_API_KEY": "AIza_old"}}
    )
    answers = wizard("1", "1", "Y", "2,3", "-", "gsk_new", *NO_PERMISSIONS)  # Google: delete
    result = CliRunner().invoke(cli, ["config"], input=answers)
    assert result.exit_code == 0, result.output
    assert ConfigManager(default_config_path()).get_api_keys() == {"GROQ_API_KEY": "gsk_new"}
