"""Tests for the CLI shell and configuration storage."""

from __future__ import annotations

from pathlib import Path

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
