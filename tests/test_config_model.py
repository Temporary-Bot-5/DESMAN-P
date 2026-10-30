"""Tests for the config schema, the provider list and `desman config show` (ROADMAP 2.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from desman import providers
from desman.cli import cli
from desman.config import ApiKey, Config, ConfigManager, Permissions, default_config_path
from helpers import flat_keys, make_config

GROQ = "gsk_" + "a" * 44 + "Z9x1"
CLOUDFLARE = "cf_" + "b" * 30 + "Q7k2"


def desman(*args: str, input: str | None = None) -> Result:
    return CliRunner().invoke(cli, list(args), input=input)


def write(data: object) -> Path:
    path = default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_schema_0_file_is_upgraded_on_load_and_save() -> None:
    # As written by the wizard before schema_version existed.
    path = write(
        {
            "configured": True,
            "language": "ru",
            "permissions": {"admin_rights": False, "shell_execution": True},
            "api_keys": {"GROQ_API_KEY": GROQ, "VERCEL_AI_GATEWAY_KEY": "vck_x"},
            "local_settings": {"OLLAMA_HOST": "http://box:11434"},
            "version": "0.1.0",
        }
    )
    manager = ConfigManager(path)
    config = manager.load()
    assert config.schema_version == 3 and config.configured and config.language == "ru"
    assert config.permissions == Permissions(shell_execution=True)
    assert flat_keys(config) == {"GROQ_API_KEY": GROQ, "AI_GATEWAY_API_KEY": "vck_x"}
    assert config.settings == {"OLLAMA_HOST": "http://box:11434"}

    manager.save(config)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 3
    assert "version" not in saved and "local_settings" not in saved
    assert ConfigManager(path).load() == config


def test_missing_file_is_an_unconfigured_default() -> None:
    config = ConfigManager(default_config_path()).load()
    assert config == Config() and not config.configured
    assert config.permissions == Permissions()  # everything off, admin_rights included
    assert not config.permissions.admin_rights


def test_provider_readiness() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "CLOUDFLARE_API_TOKEN": CLOUDFLARE})
    assert [p.id for p in providers.ready(config)] == ["groq"]  # Cloudflare lacks its account
    config.settings["CLOUDFLARE_ACCOUNT_ID"] = "acc123"
    assert [p.id for p in providers.ready(config)] == ["groq", "cloudflare"]
    assert providers.ready(make_config(api_keys={"GROQ_API_KEY": ""})) == []  # "" meant none
    names = [p.key_env for p in providers.CLOUD] + [
        s.env for p in providers.PROVIDERS for s in p.settings
    ]
    assert len(names) == len(set(names))  # one variable, one provider


@pytest.mark.parametrize(
    ("secret", "shown"),
    [("short", "****"), ("x" * 19, "****"), (GROQ, "gsk_…Z9x1"), ("", "****")],
)
def test_mask(secret: str, shown: str) -> None:
    assert providers.mask(secret) == shown


def test_config_show_masks_every_secret() -> None:
    ConfigManager(default_config_path()).save(
        Config(
            language="en",
            permissions=Permissions(admin_rights=True),
            api_keys={
                "GROQ_API_KEY": [ApiKey(value=GROQ), ApiKey(value=GROQ[::-1], label="spare")],
                "CLOUDFLARE_API_TOKEN": [ApiKey(value=CLOUDFLARE)],
                "GROK_KEY": [ApiKey(value=GROQ)],
            },
            settings={"OLLAMA_HOST": "http://box:11434"},
        )
    )
    text = desman("config", "show")
    assert text.exit_code == 0, text.output
    js = desman("config", "show", "--json")
    assert js.exit_code == 0, js.output
    for output in (text.output, js.output):
        for secret in (GROQ, GROQ[::-1], CLOUDFLARE):
            assert secret[4:-4] not in output
    assert "WARNING: admin_rights is on" in text.output
    assert "unknown key GROK_KEY=gsk_…Z9x1" in text.output
    lines = {line.split()[0]: line for line in text.output.splitlines() if line.strip()}
    assert "ready" in lines["GROQ"] and "gsk_…Z9x1  1x9Z…_ksg (spare)" in lines["GROQ"]
    assert "not set" in lines["Cloudflare"] and "CLOUDFLARE_ACCOUNT_ID=-" in lines["Cloudflare"]
    assert "OLLAMA_HOST=http://box:11434" in lines["Ollama"]

    out = json.loads(js.output)
    assert out["path"] == str(default_config_path()) and out["schema_version"] == 3
    rows = {r["id"]: r for r in out["providers"]}
    assert rows["groq"]["key_env"] == "GROQ_API_KEY"
    assert rows["groq"]["keys"] == [
        {"value": "gsk_…Z9x1", "label": ""},
        {"value": "1x9Z…_ksg", "label": "spare"},
    ]
    assert rows["groq"]["ready"] and not rows["cloudflare"]["ready"]
    assert rows["mistral"]["keys"] == []
    assert out["unknown_keys"] == {"GROK_KEY": ["gsk_…Z9x1"]}


def test_config_show_when_unconfigured_does_not_create_the_file() -> None:
    result = desman("config", "show")
    assert result.exit_code == 0, result.output
    assert "Not configured yet" in result.output
    assert not default_config_path().exists()


# ── Review regressions (PR #16) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "keys",
    [
        {"AI_GATEWAY_API_KEY": "vck_new", "VERCEL_AI_GATEWAY_KEY": "vck_old"},
        {"VERCEL_AI_GATEWAY_KEY": "vck_old", "AI_GATEWAY_API_KEY": "vck_new"},
    ],
)
def test_upgrade_keeps_the_key_under_the_new_name(keys: dict[str, str]) -> None:
    config = ConfigManager(write({"api_keys": keys, "version": "0.1.0"})).load()
    assert flat_keys(config) == {"AI_GATEWAY_API_KEY": "vck_new"}


def test_a_rejected_key_is_not_quoted_in_the_error() -> None:
    key = "sk-" + "q" * 20 + "W3e4"  # short enough that pydantic would quote it whole
    write({"schema_version": 1, "api_keys": {"GROQ_API_KEY": [key]}})
    result = desman("config", "show")
    assert result.exit_code == 1 and "api_keys.GROQ_API_KEY" in result.output
    assert "q" * 20 not in result.output


def test_unknown_settings_are_masked_too() -> None:
    ConfigManager(default_config_path()).save(Config(settings={"GROQ_KEY": GROQ}))
    for args in (["config", "show"], ["config", "show", "--json"]):
        result = desman(*args)
        assert result.exit_code == 0 and GROQ not in result.output, args
    assert "unknown setting GROQ_KEY=gsk_…Z9x1" in desman("config", "show").output


# ── Schema 2: several keys per provider ──────────────────────────────────────


def test_schema_1_file_becomes_a_list_of_keys() -> None:
    path = write({"schema_version": 1, "api_keys": {"GROQ_API_KEY": GROQ, "NVIDIA_API_KEY": ""}})
    config = ConfigManager(path).load()
    assert config.api_keys == {"GROQ_API_KEY": [ApiKey(value=GROQ)]}


def test_add_and_remove_keys() -> None:
    config = Config()
    assert config.add_key("GROQ_API_KEY", f"  {GROQ} ")
    assert not config.add_key("GROQ_API_KEY", GROQ)  # the same key twice is one key
    assert config.add_key("GROQ_API_KEY", "gsk_two", "second")
    assert [(k.value, k.label) for k in config.keys("GROQ_API_KEY")] == [
        (GROQ, ""),
        ("gsk_two", "second"),
    ]
    with pytest.raises(ValueError, match="empty"):
        config.add_key("GROQ_API_KEY", "  ")
    assert config.remove_key("GROQ_API_KEY", 0).value == GROQ
    with pytest.raises(IndexError, match="no key #2"):
        config.remove_key("GROQ_API_KEY", 1)
    config.remove_key("GROQ_API_KEY", 0)
    assert config.api_keys == {}  # no empty list left behind


@pytest.mark.parametrize("bad", [[{"value": ""}], [{"value": "k", "extra": 1}], "k"])
def test_bad_key_entries_are_rejected(bad: object) -> None:
    write({"schema_version": 2, "api_keys": {"GROQ_API_KEY": bad}})
    assert desman("config", "show").exit_code == 1
