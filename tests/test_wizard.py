"""The `desman config` wizard, driven through its numbered fallback (no terminal in tests).

Main menu: providers numbered in `providers.PROVIDERS` order (free, paid, local), then the search
engines (`providers.SEARCH`), under section
headings that take no number, then Done. A provider without
keys goes straight to the key prompt; one with keys opens its menu: 1 add, 2 remove, then its
settings, then Back. Permissions follow: four Y/N questions, Enter keeps the current value.
"""

from __future__ import annotations

import json

from click.testing import CliRunner, Result

from desman import providers
from desman.cli import cli
from desman.config import ConfigManager, Permissions, default_config_path
from helpers import flat_keys, make_config

DONE = str(len(providers.PROVIDERS + providers.SEARCH) + 1)
KEEP_PERMISSIONS = ("", "", "", "")


def menu(provider_id: str) -> str:
    return str(1 + [p.id for p in providers.PROVIDERS + providers.SEARCH].index(provider_id))


def wizard(*answers: str) -> Result:
    return CliRunner().invoke(cli, ["config"], input="\n".join(answers) + "\n")


def saved() -> ConfigManager:
    return ConfigManager(default_config_path())


def test_several_providers_and_keys_in_one_run() -> None:
    result = wizard(
        "1",  # English
        menu("groq"), "y", "gsk_first",  # no key yet: privacy once, then the key
        menu("groq"), "1", "gsk_second", "second account", "3",  # add, label, Back
        menu("google"), "AIza_one",  # privacy is not asked again
        DONE,
        *KEEP_PERMISSIONS,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert result.output.count("I understand my data") == 1
    for secret in ("gsk_first", "gsk_second", "AIza_one"):
        assert secret not in result.output
    config = saved().load()
    assert flat_keys(config) == {
        "GROQ_API_KEY": ["gsk_first", "gsk_second"],
        "GOOGLE_API_KEY": "AIza_one",
    }
    assert [k.label for k in config.keys("GROQ_API_KEY")] == ["", "second account"]
    assert config.permissions == Permissions()


def test_keys_that_are_not_touched_stay() -> None:
    saved().save(
        make_config(
            api_keys={"GROQ_API_KEY": "gsk_old", "GOOGLE_API_KEY": "AIza_old"},
            settings={"OLLAMA_HOST": "http://box:11434"},
            permissions={"shell_execution": True},
        )
    )
    result = wizard("1", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    config = saved().load()
    assert flat_keys(config) == {"GROQ_API_KEY": "gsk_old", "GOOGLE_API_KEY": "AIza_old"}
    assert config.settings == {"OLLAMA_HOST": "http://box:11434"}
    assert config.permissions == Permissions(shell_execution=True)  # Enter keeps it


def test_remove_one_key_of_several() -> None:
    saved().save(make_config(api_keys={"GROQ_API_KEY": "gsk_a"}))
    config = saved().load()
    config.add_key("GROQ_API_KEY", "gsk_b")
    saved().save(config)
    result = wizard("1", menu("groq"), "2", "1", "3", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    assert flat_keys(saved().load()) == {"GROQ_API_KEY": "gsk_b"}


def test_duplicate_key_is_not_added_twice() -> None:
    saved().save(make_config(api_keys={"GROQ_API_KEY": "gsk_a"}))
    result = wizard("1", menu("groq"), "1", "gsk_a", "", "3", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    assert "already has this key" in result.output
    assert flat_keys(saved().load()) == {"GROQ_API_KEY": "gsk_a"}


def test_cloudflare_asks_for_its_account_id() -> None:
    result = wizard("1", menu("cloudflare"), "y", "cf_token", "acc123", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    config = saved().load()
    assert flat_keys(config) == {"CLOUDFLARE_API_TOKEN": "cf_token"}
    assert config.settings == {"CLOUDFLARE_ACCOUNT_ID": "acc123"}
    assert [p.id for p in providers.ready(config)] == ["cloudflare"]


def test_local_setting_can_be_cleared() -> None:
    saved().save(make_config(settings={"OLLAMA_HOST": "http://box:11434"}))
    # Ollama's menu: 1 = set Host, 2 = Back.
    result = wizard("1", menu("ollama"), "1", "-", "2", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    assert "OLLAMA_HOST" not in saved().load().settings


def test_declining_privacy_adds_no_key() -> None:
    result = wizard("1", menu("groq"), "n", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    assert saved().load().api_keys == {}
    assert "No cloud key yet" in result.output


def test_permissions_and_the_admin_warning() -> None:
    result = wizard("2", DONE, "y", "", "", "y")  # Russian; admin_rights and shell on
    assert result.exit_code == 0, result.output
    config = saved().load()
    assert config.language == "ru"
    assert config.permissions == Permissions(admin_rights=True, shell_execution=True)
    assert "admin_rights включён" in result.output


def test_leaving_midway_saves_nothing() -> None:
    saved().save(make_config(api_keys={"GROQ_API_KEY": "gsk_a"}))
    before = default_config_path().read_text(encoding="utf-8")
    result = wizard("1", menu("google"), "y")  # input ends at the key prompt, like Ctrl+C
    assert result.exit_code == 1
    assert "nothing was saved" in result.output
    assert default_config_path().read_text(encoding="utf-8") == before


def test_first_run_of_bare_desman_opens_the_wizard() -> None:
    result = CliRunner().invoke(cli, [], input="\n".join(["1", DONE, *KEEP_PERMISSIONS]) + "\n")
    assert result.exit_code == 0, result.output
    assert json.loads(default_config_path().read_text(encoding="utf-8"))["configured"]
    assert "Decentralized Self-Evolving Modular Agents Network" in result.output


def test_free_and_paid_providers_are_listed_apart() -> None:
    result = wizard("1", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    out = result.output
    free, paid, local = (
        out.index("── Free tier"),
        out.index("── Paid: billed per use"),
        out.index("── Local"),
    )
    assert free < out.index("GROQ") < paid < out.index("OpenAI") < local < out.index("Ollama")


def test_a_paid_key_needs_its_own_consent() -> None:
    declined = wizard("1", menu("openai"), "y", "n", DONE, *KEEP_PERMISSIONS)  # privacy, cost
    assert declined.exit_code == 0, declined.output
    assert "Every call to OpenAI is billed to your account" in declined.output
    assert "Paid providers are not tested in the prototype" in declined.output
    assert saved().load().api_keys == {}
    accepted = wizard("1", menu("openai"), "y", "y", "sk-paid", DONE, *KEEP_PERMISSIONS)
    assert accepted.exit_code == 0, accepted.output
    assert flat_keys(saved().load()) == {"OPENAI_API_KEY": "sk-paid"}
