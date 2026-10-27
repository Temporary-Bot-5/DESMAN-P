"""`desman keys ...` and `desman config reset`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from desman import providers
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from helpers import flat_keys, make_config

GROQ = "gsk_" + "g" * 48 + "AbCd"
GROQ_2 = "gsk_" + "h" * 48 + "EfGh"
NVIDIA = "nvapi-" + "n" * 60 + "WxYz"


def desman(*args: str | Path, input: str | None = None) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args], input=input)


def saved() -> ConfigManager:
    return ConfigManager(default_config_path())


def test_add_list_remove() -> None:
    first = desman("keys", "add", "groq", "--stdin", input=GROQ + "\n")
    assert first.exit_code == 0, first.output
    second = desman("keys", "add", "GROQ_API_KEY", "--label", "work", "--stdin", input=GROQ_2)
    assert second.exit_code == 0 and "key #2, 2 in total" in second.output
    again = desman("keys", "add", "GROQ", "--stdin", input=GROQ)
    assert again.exit_code == 0 and "already has this key" in again.output

    listed = desman("keys")
    assert listed.exit_code == 0, listed.output
    assert "gsk_…AbCd" in listed.output and "gsk_…EfGh" in listed.output
    assert "2 key(s) from 1 of" in listed.output
    assert not saved().is_configured()  # adding a key is not running the wizard
    rows = {r["provider"]: r for r in json.loads(desman("keys", "list", "--json").output)}
    assert [k["label"] for k in rows["groq"]["keys"]] == ["", "work"]
    assert rows["groq"]["ready"] and rows["nvidia"]["keys"] == []
    for output in (listed.output, desman("keys", "list", "--json").output):
        assert GROQ not in output and GROQ_2 not in output

    removed = desman("keys", "remove", "groq", "1")
    assert removed.exit_code == 0 and "1 left" in removed.output
    assert flat_keys(saved().load()) == {"GROQ_API_KEY": GROQ_2}
    assert desman("keys", "remove", "groq", "5").exit_code == 2
    assert desman("keys", "remove", "groq").exit_code == 2  # NUMBER or --all
    assert desman("keys", "remove", "groq", "--all").exit_code == 0
    assert saved().load().api_keys == {}
    assert desman("keys", "remove", "groq", "--all").exit_code == 1  # nothing left


def test_add_prompts_for_a_hidden_key() -> None:
    result = desman("keys", "add", "nvidia", input=NVIDIA + "\n")
    assert result.exit_code == 0, result.output
    assert NVIDIA not in result.output
    assert flat_keys(saved().load()) == {"NVIDIA_API_KEY": NVIDIA}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["keys", "add", "nope", "--stdin"], "unknown provider 'nope'"),
        (["keys", "add", "ollama", "--stdin"], "local and has no API key"),
        (["keys", "add", "groq", "--stdin"], "no key given"),
    ],
)
def test_add_refuses(args: list[str], message: str) -> None:
    result = desman(*args, input="\n")
    assert result.exit_code == 2 and message in result.output
    assert not default_config_path().exists()


def test_import_tolerates_the_usual_slips(tmp_path: Path) -> None:
    # The shape a person writes by hand: JSON with missing commas.
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(
        "{\n"
        f'  "GROQ_API_KEY": "{GROQ}",\n'
        f'  "NVIDIA_API_KEY": "{NVIDIA}"\n'
        f'  "GROQ_API_KEY": "{GROQ_2}"\n'
        '  "SOMETHING_ELSE": "x"\n'
        "}\n",
        encoding="utf-8",
    )
    saved().save(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    result = desman("keys", "import", keys_file)
    assert result.exit_code == 0, result.output
    assert "2 key(s) added, 1 already saved" in result.output
    assert "skipped names no provider uses: SOMETHING_ELSE" in result.output
    for secret in (GROQ, GROQ_2, NVIDIA):
        assert secret[4:-4] not in result.output
    assert flat_keys(saved().load()) == {"GROQ_API_KEY": [GROQ, GROQ_2], "NVIDIA_API_KEY": NVIDIA}
    again = desman("keys", "import", keys_file)
    assert "0 key(s) added, 3 already saved" in again.output


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"GROQ_API_KEY": [GROQ, GROQ_2]}),
        f"GROQ_API_KEY={GROQ}\nexport GROQ_API_KEY={GROQ_2}\n# comment\n",
        f"GROQ_API_KEY = '{GROQ}'\nGROQ_API_KEY: \"{GROQ_2}\"\n",
    ],
)
def test_key_file_formats(text: str) -> None:
    assert providers.parse_key_file(text) == [("GROQ_API_KEY", GROQ), ("GROQ_API_KEY", GROQ_2)]


def test_import_of_a_file_without_keys(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("nothing here\n", encoding="utf-8")
    result = desman("keys", "import", empty)
    assert result.exit_code == 1 and "no keys found" in result.output
    assert not default_config_path().exists()


def test_config_reset() -> None:
    saved().save(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    path = default_config_path()
    (path.parent / ".config.json.abcdef123456.tmp").write_text("{}")  # left by a crash

    kept = desman("config", "reset", input="n\n")
    assert kept.exit_code == 1 and path.exists()
    assert "and the 1 API key(s) in it" in kept.output

    done = desman("config", "reset", input="y\n")
    assert done.exit_code == 0, done.output
    assert not path.exists() and list(path.parent.iterdir()) == []
    assert "nothing to delete" in desman("config", "reset", "--yes").output


def test_reset_is_the_way_out_of_a_broken_config() -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert "config reset" in desman("keys").output  # the error points at it
    result = desman("config", "reset", input="y\n")
    assert result.exit_code == 0 and "it does not load" in result.output
    assert not path.exists()
