"""`desman keys ...` and `desman config reset`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from desman import providers
from desman.cli import cli
from desman.config import CONFIG_DIR_ENV, ApiKey, ConfigError, ConfigManager, default_config_path
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
    # Only the lock is left: deleting a lock file others may hold would defeat it.
    assert [f.name for f in path.parent.iterdir()] == [".config.json.lock"]
    assert "nothing to delete" in desman("config", "reset", "--yes").output


def test_reset_is_the_way_out_of_a_broken_config() -> None:
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")
    assert "config reset" in desman("keys").output  # the error points at it
    result = desman("config", "reset", input="y\n")
    assert result.exit_code == 0 and "it does not load" in result.output
    assert not path.exists()


# ── Review regressions (PR #17) ──────────────────────────────────────────────


def test_concurrent_key_adds_both_survive(tmp_path: Path) -> None:
    """Many `keys add` processes at once: each one's key is kept (read-modify-write locked)."""
    env = {**os.environ, CONFIG_DIR_ENV: str(default_config_path().parent)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "desman", "keys", "add", "groq", "--stdin"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env=env,
        )
        for _ in range(8)
    ]
    for i, proc in enumerate(procs):
        assert proc.stdin is not None
        proc.stdin.write(f"gsk_concurrent_{i}\n".encode())
        proc.stdin.close()
    assert [proc.wait(timeout=60) for proc in procs] == [0] * 8
    values = sorted(k.value for k in saved().load().keys("GROQ_API_KEY"))
    assert values == sorted(f"gsk_concurrent_{i}" for i in range(8))


def test_wizard_does_not_write_over_a_change_made_meanwhile() -> None:
    manager = saved()
    manager.save(make_config(api_keys={"GROQ_API_KEY": "gsk_a"}))
    seen = manager.snapshot()
    with manager.edit() as config:  # e.g. `desman keys add` while the wizard is open
        config.add_key("GROQ_API_KEY", "gsk_b")
    with pytest.raises(ConfigError, match="changed while this was open"):
        manager.save_if_unchanged(make_config(), seen)
    assert flat_keys(saved().load()) == {"GROQ_API_KEY": ["gsk_a", "gsk_b"]}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('# GROQ_API_KEY="retired"\nGOOGLE_API_KEY=active', [("GOOGLE_API_KEY", "active")]),
        (
            "GROQ_API_KEY=active # primary\nGOOGLE_API_KEY=other",
            [("GROQ_API_KEY", "active"), ("GOOGLE_API_KEY", "other")],
        ),
        ('// "GROQ_API_KEY": "old",\n  "GROQ_API_KEY": "new",', [("GROQ_API_KEY", "new")]),
        (
            '{"GROQ_API_KEY": "a"\n"NVIDIA_API_KEY": "b"}',
            [("GROQ_API_KEY", "a"), ("NVIDIA_API_KEY", "b")],
        ),
        ('note: GROQ_API_KEY="in prose"', []),
    ],
)
def test_key_file_comments(text: str, expected: list[tuple[str, str]]) -> None:
    assert providers.parse_key_file(text) == expected


@pytest.mark.skipif(os.name != "posix", reason="symlinks")
def test_reset_of_a_linked_config_keeps_the_target(tmp_path: Path) -> None:
    target = tmp_path / "dotfiles" / "desman.json"
    target.parent.mkdir()
    ConfigManager(target).save(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    link = default_config_path()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    result = desman("config", "reset", "--yes")
    assert result.exit_code == 0, result.output
    assert not link.is_symlink() and target.exists()
    assert "only the link is gone" in result.output


def test_a_label_that_looks_like_a_key_is_refused_and_masked() -> None:
    result = desman("keys", "add", "groq", "--label", GROQ_2, "--stdin", input=GROQ)
    assert result.exit_code == 2 and "looks like a key" in result.output
    assert not default_config_path().exists() or saved().load().api_keys == {}
    assert desman("keys", "add", "groq", "--label", "gsk", "--stdin", input=GROQ).exit_code == 2

    # A label written by hand into the file is masked on display.
    saved().save(make_config())
    with saved().edit() as config:
        config.api_keys["GROQ_API_KEY"] = [ApiKey(value=GROQ, label=GROQ_2)]
    for args in (
        ["keys"],
        ["keys", "list", "--json"],
        ["config", "show"],
        ["config", "show", "--json"],
    ):
        output = desman(*args).output
        assert GROQ_2[4:-4] not in output, args
