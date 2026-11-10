"""Tests for consent to desman-global (ROADMAP G.3, ADR 0004 decisions 4–5, 12): the `network`
section of the config, `desman global ...`, the wizard's last step, the chat's windows and the
network's journal."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from desman import commands, net, providers, runs
from desman.cli import cli
from desman.config import (
    Config,
    ConfigManager,
    Donation,
    Network,
    default_config_path,
    upgrade,
)
from desman.core import db, netlog
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from desman.net import sign
from helpers import make_config
from test_commands import asks, chat, journal
from test_wizard import DONE, KEEP_PERMISSIONS, wizard

REGISTRY = "https://registry.example.org"
OLLAMA = {"OLLAMA_MODEL": "llama3.2"}
LENT = Donation(calls_per_day=100, tokens_per_day=100_000)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    return llm


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def ok(*args: str) -> Result:
    result = desman(*args)
    assert result.exit_code == 0, result.output
    return result


def refused(*args: str) -> str:
    result = desman(*args)
    assert result.exit_code == 1, result.output
    return result.output


def save(config: Config) -> None:
    ConfigManager(default_config_path()).save(config)


def saved() -> Config:
    return ConfigManager(default_config_path()).load()


def configured(**data: Any) -> None:
    save(make_config(language="en", configured=True, **data))


def network_log() -> list[tuple[str, dict[str, Any]]]:
    conn = db.open_db(db.default_db_path())
    try:
        return [(e.action, e.details) for e in netlog.entries(conn)]
    finally:
        conn.close()


def node_key() -> Path:
    return sign.default_node_key_path()


# ── The config ───────────────────────────────────────────────────────────────


def test_desman_global_is_off_by_default_and_has_nowhere_to_send() -> None:
    config = Config()
    assert config.network == Network() and not config.network.enabled
    assert net.registry(config) is None
    off = Config(network=Network(registry=REGISTRY, donated={"ollama": LENT}))
    assert net.registry(off) is None


def test_a_schema_4_config_upgrades_to_desman_global_off() -> None:
    data = make_config(language="ru", configured=True).model_dump()
    data["schema_version"] = 4
    del data["network"]
    config = Config.model_validate(upgrade(data))
    assert config.schema_version == 9 and config.network == Network()
    assert config.language == "ru"


@pytest.mark.parametrize(
    "url",
    [
        REGISTRY,
        "https://registry.example.org:8443/desman",
        "http://localhost:8000",
        "http://127.0.0.1:9000/api",
    ],
)
def test_a_registry_is_https_or_this_machine(url: str) -> None:
    assert Network(registry=url).registry == url


@pytest.mark.parametrize(
    "url",
    ["http://registry.example.org", "ftp://x", "https://", "registry.example.org", "https://a b"],
)
def test_anything_else_is_not_a_registry(url: str) -> None:
    with pytest.raises(ValueError):
        Network(registry=url)


def test_a_quota_has_bounds() -> None:
    Donation(calls_per_day=1, tokens_per_day=1000)
    for bad in ({"calls_per_day": 0}, {"tokens_per_day": 999}, {"calls_per_day": 10**6}):
        with pytest.raises(ValueError):
            Donation.model_validate({"calls_per_day": 100, "tokens_per_day": 1000, **bad})


# ── What may be lent ─────────────────────────────────────────────────────────


def test_no_paid_provider_and_no_service_is_ever_lent() -> None:
    for p in providers.PAID + providers.SERVICES:
        assert not providers.can_donate(p), p.id
    for p in providers.PAID:
        assert not p.donatable, f"{p.id} is paid: it is never lent"


def test_local_servers_are_lent_and_cloud_providers_only_once_checked() -> None:
    for p in providers.LOCAL:
        assert providers.can_donate(p)
    for p in providers.FREE:
        assert providers.can_donate(p) == p.donatable


# ── desman global ────────────────────────────────────────────────────────────


def test_status_says_it_is_off() -> None:
    configured()
    result = ok("global", "status")
    assert "nothing leaves this machine" in result.output
    out = json.loads(ok("global", "status", "--json").output)
    assert out == {
        "enabled": False,
        "registry": None,
        "node": None,
        "donated": {},
        "lead": True,
        "cluster": None,
    }
    assert ok("global").output == result.output
    assert not node_key().exists()


def test_join_needs_a_registry_address() -> None:
    configured()
    assert "No registry address is known yet" in refused("global", "join")
    assert "is not a registry address" in refused(
        "global", "join", "--registry", "http://example.org"
    )
    assert not saved().network.enabled and not node_key().exists()
    assert network_log() == []


def test_join_creates_the_node_key_and_switches_it_on() -> None:
    configured()
    result = ok("global", "join", "--registry", REGISTRY)
    assert f"Taking part in desman-global, registry {REGISTRY}." in result.output
    key = sign.load_node_key(node_key())
    assert key.fingerprint in result.output
    if os.name != "nt":
        assert stat.S_IMODE(node_key().stat().st_mode) == 0o600
    secret = node_key().read_text()
    assert secret.splitlines()[1] not in result.output
    network = saved().network
    assert network.enabled and network.registry == REGISTRY
    assert net.registry(saved()) == REGISTRY
    assert network_log() == [("joined", {"node": key.fingerprint, "registry": REGISTRY})]
    # Again: nothing changes, the same key.
    again = json.loads(ok("global", "join", "--json").output)
    assert again == {"changed": False, "registry": REGISTRY, "node": key.fingerprint}
    assert len(network_log()) == 1
    status = json.loads(ok("global", "status", "--json").output)
    assert status["enabled"] and status["node"] == key.fingerprint


def test_the_shipped_registry_is_used_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    configured()
    monkeypatch.setattr(net, "DEFAULT_REGISTRY", REGISTRY)
    ok("global", "join")
    assert saved().network.registry is None  # follows the shipped address
    assert net.registry(saved()) == REGISTRY


def test_donate_lends_a_local_server_with_a_quota() -> None:
    configured(settings=OLLAMA)
    assert "join it first" in refused("global", "donate", "ollama")
    ok("global", "join", "--registry", REGISTRY)
    result = ok("global", "donate", "ollama")
    assert "Ollama lent: 100 calls and 100000 tokens a day." in result.output
    assert saved().network.donated == {"ollama": LENT}
    result = ok("global", "quota", "Ollama", "--calls", "50")
    assert "100 calls and 100000 tokens a day → 50 calls and 100000 tokens a day" in result.output
    shown = json.loads(ok("global", "quota", "ollama", "--json").output)
    assert shown == {"provider": "ollama", "quota": {"calls_per_day": 50, "tokens_per_day": 100000}}
    assert "Nothing changed." in ok("global", "donate", "ollama", "--calls", "50").output
    assert "1–100000 calls" in refused("global", "quota", "ollama", "--tokens", "5")
    result = ok("global", "undonate", "ollama")
    assert "Ollama is not lent any more." in result.output
    assert saved().network.donated == {}
    assert "is not lent" in refused("global", "quota", "ollama")
    assert [action for action, _ in network_log()] == ["joined", "donated", "quota", "undonated"]
    assert network_log()[2][1] == {
        "provider": "ollama",
        "calls_per_day": 50,
        "tokens_per_day": 100000,
    }


def test_what_cannot_be_lent_is_refused() -> None:
    configured(api_keys={"GROQ_API_KEY": "gsk_" + "a" * 40, "ANTHROPIC_API_KEY": "sk-ant-x"})
    ok("global", "join", "--registry", REGISTRY)
    assert "has not been checked for serving others" in refused("global", "donate", "groq")
    paid = providers.PAID[0]
    assert "paid provider" in refused("global", "donate", paid.id)
    assert "not set up yet" in refused("global", "donate", "llamacpp")
    assert "unknown provider" in refused("global", "donate", "nobody")
    assert saved().network.donated == {}


def test_leave_takes_everything_back_and_keeps_the_key() -> None:
    configured(settings=OLLAMA)
    assert "off already" in ok("global", "leave").output
    ok("global", "join", "--registry", REGISTRY)
    ok("global", "donate", "ollama")
    fingerprint = sign.load_node_key(node_key()).fingerprint
    result = ok("global", "leave")
    assert "Taken back: Ollama" in result.output and "The node's key stays" in result.output
    network = saved().network
    assert not network.enabled and network.donated == {} and network.registry == REGISTRY
    assert net.registry(saved()) is None
    assert sign.load_node_key(node_key()).fingerprint == fingerprint
    # Saved off first, then the goodbye (no registry to reach in tests).
    assert network_log()[-2] == ("left", {"withdrawn": ["ollama"]})
    assert network_log()[-1][0] == "leave_sent" and network_log()[-1][1]["heard"] is False
    # Joining again: the same key.
    assert fingerprint in ok("global", "join").output


def test_only_leaving_reaches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Of all these commands only `leave` sends anything: its goodbye to the registry, once
    the config is saved off (G.5). The rest is the worker's (tests/test_node.py)."""
    sent: list[tuple[str, str]] = []

    def record(self: Any, request: httpx.Request) -> httpx.Response:
        sent.append((request.method, str(request.url)))
        raise httpx.ConnectError("not in tests", request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", record)
    configured(settings=OLLAMA)
    for args in (
        ("global", "status"),
        ("global", "join", "--registry", REGISTRY),
        ("global", "donate", "ollama"),
        ("global", "quota", "ollama", "--tokens", "5000"),
        ("global", "undonate", "ollama"),
        ("global", "leave"),
        ("global", "status", "--json"),
        ("global", "leave"),
    ):
        ok(*args)
    assert sent == [("POST", REGISTRY + "/v1/nodes/leave")]


# ── The journal ──────────────────────────────────────────────────────────────


def test_the_network_log_is_append_only() -> None:
    conn = db.open_db(":memory:")
    netlog.record(conn, "joined", {"node": "ab" * 32})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE network_log SET action = 'left'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM network_log")
    assert [e.action for e in netlog.entries(conn)] == ["joined"]


def test_a_v25_database_upgrades_to_the_network_log(tmp_path: Path) -> None:
    path = tmp_path / "v25.db"
    old = db.connect(path)
    for v, migration in enumerate(db.MIGRATIONS[:25], start=1):
        with db.transaction(old):
            for statement in db._split_sql(migration):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.execute("INSERT INTO tasks (id, goal) VALUES ('t1', 'goal')")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert netlog.entries(conn) == []
    assert conn.execute("SELECT goal FROM tasks").fetchone()[0] == "goal"


def test_config_changes_name_the_parts_of_the_network() -> None:
    before = make_config()
    after = before.model_copy(deep=True)
    after.network = Network(enabled=True, registry=REGISTRY, donated={"ollama": LENT})
    assert commands.config_changes(before, after) == [
        ("operator_network", {"changed": ["enabled", "registry", "donated.ollama"]})
    ]


# ── The wizard ───────────────────────────────────────────────────────────────


def test_the_wizards_last_step_joins_and_leaves() -> None:
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", "y", "http://nope", REGISTRY)
    assert result.exit_code == 0, result.output
    assert "is not a registry address" in result.output
    assert "Taking part in desman-global" in result.output
    assert saved().network.enabled and node_key().exists()
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", "")  # Enter keeps it on
    assert result.exit_code == 0, result.output
    assert saved().network.enabled and len(network_log()) == 1
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", "n")
    assert result.exit_code == 0, result.output
    assert "desman-global is off" in result.output and not saved().network.enabled


def test_the_wizard_without_an_address_stays_off() -> None:
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", "y", "-")
    assert result.exit_code == 0, result.output
    assert "Not now: desman-global stays off." in result.output
    assert not saved().network.enabled and not node_key().exists()


# ── The chat ─────────────────────────────────────────────────────────────────


def test_join_and_donate_from_the_chat_wait_for_yes(fake: FakeLLM) -> None:
    configured(settings=OLLAMA)
    fake.on(
        CHAT,
        asks("global join", registry=REGISTRY),
        asks("global donate", provider="ollama", calls=10),
        asks("global status"),
    )
    result = chat("join the network", "y", "lend my ollama", "y", "and now?")
    assert f"Take part in desman-global, registry {REGISTRY}." in result.output
    assert "This machine gets a node key of its own" in result.output
    assert "Lend Ollama to desman-global: 10 calls and 100000 tokens a day." in result.output
    assert "its key stays on this machine" in result.output
    network = saved().network
    assert network.enabled and network.donated["ollama"].calls_per_day == 10
    assert [a for a, _ in network_log()] == ["joined", "donated"]
    assert ("operator_network", {"changed": ["donated.ollama"]}) in journal()


def test_a_declined_join_changes_nothing(fake: FakeLLM) -> None:
    configured()
    fake.on(CHAT, asks("global join", registry=REGISTRY))
    result = chat("join", "n")
    assert "Nothing changed." in result.output
    assert not saved().network.enabled and not node_key().exists() and network_log() == []


def test_what_cannot_be_lent_is_told_without_a_window(fake: FakeLLM) -> None:
    configured(api_keys={"GROQ_API_KEY": "gsk_" + "a" * 40})
    fake.on(CHAT, asks("global donate", provider="groq"))
    result = chat("lend groq")
    assert "join it first" in result.output and "Apply this change?" not in result.output


def test_leave_from_the_chat_lists_what_is_taken_back(fake: FakeLLM) -> None:
    configured(settings=OLLAMA)
    ok("global", "join", "--registry", REGISTRY)
    ok("global", "donate", "ollama")
    fake.on(CHAT, asks("global leave"), asks("global quota", provider="ollama"))
    result = chat("leave", "y", "quota?")
    assert "Stop taking part in desman-global." in result.output
    assert "Take back: Ollama" in result.output
    assert not saved().network.enabled and network_log()[-2][0] == "left"
    assert "is not lent" in result.output


# ── Found in review ──────────────────────────────────────────────────────────


def test_a_save_that_fails_journals_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    configured()

    def broken(self: ConfigManager, config: Config) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as m:  # never undo(): it would drop conftest's directories too
        m.setattr(ConfigManager, "save", broken)
        assert desman("global", "join", "--registry", REGISTRY).exit_code != 0
    assert not saved().network.enabled and network_log() == []


def test_a_journal_that_fails_saves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    configured(settings=OLLAMA)
    ok("global", "join", "--registry", REGISTRY)

    def broken(conn: sqlite3.Connection, action: str, details: Any = None) -> None:
        raise sqlite3.OperationalError("database is locked")

    with monkeypatch.context() as m:
        m.setattr(netlog, "write", broken)
        assert desman("global", "donate", "ollama").exit_code != 0
        assert desman("global", "leave").exit_code != 0
    assert saved().network.enabled and saved().network.donated == {}
    assert [a for a, _ in network_log()] == ["joined"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_the_wizard_saves_nothing_when_the_key_cannot_be_used() -> None:
    sign.node_key(node_key())
    node_key().chmod(0o644)
    result = wizard("1", DONE, *KEEP_PERMISSIONS, "", "", "y", REGISTRY)
    assert result.exit_code == 1, result.output
    assert "The node's key cannot be used" in result.output
    assert not default_config_path().exists() and network_log() == []


def test_config_reset_journals_leaving() -> None:
    configured(settings=OLLAMA)
    ok("global", "join", "--registry", REGISTRY)
    ok("global", "donate", "ollama")
    ok("config", "reset", "--yes")
    assert network_log()[-1] == ("left", {"withdrawn": ["ollama"], "by": "config reset"})
    assert not default_config_path().exists()
    configured()  # off: nothing more to journal
    ok("config", "reset", "--yes")
    assert len(network_log()) == 3


def test_config_show_says_what_desman_global_does() -> None:
    configured(settings=OLLAMA)
    assert "desman-global  off" in ok("config", "show").output
    ok("global", "join", "--registry", REGISTRY)
    ok("global", "donate", "ollama")
    shown = ok("config", "show").output
    assert f"desman-global  on, registry {REGISTRY}; lent: Ollama" in shown
    out = json.loads(ok("config", "show", "--json").output)
    assert out["network"] == {
        "enabled": True,
        "registry": REGISTRY,
        "donated": {"ollama": LENT.model_dump()},
    }


def test_join_from_the_chat_brings_back_a_missing_key(fake: FakeLLM) -> None:
    configured()
    ok("global", "join", "--registry", REGISTRY)
    node_key().unlink()
    fake.on(CHAT, asks("global join"))
    result = chat("join", "y")
    assert "This machine gets a node key of its own" in result.output
    assert node_key().exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_join_from_the_chat_says_the_key_cannot_be_used(fake: FakeLLM) -> None:
    configured()
    ok("global", "join", "--registry", REGISTRY)
    node_key().chmod(0o644)
    fake.on(CHAT, asks("global join"))
    result = chat("join")
    assert "The node's key cannot be used" in result.output
    assert "Apply this change?" not in result.output
