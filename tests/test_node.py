"""The node in the network (ROADMAP G.5): registration, heartbeat and the trust list from
`desman worker`; backoff while the registry cannot be reached; a revoked node falls silent;
`desman global leave` says goodbye; everything in `network_log`. The registry is the real one,
in process, behind `httpx.MockTransport` (ADR 0004, decision 14)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import global_cli
from desman.cli import cli
from desman.config import Config, ConfigManager, Donation, Network, default_config_path
from desman.core import db, netlog
from desman.core.store import Store
from desman.net import node, protocol, sign
from desman.net.node import Change, Node
from desman.registry import admin, store
from desman.registry.server import Registry
from helpers import make_config

REGISTRY = "https://registry.test"
OLLAMA = {"OLLAMA_MODEL": "llama3.2"}
LENT = Donation(calls_per_day=100, tokens_per_day=100_000)


@pytest.fixture
def root() -> sign.NodeKey:
    return sign.NodeKey.generate()


@pytest.fixture
def root_pub(tmp_path: Path, root: sign.NodeKey) -> Path:
    path = tmp_path / "root.pub"
    path.write_bytes(sign.public_pem(root.public))
    return path


@pytest.fixture
def registry_db(tmp_path: Path) -> Path:
    path = tmp_path / "registry" / store.DB_NAME
    path.parent.mkdir()
    store.open_registry(path).close()
    return path


class World:
    """The registry in process; `down` makes every request fail as a dropped connection, and
    `status` answers every request with that code instead."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.requests: list[tuple[str, str]] = []
        self.down = False
        self.status: int | None = None

    def handle(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, req.url.path))
        if self.down:
            raise httpx.ConnectError("unreachable", request=req)
        if self.status is not None:
            return httpx.Response(self.status, text="busy")
        reply = self.registry.handle(
            req.method, req.url.raw_path.decode("ascii"), req.headers.multi_items(), req.content
        )
        return httpx.Response(reply.status, json=reply.body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


@pytest.fixture
def world(registry_db: Path, clock: FakeClock) -> World:
    return World(Registry(registry_db, clock=clock))


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = db.open_db(":memory:")
    yield c
    c.close()


@pytest.fixture
def key(tmp_path: Path) -> sign.NodeKey:
    return sign.create_node_key(tmp_path / "node.key")


class Settings:
    def __init__(self) -> None:
        self.config = make_config(
            language="en",
            configured=True,
            settings=OLLAMA,
            network=Network(enabled=True, registry=REGISTRY, donated={"ollama": LENT}),
        )

    def __call__(self) -> Config:
        return self.config.model_copy(deep=True)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def make_node(
    conn: sqlite3.Connection,
    settings: Settings,
    world: World,
    clock: FakeClock,
    tmp_path: Path,
    root_pub: Path,
    key: sign.NodeKey,
) -> Callable[..., Node]:
    def make(**kwargs: Any) -> Node:
        options: dict[str, Any] = {
            "key_path": tmp_path / "node.key",
            "transport": world.transport,
            "clock": clock,
            "root": root_pub,
        }
        return Node(conn, settings, **(options | kwargs))

    return make


def log(conn: sqlite3.Connection) -> list[tuple[str, dict[str, Any]]]:
    return [(e.action, e.details) for e in netlog.entries(conn)]


def actions(conn: sqlite3.Connection) -> list[str]:
    return [a for a, _ in log(conn)]


def registered(world: World, fingerprint: str) -> store.Node | None:
    with store.connection(world.registry.path) as c:
        return store.node(c, fingerprint)


# ── Off ──────────────────────────────────────────────────────────────────────


def test_off_sends_nothing(
    make_node: Callable[..., Node], settings: Settings, world: World, clock: FakeClock
) -> None:
    settings.config.network.enabled = False
    found = make_node()
    for _ in range(5):
        assert found.poll() is None
        clock.advance(3600)
    assert world.requests == []


def test_joining_while_the_worker_runs_starts_at_the_next_look(
    make_node: Callable[..., Node], settings: Settings, world: World, clock: FakeClock
) -> None:
    settings.config.network.enabled = False
    found = make_node()
    assert found.poll() is None
    settings.config.network.enabled = True
    assert found.poll() is None  # not yet: the config is read again a minute later
    clock.advance(node.OFF_RECHECK.total_seconds())
    event = found.poll()
    assert event is not None and event.change is Change.CONNECTED
    assert [p for _, p in world.requests] == [protocol.REGISTER, protocol.TRUST]


def test_leaving_while_the_worker_runs_stops_it_before_anything_is_sent(
    make_node: Callable[..., Node], settings: Settings, world: World, clock: FakeClock
) -> None:
    found = make_node()
    found.poll()
    settings.config.network = Network(registry=REGISTRY)  # `desman global leave` elsewhere
    sent = len(world.requests)
    clock.advance(3600)
    assert found.poll() is None
    assert len(world.requests) == sent and not found.registered


# ── Registration, heartbeat, the trust list ──────────────────────────────────


def test_a_node_registers_with_what_it_lends_then_says_it_is_alive(
    make_node: Callable[..., Node],
    settings: Settings,
    world: World,
    clock: FakeClock,
    conn: sqlite3.Connection,
    key: sign.NodeKey,
) -> None:
    found = make_node()
    event = found.poll()
    assert event == node.Event(Change.CONNECTED, REGISTRY)
    there = registered(world, key.fingerprint)
    assert there is not None and there.status == "active"
    assert [(c.provider, c.model, c.key) for c in there.capabilities] == [
        ("ollama", "llama3.2", None)
    ]
    assert log(conn) == [
        (
            "registered",
            {
                "registry": REGISTRY,
                "node": key.fingerprint,
                "capabilities": [
                    {
                        "provider": "ollama",
                        "model": "llama3.2",
                        "key": None,
                        "calls_per_day": 100,
                        "tokens_per_day": 100_000,
                    }
                ],
            },
        )
    ]

    # Not due yet: nothing sent.
    clock.advance(protocol.HEARTBEAT_SECONDS - 1)
    assert found.poll() is None
    assert len(world.requests) == 2
    # Due: a heartbeat that changes nothing is not journaled, and the person hears nothing.
    clock.advance(1)
    assert found.poll() is None
    assert [p for _, p in world.requests[2:]] == [protocol.HEARTBEAT, protocol.TRUST]
    assert actions(conn) == ["registered"]

    # A quota changed: the next heartbeat carries it.
    settings.config.network.donated["ollama"] = Donation(calls_per_day=7, tokens_per_day=7000)
    clock.advance(protocol.HEARTBEAT_SECONDS)
    found.poll()
    there = registered(world, key.fingerprint)
    assert there is not None and there.capabilities[0].calls_per_day == 7
    assert actions(conn) == ["registered", "capabilities_sent"]


def test_a_cloud_module_is_named_by_its_key_fingerprint_never_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman import providers
    from desman.llm.openai_compat import key_id

    groq = providers.find("groq")
    monkeypatch.setattr(providers, "can_donate", lambda p: not p.paid)
    secret = "gsk_" + "s" * 48
    config = make_config(
        api_keys={"GROQ_API_KEY": secret},
        network=Network(enabled=True, registry=REGISTRY, donated={groq.id: LENT}),
    )
    [lent] = node.capabilities(config)
    assert lent.key == key_id(secret) and secret not in lent.model_dump_json()
    # Not set up, unknown, or not lendable: left out.
    config.network.donated["nobody"] = LENT
    assert len(node.capabilities(config)) == 1
    monkeypatch.setattr(providers, "can_donate", lambda p: False)
    assert node.capabilities(config) == []


def test_assignments_and_revocations_are_checked_and_journaled_once(
    make_node: Callable[..., Node],
    world: World,
    clock: FakeClock,
    conn: sqlite3.Connection,
    root: sign.NodeKey,
) -> None:
    operator = sign.NodeKey.generate()
    with store.connection(world.registry.path) as c:
        admin.assign(c, root, operator.public, {sign.Scope.TASKS}, days=30, now=clock())
        admin.revoke(c, root, "ab" * 32, "lost", now=clock())
    found = make_node()
    found.poll()
    clock.advance(protocol.HEARTBEAT_SECONDS)
    found.poll()
    received = [d for a, d in log(conn) if a == "record_received"]
    assert [d["kind"] for d in received] == ["assignment", "revocation"]
    assert all(d["signer"] == root.fingerprint for d in received)
    assert found._trust is not None and found._trust.operator(operator.fingerprint) is not None


def test_records_signed_by_another_root_are_refused_and_journaled_once(
    make_node: Callable[..., Node], world: World, clock: FakeClock, conn: sqlite3.Connection
) -> None:
    stranger = sign.NodeKey.generate()
    with store.connection(world.registry.path) as c:
        admin.assign(c, stranger, sign.NodeKey.generate().public, {sign.Scope.TASKS}, days=30,
                     now=clock())  # fmt: skip
    found = make_node()
    found.poll()
    clock.advance(protocol.HEARTBEAT_SECONDS)
    found.poll()
    refused = [d for a, d in log(conn) if a == "record_refused"]
    assert len(refused) == 1 and refused[0]["reason"] == sign.Reason.UNKNOWN_SIGNER.value


# ── When the registry fails ──────────────────────────────────────────────────


def test_an_unreachable_registry_is_tried_again_later_and_later(
    make_node: Callable[..., Node], world: World, clock: FakeClock, conn: sqlite3.Connection
) -> None:
    world.down = True
    found = make_node()
    event = found.poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.retry_in == 30 and event.detail == "ConnectError"
    assert found.poll() is None and len(world.requests) == 1  # waiting
    clock.advance(30)
    assert found.poll() is None  # the second failure is journaled, not told again
    assert found.next_at == clock() + timedelta(seconds=60)
    for _ in range(12):
        clock.advance(3600)
        found.poll()
    assert found.next_at == clock() + node.RETRY_MAX

    world.down = False
    clock.advance(node.RETRY_MAX.total_seconds())
    event = found.poll()
    assert event is not None and event.change is Change.CONNECTED
    assert actions(conn)[-2:] == ["registered", "reachable"]
    failures = [d for a, d in log(conn) if a == "unreachable"]
    assert [d["attempt"] for d in failures] == list(range(1, 15))


@pytest.mark.parametrize("status", [429, 500, 503])
def test_busy_or_broken_answers_wait_like_no_answer(
    make_node: Callable[..., Node], world: World, status: int
) -> None:
    world.status = status
    event = make_node().poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.detail == f"HTTP {status}"


def test_a_reply_that_is_not_the_protocol_waits_too(
    make_node: Callable[..., Node], world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        world, "handle", lambda req: httpx.Response(200, json={"heartbeat_seconds": "soon"})
    )
    event = make_node().poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.detail == "bad_reply"


def test_a_clock_that_is_off_is_a_refusal_told_once(
    make_node: Callable[..., Node], clock: FakeClock, conn: sqlite3.Connection
) -> None:
    ahead = FakeClock()
    ahead.now = clock.now + timedelta(minutes=10)
    found = make_node(clock=ahead)
    event = found.poll()
    assert event is not None and event.change is Change.REFUSED
    assert event.detail == sign.Reason.NOT_YET.value and event.retry_in == 30
    assert actions(conn) == ["refused_by_registry"]


def test_a_node_the_registry_forgot_registers_again_at_once(
    make_node: Callable[..., Node],
    world: World,
    clock: FakeClock,
    conn: sqlite3.Connection,
    key: sign.NodeKey,
) -> None:
    found = make_node()
    found.poll()
    assert node.tell_left(REGISTRY, key, transport=world.transport, clock=clock) is None
    clock.advance(protocol.HEARTBEAT_SECONDS)
    assert found.poll() is None and not found.registered
    found.poll()  # at once, not after a backoff
    there = registered(world, key.fingerprint)
    assert there is not None and there.status == "active"
    assert actions(conn) == ["registered", "refused_by_registry", "registered"]


def test_a_revoked_node_falls_silent(
    make_node: Callable[..., Node],
    world: World,
    clock: FakeClock,
    conn: sqlite3.Connection,
    root: sign.NodeKey,
    key: sign.NodeKey,
) -> None:
    found = make_node()
    found.poll()
    with store.connection(world.registry.path) as c:
        admin.revoke(c, root, key.fingerprint, "abuse", now=clock())
    clock.advance(protocol.HEARTBEAT_SECONDS)
    event = found.poll()
    assert event is not None and event.change is Change.REVOKED and found.revoked
    sent = len(world.requests)
    for _ in range(3):
        clock.advance(3600)
        assert found.poll() is None
    assert len(world.requests) == sent
    assert actions(conn)[-1] == "revoked"


def test_a_revocation_in_the_trust_list_stops_the_node_too(
    make_node: Callable[..., Node], world: World, key: sign.NodeKey, root: sign.NodeKey,
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    """Should the registry still serve a revoked node, its own trust list says enough."""
    with store.connection(world.registry.path) as c:
        admin.revoke(c, root, key.fingerprint, "abuse", now=clock())
    monkeypatch.setattr(store, "is_revoked", lambda c, fingerprint: False)
    event = make_node().poll()
    assert event is not None and event.change is Change.REVOKED


def test_a_busy_journal_never_stops_the_worker(
    make_node: Callable[..., Node], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(*args: Any) -> None:
        raise sqlite3.OperationalError("database is locked")

    found = make_node()
    with monkeypatch.context() as m:
        m.setattr(netlog, "record", locked)
        assert found.poll() is None
    assert found.next_at == clock() + node.RETRY_FIRST


def test_without_a_usable_key_nothing_is_sent(
    make_node: Callable[..., Node], world: World, tmp_path: Path, conn: sqlite3.Connection
) -> None:
    found = make_node(key_path=tmp_path / "missing.key")
    event = found.poll()
    assert event is not None and event.change is Change.NO_KEY
    assert world.requests == [] and actions(conn) == ["node_key_unusable"]


def test_the_network_never_stops_the_workers_own_work(tmp_path: Path) -> None:
    """`desman worker` with desman-global on and no registry to reach (no internet in tests):
    it says so once and does its work."""
    sign.node_key()
    ConfigManager(default_config_path()).save(
        make_config(
            language="en",
            configured=True,
            network=Network(enabled=True, registry="https://registry.example.org"),
        )
    )
    path = tmp_path / "desman.db"
    Store(db.open_db(path)).conn.close()
    script = tmp_path / "fake.json"
    script.write_text(json.dumps({"routes": []}))
    args = ["worker", "--until-idle", "--fake-llm", str(script), "--db", str(path)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert "cannot be reached (ConnectError)" in result.output
    assert "Worker stopped" in result.output
    quiet = CliRunner().invoke(cli, [*args, "--json"])
    assert quiet.exit_code == 0, quiet.output
    json.loads(quiet.output)


# ── Leaving ──────────────────────────────────────────────────────────────────


def test_leaving_tells_the_registry_first(
    world: World, clock: FakeClock, key: sign.NodeKey, tmp_path: Path, conn: sqlite3.Connection,
    make_node: Callable[..., Node],
) -> None:  # fmt: skip
    clock.now = datetime.now(UTC)  # leaving signs with the real time
    make_node().poll()
    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    left = global_cli.leave(
        manager, conn=conn, transport=world.transport, key_path=tmp_path / "node.key"
    )
    assert left == global_cli.Left(["ollama"], None)
    there = registered(world, key.fingerprint)
    assert there is not None and there.status == "left"
    assert log(conn)[-2:] == [
        ("leave_sent", {"registry": REGISTRY, "node": key.fingerprint, "heard": True}),
        ("left", {"withdrawn": ["ollama"]}),
    ]
    assert not manager.load().network.enabled
    # Off already: nothing is sent again.
    sent = len(world.requests)
    assert global_cli.leave(manager, conn=conn, transport=world.transport) is None
    assert len(world.requests) == sent


def test_leaving_without_the_registry_still_leaves(
    world: World, tmp_path: Path, conn: sqlite3.Connection, key: sign.NodeKey
) -> None:
    world.down = True
    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    left = global_cli.leave(
        manager, conn=conn, transport=world.transport, key_path=tmp_path / "node.key"
    )
    assert left == global_cli.Left(["ollama"], "ConnectError")
    assert not manager.load().network.enabled
    assert log(conn)[-2][1]["heard"] is False
    assert global_cli.left_json(left)["registry_told"] is False


def test_leave_from_the_command_line_warns_when_the_registry_did_not_hear() -> None:
    sign.node_key()
    ConfigManager(default_config_path()).save(Settings().config)
    result = CliRunner().invoke(cli, ["global", "leave"])
    assert result.exit_code == 0, result.output
    assert "The registry could not be told (ConnectError)" in result.output
    result = CliRunner().invoke(cli, ["global", "leave", "--json"])
    assert json.loads(result.output) == {
        "changed": False,
        "withdrawn": [],
        "registry_told": False,
    }
