"""The node in the network (ROADMAP G.5): registration, heartbeat and the trust list from
`desman worker`; backoff while the registry cannot be reached; a revoked node falls silent;
`desman global leave` says goodbye; everything in `network_log`. The registry is the real one,
in process, behind `httpx.MockTransport` (ADR 0004, decision 14)."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
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
    assert [p for _, p in world.requests] == [protocol.REGISTER, protocol.TRUST, protocol.RELEASES]


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
    assert len(world.requests) == 3
    # Due: a heartbeat that changes nothing is not journaled, and the person hears nothing.
    clock.advance(1)
    assert found.poll() is None
    assert [p for _, p in world.requests[3:]] == [
        protocol.HEARTBEAT,
        protocol.TRUST,
        protocol.RELEASES,
    ]
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
    # each change signs a trust list too (G.6c): only the newest is handed out
    assert [d["kind"] for d in received] == ["assignment", "revocation", "trust_list"]
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
    assert [d["kind"] for d in refused] == ["assignment", "trust_list"]
    assert all(d["reason"] == sign.Reason.UNKNOWN_SIGNER.value for d in refused)


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


def test_a_record_the_busy_journal_missed_is_journaled_next_time(
    make_node: Callable[..., Node],
    world: World,
    clock: FakeClock,
    conn: sqlite3.Connection,
    root: sign.NodeKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with store.connection(world.registry.path) as c:
        admin.revoke(c, root, "ab" * 32, "lost", now=clock())
    record = netlog.record

    def busy_for_records(c: sqlite3.Connection, action: str, details: Any = None) -> None:
        if action == "record_received":
            raise sqlite3.OperationalError("database is locked")
        record(c, action, details)

    found = make_node()
    with monkeypatch.context() as m:
        m.setattr(netlog, "record", busy_for_records)
        assert found.poll() is None
    assert "record_received" not in actions(conn)
    clock.advance(node.RETRY_FIRST.total_seconds())
    found.poll()
    assert actions(conn).count("record_received") == 2  # the record and its trust list


def test_a_contact_gets_one_timeout_for_both_its_requests(
    make_node: Callable[..., Node], world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow registration leaves the trust list only what is left of `TIMEOUT`; once it is
    spent, the second request is not sent at all."""
    now = [1000.0]
    spent: list[float] = []
    given: list[float] = []

    class Slow(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            given.append(request.extensions["timeout"]["read"])
            now[0] += spent.pop(0)
            return world.transport.handle_request(request)

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    found = make_node(transport=Slow())
    spent[:] = [6.0, 0.0, 0.0]
    event = found.poll()
    assert event is not None and event.change is Change.CONNECTED
    assert given == [node.TIMEOUT, node.TIMEOUT - 6.0, node.TIMEOUT - 6.0]
    # The heartbeat spends it all: the trust list is not asked for.
    found.next_at = None
    given.clear()
    spent[:] = [node.TIMEOUT]
    event = found.poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.detail == "ContactTimeout" and given == [node.TIMEOUT]


def test_a_reply_that_trickles_in_is_given_up_at_the_deadline(
    make_node: Callable[..., Node], world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """httpx times each read, so a registry sending a byte now and then would never time out
    by itself; the contact's own deadline stops it."""
    now = [1000.0]
    pieces: list[int] = []

    class Trickle(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self) -> Iterator[bytes]:
            for i in range(len(self.body)):
                now[0] += 1.0  # each byte well inside any read timeout
                pieces.append(i)
                yield self.body[i : i + 1]

    class Slow(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            reply = world.transport.handle_request(request)
            reply.read()
            return httpx.Response(
                reply.status_code, headers=reply.headers, stream=Trickle(reply.content)
            )

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    event = make_node(transport=Slow()).poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.detail == "ContactTimeout"
    assert len(pieces) == node.TIMEOUT  # read until the deadline, not a byte more


def test_a_reply_whose_ending_comes_after_the_deadline_is_given_up(
    make_node: Callable[..., Node], world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All the body in time, then the end of the stream late: the reply is not accepted."""
    now = [1000.0]

    class LateEnd(httpx.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self) -> Iterator[bytes]:
            yield self.body
            now[0] += node.TIMEOUT  # the terminating chunk waits

    class Slow(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            reply = world.transport.handle_request(request)
            reply.read()
            return httpx.Response(
                reply.status_code, headers=reply.headers, stream=LateEnd(reply.content)
            )

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    found = make_node(transport=Slow())
    event = found.poll()
    assert event is not None and event.change is Change.UNREACHABLE
    assert event.detail == "ContactTimeout" and not found.registered


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


def test_leaving_saves_off_then_tells_the_registry(
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
    assert left == global_cli.Left(["ollama"], REGISTRY, told=True)
    there = registered(world, key.fingerprint)
    assert there is not None and there.status == "left"
    assert log(conn)[-2:] == [
        ("left", {"withdrawn": ["ollama"]}),
        ("leave_sent", {"registry": REGISTRY, "node": key.fingerprint, "heard": True}),
    ]
    assert global_cli.left_json(left)["registry_told"] is True
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
    assert left == global_cli.Left(["ollama"], REGISTRY, unheard="ConnectError")
    assert not manager.load().network.enabled
    assert log(conn)[-1][1]["heard"] is False
    assert global_cli.left_json(left)["registry_told"] is False


def test_leaving_without_a_usable_key_says_nothing_was_sent(
    world: World, tmp_path: Path, conn: sqlite3.Connection
) -> None:
    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    left = global_cli.leave(
        manager, conn=conn, transport=world.transport, key_path=tmp_path / "missing.key"
    )
    assert left == global_cli.Left(["ollama"], REGISTRY, unheard=global_cli.NO_KEY)
    assert world.requests == [] and not manager.load().network.enabled
    assert log(conn)[-1] == ("leave_not_sent", {"registry": REGISTRY, "why": global_cli.NO_KEY})
    assert global_cli.left_json(left)["registry_told"] is False


def test_a_wizard_whose_save_is_refused_says_no_goodbye(
    world: World, tmp_path: Path, conn: sqlite3.Connection, key: sign.NodeKey
) -> None:
    """The config changed elsewhere while the wizard was open: its "leave" saves nothing, so
    the registry must still list the node."""
    from desman import wizard
    from desman.config import ConfigError

    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    seen = manager.snapshot()
    config = manager.load()
    entries, settle, said = wizard.network_change(config, wizard.NetworkChoice(join=False))
    other = manager.load()
    other.language = "de"
    manager.save(other)  # `desman language de` meanwhile
    with pytest.raises(ConfigError, match="nothing was saved"):
        wizard.save(manager, config, seen, entries, settle)
    assert world.requests == [] and manager.load().network.enabled


def _lock_is_held(manager: ConfigManager) -> bool:
    lock = manager.config_path.parent / f".{manager.config_path.name}.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


@pytest.mark.parametrize("via", ["leave", "wizard"])
def test_the_goodbye_is_sent_under_the_configs_lock(
    via: str, world: World, tmp_path: Path, conn: sqlite3.Connection, key: sign.NodeKey
) -> None:
    """A `join` right after the save waits for the goodbye, so a late goodbye never marks
    the node left after it joined again."""
    from desman import wizard

    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    held: list[bool] = []

    def handle(request: httpx.Request) -> httpx.Response:
        held.append(_lock_is_held(manager))
        return world.handle(request)

    transport = httpx.MockTransport(handle)
    if via == "leave":
        global_cli.leave(manager, conn=conn, transport=transport, key_path=tmp_path / "node.key")
    else:
        seen = manager.snapshot()
        config = manager.load()
        entries, _, said = wizard.network_change(config, wizard.NetworkChoice(join=False))
        left, _ = global_cli.leaving(manager.load())

        def settle(c: sqlite3.Connection) -> None:
            global_cli.farewell(left, conn=c, transport=transport, key_path=tmp_path / "node.key")

        wizard.save(manager, config, seen, entries, settle)
    assert held == [True] and not manager.load().network.enabled


def test_a_busy_journal_after_the_save_still_leaves(
    world: World,
    clock: FakeClock,
    tmp_path: Path,
    conn: sqlite3.Connection,
    key: sign.NodeKey,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = netlog.record

    def busy_for_goodbyes(c: sqlite3.Connection, action: str, details: Any = None) -> None:
        if action == "leave_sent":
            raise sqlite3.OperationalError("database is locked")
        record(c, action, details)

    clock.now = datetime.now(UTC)  # leaving signs with the real time
    manager = ConfigManager(tmp_path / "config.json")
    manager.save(Settings().config)
    monkeypatch.setattr(netlog, "record", busy_for_goodbyes)
    left = global_cli.leave(
        manager, conn=conn, transport=world.transport, key_path=tmp_path / "node.key"
    )
    assert left == global_cli.Left(["ollama"], REGISTRY, told=True, journaled=False)
    assert not manager.load().network.enabled and actions(conn)[-1] == "left"


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


@pytest.mark.parametrize("mode", ["good", "forged", "operator", "old", "down"])
def test_release_poll(
    mode: str,
    make_node: Callable[..., Node],
    world: World,
    root: sign.NodeKey,
    clock: FakeClock,
    conn: sqlite3.Connection,
) -> None:
    from test_updates import envelope

    signed = envelope(root, "a" * 40, "b" * 40)
    assigned = None
    if mode == "forged":
        signed = signed.model_copy(update={"signature": "A" * 86 + "=="})
    if mode == "operator":
        op = sign.NodeKey.generate()
        assigned = sign.assign(
            root, op.public, set(sign.Scope), issued_at=clock(), lifetime=timedelta(days=7)
        )
        signed = envelope(op, "a" * 40, "b" * 40)
    # Fixtures use a deterministic clock; sign within that term.
    signed = sign.sign(
        root if mode != "operator" else op,
        sign.RecordKind.RELEASE,
        signed.record.body,
        issued_at=clock(),
        lifetime=timedelta(days=7),
    )
    if mode == "forged":
        signed = signed.model_copy(update={"signature": "A" * 86 + "=="})
    normal = world.handle

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path == protocol.RELEASES:
            world.requests.append((req.method, req.url.path))
            if mode == "old":
                return httpx.Response(404, text="not found")
            if mode == "down":
                raise httpx.ConnectError("down", request=req)
            return httpx.Response(200, json={"records": [signed.model_dump(mode="json")]})
        if req.url.path == protocol.TRUST and assigned is not None:
            return httpx.Response(200, json={"records": [assigned.model_dump(mode="json")]})
        return normal(req)

    found = make_node(transport=httpx.MockTransport(handle))
    event = found.poll()
    assert event is not None
    assert event.change is (
        Change.RELEASE
        if mode == "good"
        else Change.UNREACHABLE
        if mode == "down"
        else Change.CONNECTED
    )
    for _ in range(2):
        clock.advance(protocol.HEARTBEAT_SECONDS)
        found.poll()
    if mode == "good":
        assert conn.execute("SELECT COUNT(*) FROM releases").fetchone()[0] == 1
        assert actions(conn).count("release_received") == 1
    elif mode != "down":
        assert actions(conn).count("releases_unavailable") == 1
        assert found.failures == 0 and not found.revoked
        assert conn.execute("SELECT COUNT(*) FROM releases").fetchone()[0] == 0


def test_release_notice_survives_job_claim(
    make_node: Callable[..., Node],
    world: World,
    root: sign.NodeKey,
    root_pub: Path,
    registry_db: Path,
    clock: FakeClock,
    conn: sqlite3.Connection,
) -> None:
    from test_updates import envelope

    operator = sign.NodeKey.generate()
    with store.connection(registry_db) as registry_conn:
        admin.assign(registry_conn, root, operator.public, {sign.Scope.TASKS}, days=7, now=clock())
        job = admin.add_job(
            registry_conn,
            operator,
            protocol.Job(description="Sum 2 and 3", acceptance_criteria="Says 5"),
            days=7,
            now=clock(),
            root_pub=root_pub,
        )
        admin.sign_trust_list(registry_conn, root, days=7, now=clock())
    world.registry = Registry(registry_db, clock=clock, root=root_pub)
    signed = envelope(root, "a" * 40, "b" * 40)
    signed = sign.sign(
        root,
        sign.RecordKind.RELEASE,
        signed.record.body,
        issued_at=clock(),
        lifetime=timedelta(days=7),
    )
    normal = world.handle

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path == protocol.RELEASES:
            return httpx.Response(200, json={"records": [signed.model_dump(mode="json")]})
        return normal(req)

    found = make_node(transport=httpx.MockTransport(handle), work=True)
    taken = found.poll(idle=True)
    assert taken == node.Event(Change.WORK_TAKEN, REGISTRY, "execute")
    assert conn.execute("SELECT job FROM network_work").fetchone()[0] == job.id
    assert found.poll() == node.Event(Change.RELEASE, REGISTRY, "1")
    assert found.poll() is None
    clock.advance(protocol.HEARTBEAT_SECONDS)
    assert found.poll() is None
