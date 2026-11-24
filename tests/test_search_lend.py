"""Lend search through the signed registry and the normal web search tool."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, net, runs, searxng, searxng_native, ui
from desman.cli import cli
from desman.config import Config, ConfigManager, Donation, Network, default_config_path
from desman.core import db, netlog, operator
from desman.core.store import Store
from desman.core.tools import ToolError, WebSearchArgs, web_tools
from desman.core.web import MissingSearchEngine, SearchResult, Web, WebError
from desman.core.workspace import Workspace
from desman.gui.front import WebFront
from desman.gui.panels import snapshot
from desman.gui.runners import Runners
from desman.net import node, protocol, search_lend, sign
from desman.net.search_lend import Lent
from desman.registry import clusters, store
from desman.registry.server import Registry
from helpers import make_config
from test_registry import Node as Wire
from test_registry import root, root_pub

__all__ = ["root", "root_pub"]

REGISTRY = "https://registry.test"
SEARXNG = "http://127.0.0.1:8888"
QUERY = "a distinctive private query"
HIT = protocol.SearchHit(title="Result", url="https://example.test/item", snippet="Snippet")


@pytest.fixture
def relay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clock: FakeClock
) -> Iterator[tuple[Registry, sqlite3.Connection]]:
    with db.connect(":memory:") as conn:
        db.migrate(conn, store.MIGRATIONS[:-1])
        assert db.migrate(conn, store.MIGRATIONS) == 1
        db.verify_schema(conn, store.MIGRATIONS, fresh=store.FRESH)

        @contextmanager
        def connection(path: Path) -> Iterator[sqlite3.Connection]:
            yield conn

        monkeypatch.setattr(store, "connection", connection)
        yield Registry(tmp_path / "registry.db", clock=clock), conn
    conn.close()


def _capability(calls: int) -> dict[str, object]:
    return protocol.Capability(
        provider="searxng", calls_per_day=calls, tokens_per_day=0
    ).model_dump()


def _lend(registry: Registry, clock: FakeClock, calls: int = 100) -> Wire:
    lender = Wire(registry, clock)
    assert lender.register([_capability(calls)]).status_code == 200
    assert lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json() == {"jobs": []}
    return lender


def _ask(asker: Wire) -> httpx.Response:
    return asker.call("POST", protocol.SEARCH_ASK, {"query": QUERY, "count": 2})


def _answer(lender: Wire, id: str, hits: list[protocol.SearchHit]) -> httpx.Response:
    return lender.call(
        "POST", protocol.SEARCH_ANSWER, protocol.SearchAnswer(id=id, hits=hits).model_dump()
    )


@pytest.mark.parametrize("query", ["", "  ", "\tstuff", "foo\x85bar", "q" * 301])
def test_query_rules(query: str) -> None:
    with pytest.raises(ValidationError):
        protocol.SearchAsk(query=query)


def test_protocol_constraints_and_lender_mapping() -> None:
    assert protocol.SearchAsk(query="  hello  ").model_dump() == {"query": "hello", "count": 8}
    for bad in ({"count": "5"}, {"count": 11}, {"extra": True}):
        with pytest.raises(ValidationError):
            protocol.SearchAsk.model_validate({"query": "test", **bad})
    for key, value in (
        ("title", "x" * 301),
        ("url", "https://example.test/" + "x" * 2000),
        ("snippet", "x" * 1001),
        ("url", "file:///etc/passwd"),
    ):
        with pytest.raises(ValidationError):
            protocol.SearchHit.model_validate({**HIT.model_dump(), key: value})
    with pytest.raises(ValidationError):
        protocol.SearchAnswer(id="0" * 32, hits=[], error="x" * 201)
    found = search_lend.hits(
        [
            SearchResult(title="x" * 500, url=HIT.url, snippet="y" * 2000),
            SearchResult(url="file:///tmp/item"),
        ]
    )
    assert found == [protocol.SearchHit(title="x" * 300, url=HIT.url, snippet="y" * 1000)]


def test_relay_privacy_first_answer_and_expiry(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker, stranger = Wire(registry, clock), Wire(registry, clock)
    assert _ask(asker).json()["error"] == "not_registered"
    asker.register()
    stranger.register()
    lender = _lend(registry, clock)
    offered = protocol.SearchAsked.model_validate(_ask(asker).json())
    path = protocol.SEARCH_STATE + offered.id
    assert asker.call("GET", path).json()["state"] == "waiting"
    assert stranger.call("GET", path).status_code == 404
    assert _answer(stranger, offered.id, [HIT]).status_code == 404
    jobs = lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json()
    assert jobs == {"jobs": [{"id": offered.id, "query": QUERY, "count": 2, "expires_in": 30}]}
    assert lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json() == {"jobs": []}
    assert _answer(lender, offered.id, [HIT]).status_code == 200
    assert _answer(lender, offered.id, []).status_code == 200
    assert conn.execute("SELECT query FROM search_requests").fetchone()[0] is None
    assert asker.call("GET", path).json()["hits"] == [HIT.model_dump()]
    assert asker.call("GET", path).json()["state"] == "gone"
    assert tuple(conn.execute("SELECT query, payload FROM search_requests").fetchone()) == (
        None,
        None,
    )

    uncollected = _ask(asker).json()["id"]
    assert _answer(lender, uncollected, [HIT]).json()["state"] == "answered"
    expires = _ask(asker).json()["id"]
    clock.advance(30)
    assert asker.heartbeat().status_code == 200  # the ordinary registry tick also erases
    assert asker.call("GET", protocol.SEARCH_STATE + uncollected).json()["state"] == "gone"
    assert asker.call("GET", protocol.SEARCH_STATE + expires).json()["state"] == "expired"
    assert _answer(lender, expires, [HIT]).json()["state"] == "expired"
    assert (
        conn.execute("SELECT count(query) + count(payload) FROM search_requests").fetchone()[0] == 0
    )
    entries = store.entries(conn)
    assert QUERY not in repr(entries)
    assert {e.action for e in entries}.issuperset(
        {"search_asked", "search_answered", "search_expired"}
    )


def test_eligibility_owner_poll_and_quota(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    assert _ask(asker).json()["error"] == "no_lender"
    asker.heartbeat([_capability(100)])
    asker.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
    lender = _lend(registry, clock, 1)
    conn.execute("UPDATE nodes SET owner = 'same owner'")
    assert _ask(asker).json()["error"] == "no_lender"
    conn.execute("UPDATE nodes SET owner = NULL")
    clock.advance(31)
    assert _ask(asker).json()["error"] == "no_lender"
    lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
    id = _ask(asker).json()["id"]
    assert _ask(asker).json()["error"] == "no_lender"  # reserve the last possible answer
    assert _answer(lender, id, []).status_code == 200
    assert _ask(asker).json()["error"] == "no_lender"
    clock.advance(86400)
    lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
    assert _ask(asker).status_code == 200


def test_reassignment_is_once_and_excludes_previous_lender(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    first = _lend(registry, clock)
    id = _ask(asker).json()["id"]
    clock.advance(9)
    second = _lend(registry, clock)
    clock.advance(1)
    jobs = second.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json()["jobs"]
    assert [job["id"] for job in jobs] == [id]
    assert _answer(first, id, []).status_code == 404
    clock.advance(11)
    assert first.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json() == {"jobs": []}
    assert conn.execute(
        "SELECT lender, reassigned FROM search_requests WHERE id = ?", (id,)
    ).fetchone()[:] == (second.fingerprint, 1)
    assert _answer(second, id, [HIT]).status_code == 200


def test_requester_window_and_utc_day_limit(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    lender = _lend(registry, clock, 1000)
    for _ in range(10):
        lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
        for _ in range(30):
            assert _ask(asker).status_code == 200
        assert _ask(asker).json()["error"] == "too_many"
        clock.advance(600)
    lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
    assert _ask(asker).json()["error"] == "too_many"
    clock.advance(86400)
    lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4})
    assert _ask(asker).status_code == 200
    assert conn.execute("SELECT count(*) FROM search_requests").fetchone()[0] == 301


def _borrower() -> Config:
    return make_config(network=Network(enabled=True, registry=REGISTRY, borrow_search=True))


def _lender() -> Config:
    return make_config(
        network=Network(
            enabled=True,
            registry=REGISTRY,
            donated={"searxng": Donation(calls_per_day=1, tokens_per_day=1000)},
        ),
        settings={"SEARXNG_URL": SEARXNG, searxng.DOCKER_URL: SEARXNG},
    )


@pytest.mark.parametrize("case", ["no_lender", "timeout", "unreachable"])
def test_borrower_failures(
    relay: tuple[Registry, sqlite3.Connection],
    clock: FakeClock,
    case: str,
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    if case == "timeout":
        _lend(registry, clock)

    def unreachable(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=req)

    with (
        db.open_db(":memory:") as local,
        httpx.Client(transport=httpx.MockTransport(unreachable)) as down,
    ):
        engine = Lent(
            _borrower(),
            local,
            down if case == "unreachable" else asker.client,
            clock,
            asker.key,
            REGISTRY,
            sleep=clock.sleep,
        )
        error, match = {
            "no_lender": (MissingSearchEngine, "no node lends search now"),
            "timeout": (WebError, "the lending node did not answer in time"),
            "unreachable": (WebError, "the search relay is unreachable"),
        }[case]
        with pytest.raises(error, match=match) as exc:
            engine.search(QUERY, 1)
        assert type(exc.value) is error
        assert conn.execute("SELECT count(*) FROM search_requests").fetchone()[0] == (
            case == "timeout"
        )
    assert clock.slept == pytest.approx(30, abs=0.1) if case == "timeout" else clock.slept == 0


@pytest.mark.parametrize("fails", [False, True])
def test_web_tool_end_to_end_and_local_quota(
    relay: tuple[Registry, sqlite3.Connection],
    clock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fails: bool,
) -> None:
    registry, _ = relay
    requests: list[tuple[str, str]] = []

    def handle(req: httpx.Request) -> httpx.Response:
        requests.append((req.method, req.url.path))
        if req.url.host == "127.0.0.1":
            assert req.url.params["q"] == QUERY
            if fails:
                raise httpx.ConnectError("unreachable", request=req)
            return httpx.Response(200, json={"results": [HIT.model_dump(), {"url": "file:///a"}]})
        reply = registry.handle(req.method, req.url.path, req.headers.multi_items(), req.content)
        return httpx.Response(reply.status, json=reply.body)

    transport = httpx.MockTransport(handle)
    lender_path = tmp_path / "lender.key"
    sign.create_node_key(lender_path)
    asker_path = sign.default_node_key_path()
    asker_path.parent.mkdir(exist_ok=True, parents=True)
    sign.create_node_key(asker_path)
    lender_config = _lender()
    with (
        db.open_db(":memory:") as local,
        db.open_db(":memory:") as remote,
        httpx.Client(transport=transport) as http,
    ):
        worker = node.Node(
            remote, lambda: lender_config, key_path=lender_path, clock=clock, transport=transport
        )
        lender_config.network.donated = {}
        worker.poll(idle=False)
        assert ("POST", protocol.SEARCH_NEXT) not in requests
        lender_config.network.donated = _lender().network.donated
        worker.poll(idle=False)
        worker.poll(idle=False)
        assert requests.count(("POST", protocol.SEARCH_NEXT)) == 1

        def sleep(seconds: float) -> None:
            clock.sleep(seconds)
            worker.poll(idle=False)

        web = runs.web(_borrower(), Store(local, clock), http)
        engine = web.engines[0]
        assert isinstance(engine, Lent)
        monkeypatch.setattr(engine, "sleep", sleep)
        tool = next(tool for tool in web_tools(web) if tool.name == "web_search")
        if fails:
            with pytest.raises(ToolError, match="the lending node could not search") as exc:
                tool.run(WebSearchArgs(query=QUERY), Workspace(tmp_path, "0" * 32))
            assert type(exc.value) is ToolError
        else:
            result = tool.run(WebSearchArgs(query=QUERY), Workspace(tmp_path, "0" * 32))
            assert result["engine"] == "desman-global"
            assert result["results"] == [HIT.model_dump()]
            assert "another desman-global node" in result["untrusted"]
        assert requests.count(("POST", protocol.SEARCH_ASK)) == 2  # register and retry once
        assert requests.count(("POST", protocol.SEARCH_NEXT)) == 2  # next poll at 5s
        assert requests.count(("GET", "/search")) == 1
        clock.advance(5)
        worker.poll(idle=True)
        assert requests.count(("POST", protocol.SEARCH_NEXT)) == 2
        digest = hashlib.sha256(QUERY.encode()).hexdigest()
        for conn, action in [(remote, "lent_search"), (local, "borrowed_search")]:
            [entry] = [entry for entry in netlog.entries(conn) if entry.action == action]
            assert entry.details["sha256"] == digest and QUERY not in repr(entry)
        worker.close()


def test_engine_only_with_consent_key_and_no_local_engine(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "node.key"
    monkeypatch.setattr(sign, "default_node_key_path", lambda: key)
    config = _borrower()
    with db.open_db(":memory:") as conn, httpx.Client() as http:
        target = Store(conn, clock)
        assert runs.web(config, target, http).engines == []
        sign.create_node_key(key)
        assert isinstance(runs.web(config, target, http).engines[0], Lent)
        config.network.borrow_search = False
        assert runs.web(config, target, http).engines == []
        config.network.borrow_search = True
        config.network.enabled = False
        assert runs.web(config, target, http).engines == []
        config.network.enabled = True
        config.settings["SEARXNG_URL"] = SEARXNG
        assert [engine.name for engine in runs.web(config, target, http).engines] == ["searxng"]


def test_v9_upgrade_and_cli() -> None:
    manager = ConfigManager(default_config_path())
    manager.config_path.parent.mkdir(exist_ok=True, parents=True)
    manager.config_path.write_text(
        '{"schema_version":9,"language":"en","network":{"enabled":true}}', encoding="utf-8"
    )
    current = manager.load()
    assert current.schema_version == 15 and not current.network.borrow_search
    runner = CliRunner()
    assert "off" in runner.invoke(cli, ["global", "search"]).output
    assert "before lending it" in runner.invoke(cli, ["global", "donate", "searxng"]).output
    current.settings["SEARXNG_URL"] = SEARXNG
    manager.save(current)
    assert runner.invoke(cli, ["global", "donate", "searxng"]).exit_code == 1
    current.settings[searxng.DOCKER_URL] = SEARXNG
    manager.save(current)
    assert runner.invoke(cli, ["global", "donate", "searxng", "--calls", "3"]).exit_code == 0
    assert (
        "4 searches a day"
        in runner.invoke(cli, ["global", "quota", "searxng", "--calls", "4"]).output
    )
    assert runner.invoke(cli, ["global", "search", "on"]).exit_code == 0
    assert manager.load().network.borrow_search
    assert '"borrow_search": true' in runner.invoke(cli, ["global", "status", "--json"]).output
    assert runner.invoke(cli, ["global", "search", "off"]).exit_code == 0
    assert not manager.load().network.borrow_search
    with db.open_db(db.default_db_path()) as conn:
        assert [e.details["on"] for e in netlog.entries(conn) if e.action == "borrow_search"] == [
            True,
            False,
        ]
        assert [e.details["on"] for e in operator.entries(conn) if e.action == "borrow_search"] == [
            True,
            False,
        ]
    entry = commands.find("global search")
    assert entry is not None
    assert entry.kind_of(commands.BorrowSearchArgs()) is commands.Kind.READ
    assert entry.kind_of(commands.BorrowSearchArgs(state="on")) is commands.Kind.CHANGE


def test_unavailable_asks_are_rate_limited(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    for _ in range(30):
        assert _ask(asker).json()["error"] == "no_lender"
    assert _ask(asker).json()["error"] == "too_many"
    assert conn.execute("SELECT count(*) FROM search_requests").fetchone()[0] == 0
    assert QUERY not in repr(store.entries(conn))


def test_disabling_borrowing_stops_existing_engine(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key = tmp_path / "node.key"
    monkeypatch.setattr(sign, "default_node_key_path", lambda: key)
    sign.create_node_key(key)
    config = _borrower()

    def unexpected(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected request to {request.url.path}")

    with (
        db.open_db(":memory:") as conn,
        httpx.Client(transport=httpx.MockTransport(unexpected)) as http,
    ):
        [engine] = runs.web(config, Store(conn, clock), http, load=lambda: config).engines
        config.network.borrow_search = False
        with pytest.raises(WebError, match="lent search is unavailable"):
            engine.search(QUERY, 1)


def test_native_lender_uses_the_native_start_hook(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _lender()
    config.settings.pop(searxng.DOCKER_URL)
    monkeypatch.setattr(searxng_native, "directory", lambda: tmp_path)
    monkeypatch.setattr(searxng_native, "commit", lambda _: "installed")
    monkeypatch.setattr(searxng_native, "url", lambda _: SEARXNG)
    called = []

    def start(url: str, *, clock: object) -> bool:
        called.append(url)
        return True

    monkeypatch.setattr(searxng, "ensure_native", start)
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"results": [HIT.model_dump()]})
    )
    with db.open_db(":memory:") as conn, httpx.Client(transport=transport) as http:
        result = search_lend.answer(
            conn, config, protocol.SearchJob(id="1" * 32, query=QUERY, expires_in=30), http, clock
        )
    assert result is not None and result.hits == [HIT]
    assert called == [SEARXNG]


def test_poll_takes_own_overdue_jobs_first_and_honors_limit(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    first = _lend(registry, clock, 3)
    old = _ask(asker).json()["id"]
    clock.advance(1)
    new = _ask(asker).json()["id"]
    clock.advance(1)
    newest = _ask(asker).json()["id"]
    clock.advance(8)
    # A second lender becomes available without running reassignment yet.
    second = Wire(registry, clock)
    second.register([_capability(4)])
    conn.execute(
        "INSERT INTO search_polls VALUES (?, ?)", (second.fingerprint, clock().timestamp())
    )
    for invalid in ({}, {"limit": 0}, {"limit": 5}, {"limit": "1"}):
        assert first.call("POST", protocol.SEARCH_NEXT, invalid).status_code != 200
    assert [
        j["id"] for j in first.call("POST", protocol.SEARCH_NEXT, {"limit": 1}).json()["jobs"]
    ] == [old]
    assert first.heartbeat([_capability(2)]).status_code == 200
    assert [
        j["id"] for j in first.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json()["jobs"]
    ] == [new]
    assert _answer(first, old, []).status_code == 200
    assert first.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json() == {"jobs": []}
    assert asker.call("GET", protocol.SEARCH_STATE + newest).json()["state"] == "waiting"


def test_borrower_registration_preserves_worker_capabilities(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock, tmp_path: Path
) -> None:
    registry, conn = relay
    config = _borrower()
    config.settings["OLLAMA_MODEL"] = "qwen2.5:7b"
    config.network.donated["ollama"] = Donation(calls_per_day=5, tokens_per_day=1000)
    lent = node.capabilities(config)
    assert [cap.provider for cap in lent] == ["ollama"]
    path = tmp_path / "node.key"
    key = sign.create_node_key(path)
    wire = Wire(registry, clock, key)
    heartbeats: list[protocol.Heartbeat] = []

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path == protocol.HEARTBEAT:
            heartbeats.append(protocol.Heartbeat.model_validate_json(req.content))
        return wire._handle(req)

    transport = httpx.MockTransport(handle)

    with db.open_db(":memory:") as local, httpx.Client(transport=transport) as http:
        worker = node.Node(local, lambda: config, key_path=path, clock=clock, transport=transport)
        worker.poll(idle=False)
        before = store.node(conn, wire.fingerprint)
        assert before is not None and before.capabilities == lent
        assert wire.call("POST", protocol.LEAVE).status_code == 200
        engine = Lent(config, local, http, clock, key, REGISTRY, sleep=clock.sleep)
        with pytest.raises(MissingSearchEngine):
            engine.search(QUERY, 1)
        registrations = [entry for entry in netlog.entries(local) if entry.action == "registered"]
        assert len(registrations) == 2
        assert registrations[-1].details == {
            "registry": REGISTRY,
            "node": wire.fingerprint,
            "capabilities": [cap.model_dump(mode="json") for cap in lent],
        }
        clock.advance(protocol.HEARTBEAT_SECONDS)
        worker.poll(idle=False)
        worker.close()
        assert len(heartbeats) == 1 and heartbeats[0].capabilities is None
        found = store.node(conn, wire.fingerprint)
        assert found is not None and found.capabilities == lent


def test_registration_clears_cluster_availability_until_next_claim(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    wire = Wire(registry, clock)
    capability = protocol.Capability(
        provider="ollama", model="qwen2.5:7b", calls_per_day=5, tokens_per_day=1000
    ).model_dump()

    def available() -> bool:
        found = store.node(conn, wire.fingerprint)
        assert found is not None
        return clusters._available(found, clock(), timedelta(minutes=1))

    assert wire.register([capability]).status_code == 200
    assert wire.call("POST", protocol.CLAIM, {}).status_code == 200
    assert available()
    assert wire.register([capability]).status_code == 200
    assert not available()
    assert wire.call("POST", protocol.CLAIM, {}).status_code == 200
    assert available()


def test_offer_can_shorten_wait_and_web_reports_timeout(clock: FakeClock) -> None:
    def handle(req: httpx.Request) -> httpx.Response:
        id = "a" * 32
        if req.url.path == protocol.SEARCH_ASK:
            return httpx.Response(200, json={"id": id, "wait_seconds": 3})
        return httpx.Response(200, json={"id": id, "state": "waiting", "hits": []})

    with (
        db.open_db(":memory:") as conn,
        httpx.Client(transport=httpx.MockTransport(handle)) as http,
    ):
        engine = Lent(
            _borrower(), conn, http, clock, sign.NodeKey.generate(), REGISTRY, sleep=clock.sleep
        )
        tool = next(tool for tool in web_tools(Web(engines=[engine])) if tool.name == "web_search")
        with pytest.raises(ToolError, match="the lending node did not answer in time") as exc:
            tool.run(WebSearchArgs(query=QUERY), Workspace(Path("."), "0" * 32))
        assert type(exc.value) is ToolError
    assert clock.slept == 3


@pytest.mark.parametrize("change", ["disabled", "quota", "busy"])
def test_lender_reports_jobs_it_cannot_run(
    relay: tuple[Registry, sqlite3.Connection],
    clock: FakeClock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    config = _lender()
    config.network.donated["searxng"] = Donation(calls_per_day=3, tokens_per_day=0)
    path = tmp_path / "lender.key"
    sign.create_node_key(path)
    poll_limits: list[int] = []
    taken: list[str] = []
    write = netlog.write
    busy = change == "busy"

    def write_once_busy(
        conn: sqlite3.Connection, action: str, details: dict[str, str] | None = None
    ) -> None:
        nonlocal busy
        if action == "lent_search":
            assert conn.in_transaction
            if busy:
                busy = False
                raise sqlite3.OperationalError("database is locked")
        write(conn, action, details)

    with db.open_db(":memory:") as local:

        def handle(req: httpx.Request) -> httpx.Response:
            if req.url.host == "127.0.0.1":
                assert change == "busy"
                return httpx.Response(200, json={"results": [HIT.model_dump()]})
            reply = registry.handle(
                req.method, req.url.path, req.headers.multi_items(), req.content
            )
            if req.url.path == protocol.SEARCH_NEXT:
                poll_limits.append(json.loads(req.content)["limit"])
                if reply.body["jobs"]:
                    taken.extend(job["id"] for job in reply.body["jobs"])
                    if change == "disabled":
                        config.network.enabled = False
                    elif change == "quota":
                        config.network.donated["searxng"] = Donation(
                            calls_per_day=1, tokens_per_day=0
                        )
            return httpx.Response(reply.status, json=reply.body)

        worker = node.Node(
            local, lambda: config, key_path=path, clock=clock, transport=httpx.MockTransport(handle)
        )
        netlog.record(local, "lent_search", {"day": clock().date().isoformat()})
        worker.poll()
        first = _ask(asker).json()["id"]
        clock.advance(1)
        second = _ask(asker).json()["id"]
        clock.advance(1)
        third = _ask(asker).json()["id"]
        clock.advance(3)
        monkeypatch.setattr(netlog, "write", write_once_busy)
        worker.poll()
        assert poll_limits == [2, 2]
        assert set(taken) == {first, second}
        assert asker.call("GET", protocol.SEARCH_STATE + taken[0]).json()["error"] == (
            protocol.SEARCH_SKIPPED
        )
        second_result = asker.call("GET", protocol.SEARCH_STATE + taken[1]).json()
        assert second_result["error"] == (None if change == "busy" else protocol.SEARCH_SKIPPED)
        assert asker.call("GET", protocol.SEARCH_STATE + third).json()["state"] == "waiting"
        assert search_lend.used_today(local, clock()) == (2 if change == "busy" else 1)
        assert conn.execute("SELECT count(answered_at) FROM search_requests").fetchone()[0] == (
            1 if change == "busy" else 0
        )
        worker.close()


def test_suspending_returns_registry_revocation(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock, tmp_path: Path
) -> None:
    registry, _ = relay
    config = _lender()
    path = tmp_path / "lender.key"
    sign.create_node_key(path)

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path == protocol.HEARTBEAT:
            return httpx.Response(403, json={"error": node.REVOKED})
        reply = registry.handle(req.method, req.url.path, req.headers.multi_items(), req.content)
        return httpx.Response(reply.status, json=reply.body)

    with db.open_db(":memory:") as local:
        worker = node.Node(
            local, lambda: config, key_path=path, clock=clock, transport=httpx.MockTransport(handle)
        )
        worker.poll()
        config.network.donated.clear()
        event = worker.poll()
        assert event is not None and event.change is node.Change.REVOKED
        assert worker.revoked
        worker.close()


def test_search_refusal_records_only_reason_and_route(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    assert _ask(asker).status_code == 404
    [entry] = [entry for entry in store.entries(conn) if entry.action == "refused"]
    assert entry.details == {"reason": "no_lender", "method": "POST", "path": protocol.SEARCH_ASK}


def test_lent_search_has_its_own_gui_card_with_runners(tmp_path: Path) -> None:
    ConfigManager(default_config_path()).save(_lender())
    path = tmp_path / "panels.db"
    with closing(db.open_db(path)):
        front = WebFront()
        runners = Runners(path, front)
        try:
            [card] = snapshot(path, "modules", front)["modules"]
        finally:
            runners.close()
    assert (card["work"], card["scope"], card["scope_id"], card["role"]) == (
        [],
        "network",
        "",
        "search",
    )
    assert card["section"] == "network" and card["provider"] == "searxng"
    assert card["calls"] == 0 and card["calls_per_day"] == 1
    assert "SearXNG" in card["label"] and "tokens" not in card


def test_legacy_search_donation_upgrades_and_cli_omits_tokens() -> None:
    manager = ConfigManager(default_config_path())
    manager.config_path.parent.mkdir(exist_ok=True, parents=True)
    manager.config_path.write_text(
        json.dumps(
            {
                "schema_version": 10,
                "language": "en",
                "network": {"donated": {"searxng": {"calls_per_day": 4, "tokens_per_day": 1000}}},
            }
        ),
        encoding="utf-8",
    )
    assert manager.load().network.donated["searxng"].tokens_per_day == 0
    runner = CliRunner()
    enabled = runner.invoke(cli, ["global", "search", "on"])
    assert enabled.exit_code == 0 and "after `desman global join`" in enabled.output
    assert manager.load().network.borrow_search
    saved = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert (
        saved["schema_version"] == 15
        and saved["network"]["donated"]["searxng"]["tokens_per_day"] == 0
    )
    quota = runner.invoke(cli, ["global", "quota", "searxng", "--json"])
    assert json.loads(quota.output)["quota"] == {"calls_per_day": 4}
    assert json.loads(quota.output)["used_today"] == {"calls": 0}
    status = runner.invoke(cli, ["global", "status", "--json"])
    assert json.loads(status.output)["donated"] == {"searxng": {"calls_per_day": 4}}
    assert json.loads(status.output)["used_today"] == {"searxng": {"calls": 0}}
    for args in (["status"], ["quota", "searxng"]):
        result = runner.invoke(cli, ["global", *args])
        assert result.exit_code == 0 and "searches a day" in result.output
        assert "tokens" not in result.output


@pytest.mark.parametrize(
    ("enabled", "registry", "borrow", "key_state", "warns"),
    [
        (False, REGISTRY, True, "valid", True),
        (True, None, True, "valid", True),
        (True, REGISTRY, False, "valid", True),
        (True, REGISTRY, True, "missing", True),
        (True, REGISTRY, True, "invalid", True),
        (True, REGISTRY, True, "valid", False),
    ],
)
def test_no_engine_warning_requires_borrower_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    enabled: bool,
    registry: str | None,
    borrow: bool,
    key_state: str,
    warns: bool,
) -> None:
    config = _borrower()
    config.network.enabled = enabled
    config.network.registry = registry
    config.network.borrow_search = borrow
    config.permissions.network_access = True
    key = tmp_path / "node.key"
    if key_state != "missing":
        sign.create_node_key(key)
        if key_state == "invalid":
            key.write_bytes(b"invalid")
    monkeypatch.setattr(sign, "default_node_key_path", lambda: key)
    monkeypatch.setattr(net, "DEFAULT_REGISTRY", None)
    warnings: list[str] = []
    monkeypatch.setattr(ui, "warn", lambda text, **kwargs: warnings.append(text))
    runs.executor_access(config)
    assert any("no engine for the whole web" in warning for warning in warnings) is warns


@pytest.mark.parametrize("reply", ["gone", "missing_payload", "empty", "lender_error"])
def test_borrower_only_caches_complete_answers(clock: FakeClock, reply: str) -> None:
    id = "a" * 32
    state: dict[str, object] = {"id": id, "state": "gone" if reply == "gone" else "answered"}
    if reply == "empty":
        state["hits"] = []
    if reply == "lender_error":
        state.update(hits=[], error=f"untrusted lender text {QUERY}")

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path == protocol.SEARCH_ASK:
            return httpx.Response(200, json={"id": id, "wait_seconds": 30})
        assert req.url.path == protocol.SEARCH_STATE + id
        return httpx.Response(200, json=state)

    config = _borrower()
    with (
        db.open_db(":memory:") as conn,
        httpx.Client(transport=httpx.MockTransport(handle)) as http,
    ):
        web = runs.web(config, Store(conn, clock), http)
        web.engines = [
            Lent(config, conn, http, clock, sign.NodeKey.generate(), REGISTRY, sleep=clock.sleep)
        ]
        if reply == "empty":
            assert web.search(QUERY, 1)["results"] == []
        else:
            message = (
                "the lending node could not search"
                if reply == "lender_error"
                else "the lending node's answer is unavailable"
            )
            with pytest.raises(WebError, match=message) as exc:
                web.search(QUERY, 1)
            assert type(exc.value) is WebError
            assert "untrusted lender text" not in str(exc.value)
            assert "untrusted lender text" not in repr(netlog.entries(conn))
        assert conn.execute("SELECT count(*) FROM web_cache").fetchone()[0] == (reply == "empty")


def test_borrower_reports_registry_rate_limit(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock
) -> None:
    registry, _ = relay
    asker = Wire(registry, clock)
    asker.register()
    for _ in range(30):
        assert _ask(asker).json()["error"] == "no_lender"
    with db.open_db(":memory:") as conn:
        engine = Lent(
            _borrower(), conn, asker.client, clock, asker.key, REGISTRY, sleep=clock.sleep
        )
        with pytest.raises(
            WebError, match="lent search refused: too many searches, try later"
        ) as exc:
            engine.search(QUERY, 1)
        assert type(exc.value) is WebError


@pytest.mark.parametrize(
    ("path", "exception"),
    [
        (protocol.SEARCH_ASK, node.Problem),
        (protocol.SEARCH_STATE + "a" * 32, node.Problem),
        (protocol.SEARCH_NEXT, node.Unreachable),
        (protocol.SEARCH_ANSWER, node.Unreachable),
        (protocol.SEARCH_STATE + "z" * 32, node.Unreachable),
    ],
)
def test_only_borrower_routes_decode_429_as_problem(
    tmp_path: Path, clock: FakeClock, path: str, exception: type[Exception]
) -> None:
    key = sign.create_node_key(tmp_path / "node.key")
    transport = httpx.MockTransport(lambda _: httpx.Response(429, json={"error": "too_many"}))
    with httpx.Client(transport=transport) as http, pytest.raises(exception):
        node.call(
            http,
            key,
            REGISTRY,
            "GET",
            path,
            None,
            protocol.SearchState,
            clock(),
            time.monotonic() + node.TIMEOUT,
        )


@pytest.mark.parametrize("case", ["too_late_to_start", "expires_during_search", "failed_search"])
def test_lender_counts_runs_even_without_a_successful_delivery(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock, tmp_path: Path, case: str
) -> None:
    registry, remote = relay
    asker = Wire(registry, clock)
    asker.register()
    config = _lender()
    path = tmp_path / "lender.key"
    sign.create_node_key(path)
    polls: list[int] = []
    searches: list[str] = []

    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.host == "127.0.0.1":
            assert not local.in_transaction
            assert search_lend.used_today(local, clock()) == 1
            assert netlog.entries(local)[-1].details == {
                **search_lend.query_hash(QUERY),
                "day": clock().date().isoformat(),
            }
            searches.append(req.url.params["q"])
            if case == "failed_search":
                return httpx.Response(500)
            clock.advance(30)
            return httpx.Response(200, json={"results": [HIT.model_dump()]})
        if req.url.path == protocol.SEARCH_NEXT:
            polls.append(json.loads(req.content)["limit"])
        response = registry.handle(req.method, req.url.path, req.headers.multi_items(), req.content)
        return httpx.Response(response.status, json=response.body)

    with db.open_db(":memory:") as local:
        worker = node.Node(
            local, lambda: config, key_path=path, clock=clock, transport=httpx.MockTransport(handle)
        )
        try:
            worker.poll(idle=False)
            id = _ask(asker).json()["id"]
            clock.advance(26 if case == "too_late_to_start" else 5)
            worker.poll(idle=False)
            assert searches == ([] if case == "too_late_to_start" else [QUERY])
            assert search_lend.used_today(local, clock()) == (0 if not searches else 1)
            result = asker.call("GET", protocol.SEARCH_STATE + id).json()
            if case == "too_late_to_start":
                assert result["error"] == protocol.SEARCH_SKIPPED
            elif case == "failed_search":
                assert result["state"] == "answered" and result["error"]
            else:
                assert result["state"] == "expired"
            clock.advance(5)
            worker.poll(idle=False)
            assert polls == ([1, 1] if searches else [1, 1, 1])
            assert (
                remote.execute("SELECT count(answered_at) FROM search_requests").fetchone()[0] == 0
            )
        finally:
            worker.close()


@pytest.mark.parametrize("operation", ["donate", "quota"])
def test_searxng_cli_rejects_explicit_token_quota(operation: str) -> None:
    manager = ConfigManager(default_config_path())
    manager.save(_lender())
    before = manager.load().network.donated
    result = CliRunner().invoke(cli, ["global", operation, "searxng", "--tokens", "1000"])
    assert result.exit_code == 1
    assert "SearXNG quotas count searches only." in result.output
    assert manager.load().network.donated == before
