"""Lend search through the signed registry and the normal web search tool."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from conftest import FakeClock
from desman import commands, runs, searxng, searxng_native
from desman.cli import cli
from desman.config import Config, ConfigManager, Donation, Network, default_config_path
from desman.core import db, netlog, operator
from desman.core.search import Lent
from desman.core.store import Store
from desman.core.tools import ToolError, WebSearchArgs, web_tools
from desman.core.web import SearchResult, WebError
from desman.core.workspace import Workspace
from desman.net import node, protocol, search_lend, sign
from desman.registry import store
from desman.registry.server import Registry
from helpers import make_config
from test_registry import Node as Wire

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
    assert lender.call("POST", protocol.SEARCH_NEXT).json() == {"jobs": []}
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
    jobs = lender.call("POST", protocol.SEARCH_NEXT).json()
    assert jobs == {"jobs": [{"id": offered.id, "query": QUERY, "count": 2}]}
    assert lender.call("POST", protocol.SEARCH_NEXT).json() == {"jobs": []}
    assert _answer(lender, offered.id, [HIT]).status_code == 200
    assert _answer(lender, offered.id, []).status_code == 200
    assert conn.execute("SELECT query FROM search_requests").fetchone()[0] is None
    assert asker.call("GET", path).json()["hits"] == [HIT.model_dump()]
    assert asker.call("GET", path).json()["hits"] == []
    assert tuple(conn.execute("SELECT query, payload FROM search_requests").fetchone()) == (
        None,
        None,
    )

    expires = _ask(asker).json()["id"]
    clock.advance(60)
    assert asker.heartbeat().status_code == 200  # the ordinary registry tick also erases
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
    asker.call("POST", protocol.SEARCH_NEXT)
    lender = _lend(registry, clock, 1)
    conn.execute("UPDATE nodes SET owner = 'same owner'")
    assert _ask(asker).json()["error"] == "no_lender"
    conn.execute("UPDATE nodes SET owner = NULL")
    clock.advance(31)
    assert _ask(asker).json()["error"] == "no_lender"
    lender.call("POST", protocol.SEARCH_NEXT)
    id = _ask(asker).json()["id"]
    assert _ask(asker).json()["error"] == "no_lender"  # reserve the last possible answer
    assert _answer(lender, id, []).status_code == 200
    assert _ask(asker).json()["error"] == "no_lender"
    clock.advance(86400)
    lender.call("POST", protocol.SEARCH_NEXT)
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
    jobs = second.call("POST", protocol.SEARCH_NEXT).json()["jobs"]
    assert [job["id"] for job in jobs] == [id]
    assert _answer(first, id, []).status_code == 404
    clock.advance(11)
    assert first.call("POST", protocol.SEARCH_NEXT).json() == {"jobs": []}
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
        lender.call("POST", protocol.SEARCH_NEXT)
        for _ in range(30):
            assert _ask(asker).status_code == 200
        assert _ask(asker).json()["error"] == "too_many"
        clock.advance(600)
    lender.call("POST", protocol.SEARCH_NEXT)
    assert _ask(asker).json()["error"] == "too_many"
    clock.advance(86400)
    lender.call("POST", protocol.SEARCH_NEXT)
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
        with pytest.raises(WebError, match="no node lends search now"):
            engine.search(QUERY, 1)
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
            with pytest.raises(ToolError, match="no node lends search now"):
                tool.run(WebSearchArgs(query=QUERY), Workspace(tmp_path, "0" * 32))
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
    assert current.schema_version == 10 and not current.network.borrow_search
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
        with pytest.raises(WebError, match="no node lends search now"):
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
            conn, config, protocol.SearchJob(id="1" * 32, query=QUERY), http, clock
        )
    assert result is not None and result.hits == [HIT]
    assert called == [SEARXNG]
