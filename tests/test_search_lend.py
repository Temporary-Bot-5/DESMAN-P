"""Lend search through the signed registry and the normal web search tool."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
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
from desman.core.tools import MissingSearchToolError, ToolError, WebSearchArgs, web_tools
from desman.core.web import MissingSearchEngine, SearchResult, Web, WebError
from desman.core.workspace import Workspace
from desman.gui.module_view import cards
from desman.net import node, protocol, search_lend, sign
from desman.net.search_lend import Lent
from desman.registry import admin, clusters, store
from desman.registry.server import Registry
from helpers import make_config
from test_registry import LOCAL, root, root_pub
from test_registry import Node as Wire

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
    assert jobs == {"jobs": [{"id": offered.id, "query": QUERY, "count": 2}]}
    assert lender.call("POST", protocol.SEARCH_NEXT, {"limit": 4}).json() == {"jobs": []}
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
    clock.advance(30)
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
        error = WebError if case == "unreachable" else MissingSearchEngine
        match = "unreachable" if case == "unreachable" else "no node lends search now"
        with pytest.raises(error, match=match):
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
            with pytest.raises(ToolError, match="the lending node's SearXNG failed") as exc:
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


def test_borrower_registration_does_not_advertise_models(
    relay: tuple[Registry, sqlite3.Connection],
    clock: FakeClock,
    root: sign.NodeKey,
    root_pub: Path,
) -> None:
    registry, conn = relay
    operator_key = sign.NodeKey.generate()
    admin.assign(conn, root, operator_key.public, {sign.Scope.TASKS}, days=30, now=clock())
    job = admin.add_job(
        conn,
        operator_key,
        protocol.Job(description="Sum 2 and 3", acceptance_criteria="Says 5"),
        days=7,
        now=clock(),
        root_pub=root_pub,
    )
    asker = Wire(registry, clock)
    config = _borrower()
    config.settings["OLLAMA_MODEL"] = "qwen2.5:7b"
    config.network.donated["ollama"] = Donation(calls_per_day=5, tokens_per_day=1000)
    assert [c.provider for c in node.capabilities(config)] == ["ollama"]

    with db.open_db(":memory:") as local:
        engine = Lent(config, local, asker.client, clock, asker.key, REGISTRY, sleep=clock.sleep)
        with pytest.raises(MissingSearchEngine):
            engine.search(QUERY, 1)
        [entry] = netlog.entries(local)
        assert entry.action == "registered"
        assert entry.details == {
            "registry": REGISTRY,
            "node": asker.fingerprint,
            "capabilities": [],
        }

    offer = asker.call("POST", protocol.CLAIM, {"jobs": True, "lead": True}).json()
    assert protocol.Offer.model_validate(offer).work is None
    found = store.node(conn, asker.fingerprint)
    assert found is not None and found.capabilities == [] and found.takes_jobs
    assert clusters.free_nodes(conn, clock(), timedelta(seconds=30)) == []
    normal = Wire(registry, clock)
    normal.register([LOCAL])
    control = protocol.Offer.model_validate(normal.call("POST", protocol.CLAIM, {}).json())
    assert control.work is not None and control.work.task.record.id == job.id


def test_offer_can_shorten_wait_and_web_preserves_missing_engine(clock: FakeClock) -> None:
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
        with pytest.raises(MissingSearchToolError, match="no node lends search now"):
            tool.run(WebSearchArgs(query=QUERY), Workspace(Path("."), "0" * 32))
    assert clock.slept == 3


@pytest.mark.parametrize("change", ["disabled", "quota"])
def test_lender_reports_jobs_it_cannot_run(
    relay: tuple[Registry, sqlite3.Connection], clock: FakeClock, tmp_path: Path, change: str
) -> None:
    registry, conn = relay
    asker = Wire(registry, clock)
    asker.register()
    config = _lender()
    config.network.donated["searxng"] = Donation(calls_per_day=3, tokens_per_day=0)
    path = tmp_path / "lender.key"
    sign.create_node_key(path)
    poll_limits: list[int] = []

    with db.open_db(":memory:") as local:

        def handle(req: httpx.Request) -> httpx.Response:
            assert req.url.host != "127.0.0.1"
            reply = registry.handle(
                req.method, req.url.path, req.headers.multi_items(), req.content
            )
            if req.url.path == protocol.SEARCH_NEXT:
                poll_limits.append(json.loads(req.content)["limit"])
                if reply.body["jobs"]:
                    if change == "disabled":
                        config.network.enabled = False
                    else:
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
        worker.poll()
        assert poll_limits == [2, 2]
        for id in (first, second):
            assert (
                asker.call("GET", protocol.SEARCH_STATE + id).json()["error"]
                == protocol.SEARCH_SKIPPED
            )
        assert asker.call("GET", protocol.SEARCH_STATE + third).json()["state"] == "waiting"
        assert search_lend.used_today(local, clock()) == 1
        assert conn.execute("SELECT count(answered_at) FROM search_requests").fetchone()[0] == 0
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


def test_lent_search_has_its_own_gui_card() -> None:
    with db.open_db(":memory:") as conn:
        [card] = cards(conn, _lender(), [])
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
        saved["schema_version"] == 10
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
    ("enabled", "registry", "warns"),
    [
        (False, REGISTRY, True),
        (True, None, True),
        (True, REGISTRY, False),
    ],
)
def test_no_engine_warning_requires_registry(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, registry: str | None, warns: bool
) -> None:
    config = _borrower()
    config.network.enabled = enabled
    config.network.registry = registry
    config.permissions.network_access = True
    monkeypatch.setattr(net, "DEFAULT_REGISTRY", None)
    warnings: list[str] = []
    monkeypatch.setattr(ui, "warn", lambda text, **kwargs: warnings.append(text))
    runs.executor_access(config)
    assert any("no engine for the whole web" in warning for warning in warnings) is warns
