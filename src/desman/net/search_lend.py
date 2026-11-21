"""Execute borrowed searches within the lender's local UTC daily quota."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Self

import httpx
from pydantic import ValidationError

from desman import net, providers, searxng
from desman.config import Config, ConfigError
from desman.core import netlog
from desman.core.web import MissingSearchEngine, SearchResult, WebError
from desman.net import protocol, sign


def query_hash(query: str) -> dict[str, str]:
    return {"sha256": hashlib.sha256(query.encode()).hexdigest()}


def quota(config: Config) -> int:
    donated = config.network.donated.get(protocol.SEARCH_PROVIDER)
    if donated is None or net.registry(config) is None or not searxng.managed_search(config):
        return 0
    return donated.calls_per_day


def used_today(conn: sqlite3.Connection, now: datetime) -> int:
    return int(
        conn.execute(
            """SELECT count(*) FROM network_log
               WHERE action = 'lent_search' AND json_extract(details_json, '$.day') = ?""",
            (now.astimezone(UTC).date().isoformat(),),
        ).fetchone()[0]
    )


def hits(results: list[SearchResult]) -> list[protocol.SearchHit]:
    found = []
    for result in results[:10]:
        try:
            found.append(
                protocol.SearchHit(
                    title=result.title[:300], url=result.url, snippet=result.snippet[:1000]
                )
            )
        except ValidationError:
            continue
    return found


def answer(
    conn: sqlite3.Connection,
    config: Config,
    job: protocol.SearchJob,
    http: httpx.Client,
    clock: Callable[[], datetime],
) -> protocol.SearchAnswer:
    limit = quota(config)
    now = clock()
    if not limit or used_today(conn, now) >= limit:
        return protocol.SearchAnswer(id=job.id, hits=[], error=protocol.SEARCH_SKIPPED)
    try:
        found = searxng.engine(config, http, clock).search(job.query, job.count)
    except WebError:
        return protocol.SearchAnswer(id=job.id, hits=[], error="the lending node's SearXNG failed")
    return protocol.SearchAnswer(id=job.id, hits=hits(found))


def record_delivery(conn: sqlite3.Connection, job: protocol.SearchJob, now: datetime) -> None:
    netlog.record(
        conn,
        "lent_search",
        {**query_hash(job.query), "day": now.astimezone(UTC).date().isoformat()},
    )


def borrower_identity(config: Config) -> tuple[str, sign.NodeKey] | None:
    url = net.registry(config)
    if url is None or not config.network.borrow_search or providers.search_engines(config):
        return None
    try:
        key = sign.load_node_key(sign.default_node_key_path())
    except (sign.KeyFileError, OSError):
        return None
    return url, key


class Lent:
    """The whole-web engine relayed by the registry with the person's consent."""

    name = "desman-global"

    def __init__(
        self,
        config: Config,
        conn: sqlite3.Connection,
        http: httpx.Client,
        clock: Callable[[], datetime],
        key: sign.NodeKey,
        url: str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        load: Callable[[], Config] | None = None,
    ) -> None:
        self.load = load or (lambda: config)
        self.conn = conn
        self.http = http
        self.clock = clock
        self.key = key
        self.url = url
        self.sleep = sleep

    @classmethod
    def from_config(
        cls,
        config: Config,
        conn: sqlite3.Connection,
        http: httpx.Client,
        clock: Callable[[], datetime],
        *,
        load: Callable[[], Config] | None = None,
    ) -> Self | None:

        identity = borrower_identity(config)
        if identity is None:
            return None
        url, key = identity
        return cls(config, conn, http, clock, key, url, load=load)

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        from desman.net import node

        try:
            request = protocol.SearchAsk(query=query, count=max_results)
        except ValidationError:
            raise WebError(
                "lent search requires a query of 1–300 characters without controls"
            ) from None
        budget = protocol.SEARCH_WAIT
        started = time.monotonic()
        start = self.clock()

        def remaining() -> float:
            return min(
                budget - (time.monotonic() - started),
                budget - (self.clock() - start).total_seconds(),
            )

        def check_config() -> Config:
            config = self.load()
            if (
                not config.network.borrow_search
                or net.registry(config) != self.url
                or providers.search_engines(config)
            ):
                raise WebError("lent search is unavailable")
            return config

        def ask() -> protocol.SearchAsked:
            check_config()
            return node.call(
                self.http,
                self.key,
                self.url,
                "POST",
                protocol.SEARCH_ASK,
                request,
                protocol.SearchAsked,
                self.clock(),
                time.monotonic() + remaining(),
            )

        offered = False
        try:
            try:
                offer = ask()
            except node.Problem as e:
                if e.error != node.NOT_REGISTERED:
                    raise
                lent = node.capabilities(check_config())
                node.call(
                    self.http,
                    self.key,
                    self.url,
                    "POST",
                    protocol.REGISTER,
                    protocol.Register(capabilities=lent),
                    protocol.NodeState,
                    self.clock(),
                    time.monotonic() + remaining(),
                )
                netlog.record(
                    self.conn,
                    "registered",
                    {
                        "registry": self.url,
                        "node": self.key.fingerprint,
                        "capabilities": [cap.model_dump(mode="json") for cap in lent],
                    },
                )
                offer = ask()
            offered = True
            budget = min(offer.wait_seconds, protocol.SEARCH_WAIT)
            netlog.record(self.conn, "borrowed_search", query_hash(request.query))
            while (left := remaining()) > 0:
                self.sleep(min(1, left))
                if remaining() <= 0:
                    break
                check_config()
                state = node.call(
                    self.http,
                    self.key,
                    self.url,
                    "GET",
                    protocol.SEARCH_STATE + offer.id,
                    None,
                    protocol.SearchState,
                    self.clock(),
                    time.monotonic() + remaining(),
                )
                if state.id != offer.id or state.state == "gone":
                    raise WebError("the lending node's answer is unavailable")
                if state.state == "expired":
                    break
                if state.state == "answered":
                    if state.error is not None:
                        raise WebError("the lending node could not search")
                    if state.hits is None:
                        raise WebError("the lending node's answer is unavailable")
                    return [SearchResult(**hit.model_dump()) for hit in state.hits[:max_results]]
        except node.Problem as e:
            if e.error == "no_lender":
                raise MissingSearchEngine("no node lends search now") from None
            if e.error == "too_many":
                raise WebError("lent search refused: too many searches, try later") from None
            raise WebError(f"lent search refused: {e.error}") from None
        except node.Unreachable as e:
            if not offered or str(e) != "ContactTimeout":
                raise WebError("the search relay is unreachable") from None
        except (ConfigError, sqlite3.OperationalError):
            raise WebError("lent search is unavailable") from None
        raise WebError("the lending node did not answer in time")
