"""Execute borrowed searches within the lender's local UTC daily quota."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from pydantic import ValidationError

from desman import net, searxng
from desman.config import Config
from desman.core import db, netlog
from desman.core.web import SearchResult, WebError
from desman.net import protocol


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
) -> protocol.SearchAnswer | None:
    limit = quota(config)
    now = clock()
    with db.transaction(conn):
        if not limit or used_today(conn, now) >= limit:
            return None
        netlog.write(
            conn,
            "lent_search",
            {**query_hash(job.query), "day": now.astimezone(UTC).date().isoformat()},
        )
    try:
        found = searxng.engine(config, http, clock).search(job.query, job.count)
    except WebError:
        return protocol.SearchAnswer(id=job.id, hits=[], error="SearXNG search failed")
    return protocol.SearchAnswer(id=job.id, hits=hits(found))
