"""Short-lived search requests relayed between registered nodes."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from uuid import uuid4

from desman.core import db
from desman.net import protocol
from desman.registry import store

LIFETIME = protocol.SEARCH_WAIT
POLL_FRESHNESS = 30
TAKE_WINDOW = 10
DAY = 86400


class Refusal(Exception):
    pass


def _expire(conn: sqlite3.Connection, stamp: float) -> None:
    rows = conn.execute(
        """UPDATE search_requests SET state = 'expired', query = NULL
           WHERE (state = 'waiting' OR payload IS NOT NULL)
             AND asked_at <= ? AND state = 'waiting' RETURNING id, asker, lender""",
        (stamp - LIFETIME,),
    ).fetchall()
    for row in rows:
        _log(conn, "expired", row["id"], row["asker"], row["lender"])
    conn.execute(
        """UPDATE search_requests SET payload = NULL
           WHERE (state = 'waiting' OR payload IS NOT NULL)
             AND asked_at <= ? AND payload IS NOT NULL""",
        (stamp - LIFETIME,),
    )


def expire(conn: sqlite3.Connection, now: datetime) -> None:
    stamp = now.timestamp()
    with db.transaction(conn):
        _expire(conn, stamp)
        conn.execute("DELETE FROM search_ask_attempts WHERE asked_at < ?", (stamp - DAY,))


def _log(conn: sqlite3.Connection, action: str, id: str, asker: str, lender: str) -> None:
    store.log(conn, f"search_{action}", asker, {"id": id, "lender": lender})


def _quota(node: store.Node) -> int:
    return next(
        (c.calls_per_day for c in node.capabilities if c.provider == protocol.SEARCH_PROVIDER), 0
    )


def _used(conn: sqlite3.Connection, lender: str, stamp: float) -> int:
    return int(
        conn.execute(
            """SELECT count(*) FROM search_requests
               WHERE lender = ? AND state = 'answered' AND answered_at >= ?""",
            (lender, stamp // DAY * DAY),
        ).fetchone()[0]
    )


def _lender(
    conn: sqlite3.Connection, asker: store.Node, stamp: float, previous: str = ""
) -> str | None:
    for row in conn.execute(
        """SELECT lender FROM search_polls WHERE polled_at >= ? AND lender NOT IN (?, ?)
           ORDER BY polled_at DESC, lender""",
        (stamp - POLL_FRESHNESS, asker.fingerprint, previous),
    ):
        lender = store.node(conn, row["lender"])
        if (
            lender is None
            or lender.status != "active"
            or lender.revoked
            or lender.owned_by == asker.owned_by
        ):
            continue
        pending = conn.execute(
            "SELECT count(*) FROM search_requests WHERE lender = ? AND state = 'waiting'",
            (lender.fingerprint,),
        ).fetchone()[0]
        if _used(conn, lender.fingerprint, stamp) + pending < _quota(lender):
            return lender.fingerprint
    return None


def ask(
    conn: sqlite3.Connection, asker: store.Node, request: protocol.SearchAsk, now: datetime
) -> protocol.SearchAsked:
    stamp = now.timestamp()
    with db.transaction(conn):
        recent, daily = conn.execute(
            """SELECT count(*) FILTER (WHERE asked_at > ?),
                      count(*) FILTER (WHERE asked_at >= ?)
               FROM search_ask_attempts WHERE asker = ? AND asked_at >= ?""",
            (
                stamp - 600,
                stamp // DAY * DAY,
                asker.fingerprint,
                min(stamp - 600, stamp // DAY * DAY),
            ),
        ).fetchone()
        if recent >= 30 or daily >= 300:
            raise Refusal("too_many")
        conn.execute(
            "INSERT INTO search_ask_attempts (asker, asked_at) VALUES (?, ?)",
            (asker.fingerprint, stamp),
        )
        lender = _lender(conn, asker, stamp)
        id = uuid4().hex
        if lender is not None:
            conn.execute(
                """INSERT INTO search_requests
                   (id, asker, lender, asked_at, assigned_at, query, count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (id, asker.fingerprint, lender, stamp, stamp, request.query, request.count),
            )
            _log(conn, "asked", id, asker.fingerprint, lender)
    if lender is None:
        raise Refusal("no_lender")
    return protocol.SearchAsked(id=id, wait_seconds=LIFETIME)


def next_jobs(
    conn: sqlite3.Connection, lender: store.Node, limit: int, now: datetime
) -> protocol.SearchJobs:
    stamp = now.timestamp()

    def take(count: int) -> list[sqlite3.Row]:
        return conn.execute(
            """UPDATE search_requests SET taken = 1 WHERE id IN (
                   SELECT id FROM search_requests
                   WHERE lender = ? AND state = 'waiting' AND taken = 0
                   ORDER BY assigned_at, id LIMIT ?
               ) RETURNING id, query, count""",
            (lender.fingerprint, count),
        ).fetchall()

    with db.transaction(conn):
        taken = conn.execute(
            "SELECT count(*) FROM search_requests WHERE lender = ? AND state = 'waiting' AND taken = 1",
            (lender.fingerprint,),
        ).fetchone()[0]
        remaining = min(limit, _quota(lender) - _used(conn, lender.fingerprint, stamp) - taken)
        if remaining <= 0:
            return protocol.SearchJobs(jobs=[])
        conn.execute(
            """INSERT INTO search_polls VALUES (?, ?)
               ON CONFLICT (lender) DO UPDATE SET polled_at = max(polled_at, excluded.polled_at)""",
            (lender.fingerprint, stamp),
        )
        rows = take(remaining)
        for row in conn.execute(
            """SELECT id, asker, lender FROM search_requests
               WHERE state = 'waiting' AND taken = 0 AND reassigned = 0 AND assigned_at <= ?
                 AND lender != ?
               ORDER BY assigned_at, id""",
            (stamp - TAKE_WINDOW, lender.fingerprint),
        ).fetchall():
            asker = store.node(conn, row["asker"])
            if asker is None or asker.status != "active" or asker.revoked:
                continue
            replacement = _lender(conn, asker, stamp, row["lender"])
            if replacement is not None:
                conn.execute(
                    """UPDATE search_requests SET lender = ?, reassigned = 1, assigned_at = ?
                       WHERE id = ?""",
                    (replacement, stamp, row["id"]),
                )
        if len(rows) < remaining:
            rows.extend(take(remaining - len(rows)))
    return protocol.SearchJobs(jobs=[protocol.SearchJob(**dict(row)) for row in rows])


def answer(
    conn: sqlite3.Connection, lender: store.Node, result: protocol.SearchAnswer, now: datetime
) -> protocol.SearchState:
    stamp = now.timestamp()
    with db.transaction(conn):
        row = conn.execute(
            "SELECT * FROM search_requests WHERE id = ? AND lender = ?",
            (result.id, lender.fingerprint),
        ).fetchone()
        if row is None:
            raise Refusal("not_found")
        if row["state"] == "waiting":
            # A taken job the node could not run still needs a terminal reply.
            skipped = row["taken"] and result.error == protocol.SEARCH_SKIPPED and not result.hits
            if not skipped and _used(conn, lender.fingerprint, stamp) >= _quota(lender):
                raise Refusal("too_many")
            conn.execute(
                """UPDATE search_requests SET state = 'answered', query = NULL, payload = ?,
                   answered_at = ? WHERE id = ?""",
                (result.model_dump_json(), None if skipped else stamp, result.id),
            )
            _log(conn, "answered", result.id, row["asker"], lender.fingerprint)
            return protocol.SearchState(id=result.id, state="answered", hits=[])
        return protocol.SearchState(id=result.id, state=row["state"], hits=[])


def get(conn: sqlite3.Connection, asker: store.Node, id: str) -> protocol.SearchState:
    with db.transaction(conn):
        row = conn.execute(
            "SELECT state, payload FROM search_requests WHERE id = ? AND asker = ?",
            (id, asker.fingerprint),
        ).fetchone()
        if row is None:
            raise Refusal("not_found")
        payload = row["payload"]
        if payload is None:
            return protocol.SearchState(id=id, state=row["state"], hits=[])
        found = protocol.SearchAnswer.model_validate_json(payload)
        conn.execute("UPDATE search_requests SET payload = NULL WHERE id = ?", (id,))
        return protocol.SearchState(id=id, state="answered", hits=found.hits, error=found.error)
