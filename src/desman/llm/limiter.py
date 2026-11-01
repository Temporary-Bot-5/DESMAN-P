"""A pace per provider and model, kept in the database (ROADMAP 4.5): it outlives a restart and
is shared by every worker on the database.

Each provider/model pair has a token bucket (`rate_buckets`): it holds up to a minute's worth of
requests (`catalog.Model.rpm` times the provider's keys) and refills at that rate. A call takes
one token. A limit the provider reported (429 with a wait, a used-up quota with its reset) is
remembered too, as `blocked_until`, so no process calls it again before then: free tiers count
per model, and a restarted run would otherwise spend its first calls finding out again.

A short wait (`MAX_WAIT` seconds or less) is waited out here; a longer one is `RateLimited`
raised without a call, so the run moves on to the next model (`router.Fallback`) or, when
every model has to wait, the subtask pauses until the soonest.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from desman.core import db
from desman.llm.base import LLM, Completion, QuotaExhausted, RateLimited, Request
from desman.llm.openai_compat import OpenAICompatible

MAX_WAIT = 10.0
"""Seconds a call may wait for its turn here (as long as `router.Backoff` waits out a limit)."""
DEFAULT_BLOCK = 60.0
"""How long a 429 that named no wait keeps the model aside."""


def _now() -> datetime:
    return datetime.now(UTC)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)  # looked up at call time, so tests replace it


class Paced(RateLimited):
    """Raised here, before anything was sent: the bucket's own wait, not a limit the provider
    reported (so it blocks nothing)."""


class Buckets:
    """The token buckets in the database."""

    def __init__(
        self, conn: sqlite3.Connection, clock: Callable[[], datetime] | None = None
    ) -> None:
        self.conn = conn
        self.clock = clock or _now

    def take(self, scope: str, per_minute: float) -> float:
        """Takes a token of `scope`, refilled at `per_minute`: 0.0 if one was taken, else the
        seconds until one can be."""
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        rate = per_minute / 60.0
        now = self.clock()
        with db.transaction(self.conn):
            row = self.conn.execute(
                "SELECT tokens, updated_at, blocked_until FROM rate_buckets WHERE scope = ?",
                (scope,),
            ).fetchone()
            if row is None:
                tokens, blocked = per_minute, None
            else:
                # Another process's clock may be a little ahead: time never runs backwards here.
                elapsed = max(0.0, (now - db.parse_iso(row["updated_at"])).total_seconds())
                tokens = min(per_minute, row["tokens"] + elapsed * rate)
                blocked = db.parse_iso(row["blocked_until"]) if row["blocked_until"] else None
            if blocked is not None and blocked > now:
                wait = (blocked - now).total_seconds()
            elif tokens >= 1.0:
                tokens, wait, blocked = tokens - 1.0, 0.0, None
            else:
                wait, blocked = (1.0 - tokens) / rate, None
            self._write(scope, tokens, now, blocked)
        return wait

    def block(self, scope: str, until: datetime) -> None:
        """No token of `scope` before `until` (a limit the provider reported); an earlier block
        already set stays if it ends later."""
        now = self.clock()
        with db.transaction(self.conn):
            row = self.conn.execute(
                "SELECT tokens, blocked_until FROM rate_buckets WHERE scope = ?", (scope,)
            ).fetchone()
            if row is not None and row["blocked_until"]:
                until = max(until, db.parse_iso(row["blocked_until"]))
            # The bucket is emptied: when the block ends, the pace starts again from nothing.
            self._write(scope, 0.0, now, until)

    def blocked_until(self, scope: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT blocked_until FROM rate_buckets WHERE scope = ?", (scope,)
        ).fetchone()
        if row is None or not row["blocked_until"]:
            return None
        until = db.parse_iso(row["blocked_until"])
        return until if until > self.clock() else None

    def _write(self, scope: str, tokens: float, now: datetime, blocked: datetime | None) -> None:
        self.conn.execute(
            "INSERT INTO rate_buckets (scope, tokens, updated_at, blocked_until) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (scope) DO UPDATE SET tokens = excluded.tokens, "
            "updated_at = excluded.updated_at, blocked_until = excluded.blocked_until",
            (scope, tokens, db.to_iso(now), None if blocked is None else db.to_iso(blocked)),
        )


class Limited:
    """An LLM that calls `inner` only when the bucket of `scope` gives it a token.

    The bucket counts requests sent, not calls: a client that may send several for one call (a
    key refused, the next one tried) takes a token before each (`OpenAICompatible.pace`), and
    every repeat of `router.Backoff`, which wraps this, is a call of its own.
    """

    def __init__(
        self,
        llm: LLM,
        buckets: Buckets,
        scope: str,
        per_minute: float,
        *,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.inner = llm
        self.name = llm.name
        self.buckets = buckets
        self.scope = scope
        self.per_minute = per_minute
        self.sleep = sleep
        self._per_request = isinstance(llm, OpenAICompatible)
        if isinstance(llm, OpenAICompatible):
            llm.pace = self.pace

    def pace(self) -> None:
        """Takes a token, waiting for it if that is short; raises `RateLimited` if it is not
        (nothing is sent: the provider would only refuse it)."""
        waited = 0.0
        while (wait := self.buckets.take(self.scope, self.per_minute)) > 0.0:
            if waited + wait > MAX_WAIT:
                raise Paced(self.name, retry_after=wait)
            (self.sleep or _sleep)(wait)
            waited += wait

    def complete(self, request: Request) -> Completion:
        if not self._per_request:
            self.pace()
        try:
            return self.inner.complete(request)
        except Paced:
            raise
        except RateLimited as e:
            after = DEFAULT_BLOCK if e.retry_after is None else e.retry_after
            self.buckets.block(self.scope, self.buckets.clock() + timedelta(seconds=after))
            raise
        except QuotaExhausted as e:
            until = e.paused_until(self.buckets.clock())
            if until is not None:
                self.buckets.block(self.scope, until)
            raise
