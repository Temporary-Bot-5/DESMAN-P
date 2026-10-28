"""Replies kept by request (ROADMAP 2.6), so a resume or a re-review does not pay twice.

A request is identified by provider, model, messages, temperature and max_tokens. A retry
after a reviewer's remarks asks something else (the attempt number and the remarks are in the
prompt), so it is never served from here; what is served is the exact same question asked
again, typically after a crash or an interrupted review.

Only replies the caller *accepted* are kept: `complete_json()` calls `accept()` once a reply
passed its schema. A reply that turned out unusable is never cached, or asking again would
return the same garbage until the attempts ran out. Entries older than `TTL` are ignored and
pruned: free models change behind the same name.

Lives in the task database (`response_cache`), next to the call log.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from desman.core import db
from desman.llm.base import LLM, Completion, Request, Usage

TTL = timedelta(days=7)


def request_key(provider: str, model: str, request: Request) -> str:
    body = {
        "provider": provider,
        "model": request.model or model,
        "messages": [m.model_dump() for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class Cached:
    """An LLM that answers a request it has an accepted reply for without calling `inner`."""

    def __init__(
        self,
        llm: LLM,
        conn: sqlite3.Connection,
        model: str,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.inner = llm
        self.name = llm.name
        self.conn = conn
        self.model = model
        self.clock = clock
        self.hits = 0

    def complete(self, request: Request) -> Completion:
        row = self.conn.execute(
            "SELECT text, finish_reason, prompt_tokens, completion_tokens, model "
            "FROM response_cache WHERE key = ? AND created_at >= ?",
            (self._key(request), db.to_iso(self.clock() - TTL)),
        ).fetchone()
        if row is None:
            return self.inner.complete(request)
        self.hits += 1
        return Completion(
            text=row["text"],
            provider=self.name,
            model=row["model"],
            usage=Usage(
                prompt_tokens=row["prompt_tokens"], completion_tokens=row["completion_tokens"]
            ),
            finish_reason=row["finish_reason"],
        )

    def accept(self, request: Request, reply: Completion) -> None:
        """Keep `reply` for `request`: the caller found it usable."""
        now = self.clock()
        with db.transaction(self.conn):
            self.conn.execute(
                "DELETE FROM response_cache WHERE created_at < ?", (db.to_iso(now - TTL),)
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO response_cache (key, provider, model, text, "
                "finish_reason, prompt_tokens, completion_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._key(request),
                    self.name,
                    reply.model,
                    reply.text,
                    reply.finish_reason,
                    reply.usage.prompt_tokens,
                    reply.usage.completion_tokens,
                    db.to_iso(now),
                ),
            )

    def _key(self, request: Request) -> str:
        return request_key(self.name, self.model, request)
