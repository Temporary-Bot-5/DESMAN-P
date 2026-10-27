"""The log of model calls (ROADMAP 2.4): what was spent, per provider and per key.

Rows are written by the LLM client through `recorder()` and never changed. A key appears only
as its fingerprint (`openai_compat.key_id`), so the log can be shared without leaking keys.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from desman.core import db
from desman.llm.openai_compat import CallRecord


def recorder(conn: sqlite3.Connection) -> Callable[[CallRecord], None]:
    """An `on_call` for `OpenAICompatible` that appends to model_calls."""

    def record(call: CallRecord) -> None:
        conn.execute(
            "INSERT INTO model_calls (provider, model, key_id, prompt_hash, response_hash, "
            "prompt_tokens, completion_tokens, latency_ms, http_status, error_kind, error, "
            "limits_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                call.provider,
                call.model,
                call.key_id,
                call.prompt_hash,
                call.response_hash,
                call.prompt_tokens,
                call.completion_tokens,
                call.latency_ms,
                call.http_status,
                call.error_kind,
                call.error,
                json.dumps(call.limits),
            ),
        )

    return record


@dataclass(frozen=True)
class KeyUsage:
    provider: str
    key_id: str | None
    calls: int
    failed: int
    limited: int  # calls that hit a rate limit or quota
    prompt_tokens: int
    completion_tokens: int
    last_at: str | None
    last_error: str | None
    limits: dict[str, str]  # the rate-limit headers of the latest call that had any


def usage(conn: sqlite3.Connection, since: datetime | None = None) -> list[KeyUsage]:
    """Per provider and key, oldest provider first; `since` limits the window."""
    where, args = ("WHERE at >= ?", [db.to_iso(since)]) if since else ("", [])
    rows = conn.execute(
        f"""
        SELECT provider, key_id, COUNT(*) AS calls,
               SUM(error_kind IS NOT NULL) AS failed,
               SUM(error_kind = 'limit') AS limited,
               SUM(prompt_tokens) AS prompt_tokens,
               SUM(completion_tokens) AS completion_tokens,
               MAX(at) AS last_at
        FROM model_calls {where}
        GROUP BY provider, key_id
        ORDER BY MIN(seq)
        """,
        args,
    ).fetchall()
    out = []
    for r in rows:
        last = conn.execute(
            "SELECT error FROM model_calls WHERE provider = ? AND key_id IS ? "
            "ORDER BY seq DESC LIMIT 1",
            (r["provider"], r["key_id"]),
        ).fetchone()
        limits = conn.execute(
            "SELECT limits_json FROM model_calls WHERE provider = ? AND key_id IS ? "
            "AND limits_json != '{}' ORDER BY seq DESC LIMIT 1",
            (r["provider"], r["key_id"]),
        ).fetchone()
        out.append(
            KeyUsage(
                provider=r["provider"],
                key_id=r["key_id"],
                calls=r["calls"],
                failed=r["failed"] or 0,
                limited=r["limited"] or 0,
                prompt_tokens=r["prompt_tokens"] or 0,
                completion_tokens=r["completion_tokens"] or 0,
                last_at=r["last_at"],
                last_error=last["error"] if last else None,
                limits=json.loads(limits["limits_json"]) if limits else {},
            )
        )
    return out
