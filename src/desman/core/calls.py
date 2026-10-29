"""The log of model calls (ROADMAP 2.4): what was spent, per provider and per key.

Rows are written by the LLM client through `recorder()` and never changed. A key appears only
as its fingerprint (`openai_compat.key_id`), so the log can be shared without leaking keys.

A call is attributed to what it was made for by whoever makes it: the coordinator around its
executor, reviewer and decomposer, the chat around the interlocutor and the planner (`for_`).
The client itself knows nothing of tasks; the recorder reads the attribution when it writes.
Whoever shows progress watches the calls as they are written (`watching`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from desman.core import db
from desman.llm.openai_compat import CallRecord


class Role(StrEnum):
    """What a call was made for. Keep in sync with the `model_calls.role` CHECK (a test does)."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    DECOMPOSER = "decomposer"
    INTERLOCUTOR = "interlocutor"


@dataclass(frozen=True)
class Purpose:
    role: Role | None
    task_id: str | None = None
    subtask_id: str | None = None
    session_id: str | None = None
    """The chat session of the interlocutor's own calls (and of planning a chat task)."""


_UNATTRIBUTED = Purpose(None)  # e.g. `desman keys check`: no role, no task
_purpose: ContextVar[Purpose | None] = ContextVar("purpose", default=None)
_watchers: ContextVar[tuple[Callable[[], None], ...]] = ContextVar("watchers", default=())


@contextmanager
def for_(purpose: Purpose) -> Iterator[None]:
    """Calls made inside are recorded as made for `purpose`."""
    token = _purpose.set(purpose)
    try:
        yield
    finally:
        _purpose.reset(token)


@contextmanager
def watching(watcher: Callable[[], None]) -> Iterator[None]:
    """`watcher` is called after each call recorded inside (it reads the log itself)."""
    token = _watchers.set((*_watchers.get(), watcher))
    try:
        yield
    finally:
        _watchers.reset(token)


def recorder(conn: sqlite3.Connection) -> Callable[[CallRecord], None]:
    """An `on_call` for `OpenAICompatible` that appends to model_calls."""

    def record(call: CallRecord) -> None:
        purpose = _purpose.get() or _UNATTRIBUTED
        conn.execute(
            "INSERT INTO model_calls (provider, model, key_id, prompt_hash, response_hash, "
            "prompt_tokens, completion_tokens, latency_ms, http_status, error_kind, error, "
            "limits_json, error_type, role, task_id, subtask_id, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                call.error_type,
                None if purpose.role is None else purpose.role.value,
                purpose.task_id,
                purpose.subtask_id,
                purpose.session_id,
            ),
        )
        for watcher in _watchers.get():
            watcher()

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
    limits_current: bool = True
    """The limits came with the key's latest call: no call has been made since without them."""


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
            "SELECT seq, error FROM model_calls WHERE provider = ? AND key_id IS ? "
            "ORDER BY seq DESC LIMIT 1",
            (r["provider"], r["key_id"]),
        ).fetchone()
        limits = conn.execute(
            "SELECT seq, limits_json FROM model_calls WHERE provider = ? AND key_id IS ? "
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
                limits_current=bool(limits) and limits["seq"] == last["seq"],
            )
        )
    return out


@dataclass(frozen=True)
class ModuleUse:
    """One module (provider, model and key) as it worked on a task: what `desman` shows live."""

    provider: str
    model: str
    key_id: str | None
    """None: a model without a key (a local one)."""
    calls: int
    prompt_tokens: int
    completion_tokens: int
    role: Role | None
    """Of its latest call: what it is doing now, or did last."""
    subtask_id: str | None
    error_type: str | None
    """Of its latest call; None if that call succeeded."""
    limits: dict[str, str]
    """The rate-limit headers of its latest call that had any."""


def modules(conn: sqlite3.Connection, task_id: str) -> list[ModuleUse]:
    """The modules that made calls for the task, the first to start first."""
    rows = conn.execute(
        """
        SELECT provider, model, key_id, COUNT(*) AS calls,
               SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens,
               MAX(seq) AS last_seq,
               MAX(CASE WHEN limits_json != '{}' THEN seq END) AS limits_seq
        FROM model_calls WHERE task_id = ?
        GROUP BY provider, model, key_id
        ORDER BY MIN(seq)
        """,
        (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        last = conn.execute(
            "SELECT role, subtask_id, error_type FROM model_calls WHERE seq = ?", (r["last_seq"],)
        ).fetchone()
        limits = None
        if r["limits_seq"] is not None:
            limits = conn.execute(
                "SELECT limits_json FROM model_calls WHERE seq = ?", (r["limits_seq"],)
            ).fetchone()
        out.append(
            ModuleUse(
                provider=r["provider"],
                model=r["model"],
                key_id=r["key_id"],
                calls=r["calls"],
                prompt_tokens=r["prompt_tokens"] or 0,
                completion_tokens=r["completion_tokens"] or 0,
                role=None if last["role"] is None else Role(last["role"]),
                subtask_id=last["subtask_id"],
                error_type=last["error_type"],
                limits=json.loads(limits["limits_json"]) if limits else {},
            )
        )
    return out


@dataclass(frozen=True)
class Limit:
    """One rate limit as a provider reported it: e.g. requests, 998 of 1000 left, reset in 2m."""

    name: str
    remaining: str | None
    limit: str | None
    reset: str | None

    @property
    def exhausted_for_good(self) -> bool:
        """A limit of 0: the account has no access at all (e.g. Mistral's free plan before
        activation), waiting will not help."""
        return self.limit is not None and _number(self.limit) == 0


def limits(headers: dict[str, str]) -> tuple[list[Limit], dict[str, str]]:
    """The `x-ratelimit-{limit,remaining,reset}-<name>` headers grouped by limit, in the order
    the provider sent them, and whatever else did not fit that pattern."""
    found: dict[str, dict[str, str]] = {}
    other: dict[str, str] = {}
    for header, value in headers.items():
        name = header.lower().removeprefix("x-ratelimit-")
        kind, _, limit = name.partition("-")
        if kind in ("limit", "remaining", "reset") and limit:
            found.setdefault(limit, {})[kind] = value
        else:
            other[header] = value
    return [
        Limit(name, parts.get("remaining"), parts.get("limit"), parts.get("reset"))
        for name, parts in found.items()
    ], other


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
