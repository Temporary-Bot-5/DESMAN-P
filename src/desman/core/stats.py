"""How well DESMAN works (ROADMAP 5.2): what `desman stats` shows, read from what is already
kept: subtasks, the audit log and the call log. Nothing new is recorded for it.

A window (`since`) counts what *happened* in it: tasks created, attempts started, verdicts,
failures and model calls made since then. The subtask counts are of the tasks created in the
window, as they are now. Everything is read in one snapshot.

The Improvement Analyst (5.4) and model choice (4.5) are meant to read the same numbers, so
they are plain data: `as_json()` is what `--json` prints.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from desman.core import db
from desman.core.retry import Strategy
from desman.core.states import SubtaskStatus, TaskStatus, derive_task_status

S = SubtaskStatus


@dataclass(frozen=True)
class SubtaskCounts:
    total: int = 0
    done: int = 0
    first_try: int = 0
    """Done after a single attempt, accepted by the reviewer (not by the operator)."""
    retried: int = 0
    """Took more than one attempt, however they ended."""
    escalated: int = 0
    """Reached needs_human at least once: the operator had to decide."""
    split: int = 0
    """Split into parts after failed attempts (the parts are counted as subtasks too)."""


@dataclass(frozen=True)
class StrategyUse:
    """Attempts made a given way (`retry.Strategy`) and how many of them the reviewer accepted."""

    strategy: str
    started: int
    accepted: int


@dataclass(frozen=True)
class RoleUse:
    role: str
    """`calls.Role`, or "other" for calls made for no role (e.g. `desman keys check`)."""
    calls: int
    failed: int
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ModelUse:
    provider: str
    model: str
    calls: int
    failed: int
    limited: int
    """Calls that hit a rate limit or quota."""
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    """Average over its calls."""


@dataclass(frozen=True)
class Stats:
    since: str | None
    """The window's start, None for all time."""
    tasks: dict[str, int] = field(default_factory=dict)
    """Tasks created in the window, by `TaskStatus` now."""
    subtasks: SubtaskCounts = field(default_factory=SubtaskCounts)
    strategies: list[StrategyUse] = field(default_factory=list)
    reviews: dict[str, int] = field(default_factory=dict)
    """Verdicts (accept, retry, escalate) and reviews that failed to give one ("failed"):
    those that cost an attempt and those to be tried again (`review_interrupted`)."""
    failures: dict[str, int] = field(default_factory=dict)
    """Failed attempts and reviews, by `FailureKind`."""
    roles: list[RoleUse] = field(default_factory=list)
    models: list[ModelUse] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def empty(self) -> bool:
        """Nothing happened in the window: no task created, no attempt, review, failure or
        model call (an older task may still have been worked on in it)."""
        return not (
            self.tasks
            or self.subtasks.total
            or self.strategies
            or self.reviews
            or self.failures
            or self.roles
        )


def collect(conn: sqlite3.Connection, since: datetime | None = None) -> Stats:
    start = None if since is None else db.to_iso(since)
    with db.read_snapshot(conn):
        return Stats(
            since=start,
            tasks=_tasks(conn, start),
            subtasks=_subtasks(conn, start),
            strategies=_strategies(conn, start),
            reviews=_reviews(conn, start),
            failures=_failures(conn, start),
            roles=_roles(conn, start),
            models=_models(conn, start),
        )


def _after(column: str, start: str | None) -> tuple[str, list[str]]:
    return (f"{column} >= ?", [start]) if start is not None else ("1", [])


def _tasks(conn: sqlite3.Connection, start: str | None) -> dict[str, int]:
    where, args = _after("t.created_at", start)
    statuses: dict[str, list[SubtaskStatus]] = {}
    for task_id, value in conn.execute(
        f"""SELECT t.id, s.status FROM tasks t LEFT JOIN subtasks s ON s.task_id = t.id
            WHERE {where}""",
        args,
    ):
        statuses.setdefault(task_id, [])
        if value is not None:
            statuses[task_id].append(SubtaskStatus(value))
    counts = Counter(derive_task_status(s).value for s in statuses.values())
    return {status.value: counts[status.value] for status in TaskStatus if counts[status.value]}


def _subtasks(conn: sqlite3.Connection, start: str | None) -> SubtaskCounts:
    where, args = _after("t.created_at", start)
    row = conn.execute(
        f"""
        WITH subs AS (
            SELECT s.id, s.status,
                   (SELECT COUNT(*) FROM audit_log a
                    WHERE a.subtask_id = s.id AND a.action = 'attempt_started') AS attempts,
                   EXISTS (SELECT 1 FROM audit_log a
                           WHERE a.subtask_id = s.id AND a.to_status = 'needs_human')
                       AS escalated,
                   EXISTS (SELECT 1 FROM audit_log a
                           WHERE a.subtask_id = s.id AND a.action = 'decomposed') AS split,
                   (SELECT a.actor FROM audit_log a
                    WHERE a.subtask_id = s.id AND a.to_status = 'done'
                    ORDER BY a.seq DESC LIMIT 1) AS done_by
            FROM subtasks s JOIN tasks t ON t.id = s.task_id
            WHERE {where}
        )
        SELECT COUNT(*) AS total,
               SUM(status = 'done') AS done,
               SUM(status = 'done' AND attempts = 1 AND done_by = 'reviewer') AS first_try,
               SUM(attempts > 1) AS retried,
               SUM(escalated) AS escalated,
               SUM(split) AS split
        FROM subs
        """,
        args,
    ).fetchone()
    return SubtaskCounts(
        **{key: row[key] or 0 for key in row.keys()}  # noqa: SIM118 (a Row, not a dict)
    )


def _strategies(conn: sqlite3.Connection, start: str | None) -> list[StrategyUse]:
    """Each attempt_started opens an attempt; the next verdict on that subtask closes it. An
    attempt with no verdict (failed, interrupted, still running) was not accepted."""
    where, args = _after("ts", start)
    started: Counter[str] = Counter()
    accepted: Counter[str] = Counter()
    open_: dict[str, str] = {}  # subtask id -> strategy of its attempt waiting for a verdict
    for subtask_id, action, details_json in conn.execute(
        f"""SELECT subtask_id, action, details_json FROM audit_log
            WHERE action IN ('attempt_started', 'verdict') AND {where} ORDER BY seq""",
        args,
    ):
        details = json.loads(details_json)
        if action == "attempt_started":
            strategy = str(details.get("strategy", Strategy.FIRST.value))
            started[strategy] += 1
            open_[subtask_id] = strategy
        elif subtask_id in open_:
            strategy = open_.pop(subtask_id)
            if details.get("verdict") == "accept":
                accepted[strategy] += 1
    order = [s.value for s in Strategy]
    names = sorted(started, key=lambda s: (order.index(s) if s in order else len(order), s))
    return [StrategyUse(s, started[s], accepted[s]) for s in names]


def _reviews(conn: sqlite3.Connection, start: str | None) -> dict[str, int]:
    where, args = _after("ts", start)
    counts: Counter[str] = Counter()
    for action, details_json in conn.execute(
        f"""SELECT action, details_json FROM audit_log
            WHERE action IN ('verdict', 'review_failed', 'review_interrupted') AND {where}""",
        args,
    ):
        if action != "verdict":
            counts["failed"] += 1
        else:
            counts[str(json.loads(details_json).get("verdict", "?"))] += 1
    return dict(sorted(counts.items(), key=lambda kv: _REVIEW_ORDER.get(kv[0], 9)))


_REVIEW_ORDER = {"accept": 0, "retry": 1, "escalate": 2, "failed": 3}


def _failures(conn: sqlite3.Connection, start: str | None) -> dict[str, int]:
    where, args = _after("ts", start)
    rows = conn.execute(
        f"""SELECT json_extract(details_json, '$.failure') AS kind, COUNT(*) AS n
            FROM audit_log WHERE kind IS NOT NULL AND {where}
            GROUP BY kind ORDER BY n DESC, kind""",
        args,
    ).fetchall()
    return {r["kind"]: r["n"] for r in rows}


def _roles(conn: sqlite3.Connection, start: str | None) -> list[RoleUse]:
    where, args = _after("at", start)
    rows = conn.execute(
        f"""
        SELECT COALESCE(role, 'other') AS role, COUNT(*) AS calls,
               SUM(error_kind IS NOT NULL) AS failed,
               SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens
        FROM model_calls WHERE {where}
        GROUP BY COALESCE(role, 'other') ORDER BY MIN(seq)
        """,
        args,
    ).fetchall()
    return [
        RoleUse(
            r["role"],
            r["calls"],
            r["failed"] or 0,
            r["prompt_tokens"] or 0,
            r["completion_tokens"] or 0,
        )
        for r in rows
    ]


def _models(conn: sqlite3.Connection, start: str | None) -> list[ModelUse]:
    where, args = _after("at", start)
    rows = conn.execute(
        f"""
        SELECT provider, model, COUNT(*) AS calls,
               SUM(error_kind IS NOT NULL) AS failed, SUM(error_kind = 'limit') AS limited,
               SUM(prompt_tokens) AS prompt_tokens, SUM(completion_tokens) AS completion_tokens,
               CAST(ROUND(AVG(latency_ms)) AS INTEGER) AS latency_ms
        FROM model_calls WHERE {where}
        GROUP BY provider, model ORDER BY calls DESC, MIN(seq)
        """,
        args,
    ).fetchall()
    return [
        ModelUse(
            r["provider"],
            r["model"],
            r["calls"],
            r["failed"] or 0,
            r["limited"] or 0,
            r["prompt_tokens"] or 0,
            r["completion_tokens"] or 0,
            r["latency_ms"] or 0,
        )
        for r in rows
    ]
