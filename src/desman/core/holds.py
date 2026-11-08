"""Operator-owned module exclusions and the task pauses owned by a stop (ADR 0008)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desman.core import db, operator
from desman.core.store import Clock, Store

Scope = Literal["all", "task", "session", "network"]


def _now() -> datetime:
    return datetime.now(UTC)


class Hold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(pattern=r"^(?:[0-9a-f]{12}|ollama|llamacpp)$")
    model: str = Field(pattern=r"^[a-z0-9_-]+/\S+$", max_length=200)
    scope: Scope = "all"
    scope_id: str = Field(default="", max_length=128)
    state: Literal["stopped", "removed"] = "stopped"
    at: str = ""
    paused_tasks: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_scope(self) -> Hold:
        if bool(self.scope_id) != (self.scope in {"task", "session"}):
            raise ValueError("invalid hold scope id")
        return self


def active(conn: sqlite3.Connection, scope: Scope = "all", scope_id: str = "") -> list[Hold]:
    return [
        Hold(
            **{k: r[k] for k in ("source", "model", "scope", "scope_id", "state", "at")},
            paused_tasks=json.loads(r["paused_tasks_json"]),
        )
        for r in conn.execute(
            "SELECT * FROM module_holds WHERE scope = 'all' OR (scope = ? AND scope_id = ?)",
            (scope, scope_id),
        )
    ]


def listed(conn: sqlite3.Connection) -> list[Hold]:
    return [
        Hold(
            **{k: r[k] for k in ("source", "model", "scope", "scope_id", "state", "at")},
            paused_tasks=json.loads(r["paused_tasks_json"]),
        )
        for r in conn.execute("SELECT * FROM module_holds ORDER BY at, source, model")
    ]


def blocked(
    conn: sqlite3.Connection, source: str, model: str, scope: Scope = "all", scope_id: str = ""
) -> bool:
    return any(h.source == source and h.model == model for h in active(conn, scope, scope_id))


def _pause_token(conn: sqlite3.Connection, identity: str) -> str | None:
    """The audit sequence owns a pause even when two operator actions share a millisecond."""
    row = conn.execute(
        "SELECT a.seq, a.action FROM audit_log a JOIN tasks t ON t.id=a.task_id "
        "WHERE t.id=? AND t.paused_at IS NOT NULL AND a.action IN ('task_paused','task_resumed') "
        "ORDER BY a.seq DESC LIMIT 1",
        (identity,),
    ).fetchone()
    return str(row["seq"]) if row is not None and row["action"] == "task_paused" else None


def _change(
    conn: sqlite3.Connection,
    hold: Hold,
    operation: str,
    tasks: Iterable[str] = (),
    *,
    clock: Clock = _now,
) -> None:
    with db.transaction(conn):
        if hold.scope in {"task", "session"}:
            table = "tasks" if hold.scope == "task" else "sessions"
            if (
                conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (hold.scope_id,)).fetchone()
                is None
            ):
                raise ValueError("unknown hold scope id")
        existing = next(
            (
                h
                for h in active(conn, hold.scope, hold.scope_id)
                if (h.source, h.model, h.scope, h.scope_id)
                == (hold.source, hold.model, hold.scope, hold.scope_id)
            ),
            None,
        )
        store = Store(conn, clock=clock)
        if operation in {"continue", "restore"}:
            expected = "stopped" if operation == "continue" else "removed"
            if existing is None or existing.state != expected:
                return
            conn.execute(
                "DELETE FROM module_holds WHERE source = ? AND model = ? AND scope = ? AND scope_id = ?",
                (hold.source, hold.model, hold.scope, hold.scope_id),
            )
            for identity, stamp in existing.paused_tasks.items():
                if _pause_token(conn, identity) == stamp:
                    store.resume_task(identity)
        else:
            state = "stopped" if operation == "stop" else "removed"
            if existing and existing.state == state:
                return
            paused = existing.paused_tasks.copy() if existing else {}
            if state == "stopped":
                for identity in tasks:
                    row = conn.execute(
                        "SELECT paused_at FROM tasks WHERE id = ?", (identity,)
                    ).fetchone()
                    if row and row[0] is None and store.pause_task(identity):
                        token = _pause_token(conn, identity)
                        assert token is not None
                        paused[identity] = token
            conn.execute(
                "INSERT INTO module_holds VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(source, model, scope, scope_id) "
                "DO UPDATE SET state=excluded.state, at=excluded.at, paused_tasks_json=excluded.paused_tasks_json",
                (
                    hold.source,
                    hold.model,
                    hold.scope,
                    hold.scope_id,
                    state,
                    db.to_iso(clock()),
                    json.dumps(paused),
                ),
            )
            # Replacing a stop releases only its own pauses; removal allows fallback work.
            if state == "removed" and existing:
                for identity, stamp in paused.items():
                    if _pause_token(conn, identity) == stamp:
                        store.resume_task(identity)
                conn.execute(
                    "UPDATE module_holds SET paused_tasks_json = '{}' WHERE source=? AND model=? AND scope=? AND scope_id=?",
                    (hold.source, hold.model, hold.scope, hold.scope_id),
                )
        operator.write(
            conn,
            f"operator_module_{operation}",
            {
                "source": hold.source,
                "model": hold.model,
                "scope": hold.scope,
                "scope_id": hold.scope_id,
            },
            session_id=hold.scope_id if hold.scope == "session" else None,
        )


def stop(
    conn: sqlite3.Connection, hold: Hold, tasks: Iterable[str] = (), *, clock: Clock = _now
) -> None:
    _change(conn, hold, "stop", tasks, clock=clock)


def continue_(conn: sqlite3.Connection, hold: Hold, *, clock: Clock = _now) -> None:
    _change(conn, hold, "continue", clock=clock)


def remove(conn: sqlite3.Connection, hold: Hold, *, clock: Clock = _now) -> None:
    _change(conn, hold, "remove", clock=clock)


def restore(conn: sqlite3.Connection, hold: Hold, *, clock: Clock = _now) -> None:
    _change(conn, hold, "restore", clock=clock)
