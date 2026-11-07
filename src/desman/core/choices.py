"""The person's replaceable choices for one task or chat; changes are journaled atomically."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from desman.config import Config, KeyFingerprint, ModelChoices, ModelLabel
from desman.core import db, operator

Scope = Literal["task", "session"]

ROLES_BY_SCOPE: dict[str, tuple[str, ...]] = {
    # The planner plans a task before it exists, so a task's own choice could never reach it.
    "task": ("reviewer", "decomposer", "text", "code", "analysis"),
    "session": ("interlocutor",),
}
"""The roles whose calls read a choice of that scope: anything else would be kept and never used."""


class Choice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelLabel | None = None
    key: KeyFingerprint | None = None


def validate(
    conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str | None = None
) -> None:
    if scope not in ("task", "session"):
        raise ValueError("unknown module choice scope")
    if role is not None and role not in ModelChoices.model_fields:
        raise ValueError("unknown module role")
    if role is not None and role not in ROLES_BY_SCOPE[scope]:
        raise ValueError(f"a {scope} choice is not read by the {role}")
    table = "tasks" if scope == "task" else "sessions"
    if conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (scope_id,)).fetchone() is None:
        raise ValueError("unknown module choice scope id")


def get(conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str) -> Choice:
    row = conn.execute(
        "SELECT model, key FROM module_choices WHERE scope = ? AND scope_id = ? AND role = ?",
        (scope, scope_id, role),
    ).fetchone()
    return Choice(**dict(row)) if row else Choice()


def list_choices(conn: sqlite3.Connection, scope: Scope, scope_id: str) -> dict[str, Choice]:
    validate(conn, scope, scope_id)
    return {
        r["role"]: Choice(model=r["model"], key=r["key"])
        for r in conn.execute(
            "SELECT role, model, key FROM module_choices WHERE scope = ? AND scope_id = ? "
            "ORDER BY role",
            (scope, scope_id),
        )
    }


def set_choice(
    conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str, choice: Choice
) -> tuple[Choice, Choice]:
    with db.transaction(conn):
        return write_set(conn, scope, scope_id, role, choice)


def write_set(
    conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str, choice: Choice
) -> tuple[Choice, Choice]:
    """Set within the caller's transaction (also used with a chat's first message)."""
    validate(conn, scope, scope_id, role)
    old = get(conn, scope, scope_id, role)
    if old == choice:
        return old, choice
    if choice.model is None and choice.key is None:
        conn.execute(
            "DELETE FROM module_choices WHERE scope = ? AND scope_id = ? AND role = ?",
            (scope, scope_id, role),
        )
    else:
        conn.execute(
            "INSERT INTO module_choices (scope, scope_id, role, model, key, set_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (scope, scope_id, role) DO UPDATE SET "
            "model = excluded.model, key = excluded.key, set_at = excluded.set_at",
            (scope, scope_id, role, choice.model, choice.key, db.to_iso(datetime.now(UTC))),
        )
    operator.write(
        conn,
        "operator_module_choice",
        {
            "scope": scope,
            "scope_id": scope_id,
            "role": role,
            "old": old.model_dump(),
            "new": choice.model_dump(),
        },
        session_id=scope_id if scope == "session" else None,
    )

    return old, choice


def clear(conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str | None = None) -> None:
    with db.transaction(conn):
        write_clear(conn, scope, scope_id, role)


def write_clear(
    conn: sqlite3.Connection, scope: Scope, scope_id: str, role: str | None = None
) -> None:
    """Clear inside the caller's transaction (also used when erasing a temporary chat)."""
    validate(conn, scope, scope_id, role)
    for name in [role] if role else list_choices(conn, scope, scope_id):
        write_set(conn, scope, scope_id, name, Choice())


def effective(config: Config, role: str, choice: Choice) -> Config:
    """Inherit each unset part independently; never mutate the caller's config."""
    tuned = config.tuning.model_copy(deep=True)
    if choice.model is not None:
        setattr(tuned.models, role, choice.model)
    if choice.key is not None:
        setattr(tuned.keys, role, choice.key)
    return config.model_copy(update={"tuning": tuned})
