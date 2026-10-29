"""Chat sessions and their messages (ROADMAP C.2, ADR 0003 §8).

A session is one conversation in bare `desman`. Its messages are append-only (triggers refuse
UPDATE and DELETE): a summary that later stands in for old messages (C.6) is a message of its
own, and the originals stay in the database.

A session exists from its first message on: `/new` only starts a fresh one, so leaving the chat
without writing anything leaves no empty session behind.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from desman.core import db
from desman.core.store import Clock


class Role(StrEnum):
    """Who said it. Keep in sync with the `messages.role` CHECK (a test enforces it)."""

    USER = "user"
    INTERLOCUTOR = "interlocutor"  # the module that talks with the person (C.3)
    DESMAN = "desman"  # the program itself: events, results of commands


class Kind(StrEnum):
    """Keep in sync with the `messages.kind` CHECK (a test enforces it)."""

    MESSAGE = "message"
    SUMMARY = "summary"  # stands in for older messages in the model's context (C.6)
    EVENT = "event"  # something that happened: a task started or finished, a command ran


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    session_id: str
    at: str
    role: Role
    kind: Kind
    text: str
    task_id: str | None = None


class SessionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created_at: str
    last_at: str
    messages: int
    opening: str
    """The first thing the person wrote: how they recognise the session in a list."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Sessions:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = _utcnow) -> None:
        self.conn = conn
        self.clock = clock

    def add(
        self,
        session_id: str | None,
        role: Role,
        text: str,
        kind: Kind = Kind.MESSAGE,
        task_id: str | None = None,
    ) -> Message:
        """Append a message; `session_id` None starts a new session with it."""
        now = db.to_iso(self.clock())
        with db.transaction(self.conn):
            if session_id is None:
                session_id = uuid.uuid4().hex
                self.conn.execute(
                    "INSERT INTO sessions (id, created_at) VALUES (?, ?)", (session_id, now)
                )
            cursor = self.conn.execute(
                "INSERT INTO messages (session_id, at, role, kind, text, task_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, now, role.value, kind.value, text, task_id),
            )
        assert cursor.lastrowid is not None
        return Message(
            seq=cursor.lastrowid,
            session_id=session_id,
            at=now,
            role=role,
            kind=kind,
            text=text,
            task_id=task_id,
        )

    def messages(self, session_id: str, last: int | None = None) -> list[Message]:
        """The session's messages in order; only the `last` few if given."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
            (session_id, -1 if last is None else last),
        ).fetchall()
        return [Message.model_validate(dict(r)) for r in reversed(rows)]

    def recent(self, limit: int = 20, offset: int = 0) -> list[SessionInfo]:
        """Sessions, the one written to last first; `offset` skips that many."""
        rows = self.conn.execute(
            """
            SELECT s.id, s.created_at, MAX(m.at) AS last_at, COUNT(m.seq) AS messages,
                   COALESCE((SELECT u.text FROM messages u
                             WHERE u.session_id = s.id AND u.role = 'user'
                             ORDER BY u.seq LIMIT 1), '') AS opening
            FROM sessions s JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY MAX(m.seq) DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return [SessionInfo.model_validate(dict(r)) for r in rows]

    def latest(self) -> SessionInfo | None:
        """The session the chat continues: the one written to last."""
        recent = self.recent(limit=1)
        return recent[0] if recent else None
