"""Chat sessions and their messages (ROADMAP C.2, ADR 0003 §8).

A session is one conversation in bare `desman`. Its messages are append-only (triggers refuse
UPDATE and DELETE): a summary that later stands in for old messages (C.6) is a message of its
own, and the originals stay in the database.

What the interlocutor sees of a session is its `History`: the latest summary, if there is one,
and the messages after those it covers.

A session exists from its first message on: `/new` only starts a fresh one, so leaving the chat
without writing anything leaves no empty session behind. The person may name a session and
delete it: a deleted session leaves the lists and is never continued, but its messages stay.

A temporary session (C.11, `desman -t`) is the exception: it is never listed, and its messages
are erased (`erase`) when the chat that holds it ends, or by the next chat once that one is
certainly gone (`sweep`, after a crash). The database refuses to erase any other session.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from desman.core import db
from desman.core.owner import current_owner, owner_is_gone
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
    covers: int | None = None
    """A summary's: the last message (by `seq`) it stands in for."""


class History(BaseModel):
    """What the interlocutor sees of a session (C.6): the latest summary and what came after
    the messages it covers (other summaries never: the latest one includes them)."""

    model_config = ConfigDict(frozen=True)

    summary: Message | None
    messages: list[Message]
    earlier: int = 0
    """Messages after the summary (or from the start) left out of `messages` by its limit."""


class SessionInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created_at: str
    last_at: str
    messages: int
    opening: str
    """The first thing the person wrote: how they recognise the session in a list."""
    pinned: bool = False
    title: str | None = None
    """The name the person gave it (`/rename`), shown instead of the opening."""

    @property
    def name(self) -> str:
        return self.title or self.opening


TITLE_CHARS = 80
"""The longest name a session keeps."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Sessions:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = _utcnow) -> None:
        self.conn = conn
        self.clock = clock

    def terminal(self) -> str:
        """Return the one permanent console, creating it atomically on first use."""
        with db.transaction(self.conn):
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at, terminal, title) "
                "VALUES (?, ?, 1, 'General_Terminal')",
                (uuid.uuid4().hex, db.to_iso(self.clock())),
            )
            row = self.conn.execute("SELECT id FROM sessions WHERE terminal = 1").fetchone()
        return str(row["id"])

    def is_terminal(self, session_id: str) -> bool:
        row = self.conn.execute(
            "SELECT terminal FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row is not None and bool(row["terminal"])

    def add(
        self,
        session_id: str | None,
        role: Role,
        text: str,
        kind: Kind = Kind.MESSAGE,
        task_id: str | None = None,
        covers: int | None = None,
        temporary: bool = False,
    ) -> Message:
        """Append a message; `session_id` None starts a new session with it, `temporary` one if
        asked (this process holds it). A summary says which messages it `covers`."""
        if (covers is not None) != (kind is Kind.SUMMARY):
            raise ValueError("a summary, and only a summary, covers earlier messages")
        if kind is Kind.SUMMARY and session_id is not None and self.is_terminal(session_id):
            raise ValueError("the terminal cannot have summaries")
        now = db.to_iso(self.clock())
        with db.transaction(self.conn):
            if session_id is None:
                session_id = uuid.uuid4().hex
                if temporary:  # the columns are newer than the table (migration 24)
                    self.conn.execute(
                        "INSERT INTO sessions (id, created_at, temporary, owner) "
                        "VALUES (?, ?, 1, ?)",
                        (session_id, now, current_owner()),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO sessions (id, created_at) VALUES (?, ?)", (session_id, now)
                    )
            columns: dict[str, str | int | None] = {
                "session_id": session_id,
                "at": now,
                "role": role.value,
                "kind": kind.value,
                "text": text,
                "task_id": task_id,
            }
            if covers is not None:  # the column is newer than the other ones (migration 14)
                columns["covers"] = covers
            cursor = self.conn.execute(
                f"INSERT INTO messages ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' * len(columns))})",
                tuple(columns.values()),
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
            covers=covers,
        )

    def messages(self, session_id: str, last: int | None = None) -> list[Message]:
        """The session's messages in order; only the `last` few if given."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
            (session_id, -1 if last is None else last),
        ).fetchall()
        return [Message.model_validate(dict(r)) for r in reversed(rows)]

    def history(self, session_id: str, last: int | None = None) -> History:
        """The latest summary of the session and the messages after those it covers (only the
        `last` few of them if given), summaries left out."""
        if self.is_terminal(session_id):
            return History(summary=None, messages=[])
        row = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND kind = 'summary' "
            "ORDER BY seq DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        summary = None if row is None else Message.model_validate(dict(row))
        after = 0 if summary is None else summary.covers
        after_summary = "FROM messages WHERE session_id = ? AND seq > ? AND kind != 'summary'"
        rows = self.conn.execute(
            f"SELECT * {after_summary} ORDER BY seq DESC LIMIT ?",
            (session_id, after, -1 if last is None else last),
        ).fetchall()
        earlier = 0
        if last is not None and len(rows) == last:
            total = self.conn.execute(f"SELECT COUNT(*) {after_summary}", (session_id, after))
            earlier = total.fetchone()[0] - last
        return History(
            summary=summary,
            messages=[Message.model_validate(dict(r)) for r in reversed(rows)],
            earlier=earlier,
        )

    def changed_tasks(self, session_id: str) -> list[str]:
        """The session's tasks with audit entries newer than the last message about them: what
        happened while the chat was not watching (`desman resume`, decisions in the terminal).
        The messages are the marker, so a task is retold once for each change."""
        rows = self.conn.execute(
            """
            SELECT t.id FROM tasks t
            WHERE t.session_id = ?
              AND (SELECT MAX(a.ts) FROM audit_log a WHERE a.task_id = t.id)
                  > COALESCE((SELECT MAX(m.at) FROM messages m
                              WHERE m.session_id = t.session_id AND m.task_id = t.id), '')
            ORDER BY t.created_at, t.id
            """,
            (session_id,),
        ).fetchall()
        return [r["id"] for r in rows]

    def rename(self, session_id: str, title: str | None) -> None:
        """Name a session; None (or blank) gives it back its opening as the name."""
        title = " ".join((title or "").split())[:TITLE_CHARS] or None
        with db.transaction(self.conn):
            self.conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND terminal = 0", (title, session_id)
            )

    def delete(self, session_id: str) -> None:
        """Take a session out of the lists and the chat. Its messages are kept."""
        now = db.to_iso(self.clock())
        with db.transaction(self.conn):
            self.conn.execute(
                "UPDATE sessions SET deleted_at = ? WHERE id = ? AND terminal = 0 AND deleted_at IS NULL",
                (now, session_id),
            )

    def pin(self, session_id: str, pinned: bool) -> None:
        """Pin a kept session; the database refuses pinning a temporary one."""
        with db.transaction(self.conn):
            self.conn.execute(
                "UPDATE sessions SET pinned_at = ? WHERE id = ? AND terminal = 0",
                (db.to_iso(self.clock()) if pinned else None, session_id),
            )

    def delete_all(self) -> int:
        """Soft-delete listed, unpinned sessions, retaining their messages."""
        with db.transaction(self.conn):
            return self.conn.execute(
                "UPDATE sessions SET deleted_at = ? WHERE deleted_at IS NULL "
                "AND temporary = 0 AND terminal = 0 AND pinned_at IS NULL "
                "AND EXISTS (SELECT 1 FROM messages WHERE session_id = sessions.id)",
                (db.to_iso(self.clock()),),
            ).rowcount

    def erase(self, session_id: str) -> None:
        """Erase the messages of a temporary session (the database refuses any other)."""
        with db.transaction(self.conn):
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    def sweep(self) -> int:
        """Erase what temporary sessions left behind when their chat could not (it crashed, its
        terminal was closed): only of a chat certainly gone. How many were erased."""
        rows = self.conn.execute(
            "SELECT DISTINCT s.id, s.owner FROM sessions s JOIN messages m ON m.session_id = s.id "
            "WHERE s.temporary = 1"
        ).fetchall()
        gone = [r["id"] for r in rows if owner_is_gone(r["owner"])]
        for session_id in gone:
            self.erase(session_id)
        return len(gone)

    def get(self, session_id: str) -> SessionInfo | None:
        """A session that is not deleted, or None."""
        found = self._infos("AND s.id = ?", (session_id,), limit=1, offset=0)
        return found[0] if found else None

    def recent(self, limit: int = 20, offset: int = 0) -> list[SessionInfo]:
        """Sessions not deleted, the one written to last first; `offset` skips that many."""
        return self._infos("", (), limit=limit, offset=offset)

    def _infos(
        self, where: str, params: tuple[str, ...], *, limit: int, offset: int
    ) -> list[SessionInfo]:
        rows = self.conn.execute(
            f"""
            SELECT s.id, s.created_at, s.title, s.pinned_at IS NOT NULL AS pinned, MAX(m.at) AS last_at, COUNT(m.seq) AS messages,
                   COALESCE((SELECT u.text FROM messages u
                             WHERE u.session_id = s.id AND u.role = 'user'
                             ORDER BY u.seq LIMIT 1), '') AS opening
            FROM sessions s JOIN messages m ON m.session_id = s.id
            WHERE s.deleted_at IS NULL AND s.temporary = 0 AND s.terminal = 0 {where}
            GROUP BY s.id
            ORDER BY (s.pinned_at IS NOT NULL) DESC, MAX(m.seq) DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [SessionInfo.model_validate(dict(r)) for r in rows]

    def latest(self) -> SessionInfo | None:
        """The session the chat continues: the one written to last."""
        row = self.conn.execute(
            "SELECT s.id FROM sessions s JOIN messages m ON m.session_id = s.id "
            "WHERE s.deleted_at IS NULL AND s.temporary = 0 AND s.terminal = 0 "
            "GROUP BY s.id ORDER BY MAX(m.seq) DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self.get(row["id"])
