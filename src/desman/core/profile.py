"""The user's profile (ROADMAP C.7, ADR 0003 §6): short notes about the person.

The interlocutor keeps them (how the person likes to be answered, their interests, requests
they make often) and sees them with every message, so they reach the model providers: they are
personal data. The person sees them in `/profile`, changes or erases them there, or turns the
profile off (`Config.profile`), and then the interlocutor neither sees nor writes any.

Unlike messages, notes are changed and deleted in place: erased means gone. Every change is
written to the operator's journal (`core.operator`) with the note's number and who made it,
never its text, so the journal does not keep what was erased.

Notes change how the interlocutor talks, nothing else: no permission, rule or confirmation
reads them.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from desman.core import db, operator
from desman.core.store import Clock

NOTE_CHARS = 200
"""The longest note (the `profile_notes.text` CHECK says the same)."""
MAX_NOTES = 20
"""The most notes the profile keeps: the interlocutor sees them all with every message."""


class Author(StrEnum):
    """Who wrote a note's text. Keep in sync with the `profile_notes.author` CHECK (a test
    enforces it)."""

    PERSON = "person"
    INTERLOCUTOR = "interlocutor"


class Note(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    text: str
    author: Author
    created_at: str
    updated_at: str


class ProfileFull(ValueError):
    """The profile has `MAX_NOTES` already."""


def normalized(text: str) -> str:
    """A note as kept: one line, without surrounding blanks. ValueError if empty or too long."""
    text = " ".join(text.split())
    if not text:
        raise ValueError("a note cannot be empty")
    if len(text) > NOTE_CHARS:
        raise ValueError(f"a note is at most {NOTE_CHARS} characters, not {len(text)}")
    return text


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Profile:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = _utcnow) -> None:
        self.conn = conn
        self.clock = clock

    def notes(self) -> list[Note]:
        rows = self.conn.execute("SELECT * FROM profile_notes ORDER BY id").fetchall()
        return [Note.model_validate(dict(r)) for r in rows]

    def get(self, note_id: int) -> Note | None:
        row = self.conn.execute("SELECT * FROM profile_notes WHERE id = ?", (note_id,)).fetchone()
        return None if row is None else Note.model_validate(dict(row))

    def add(self, text: str, author: Author, *, session_id: str | None = None) -> Note:
        """A new note; the same text again (in any case) gives the note there is. ValueError
        for a bad text, `ProfileFull` when there is no room."""
        text = normalized(text)
        with db.transaction(self.conn):
            for row in self.conn.execute("SELECT * FROM profile_notes").fetchall():
                if row["text"].casefold() == text.casefold():
                    return Note.model_validate(dict(row))
            count = self.conn.execute("SELECT COUNT(*) FROM profile_notes").fetchone()[0]
            if count >= MAX_NOTES:
                raise ProfileFull(f"the profile has {MAX_NOTES} notes already")
            now = db.to_iso(self.clock())
            cursor = self.conn.execute(
                "INSERT INTO profile_notes (text, author, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (text, author.value, now, now),
            )
            assert cursor.lastrowid is not None
            self._journal("profile_note_added", cursor.lastrowid, author, session_id)
        note = self.get(cursor.lastrowid)
        assert note is not None
        return note

    def change(
        self, note_id: int, text: str, author: Author, *, session_id: str | None = None
    ) -> Note | None:
        """Give a note a new text (its author is who wrote that); None if there is no such
        note. ValueError for a bad text."""
        text = normalized(text)
        with db.transaction(self.conn):
            now = db.to_iso(self.clock())
            changed = self.conn.execute(
                "UPDATE profile_notes SET text = ?, author = ?, updated_at = ? "
                "WHERE id = ? AND text != ?",
                (text, author.value, now, note_id, text),
            ).rowcount
            if changed:
                self._journal("profile_note_changed", note_id, author, session_id)
        return self.get(note_id)

    def forget(self, note_id: int, author: Author, *, session_id: str | None = None) -> bool:
        """Erase a note; False if there was no such note."""
        with db.transaction(self.conn):
            gone = self.conn.execute("DELETE FROM profile_notes WHERE id = ?", (note_id,)).rowcount
            if gone:
                self._journal("profile_note_forgotten", note_id, author, session_id)
        return bool(gone)

    def clear(self, *, session_id: str | None = None) -> int:
        """Erase every note (only the person does that); how many there were."""
        with db.transaction(self.conn):
            gone = self.conn.execute("DELETE FROM profile_notes").rowcount
            if gone:
                operator.write(self.conn, "profile_cleared", {"notes": gone}, session_id=session_id)
        return gone

    def _journal(self, action: str, note_id: int, author: Author, session_id: str | None) -> None:
        details = {"note": note_id, "by": author.value}
        operator.write(self.conn, action, details, session_id=session_id)
