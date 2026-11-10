"""The user's profile (ROADMAP C.7, ADR 0003 §6): short notes about the person.

The interlocutor keeps them (how the person likes to be answered, their interests, requests
they make often) and sees them with every message, so they reach the model providers: they are
personal data. They exist only if the person agreed to personalisation (`Config.personalize`,
off by default); otherwise the interlocutor neither sees nor writes any. The person sees them in
`/profile` and changes or erases them there. How the person set the interlocutor to talk is
apart from them (`Config.behaviour`), and holds either way.

Since K.5 (ADR 0009) the notes live in the knowledge tree: one personal node,
`Personal/Profile` (`core.nodes.PROFILE_PATH`), one note per line. A task's planner may choose
it like any of the person's nodes; the person sees it in `desman nodes` too and may write it
there, as notes. The interlocutor edits the notes without a window (owner, 2026-11-10: the one
exception to ADR 0009 §6); the node itself only the person writes, moves or removes. A note's
number is its line: it holds within one look at the notes, so a change names the text it saw
(`was`) and finds that note again if the lines moved meanwhile. The first note makes the node,
erasing the last one removes it.

Unlike messages, notes are changed and deleted in place. Every change is written to the
operator's journal (`core.operator`) with the note's number and who made it, never its text, so
the journal does not keep what was erased; a changed or erased text also leaves the response
cache (`llm.cache`: the turn that wrote the note is a reply kept there). What the person said
in the conversation stays, like every message (`core.sessions`): erasing a note stops it from
going to the models with every message, it does not rewrite the past.

Notes change how the interlocutor talks, nothing else: no permission, rule or confirmation
reads them.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from desman.core import db, nodes, operator
from desman.core.store import Clock

NOTE_CHARS = nodes.NOTE_CHARS
"""The longest note."""
MAX_NOTES = nodes.MAX_NOTES
"""The most notes the profile keeps: the interlocutor sees them all with every message."""


class Author(StrEnum):
    """Who changed the notes (the profile's node keeps who wrote it last)."""

    PERSON = "person"
    INTERLOCUTOR = "interlocutor"


class Note(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    """Its line in the profile's node, from 1: valid within one look at the notes."""
    text: str
    created_at: str
    updated_at: str
    """The profile's node's: notes are lines of one text."""


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
        row = self._row()
        if row is None:
            return []
        return [
            Note(id=i, text=text, created_at=row["created_at"], updated_at=row["updated_at"])
            for i, text in enumerate(_parsed(row["text"]), start=1)
        ]

    def get(self, note_id: int) -> Note | None:
        return next((n for n in self.notes() if n.id == note_id), None)

    def add(self, text: str, author: Author, *, session_id: str | None = None) -> Note:
        """A new note; the same text again (in any case) gives the note there is. ValueError
        for a bad text, `ProfileFull` when there is no room."""
        text = normalized(text)
        with db.transaction(self.conn):
            texts = self._texts()
            for i, kept in enumerate(texts, start=1):
                if kept.casefold() == text.casefold():
                    note = self.get(i)
                    assert note is not None
                    return note
            if len(texts) >= MAX_NOTES:
                raise ProfileFull(f"the profile has {MAX_NOTES} notes already")
            self._write([*texts, text], author)
            self._journal("profile_note_added", len(texts) + 1, author, session_id)
        note = self.get(len(texts) + 1)
        assert note is not None
        return note

    def change(
        self,
        note_id: int,
        text: str,
        author: Author,
        *,
        was: str | None = None,
        session_id: str | None = None,
    ) -> Note | None:
        """Give a note a new text; None if there is no such note. `was`: the text the caller
        saw at `note_id`; when the lines moved since, the note with that text is changed (none
        → None). ValueError for a bad text."""
        text = normalized(text)
        with db.transaction(self.conn):
            texts = self._texts()
            i = _index(texts, note_id, was)
            if i is None:
                return None
            if texts[i] != text:
                old = texts[i]
                texts[i] = text
                self._write(texts, author)
                nodes.uncache(self.conn, [old])
                self._journal("profile_note_changed", i + 1, author, session_id)
        return self.get(i + 1)

    def forget(
        self,
        note_id: int,
        author: Author,
        *,
        was: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Erase a note; False if there was no such note. `was` as in `change`."""
        with db.transaction(self.conn):
            texts = self._texts()
            i = _index(texts, note_id, was)
            if i is None:
                return False
            gone = texts.pop(i)
            self._write(texts, author)
            nodes.uncache(self.conn, [gone])
            self._journal("profile_note_forgotten", i + 1, author, session_id)
        return True

    def clear(self, *, session_id: str | None = None) -> int:
        """Erase every note (only the person does that); how many there were."""
        with db.transaction(self.conn):
            gone = self._texts()
            if gone:
                self._write([], Author.PERSON)
                nodes.uncache(self.conn, gone)
                details = {"notes": len(gone)}
                operator.write(self.conn, "profile_cleared", details, session_id=session_id)
        return len(gone)

    def _row(self) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM personal_nodes WHERE path = ?", (nodes.PROFILE_PATH,)
        ).fetchone()
        return row

    def _texts(self) -> list[str]:
        row = self._row()
        return [] if row is None else _parsed(row["text"])

    def _write(self, texts: list[str], author: Author) -> None:
        """The profile's node holds `texts`: made by the first note, removed with the last."""
        row = self._row()
        if not texts:
            if row is not None:
                self.conn.execute("DELETE FROM personal_nodes WHERE id = ?", (row["id"],))
            return
        now = db.to_iso(self.clock())
        text = nodes.profile_text(texts)
        if row is None:
            self.conn.execute(
                "INSERT INTO personal_nodes "
                "(path, about, text, author, untrusted, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 0, ?, ?)",
                (nodes.PROFILE_PATH, nodes.PROFILE_ABOUT, text, author.value, now, now),
            )
        else:
            self.conn.execute(
                "UPDATE personal_nodes SET text = ?, author = ?, untrusted = 0, updated_at = ? "
                "WHERE id = ?",
                (text, author.value, now, row["id"]),
            )

    def _journal(self, action: str, note_id: int, author: Author, session_id: str | None) -> None:
        details = {"note": note_id, "by": author.value}
        operator.write(self.conn, action, details, session_id=session_id)


def _parsed(text: str) -> list[str]:
    """The notes of the node's text. A text that does not hold as notes (written before the
    rules, by hand in the database) gives what fits rather than nothing."""
    try:
        return nodes.profile_notes(text)
    except nodes.NodeError:
        lines = [" ".join(line.split()).removeprefix("- ").strip() for line in text.splitlines()]
        return [line[:NOTE_CHARS] for line in lines if line][:MAX_NOTES]


def _index(texts: list[str], note_id: int, was: str | None) -> int | None:
    """Where the note is: line `note_id`, unless `was` says it held another text then."""
    if 1 <= note_id <= len(texts) and (was is None or texts[note_id - 1] == was):
        return note_id - 1
    if was is None:
        return None
    return next((i for i, t in enumerate(texts) if t.casefold() == was.casefold()), None)
