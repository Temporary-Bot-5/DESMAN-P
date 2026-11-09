"""The person's own nodes of the knowledge tree (ADR 0009 §6, ROADMAP K.4).

Release nodes come with DESMAN and are built from the repository (`desman.knowledge`); these are
the person's: "Projects/X/Information", "Personal/Style". A model choosing its context sees their
paths and descriptions beside the release ones, and reads a text only when it names its node.

They are personal data, like the profile's notes (`core.profile`): they exist for the models
only while the person agreed to personalisation (`Config.personalize`); with it off they stay,
unused, until the person removes them. The person writes them (`desman nodes`, the page); an
agent (the interlocutor, or a task's retelling) only proposes a change, applied after the person
confirms it in a window rendered by code. A temporary chat never proposes one, and neither its
interlocutor nor the planning of its tasks sees them (an executor's `read_node` reads the tree
as the person's settings say: it does not know which chat its task came from).

Nodes are changed and erased in place. Every change is written to the operator's journal by
the node's number and who made it, never its path or text, so the journal does not keep what was
erased; a changed or erased text also leaves the response cache. A text that came from a task's
result keeps the `untrusted` mark: data, never instructions, whoever reads it.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict

from desman.core import db, operator
from desman.core.store import Clock

TEXT_CHARS: Final = 6000
"""The longest text of a node (~1.5k tokens; the `personal_nodes.text` CHECK says the same)."""
ABOUT_CHARS: Final = 100
"""A node's description: one line for choosing."""
PATH_CHARS: Final = 200
PART_CHARS: Final = 60
"""The longest section or name in a path."""
MAX_DEPTH: Final = 5
MAX_NODES: Final = 100
"""The most nodes the person keeps: their table of contents is in every prompt that chooses."""
RELEASE_SECTIONS: Final = ("Commands", "DESMAN")
"""Top sections that come with DESMAN: a personal node never goes under them."""


class Author(StrEnum):
    """Who wrote a node's text. Keep in sync with the `personal_nodes.author` CHECK (a test
    enforces it)."""

    PERSON = "person"
    INTERLOCUTOR = "interlocutor"
    TASK = "task"
    """Proposed by a task's retelling, from its result: always untrusted."""


class PersonalNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    path: str
    about: str
    text: str
    author: Author
    untrusted: bool
    created_at: str
    updated_at: str


class NodeError(ValueError):
    """A path, description or text that cannot be kept (said in English: core's messages are
    not translated; the CLI words its own)."""


class NodeExists(NodeError):
    pass


class NodesFull(NodeError):
    pass


def normalized_path(path: str) -> str:
    """A path as kept: parts without surrounding blanks, joined by "/". NodeError when it is
    empty, too long or deep, or under a section that comes with DESMAN."""
    parts = [" ".join(part.split()) for part in path.strip().strip("/").split("/")]
    if not path.strip().strip("/") or any(not part for part in parts):
        raise NodeError("a path is sections and a name joined by /, none of them empty")
    if len(parts) > MAX_DEPTH:
        raise NodeError(f"a path has at most {MAX_DEPTH} parts")
    if any(len(part) > PART_CHARS for part in parts):
        raise NodeError(f"a section or name is at most {PART_CHARS} characters")
    if any(not ch.isprintable() for ch in "".join(parts)):
        raise NodeError("a path is plain characters")
    if parts[0].casefold() in {s.casefold() for s in RELEASE_SECTIONS}:
        raise NodeError(f"{parts[0]}/ comes with DESMAN: put your node elsewhere")
    joined = "/".join(parts)
    if len(joined) > PATH_CHARS:
        raise NodeError(f"a path is at most {PATH_CHARS} characters")
    return joined


def normalized_about(about: str) -> str:
    about = " ".join(about.split())
    if not about:
        raise NodeError("a description cannot be empty")
    if len(about) > ABOUT_CHARS:
        raise NodeError(f"a description is at most {ABOUT_CHARS} characters")
    return about


def normalized_text(text: str) -> str:
    text = text.strip("\n").rstrip()
    if not text.strip():
        raise NodeError("a text cannot be empty")
    if len(text) > TEXT_CHARS:
        raise NodeError(f"a text is at most {TEXT_CHARS} characters, not {len(text)}")
    return text


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PersonalNodes:
    def __init__(self, conn: sqlite3.Connection, clock: Clock = _utcnow) -> None:
        self.conn = conn
        self.clock = clock

    def nodes(self) -> list[PersonalNode]:
        rows = self.conn.execute("SELECT * FROM personal_nodes ORDER BY path, id").fetchall()
        return [_node(r) for r in rows]

    def get(self, path: str) -> PersonalNode | None:
        """The node at `path`, written in any case; None if there is none."""
        try:
            wanted = normalized_path(path).casefold()
        except NodeError:
            return None
        return next((n for n in self.nodes() if n.path.casefold() == wanted), None)

    def add(
        self,
        path: str,
        about: str,
        text: str,
        author: Author,
        *,
        untrusted: bool = False,
        session_id: str | None = None,
    ) -> PersonalNode:
        """A new node. NodeError for a bad path, description or text; `NodeExists` when the
        path is taken; `NodesFull` when there is no room."""
        path, about, text = normalized_path(path), normalized_about(about), normalized_text(text)
        untrusted = untrusted or author is Author.TASK
        with db.transaction(self.conn):
            if self._id(path) is not None:
                raise NodeExists(f"there is a node {path} already")
            count = self.conn.execute("SELECT COUNT(*) FROM personal_nodes").fetchone()[0]
            if count >= MAX_NODES:
                raise NodesFull(f"there are {MAX_NODES} nodes already")
            now = db.to_iso(self.clock())
            cursor = self.conn.execute(
                "INSERT INTO personal_nodes "
                "(path, about, text, author, untrusted, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, about, text, author.value, int(untrusted), now, now),
            )
            assert cursor.lastrowid is not None
            self._journal("knowledge_node_added", cursor.lastrowid, author, session_id)
        node = self._by_id(cursor.lastrowid)
        assert node is not None
        return node

    def change(
        self,
        path: str,
        author: Author,
        *,
        new_path: str | None = None,
        about: str | None = None,
        text: str | None = None,
        untrusted: bool = False,
        session_id: str | None = None,
    ) -> PersonalNode | None:
        """Move, describe or rewrite the node at `path` (what is given); None if there is no
        such node. A new text from a task keeps the untrusted mark; a text the person wrote
        clears it; anything else keeps it, and a task's change of any part sets it."""
        new_path = None if new_path is None else normalized_path(new_path)
        about = None if about is None else normalized_about(about)
        text = None if text is None else normalized_text(text)
        with db.transaction(self.conn):
            was = self.get(path)
            if was is None:
                return None
            moved = new_path is not None and new_path.casefold() != was.path.casefold()
            if moved and new_path is not None and self._id(new_path) is not None:
                raise NodeExists(f"there is a node {new_path} already")
            now_path = new_path if new_path is not None else was.path
            now_about = about if about is not None else was.about
            now_text = text if text is not None else was.text
            if untrusted or author is Author.TASK:
                mark = True  # a task's path or description is no more trusted than its text
            elif author is Author.PERSON and text is not None and text != was.text:
                mark = False  # the person wrote it
            else:
                mark = was.untrusted  # an agent's rewrite never launders a task's text
            if (now_path, now_about, now_text, mark) == (
                was.path,
                was.about,
                was.text,
                was.untrusted,
            ):
                return was
            self.conn.execute(
                "UPDATE personal_nodes SET path = ?, about = ?, text = ?, author = ?, "
                "untrusted = ?, updated_at = ? WHERE id = ?",
                (
                    now_path,
                    now_about,
                    now_text,
                    author.value,
                    int(mark),
                    db.to_iso(self.clock()),
                    was.id,
                ),
            )
            self._uncache(
                [
                    old
                    for old, new in (
                        (was.text, now_text),
                        (was.about, now_about),
                        (was.path, now_path),
                    )
                    if old != new
                ]
            )
            self._journal("knowledge_node_changed", was.id, author, session_id)
        return self._by_id(was.id)

    def remove(self, path: str, author: Author, *, session_id: str | None = None) -> bool:
        """Erase the node at `path`; False if there was none."""
        with db.transaction(self.conn):
            was = self.get(path)
            if was is None:
                return False
            self.conn.execute("DELETE FROM personal_nodes WHERE id = ?", (was.id,))
            self._uncache([was.text, was.about, was.path])
            self._journal("knowledge_node_removed", was.id, author, session_id)
        return True

    def _id(self, path: str) -> int | None:
        wanted = path.casefold()
        for row in self.conn.execute("SELECT id, path FROM personal_nodes").fetchall():
            if row["path"].casefold() == wanted:
                return int(row["id"])
        return None

    def _by_id(self, node_id: int) -> PersonalNode | None:
        row = self.conn.execute("SELECT * FROM personal_nodes WHERE id = ?", (node_id,)).fetchone()
        return None if row is None else _node(row)

    def _uncache(self, texts: list[str]) -> None:
        """Drop cached replies that hold any of `texts`, as written or escaped in JSON (a very
        short one, like a path "X", is left: it would drop nearly every reply)."""
        for text in texts:
            if len(text) < 4:
                continue
            for form in {text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1]}:
                self.conn.execute("DELETE FROM response_cache WHERE instr(text, ?) > 0", (form,))

    def _journal(self, action: str, node_id: int, author: Author, session_id: str | None) -> None:
        details = {"node": node_id, "by": author.value}
        operator.write(self.conn, action, details, session_id=session_id)


def _node(row: sqlite3.Row) -> PersonalNode:
    return PersonalNode.model_validate({**dict(row), "untrusted": bool(row["untrusted"])})
