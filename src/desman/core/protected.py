"""Protected files (ROADMAP 5.3): what decides what agents may do and whether their work is good.

The files are listed in `protected.toml` (shipped in the package, next to `policy.toml`) with the
SHA-256 of each: the tool policy and the code that enforces it, the reviewer, the sandbox and
the workspace guard, the human-only edges of the state machine, this check and the test
harness. Work runs only while every one of them is as the person approved it: a change nobody
approved (an edit, a pull, later an applied proposal, 5.6) blocks every run until the person
looks at it (`desman protected show --diff`) and approves it (`desman protected approve`).

What was approved is kept in the database (`protected_approvals`, `protected_files`), with
each file's content so a change can be shown as a diff; both tables are append-only and every
approval is a full snapshot. The first check of a database with no approval takes the release
as approved, but only if every file matches the hashes the release shipped; otherwise it is
blocked like any other change. The manifest itself is part of every snapshot: a change to the
list (a file taken off it) or to its hashes needs approval too. A manifest that cannot be read
blocks everything: it is never treated as "nothing is protected".

Paths in the manifest are relative to the package (`[package]`, kept as `desman/...`) or to the
source checkout (`[checkout]`: the test harness, which an installed package does not have and
then does not check). Line endings are not part of a file's hash, so a checkout on Windows
matches the release.

Agents cannot change any of this themselves: they reach files only through a task's workspace,
and the approval is a command of the person's (in the chat, only after their "yes" in a window
built by code).
"""

from __future__ import annotations

import difflib
import hashlib
import re
import sqlite3
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from desman.core import db, operator

PACKAGE: Final = Path(__file__).resolve().parent.parent
MANIFEST: Final = "protected.toml"
PREFIX: Final = "desman/"
"""How the package's files are named in snapshots and messages."""


class ProtectedError(ValueError):
    """The manifest cannot be used: nothing can be checked, so nothing runs."""


class Blocked(Exception):
    """Protected files changed since the person approved them."""

    def __init__(self, changes: list[Change]) -> None:
        self.changes = changes
        super().__init__(
            "protected files changed without approval: " + ", ".join(c.path for c in changes)
        )


@dataclass(frozen=True)
class Layout:
    """Where the protected files are: the package, and the source checkout if there is one."""

    package: Path
    checkout: Path | None

    @property
    def manifest(self) -> Path:
        return self.package / MANIFEST

    def file(self, path: str) -> Path:
        """The file a snapshot's `path` names."""
        if path.startswith(PREFIX):
            return self.package / path.removeprefix(PREFIX)
        assert self.checkout is not None, path
        return self.checkout / path


def here() -> Layout:
    """This installation: a source checkout is the package's `src/..` with a pyproject.toml."""
    root = PACKAGE.parent.parent
    checkout = PACKAGE.parent.name == "src" and (root / "pyproject.toml").is_file()
    return Layout(PACKAGE, root if checkout else None)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package: dict[str, str] = Field(min_length=1)
    checkout: dict[str, str] = Field(default_factory=dict)

    @field_validator("package", "checkout")
    @classmethod
    def _plain_paths(cls, files: dict[str, str]) -> dict[str, str]:
        for path, sha in files.items():
            parts = PurePosixPath(path).parts
            if not parts or path.startswith("/") or ".." in parts or "\\" in path:
                raise ValueError(f"not a relative path: {path!r}")
            if path == MANIFEST:
                raise ValueError(f"{MANIFEST} is always protected and cannot list its own hash")
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError(f"not a SHA-256 for {path!r}: {sha!r}")
        return files

    def expected(self, layout: Layout) -> dict[str, str]:
        """The shipped hash of each file this installation has to check, by snapshot path."""
        found = {PREFIX + path: sha for path, sha in self.package.items()}
        if layout.checkout is not None:
            found.update(self.checkout)
        return found


def load_manifest(layout: Layout) -> Manifest:
    try:
        data = tomllib.loads(layout.manifest.read_text(encoding="utf-8"))
        return Manifest.model_validate(data)
    except FileNotFoundError as e:
        raise ProtectedError(f"{layout.manifest} is missing") from e
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValidationError) as e:
        raise ProtectedError(f"{layout.manifest} cannot be used: {e}") from e


def content(path: Path) -> bytes | None:
    """A file as it is hashed (line endings as LF); None if it is not there."""
    try:
        return path.read_bytes().replace(b"\r\n", b"\n")
    except (FileNotFoundError, IsADirectoryError):
        return None


def sha256(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def snapshot(layout: Layout, manifest: Manifest) -> dict[str, bytes | None]:
    """Every protected file now, by path: the manifest's list and the manifest itself."""
    paths = [PREFIX + MANIFEST, *manifest.expected(layout)]
    return {path: content(layout.file(path)) for path in paths}


class Kind(StrEnum):
    CHANGED = "changed"
    MISSING = "missing"
    ADDED = "added"
    """Put on the list since the approval."""
    REMOVED = "removed"
    """Taken off the list since the approval."""


class Change(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    kind: Kind
    approved: str | None
    """The approved hash (None: missing then, or not on the list)."""
    current: str | None


def _diff(approved: dict[str, str | None], now: dict[str, str | None]) -> list[Change]:
    changes = []
    for path in sorted(approved.keys() | now.keys()):
        before, after = approved.get(path), now.get(path)
        if path not in approved:
            changes.append(Change(path=path, kind=Kind.ADDED, approved=None, current=after))
        elif path not in now:
            changes.append(Change(path=path, kind=Kind.REMOVED, approved=before, current=None))
        elif before != after:
            kind = Kind.MISSING if after is None else Kind.CHANGED
            changes.append(Change(path=path, kind=kind, approved=before, current=after))
    return changes


def _latest(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT max(id) AS id FROM protected_approvals").fetchone()
    return None if row["id"] is None else int(row["id"])


def _approved(conn: sqlite3.Connection, approval: int, layout: Layout) -> dict[str, str | None]:
    rows = conn.execute(
        "SELECT path, sha256 FROM protected_files WHERE approval_id = ?", (approval,)
    ).fetchall()
    # An installation without a checkout does not check the test harness, whoever approved it.
    return {
        r["path"]: r["sha256"]
        for r in rows
        if layout.checkout is not None or r["path"].startswith(PREFIX)
    }


def _write(
    conn: sqlite3.Connection,
    files: dict[str, bytes | None],
    source: str,
    note: str | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO protected_approvals (source, note) VALUES (?, ?)", (source, note)
    )
    approval = cur.lastrowid
    assert approval is not None
    conn.executemany(
        "INSERT INTO protected_files (approval_id, path, sha256, content) VALUES (?, ?, ?, ?)",
        [(approval, path, sha256(data), data) for path, data in files.items()],
    )
    return approval


def _state(
    conn: sqlite3.Connection, layout: Layout
) -> tuple[dict[str, bytes | None], list[Change], bool]:
    """The files now, what changed, and whether this is the release on a database with no
    approval yet (then the changes are against the shipped hashes)."""
    manifest = load_manifest(layout)
    files = snapshot(layout, manifest)
    now = {path: sha256(data) for path, data in files.items()}
    approval = _latest(conn)
    if approval is not None:
        return files, _diff(_approved(conn, approval, layout), now), False
    shipped: dict[str, str | None] = {PREFIX + MANIFEST: now[PREFIX + MANIFEST]}
    shipped.update(manifest.expected(layout))
    found = _diff(shipped, now)
    return files, found, not found


def changes(conn: sqlite3.Connection, layout: Layout | None = None) -> list[Change]:
    """What changed since the last approval; empty when work may run.

    A database with no approval yet takes the release as approved (recorded as such) when every
    file matches the shipped hashes; otherwise the changes are against those hashes."""
    layout = here() if layout is None else layout
    files, found, release = _state(conn, layout)
    if release:
        with db.transaction(conn):
            if _latest(conn) is None:  # another process may have recorded it meanwhile
                _write(conn, files, "release", None)
    return found


def require(conn: sqlite3.Connection, layout: Layout | None = None) -> None:
    """Raise `Blocked` unless every protected file is as approved."""
    found = changes(conn, layout)
    if found:
        raise Blocked(found)


def approve(
    conn: sqlite3.Connection,
    note: str | None = None,
    *,
    expected: list[Change] | None = None,
    session_id: str | None = None,
    layout: Layout | None = None,
) -> list[Change]:
    """The person approves the protected files as they are now; the changes approved (empty:
    there was nothing to approve, and nothing is written). `expected`: the changes the person
    was shown; if the files changed since, nothing is approved (`ProtectedError`). Recorded in
    the operator's journal in the same transaction."""
    layout = here() if layout is None else layout
    with db.transaction(conn):
        files, found, _ = _state(conn, layout)
        if expected is not None and found != expected:
            raise ProtectedError("the protected files changed since they were shown: look again")
        if not found:
            return []
        approval = _write(conn, files, "operator", note)
        operator.write(
            conn,
            "operator_protected_approved",
            {
                "approval": approval,
                "note": note,
                "changes": [c.model_dump(mode="json") for c in found],
            },
            session_id=session_id,
        )
    return found


@dataclass(frozen=True)
class Approval:
    id: int
    ts: str
    source: str
    note: str | None


def last_approval(conn: sqlite3.Connection) -> Approval | None:
    row = conn.execute(
        "SELECT id, ts, source, note FROM protected_approvals ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return Approval(id=row["id"], ts=row["ts"], source=row["source"], note=row["note"])


def diff(conn: sqlite3.Connection, change: Change, layout: Layout | None = None) -> str | None:
    """A unified diff from the approved content to the file now; None when the approved
    content is not known (no approval yet: compare with the release in git)."""
    layout = here() if layout is None else layout
    approval = _latest(conn)
    if approval is None:
        return None
    row = conn.execute(
        "SELECT content FROM protected_files WHERE approval_id = ? AND path = ?",
        (approval, change.path),
    ).fetchone()
    before = b"" if row is None or row["content"] is None else bytes(row["content"])
    after = b""
    if change.kind is not Kind.REMOVED:
        after = content(layout.file(change.path)) or b""
    lines = difflib.unified_diff(
        before.decode("utf-8", "replace").splitlines(keepends=True),
        after.decode("utf-8", "replace").splitlines(keepends=True),
        fromfile=f"{change.path} (approved)",
        tofile=f"{change.path} (now)",
    )
    return "".join(lines)
