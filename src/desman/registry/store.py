"""The registry's database (ADR 0004; ROADMAP G.4): its own SQLite with its own migrations.

The same rules as the node's database (`core/db.py`, whose helpers it uses): append-only
migrations, WAL, `BEGIN IMMEDIATE` for writes, a schema check on open.

- `nodes`: every node that registered — its public key, whether it takes part (`active`) or
  `left`, what it lends, when it was seen last. A node is never deleted: its history is the
  log's.
- `records`: the state the registry hands out (`protocol.TRUST`) — assignments and revocations
  signed by the root. Append-only. Whether a key is revoked is read from them, as a node reads
  them (`sign.Trust`): the latest record about the key counts.
- `nonces`: what `net.request` saw, until its time window closes.
- `registry_log`: the registry's journal (ADR 0004, decision 12) — registrations, changes of
  what is lent, departures, assignments, revocations, refused requests. Keys by fingerprint;
  nothing in it is secret. Append-only.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.core import db
from desman.net import sign
from desman.net.protocol import Capability

DIR_ENV: Final = "DESMAN_REGISTRY_DIR"
DB_NAME: Final = "registry.db"
ROOT_KEY: Final = "root.key"
OPERATOR_KEY: Final = "operator.key"
FRESH: Final = "`desman-registry serve`"

_NOW_SQL: Final = "(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"

MIGRATIONS: Final[tuple[str, ...]] = (
    # 1 — nodes, the records handed out, nonces seen, the journal
    f"""
    CREATE TABLE nodes (
        fingerprint        TEXT PRIMARY KEY CHECK (length(fingerprint) = 64),
        public_key         TEXT NOT NULL,
        status             TEXT NOT NULL CHECK (status IN ('active', 'left')),
        capabilities_json  TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(capabilities_json)),
        registered_at      TEXT NOT NULL,
        last_seen          TEXT NOT NULL
    ) STRICT;
    CREATE TRIGGER nodes_key_fixed BEFORE UPDATE OF fingerprint, public_key ON nodes
        WHEN NEW.fingerprint IS NOT OLD.fingerprint OR NEW.public_key IS NOT OLD.public_key
        BEGIN SELECT RAISE(ABORT, 'a node keeps its key'); END;
    CREATE TRIGGER nodes_no_delete BEFORE DELETE ON nodes
        BEGIN SELECT RAISE(ABORT, 'nodes are never deleted'); END;

    CREATE TABLE records (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        id           TEXT NOT NULL UNIQUE,
        kind         TEXT NOT NULL CHECK (kind IN ('assignment', 'revocation')),
        subject      TEXT NOT NULL CHECK (length(subject) = 64),
        signer       TEXT NOT NULL,
        issued_at    TEXT NOT NULL,
        expires_at   TEXT NOT NULL,
        signed_json  TEXT NOT NULL CHECK (json_valid(signed_json)),
        created_at   TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;
    CREATE INDEX records_subject ON records (subject, issued_at);
    CREATE TRIGGER records_no_update BEFORE UPDATE ON records
        BEGIN SELECT RAISE(ABORT, 'records are append-only'); END;
    CREATE TRIGGER records_no_delete BEFORE DELETE ON records
        BEGIN SELECT RAISE(ABORT, 'records are append-only'); END;

    CREATE TABLE nonces (
        id          TEXT PRIMARY KEY,
        expires_at  TEXT NOT NULL
    ) STRICT;
    CREATE INDEX nonces_expiry ON nonces (expires_at);

    CREATE TABLE registry_log (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT NOT NULL DEFAULT {_NOW_SQL},
        action        TEXT NOT NULL,
        node          TEXT,
        details_json  TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json))
    ) STRICT;
    CREATE TRIGGER registry_log_no_update BEFORE UPDATE ON registry_log
        BEGIN SELECT RAISE(ABORT, 'registry_log is append-only'); END;
    CREATE TRIGGER registry_log_no_delete BEFORE DELETE ON registry_log
        BEGIN SELECT RAISE(ABORT, 'registry_log is append-only'); END;
    """,
)


def default_dir() -> Path:
    """$DESMAN_REGISTRY_DIR or ~/.local/share/desman-registry: the database and the keys."""
    configured = os.environ.get(DIR_ENV)
    return Path(configured) if configured else Path.home() / ".local" / "share" / "desman-registry"


def open_registry(path: Path) -> sqlite3.Connection:
    """Connect, bring the schema up to date and check it."""
    conn = db.connect(path)
    try:
        db.migrate(conn, MIGRATIONS)
        db.verify_schema(conn, MIGRATIONS, fresh=FRESH)
    except BaseException:
        conn.close()
        raise
    return conn


@contextmanager
def connection(path: Path) -> Iterator[sqlite3.Connection]:
    """One connection for one request or command (the server is threaded: never shared)."""
    conn = db.connect(path)
    try:
        yield conn
    finally:
        conn.close()


# ── Journal ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LogEntry:
    seq: int
    ts: str
    action: str
    node: str | None
    details: dict[str, Any]


def log(
    conn: sqlite3.Connection,
    action: str,
    node: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Inside the caller's transaction, if it has one: the entry and its change commit together."""
    conn.execute(
        "INSERT INTO registry_log (action, node, details_json) VALUES (?, ?, ?)",
        (action, node, json.dumps(details or {}, ensure_ascii=False, sort_keys=True)),
    )


def entries(conn: sqlite3.Connection, limit: int | None = None) -> list[LogEntry]:
    """Oldest first; with `limit`, the latest ones."""
    if limit is None:
        rows = conn.execute("SELECT * FROM registry_log ORDER BY seq").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM (SELECT * FROM registry_log ORDER BY seq DESC LIMIT ?) ORDER BY seq",
            (limit,),
        ).fetchall()
    return [
        LogEntry(r["seq"], r["ts"], r["action"], r["node"], json.loads(r["details_json"]))
        for r in rows
    ]


# ── Nonces ───────────────────────────────────────────────────────────────────


class NonceStore:
    """`sign.Seen` in the database: shared by the server's threads, outlives a restart."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(self, id: str, expires_at: datetime, now: datetime) -> bool:
        with db.transaction(self._conn):
            self._conn.execute("DELETE FROM nonces WHERE expires_at <= ?", (db.to_iso(now),))
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO nonces (id, expires_at) VALUES (?, ?)",
                (id, db.to_iso(expires_at)),
            )
            return cursor.rowcount == 1


# ── Records ──────────────────────────────────────────────────────────────────


def _subject(signed: sign.Signed) -> str:
    record = signed.record
    if record.kind is sign.RecordKind.ASSIGNMENT:
        return sign.Assignment.model_validate(record.body).fingerprint
    if record.kind is sign.RecordKind.REVOCATION:
        return sign.Revocation.model_validate(record.body).key
    raise ValueError(f"the registry keeps assignments and revocations, not {record.kind}")


def all_records(conn: sqlite3.Connection) -> list[sign.Signed]:
    rows = conn.execute("SELECT signed_json FROM records ORDER BY issued_at, seq").fetchall()
    return [sign.Signed.model_validate_json(r["signed_json"]) for r in rows]


def handed_out(conn: sqlite3.Connection, now: datetime) -> list[sign.Signed]:
    """Assignments still in their term and every revocation (one is honoured past its term)."""
    rows = conn.execute(
        "SELECT signed_json FROM records WHERE kind = 'revocation' OR expires_at > ?"
        " ORDER BY issued_at, seq",
        (db.to_iso(now),),
    ).fetchall()
    return [sign.Signed.model_validate_json(r["signed_json"]) for r in rows]


def trust(conn: sqlite3.Connection, root: Ed25519PublicKey, now: datetime) -> sign.Trust:
    """What a node would believe after receiving every record kept here (an assignment past its
    term is refused and counts for nothing, as on a node)."""
    believed = sign.Trust(root, clock=lambda: now)
    for signed in all_records(conn):
        try:
            believed.receive(signed.model_dump(mode="json"))
        except sign.Refused:
            continue
    return believed


def add_record(conn: sqlite3.Connection, signed: sign.Signed) -> str:
    """Keep a record (inside the caller's transaction); returns its subject's fingerprint."""
    record = signed.record
    subject = _subject(signed)
    conn.execute(
        "INSERT INTO records (id, kind, subject, signer, issued_at, expires_at, signed_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record.id,
            record.kind.value,
            subject,
            record.signer,
            db.to_iso(record.issued_at),
            db.to_iso(record.expires_at),
            json.dumps(signed.model_dump(mode="json"), sort_keys=True),
        ),
    )
    return subject


def is_revoked(conn: sqlite3.Connection, fingerprint: str) -> bool:
    """The latest record about this key revokes it (an assignment after it lifts that, as on
    a node)."""
    row = conn.execute(
        "SELECT kind FROM records WHERE subject = ? ORDER BY issued_at DESC, seq DESC LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row is not None and row["kind"] == sign.RecordKind.REVOCATION.value


# ── Nodes ────────────────────────────────────────────────────────────────────

Status = Literal["active", "left"]


@dataclass(frozen=True)
class Node:
    fingerprint: str
    public_key: str
    status: Status
    capabilities: list[Capability]
    registered_at: str
    last_seen: str
    revoked: bool

    def as_json(self) -> dict[str, Any]:
        return {
            "node": self.fingerprint,
            "status": "revoked" if self.revoked else self.status,
            "capabilities": [c.model_dump(mode="json") for c in self.capabilities],
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
        }


def _node(conn: sqlite3.Connection, row: sqlite3.Row) -> Node:
    return Node(
        fingerprint=row["fingerprint"],
        public_key=row["public_key"],
        status=row["status"],
        capabilities=[Capability.model_validate(c) for c in json.loads(row["capabilities_json"])],
        registered_at=row["registered_at"],
        last_seen=row["last_seen"],
        revoked=is_revoked(conn, row["fingerprint"]),
    )


def node(conn: sqlite3.Connection, fingerprint: str) -> Node | None:
    row = conn.execute("SELECT * FROM nodes WHERE fingerprint = ?", (fingerprint,)).fetchone()
    return None if row is None else _node(conn, row)


def nodes(conn: sqlite3.Connection) -> list[Node]:
    rows = conn.execute("SELECT * FROM nodes ORDER BY registered_at, fingerprint").fetchall()
    return [_node(conn, r) for r in rows]


def capabilities_json(capabilities: list[Capability]) -> str:
    return json.dumps([c.model_dump(mode="json") for c in capabilities], sort_keys=True)
