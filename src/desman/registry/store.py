"""The registry's database (ADR 0004; ROADMAP G.4): its own SQLite with its own migrations.

G.7b adds independent proposal checks and leases (migration 6). G.7c adds final operator
decisions and append-only root-signed releases (migration 7); rejection reasons use `reason`.

The same rules as the node's database (`core/db.py`, whose helpers it uses): append-only
migrations, WAL, `BEGIN IMMEDIATE` for writes, a schema check on open.

- `nodes`: every node that registered — its public key, whether it takes part (`active`) or
  `left`, what it lends, when it was seen last. A node is never deleted: its history is the
  log's.
- `records`: the state the registry hands out (`protocol.TRUST`) — assignments and revocations
  signed by the root. Append-only. Whether a key is revoked is read from them, as a node reads
  them (`sign.Trust`): the latest record about the key counts.
- `trust_lists` (G.6c): the trust lists the person signed with the root (`desman-registry trust
  sign`); the newest still in its term is handed out after the records it names. Append-only.
- `nonces`: what `net.request` saw, until its time window closes.
- `nodes.owner`: whose node it is, as the operator says (`desman-registry owner`); unset, a
  node is its own owner. A result is rechecked by a node of another owner.
- `jobs`, `leases`, `results` (G.6): work an operator signed, lent to one node at a time (see
  `registry/work.py`), and what nodes handed in (append-only).
- `registry_log`: the registry's journal (ADR 0004, decision 12) — registrations, changes of
  what is lent, departures, assignments, revocations, refused requests. Keys by fingerprint;
  nothing in it is secret. Append-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.core import db
from desman.net import protocol, sign
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
    # 2 — work (G.6): owners of nodes, jobs, their leases, what nodes handed in
    f"""
    ALTER TABLE nodes ADD COLUMN owner TEXT
        CHECK (owner IS NULL OR length(owner) BETWEEN 1 AND 64);

    CREATE TABLE jobs (
        id           TEXT PRIMARY KEY CHECK (length(id) = 32),
        signer       TEXT NOT NULL CHECK (length(signer) = 64),
        signed_json  TEXT NOT NULL CHECK (json_valid(signed_json)),
        expires_at   TEXT NOT NULL,
        status       TEXT NOT NULL CHECK (status IN
                         ('execute', 'recheck', 'accepted', 'operator', 'dropped', 'expired')),
        result       TEXT,
        reason       TEXT,
        round        INTEGER NOT NULL DEFAULT 1 CHECK (round >= 1),
        created_at   TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;
    CREATE INDEX jobs_status ON jobs (status, created_at);
    CREATE TRIGGER jobs_record_fixed BEFORE UPDATE OF id, signer, signed_json, expires_at ON jobs
        BEGIN SELECT RAISE(ABORT, 'a job keeps its record'); END;
    CREATE TRIGGER jobs_no_delete BEFORE DELETE ON jobs
        BEGIN SELECT RAISE(ABORT, 'jobs are never deleted'); END;

    CREATE TABLE leases (
        token       TEXT PRIMARY KEY CHECK (length(token) = 32),
        job         TEXT NOT NULL REFERENCES jobs (id),
        node        TEXT NOT NULL REFERENCES nodes (fingerprint),
        stage       TEXT NOT NULL CHECK (stage IN ('execute', 'recheck')),
        round       INTEGER NOT NULL,
        granted_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        ended       TEXT CHECK (ended IN ('handed_in', 'expired', 'withdrawn')),
        ended_at    TEXT
    ) STRICT;
    CREATE UNIQUE INDEX leases_one_held ON leases (job) WHERE ended IS NULL;
    CREATE INDEX leases_held ON leases (expires_at) WHERE ended IS NULL;
    CREATE TRIGGER leases_fixed BEFORE UPDATE OF token, job, node, stage, round, granted_at
        ON leases
        BEGIN SELECT RAISE(ABORT, 'a lease keeps who holds what'); END;
    CREATE TRIGGER leases_end_once BEFORE UPDATE OF ended ON leases WHEN OLD.ended IS NOT NULL
        BEGIN SELECT RAISE(ABORT, 'a lease ends once'); END;
    CREATE TRIGGER leases_no_delete BEFORE DELETE ON leases
        BEGIN SELECT RAISE(ABORT, 'leases are never deleted'); END;

    CREATE TABLE results (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        id           TEXT NOT NULL UNIQUE CHECK (length(id) = 32),
        job          TEXT NOT NULL REFERENCES jobs (id),
        node         TEXT NOT NULL REFERENCES nodes (fingerprint),
        lease        TEXT NOT NULL UNIQUE REFERENCES leases (token),
        kind         TEXT NOT NULL CHECK (kind IN ('result', 'verdict')),
        round        INTEGER NOT NULL,
        outcome      TEXT NOT NULL CHECK (outcome IN ('done', 'failed', 'accept', 'reject')),
        signed_json  TEXT NOT NULL CHECK (json_valid(signed_json)),
        received_at  TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;
    CREATE INDEX results_job ON results (job, seq);
    CREATE TRIGGER results_no_update BEFORE UPDATE ON results
        BEGIN SELECT RAISE(ABORT, 'results are append-only'); END;
    CREATE TRIGGER results_no_delete BEFORE DELETE ON results
        BEGIN SELECT RAISE(ABORT, 'results are append-only'); END;
    """,
    # 3 — whose node did the work, as it was when the result came: a later `owner` cannot make
    # a node of the same owner its reviewer
    """
    ALTER TABLE results ADD COLUMN owner TEXT;
    """,
    # 4 — trust lists signed by the root (G.6c)
    """
    CREATE TABLE trust_lists (
        seq          INTEGER PRIMARY KEY AUTOINCREMENT,
        id           TEXT NOT NULL UNIQUE,
        issued_at    TEXT NOT NULL,
        expires_at   TEXT NOT NULL,
        signed_json  TEXT NOT NULL CHECK (json_valid(signed_json))
    ) STRICT;
    CREATE TRIGGER trust_lists_no_update BEFORE UPDATE ON trust_lists
        BEGIN SELECT RAISE(ABORT, 'trust lists are append-only'); END;
    CREATE TRIGGER trust_lists_no_delete BEFORE DELETE ON trust_lists
        BEGIN SELECT RAISE(ABORT, 'trust lists are append-only'); END;
    """,
    # 5 — improvement proposals submitted by a node after its person's confirmation (G.7a)
    f"""
    CREATE TABLE proposals (
        id              TEXT PRIMARY KEY CHECK (length(id) = 32),
        node            TEXT NOT NULL REFERENCES nodes (fingerprint),
        proposal        INTEGER NOT NULL CHECK (proposal >= 1),
        evaluation      INTEGER NOT NULL CHECK (evaluation >= 1),
        submission_json TEXT NOT NULL CHECK (json_valid(submission_json)),
        status          TEXT NOT NULL DEFAULT 'submitted'
                        CHECK (status IN ('submitted', 'reviewing', 'reviewed', 'accepted', 'rejected')),
        created_at      TEXT NOT NULL DEFAULT {_NOW_SQL},
        UNIQUE (node, proposal, evaluation)
    ) STRICT;
    CREATE INDEX proposals_status ON proposals (status, created_at);
    CREATE TRIGGER proposals_fixed BEFORE UPDATE OF id, node, proposal, evaluation,
        submission_json, created_at ON proposals
        BEGIN SELECT RAISE(ABORT, 'a proposal submission keeps its contents'); END;
    CREATE TRIGGER proposals_no_delete BEFORE DELETE ON proposals
        BEGIN SELECT RAISE(ABORT, 'proposals are never deleted'); END;
    """,
    # 6 — independent proposal checks and their leases (G.7b).
    """
    ALTER TABLE proposals ADD COLUMN reason TEXT;
    CREATE TABLE proposal_leases (
        token TEXT PRIMARY KEY CHECK (length(token) = 32 AND token NOT GLOB '*[^0-9a-f]*'),
        proposal TEXT NOT NULL REFERENCES proposals(id),
        node TEXT NOT NULL REFERENCES nodes(fingerprint),
        granted_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        ended TEXT CHECK (ended IN ('handed_in', 'expired', 'withdrawn')),
        ended_at TEXT
    ) STRICT;
    CREATE INDEX proposal_leases_active ON proposal_leases(node, ended);
    CREATE TABLE proposal_checks (
        id TEXT PRIMARY KEY,
        proposal TEXT NOT NULL REFERENCES proposals(id),
        node TEXT NOT NULL REFERENCES nodes(fingerprint),
        owner TEXT NOT NULL,
        lease TEXT NOT NULL UNIQUE REFERENCES proposal_leases(token),
        outcome TEXT NOT NULL CHECK (outcome IN ('refused', 'failed', 'unable', 'checked')),
        signed_json TEXT NOT NULL CHECK (json_valid(signed_json)),
        received_at TEXT NOT NULL
    ) STRICT;
    CREATE TRIGGER proposal_checks_no_update BEFORE UPDATE ON proposal_checks
        BEGIN SELECT RAISE(ABORT, 'proposal checks are append-only'); END;
    CREATE TRIGGER proposal_checks_no_delete BEFORE DELETE ON proposal_checks
        BEGIN SELECT RAISE(ABORT, 'proposal checks are append-only'); END;
    """,
    # 7 — operator decisions and root-signed releases (G.7c).
    """
    CREATE TABLE releases (
        number INTEGER PRIMARY KEY CHECK (number >= 1),
        id TEXT NOT NULL UNIQUE,
        proposal TEXT NOT NULL UNIQUE REFERENCES proposals(id),
        commit_sha TEXT NOT NULL,
        parent TEXT NOT NULL,
        branch TEXT NOT NULL,
        signed_json TEXT NOT NULL CHECK (json_valid(signed_json)),
        created_at TEXT NOT NULL
    ) STRICT;
    CREATE TRIGGER releases_no_update BEFORE UPDATE ON releases
        BEGIN SELECT RAISE(ABORT, 'releases are append-only'); END;
    CREATE TRIGGER releases_no_delete BEFORE DELETE ON releases
        BEGIN SELECT RAISE(ABORT, 'releases are append-only'); END;
    CREATE TRIGGER proposals_decision_final BEFORE UPDATE OF status ON proposals
        WHEN OLD.status IN ('accepted', 'rejected') AND NEW.status IS NOT OLD.status
        BEGIN SELECT RAISE(ABORT, 'proposal decisions are final'); END;
    """,
    # 8 — reports and one-shot shared memory (G.9).
    """
    CREATE TABLE reports (
        id TEXT PRIMARY KEY,
        node TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('bug', 'memory')),
        report_json TEXT NOT NULL CHECK (json_valid(report_json)),
        created_at TEXT NOT NULL
    ) STRICT;
    CREATE INDEX reports_kind_created ON reports(kind, created_at);
    CREATE INDEX reports_node_created ON reports(node, created_at);
    CREATE TRIGGER reports_no_update BEFORE UPDATE ON reports
        BEGIN SELECT RAISE(ABORT, 'reports are append-only'); END;
    CREATE TRIGGER reports_no_delete BEFORE DELETE ON reports
        BEGIN SELECT RAISE(ABORT, 'reports are append-only'); END;
    """,
    # 9 — human correspondence (G.10a). Events survive withdrawn requests.
    """
    CREATE TABLE messenger_names (
        node TEXT PRIMARY KEY REFERENCES nodes(fingerprint),
        name TEXT NOT NULL UNIQUE, set_at TEXT NOT NULL
    ) STRICT;
    CREATE TABLE contact_requests (
        from_node TEXT NOT NULL REFERENCES nodes(fingerprint),
        to_node TEXT NOT NULL REFERENCES nodes(fingerprint), created_at TEXT NOT NULL,
        PRIMARY KEY(from_node, to_node)
    ) STRICT;
    CREATE TABLE contact_events (
        from_node TEXT NOT NULL REFERENCES nodes(fingerprint),
        to_node TEXT NOT NULL REFERENCES nodes(fingerprint), created_at TEXT NOT NULL
    ) STRICT;
    CREATE INDEX contact_events_day ON contact_events(from_node, created_at);
    CREATE TABLE contacts (
        a TEXT NOT NULL REFERENCES nodes(fingerprint),
        b TEXT NOT NULL REFERENCES nodes(fingerprint), since TEXT NOT NULL,
        PRIMARY KEY(a,b), CHECK(a < b)
    ) STRICT;
    CREATE TABLE contact_history (
        a TEXT NOT NULL REFERENCES nodes(fingerprint),
        b TEXT NOT NULL REFERENCES nodes(fingerprint), PRIMARY KEY(a,b), CHECK(a < b)
    ) STRICT;
    CREATE TABLE blocks (
        node TEXT NOT NULL REFERENCES nodes(fingerprint),
        blocked TEXT NOT NULL REFERENCES nodes(fingerprint), created_at TEXT NOT NULL,
        PRIMARY KEY(node,blocked)
    ) STRICT;
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sender TEXT NOT NULL REFERENCES nodes(fingerprint),
        recipient TEXT NOT NULL REFERENCES nodes(fingerprint),
        text TEXT NOT NULL, sent_at TEXT NOT NULL
    ) STRICT;
    CREATE INDEX messages_inbox ON messages(recipient,id);
    CREATE INDEX messages_sender ON messages(sender,id);
    CREATE INDEX messages_day ON messages(sender,sent_at);
    CREATE TABLE complaints (
        id TEXT PRIMARY KEY, from_node TEXT NOT NULL REFERENCES nodes(fingerprint),
        about_node TEXT NOT NULL REFERENCES nodes(fingerprint), message_id INTEGER,
        message_text TEXT, text TEXT NOT NULL, created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','settled'))
    ) STRICT;
    CREATE INDEX complaints_day ON complaints(from_node,created_at);
    """,
    # 10: moderation stays local to the messenger, never in the trust feed.
    """
    CREATE TABLE moderations (
        id TEXT PRIMARY KEY, node TEXT NOT NULL,
        action TEXT NOT NULL CHECK(action IN ('suspend','lift')), lifts TEXT,
        signer TEXT NOT NULL, issued_at TEXT NOT NULL, expires_at TEXT NOT NULL,
        signed_json TEXT NOT NULL
    ) STRICT;
    CREATE INDEX moderations_node_action ON moderations(node, action);
    CREATE TRIGGER moderations_no_update BEFORE UPDATE ON moderations
    BEGIN SELECT RAISE(ABORT, 'moderations are append-only'); END;
    CREATE TRIGGER moderations_no_delete BEFORE DELETE ON moderations
    BEGIN SELECT RAISE(ABORT, 'moderations are append-only'); END;
    CREATE TABLE complaint_decisions (
        complaint TEXT NOT NULL UNIQUE REFERENCES complaints(id),
        decision TEXT NOT NULL CHECK(decision IN ('suspended','dismissed')),
        record TEXT REFERENCES moderations(id), note TEXT, decided_at TEXT NOT NULL,
        CHECK ((decision = 'suspended' AND record IS NOT NULL)
            OR (decision = 'dismissed' AND record IS NULL))
    ) STRICT;
    CREATE TRIGGER complaint_decisions_no_update BEFORE UPDATE ON complaint_decisions
    BEGIN SELECT RAISE(ABORT, 'complaint decisions are append-only'); END;
    CREATE TRIGGER complaint_decisions_no_delete BEFORE DELETE ON complaint_decisions
    BEGIN SELECT RAISE(ABORT, 'complaint decisions are append-only'); END;
    CREATE TRIGGER complaints_status_forward BEFORE UPDATE OF status ON complaints
    WHEN OLD.status != 'open' OR NEW.status != 'settled'
    BEGIN SELECT RAISE(ABORT, 'complaints can only be settled once'); END;
    """,
    # 11: look up the lifts of a suspension without scanning moderation history.
    """
    CREATE INDEX moderations_lifts ON moderations(lifts);
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


def in_effect(conn: sqlite3.Connection, now: datetime) -> list[sign.Signed]:
    """Assignments still in their term and every revocation (one is honoured past its term)."""
    rows = conn.execute(
        "SELECT signed_json FROM records WHERE kind = 'revocation' OR expires_at > ?"
        " ORDER BY issued_at, seq",
        (db.to_iso(now),),
    ).fetchall()
    return [sign.Signed.model_validate_json(r["signed_json"]) for r in rows]


def handed_out(conn: sqlite3.Connection, now: datetime) -> list[sign.Signed]:
    """The records in effect, then the newest trust list still in its term (if any)."""
    found = latest_trust_list(conn, now)
    return in_effect(conn, now) + ([] if found is None else [found])


def latest_trust_list(conn: sqlite3.Connection, now: datetime) -> sign.Signed | None:
    row = conn.execute(
        "SELECT signed_json FROM trust_lists WHERE expires_at > ?"
        " ORDER BY issued_at DESC, seq DESC LIMIT 1",
        (db.to_iso(now),),
    ).fetchone()
    return None if row is None else sign.Signed.model_validate_json(row["signed_json"])


def add_trust_list(conn: sqlite3.Connection, signed: sign.Signed) -> None:
    record = signed.record
    conn.execute(
        "INSERT INTO trust_lists (id, issued_at, expires_at, signed_json) VALUES (?, ?, ?, ?)",
        (
            record.id,
            db.to_iso(record.issued_at),
            db.to_iso(record.expires_at),
            json.dumps(signed.model_dump(mode="json"), sort_keys=True),
        ),
    )


@dataclass(frozen=True)
class ProposalSubmission:
    id: str
    node: str
    signed: protocol.SignedProposalSubmission
    status: str
    created_at: str
    reason: str | None = None

    @property
    def submission(self) -> protocol.ProposalSubmission:
        return self.signed.submission


class ProposalLimit(Exception):
    """The node already submitted `protocol.MAX_PROPOSALS_PER_DAY` in the last 24 hours."""


def submit_proposal(
    conn: sqlite3.Connection,
    node: str,
    signed: protocol.SignedProposalSubmission,
    now: datetime,
) -> tuple[ProposalSubmission, bool]:
    """Keep a person's confirmed upload once; an identical retry returns the same row (even
    past the limit). A node keeps at most `MAX_PROPOSALS_PER_DAY` new ones a day."""
    submission = signed.submission
    encoded = json.dumps(signed.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    with db.transaction(conn):
        row = conn.execute(
            "SELECT * FROM proposals WHERE node = ? AND proposal = ? AND evaluation = ?",
            (node, submission.proposal, submission.evaluation),
        ).fetchone()
        if row is not None:
            if row["submission_json"] != encoded:
                raise ValueError(
                    "this proposal evaluation was already submitted with other contents"
                )
            return _proposal_submission(row), False
        (recent,) = conn.execute(
            "SELECT count(*) FROM proposals WHERE node = ? AND created_at > ?",
            (node, db.to_iso(now - timedelta(days=1))),
        ).fetchone()
        if recent >= protocol.MAX_PROPOSALS_PER_DAY:
            raise ProposalLimit(node)
        id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO proposals (id, node, proposal, evaluation, submission_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (id, node, submission.proposal, submission.evaluation, encoded, db.to_iso(now)),
        )
        log(
            conn,
            "proposal_submitted",
            node,
            {"id": id, "proposal": submission.proposal, "evaluation": submission.evaluation},
        )
        row = conn.execute("SELECT * FROM proposals WHERE id = ?", (id,)).fetchone()
    assert row is not None
    return _proposal_submission(row), True


def proposals(conn: sqlite3.Connection, status: str | None = None) -> list[ProposalSubmission]:
    """Submissions, oldest first; optionally by their operator-facing status."""
    if status is None:
        rows = conn.execute("SELECT * FROM proposals ORDER BY created_at, id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM proposals WHERE status = ? ORDER BY created_at, id", (status,)
        ).fetchall()
    return [_proposal_submission(row) for row in rows]


def proposal(conn: sqlite3.Connection, id: str) -> ProposalSubmission | None:
    row = conn.execute("SELECT * FROM proposals WHERE id = ?", (id,)).fetchone()
    return None if row is None else _proposal_submission(row)


def find_proposal(conn: sqlite3.Connection, prefix: str) -> list[str]:
    if not prefix or any(c not in "0123456789abcdef" for c in prefix):
        return []
    rows = conn.execute(
        "SELECT id FROM proposals WHERE substr(id, 1, ?) = ? ORDER BY id LIMIT 2",
        (len(prefix), prefix),
    ).fetchall()
    return [row["id"] for row in rows]


def _proposal_submission(row: sqlite3.Row) -> ProposalSubmission:
    return ProposalSubmission(
        id=row["id"],
        node=row["node"],
        signed=protocol.SignedProposalSubmission.model_validate_json(row["submission_json"]),
        reason=row["reason"],
        status=row["status"],
        created_at=row["created_at"],
    )


def trust(conn: sqlite3.Connection, root: Ed25519PublicKey, now: datetime) -> sign.Trust:
    """What a node would believe after receiving every record kept here and the newest trust
    list (an assignment past its term is refused and counts for nothing, as on a node)."""
    believed = sign.Trust(root, clock=lambda: now)
    found = latest_trust_list(conn, now)
    for signed in all_records(conn) + ([] if found is None else [found]):
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
    owner: str | None = None

    @property
    def owned_by(self) -> str:
        """Its owner, or itself when the operator named none."""
        return self.owner or self.fingerprint

    def as_json(self) -> dict[str, Any]:
        return {
            "node": self.fingerprint,
            "status": "revoked" if self.revoked else self.status,
            "capabilities": [c.model_dump(mode="json") for c in self.capabilities],
            "registered_at": self.registered_at,
            "last_seen": self.last_seen,
            "owner": self.owner,
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
        owner=row["owner"],
    )


def node(conn: sqlite3.Connection, fingerprint: str) -> Node | None:
    row = conn.execute("SELECT * FROM nodes WHERE fingerprint = ?", (fingerprint,)).fetchone()
    return None if row is None else _node(conn, row)


def nodes(conn: sqlite3.Connection) -> list[Node]:
    rows = conn.execute("SELECT * FROM nodes ORDER BY registered_at, fingerprint").fetchall()
    return [_node(conn, r) for r in rows]


def capabilities_json(capabilities: list[Capability]) -> str:
    return json.dumps([c.model_dump(mode="json") for c in capabilities], sort_keys=True)


class ReportLimit(Exception):
    """Five new reports per node and kind in a rolling day."""


def submit_report(
    conn: sqlite3.Connection, node: str, signed: protocol.SignedReport, now: datetime
) -> tuple[protocol.ReportReceived, bool]:
    digest = hashlib.sha256(sign.canonical(signed.report.model_dump(mode="json"))).hexdigest()
    id = hashlib.sha256(f"{node}:{digest}".encode()).hexdigest()
    kind = signed.report.kind
    received = protocol.ReportReceived(id=id, kind=kind)
    with db.transaction(conn):
        if conn.execute("SELECT 1 FROM reports WHERE id = ?", (id,)).fetchone():
            return received, False
        count = conn.execute(
            "SELECT count(*) FROM reports WHERE node = ? AND kind = ? AND created_at > ?",
            (node, kind, db.to_iso(now - timedelta(days=1))),
        ).fetchone()[0]
        if count >= protocol.MAX_REPORTS_PER_DAY:
            raise ReportLimit(node)
        conn.execute(
            "INSERT INTO reports VALUES (?, ?, ?, ?, ?)",
            (id, node, kind, signed.model_dump_json(), db.to_iso(now)),
        )
        log(conn, "report_received", node, {"id": id, "kind": kind})
    return received, True
