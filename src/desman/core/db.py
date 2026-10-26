"""SQLite connection, schema and migrations.

The database is the only place where task state lives (PROJECT_PLAN: "durable state first").

- Connections run in autocommit mode; writes go through `transaction()`, which takes the write
  lock up front (`BEGIN IMMEDIATE`) so concurrent workers queue on `busy_timeout` instead of
  failing halfway through.
- WAL lets readers (`desman status`) work while a worker writes.
- Migrations are append-only: never edit a released one, add the next number instead.
- Timestamps are UTC ISO-8601 strings with millisecond precision (`now_iso()`), so they compare
  correctly as text inside SQL.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

DATA_DIR_ENV: Final = "DESMAN_DATA_DIR"
BUSY_TIMEOUT_MS: Final = 5000

# Same format as SQLite's strftime('%Y-%m-%dT%H:%M:%fZ', 'now').
_NOW_SQL: Final = "(strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"


class SchemaVersionError(RuntimeError):
    """The database was created by a newer DESMAN than this one."""


def default_db_path() -> Path:
    """$DESMAN_DATA_DIR/desman.db or ~/.local/share/desman/desman.db."""
    data_dir = os.environ.get(DATA_DIR_ENV)
    base = Path(data_dir) if data_dir else Path.home() / ".local" / "share" / "desman"
    return base / "desman.db"


def to_iso(moment: datetime) -> str:
    """Format an aware datetime in the database timestamp format."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def now_iso() -> str:
    """Current UTC time in the database timestamp format."""
    return to_iso(datetime.now(UTC))


MIGRATIONS: Final[tuple[str, ...]] = (
    # 1 — initial schema
    f"""
    CREATE TABLE tasks (
        id              TEXT PRIMARY KEY,
        goal            TEXT NOT NULL,
        priority        INTEGER NOT NULL DEFAULT 50,
        parent_task_id  TEXT REFERENCES tasks(id),
        created_at      TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;
    -- Task status is not stored: it is derived from its subtasks, so the two can never disagree.

    CREATE TABLE subtasks (
        id                  TEXT PRIMARY KEY,
        task_id             TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
        seq                 INTEGER NOT NULL,
        description         TEXT NOT NULL,
        acceptance_criteria TEXT NOT NULL,
        inputs_json         TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(inputs_json)),
        priority            INTEGER NOT NULL DEFAULT 50,
        status              TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                                'pending', 'ready', 'running', 'review', 'done', 'paused',
                                'needs_human', 'failed', 'blocked', 'cancelled')),
        -- Lease: present exactly while an executor holds the subtask.
        lease_owner         TEXT,
        lease_until         TEXT,
        -- Pause: where to resume and not before when.
        resume_status       TEXT CHECK (resume_status IN ('ready', 'review')),
        paused_until        TEXT,
        checkpoint_json     TEXT CHECK (checkpoint_json IS NULL OR json_valid(checkpoint_json)),
        attempts_quality    INTEGER NOT NULL DEFAULT 0 CHECK (attempts_quality >= 0),
        attempts_transient  INTEGER NOT NULL DEFAULT 0 CHECK (attempts_transient >= 0),
        created_at          TEXT NOT NULL DEFAULT {_NOW_SQL},
        updated_at          TEXT NOT NULL DEFAULT {_NOW_SQL},
        UNIQUE (task_id, seq),
        CHECK ((status = 'running') = (lease_owner IS NOT NULL AND lease_until IS NOT NULL)),
        CHECK ((status = 'paused') = (resume_status IS NOT NULL))
    ) STRICT;
    CREATE INDEX ix_subtasks_status_priority ON subtasks (status, priority DESC, created_at);
    CREATE INDEX ix_subtasks_task ON subtasks (task_id, seq);

    CREATE TABLE subtask_deps (
        subtask_id  TEXT NOT NULL REFERENCES subtasks(id) ON DELETE CASCADE,
        depends_on  TEXT NOT NULL REFERENCES subtasks(id) ON DELETE CASCADE,
        PRIMARY KEY (subtask_id, depends_on),
        CHECK (subtask_id != depends_on)
    ) STRICT, WITHOUT ROWID;
    CREATE INDEX ix_subtask_deps_depends_on ON subtask_deps (depends_on);

    CREATE TABLE artifacts (
        id            TEXT PRIMARY KEY,
        subtask_id    TEXT NOT NULL REFERENCES subtasks(id) ON DELETE CASCADE,
        attempt       INTEGER NOT NULL,
        kind          TEXT NOT NULL,
        content_json  TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(content_json)),
        path          TEXT,
        created_at    TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;
    CREATE INDEX ix_artifacts_subtask ON artifacts (subtask_id, attempt);

    -- Append-only history of every decision. `seq` gives a total order even within one millisecond.
    CREATE TABLE audit_log (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT NOT NULL DEFAULT {_NOW_SQL},
        task_id       TEXT,
        subtask_id    TEXT,
        actor         TEXT NOT NULL,
        action        TEXT NOT NULL,
        from_status   TEXT,
        to_status     TEXT,
        details_json  TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json))
    ) STRICT;
    CREATE INDEX ix_audit_task ON audit_log (task_id, seq);
    CREATE INDEX ix_audit_subtask ON audit_log (subtask_id, seq);
    CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    """,
)

SCHEMA_VERSION: Final = len(MIGRATIONS)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database with DESMAN's settings. Use ":memory:" for tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    _enable_wal(conn)
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL, waiting for other connections like busy_timeout would.

    The switch needs an exclusive lock and SQLite reports "database is locked" at once instead of
    waiting, so several processes opening a fresh database together would otherwise fail.
    The mode is stored in the file, so later connections find it already set.
    """
    deadline = time.monotonic() + BUSY_TIMEOUT_MS / 1000
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Write transaction: takes the write lock immediately, commits or rolls back as a whole."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    """Version of the schema in this database; 0 for an empty database."""
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    return version


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations, each in its own transaction. Safe to call repeatedly and from
    several processes at once.

    Returns the number of migrations this call applied.
    """
    applied = 0
    while True:
        # The version is read under the write lock, so another migrator cannot change it between
        # the read and the migration (two fresh `init-db` runs would otherwise both see v0).
        with transaction(conn):
            current = schema_version(conn)
            if current > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database schema v{current} is newer than this DESMAN (v{SCHEMA_VERSION}); "
                    "upgrade DESMAN"
                )
            if current == SCHEMA_VERSION:
                return applied
            # executescript() would COMMIT on its own, so run statements one by one inside the
            # transaction; user_version is bumped in the same transaction as the schema change.
            for statement in _split_sql(MIGRATIONS[current]):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {current + 1}")
        applied += 1


def _split_sql(script: str) -> list[str]:
    """Split a migration into complete statements (triggers contain `;` inside BEGIN...END)."""
    statements: list[str] = []
    buffer = ""
    *pieces, tail = script.split(";")
    for piece in pieces:
        buffer += piece + ";"
        if sqlite3.complete_statement(buffer):
            statement = buffer.strip()
            if statement != ";":
                statements.append(statement)
            buffer = ""
    # Whatever follows the last complete statement may only be whitespace or comments.
    rest = (buffer + tail).strip()
    if rest and not all(line.strip().startswith("--") for line in rest.splitlines()):
        raise ValueError(f"incomplete SQL statement in migration: {rest[:80]}")
    return statements


def open_db(path: Path | str | None = None) -> sqlite3.Connection:
    """Connect and bring the schema up to date."""
    conn = connect(path if path is not None else default_db_path())
    migrate(conn)
    return conn
