"""SQLite connection, schema and migrations.

The database is the only place where task state lives (PROJECT_PLAN: "durable state first").

- Connections run in autocommit mode; writes go through `transaction()`, which takes the write
  lock up front (`BEGIN IMMEDIATE`) so concurrent workers queue on `busy_timeout` instead of
  failing halfway through.
- WAL lets readers (`desman status`) work while a worker writes.
- Migrations are append-only: once a migration has been pushed (to any branch: someone may
  have a database from it) never edit it, add the next number instead. `verify_schema()`
  catches databases whose schema does not match their version.
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


def parse_iso(moment: str) -> datetime:
    """Parse a timestamp that carries a timezone (e.g. "2026-01-01T12:00:00Z", "...+02:00")."""
    parsed = datetime.fromisoformat(moment)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp without timezone: {moment!r}")
    return parsed


def normalize_iso(moment: str) -> str:
    """Validate a timestamp from outside (e.g. a provider's reset time) into the database format."""
    return to_iso(parse_iso(moment))


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
        -- Lease: present exactly while an executor holds the subtask. The token is new for
        -- every claim, so a stalled executor cannot act on a later claim (even under its name).
        lease_owner         TEXT,
        lease_token         TEXT,
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
        CHECK (CASE WHEN status = 'running'
                    THEN lease_owner IS NOT NULL AND lease_token IS NOT NULL
                         AND lease_until IS NOT NULL
                    ELSE lease_owner IS NULL AND lease_token IS NULL AND lease_until IS NULL END),
        CHECK (CASE WHEN status = 'paused'
                    THEN resume_status IS NOT NULL
                    ELSE resume_status IS NULL AND paused_until IS NULL END)
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
    # 2 — which artifact a DONE verdict accepted (NULL: done without one)
    """
    ALTER TABLE subtasks ADD COLUMN accepted_attempt INTEGER
        CHECK (accepted_attempt IS NULL OR status = 'done');

    -- Backfill with the rule Store._accepted_attempt applies from now on: the reviewer accepted
    -- the latest attempt; the operator did so only when resolving an escalation from REVIEW.
    UPDATE subtasks SET accepted_attempt = (
        SELECT MAX(a.attempt) FROM artifacts a WHERE a.subtask_id = subtasks.id)
    WHERE status = 'done' AND (
        (SELECT from_status FROM audit_log l WHERE l.subtask_id = subtasks.id
           AND l.to_status = 'done' ORDER BY l.seq DESC LIMIT 1) = 'review'
        OR (SELECT from_status FROM audit_log l WHERE l.subtask_id = subtasks.id
              AND l.to_status = 'needs_human' ORDER BY l.seq DESC LIMIT 1) = 'review');
    """,
    # 3 — every audit entry belongs to an existing task, and its subtask (if any) to that task
    """
    CREATE TRIGGER audit_log_needs_task BEFORE INSERT ON audit_log
        WHEN NEW.task_id IS NULL OR NOT EXISTS (SELECT 1 FROM tasks WHERE id = NEW.task_id)
        BEGIN SELECT RAISE(ABORT, 'audit entry must name an existing task'); END;
    CREATE TRIGGER audit_log_subtask_in_task BEFORE INSERT ON audit_log
        WHEN NEW.subtask_id IS NOT NULL AND NOT EXISTS (
            SELECT 1 FROM subtasks WHERE id = NEW.subtask_id AND task_id = NEW.task_id)
        BEGIN SELECT RAISE(ABORT, 'audit entry subtask must belong to its task'); END;
    """,
    # 4 — every call to a model provider (ROADMAP 2.4): usage per provider and key, limits seen
    f"""
    CREATE TABLE model_calls (
        seq                INTEGER PRIMARY KEY AUTOINCREMENT,
        at                 TEXT NOT NULL DEFAULT {_NOW_SQL},
        provider           TEXT NOT NULL,
        model              TEXT NOT NULL,
        key_id             TEXT,      -- fingerprint of the key, never the key
        task_id            TEXT,
        subtask_id         TEXT,
        prompt_hash        TEXT NOT NULL,
        response_hash      TEXT,
        prompt_tokens      INTEGER NOT NULL DEFAULT 0 CHECK (prompt_tokens >= 0),
        completion_tokens  INTEGER NOT NULL DEFAULT 0 CHECK (completion_tokens >= 0),
        latency_ms         INTEGER NOT NULL CHECK (latency_ms >= 0),
        http_status        INTEGER,
        error_kind         TEXT,      -- NULL: the call succeeded
        error              TEXT,
        limits_json        TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(limits_json))
    ) STRICT;
    CREATE INDEX ix_model_calls_provider ON model_calls (provider, at);
    CREATE INDEX ix_model_calls_key ON model_calls (key_id, at);
    CREATE TRIGGER model_calls_no_update BEFORE UPDATE ON model_calls
        BEGIN SELECT RAISE(ABORT, 'model_calls is append-only'); END;
    """,
    # 5 — replies the caller accepted, by request (ROADMAP 2.6): a resume does not pay twice
    """
    CREATE TABLE response_cache (
        key                TEXT PRIMARY KEY,  -- sha256 of provider, model, messages, parameters
        provider           TEXT NOT NULL,
        model              TEXT NOT NULL,
        text               TEXT NOT NULL,
        finish_reason      TEXT NOT NULL,
        prompt_tokens      INTEGER NOT NULL CHECK (prompt_tokens >= 0),
        completion_tokens  INTEGER NOT NULL CHECK (completion_tokens >= 0),
        created_at         TEXT NOT NULL
    ) STRICT;
    CREATE INDEX ix_response_cache_created ON response_cache (created_at);
    """,
    # 6 — a subtask split into smaller ones (ROADMAP 2.9b) is their parent
    """
    ALTER TABLE subtasks ADD COLUMN parent_id TEXT REFERENCES subtasks(id) ON DELETE CASCADE;
    CREATE INDEX ix_subtasks_parent ON subtasks (parent_id);
    """,
    # 7 — tool calls with side effects (ROADMAP 3.4, D6): intent before the call, result after,
    # so a rerun of the same execution replays them instead of acting twice (core/effects.py)
    """
    CREATE TABLE effects (
        subtask_id   TEXT NOT NULL REFERENCES subtasks(id) ON DELETE CASCADE,
        execution    INTEGER NOT NULL CHECK (execution >= 0),
        ordinal      INTEGER NOT NULL CHECK (ordinal >= 0),
        tool         TEXT NOT NULL,
        key          TEXT NOT NULL,  -- sha256 of the tool and its arguments, never the arguments
        lease_token  TEXT NOT NULL,  -- whose call it is: a late result from a lost lease is ignored
        state        TEXT NOT NULL CHECK (state IN ('started', 'done', 'interrupted')),
        result_json  TEXT CHECK (result_json IS NULL OR json_valid(result_json)),
        started_at   TEXT NOT NULL,
        finished_at  TEXT,
        PRIMARY KEY (subtask_id, execution, ordinal),
        CHECK ((state = 'started') = (result_json IS NULL AND finished_at IS NULL))
    ) STRICT, WITHOUT ROWID;
    """,
    # 8 — answers of the web tools (ROADMAP 3.5), kept a day: a page read twice is fetched once
    """
    CREATE TABLE web_cache (
        key         TEXT PRIMARY KEY,  -- sha256 of the kind and the request (query or URL)
        kind        TEXT NOT NULL CHECK (kind IN ('search', 'page')),
        body_json   TEXT NOT NULL CHECK (json_valid(body_json)),
        created_at  TEXT NOT NULL
    ) STRICT;
    CREATE INDEX ix_web_cache_created ON web_cache (created_at);
    """,
    # 9 — chat sessions (ROADMAP C.2, ADR 0003): what was said stays as it was said; a summary
    # (C.6) is a new message, the originals are never changed or removed
    f"""
    CREATE TABLE sessions (
        id          TEXT PRIMARY KEY,
        created_at  TEXT NOT NULL DEFAULT {_NOW_SQL}
    ) STRICT;

    CREATE TABLE messages (
        seq         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL REFERENCES sessions(id),
        at          TEXT NOT NULL DEFAULT {_NOW_SQL},
        role        TEXT NOT NULL CHECK (role IN ('user', 'interlocutor', 'desman')),
        kind        TEXT NOT NULL CHECK (kind IN ('message', 'summary', 'event')),
        text        TEXT NOT NULL,
        task_id     TEXT REFERENCES tasks(id)
    ) STRICT;
    CREATE INDEX ix_messages_session ON messages (session_id, seq);
    CREATE TRIGGER messages_no_update BEFORE UPDATE ON messages
        BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
    CREATE TRIGGER messages_no_delete BEFORE DELETE ON messages
        BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
    """,
    # 10 — a task asked for in the chat (ROADMAP C.3) belongs to its session; NULL: `desman run`
    """
    ALTER TABLE tasks ADD COLUMN session_id TEXT REFERENCES sessions(id);
    CREATE INDEX ix_tasks_session ON tasks (session_id);
    """,
    # 11 — what a model call was for (ROADMAP C.4): the role that made it, the chat session of
    # the interlocutor's own calls, and the error's class (a rate limit is not a used-up quota)
    """
    ALTER TABLE model_calls ADD COLUMN role TEXT
        CHECK (role IN ('planner', 'executor', 'reviewer', 'decomposer', 'interlocutor'));
    ALTER TABLE model_calls ADD COLUMN session_id TEXT REFERENCES sessions(id);
    ALTER TABLE model_calls ADD COLUMN error_type TEXT;
    CREATE INDEX ix_model_calls_task ON model_calls (task_id, seq);
    """,
    # 12 — what the person did outside any task (ROADMAP C.5, ADR 0003): commands confirmed or
    # declined in the chat, keys added (provider and fingerprint, never the key), permissions
    # granted or revoked. `audit_log` tells the story of tasks; this is the operator's own
    f"""
    CREATE TABLE operator_log (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT NOT NULL DEFAULT {_NOW_SQL},
        session_id    TEXT REFERENCES sessions(id),
        action        TEXT NOT NULL,
        details_json  TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json))
    ) STRICT;
    CREATE TRIGGER operator_log_no_update BEFORE UPDATE ON operator_log
        BEGIN SELECT RAISE(ABORT, 'operator_log is append-only'); END;
    CREATE TRIGGER operator_log_no_delete BEFORE DELETE ON operator_log
        BEGIN SELECT RAISE(ABORT, 'operator_log is append-only'); END;
    """,
    # 13 — a session the person named or deleted (`/rename`, `/delete`). Deleting hides it from
    # the lists and the chat; its messages stay, as every message does (migration 9)
    """
    ALTER TABLE sessions ADD COLUMN title TEXT CHECK (title IS NULL OR length(title) > 0);
    ALTER TABLE sessions ADD COLUMN deleted_at TEXT;
    """,
    # 14 — a summary (ROADMAP C.6) says which messages it stands in for: every message of its
    # session up to `covers`. Only a summary has it, and every summary does (none were kept
    # before this migration)
    """
    ALTER TABLE messages ADD COLUMN covers INTEGER
        CHECK ((covers IS NOT NULL) = (kind = 'summary'));
    """,
    # 15 — the user's profile (ROADMAP C.7): short notes about the person that the interlocutor
    # sees. Personal data, so they can be changed and erased for real; who changed which note
    # is kept in operator_log (by number, never the text)
    """
    CREATE TABLE profile_notes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        text        TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND 200),
        author      TEXT NOT NULL CHECK (author IN ('person', 'interlocutor')),
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    ) STRICT;
    """,
    # 16 — several workers on one database (ROADMAP 4.1). A task keeps its own retry budgets, so
    # whoever recovers its subtasks charges them by the task's policy, not their own; and the
    # tools its executor may have (`none`: a chat summary, whose input may say anything). A
    # review in progress is reserved by its reviewer for a short while: two workers do not pay
    # for the same verdict
    """
    ALTER TABLE tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 5 CHECK (max_attempts >= 1);
    ALTER TABLE tasks ADD COLUMN max_transient INTEGER NOT NULL DEFAULT 5
        CHECK (max_transient >= 1);
    ALTER TABLE tasks ADD COLUMN tools TEXT NOT NULL DEFAULT 'granted'
        CHECK (tools IN ('granted', 'none'));
    ALTER TABLE subtasks ADD COLUMN review_owner TEXT;
    ALTER TABLE subtasks ADD COLUMN review_until TEXT
        CHECK ((review_until IS NULL) = (review_owner IS NULL)
               AND (review_until IS NULL OR status = 'review'));
    """,
    # 17 — a review reservation has a token, like a lease: a reviewer's verdict is written only
    # with the token of the reservation that is still current, so a reviewer whose reservation
    # ran out and was taken over cannot decide after all. Reservations in flight are dropped
    # (taken over at once, as if expired): they hold no work, only the right to one model call
    """
    UPDATE subtasks SET review_owner = NULL, review_until = NULL;
    ALTER TABLE subtasks ADD COLUMN review_token TEXT
        CHECK ((review_token IS NULL) = (review_owner IS NULL));
    """,
    # 18 — when a subtask joined the queue it waits in (READY to run, REVIEW for a verdict): the
    # scheduler ages waiting work by it (core/scheduler.py). Work already waiting is taken as
    # queued at its last change
    """
    ALTER TABLE subtasks ADD COLUMN queued_at TEXT;
    UPDATE subtasks SET queued_at = updated_at WHERE status IN ('ready', 'review');
    """,
    # 19 — the pace of each provider's model (ROADMAP 4.5, llm/limiter.py): a token bucket per
    # provider/model, and until when a limit the provider reported keeps it aside. Shared by the
    # workers on the database and kept across restarts
    """
    CREATE TABLE rate_buckets (
        scope TEXT PRIMARY KEY,
        tokens REAL NOT NULL CHECK (tokens >= 0),
        updated_at TEXT NOT NULL,
        blocked_until TEXT
    );
    """,
    # 20 — what sort of work a subtask is (ROADMAP 4.5, `models.Kind`): its executor's model is
    # picked by it. Subtasks made before were not told apart: text, the general sort
    """
    ALTER TABLE subtasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'text'
        CHECK (kind IN ('text', 'code', 'analysis'));
    """,
    # 21 — the protected files as the person approved them (ROADMAP 5.3, core/protected.py):
    # each approval is a full snapshot, with each file's content for a diff. `release`: taken
    # as approved on a database with no approval, because every file matched the shipped
    # hashes. Append-only, like the logs
    f"""
    CREATE TABLE protected_approvals (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        ts      TEXT NOT NULL DEFAULT {_NOW_SQL},
        source  TEXT NOT NULL CHECK (source IN ('release', 'operator')),
        note    TEXT
    ) STRICT;
    CREATE TABLE protected_files (
        approval_id  INTEGER NOT NULL REFERENCES protected_approvals(id),
        path         TEXT NOT NULL,
        sha256       TEXT CHECK (sha256 IS NULL OR length(sha256) = 64),
        content      BLOB CHECK ((content IS NULL) = (sha256 IS NULL)),
        PRIMARY KEY (approval_id, path)
    ) STRICT;
    CREATE TRIGGER protected_approvals_no_update BEFORE UPDATE ON protected_approvals
        BEGIN SELECT RAISE(ABORT, 'protected_approvals is append-only'); END;
    CREATE TRIGGER protected_approvals_no_delete BEFORE DELETE ON protected_approvals
        BEGIN SELECT RAISE(ABORT, 'protected_approvals is append-only'); END;
    CREATE TRIGGER protected_files_no_update BEFORE UPDATE ON protected_files
        BEGIN SELECT RAISE(ABORT, 'protected_files is append-only'); END;
    CREATE TRIGGER protected_files_no_delete BEFORE DELETE ON protected_files
        BEGIN SELECT RAISE(ABORT, 'protected_files is append-only'); END;
    """,
    # 22 — the Improvement Analyst's proposals (ROADMAP 5.4, core/proposals.py): one per
    # analysis task. While the task runs it is `analysing`; then it is settled once (proposed,
    # nothing to propose, or unusable) and never changes again: the diff the person reads is
    # the diff that was proposed. What happens to it later (5.5, 5.6) goes elsewhere
    f"""
    CREATE TABLE proposals (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id     TEXT NOT NULL UNIQUE REFERENCES tasks(id),
        created_at  TEXT NOT NULL DEFAULT {_NOW_SQL},
        since       TEXT,
        status      TEXT NOT NULL DEFAULT 'analysing'
                    CHECK (status IN ('analysing', 'proposed', 'nothing', 'unusable')),
        settled_at  TEXT CHECK ((settled_at IS NULL) = (status = 'analysing')),
        summary     TEXT,
        reason      TEXT,
        details_json TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json)),
        diff        TEXT CHECK ((diff IS NOT NULL) = (status = 'proposed'))
    ) STRICT;
    CREATE TRIGGER proposals_settled_once BEFORE UPDATE ON proposals
        WHEN OLD.status <> 'analysing'
        BEGIN SELECT RAISE(ABORT, 'a settled proposal never changes'); END;
    CREATE TRIGGER proposals_no_delete BEFORE DELETE ON proposals
        BEGIN SELECT RAISE(ABORT, 'proposals are kept'); END;
    """,
    # 23 — evaluations of proposals (ROADMAP 5.5, desman/evaluation.py): the benchmark run on
    # the version at `commit` and on it with the proposal, each run's result as it reported it,
    # and the verdict; `failed` when the evaluation could not be finished (`reason`). Written
    # once when the evaluation ends, never changed or removed: a proposal may be evaluated
    # again, and every evaluation stays
    f"""
    CREATE TABLE evaluations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        proposal_id INTEGER NOT NULL REFERENCES proposals(id),
        created_at  TEXT NOT NULL DEFAULT {_NOW_SQL},
        mode        TEXT NOT NULL CHECK (mode IN ('fake', 'real')),
        git_commit  TEXT NOT NULL,
        verdict     TEXT NOT NULL CHECK (verdict IN ('better', 'worse', 'same', 'failed')),
        reason      TEXT CHECK ((reason IS NOT NULL) = (verdict = 'failed')),
        base_json   TEXT CHECK (base_json IS NULL OR json_valid(base_json)),
        candidate_json TEXT CHECK (candidate_json IS NULL OR json_valid(candidate_json))
    ) STRICT;
    CREATE INDEX evaluations_proposal ON evaluations(proposal_id);
    CREATE TRIGGER evaluations_no_update BEFORE UPDATE ON evaluations
        BEGIN SELECT RAISE(ABORT, 'evaluations are append-only'); END;
    CREATE TRIGGER evaluations_no_delete BEFORE DELETE ON evaluations
        BEGIN SELECT RAISE(ABORT, 'evaluations are append-only'); END;
    """,
    # 24 — temporary chats (ROADMAP C.11, `desman -t`): a session marked temporary when it is
    # created, never later, by the chat that holds it (`owner`, "host:pid"). Out of the lists;
    # its messages are erased when that chat ends, or after it is gone. Only a temporary
    # session's messages may be deleted: every other session stays append-only (migration 9).
    # The session's row stays, so the tasks and calls that name it keep their link
    """
    ALTER TABLE sessions ADD COLUMN temporary INTEGER NOT NULL DEFAULT 0
        CHECK (temporary IN (0, 1));
    ALTER TABLE sessions ADD COLUMN owner TEXT CHECK ((owner IS NOT NULL) = (temporary = 1));
    CREATE TRIGGER sessions_temporary_fixed BEFORE UPDATE OF temporary, owner ON sessions
        BEGIN SELECT RAISE(ABORT, 'a session is temporary from its start or never'); END;
    DROP TRIGGER messages_no_delete;
    CREATE TRIGGER messages_no_delete BEFORE DELETE ON messages
        WHEN NOT EXISTS (SELECT 1 FROM sessions WHERE id = OLD.session_id AND temporary = 1)
        BEGIN SELECT RAISE(ABORT, 'messages are append-only'); END;
    """,
    # 25 — a proposal applied and reverted by the person (ROADMAP 5.6, desman/applying.py):
    # each time, the commit made (none when only settings changed), the HEAD it was made on
    # and its branch, and in `details_json` the files and the settings changed. Applied and
    # reverted in turn, starting with applied, only a proposed change; never changed or removed
    f"""
    CREATE TABLE applications (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        proposal_id INTEGER NOT NULL REFERENCES proposals(id),
        created_at  TEXT NOT NULL DEFAULT {_NOW_SQL},
        action      TEXT NOT NULL CHECK (action IN ('applied', 'reverted')),
        git_commit  TEXT,
        base_commit TEXT CHECK ((base_commit IS NULL) = (git_commit IS NULL)),
        branch      TEXT CHECK (branch IS NULL OR git_commit IS NOT NULL),
        details_json TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json))
    ) STRICT;
    CREATE INDEX applications_proposal ON applications(proposal_id);
    CREATE TRIGGER applications_in_turn BEFORE INSERT ON applications
        WHEN NEW.action = COALESCE((SELECT action FROM applications
                                    WHERE proposal_id = NEW.proposal_id
                                    ORDER BY id DESC LIMIT 1), 'reverted')
        BEGIN SELECT RAISE(ABORT, 'a proposal is applied and reverted in turn'); END;
    CREATE TRIGGER applications_proposed_only BEFORE INSERT ON applications
        WHEN (SELECT status FROM proposals WHERE id = NEW.proposal_id) IS NOT 'proposed'
        BEGIN SELECT RAISE(ABORT, 'only a proposed change is applied'); END;
    CREATE TRIGGER applications_no_update BEFORE UPDATE ON applications
        BEGIN SELECT RAISE(ABORT, 'applications are append-only'); END;
    CREATE TRIGGER applications_no_delete BEFORE DELETE ON applications
        BEGIN SELECT RAISE(ABORT, 'applications are append-only'); END;
    """,
    # 26 — the network's journal (ROADMAP G.3, ADR 0004 decision 12): joining and leaving
    # desman-global, modules lent and taken back, later every record sent or received and every
    # one refused. Keys by fingerprint, never a secret; append-only, like operator_log
    f"""
    CREATE TABLE network_log (
        seq           INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT NOT NULL DEFAULT {_NOW_SQL},
        action        TEXT NOT NULL,
        details_json  TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(details_json))
    ) STRICT;
    CREATE TRIGGER network_log_no_update BEFORE UPDATE ON network_log
        BEGIN SELECT RAISE(ABORT, 'network_log is append-only'); END;
    CREATE TRIGGER network_log_no_delete BEFORE DELETE ON network_log
        BEGIN SELECT RAISE(ABORT, 'network_log is append-only'); END;
    """,
    # 27 — a subtask that must run code and report what it printed (`SubtaskSpec.runs_code`):
    # it is done only through `run_command`, which the coordinator checks, never on a model's
    # word. Subtasks made before were not told apart: none of them is held to it
    """
    ALTER TABLE subtasks ADD COLUMN runs_code INTEGER NOT NULL DEFAULT 0
        CHECK (runs_code IN (0, 1));
    """,
)

SCHEMA_VERSION: Final = len(MIGRATIONS)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database with DESMAN's settings. Use ":memory:" for tests."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Created private: the call log holds provider error text and hashes of prompts.
        # SQLite gives the -wal and -shm files the same mode as the database.
        os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
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


@contextmanager
def read_snapshot(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Consistent reads: every SELECT inside sees the same committed state (one WAL snapshot),
    without blocking writers. Inside an open transaction it adds nothing."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    """Version of the schema in this database; 0 for an empty database."""
    version: int = conn.execute("PRAGMA user_version").fetchone()[0]
    return version


def migrate(conn: sqlite3.Connection, migrations: tuple[str, ...] = MIGRATIONS) -> int:
    """Apply pending migrations, each in its own transaction. Safe to call repeatedly and from
    several processes at once. `migrations`: another database's own list (the registry's).

    Returns the number of migrations this call applied.
    """
    latest = len(migrations)
    applied = 0
    while True:
        # The version is read under the write lock, so another migrator cannot change it between
        # the read and the migration (two fresh `init-db` runs would otherwise both see v0).
        with transaction(conn):
            current = schema_version(conn)
            if current > latest:
                raise SchemaVersionError(
                    f"database schema v{current} is newer than this DESMAN (v{latest}); "
                    "upgrade DESMAN"
                )
            if current == latest:
                return applied
            # executescript() would COMMIT on its own, so run statements one by one inside the
            # transaction; user_version is bumped in the same transaction as the schema change.
            for statement in _split_sql(migrations[current]):
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


class SchemaMismatchError(RuntimeError):
    """The database's tables do not match what its schema version promises."""


def expected_schema(migrations: tuple[str, ...] = MIGRATIONS) -> dict[str, list[str]]:
    """Columns per table that the migrations produce, built from scratch in memory."""
    reference = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for script in migrations:
            for statement in _split_sql(script):
                reference.execute(statement)
        return _columns(reference)
    finally:
        reference.close()


def expected_triggers(migrations: tuple[str, ...] = MIGRATIONS) -> set[str]:
    """Trigger names the migrations produce (they carry invariants such as append-only audit)."""
    reference = sqlite3.connect(":memory:", isolation_level=None)
    try:
        for script in migrations:
            for statement in _split_sql(script):
                reference.execute(statement)
        return _triggers(reference)
    finally:
        reference.close()


def _triggers(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}


def _columns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    return {t: sorted(r[1] for r in conn.execute(f"PRAGMA table_info({t})")) for t in tables}


def verify_schema(
    conn: sqlite3.Connection,
    migrations: tuple[str, ...] = MIGRATIONS,
    *,
    fresh: str = "`desman init-db`",
) -> None:
    """Refuse databases whose columns or triggers differ from the current schema.

    Happens with a database created by an unreleased development revision whose migration was
    later changed; failing here with a clear message beats a KeyError deep inside a task run.
    """
    expected, actual = expected_schema(migrations), _columns(conn)
    problems = []
    for table, columns in expected.items():
        have = actual.get(table)
        if have is None:
            problems.append(f"missing table {table}")
            continue
        if missing := sorted(set(columns) - set(have)):
            problems.append(f"{table}: missing {', '.join(missing)}")
        if extra := sorted(set(have) - set(columns)):
            problems.append(f"{table}: unexpected {', '.join(extra)}")
    if missing_triggers := sorted(expected_triggers(migrations) - _triggers(conn)):
        problems.append(f"missing triggers {', '.join(missing_triggers)}")
    if problems:
        raise SchemaMismatchError(
            f"database schema does not match v{schema_version(conn)} ({'; '.join(problems)}). "
            f"It was probably created by a development revision; move it away and run "
            f"{fresh} to start fresh."
        )


def open_db(path: Path | str | None = None) -> sqlite3.Connection:
    """Connect, bring the schema up to date and check it matches."""
    conn = connect(path if path is not None else default_db_path())
    try:
        migrate(conn)
        verify_schema(conn)
    except BaseException:
        conn.close()
        raise
    return conn
