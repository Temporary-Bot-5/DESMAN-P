"""Tests for the database schema and migrations."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from desman.cli import cli
from desman.core import db, operator
from desman.core.audit import check_history, history
from desman.core.sessions import Kind, Role, Sessions
from desman.core.states import SubtaskStatus
from desman.core.store import Store


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    connection = db.open_db(":memory:")
    yield connection
    connection.close()


def new_id() -> str:
    return uuid.uuid4().hex


def add_task(conn: sqlite3.Connection) -> str:
    task_id = new_id()
    conn.execute("INSERT INTO tasks (id, goal) VALUES (?, ?)", (task_id, "goal"))
    return task_id


def add_subtask(conn: sqlite3.Connection, task_id: str, seq: int = 0, **cols: object) -> str:
    subtask_id = new_id()
    values = {
        "id": subtask_id,
        "task_id": task_id,
        "seq": seq,
        "description": "d",
        "acceptance_criteria": "a",
        **cols,
    }
    names = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    conn.execute(f"INSERT INTO subtasks ({names}) VALUES ({marks})", tuple(values.values()))
    return subtask_id


# ── Migrations ───────────────────────────────────────────────────────────────


def test_fresh_database_is_migrated(conn: sqlite3.Connection) -> None:
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "subtasks", "subtask_deps", "artifacts", "audit_log"} <= tables


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    first = db.connect(path)
    assert db.migrate(first) == db.SCHEMA_VERSION
    first.close()
    second = db.connect(path)
    assert db.migrate(second) == 0
    assert db.schema_version(second) == db.SCHEMA_VERSION
    second.close()


def test_newer_schema_is_refused(conn: sqlite3.Connection) -> None:
    conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    with pytest.raises(db.SchemaVersionError):
        db.migrate(conn)


def test_failed_migration_leaves_database_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    broken = (*db.MIGRATIONS, "CREATE TABLE extra (x INTEGER) STRICT; INSERT INTO nope VALUES (1);")
    monkeypatch.setattr(db, "MIGRATIONS", broken)
    monkeypatch.setattr(db, "SCHEMA_VERSION", len(broken))
    conn = db.connect(":memory:")
    with pytest.raises(sqlite3.OperationalError):
        db.migrate(conn)
    # The real migrations committed, the broken one rolled back completely.
    assert db.schema_version(conn) == len(broken) - 1
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'extra'").fetchone() is None


def test_split_sql_keeps_trigger_bodies() -> None:
    script = (
        "CREATE TABLE t (x INTEGER);\nCREATE TRIGGER g BEFORE DELETE ON t\nBEGIN SELECT 1; END;\n"
    )
    assert len(db._split_sql(script)) == 2
    with pytest.raises(ValueError):
        db._split_sql("CREATE TABLE t (x INTEGER")


def test_connection_settings(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "d.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == db.BUSY_TIMEOUT_MS
    conn.close()


def test_default_db_path_honours_env(tmp_path: Path) -> None:
    assert db.default_db_path() == tmp_path / "data" / "desman.db"


# ── Transactions ─────────────────────────────────────────────────────────────


def test_transaction_rolls_back_on_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError), db.transaction(conn):
        add_task(conn)
        raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_transaction_commits(conn: sqlite3.Connection) -> None:
    with db.transaction(conn):
        add_task(conn)
    assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


# ── Constraints ──────────────────────────────────────────────────────────────


def test_status_check_matches_state_machine(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    lease: dict[str, object] = {"lease_owner": "w", "lease_token": "t", "lease_until": db.now_iso()}
    for seq, status in enumerate(SubtaskStatus):
        extra: dict[str, object] = {}
        if status is SubtaskStatus.RUNNING:
            extra = lease
        if status is SubtaskStatus.PAUSED:
            extra = {"resume_status": "ready"}
        add_subtask(conn, task_id, seq, status=status.value, **extra)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, 99, status="leased")


def test_running_requires_lease(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="running")
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="ready", lease_owner="w", lease_until=db.now_iso())
    with pytest.raises(sqlite3.IntegrityError):  # half a lease is not a lease
        add_subtask(conn, task_id, status="running", lease_owner="w", lease_until=db.now_iso())


def test_paused_requires_resume_status(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="paused")
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="paused", resume_status="running")
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="ready", paused_until=db.now_iso())


def test_json_columns_are_validated(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, inputs_json="{broken")


def test_seq_unique_within_task(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    add_subtask(conn, task_id, 0)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, 0)


def test_dependencies(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    a = add_subtask(conn, task_id, 0)
    b = add_subtask(conn, task_id, 1)
    conn.execute("INSERT INTO subtask_deps VALUES (?, ?)", (b, a))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO subtask_deps VALUES (?, ?)", (a, a))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO subtask_deps VALUES (?, ?)", (a, "missing"))


def test_deleting_task_cascades(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    a = add_subtask(conn, task_id, 0)
    b = add_subtask(conn, task_id, 1)
    conn.execute("INSERT INTO subtask_deps VALUES (?, ?)", (b, a))
    conn.execute(
        "INSERT INTO artifacts (id, subtask_id, attempt, kind) VALUES (?, ?, 1, 'r')", (new_id(), a)
    )
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    for table in ("subtasks", "subtask_deps", "artifacts"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_strict_types(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO tasks (id, goal, priority) VALUES ('t', 'g', 'high')")


def test_audit_log_is_append_only(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    for action in ("test", "test2"):
        conn.execute(
            "INSERT INTO audit_log (task_id, actor, action) VALUES (?, 'human', ?)",
            (task_id, action),
        )
    seqs = [r["seq"] for r in conn.execute("SELECT seq FROM audit_log ORDER BY seq")]
    assert seqs == [1, 2]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE audit_log SET action = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM audit_log")


def test_audit_entries_must_belong_to_a_task(conn: sqlite3.Connection) -> None:
    task_id, other = add_task(conn), add_task(conn)
    subtask = add_subtask(conn, task_id)
    insert = (
        "INSERT INTO audit_log (task_id, subtask_id, actor, action) VALUES (?, ?, 'human', 'x')"
    )
    conn.execute(insert, (task_id, subtask))
    conn.execute(insert, (task_id, None))
    with pytest.raises(sqlite3.IntegrityError, match="existing task"):
        conn.execute(insert, (None, None))
    with pytest.raises(sqlite3.IntegrityError, match="existing task"):
        conn.execute(insert, ("no-such-task", None))
    with pytest.raises(sqlite3.IntegrityError):  # either trigger may fire first
        conn.execute(insert, (None, subtask))
    with pytest.raises(sqlite3.IntegrityError, match="belong to its task"):
        conn.execute(insert, (other, subtask))
    with pytest.raises(sqlite3.IntegrityError, match="belong to its task"):
        conn.execute(insert, (task_id, "no-such-subtask"))


def test_v2_database_upgrades_and_keeps_its_history(tmp_path: Path) -> None:
    path = tmp_path / "v2.db"
    old = _db_at_version(path, 2)
    task_id = add_task(old)
    old.execute("INSERT INTO audit_log (actor, action) VALUES ('human', 'legacy, no task')")
    old.execute(
        "INSERT INTO audit_log (task_id, actor, action) VALUES (?, 'human', 'old')", (task_id,)
    )
    old.close()
    conn = db.open_db(path)
    assert [r[0] for r in conn.execute("SELECT action FROM audit_log ORDER BY seq")] == [
        "legacy, no task",
        "old",
    ]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO audit_log (actor, action) VALUES ('human', 'new')")
    conn.close()


def test_legacy_entries_without_task_id_stay_in_the_task_history(tmp_path: Path) -> None:
    # Before migration 3 nothing forced task_id; the log is append-only, so such rows are found
    # through their subtask rather than backfilled.
    path = tmp_path / "v2.db"
    old = _db_at_version(path, 2)
    task_id = add_task(old)
    subtask = add_subtask(old, task_id, status="ready")
    old.execute(
        "INSERT INTO audit_log (task_id, actor, action, details_json) "
        "VALUES (?, 'human', 'task_created', ?)",
        (task_id, json.dumps({"subtasks": [subtask]})),
    )
    old.execute(
        "INSERT INTO audit_log (subtask_id, actor, action, from_status, to_status) "
        "VALUES (?, 'scheduler', 'dependencies_met', 'pending', 'ready')",
        (subtask,),
    )
    old.close()
    conn = db.open_db(path)
    entries = history(conn, task_id)
    assert [(e.action, e.task_id) for e in entries] == [
        ("task_created", task_id),
        ("dependencies_met", task_id),
    ]
    assert check_history(conn, task_id) == []
    assert len(Store(conn).audit_log(task_id)) == 2
    conn.close()


def test_database_missing_a_trigger_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.db"
    conn = db.open_db(path)
    conn.execute("DROP TRIGGER audit_log_no_delete")
    conn.close()
    with pytest.raises(db.SchemaMismatchError, match="missing triggers audit_log_no_delete"):
        db.open_db(path)


def test_python_and_sql_timestamps_share_format(conn: sqlite3.Connection) -> None:
    sql_now = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0]
    py_now = db.now_iso()
    assert len(sql_now) == len(py_now) == len("2026-01-01T00:00:00.000Z")
    assert sql_now[:16] <= py_now


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_init_db_cli_is_idempotent(tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, ["init-db"])
    assert first.exit_code == 0, first.output
    assert f"({db.SCHEMA_VERSION} migrations applied)" in first.output
    assert (tmp_path / "data" / "desman.db").exists()
    second = runner.invoke(cli, ["init-db"])
    assert "(0 migrations applied)" in second.output


def test_init_db_cli_refuses_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    conn = db.connect(path)
    conn.execute("PRAGMA user_version = 999")
    conn.close()
    result = CliRunner().invoke(cli, ["init-db", "--db", str(path)])
    assert result.exit_code != 0
    assert "newer" in result.output


def test_concurrent_migrators_do_not_collide(tmp_path: Path) -> None:
    # Another process holds the write lock and is migrating; we must wait for it and then see
    # the version it committed, instead of re-applying migrations from a stale version read.
    path = tmp_path / "d.db"
    other = db.connect(path)
    other.execute("BEGIN IMMEDIATE")

    outcome: dict[str, object] = {}

    def migrate_ours() -> None:
        ours = db.connect(path)  # sqlite3 connections belong to the thread that opened them
        try:
            outcome["applied"] = db.migrate(ours)
            outcome["version"] = db.schema_version(ours)
        except Exception as e:  # reported via the assertion below
            outcome["error"] = e
        finally:
            ours.close()

    thread = threading.Thread(target=migrate_ours)
    thread.start()
    time.sleep(0.3)  # let our migrator reach the lock
    for version, script in enumerate(db.MIGRATIONS, start=1):
        for statement in db._split_sql(script):
            other.execute(statement)
        other.execute(f"PRAGMA user_version = {version}")
    other.execute("COMMIT")
    thread.join(timeout=10)

    assert "error" not in outcome, outcome.get("error")
    assert outcome["applied"] == 0
    assert outcome["version"] == db.SCHEMA_VERSION
    other.close()


def test_connect_waits_for_lock_when_switching_to_wal(tmp_path: Path) -> None:
    # Switching a fresh database to WAL needs an exclusive lock, and SQLite does not apply
    # busy_timeout to it: concurrent first connections used to fail with "database is locked".
    path = tmp_path / "d.db"
    other = sqlite3.connect(path, isolation_level=None)
    other.execute("CREATE TABLE t (x INTEGER)")
    other.execute("BEGIN IMMEDIATE")

    outcome: dict[str, object] = {}

    def connect_ours() -> None:
        try:
            conn = db.connect(path)
            outcome["mode"] = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
        except Exception as e:  # reported via the assertion below
            outcome["error"] = e

    thread = threading.Thread(target=connect_ours)
    thread.start()
    time.sleep(0.3)
    other.execute("COMMIT")
    thread.join(timeout=10)
    other.close()

    assert "error" not in outcome, outcome.get("error")
    assert outcome["mode"] == "wal"


# ── Upgrades and schema drift ────────────────────────────────────────────────


def _db_at_version(path: Path, version: int) -> sqlite3.Connection:
    """A database as an older DESMAN with only the first `version` migrations would leave it."""
    conn = db.connect(path)
    for v, script in enumerate(db.MIGRATIONS[:version], start=1):
        with db.transaction(conn):
            for statement in db._split_sql(script):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {v}")
    return conn


def test_v1_database_upgrades_and_backfills_accepted_attempt(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    old = _db_at_version(path, 1)
    task_id = add_task(old)
    reviewed = add_subtask(old, task_id, 0, status="done")
    by_hand = add_subtask(old, task_id, 1, status="done")
    for subtask_id in (reviewed, by_hand):
        for attempt in (1, 2):
            old.execute(
                "INSERT INTO artifacts (id, subtask_id, attempt, kind) VALUES (?, ?, ?, 'result')",
                (new_id(), subtask_id, attempt),
            )
    old.execute(
        "INSERT INTO audit_log (subtask_id, actor, action, from_status, to_status) "
        "VALUES (?, 'reviewer', 'transition', 'review', 'done')",
        (reviewed,),
    )
    old.executemany(
        "INSERT INTO audit_log (subtask_id, actor, action, from_status, to_status) VALUES (?, ?, 'transition', ?, ?)",
        [
            (by_hand, "executor", "running", "needs_human"),
            (by_hand, "human", "needs_human", "done"),
        ],
    )
    old.close()

    conn = db.open_db(path)  # migrates to the current version and verifies
    accepted = dict(conn.execute("SELECT id, accepted_attempt FROM subtasks").fetchall())
    assert accepted == {reviewed: 2, by_hand: None}
    store = Store(conn)
    assert store.get_subtask(reviewed).accepted_attempt == 2  # rows convert fine after upgrade
    conn.close()


def test_drifted_development_database_is_refused_clearly(tmp_path: Path) -> None:
    path = tmp_path / "drift.db"
    conn = db.open_db(path)
    # Simulate a database from a revision whose migration 1 lacked a column added later.
    conn.execute("ALTER TABLE artifacts DROP COLUMN path")
    conn.close()
    with pytest.raises(db.SchemaMismatchError, match="artifacts: missing path"):
        db.open_db(path)
    result = CliRunner().invoke(cli, ["init-db", "--db", str(path)])
    assert result.exit_code != 0
    assert "development revision" in result.output


def test_current_schema_verifies(conn: sqlite3.Connection) -> None:
    db.verify_schema(conn)
    assert "accepted_attempt" in db.expected_schema()["subtasks"]


def test_v3_database_with_history_upgrades_to_the_call_log(tmp_path: Path) -> None:
    path = tmp_path / "v3.db"
    old = _db_at_version(path, 3)
    task_id = add_task(old)
    sub = add_subtask(old, task_id, 0, status="done")
    old.execute(
        "INSERT INTO audit_log (task_id, subtask_id, actor, action, from_status, to_status) "
        "VALUES (?, ?, 'reviewer', 'verdict', 'review', 'done')",
        (task_id, sub),
    )
    old.close()

    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1  # history kept
    assert conn.execute("SELECT status FROM subtasks").fetchone()[0] == "done"
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0
    conn.close()


def test_v4_database_with_a_call_log_upgrades_to_the_response_cache(tmp_path: Path) -> None:
    path = tmp_path / "v4.db"
    old = _db_at_version(path, 4)
    task_id = add_task(old)
    old.execute(
        "INSERT INTO model_calls (provider, model, prompt_hash, latency_ms) "
        "VALUES ('groq', 'm', 'h', 5)"
    )
    old.close()

    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT id FROM tasks").fetchone()[0] == task_id
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1  # log kept
    assert conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 0
    conn.close()


def test_v5_database_with_tasks_upgrades_to_split_subtasks(tmp_path: Path) -> None:
    path = tmp_path / "v5.db"
    old = _db_at_version(path, 5)
    task_id = add_task(old)
    add_subtask(old, task_id)
    old.close()

    store = Store(db.open_db(path))
    assert db.schema_version(store.conn) == db.SCHEMA_VERSION
    [existing] = store.list_subtasks(task_id)
    assert existing.parent_id is None and store.parts(existing.id) == []
    store.conn.close()


def test_v6_database_with_running_work_upgrades_to_the_effect_journal(tmp_path: Path) -> None:
    path = tmp_path / "v6.db"
    old = _db_at_version(path, 6)
    task_id = add_task(old)
    add_subtask(old, task_id)
    old.close()

    store = Store(db.open_db(path))
    assert db.schema_version(store.conn) == db.SCHEMA_VERSION
    [existing] = store.list_subtasks(task_id)
    assert store.executions(existing.id) == 0 and store.effects(existing.id, 0) == []
    store.conn.close()


def test_v7_database_upgrades_to_the_web_cache(tmp_path: Path) -> None:
    path = tmp_path / "v7.db"
    _db_at_version(path, 7).close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM web_cache").fetchone()[0] == 0
    conn.close()


def test_v8_database_with_tasks_upgrades_to_chat_sessions(tmp_path: Path) -> None:
    path = tmp_path / "v8.db"
    old = _db_at_version(path, 8)
    task_id = add_task(old)
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    sessions = Sessions(conn)
    assert sessions.latest() is None
    message = sessions.add(None, Role.DESMAN, "task created", kind=Kind.EVENT, task_id=task_id)
    assert sessions.messages(message.session_id)[0].task_id == task_id
    conn.close()


def test_v9_database_with_a_session_upgrades_to_tasks_of_sessions(tmp_path: Path) -> None:
    path = tmp_path / "v9.db"
    old = _db_at_version(path, 9)
    task_id = add_task(old)
    session_id = Sessions(old).add(None, Role.USER, "find flights").session_id
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    # Tasks of `desman run` belong to no session.
    assert (
        conn.execute("SELECT session_id FROM tasks WHERE id = ?", (task_id,)).fetchone()[0] is None
    )
    with pytest.raises(sqlite3.IntegrityError):  # only a session that exists
        conn.execute("UPDATE tasks SET session_id = 'nope' WHERE id = ?", (task_id,))
    conn.execute("UPDATE tasks SET session_id = ? WHERE id = ?", (session_id, task_id))
    conn.close()


@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_a_new_database_is_private(tmp_path: Path) -> None:
    old_umask = os.umask(0o022)
    try:
        conn = db.open_db(tmp_path / "data" / "desman.db")
        conn.execute("INSERT INTO tasks (id, goal) VALUES ('t', 'g')")  # creates the WAL
        conn.close()
    finally:
        os.umask(old_umask)
    modes = {p.name: stat.S_IMODE(p.stat().st_mode) for p in (tmp_path / "data").iterdir()}
    assert modes and all(mode == 0o600 for mode in modes.values()), modes
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700


def test_v10_database_with_a_call_log_upgrades_to_attributed_calls(tmp_path: Path) -> None:
    path = tmp_path / "v10.db"
    old = _db_at_version(path, 10)
    task_id = add_task(old)
    old.execute(
        "INSERT INTO model_calls (provider, model, task_id, prompt_hash, latency_ms) "
        "VALUES ('groq', 'm', ?, 'h', 5)",
        (task_id,),
    )
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    row = conn.execute("SELECT task_id, role, session_id, error_type FROM model_calls").fetchone()
    assert tuple(row) == (task_id, None, None, None)  # the old calls stay, unattributed
    with pytest.raises(sqlite3.IntegrityError):  # only the known roles
        conn.execute(
            "INSERT INTO model_calls (provider, model, prompt_hash, latency_ms, role) "
            "VALUES ('groq', 'm', 'h', 5, 'chef')"
        )
    conn.close()


def test_v11_database_with_a_session_upgrades_to_the_operator_log(tmp_path: Path) -> None:
    path = tmp_path / "v11.db"
    old = _db_at_version(path, 11)
    old.execute("INSERT INTO sessions (id) VALUES ('s1')")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    operator.record(conn, "operator_command", {"command": "language"}, session_id="s1")
    [entry] = operator.entries(conn)
    assert (entry.session_id, entry.action, entry.details) == (
        "s1",
        "operator_command",
        {"command": "language"},
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE operator_log SET action = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM operator_log")
    with pytest.raises(sqlite3.IntegrityError):  # a session that exists, or none
        operator.record(conn, "operator_command", session_id="no-such-session")
    conn.close()
