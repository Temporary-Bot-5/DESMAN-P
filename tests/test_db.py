"""Tests for the database schema and migrations."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from desman.cli import cli
from desman.core import db
from desman.core.states import SubtaskStatus


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
    # Migration 1 committed, migration 2 rolled back completely.
    assert db.schema_version(conn) == 1
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
    lease: dict[str, object] = {"lease_owner": "w", "lease_until": db.now_iso()}
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


def test_paused_requires_resume_status(conn: sqlite3.Connection) -> None:
    task_id = add_task(conn)
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="paused")
    with pytest.raises(sqlite3.IntegrityError):
        add_subtask(conn, task_id, status="paused", resume_status="running")


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
    conn.execute("INSERT INTO audit_log (actor, action) VALUES ('human', 'test')")
    conn.execute("INSERT INTO audit_log (actor, action) VALUES ('human', 'test2')")
    seqs = [r["seq"] for r in conn.execute("SELECT seq FROM audit_log ORDER BY seq")]
    assert seqs == [1, 2]
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE audit_log SET action = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM audit_log")


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
    assert f"({db.SCHEMA_VERSION} migration(s) applied)" in first.output
    assert (tmp_path / "data" / "desman.db").exists()
    second = runner.invoke(cli, ["init-db"])
    assert "(0 migration(s) applied)" in second.output


def test_init_db_cli_refuses_newer_schema(tmp_path: Path) -> None:
    path = tmp_path / "x.db"
    conn = db.connect(path)
    conn.execute("PRAGMA user_version = 999")
    conn.close()
    result = CliRunner().invoke(cli, ["init-db", "--db", str(path)])
    assert result.exit_code != 0
    assert "newer" in result.output
