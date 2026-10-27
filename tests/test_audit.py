"""Tests for the audit trail (D7): what is recorded, the completeness check and `desman audit`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.cli import cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.coordinator import Coordinator, Failure, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, AmbiguousIdError, NotFoundError, Store

S = SubtaskStatus


def one_step(description: str = "a") -> TaskSpec:
    return TaskSpec(
        goal="g", subtasks=[SubtaskSpec(description=description, acceptance_criteria="ok")]
    )


def ok(work: Work) -> Success:
    return Success(content={"ok": True})


def reviews(*verdicts: Review | Failure) -> Any:
    queue = iter(verdicts)

    def reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review | Failure:
        return next(queue)

    return reviewer


def actions(store: Store, task_id: str) -> list[str]:
    return [e.action for e in history(store.conn, task_id)]


# ── What is recorded ─────────────────────────────────────────────────────────


def test_reviewer_verdicts_are_recorded_as_verdicts(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(one_step())
    reviewer = reviews(
        Review(verdict=Verdict.RETRY, feedback="cite sources"),
        Failure(kind=FailureKind.TRANSIENT, reason="reviewer timed out"),
        Review(verdict=Verdict.ACCEPT, feedback="good"),
    )
    Coordinator(store, ok, reviewer, sleep=clock.sleep).run(task_id)
    verdicts = [
        (e.details["verdict"], e.details["feedback"], e.to_status)
        for e in history(store.conn, task_id)
        if e.action == "verdict"
    ]
    assert verdicts == [("retry", "cite sources", S.READY), ("accept", "good", S.DONE)]
    # A transient reviewer failure keeps the work in REVIEW (no status change).
    [failed] = [e for e in history(store.conn, task_id) if e.action == "review_interrupted"]
    assert failed.details["failure"] == "transient" and "timed out" in failed.details["reason"]
    assert check_history(store.conn, task_id) == []


def test_escalation_is_a_verdict(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(one_step())
    reviewer = reviews(Review(verdict=Verdict.ESCALATE, feedback="unsure"))
    Coordinator(store, ok, reviewer, sleep=clock.sleep).run(task_id)
    [entry] = [e for e in history(store.conn, task_id) if e.action == "verdict"]
    assert (entry.details["verdict"], entry.to_status) == ("escalate", S.NEEDS_HUMAN)


def test_executor_failures_and_submissions_are_labelled(store: Store, clock: FakeClock) -> None:
    outcomes: list[Success | Failure] = [
        Failure(kind=FailureKind.QUALITY, reason="bad json"),
        Success(content={}),
    ]

    def executor(work: Work) -> Success | Failure:
        return outcomes.pop(0)

    task_id = store.create_task(one_step())
    reviewer = reviews(Review(verdict=Verdict.ACCEPT))
    Coordinator(store, executor, reviewer, sleep=clock.sleep).run(task_id)
    assert actions(store, task_id) == [
        "task_created",
        "dependencies_met",
        "claim",
        "execution_failed",
        "claim",
        "submit",
        "verdict",
    ]


def test_housekeeping_and_cancel_have_their_own_actions(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="g", subtasks=[SubtaskSpec(description=n, acceptance_criteria="ok") for n in "ab"]
        )
    )
    a = store.claim("w")
    assert a is not None
    clock.advance(DEFAULT_LEASE_SECONDS + 1)
    store.recover_expired()
    b = store.claim("w")
    assert b is not None and b.lease_token is not None
    store.record_failure(
        b.id, S.RUNNING, Actor.EXECUTOR, FailureKind.LIMIT, lease=b.lease_token, reason="429"
    )
    clock.advance(3600)
    store.resume_due()
    store.cancel_task(task_id, reason="enough")
    recorded = actions(store, task_id)
    assert {"lease_expired", "pause_over", "cancel", "failure"} <= set(recorded)
    assert check_history(store.conn, task_id) == []


def test_lost_lease_is_recorded_as_discarded_work(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(one_step())

    def cancelled_midway(work: Work) -> Success:
        store.cancel_task(task_id, reason="operator stop")
        work.heartbeat(None)
        return Success(content={})

    Coordinator(store, cancelled_midway, reviews(), sleep=clock.sleep).run(task_id)
    [discarded] = [e for e in history(store.conn, task_id) if e.action == "work_discarded"]
    assert discarded.actor is Actor.EXECUTOR and discarded.to_status is None
    assert "lease lost" in discarded.details["reason"]


def test_lease_lost_before_submit_is_recorded(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(one_step())

    def outlived_lease(work: Work) -> Success:
        clock.advance(DEFAULT_LEASE_SECONDS + 1)
        store.recover_expired()  # someone else's housekeeping took the subtask back
        return Success(content={"late": True})

    loop = Coordinator(store, outlived_lease, reviews(), sleep=clock.sleep)
    loop.tick(task_id, [])
    [discarded] = [e for e in history(store.conn, task_id) if e.action == "work_discarded"]
    assert discarded.details["reason"] == "lease lost before success"
    assert store.artifacts(store.list_subtasks(task_id)[0].id) == []


def test_every_entry_names_its_task(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(one_step())
    Coordinator(store, ok, reviews(Review(verdict=Verdict.ACCEPT)), sleep=clock.sleep).run(task_id)
    rows = store.conn.execute("SELECT task_id FROM audit_log").fetchall()
    assert rows and all(r[0] == task_id for r in rows)


# ── The completeness check ───────────────────────────────────────────────────


def test_unrecorded_status_change_is_detected(store: Store) -> None:
    task_id = store.create_task(one_step())
    assert check_history(store.conn, task_id) == []
    store.conn.execute("UPDATE subtasks SET status = 'done'")  # bypasses the store
    [problem] = check_history(store.conn, task_id)
    assert "is done but its history ends at ready" in problem


def test_gap_in_the_middle_is_detected(store: Store) -> None:
    task_id = store.create_task(one_step())
    [sub] = store.list_subtasks(task_id)
    store.conn.execute("UPDATE subtasks SET status = 'needs_human'")  # unrecorded
    store.transition(sub.id, S.NEEDS_HUMAN, S.DONE, Actor.HUMAN)
    [problem] = check_history(store.conn, task_id)
    assert "left needs_human but history had it ready" in problem


def test_task_without_creation_entry_is_detected(store: Store) -> None:
    store.conn.execute("INSERT INTO tasks (id, goal) VALUES ('raw', 'g')")
    assert check_history(store.conn, "raw") == ["no task_created entry"]


# ── Finding a task ───────────────────────────────────────────────────────────


def test_find_task_by_id_or_unique_prefix(store: Store) -> None:
    for task_id in ("abcd1111", "abcd2222", "ffff0000"):
        store.conn.execute("INSERT INTO tasks (id, goal) VALUES (?, 'g')", (task_id,))
    assert store.find_task("ffff0000")["id"] == "ffff0000"
    assert store.find_task("ffff")["id"] == "ffff0000"
    assert store.find_task("abcd1")["id"] == "abcd1111"
    with pytest.raises(AmbiguousIdError, match="abcd1111, abcd2222"):
        store.find_task("abcd")
    with pytest.raises(NotFoundError, match="at least"):
        store.find_task("fff")
    with pytest.raises(NotFoundError):
        store.find_task("0000")
    with pytest.raises(NotFoundError):
        store.find_task("ab%_")  # no LIKE wildcards


# ── `desman audit` ───────────────────────────────────────────────────────────


@pytest.fixture
def db_file(tmp_path: Path, clock: FakeClock) -> tuple[Path, str]:
    path = tmp_path / "desman.db"
    conn = db.open_db(path)
    store = Store(conn, clock=clock)
    task_id = store.create_task(one_step("draft the\nintroduction"))
    reviewer = reviews(
        Review(verdict=Verdict.RETRY, feedback="x" * 200), Review(verdict=Verdict.ACCEPT)
    )
    Coordinator(store, ok, reviewer, sleep=clock.sleep, workspaces=tmp_path / "ws").run(task_id)
    conn.close()
    return path, task_id


def test_audit_shows_the_history(db_file: tuple[Path, str]) -> None:
    path, task_id = db_file
    result = CliRunner().invoke(cli, ["audit", task_id[:8], "--db", str(path)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"Task {task_id}  done"
    assert lines[1] == "Goal: g"
    body = lines[2:]
    assert len(body) == 8  # one line per entry, even with a newline in the description
    assert "verdict" in body[-1] and "review -> done" in body[-1]
    assert any("verdict=retry" in line and "x" * 79 + "…" in line for line in body)
    assert not any("feedback=  " in line or line.endswith("feedback=") for line in body)


def test_audit_json(db_file: tuple[Path, str]) -> None:
    path, task_id = db_file
    result = CliRunner().invoke(cli, ["audit", task_id, "--db", str(path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["task"] == {"id": task_id, "goal": "g", "status": "done"}
    assert report["problems"] == []
    assert [e["action"] for e in report["entries"]][-1] == "verdict"
    assert report["entries"][-1]["details"]["verdict"] == "accept"


def test_audit_reports_incomplete_history(db_file: tuple[Path, str]) -> None:
    path, task_id = db_file
    conn = db.open_db(path)
    conn.execute("UPDATE subtasks SET status = 'review', accepted_attempt = NULL")
    conn.close()
    result = CliRunner().invoke(cli, ["audit", task_id, "--db", str(path)])
    assert result.exit_code == 1
    assert "History is incomplete" in result.stderr
    assert "is review but its history ends at done" in result.stderr


def test_audit_errors(tmp_path: Path, db_file: tuple[Path, str]) -> None:
    missing = tmp_path / "nope.db"
    result = CliRunner().invoke(cli, ["audit", "abcd", "--db", str(missing)])
    assert result.exit_code != 0 and "no database" in result.output
    assert not missing.exists()  # a read command never creates an empty database
    path, _ = db_file
    result = CliRunner().invoke(cli, ["audit", "zzzz", "--db", str(path)])
    assert result.exit_code != 0 and "not found" in result.output


def test_audit_uses_the_default_database(clock: FakeClock) -> None:
    conn = db.open_db()
    task_id = Store(conn, clock=clock).create_task(one_step())
    conn.close()
    result = CliRunner().invoke(cli, ["audit", task_id])
    assert result.exit_code == 0, result.output
    assert "task_created" in result.output


# ── Review regressions (PR #10) ──────────────────────────────────────────────


def test_history_check_reads_one_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A worker commits between the check's reads: the check must not see half of it.
    path = tmp_path / "live.db"
    reader, writer = db.open_db(path), db.open_db(path)
    task_id = Store(writer).create_task(one_step())
    import desman.core.audit as audit_module

    original = audit_module.history

    def history_then_claim(conn: Any, task: str) -> Any:
        entries = original(conn, task)
        Store(writer).claim("other-worker")  # commits ready -> running and its entry
        return entries

    monkeypatch.setattr(audit_module, "history", history_then_claim)
    assert check_history(reader, task_id) == []
    monkeypatch.setattr(audit_module, "history", original)
    assert check_history(reader, task_id) == []  # and the next check sees the claim
    assert [e.action for e in history(reader, task_id)][-1] == "claim"
    reader.close()
    writer.close()


def test_subtask_added_without_a_record_is_detected(store: Store) -> None:
    task_id = store.create_task(one_step())
    store.conn.execute(
        "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria) "
        "VALUES ('sneaky', ?, 1, 'd', 'c')",
        (task_id,),
    )
    assert check_history(store.conn, task_id) == ["subtask sneaky was never recorded as created"]


def test_legacy_creation_entry_with_a_count_still_replays(store: Store) -> None:
    task_id = "legacy"
    store.conn.execute("INSERT INTO tasks (id, goal) VALUES (?, 'g')", (task_id,))
    store.conn.execute(
        "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria) "
        "VALUES ('s1', ?, 0, 'd', 'c')",
        (task_id,),
    )
    store.conn.execute(
        "INSERT INTO audit_log (task_id, actor, action, details_json) "
        "VALUES (?, 'human', 'task_created', '{\"subtasks\": 1}')",
        (task_id,),
    )
    assert check_history(store.conn, task_id) == []


def test_audit_output_does_not_show_live_lease_tokens(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "desman.db"
    conn = db.open_db(path)
    store = Store(conn, clock=clock)
    task_id = store.create_task(one_step())
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    conn.close()
    for args in ([], ["--json"]):
        result = CliRunner().invoke(cli, ["audit", task_id, "--db", str(path), *args])
        assert result.exit_code == 0, result.output
        assert claimed.lease_token not in result.output
        assert claimed.lease_token[:8] in result.output  # enough to tell leases apart
