"""Tests for the operator commands: inbox, retry, accept, fail, cancel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman.cli import cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store

S = SubtaskStatus


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def chain(*names: str) -> TaskSpec:
    return TaskSpec(
        goal="the goal",
        subtasks=[
            SubtaskSpec(description=n, acceptance_criteria="ok", depends_on=[i - 1] if i else [])
            for i, n in enumerate(names)
        ],
    )


def escalated(store: Store, spec: TaskSpec) -> tuple[str, str]:
    """A task whose first subtask waits for the operator after a 401."""
    task_id = store.create_task(spec)
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    store.record_failure(
        claimed.id,
        S.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=claimed.lease_token,
        reason="401 bad key",
    )
    return task_id, claimed.id


@pytest.fixture
def store() -> Any:
    s = Store(db.open_db())
    yield s
    s.conn.close()


def test_inbox_lists_what_waits_for_the_operator(store: Store) -> None:
    task_id, _ = escalated(store, chain("fetch", "analyse"))
    result = desman("inbox")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert (
        lines[0].startswith(f"{task_id[:8]}/0 ")
        and "needs_human" in lines[0]
        and "fetch" in lines[0]
    )
    assert "401 bad key" in lines[1]
    [item] = json.loads(desman("inbox", "--json").output)
    assert (item["ref"], item["status"], item["goal"]) == (
        f"{task_id[:8]}/0",
        "needs_human",
        "the goal",
    )


def test_inbox_when_nothing_waits(store: Store) -> None:
    store.create_task(chain("a"))
    assert desman("inbox").output.strip() == "Nothing is waiting for you."
    assert json.loads(desman("inbox", "--json").output) == []


def test_retry_with_a_note_reaches_the_next_attempt(store: Store, tmp_path: Path) -> None:
    task_id, _ = escalated(store, chain("fetch"))
    result = desman("retry", f"{task_id[:8]}/0", "--note", "use the public mirror")
    assert result.exit_code == 0, result.output
    assert (
        "needs_human -> ready" in result.output and f"desman resume {task_id[:8]}" in result.output
    )

    script = tmp_path / "s.json"
    script.write_text(
        json.dumps(
            {
                "routes": [
                    {"match": "Execute", "steps": ['{"ok": 1}']},
                    {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
                ]
            }
        )
    )
    resumed = desman("resume", task_id, "--fake-llm", script)
    assert resumed.exit_code == 0, resumed.output
    assert store.list_subtasks(task_id)[0].status is S.DONE
    [entry] = [e for e in history(store.conn, task_id) if e.action == "operator_retry"]
    assert entry.actor is Actor.HUMAN and entry.details["reason"] == "use the public mirror"
    assert check_history(store.conn, task_id) == []


def test_note_is_the_feedback_of_the_next_attempt(store: Store) -> None:
    task_id, sub_id = escalated(store, chain("fetch"))
    desman("retry", sub_id[:10], "--note", "use the public mirror")  # by subtask id prefix
    feedback = store.last_feedback(sub_id)
    assert feedback is not None and (feedback.source, feedback.text) == (
        "human",
        "use the public mirror",
    )
    assert store.get_subtask(sub_id).attempts.quality == 0  # a fresh budget


def test_accept_unblocks_the_next_step(store: Store) -> None:
    task_id, _ = escalated(store, chain("fetch", "analyse"))
    result = desman("accept", f"{task_id}/0", "--json")
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert (out["from"], out["to"], out["task_status"]) == ("needs_human", "done", "active")
    assert [s.status for s in store.list_subtasks(task_id)] == [S.DONE, S.READY]


def test_fail_blocks_dependents_and_retry_revives_them(store: Store) -> None:
    task_id, _ = escalated(store, chain("fetch", "analyse", "report"))
    assert desman("fail", f"{task_id[:8]}/0").exit_code != 0  # --note is required
    failed = desman("fail", f"{task_id[:8]}/0", "--note", "source is gone")
    assert failed.exit_code == 0, failed.output
    assert "Task is failed." in failed.output
    assert [s.status for s in store.list_subtasks(task_id)] == [S.FAILED, S.BLOCKED, S.BLOCKED]
    assert [i["status"] for i in json.loads(desman("inbox", "--json").output)] == ["failed"]

    revived = desman("retry", f"{task_id[:8]}/0")
    assert revived.exit_code == 0, revived.output
    assert [s.status for s in store.list_subtasks(task_id)] == [S.READY, S.PENDING, S.PENDING]
    assert check_history(store.conn, task_id) == []


def test_decisions_that_are_not_the_operators_to_make(store: Store) -> None:
    task_id = store.create_task(chain("a", "b"))
    result = desman("accept", f"{task_id[:8]}/0")  # ready: the system's to run, not ours to accept
    assert result.exit_code == 1
    assert "[0] is ready: the operator can move it to cancelled, not done" in result.output
    pending = desman("retry", f"{task_id[:8]}/1")
    assert pending.exit_code == 1 and "is pending" in pending.output
    assert [s.status for s in store.list_subtasks(task_id)] == [S.READY, S.PENDING]


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("TASK/9", "has no subtask 9"),
        ("TASK/x", "expected <task>/<number>"),
        ("zzzzzz/0", "task 'zzzzzz' not found"),
        ("zzzzzz", "subtask 'zzzzzz' not found"),
        ("ab", "at least 4"),
    ],
)
def test_bad_subtask_references(store: Store, ref: str, message: str) -> None:
    task_id = store.create_task(chain("a"))
    result = desman("retry", ref.replace("TASK", task_id[:8]))
    assert result.exit_code == 1
    assert message in result.output


def test_cancel(store: Store) -> None:
    task_id, _ = escalated(store, chain("a", "b"))
    result = desman("cancel", task_id[:8], "--note", "not needed")
    assert result.exit_code == 0, result.output
    assert "2 subtask(s) cancelled, task is cancelled" in result.output
    assert store.task_status(task_id) is TaskStatus.CANCELLED
    again = json.loads(desman("cancel", task_id[:8], "--json").output)
    assert again["cancelled"] == [] and again["status"] == "cancelled"
