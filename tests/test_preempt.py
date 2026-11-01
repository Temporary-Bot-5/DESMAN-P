"""Cooperative preemption (ROADMAP 4.4, D5): critical work that appears while a worker runs
something less urgent is taken over at the worker's next step; the dropped work resumes later
from its checkpoint, without repeating its tool calls and without spending an attempt."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from conftest import FakeClock
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Priority, Subtask, SubtaskSpec, TaskSpec, Tools
from desman.core.policy import Policy, Role, RolePolicy
from desman.core.retry import Attempts
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import PreemptedError, StaleStateError, Store
from desman.core.tools import WRITE_FILE, Access, Registry, ToolCall

S = SubtaskStatus
A = Actor


def task(store: Store, name: str, priority: int = Priority.NORMAL) -> str:
    return store.create_task(
        TaskSpec(
            goal=name,
            priority=priority,
            subtasks=[SubtaskSpec(description=name, acceptance_criteria="ok")],
            tools=Tools.GRANTED,
        )
    )


def only(store: Store, task_id: str) -> Subtask:
    [subtask] = store.list_subtasks(task_id)
    return subtask


def claim(store: Store, owner: str = "w1", task_id: str | None = None) -> Subtask:
    claimed = store.claim(owner, task_id=task_id)
    assert claimed is not None and claimed.lease_token is not None
    return claimed


def actions(store: Store, task_id: str) -> list[str]:
    return [row["action"] for row in store.audit_log(task_id)]


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


# ── Store.preempt ────────────────────────────────────────────────────────────


def test_critical_work_takes_over_and_the_dropped_work_keeps_its_progress(store: Store) -> None:
    low = task(store, "low", Priority.LOW)
    running = claim(store)
    assert running.lease_token is not None
    store.heartbeat(running.id, running.lease_token, checkpoint={"step": 2})
    urgent = task(store, "urgent", Priority.CRITICAL)

    claimed = store.preempt(running.id, running.lease_token, "w1")

    assert claimed is not None and claimed.task_id == urgent
    assert claimed.status is S.RUNNING and claimed.lease_owner == "w1"
    dropped = only(store, low)
    assert dropped.status is S.READY and dropped.lease_token is None
    assert dropped.checkpoint == {"step": 2}
    assert dropped.attempts == Attempts()  # giving way is not a failure
    [entry] = [r for r in store.audit_log(low) if r["action"] == "preempted"]
    assert (entry["from_status"], entry["to_status"], entry["actor"]) == (
        "running",
        "ready",
        "executor",
    )
    assert json.loads(entry["details_json"])["by"] == claimed.id
    [claim_entry] = [r for r in store.audit_log(urgent) if r["action"] == "claim"]
    assert json.loads(claim_entry["details_json"])["preempting"] == running.id
    assert check_history(store.conn, low) == [] and check_history(store.conn, urgent) == []


def test_nothing_to_yield_to(store: Store) -> None:
    task(store, "low", Priority.LOW)
    running = claim(store)
    assert running.lease_token is not None
    task(store, "high", Priority.HIGH)  # urgent, but not critical: it waits its turn
    assert store.preempt(running.id, running.lease_token, "w1") is None
    assert store.get_subtask(running.id).status is S.RUNNING


def test_critical_work_is_never_preempted(store: Store) -> None:
    task(store, "first", Priority.CRITICAL)
    running = claim(store)
    assert running.lease_token is not None
    task(store, "second", Priority.CRITICAL)
    assert store.preempt(running.id, running.lease_token, "w1") is None
    assert store.get_subtask(running.id).status is S.RUNNING


def test_only_critical_work_in_scope_counts(store: Store) -> None:
    mine = task(store, "mine", Priority.LOW)
    running = claim(store, task_id=mine)
    assert running.lease_token is not None
    task(store, "elsewhere", Priority.CRITICAL)
    assert store.preempt(running.id, running.lease_token, "w1", task_id=mine) is None
    assert store.preempt(running.id, running.lease_token, "w1") is not None


def test_one_executor_yields_per_critical_subtask(store: Store) -> None:
    task(store, "a", Priority.LOW)
    task(store, "b", Priority.LOW)
    first, second = claim(store, "w1"), claim(store, "w2")
    assert first.lease_token is not None and second.lease_token is not None
    task(store, "urgent", Priority.CRITICAL)
    assert store.preempt(first.id, first.lease_token, "w1") is not None
    assert store.preempt(second.id, second.lease_token, "w2") is None
    assert store.get_subtask(second.id).status is S.RUNNING


def test_a_lost_lease_cannot_yield(store: Store, clock: FakeClock) -> None:
    task(store, "low", Priority.LOW)
    running = claim(store)
    assert running.lease_token is not None
    task(store, "urgent", Priority.CRITICAL)
    store.recover_expired(abandoned=lambda owner: True)
    with pytest.raises(StaleStateError):
        store.preempt(running.id, running.lease_token, "w1")
    assert only(store, store.get_subtask(running.id).task_id).status is S.READY


# ── The coordinator ──────────────────────────────────────────────────────────


def test_a_worker_gives_way_to_critical_work_and_comes_back(store: Store, clock: FakeClock) -> None:
    """D5: a critical task appears while low work runs; the worker saves its checkpoint, does
    the critical work, then finishes the low work from that checkpoint."""
    low = task(store, "low", Priority.LOW)
    seen: list[tuple[str, dict[str, Any] | None]] = []
    urgent: list[str] = []

    def executor(work: Work) -> Success:
        name = work.subtask.description
        seen.append((name, work.context.checkpoint))
        if name == "low" and not urgent:
            work.heartbeat({"step": 1})
            urgent.append(task(store, "urgent", Priority.CRITICAL))  # the operator adds it now
            work.heartbeat({"step": 2})  # the next step: given up here, after the save
            raise AssertionError("not reached")
        return Success(content={"by": name})

    worker = Coordinator(store, executor, accept, owner="w1", sleep=clock.sleep)
    result = worker.work(until_idle=True)

    assert seen == [("low", None), ("urgent", None), ("low", {"step": 2})]
    assert store.task_status(low) is TaskStatus.DONE
    assert store.task_status(urgent[0]) is TaskStatus.DONE
    assert result.preempted == 1 and worker.preempted == 1
    assert only(store, low).attempts == Attempts()
    assert "preempted" in actions(store, low)
    assert "work_discarded" not in actions(store, low)
    assert check_history(store.conn, low) == [] and check_history(store.conn, urgent[0]) == []


def test_a_tool_call_is_a_step_and_done_calls_are_not_repeated(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    """Preempted before its second call, the rerun replays the first one (the journal, D6) and
    the second runs once, after the critical work."""
    low = task(store, "low", Priority.LOW)
    written: list[str] = []
    urgent: list[str] = []

    def executor(work: Work) -> Success:
        name = work.subtask.description
        if name == "low":
            first = work.tools.call(ToolCall(tool="write_file", args={"path": "a", "content": "1"}))
            assert first.ok
            if not urgent:
                urgent.append(task(store, "urgent", Priority.CRITICAL))
            second = work.tools.call(
                ToolCall(tool="write_file", args={"path": "b", "content": "2"})
            )
            assert second.ok
        written.append(name)
        return Success(content={"by": name})

    access = Access(Policy(roles={Role.EXECUTOR: RolePolicy(tools=["write_file"])}), Role.EXECUTOR)
    worker = Coordinator(
        store,
        executor,
        accept,
        owner="w1",
        sleep=clock.sleep,
        workspaces=tmp_path,
        tools=Registry([WRITE_FILE]),
        access=access,
    )
    worker.work(until_idle=True)

    assert written == ["urgent", "low"]
    tool_entries = [r["action"] for r in store.audit_log(low) if r["action"].startswith("tool_")]
    assert tool_entries == ["tool_called", "tool_replayed", "tool_called"]
    assert (tmp_path / low / "b").read_text() == "2"
    assert store.task_status(low) is TaskStatus.DONE


def test_an_executor_that_ignores_the_preemption_has_its_work_dropped(
    store: Store, clock: FakeClock
) -> None:
    low = task(store, "low", Priority.LOW)
    runs: list[str] = []

    def executor(work: Work) -> Success:
        name = work.subtask.description
        runs.append(name)
        if name == "low" and len(runs) == 1:
            task(store, "urgent", Priority.CRITICAL)
            with contextlib.suppress(PreemptedError):  # carries on regardless
                work.heartbeat(None)
        return Success(content={"by": name})

    worker = Coordinator(store, executor, accept, owner="w1", sleep=clock.sleep)
    worker.work(until_idle=True)

    assert runs == ["low", "urgent", "low"]  # the critical work still went first
    assert "work_discarded" in actions(store, low)
    assert store.task_status(low) is TaskStatus.DONE
    assert check_history(store.conn, low) == []


def test_a_run_of_one_task_gives_way_only_within_it(store: Store, clock: FakeClock) -> None:
    """`desman run`/the chat drive their own task: critical work of another task is for the
    workers, not a reason to drop this one."""
    mine = task(store, "mine", Priority.LOW)
    other: list[str] = []

    def executor(work: Work) -> Success:
        if not other:
            other.append(task(store, "other", Priority.CRITICAL))
        work.heartbeat(None)
        return Success(content={"by": work.subtask.description})

    coordinator = Coordinator(store, executor, accept, owner="run", sleep=clock.sleep)
    assert coordinator.run(mine).status is TaskStatus.DONE
    assert coordinator.preempted == 0
    assert store.task_status(other[0]) is TaskStatus.ACTIVE
