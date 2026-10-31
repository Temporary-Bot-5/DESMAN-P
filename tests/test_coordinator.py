"""Tests for the coordinator loop: it finishes work, recovers, and always stops."""

from __future__ import annotations

import random
from datetime import timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import FakeClock
from desman.core import db
from desman.core.audit import check_history
from desman.core.coordinator import (
    Coordinator,
    Executor,
    Failure,
    Review,
    Reviewer,
    Success,
    Verdict,
    Work,
)
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import Attempts, FailureKind, RetryPolicy
from desman.core.states import Actor, SubtaskStatus, TaskStatus, allowed_targets
from desman.core.store import DEFAULT_LEASE_SECONDS, Store
from desman.llm.base import GenerationFailed

S = SubtaskStatus
K = FailureKind


def chain(*names: str) -> TaskSpec:
    return TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description=n, acceptance_criteria="ok", depends_on=[i - 1] if i else [])
            for i, n in enumerate(names)
        ],
    )


def ok_executor(work: Work) -> Success:
    return Success(content={"by": work.subtask.description})


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def coordinator(
    store: Store,
    clock: FakeClock,
    executor: Executor = ok_executor,
    reviewer: Reviewer = accept,
    **kwargs: Any,
) -> Coordinator:
    return Coordinator(store, executor, reviewer, sleep=clock.sleep, **kwargs)


def test_runs_chain_to_completion(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(chain("a", "b", "c"))
    result = coordinator(store, clock).run(task_id)
    assert result.status is TaskStatus.DONE
    subtasks = store.list_subtasks(task_id)
    assert {s.status for s in subtasks} == {S.DONE}
    assert [store.latest_artifact(s.id) for s in subtasks] == [
        {"by": "a"},
        {"by": "b"},
        {"by": "c"},
    ]


def test_reviewer_sees_submitted_artifact_and_retry_feedback_is_audited(
    store: Store, clock: FakeClock
) -> None:
    seen: list[dict[str, Any] | None] = []

    def picky(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        seen.append(artifact)
        return (
            Review(verdict=Verdict.RETRY, feedback="too short")
            if len(seen) == 1
            else Review(verdict=Verdict.ACCEPT)
        )

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, reviewer=picky).run(task_id).status is TaskStatus.DONE
    assert seen == [{"by": "a"}, {"by": "a"}]
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts == Attempts(quality=1)
    reasons = [r["details_json"] for r in store.audit_log(task_id) if r["actor"] == "reviewer"]
    assert any("too short" in r for r in reasons)


def test_a_reply_the_provider_refused_is_tried_again_not_escalated(
    store: Store, clock: FakeClock
) -> None:
    """Groq's `tool_use_failed` (a model calling a tool natively) is the reply's fault, not the
    request's: the next attempt goes on, told why, instead of waiting for a person."""
    seen: list[Work] = []

    def natively_once(work: Work) -> Success:
        seen.append(work)
        if len(seen) == 1:
            raise GenerationFailed("groq", "Tool choice is none, but model called a tool")
        return ok_executor(work)

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, executor=natively_once).run(task_id).status is TaskStatus.DONE
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts == Attempts(quality=1)
    feedback = seen[1].context.feedback
    assert feedback is not None and "called a tool natively" in str(feedback)


def test_old_prototype_hang_now_ends_with_operator(store: Store, clock: FakeClock) -> None:
    # B1: the first subtask never passes review. The old loop waited forever on its dependents.
    def never(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        return Review(verdict=Verdict.RETRY, feedback="still wrong")

    task_id = store.create_task(chain("a", "b", "c"))
    result = coordinator(store, clock, reviewer=never).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    a, b, c = store.list_subtasks(task_id)
    assert (a.status, b.status, c.status) == (S.NEEDS_HUMAN, S.PENDING, S.PENDING)

    store.transition(a.id, S.NEEDS_HUMAN, S.FAILED, Actor.HUMAN, reason="give up")
    assert coordinator(store, clock, reviewer=never).run(task_id).status is TaskStatus.FAILED
    assert {s.status for s in store.list_subtasks(task_id)[1:]} == {S.BLOCKED}


def test_independent_work_continues_while_one_branch_waits_for_human(
    store: Store, clock: FakeClock
) -> None:
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="bad", acceptance_criteria="ok"),
            SubtaskSpec(description="good", acceptance_criteria="ok"),
        ],
    )

    def judge(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        bad = subtask.description == "bad"
        return (
            Review(verdict=Verdict.ESCALATE, feedback="unclear goal")
            if bad
            else Review(verdict=Verdict.ACCEPT)
        )

    task_id = store.create_task(spec)
    result = coordinator(store, clock, reviewer=judge).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    assert [s.status for s in store.list_subtasks(task_id)] == [S.NEEDS_HUMAN, S.DONE]


def test_crashing_executor_is_bounded(store: Store, clock: FakeClock) -> None:
    def crash(work: Work) -> Success:
        raise RuntimeError("segfault, sort of")

    task_id = store.create_task(chain("a"))
    result = coordinator(store, clock, crash, policy=RetryPolicy(max_transient=3)).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts == Attempts(transient=3)
    assert any(
        "executor raised RuntimeError" in r["details_json"] for r in store.audit_log(task_id)
    )


def test_failing_reviewer_is_bounded(store: Store, clock: FakeClock) -> None:
    def broken(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        raise ConnectionError("provider down")

    task_id = store.create_task(chain("a"))
    result = coordinator(store, clock, reviewer=broken, policy=RetryPolicy(max_transient=2)).run(
        task_id
    )
    assert result.status is TaskStatus.NEEDS_HUMAN


def test_permanent_failure_goes_straight_to_human(store: Store, clock: FakeClock) -> None:
    calls = []

    def unauthorized(work: Work) -> Failure:
        calls.append(1)
        return Failure(kind=K.PERMANENT, reason="401 invalid API key")

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, unauthorized).run(task_id).status is TaskStatus.NEEDS_HUMAN
    assert len(calls) == 1


def limited_once(clock: FakeClock, minutes: int = 10) -> Executor:
    state = {"hit": False}

    def run(work: Work) -> Success | Failure:
        if not state["hit"]:
            state["hit"] = True
            return Failure(
                kind=K.LIMIT,
                reason="429",
                paused_until=db.to_iso(clock.now + timedelta(minutes=minutes)),
            )
        return ok_executor(work)

    return run


def test_pause_returns_control_and_resumes_later(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(chain("a"))
    co = coordinator(store, clock, limited_once(clock))
    first = co.run(task_id)
    assert first.status is TaskStatus.ACTIVE
    assert first.paused_until == db.to_iso(clock.now + timedelta(minutes=10))
    assert clock.slept == 0
    clock.advance(10 * 60)
    assert co.run(task_id).status is TaskStatus.DONE
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts == Attempts()  # the pause spent nothing


def test_pause_can_be_waited_out(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(chain("a"))
    result = coordinator(store, clock, limited_once(clock)).run(task_id, wait_for_pauses=True)
    assert result.status is TaskStatus.DONE
    assert clock.slept == pytest.approx(600)


def test_endless_quota_trouble_does_not_hang(store: Store, clock: FakeClock) -> None:
    def always_limited(work: Work) -> Failure:
        return Failure(kind=K.LIMIT, reason="429")

    task_id = store.create_task(chain("a"))
    co = coordinator(store, clock, always_limited)
    for _ in range(20):
        result = co.run(task_id)
        assert result.status is TaskStatus.ACTIVE and result.paused_until is not None
        clock.advance(3600)
    # Never escalated for quota trouble alone.
    assert store.list_subtasks(task_id)[0].status is S.PAUSED


def test_resumes_after_own_crash_from_checkpoint(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(chain("a"))
    # A previous coordinator process claimed the subtask, saved a checkpoint and died.
    previous = store.claim("coordinator-old")
    assert previous is not None and previous.lease_token is not None
    store.heartbeat(previous.id, previous.lease_token, checkpoint={"step": 2})

    resumed_from: list[dict[str, Any] | None] = []

    def resume(work: Work) -> Success:
        resumed_from.append(work.subtask.checkpoint)
        return Success(content={"done": True})

    result = coordinator(store, clock, resume).run(task_id)
    assert result.status is TaskStatus.DONE
    assert resumed_from == [{"step": 2}]
    assert clock.slept == pytest.approx(DEFAULT_LEASE_SECONDS, abs=0.01)  # waited out the lease


def test_executor_checkpoints_through_heartbeat(store: Store, clock: FakeClock) -> None:
    def stepwise(work: Work) -> Success:
        for step in range(3):
            work.heartbeat({"step": step})
        return Success(content={})

    task_id = store.create_task(chain("a"))
    coordinator(store, clock, stepwise).run(task_id)
    saved = [r for r in store.audit_log(task_id) if r["action"] == "checkpoint"]
    assert len(saved) == 3


def test_cancel_while_running_stops_cleanly(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(chain("a", "b"))

    def cancelled_midway(work: Work) -> Success:
        store.cancel_task(task_id, reason="operator stop")
        work.heartbeat(None)  # raises: the lease is gone
        return Success(content={})

    result = coordinator(store, clock, cancelled_midway).run(task_id)
    assert result.status is TaskStatus.CANCELLED
    assert "lost lease" in result.events[-1]


# ── Termination under arbitrary behaviour ────────────────────────────────────


def random_task(rng: random.Random) -> TaskSpec:
    count = rng.randint(1, 6)
    return TaskSpec(
        goal="random",
        subtasks=[
            SubtaskSpec(
                description=f"s{i}",
                acceptance_criteria="ok",
                depends_on=rng.sample(range(i), rng.randint(0, min(i, 2))),
            )
            for i in range(count)
        ],
    )


@pytest.mark.parametrize("seed", range(200))
def test_loop_always_terminates(store: Store, clock: FakeClock, seed: int) -> None:
    rng = random.Random(seed)

    def chaotic_executor(work: Work) -> Success | Failure:
        roll = rng.random()
        if roll < 0.1:
            raise RuntimeError("boom")
        if roll < 0.2:
            return Failure(kind=K.TRANSIENT, reason="timeout")
        if roll < 0.25:
            return Failure(kind=K.PERMANENT, reason="403")
        if roll < 0.3:
            return Failure(kind=K.LIMIT, reason="429")
        return Success(content={"roll": roll})

    def chaotic_reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        roll = rng.random()
        if roll < 0.1:
            raise RuntimeError("reviewer boom")
        if roll < 0.15:
            return Review(verdict=Verdict.ESCALATE)
        if roll < 0.45:
            return Review(verdict=Verdict.RETRY)
        return Review(verdict=Verdict.ACCEPT)

    task_id = store.create_task(random_task(rng))
    loop = coordinator(store, clock, chaotic_executor, chaotic_reviewer)
    result = loop.run(task_id, wait_for_pauses=True, max_ticks=500)
    assert result.status is not TaskStatus.ACTIVE
    assert check_history(store.conn, task_id) == []  # D7: the log explains the outcome

    # The operator steps in, at random, until the task is finished one way or another.
    for _ in range(20):
        if result.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            break
        if rng.random() < 0.05:
            store.cancel_task(task_id, reason="operator gives up")
        for sub in store.list_subtasks(task_id):
            if sub.status in (S.NEEDS_HUMAN, S.FAILED) and rng.random() < 0.8:
                target = rng.choice(sorted(allowed_targets(sub.status, Actor.HUMAN)))
                store.transition(sub.id, sub.status, target, Actor.HUMAN, reason="operator")
        result = loop.run(task_id, wait_for_pauses=True, max_ticks=500)
        assert check_history(store.conn, task_id) == []


# ── Review fixes (PR #6) ─────────────────────────────────────────────────────


def test_run_does_not_touch_other_tasks(store: Store, clock: FakeClock) -> None:
    other = store.create_task(chain("other"))
    stuck = store.claim("other-worker", task_id=other)
    assert stuck is not None
    mine = store.create_task(chain("mine"))
    clock.advance(DEFAULT_LEASE_SECONDS + 1)  # the other task's lease has expired meanwhile
    result = coordinator(store, clock, policy=RetryPolicy(max_transient=1)).run(mine)
    assert result.status is TaskStatus.DONE
    untouched = store.get_subtask(stuck.id)
    # Not recovered, and certainly not escalated under this run's tiny budget.
    assert (untouched.status, untouched.attempts) == (S.RUNNING, Attempts())


def test_reviewer_outage_does_not_rerun_the_work(store: Store, clock: FakeClock) -> None:
    executions = []

    def counting(work: Work) -> Success:
        executions.append(work.subtask.id)
        return Success(content={"n": len(executions)})

    reviews = []

    def flaky(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        reviews.append(artifact)
        if len(reviews) < 3:
            raise TimeoutError("reviewer API timed out")
        return Review(verdict=Verdict.ACCEPT)

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, counting, flaky).run(task_id).status is TaskStatus.DONE
    assert len(executions) == 1
    assert reviews == [{"n": 1}] * 3  # the same artifact reviewed three times
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts == Attempts(transient=2)
    assert len(store.artifacts(subtask.id)) == 1


def test_verdict_from_parsed_text_is_validated(store: Store, clock: FakeClock) -> None:
    def stringly(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        return Review.model_validate({"verdict": "accept"})

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, reviewer=stringly).run(task_id).status is TaskStatus.DONE


def test_reviewer_may_return_plain_mapping(store: Store, clock: FakeClock) -> None:
    def raw(subtask: Subtask, artifact: dict[str, Any] | None) -> Any:
        return {"verdict": "accept", "feedback": "fine"}

    task_id = store.create_task(chain("a"))
    assert coordinator(store, clock, reviewer=raw).run(task_id).status is TaskStatus.DONE


def test_unknown_verdict_is_a_reviewer_failure_not_an_escalation(
    store: Store, clock: FakeClock
) -> None:
    def confused(subtask: Subtask, artifact: dict[str, Any] | None) -> Any:
        return {"verdict": "maybe"}

    task_id = store.create_task(chain("a"))
    co = coordinator(store, clock, reviewer=confused, policy=RetryPolicy(max_transient=2))
    assert co.run(task_id).status is TaskStatus.NEEDS_HUMAN
    reasons = [r["details_json"] for r in store.audit_log(task_id) if r["actor"] == "reviewer"]
    assert any("ValidationError" in r for r in reasons)
    assert not any("escalated by reviewer" in r for r in reasons)


def test_executor_returning_garbage_is_a_transient_failure(store: Store, clock: FakeClock) -> None:
    def sloppy(work: Work) -> Any:
        return {"content": "not wrapped"}

    task_id = store.create_task(chain("a"))
    co = coordinator(store, clock, sloppy, policy=RetryPolicy(max_transient=1))
    assert co.run(task_id).status is TaskStatus.NEEDS_HUMAN
    assert any("returned dict" in r["details_json"] for r in store.audit_log(task_id))


def test_provider_deadline_is_validated_and_normalized() -> None:
    normalized = Failure(kind=K.LIMIT, reason="429", paused_until="2026-01-01T14:00:00+02:00")
    assert normalized.paused_until == "2026-01-01T12:00:00.000Z"
    with pytest.raises(ValidationError):
        Failure(kind=K.LIMIT, reason="429", paused_until="tomorrow-ish")
    with pytest.raises(ValidationError, match="timezone"):
        Failure(kind=K.LIMIT, reason="429", paused_until="2026-01-01T12:00:00")


def test_bad_provider_deadline_does_not_crash_the_loop(store: Store, clock: FakeClock) -> None:
    def bad_deadline(work: Work) -> Failure:
        return Failure(kind=K.LIMIT, reason="429", paused_until="soon")  # raises inside executor

    task_id = store.create_task(chain("a"))
    co = coordinator(store, clock, bad_deadline, policy=RetryPolicy(max_transient=2))
    assert co.run(task_id).status is TaskStatus.NEEDS_HUMAN


# ── Error reports (ROADMAP 2.8) ──────────────────────────────────────────────


def test_an_error_report_is_a_failed_attempt_never_a_result(store: Store, clock: FakeClock) -> None:
    reviewed: list[dict[str, Any] | None] = []

    def gives_up(work: Work) -> Success:
        return Success(content={"type": "error", "reason": "no access to the data"})

    def records(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        reviewed.append(artifact)
        return Review(verdict=Verdict.ACCEPT)

    task_id = store.create_task(chain("a"))
    result = coordinator(store, clock, executor=gives_up, reviewer=records).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN  # the quality budget ran out
    [sub] = store.list_subtasks(task_id)
    assert reviewed == [] and store.latest_artifact(sub.id) is None  # nothing to accept
    reasons = [r["details_json"] for r in store.audit_log(task_id) if r["actor"] == "executor"]
    assert any("executor reported an error: no access to the data" in r for r in reasons)
    assert check_history(store.conn, task_id) == []


def test_an_accepted_error_report_is_overruled(store: Store, clock: FakeClock) -> None:
    # However it got stored, an error artifact is sent back even if the reviewer accepts it.
    task_id = store.create_task(chain("a"))
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    store.submit(claimed.id, claimed.lease_token, {"type": "error", "message": "gave up"})
    calls = 0

    def fixed(work: Work) -> Success:
        nonlocal calls
        calls += 1
        assert work.context.feedback is not None
        assert "gave up" in work.context.feedback.text and "overruled" in work.context.feedback.text
        return Success(content={"ok": 1})

    assert coordinator(store, clock, executor=fixed).run(task_id).status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert calls == 1 and sub.attempts == Attempts(quality=1)
    assert store.latest_artifact(sub.id) == {"ok": 1}
