"""Tests for executor context: dependency results, budget fitting, feedback."""

from __future__ import annotations

from typing import Any

import pytest

from conftest import FakeClock
from desman.core.context import (
    CHARS_PER_TOKEN,
    DependencyResult,
    _fair_shares,
    _render,
    estimate_tokens,
    fit_dependencies,
)
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store

S = SubtaskStatus


def result(name: str, content: dict[str, Any] | None) -> DependencyResult:
    return DependencyResult(subtask_id=name, description=f"do {name}", content=content)


# ── Budget fitting ───────────────────────────────────────────────────────────


def test_fair_shares_give_small_items_all_they_need() -> None:
    assert _fair_shares([10, 100, 1000], 300) == [10, 100, 190]
    assert _fair_shares([500, 500], 300) == [150, 150]
    assert _fair_shares([1, 2, 3], 1000) == [1, 2, 3]
    assert _fair_shares([], 100) == []
    assert _fair_shares([5], 0) == [0]


@pytest.mark.parametrize("budget", [0, 1, 50, 333, 10_000])
def test_fair_shares_never_exceed_budget(budget: int) -> None:
    sizes = [7, 400, 3, 1200, 90]
    shares = _fair_shares(sizes, budget)
    assert sum(shares) <= budget
    assert all(0 <= share <= size for share, size in zip(shares, sizes, strict=True))


def test_results_that_fit_are_passed_whole() -> None:
    fitted = fit_dependencies([result("a", {"x": 1}), result("b", None)], budget_tokens=100)
    assert [f.content for f in fitted] == [{"x": 1}, None]
    assert not any(f.truncated for f in fitted)
    assert [f.description for f in fitted] == ["do a", "do b"]


def test_large_result_is_truncated_and_small_kept() -> None:
    small = result("small", {"answer": 42})
    large = result("large", {"text": "x" * 10_000})
    budget = 200
    fitted = fit_dependencies([large, small], budget_tokens=budget)
    assert [f.subtask_id for f in fitted] == ["large", "small"]  # order kept
    big, kept = fitted
    assert kept.content == {"answer": 42} and not kept.truncated
    assert big.truncated and isinstance(big.content, str)
    assert big.content.endswith(f"…[truncated, {len(_render(large.content))} chars in full]")
    used = len(big.content) + len(_render(small.content))
    assert used <= budget * CHARS_PER_TOKEN


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcde") == 2


# ── Through the store and the coordinator ────────────────────────────────────


def diamond() -> TaskSpec:
    def sub(name: str, *deps: int) -> SubtaskSpec:
        return SubtaskSpec(description=name, acceptance_criteria="ok", depends_on=list(deps))

    return TaskSpec(goal="g", subtasks=[sub("a"), sub("b", 0), sub("c", 0), sub("d", 2, 1)])


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def test_each_subtask_receives_its_dependencies_results(store: Store, clock: FakeClock) -> None:
    # B4: the old prototype passed the string "TBD" instead of earlier results.
    seen: dict[str, list[tuple[str, Any]]] = {}

    def executor(work: Work) -> Success:
        name = work.subtask.description
        seen[name] = [(d.description, d.content) for d in work.context.dependencies]
        return Success(content={"made_by": name})

    task_id = store.create_task(diamond())
    result = Coordinator(store, executor, accept, sleep=clock.sleep).run(task_id)
    assert result.status is TaskStatus.DONE
    assert seen["a"] == []
    assert seen["b"] == [("a", {"made_by": "a"})]
    assert seen["c"] == [("a", {"made_by": "a"})]
    # Task order, not the order the dependencies were declared in (2, 1).
    assert seen["d"] == [("b", {"made_by": "b"}), ("c", {"made_by": "c"})]


def test_dependency_result_is_the_accepted_attempt(store: Store, clock: FakeClock) -> None:
    attempts = {"a": 0}

    def executor(work: Work) -> Success:
        name = work.subtask.description
        if name == "a":
            attempts["a"] += 1
            return Success(content={"version": attempts["a"]})
        return Success(content={"saw": [d.content for d in work.context.dependencies]})

    def second_try_only(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        if subtask.description == "a" and artifact == {"version": 1}:
            return Review(verdict=Verdict.RETRY, feedback="v1 is wrong")
        return Review(verdict=Verdict.ACCEPT)

    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="a", acceptance_criteria="ok"),
            SubtaskSpec(description="b", acceptance_criteria="ok", depends_on=[0]),
        ],
    )
    task_id = store.create_task(spec)
    Coordinator(store, executor, second_try_only, sleep=clock.sleep).run(task_id)
    b = store.list_subtasks(task_id)[1]
    assert store.latest_artifact(b.id) == {"saw": [{"version": 2}]}


def test_retry_gets_reviewer_feedback_and_attempt_number(store: Store, clock: FakeClock) -> None:
    contexts: list[tuple[int, str | None]] = []

    def executor(work: Work) -> Success:
        fb = work.context.feedback
        contexts.append((work.context.attempt, fb.text if fb else None))
        return Success(content={})

    reviews = iter(
        [
            Review(verdict=Verdict.RETRY, feedback="cite sources"),
            Review(verdict=Verdict.ACCEPT),
        ]
    )

    def reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        return next(reviews)

    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="ok")])
    )
    Coordinator(store, executor, reviewer, sleep=clock.sleep).run(task_id)
    assert contexts == [(1, None), (2, "cite sources")]


def test_operator_guidance_reaches_the_next_attempt(store: Store) -> None:
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="ok")])
    )
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    store.record_failure(
        claimed.id,
        S.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=claimed.lease_token,
        reason="403",
    )
    store.transition(
        claimed.id, S.NEEDS_HUMAN, S.READY, Actor.HUMAN, reason="use the public mirror"
    )
    feedback = store.last_feedback(claimed.id)
    assert feedback is not None
    assert (feedback.source, feedback.text) == ("human", "use the public mirror")
    assert store.last_feedback("nope") is None
    assert task_id


def test_operator_accepting_without_artifact_gives_empty_result(store: Store) -> None:
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="a", acceptance_criteria="ok"),
            SubtaskSpec(description="b", acceptance_criteria="ok", depends_on=[0]),
        ],
    )
    task_id = store.create_task(spec)
    a, b = store.list_subtasks(task_id)
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    store.transition(a.id, S.RUNNING, S.NEEDS_HUMAN, Actor.EXECUTOR, lease=claimed.lease_token)
    store.transition(a.id, S.NEEDS_HUMAN, S.DONE, Actor.HUMAN, reason="done by hand")
    [dep] = store.dependency_results(b.id)
    assert (dep.description, dep.content) == ("a", None)


def test_context_budget_is_applied_by_coordinator(store: Store, clock: FakeClock) -> None:
    truncated: list[bool] = []

    def executor(work: Work) -> Success:
        truncated.extend(d.truncated for d in work.context.dependencies)
        return Success(content={"blob": "y" * 5000} if work.subtask.description == "a" else {})

    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="a", acceptance_criteria="ok"),
            SubtaskSpec(description="b", acceptance_criteria="ok", depends_on=[0]),
        ],
    )
    task_id = store.create_task(spec)
    Coordinator(store, executor, accept, context_budget_tokens=100, sleep=clock.sleep).run(task_id)
    assert truncated == [True]


def test_tiny_budget_is_never_exceeded() -> None:
    results = [result(str(i), {"text": "z" * 500}) for i in range(5)]
    for budget in range(0, 40):
        fitted = fit_dependencies(results, budget_tokens=budget)
        used = sum(len(f.content) if f.truncated else len(_render(f.content)) for f in fitted)
        assert used <= budget * CHARS_PER_TOKEN


def two_step() -> TaskSpec:
    return TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="a", acceptance_criteria="ok"),
            SubtaskSpec(description="b", acceptance_criteria="ok", depends_on=[0]),
        ],
    )


def test_rejected_attempt_is_never_passed_on(store: Store) -> None:
    # Attempt 1 is rejected; attempt 2 escalates from RUNNING; the operator marks it done by hand.
    task_id = store.create_task(two_step())
    a, b = store.list_subtasks(task_id)
    first = store.claim("w")
    assert first is not None and first.lease_token is not None
    store.submit(a.id, first.lease_token, {"wrong": True})
    store.record_failure(a.id, S.REVIEW, Actor.REVIEWER, FailureKind.QUALITY, reason="wrong")
    second = store.claim("w")
    assert second is not None and second.lease_token is not None
    store.transition(a.id, S.RUNNING, S.NEEDS_HUMAN, Actor.EXECUTOR, lease=second.lease_token)
    done = store.transition(a.id, S.NEEDS_HUMAN, S.DONE, Actor.HUMAN, reason="did it myself")
    assert done.accepted_attempt is None
    [dep] = store.dependency_results(b.id)
    assert dep.content is None


def test_operator_accepting_escalated_review_passes_that_artifact(store: Store) -> None:
    task_id = store.create_task(two_step())
    a, b = store.list_subtasks(task_id)
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    store.submit(a.id, claimed.lease_token, {"borderline": True})
    store.transition(a.id, S.REVIEW, S.NEEDS_HUMAN, Actor.REVIEWER, reason="not sure")
    done = store.transition(a.id, S.NEEDS_HUMAN, S.DONE, Actor.HUMAN, reason="it is fine")
    assert done.accepted_attempt == 1
    [dep] = store.dependency_results(b.id)
    assert dep.content == {"borderline": True}
