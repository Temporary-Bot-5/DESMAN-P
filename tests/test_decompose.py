"""Tests for splitting a subtask whose retries keep failing (ROADMAP 2.9b)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import FakeClock
from desman.agents import EXECUTOR_SYSTEM_COMBINE, llm_executor, llm_reviewer
from desman.core.audit import check_history, history
from desman.core.coordinator import Coordinator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.retry import RetryPolicy
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import StaleStateError, Store
from desman.llm.fake import FakeLLM, garbage
from desman.planner import MAX_PARTS, SPLIT, decomposer

S = SubtaskStatus
REJECT = '{"verdict": "retry", "feedback": "1. too vague"}'
ACCEPT = '{"verdict": "accept"}'


def part(description: str, *depends_on: int) -> SubtaskSpec:
    return SubtaskSpec(
        description=description, acceptance_criteria="ok", depends_on=list(depends_on)
    )


def split_reply(*descriptions: str) -> str:
    subtasks = [{"description": d, "acceptance_criteria": "ok"} for d in descriptions]
    return json.dumps({"subtasks": subtasks})


# ── Store.decompose ──────────────────────────────────────────────────────────


def two_steps(store: Store) -> tuple[str, str]:
    """A task "base" -> "hard" with "hard" running; returns (task id, hard's id)."""
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="base", acceptance_criteria="ok"),
                SubtaskSpec(
                    description="hard",
                    acceptance_criteria="ok",
                    inputs={"topic": "WAL"},
                    depends_on=[0],
                    priority=70,
                ),
            ],
        )
    )
    base = store.claim("w")
    assert base is not None and base.lease_token is not None
    store.submit(base.id, base.lease_token, {"v": 0})
    store.transition(base.id, S.REVIEW, S.DONE, Actor.REVIEWER)
    return task_id, base.id


def test_parts_take_over_the_subtasks_place(store: Store) -> None:
    task_id, base_id = two_steps(store)
    hard = store.claim("w")
    assert hard is not None and hard.lease_token is not None
    parts = store.decompose(hard.id, hard.lease_token, [part("x"), part("y", 0)])

    x, y = parts
    assert [p.seq for p in parts] == [2, 3]
    assert {p.parent_id for p in parts} == {hard.id}
    assert all(p.inputs == {"topic": "WAL"} and p.priority == 70 for p in parts)
    assert store.dependencies(x.id) == [base_id]  # the parent's dependencies, inherited
    assert set(store.dependencies(y.id)) == {base_id, x.id}
    assert (x.status, y.status) == (S.READY, S.PENDING)

    parent = store.get_subtask(hard.id)
    assert parent.status is S.PENDING and parent.lease_token is None
    assert set(store.dependencies(hard.id)) == {base_id, x.id, y.id}
    assert store.parts(hard.id) == [x.id, y.id]
    [entry] = [e for e in history(store.conn, task_id) if e.action == "decomposed"]
    assert entry.details["parts"] == [x.id, y.id] and entry.to_status is S.PENDING
    assert check_history(store.conn, task_id) == []


def test_the_final_result_is_the_last_subtasks_not_its_parts(store: Store) -> None:
    """The chat retells `final_result` (ROADMAP C.3): a split last subtask's parts come after
    it in order, but the result of the goal is the one it combines from them."""
    task_id, _ = two_steps(store)
    hard = store.claim("w")
    assert hard is not None and hard.lease_token is not None
    store.decompose(hard.id, hard.lease_token, [part("x"), part("y")])
    for content in ({"x": 1}, {"y": 2}):
        claimed = store.claim("w")
        assert claimed is not None and claimed.lease_token is not None
        store.submit(claimed.id, claimed.lease_token, content)
        store.transition(claimed.id, S.REVIEW, S.DONE, Actor.REVIEWER)
        assert store.final_result(task_id) is None  # the last subtask is not done yet
    combined = store.claim("w")
    assert combined is not None and combined.id == hard.id and combined.lease_token is not None
    store.submit(hard.id, combined.lease_token, {"answer": 3})
    assert store.final_result(task_id) is None  # under review: not accepted yet
    store.transition(hard.id, S.REVIEW, S.DONE, Actor.REVIEWER)
    assert store.final_result(task_id) == {"answer": 3}


def test_only_one_level_and_at_least_two_parts(store: Store) -> None:
    two_steps(store)
    hard = store.claim("w")
    assert hard is not None and hard.lease_token is not None
    with pytest.raises(ValueError, match="at least two"):
        store.decompose(hard.id, hard.lease_token, [part("x")])
    with pytest.raises(ValueError, match="cycle"):
        store.decompose(hard.id, hard.lease_token, [part("x", 1), part("y", 0)])
    x, _ = store.decompose(hard.id, hard.lease_token, [part("x"), part("y")])
    claimed = store.claim("w")
    assert claimed is not None and claimed.id == x.id and claimed.lease_token is not None
    with pytest.raises(ValueError, match="cannot be split again"):
        store.decompose(x.id, claimed.lease_token, [part("a"), part("b")])


def test_a_lost_lease_splits_nothing(store: Store) -> None:
    task_id, _ = two_steps(store)
    hard = store.claim("w")
    assert hard is not None
    with pytest.raises(StaleStateError):
        store.decompose(hard.id, "not-the-lease", [part("x"), part("y")])
    assert len(store.list_subtasks(task_id)) == 2
    assert store.get_subtask(hard.id).status is S.RUNNING


# ── Through the coordinator ──────────────────────────────────────────────────


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="hard", acceptance_criteria="c")])


def coordinator(store: Store, clock: FakeClock, llm: FakeLLM, **kw: Any) -> Coordinator:
    return Coordinator(
        store,
        llm_executor(llm),
        llm_reviewer(llm),
        decomposer=decomposer(llm),
        sleep=clock.sleep,
        **kw,
    )


def strategies(store: Store, task_id: str) -> list[tuple[str, str]]:
    seq = {s.id: s.seq for s in store.list_subtasks(task_id)}
    return [
        (f"[{seq[e.subtask_id]}]", e.details["strategy"])
        for e in history(store.conn, task_id)
        if e.action == "attempt_started" and e.subtask_id is not None
    ]


def test_the_fifth_attempt_splits_and_the_parent_combines(store: Store, clock: FakeClock) -> None:
    results = iter(range(100))
    llm = (
        FakeLLM()
        .on("Execute", lambda request: f'{{"v": {next(results)}}}', repeat_last=True)
        .on("Review", REJECT, REJECT, REJECT, REJECT, ACCEPT, repeat_last=True)
        .on(SPLIT, split_reply("first half", "second half"))
    )
    task_id = store.create_task(one_step())
    assert coordinator(store, clock, llm).run(task_id).status is TaskStatus.DONE

    hard, first, second = store.list_subtasks(task_id)
    assert (first.parent_id, second.parent_id) == (hard.id, hard.id)
    assert {s.status for s in (hard, first, second)} == {S.DONE}
    assert strategies(store, task_id) == [
        ("[0]", "first"),
        ("[0]", "feedback"),
        ("[0]", "rephrase"),
        ("[0]", "rephrase"),  # no other model here: the audit says so
        ("[0]", "decompose"),
        ("[1]", "first"),
        ("[2]", "first"),
        ("[0]", "combine"),
    ]
    [split] = [r for r in llm.calls if r.prompt.startswith(SPLIT)]
    assert "1. too vague" in split.prompt and str(MAX_PARTS) in split.messages[0].content
    combine = [r for r in llm.calls if r.prompt.startswith("Execute")][-1]
    assert combine.messages[0].content.startswith(EXECUTOR_SYSTEM_COMBINE)
    payload = json.loads(combine.prompt.partition(":\n")[2])
    assert [d["subtask"] for d in payload["results_of_dependencies"]] == [
        "first half",
        "second half",
    ]
    assert check_history(store.conn, task_id) == []
    report = task_report(store, task_id)
    assert [s.part_of for s in report.subtasks] == [None, 0, 0]
    assert report.subtasks[0].depends_on == [1, 2]


def test_an_unusable_split_falls_back_to_a_normal_attempt(store: Store, clock: FakeClock) -> None:
    prose = garbage("prose").text
    llm = (
        FakeLLM()
        .on("Execute", '{"v": 1}', '{"v": 2}', '{"v": 3}', '{"v": 4}', '{"v": 5}')
        .on("Review", REJECT, REJECT, REJECT, REJECT, ACCEPT)
        .on(SPLIT, prose, prose)
    )
    task_id = store.create_task(one_step())
    assert coordinator(store, clock, llm).run(task_id).status is TaskStatus.DONE
    assert len(store.list_subtasks(task_id)) == 1
    assert [s for _, s in strategies(store, task_id)][4:] == ["decompose", "rephrase"]
    [failed] = [e for e in history(store.conn, task_id) if e.action == "decomposition_failed"]
    assert "no parts" in failed.details["reason"]


def test_a_part_is_never_split_again(store: Store, clock: FakeClock) -> None:
    results = iter(range(100))
    llm = (
        FakeLLM()
        .on("Execute", lambda request: f'{{"v": {next(results)}}}', repeat_last=True)
        # Four rejections of the parent, then its first part is rejected five times.
        .on("Review", *[REJECT] * 9, ACCEPT, repeat_last=True)
        .on(SPLIT, split_reply("a", "b"))
    )
    task_id = store.create_task(one_step())
    result = coordinator(store, clock, llm, policy=RetryPolicy(max_attempts=6)).run(task_id)
    assert result.status is TaskStatus.DONE
    assert len(store.list_subtasks(task_id)) == 3  # one split only
    part_a = [s for seq, s in strategies(store, task_id) if seq == "[1]"]
    assert part_a == ["first", "feedback", "rephrase", "rephrase", "rephrase", "rephrase"]
    assert sum(r.prompt.startswith(SPLIT) for r in llm.calls) == 1


def test_without_a_decomposer_nothing_is_split(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM().on("Execute", '{"v": 1}', repeat_last=True).on("Review", REJECT, repeat_last=True)
    )
    task_id = store.create_task(one_step())
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.NEEDS_HUMAN
    assert "decompose" not in [s for _, s in strategies(store, task_id)]
