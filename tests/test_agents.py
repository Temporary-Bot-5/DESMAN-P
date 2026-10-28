"""Tests for the minimal LLM-backed agents behind `desman run`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import FakeClock
from desman.agents import llm_executor, llm_reviewer
from desman.core.context import build_context
from desman.core.coordinator import Coordinator, Failure, Success, Work
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.core.workspace import Workspace
from desman.llm.fake import FakeLLM, garbage
from desman.llm.structured import json_object as _json_object


def no_heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
    pass


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="c")])


def payload(prompt: str) -> Any:
    heading, _, body = prompt.partition(":\n")
    return heading, json.loads(body)


def test_executor_prompt_carries_the_whole_context(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM()
        .on("Execute", '{"facts": [1]}', '{"draft": "v1"}', '{"draft": "v2"}')
        .on(
            "Review",
            '{"verdict": "accept"}',
            '{"verdict": "retry", "feedback": "shorter"}',
            '{"verdict": "accept"}',
        )
    )
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="collect", acceptance_criteria="has facts"),
            SubtaskSpec(
                description="write",
                acceptance_criteria="short",
                depends_on=[0],
                inputs={"tone": "dry"},
            ),
        ],
    )
    task_id = store.create_task(spec)
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.DONE
    executions = [payload(r.prompt) for r in llm.calls if r.prompt.startswith("Execute")]
    assert [h for h, _ in executions] == ["Execute subtask"] * 3
    _, retry = executions[2]
    assert retry["subtask"] == "write" and retry["acceptance_criteria"] == "short"
    assert retry["inputs"] == {"tone": "dry"}
    assert retry["results_of_dependencies"] == [
        {"subtask": "collect", "result": {"facts": [1]}, "truncated": False}
    ]
    assert (retry["attempt"], retry["feedback_on_previous_attempt"]) == (2, "shorter")
    reviews = [payload(r.prompt) for r in llm.calls if r.prompt.startswith("Review")]
    assert reviews[1][1] == {
        "subtask": "write",
        "acceptance_criteria": "short",
        "result": {"draft": "v1"},
    }
    assert all(r.messages[0].role == "system" for r in llm.calls)


def test_executor_turns_non_objects_into_quality_failures(store: Store, tmp_path: Path) -> None:
    # Each call: a reply, then (if unusable) one repair request.
    llm = FakeLLM(garbage("prose"), garbage("not_object"), garbage("truncated"), '{"ok": 1}')
    execute = llm_executor(llm)
    store.create_task(one_step())
    claimed = store.claim("w")
    assert claimed is not None
    work = Work(
        subtask=claimed,
        context=build_context(claimed, [], None),
        heartbeat=no_heartbeat,
        workspace=Workspace(tmp_path, claimed.task_id),
    )
    first, second = execute(work), execute(work)
    assert isinstance(first, Failure) and first.kind is FailureKind.QUALITY
    assert "not a JSON object" in first.reason and "after a repair request" in first.reason
    assert isinstance(second, Success) and second.content == {"ok": 1}
    assert len(llm.calls) == 4


def test_unusable_verdict_is_an_interrupted_review(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM()
        .on("Execute", '{"ok": 1}')
        .on(
            "Review",
            garbage("prose"),
            '{"verdict": "maybe"}',  # the repair fails too: one interrupted review
            '{"verdict": "maybe"}',
            garbage("not_object"),  # and again
            '{"verdict": "accept"}',
        )
    )
    task_id = store.create_task(one_step())
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert (sub.attempts.quality, sub.attempts.transient) == (0, 2)  # the work was never redone
    assert sum(r.prompt.startswith("Execute") for r in llm.calls) == 1


# ── Review regressions (PR #11) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply", ['{"value": 1e999}', '{"value": -1e999}', '{"v": NaN}', '{"v": Infinity}']
)
def test_non_finite_numbers_are_a_quality_failure(
    store: Store, clock: FakeClock, reply: str
) -> None:
    llm = (
        FakeLLM()
        .on("Execute", reply, reply, '{"ok": 1}')
        .on("Review", '{"verdict": "accept"}', repeat_last=True)
    )
    task_id = store.create_task(one_step())
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert sub.attempts.quality == 1
    assert store.latest_artifact(sub.id) == {"ok": 1}


def test_success_content_must_be_strict_json() -> None:
    with pytest.raises(ValidationError):
        Success(content={"x": float("inf")})
    with pytest.raises(ValidationError):
        Success(content={"when": object()})
    assert Success(content={"nested": [1, "a", None, {"b": 2.5}]}).content["nested"][3] == {
        "b": 2.5
    }


def test_task_text_naming_the_other_role_does_not_misroute(store: Store, clock: FakeClock) -> None:
    # Routes in the order a careless script would write them: Execute first.
    llm = (
        FakeLLM()
        .on("Execute", '{"plan": "Review subtask: done"}')
        .on("Review", '{"verdict": "accept"}', repeat_last=True)
    )
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(
                description="Execute this plan, then Review it",
                acceptance_criteria="Execute subtask",
            )
        ],
    )
    task_id = store.create_task(spec)
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.DONE


# ── JSON inside a reply (seen on Gemini, found on a real provider run) ─────────


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (garbage("fenced").text, {"answer": 42}),
        (garbage("wrapped").text, {"answer": 42}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Result:\n```JSON\n{\n "a": [1, 2]\n}\n```\nDone.', {"a": [1, 2]}),
        ('```json\n[1]\n```\n```json\n{"second": true}\n```', {"second": True}),
    ],
)
def test_json_is_found_inside_a_reply(reply: str, expected: dict[str, Any]) -> None:
    assert _json_object(reply) == expected


@pytest.mark.parametrize(
    "name", ["empty", "prose", "truncated", "trailing_comma", "single_quotes", "not_object"]
)
def test_what_is_not_json_stays_a_failure(name: str) -> None:
    assert _json_object(garbage(name).text) is None


def test_strictness_holds_inside_a_fence() -> None:
    assert _json_object('```json\n{"v": NaN}\n```') is None
    assert _json_object('note {"v": 1e999} end') is None


# ── Repair request (2.3) ───────────────────────────────────────────────────────


def test_repair_request_shows_the_reply_and_the_problem(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM()
        .on("Execute", '{"ok": 1}')
        .on("Review", '{"verdict": "maybe", "feedback": 3}', '{"verdict": "accept"}')
    )
    task_id = store.create_task(one_step())
    result = Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(
        task_id
    )
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert (sub.attempts.quality, sub.attempts.transient) == (0, 0)  # repaired in place
    first, repair = (r for r in llm.calls if r.prompt.startswith("Review"))
    assert repair.messages[: len(first.messages)] == first.messages
    assert [m.role for m in repair.messages[len(first.messages) :]] == ["assistant", "user"]
    assert repair.messages[-2].content == '{"verdict": "maybe", "feedback": 3}'
    assert repair.prompt.startswith("Review subtask (repair):\n")
    assert "- verdict: Input should be 'accept', 'retry' or 'escalate'" in repair.prompt
    assert "- feedback: Input should be a valid string" in repair.prompt
