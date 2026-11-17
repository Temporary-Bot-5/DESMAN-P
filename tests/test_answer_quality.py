"""Final answer formatting is best effort within the successful attempt (T.6o)."""

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.actions import result_text
from desman.agents import (
    EXECUTOR_SYSTEM,
    REVIEWER_EVIDENCE,
    REVIEWER_SYSTEM,
    llm_executor,
    llm_reviewer,
)
from desman.cli import cli
from desman.compaction import summary_of
from desman.core import calls, db
from desman.core.coordinator import Coordinator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.llm.base import Completion, Request
from desman.llm.fake import FakeLLM, RateLimit, Timeout
from desman.llm.openai_compat import CallRecord
from desman.result_prose import has_prose

DATA = {"items": [{"name": "meal", "cost": 5}], "total_cost": 5}


def run_final(store: Store, clock: FakeClock, llm: FakeLLM, *, format_final: bool = True) -> str:
    task = store.create_task(
        TaskSpec(
            goal="Составь меню",
            subtasks=[
                SubtaskSpec(
                    description="Answer",
                    acceptance_criteria="Answers the goal",
                )
            ],
        )
    )
    outcome = Coordinator(
        store, llm_executor(llm, format_final=format_final), llm_reviewer(llm), sleep=clock.sleep
    ).run(task)
    assert outcome.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task)
    assert sub.attempts.quality == sub.attempts.transient == 0
    return task


def test_final_data_becomes_stored_markdown(store: Store, clock: FakeClock) -> None:
    text = "| Блюдо | Цена |\n| --- | --- |\n| meal | 5 |\n\nИтого: 5."
    llm = (
        FakeLLM()
        .on("Execute", json.dumps(DATA))
        .on("Format", json.dumps({"text": text}))
        .on("Review", '{"verdict":"accept"}')
    )
    task = run_final(store, clock, llm)
    assert store.final_result(task) == {"text": text}
    assert result_text(store.final_result(task)) == text
    execute, follow, review = llm.calls
    payload = json.loads(follow.prompt.split(":\n", 1)[1])
    assert payload == {"goal": "Составь меню", "result": DATA}
    system = follow.messages[0].content
    for phrase in (
        "Markdown",
        "language of the goal",
        "every fact, number, date and source",
        "JSON/CSV",
        "fenced block",
        "No tools",
    ):
        assert phrase in system
    assert "tools" not in payload
    assert json.loads(review.prompt.split(":\n", 1)[1])["result"] == {"text": text}


@pytest.mark.parametrize(
    "bad",
    [
        "garbage",
        '{"text":""}',
        '{"text":"  "}',
        '{"text":"Однопarentalные"}',
        RateLimit(),
        Timeout(),
    ],
)
def test_failed_format_preserves_success(store: Store, clock: FakeClock, bad: Any) -> None:
    llm = (
        FakeLLM(clock=clock, sleep=clock.sleep)
        .on("Execute", json.dumps(DATA))
        .on("Format", bad, bad)
        .on("Review", '{"verdict":"accept"}')
    )
    task = run_final(store, clock, llm)
    assert store.final_result(task) == DATA
    expected = 1 if isinstance(bad, (RateLimit, Timeout)) else 2
    assert sum(call.prompt.startswith("Format") for call in llm.calls) == expected


@pytest.mark.parametrize(
    "content",
    [{"text": "Ready"}, {"summary": "Summary"}, {"type": "result", "result": {"answer": "Ready"}}],
)
def test_prose_needs_no_followup(store: Store, clock: FakeClock, content: dict[str, Any]) -> None:
    llm = FakeLLM().on("Execute", json.dumps(content)).on("Review", '{"verdict":"accept"}')
    task = run_final(store, clock, llm)
    assert store.final_result(task) == content
    assert len(llm.calls) == 2
    if "summary" in content:
        assert summary_of(store.final_result(task)) == "Summary"


def test_network_contract_can_disable_formatting(store: Store, clock: FakeClock) -> None:
    llm = FakeLLM().on("Execute", json.dumps(DATA)).on("Review", '{"verdict":"accept"}')
    task = run_final(store, clock, llm, format_final=False)
    assert store.final_result(task) == DATA
    assert len(llm.calls) == 2


def test_nonfinal_data_is_left_alone(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM()
        .on("Execute", json.dumps(DATA), '{"text":"Ready"}')
        .on("Review", '{"verdict":"accept"}', repeat_last=True)
    )
    task = store.create_task(
        TaskSpec(
            goal="Answer",
            subtasks=[
                SubtaskSpec(description="Collect", acceptance_criteria="Data"),
                SubtaskSpec(description="Answer", acceptance_criteria="Prose", depends_on=[0]),
            ],
        )
    )
    assert (
        Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(task).status
        is TaskStatus.DONE
    )
    assert len(llm.calls) == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        *[
            ({key: "Prose", "count": 3}, True)
            for key in (
                "content",
                "text",
                "answer",
                "result",
                "output",
                "response_text",
                "response",
                "message",
                "summary",
                "final_answer",
                "answer_text",
                "reply",
            )
        ],
        ({"custom": "x" * 80, "count": 3}, True),
        ({"custom": "line\nline", "count": 3}, True),
        ({"custom": "short"}, True),
        ({"custom": "short", "count": 3}, False),
        ({"custom": "x" * 80, "items": []}, False),
        ({"type": "result", "result": {"answer": "Prose"}}, True),
        ({"type": "result", "result": DATA}, False),
        (DATA, False),
        ("Prose", True),
        ([], False),
    ],
)
def test_shared_prose_predicate(value: Any, expected: bool) -> None:
    assert has_prose(value) is expected


def test_quality_prompt_rules_are_narrow_and_always_check_totals() -> None:
    for phrase in (
        "does not exist",
        "specific versions or features",
        "from memory and may be outdated",
        "common words in the language of the goal",
        "Totals, sums, averages and ranges",
        "match them",
    ):
        assert phrase in EXECUTOR_SYSTEM
    assert "categorical claim" in REVIEWER_EVIDENCE
    assert "tools_available" in REVIEWER_EVIDENCE
    assert "Do not judge this claim by memory" in REVIEWER_EVIDENCE
    assert "own itemised" in REVIEWER_SYSTEM
    assert "name the mismatch" in REVIEWER_SYSTEM


def test_followup_is_journaled_as_executor_usage(store: Store, clock: FakeClock) -> None:
    record = calls.recorder(store.conn)

    class RecordingFake(FakeLLM):
        def complete(self, request: Request) -> Completion:
            reply = super().complete(request)
            record(
                CallRecord(
                    provider=reply.provider,
                    model=reply.model,
                    key_id=None,
                    prompt_hash="prompt",
                    response_hash="reply",
                    prompt_tokens=reply.usage.prompt_tokens,
                    completion_tokens=reply.usage.completion_tokens,
                    latency_ms=0,
                    http_status=200,
                    error_kind=None,
                    error=None,
                )
            )
            return reply

    llm = (
        RecordingFake()
        .on("Execute", json.dumps(DATA))
        .on("Format", '{"text":"Total: 5"}')
        .on("Review", '{"verdict":"accept"}')
    )
    task = run_final(store, clock, llm)
    [sub] = store.list_subtasks(task)
    rows = store.conn.execute(
        "SELECT role, task_id, subtask_id, prompt_tokens, completion_tokens FROM model_calls ORDER BY seq"
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("executor", task, sub.id),
        ("executor", task, sub.id),
        ("reviewer", task, sub.id),
    ]
    assert all(row[3] > 0 and row[4] > 0 for row in rows)


def test_run_and_status_show_formatted_final_text(tmp_path: Path) -> None:
    script = tmp_path / "answers.json"
    script.write_text(
        json.dumps(
            {
                "routes": [
                    {"match": "Execute", "steps": [json.dumps(DATA)]},
                    {"match": "Format", "steps": [json.dumps({"text": "## Menu\nTotal: 5"})]},
                    {"match": "Review", "steps": ['{"verdict":"accept"}']},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "Menu", "--fake-llm", str(script)])
    assert result.exit_code == 0, result.output
    assert "Total: 5" in result.output
    conn = db.open_db()
    try:
        task = conn.execute("SELECT id FROM tasks").fetchone()[0]
    finally:
        conn.close()
    status = runner.invoke(cli, ["status", task])
    assert status.exit_code == 0, status.output
    assert "Total: 5" in status.output
    assert "total_cost:" not in status.output


def test_final_format_gets_one_structured_repair(store: Store, clock: FakeClock) -> None:
    llm = (
        FakeLLM()
        .on("Execute", json.dumps(DATA))
        .on("Format", '{"text":""}', '{"text":"Total: 5"}')
        .on("Review", '{"verdict":"accept"}')
    )
    task = run_final(store, clock, llm)
    assert store.final_result(task) == {"text": "Total: 5"}
    assert llm.calls[2].prompt.startswith("Format final answer (repair):")
    assert llm.calls[2].messages[0] == llm.calls[1].messages[0]
