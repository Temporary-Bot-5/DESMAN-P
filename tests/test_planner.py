"""Tests for the LLM planner: goal → validated subtask graph, template fallback (2.7, D1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner, Result

from desman.cli import EXIT_DONE, cli
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.models import Priority
from desman.core.store import Store
from desman.llm.fake import FakeLLM, RateLimit, garbage
from desman.planner import (
    CRITERIA_RULES,
    MAX_SUBTASKS,
    PLAN,
    SPLITTER_SYSTEM,
    TEMPLATE_CRITERIA,
    plan,
)


def sub(description: str, *depends_on: int) -> dict[str, Any]:
    return {
        "description": description,
        "acceptance_criteria": f"{description} is done",
        "depends_on": list(depends_on),
    }


def reply(*subtasks: dict[str, Any]) -> str:
    return json.dumps({"subtasks": list(subtasks)})


# ── plan() ───────────────────────────────────────────────────────────────────


def test_the_models_plan_becomes_the_task_graph() -> None:
    llm = FakeLLM(reply(sub("outline"), sub("draft", 0), sub("edit", 1)))
    planned = plan(llm, "write an essay", Priority.HIGH)
    assert planned.source == "llm" and planned.error is None
    assert planned.details() == {"plan": "llm"}
    spec = planned.spec
    assert spec.goal == "write an essay" and spec.priority == Priority.HIGH
    assert [s.description for s in spec.subtasks] == ["outline", "draft", "edit"]
    assert [s.depends_on for s in spec.subtasks] == [[], [0], [1]]
    assert spec.subtasks[1].acceptance_criteria == "draft is done"
    [call] = llm.calls
    assert call.prompt.startswith(f"{PLAN}:\nwrite an essay")
    assert str(MAX_SUBTASKS) in call.messages[0].content  # the limit is in the instructions


def test_criteria_are_ones_the_tools_can_confirm() -> None:
    """C.9: a plan that asked for a flight's exact time on a date sent an honest result back
    until the attempts ran out. The planner and the splitter are told what a criterion may ask."""
    llm = FakeLLM(reply(sub("find cheap tickets")))
    plan(llm, "find cheap tickets Berlin - Lisbon", Priority.NORMAL)
    [call] = llm.calls
    assert call.messages[0].content.endswith(CRITERIA_RULES)
    assert SPLITTER_SYSTEM.endswith(CRITERIA_RULES)
    assert "with its source" in CRITERIA_RULES and "not found" in CRITERIA_RULES


def test_a_broken_graph_gets_the_repair_request_with_the_reason() -> None:
    llm = FakeLLM(reply(sub("a", 1), sub("b", 0), sub("c", 1)), reply(sub("a"), sub("b", 0)))
    planned = plan(llm, "g")
    assert planned.source == "llm"
    assert [s.depends_on for s in planned.spec.subtasks] == [[], [0]]
    repair = llm.calls[1].prompt
    assert repair.startswith(f"{PLAN} (repair):")
    assert "- (object): Value error, dependency cycle among subtasks [0, 1, 2]\n" in repair


def test_work_nothing_uses_feeds_the_final_subtask() -> None:
    # "a" and "b" are independent; without the added edges "summary" would never see "a".
    llm = FakeLLM(reply(sub("a"), sub("b"), sub("c", 1), sub("summary", 2)))
    assert [s.depends_on for s in plan(llm, "g").spec.subtasks] == [[], [], [1], [2, 0]]


def test_the_last_subtask_must_be_the_final_one() -> None:
    llm = FakeLLM(reply(sub("a", 1), sub("b")), reply(sub("b"), sub("a", 0)))
    assert plan(llm, "g").source == "llm"
    assert "subtasks [0] depend on it" in llm.calls[1].prompt


def test_bad_indices_and_empty_texts_are_rejected() -> None:
    for bad in (
        reply(sub("a", 0)),  # depends on itself
        reply(sub("a", 3)),  # no such subtask
        reply({"description": "  ", "acceptance_criteria": "c"}),
        reply(),
        reply(*(sub(str(i)) for i in range(MAX_SUBTASKS + 1))),
    ):
        planned = plan(FakeLLM(bad, bad), "the goal")
        assert planned.source == "template", bad


def test_a_reply_that_stays_wrong_falls_back_to_the_template() -> None:
    llm = FakeLLM(garbage("prose"), garbage("prose"))
    planned = plan(llm, "the goal", Priority.LOW)
    assert planned.source == "template"
    assert planned.error is not None and planned.error.startswith("StructuredOutputError:")
    assert planned.details() == {"plan": "template", "plan_error": planned.error}
    assert garbage("prose").text[:20] not in planned.error  # the reply may echo a secret
    [only] = planned.spec.subtasks
    assert only.description == "the goal" and only.acceptance_criteria == TEMPLATE_CRITERIA
    assert planned.spec.priority == Priority.LOW
    assert len(llm.calls) == 2


def test_a_provider_failure_falls_back_to_the_template() -> None:
    llm = FakeLLM(RateLimit(retry_after=30))
    planned = plan(llm, "the goal")
    assert planned.source == "template"
    assert planned.error is not None and planned.error.startswith("RateLimited:")
    assert len(llm.calls) == 1  # no repair request for a provider failure


# ── desman run ───────────────────────────────────────────────────────────────


def script(tmp_path: Path, planner: list[Any], execute: list[Any]) -> Path:
    path = tmp_path / "script.json"
    routes = [
        {"match": "Plan", "steps": planner},
        {"match": "Execute", "steps": execute},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    path.write_text(json.dumps({"routes": routes}))
    return path


def desman(*args: str | Path) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_run_works_through_the_planned_graph_in_order(tmp_path: Path) -> None:
    fake = script(
        tmp_path,
        [reply(sub("find facts"), sub("write the answer", 0))],
        ['{"facts": ["WAL"]}', '{"answer": "use WAL"}'],
    )
    result = desman("run", "explain WAL", "--fake-llm", fake, "--plan", "llm", "--json")
    assert result.exit_code == EXIT_DONE, result.output
    out = json.loads(result.output)
    assert out["plan"] == {"plan": "llm"}
    assert [s["description"] for s in out["task"]["subtasks"]] == ["find facts", "write the answer"]

    store = Store(db.open_db())
    try:
        task_id = out["task"]["id"]
        entries = history(store.conn, task_id)
        created = next(e for e in entries if e.action == "task_created")
        assert created.details["plan"] == "llm" and len(created.details["subtasks"]) == 2
        assert check_history(store.conn, task_id) == []
    finally:
        store.conn.close()


def test_run_tells_the_operator_when_the_template_is_used(tmp_path: Path) -> None:
    prose = garbage("prose").text
    fake = script(tmp_path, [prose, prose], ['{"answer": 1}'])
    result = desman("run", "the goal", "--fake-llm", fake, "--plan", "llm")
    assert result.exit_code == EXIT_DONE, result.output
    assert "Planning…" in result.output
    assert "the goal is the only subtask: StructuredOutputError" in " ".join(result.output.split())
    assert "(1 subtask)" in result.output


def test_a_fake_run_uses_the_template_unless_asked(tmp_path: Path) -> None:
    fake = script(tmp_path, [], ['{"answer": 1}'])  # no planner step: it must not be asked
    result = desman("run", "the goal", "--fake-llm", fake, "--json")
    assert result.exit_code == EXIT_DONE, result.output
    assert json.loads(result.output)["plan"] == {"plan": "template"}


def test_plan_does_not_go_with_a_spec(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"goal": "g", "subtasks": [sub("a")]}))
    result = desman("run", "--spec", spec, "--plan", "template")
    assert result.exit_code == 2 and "--spec is already a plan" in result.output
