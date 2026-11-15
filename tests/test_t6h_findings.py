"""Regressions from the fourth round of user findings."""

import json
from pathlib import Path

import pytest

from conftest import FakeClock
from desman.agents import llm_executor
from desman.core.context import build_context
from desman.core.coordinator import Work
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.core.tools import Registry, Toolbox
from desman.core.workspace import Workspace
from desman.interlocutor import Interlocutor, Reply, wrong_script
from desman.llm.fake import FakeLLM
from desman.planner import CONTINUE_RULES, LANGUAGE_RULES, must_run, planner_system, template_plan
from test_agents import no_heartbeat
from test_honesty import access, run_tool


@pytest.mark.parametrize("reply", ["17 × 23 = 391", "391 kg", "42 мс", "", "123 abcd"])
def test_short_answers_do_not_trigger_language_correction(reply: str) -> None:
    assert not wrong_script("Сколько будет семнадцать на двадцать три?", reply)


def test_longer_wrong_script_still_triggers_correction() -> None:
    assert wrong_script("Какой результат?", "The result is 391")
    assert not wrong_script("Какой результат?", "Результат: 391")


def test_numeric_reply_needs_only_one_call(store: Store, clock: FakeClock) -> None:
    sessions = Sessions(store.conn, clock=clock)
    message = sessions.add(None, Role.USER, "Сколько будет 17 на 23?")
    llm = FakeLLM().on("Chat turn", '{"type":"reply","text":"17 × 23 = 391"}')
    reply = Interlocutor(llm, "ru", clock=clock).turn(sessions.history(message.session_id))
    assert isinstance(reply, Reply) and reply.text == "17 × 23 = 391"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("marked, offered", [(True, True), (False, True), (True, False)])
def test_run_requirement_is_given_only_when_applicable(
    store: Store, clock: FakeClock, tmp_path: Path, marked: bool, offered: bool
) -> None:
    task = store.create_task(
        TaskSpec(
            goal="run",
            subtasks=[
                SubtaskSpec(
                    description="Report the result", acceptance_criteria="output", runs_code=marked
                )
            ],
        )
    )
    subtask = store.claim("test", task_id=task)
    assert subtask is not None
    from desman.core.tools import Permission

    tools = Toolbox(
        Workspace(tmp_path, task),
        Registry([run_tool([])]) if offered else Registry([]),
        access=access(Permission.SHELL),
    )
    work = Work(
        subtask=subtask,
        context=build_context(subtask, [], None),
        heartbeat=no_heartbeat,
        tools=tools,
    )
    llm = FakeLLM().on("Execute", json.dumps({"text": "result"}))
    llm_executor(llm, clock=clock)(work)
    system = llm.calls[0].messages[0].content
    assert ("This subtask requires running code" in system) == (marked and offered)
    if marked and offered:
        assert "successful run_command" in system
        assert "earlier subtask does not count" in system


def test_planning_keeps_writing_and_running_together() -> None:
    system = planner_system()
    assert "ONE code subtask" in system
    assert "Reading the files a subtask works on is never a separate subtask" in system
    assert "several independent parts" in system
    assert system.endswith(LANGUAGE_RULES)
    assert "ONE subtask" in CONTINUE_RULES
    assert "same code subtask" in CONTINUE_RULES
    spec = template_plan("Write and run palindrome")
    assert must_run(spec).subtasks[-1].runs_code
