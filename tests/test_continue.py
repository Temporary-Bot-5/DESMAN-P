"""Continuing a task (ADR 0010): new subtasks after the done ones, from their results; a done
step redone without rewriting it; the person's command and the chat's window."""

from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import actions, commands
from desman.cli import cli
from desman.core.audit import check_history
from desman.core.models import Origin, SubtaskSpec, TaskSpec
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.llm.fake import FakeLLM
from desman.planner import CONTINUE
from test_worker import worker

S = SubtaskStatus


def done_task(store: Store, clock: FakeClock, steps: int = 2) -> str:
    task = store.create_task(
        TaskSpec(
            goal="a site about X",
            subtasks=[
                SubtaskSpec(
                    description=f"step {i}",
                    acceptance_criteria="ok",
                    depends_on=[i - 1] if i else [],
                )
                for i in range(steps)
            ],
        )
    )
    worker(store, clock, "w").run(task)
    assert store.task_status(task) is TaskStatus.DONE
    return task


def plan_of(*descriptions: str) -> str:
    return json.dumps(
        {
            "subtasks": [
                {
                    "description": d,
                    "acceptance_criteria": "ok",
                    "depends_on": [i - 1] if i else [],
                }
                for i, d in enumerate(descriptions)
            ]
        }
    )


def test_extend_appends_after_the_result_and_the_result_moves(
    store: Store, clock: FakeClock
) -> None:
    task = done_task(store, clock)
    old = store.list_subtasks(task)
    before = store.final_result(task)
    llm = FakeLLM().on(CONTINUE, plan_of("add a contacts page", "the whole site again"))
    pending = actions.prepare_continuation(store, task[:8], "add a contacts page", None)
    ids, error = actions.continue_task(store, llm, pending, limit=None, by="person")
    assert error is None and len(ids) == 2
    [request] = llm.calls
    asked = json.loads(request.prompt.split("\n", 1)[1])
    assert asked["add"] == "add a contacts page" and len(asked["subtasks"]) == 2
    new = [store.get_subtask(i) for i in ids]
    assert [s.seq for s in new] == [2, 3]
    assert store.dependencies(ids[0]) == [old[-1].id]  # starts from the result so far
    assert store.dependencies(ids[1]) == [ids[0]]
    assert store.task_status(task) is TaskStatus.ACTIVE  # it has work again
    assert store.final_result(task) is None  # the new last step gives the result
    worker(store, clock, "w").run(task)
    assert store.final_result(task) == {"by": "the whole site again"} != before
    # Done work kept as it was, and the history accounts for every subtask.
    assert all(store.get_subtask(s.id).status is S.DONE for s in old)
    assert check_history(store.conn, task) == []
    [entry] = [e for e in store.audit_log(task) if e["action"] == "task_extended"]
    details = json.loads(entry["details_json"])
    assert details["subtasks"] == ids and details["by"] == "person"


def test_redo_starts_from_that_step_and_the_result(store: Store, clock: FakeClock) -> None:
    task = done_task(store, clock, steps=3)
    first, _, last = store.list_subtasks(task)
    llm = FakeLLM().on(CONTINUE, plan_of("redo the data"))
    ref = f"{store.short_id(task)}/0"
    pending = actions.prepare_continuation(store, None, "the data was wrong", ref)
    ids, _ = actions.continue_task(store, llm, pending, limit=None, by="person")
    asked = json.loads(llm.calls[0].prompt.split("\n", 1)[1])
    assert asked["redo_step"] == {"seq": 0, "why": "the data was wrong"}
    assert sorted(store.dependencies(ids[0])) == sorted([last.id, first.id])
    assert store.get_subtask(first.id).status is S.DONE  # never rewound


def test_a_failed_plan_falls_back_to_one_step(store: Store, clock: FakeClock) -> None:
    task = done_task(store, clock, steps=1)
    llm = FakeLLM().on(CONTINUE, "not json", "still not json")
    pending = actions.prepare_continuation(store, task, "add a footer", None)
    ids, error = actions.continue_task(store, llm, pending, limit=None, by="person")
    assert error is not None and len(ids) == 1
    assert store.get_subtask(ids[0]).description.startswith("add a footer")


@pytest.mark.parametrize("case", ["running step", "cancelled", "network", "empty"])
def test_what_is_not_continued(store: Store, clock: FakeClock, case: str) -> None:
    task = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="s", acceptance_criteria="ok")]),
        origin=Origin.NETWORK if case == "network" else Origin.LOCAL,
    )
    if case == "cancelled":
        store.cancel_task(task, "no")
    with pytest.raises(click.ClickException):
        if case == "running step":
            actions.prepare_continuation(store, None, "again", f"{store.short_id(task)}/0")
        else:
            actions.prepare_continuation(store, task, "" if case == "empty" else "more", None)


def test_the_cli_plans_and_adds(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    from desman import runs
    from desman.config import ConfigManager, default_config_path
    from desman.core import db
    from helpers import make_config

    store = Store(db.open_db(), clock=clock)
    task = done_task(store, clock, steps=1)
    llm = FakeLLM().on(CONTINUE, plan_of("a contacts page"))
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    result = CliRunner().invoke(cli, ["task", "extend", task[:8], "add a contacts page"])
    assert result.exit_code == 0, result.output
    assert f"Task {store.short_id(task)} continues: 1 new subtask." in result.output
    assert len(store.list_subtasks(task)) == 2
    result = CliRunner().invoke(cli, ["task", "redo", f"{task[:8]}/1", "use the new data"])
    assert result.exit_code != 0 and "Only a done step is redone" in result.output
    store.conn.close()


def test_the_catalog_asks_before_planning(store: Store, clock: FakeClock) -> None:
    task = done_task(store, clock, steps=1)
    planned: list[actions.Continuation] = []
    ctx = commands.Context(
        store,
        None,  # type: ignore[arg-type]
        start_task=lambda *_: None,
        resume_task=lambda _: None,
        continue_task=planned.append,
    )
    entry = commands.find("task extend")
    assert (
        entry is not None
        and entry.kind_of(commands.TaskExtendArgs(task=task, goal="x")) is commands.Kind.CHANGE
    )
    proposal = entry.handler(ctx, commands.TaskExtendArgs(task=task, goal="add a contacts page"))
    assert isinstance(proposal, commands.Proposal) and not planned  # nothing before yes
    assert any("add a contacts page" in line for line in proposal.lines)
    proposal.apply()
    assert [p.task_id for p in planned] == [task]


def test_the_chat_continues_a_task_after_the_window(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import runs
    from desman.config import ConfigManager, default_config_path
    from desman.core import db
    from helpers import make_config

    store = Store(db.open_db(), clock=clock)
    task = done_task(store, clock, steps=1)
    turn = {
        "type": "command",
        "name": "task extend",
        "args": {"task": task[:8], "goal": "add a map"},
    }
    llm = (
        FakeLLM()
        .on("Chat turn", json.dumps(turn), repeat_last=True)
        .on(CONTINUE, plan_of("add a map to the site"))
        .on("Execute subtask", '{"site": "with a map"}')
        .on("Review subtask", '{"verdict": "accept"}')
        .on("Retell task", '{"text": "The site now has a map."}')
    )
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    declined = CliRunner().invoke(cli, [], input="add a map to that site\nn\n")
    assert declined.exit_code == 0 and len(store.list_subtasks(task)) == 1  # no means no
    result = CliRunner().invoke(cli, [], input="add a map to that site\ny\n")
    assert result.exit_code == 0, result.output
    assert "Add: add a map" in result.output
    assert f"Task {store.short_id(task)} continues: 1 new subtask." in result.output
    assert store.task_status(task) is TaskStatus.DONE  # the terminal chat worked it
    assert store.final_result(task) == {"site": "with a map"}
    assert "The site now has a map." in result.output
    store.conn.close()
