"""Nothing is told as done that was not done: work that must run code runs it, a retelling copies
code instead of writing it, and the chat says where a task's files are (code, not a model)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result
from pydantic import BaseModel, ConfigDict

from conftest import FakeClock
from desman import runs, task_files
from desman.agents import HONEST, NO_TOOLS, llm_executor
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Verdict
from desman.core.models import SubtaskSpec, TaskSpec, Tools
from desman.core.policy import Access, Policy, Role, RolePolicy
from desman.core.retry import strategy_for
from desman.core.sessions import Kind, Sessions
from desman.core.sessions import Role as Said
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import READ_FILE, WRITE_FILE, Permission, Registry, Tool
from desman.core.workspace import Workspace, workspace_root_for
from desman.interlocutor import CHAT, RETELL, NewTask, unbacked_code
from desman.llm.fake import FakeLLM
from desman.planner import Plan, Split, planner_system, splitter_system
from helpers import make_config

S = SubtaskStatus


class _Code(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str


def run_tool(ran: list[str]) -> Tool[_Code]:
    """A stand-in for the sandboxed `run_command`, with the permission the real one needs."""

    def run(args: _Code, workspace: Workspace) -> dict[str, Any]:
        ran.append(args.command)
        return {"exit_code": 0, "stdout": "5\n"}

    return Tool("run_command", "Run a command.", _Code, Permission.SHELL, run)


def access(*switches: Permission) -> Access:
    tools = ["run_command", "read_file", "write_file"]
    return Access(Policy(roles={Role.EXECUTOR: RolePolicy(tools=tools)}), Role.EXECUTOR, switches)


def runs_code() -> TaskSpec:
    return TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(
                description="Run calc.py and report what it prints",
                acceptance_criteria="the output",
                runs_code=True,
            )
        ],
    )


def accept(subtask: object, artifact: object) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def coordinator(
    store: Store, executor: Any, tmp_path: Path, clock: FakeClock, ran: list[str], *on: Permission
) -> Coordinator:
    return Coordinator(
        store,
        executor,
        accept,
        workspaces=tmp_path,
        tools=Registry([run_tool(ran), READ_FILE, WRITE_FILE]),
        access=access(*on),
        sleep=clock.sleep,
        strategies=strategy_for,
        decomposer=lambda work: None,
    )


# ── Work that must run code ──────────────────────────────────────────────────


def test_without_the_tool_to_run_code_the_subtask_waits_for_the_person_at_once(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    """The crash test of the chat: no model is asked to make up what running would print."""
    fake = FakeLLM().on("Execute", '{"stdout": "5"}')
    task_id = store.create_task(runs_code())
    result = coordinator(store, llm_executor(fake), tmp_path, clock, []).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    assert fake.calls == []
    [subtask] = store.list_subtasks(task_id)
    assert subtask.status is S.NEEDS_HUMAN and subtask.attempts.quality == 0
    [entry] = [r for r in store.audit_log(task_id) if r["action"] == "tool_unavailable"]
    reason = json.loads(entry["details_json"])["reason"]
    assert "shell_execution" in reason and "must run code" in reason
    assert check_history(store.conn, task_id) == []


def test_a_result_handed_in_without_running_the_code_is_not_a_result(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    ran: list[str] = []
    call = {"tool_calls": [{"tool": "run_command", "args": {"command": "python calc.py"}}]}
    fake = FakeLLM().on(
        "Execute",
        '{"stdout": "5"}',  # told, not run
        json.dumps(call),
        '{"stdout": "5"}',  # what the run returned
    )
    task_id = store.create_task(runs_code())
    run = coordinator(store, llm_executor(fake), tmp_path, clock, ran, Permission.SHELL)
    assert run.run(task_id).status is TaskStatus.DONE
    assert ran == ["python calc.py"]
    failed = [r for r in store.audit_log(task_id) if r["action"] == "execution_failed"]
    assert len(failed) == 1 and "without a successful run_command call" in json.loads(
        failed[0]["details_json"]
    ).get("reason", "")
    assert check_history(store.conn, task_id) == []


def test_a_subtask_that_runs_code_is_never_split(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    """Parts could hand in what only running the code can tell: the attempts go on instead,
    until the person decides."""
    fake = FakeLLM().on("Execute", *['{"stdout": "5"}'] * 5)
    split: list[object] = []
    task_id = store.create_task(runs_code())
    run = coordinator(store, llm_executor(fake), tmp_path, clock, [], Permission.SHELL)

    def decomposer(work: object) -> None:
        split.append(work)

    run.decomposer = decomposer
    assert run.run(task_id).status is TaskStatus.NEEDS_HUMAN
    strategies = [
        json.loads(r["details_json"])["strategy"]
        for r in store.audit_log(task_id)
        if r["action"] == "attempt_started"
    ]
    assert len(strategies) == 5 and "decompose" not in strategies and split == []


def test_the_planner_marks_work_that_runs_code_and_it_reaches_the_store(store: Store) -> None:
    plan = Plan.model_validate(
        {
            "subtasks": [
                {"description": "write it", "acceptance_criteria": "c", "kind": "code"},
                {
                    "description": "run it",
                    "acceptance_criteria": "c",
                    "depends_on": [0],
                    "runs_code": True,
                },
            ]
        }
    )
    task_id = store.create_task(plan.spec("g"))
    assert [s.runs_code for s in store.list_subtasks(task_id)] == [False, True]
    parts = Split.model_validate(
        {
            "subtasks": [
                {"description": "a", "acceptance_criteria": "c"},
                {"description": "b", "acceptance_criteria": "c", "runs_code": True},
            ]
        }
    ).parts()
    assert [p.runs_code for p in parts] == [False, True]
    assert '"runs_code": true' in planner_system() and '"runs_code": true' in splitter_system()


def test_a_database_before_runs_code_upgrades_with_none_held_to_it(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    for v, script in enumerate(db.MIGRATIONS[:26], start=1):
        with db.transaction(conn):
            for statement in db._split_sql(script):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {v}")
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO tasks (id, goal, priority, created_at) VALUES ('t', 'g', 50, 'now')"
        )
        conn.execute(
            "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria, "
            "priority, status, created_at, updated_at) "
            "VALUES ('s', 't', 0, 'd', 'c', 50, 'ready', 'now', 'now')"
        )
    conn.close()
    upgraded = db.open_db(path)
    [subtask] = Store(upgraded).list_subtasks("t")
    assert subtask.runs_code is False
    with pytest.raises(sqlite3.IntegrityError):
        upgraded.execute("UPDATE subtasks SET runs_code = 2")


def test_the_executor_is_told_what_it_cannot_do(store: Store, tmp_path: Path) -> None:
    fake = FakeLLM().on("Execute", '{"r": 1}', '{"r": 2}')
    for tools in (Tools.NONE, Tools.GRANTED):
        spec = TaskSpec(
            goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")], tools=tools
        )
        task_id = store.create_task(spec)
        Coordinator(store, llm_executor(fake), accept, workspaces=tmp_path).run(task_id)
    without, granted = (r.messages[0].content for r in fake.calls)
    assert HONEST in without and without.endswith(NO_TOOLS)
    assert HONEST in granted and NO_TOOLS not in granted


# ── The retelling copies code ────────────────────────────────────────────────


CODE = "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n"


def test_code_found_in_the_task_s_work_is_backed_and_made_up_code_is_not() -> None:
    copied = f"Here it is:\n```python\n{CODE.replace('    ', '  ')}```\nRun it."
    assert unbacked_code(copied, [None, [CODE]]) == []  # indentation does not tell it apart
    assert unbacked_code(copied, [{"code": CODE}, []]) == []
    made_up = "```python\ndef main():\n    print('hello')\n```"
    assert unbacked_code(made_up, [{"code": CODE}, [CODE]]) == ["def main():\nprint('hello')"]
    assert unbacked_code("Run it with\n```\npython calc.py\n```", [None, []]) == []  # one line


def test_a_task_turn_may_name_earlier_tasks_for_their_files() -> None:
    turn = NewTask.model_validate(
        {"type": "task", "title": "t", "goal": "g", "files_from": ["6667F475"]}
    )
    assert turn.files_from == ["6667f475"]
    with pytest.raises(ValueError):
        NewTask.model_validate({"type": "task", "title": "t", "goal": "g", "files_from": ["../x"]})


# ── In the chat ──────────────────────────────────────────────────────────────


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, ["-c"], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def opened() -> sqlite3.Connection:
    return db.open_db(db.default_db_path())


def said() -> list[tuple[Said, Kind, str]]:
    sessions = Sessions(opened())
    latest = sessions.latest()
    assert latest is not None
    return [(m.role, m.kind, m.text) for m in sessions.messages(latest.id)]


PLAN = json.dumps({"subtasks": [{"description": "Write calc.py", "acceptance_criteria": "c"}]})
WRITE = json.dumps(
    {"tool_calls": [{"tool": "write_file", "args": {"path": "calc.py", "content": CODE}}]}
)
TASK = json.dumps({"type": "task", "title": "Calculator", "goal": "Write a calculator"})


def written(fake: FakeLLM, retold: str) -> FakeLLM:
    return (
        fake.on(CHAT, TASK)
        .on("Plan task", PLAN)
        .on("Execute subtask", WRITE, '{"files_written": ["calc.py"]}')
        .on("Review subtask", '{"verdict": "accept"}')
        .on(RETELL, json.dumps({"text": retold}))
    )


def test_the_chat_says_where_the_files_are_and_the_retelling_sees_them(fake: FakeLLM) -> None:
    written(fake, f"Done:\n```python\n{CODE}```")
    result = chat("write me a calculator")
    assert "def add(a, b):" in result.output
    [retell] = [r for r in fake.calls if r.prompt.startswith(RETELL)]
    outcome = json.loads(retell.prompt.split("\n", 1)[1])
    [file] = outcome["files"]
    assert file["path"] == "calc.py" and file["content"] == CODE and "untrusted" in file
    [(task_id,)] = opened().execute("SELECT id FROM tasks").fetchall()
    folder = task_files.folder(workspace_root_for(opened()), task_id)
    [listed] = [text for role, kind, text in said() if text.startswith("Files of task")]
    assert str(folder) in listed and "calc.py (" in listed  # an event: the interlocutor sees it


def test_code_the_task_never_wrote_is_not_shown_as_its_work(fake: FakeLLM) -> None:
    written(fake, "Done:\n```python\ndef main():\n    print(eval(input()))\n```")
    result = chat("write me a calculator")
    assert "print(eval(input()))" not in result.output
    assert "did not produce, so it is not shown" in result.output
    assert '"files_written"' in result.output  # the result as it is
    assert not any(role is Said.INTERLOCUTOR and "eval" in text for role, _, text in said())


def first_task() -> str:
    [(task_id,)] = opened().execute("SELECT id FROM tasks ORDER BY created_at LIMIT 1").fetchall()
    return str(task_id)


def test_a_later_task_gets_copies_of_the_files_it_builds_on(fake: FakeLLM) -> None:
    def test_it(request: object) -> str:
        turn = {"type": "task", "title": "Test", "goal": "Test calc.py"}
        return json.dumps({**turn, "files_from": [first_task()[:8]]})

    fake.on(CHAT, TASK, test_it).on("Plan task", PLAN, PLAN)
    fake.on("Execute subtask", WRITE, '{"files_written": ["calc.py"]}', '{"read": true}')
    fake.on("Review subtask", '{"verdict": "accept"}', '{"verdict": "accept"}')
    fake.on(RETELL, '{"text": "Done."}', '{"text": "Tested."}')
    chat("write me a calculator", "now test it")
    first = first_task()
    ref = first[:8]
    events = [text for role, kind, text in said() if kind is Kind.EVENT]
    assert any(f"got copies of the files of task {ref}: from-{ref}/calc.py" in e for e in events)
    plans = [r for r in fake.calls if r.prompt.startswith("Plan task")]
    assert f"from-{ref}/calc.py" in plans[-1].prompt  # named in the goal, by code
    [(second,)] = opened().execute("SELECT id FROM tasks WHERE id != ?", (first,)).fetchall()
    root = workspace_root_for(opened())
    assert Workspace(root, second).read_text(f"from-{ref}/calc.py") == CODE
    assert Workspace(root, first).files() == ["calc.py"]  # the original stays as it was


def test_a_task_of_another_chat_gives_no_files(fake: FakeLLM) -> None:
    store = Store(opened())
    other = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    Workspace(workspace_root_for(store.conn), other).write("secret.txt", "not yours")
    turn = {"type": "task", "title": "t", "goal": "g", "files_from": [other[:8]]}
    fake.on(CHAT, json.dumps(turn)).on("Plan task", PLAN).on("Execute subtask", '{"r": 1}')
    fake.on("Review subtask", '{"verdict": "accept"}').on(RETELL, '{"text": "Done."}')
    result = chat("use that")
    assert "is not a task of this chat, so its files are not given" in result.output
    plans = [r for r in fake.calls if r.prompt.startswith("Plan task")]
    assert "secret.txt" not in plans[-1].prompt


def test_the_interlocutor_is_told_it_never_did_anything(fake: FakeLLM) -> None:
    fake.on(CHAT, '{"type": "reply", "text": "I cannot save it myself."}')
    chat("save the code there")
    system = fake.calls[0].messages[0].content
    assert "never says that something was done, saved" in system
    assert '"files_from"' in system


# ── After review (Devin, PR #76) ─────────────────────────────────────────────


def test_a_template_plan_is_held_to_running_code_when_the_person_says_so(tmp_path: Path) -> None:
    """`--plan template` (and a planner that failed) marks nothing: `--runs-code` does."""
    from desman.cli import EXIT_NEEDS_HUMAN

    script = tmp_path / "script.json"
    routes = [{"match": "Execute", "steps": ['{"stdout": "5"}'], "repeat_last": True}]
    script.write_text(json.dumps({"routes": routes}))
    args = ["run", "run calc.py and show its output", "--plan", "template", "--runs-code"]
    result = CliRunner().invoke(cli, [*args, "--fake-llm", str(script)])
    assert result.exit_code == EXIT_NEEDS_HUMAN, result.output
    store = Store(db.open_db())
    [(task_id,)] = store.conn.execute("SELECT id FROM tasks").fetchall()
    [subtask] = store.list_subtasks(task_id)
    assert subtask.runs_code and subtask.status is S.NEEDS_HUMAN
    assert not any(r["action"] == "attempt_started" for r in store.audit_log(task_id))


def test_must_run_marks_the_last_subtask_and_keeps_the_plan_s_marks() -> None:
    from desman.planner import must_run, template_plan

    assert [s.runs_code for s in must_run(template_plan("g")).subtasks] == [True]
    plan = Plan.model_validate(
        {
            "subtasks": [
                {"description": "a", "acceptance_criteria": "c", "runs_code": True},
                {"description": "b", "acceptance_criteria": "c", "depends_on": [0]},
            ]
        }
    ).spec("g")
    # An earlier run (a setup step) proves nothing about the result: the last one runs too.
    assert [s.runs_code for s in must_run(plan).subtasks] == [True, True]
    assert must_run(must_run(plan)) == must_run(plan)


def test_the_interlocutor_s_word_holds_a_failed_plan_to_running_code(fake: FakeLLM) -> None:
    turn = {"type": "task", "title": "Crash test", "goal": "Crash-test calc.py", "runs_code": True}
    fake.on(CHAT, json.dumps(turn)).on("Plan task", "not json", "still not json")
    fake.on(RETELL, '{"text": "It needs shell_execution."}')
    result = chat("crash test it")
    assert "waits for you" in result.output
    [(task_id,)] = opened().execute("SELECT id FROM tasks").fetchall()
    [subtask] = Store(opened()).list_subtasks(task_id)
    assert subtask.runs_code and subtask.status is S.NEEDS_HUMAN
    assert not [r for r in fake.calls if r.prompt.startswith("Execute")]


def test_a_failed_copy_leaves_no_task_for_a_worker_to_claim(
    fake: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copies are made in the transaction that creates the task: a worker cannot claim
    work whose files are not there yet, and a copy that fails leaves nothing runnable."""

    def test_it(request: object) -> str:
        turn = {"type": "task", "title": "Test", "goal": "Test calc.py"}
        return json.dumps({**turn, "files_from": [first_task()[:8]]})

    fake.on(CHAT, TASK, test_it).on("Plan task", PLAN, PLAN)
    fake.on("Execute subtask", WRITE, '{"files_written": ["calc.py"]}')
    fake.on("Review subtask", '{"verdict": "accept"}').on(RETELL, '{"text": "Done."}')
    seen: list[int] = []

    def broken(root: Path, target: str, given: task_files.Handover) -> list[str]:
        # What another connection (a worker) sees while the copy runs: not the new task.
        reader = sqlite3.connect(f"file:{db.default_db_path()}?mode=ro", uri=True)
        seen.append(len(reader.execute("SELECT id FROM tasks").fetchall()))
        reader.close()
        raise OSError("disk full")

    monkeypatch.setattr(task_files, "hand_over", broken)
    result = chat("write me a calculator", "now test it")
    assert "could not be copied, so the task was not started: disk full" in result.output
    assert seen == [1]
    assert len(opened().execute("SELECT id FROM tasks").fetchall()) == 1


def test_files_over_the_limit_are_neither_named_nor_copied_and_the_person_is_told(
    fake: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(task_files, "MAX_HANDED_BYTES", len(CODE) + 10)
    big = json.dumps(
        {"tool_calls": [{"tool": "write_file", "args": {"path": "big.txt", "content": "x" * 500}}]}
    )

    def test_it(request: object) -> str:
        turn = {"type": "task", "title": "Test", "goal": "Test calc.py"}
        return json.dumps({**turn, "files_from": [first_task()[:8]]})

    fake.on(CHAT, TASK, test_it).on("Plan task", PLAN, PLAN)
    fake.on("Execute subtask", big, WRITE, '{"files_written": ["calc.py"]}', '{"read": true}')
    fake.on("Review subtask", '{"verdict": "accept"}', '{"verdict": "accept"}')
    fake.on(RETELL, '{"text": "Done."}', '{"text": "Tested."}')
    result = chat("write me a calculator", "now test it")
    assert "too many or too large to copy" in result.output and "big.txt" in result.output
    ref = first_task()[:8]
    plans = [r for r in fake.calls if r.prompt.startswith("Plan task")]
    assert f"from-{ref}/calc.py" in plans[-1].prompt and "big.txt" not in plans[-1].prompt


def test_a_hard_link_in_a_workspace_is_never_read_or_listed(tmp_path: Path) -> None:
    """Generated code could link a file from outside: it is the outside file, not the task's."""
    import os

    from desman.core.workspace import PathEscapeError

    outside = tmp_path / "secret.txt"
    outside.write_text("not the task's")
    workspace = Workspace(tmp_path / "ws", "a" * 32)
    workspace.write("own.txt", "mine")
    os.link(outside, workspace.root / "linked.txt")
    assert workspace.files() == ["own.txt"]
    with pytest.raises(PathEscapeError, match="hard links"):
        workspace.read("linked.txt")
    assert task_files.for_retelling(tmp_path / "ws", "a" * 32) == [
        {"path": "own.txt", "bytes": 4, "content": "mine", "untrusted": task_files.UNTRUSTED}
    ]
