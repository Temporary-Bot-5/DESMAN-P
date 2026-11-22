"""Master delegation, chat opt-in, and continuation (ROADMAP M.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from desman import actions, commands, runs, ui
from desman.agents import EXECUTE, REVIEW, llm_executor, llm_reviewer
from desman.chat import Chat, _commands
from desman.cli import continued
from desman.config import open_config
from desman.core import stats
from desman.core.audit import check_history, history
from desman.core.context import CHARS_PER_TOKEN, estimate_tokens
from desman.core.coordinator import Coordinator, Crew, Delegation, Failure, Success, Work
from desman.core.models import MAX_DELEGATIONS, Kind, Origin, SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.retry import Strategy
from desman.core.sessions import Role, Sessions
from desman.core.states import SubtaskStatus as S
from desman.core.states import TaskStatus
from desman.core.store import StaleStateError, Store
from desman.interlocutor import CHAT
from desman.llm.cache import Cached
from desman.llm.fake import FakeLLM
from desman.planner import CONTINUE, MASTER_CRITERIA, master_plan
from desman.task_files import Given
from helpers import make_config
from test_continue import plan_of
from test_interlocutor import calls, payload
from test_master import GROQ, work_of

PARTS: list[dict[str, JsonValue]] = [
    {"description": "first piece", "acceptance_criteria": "first fact"},
    {
        "description": "second piece",
        "acceptance_criteria": "second fact",
        "kind": "text",
        "depends_on": [0],
    },
]
NOTE = "Outlined the answer. I will connect the two facts."
REPLY = json.dumps({"type": "delegate", "note": NOTE, "parts": PARTS})
ANSWER = '{"text": "Done"}'
ACCEPT = '{"verdict": "accept"}'


def parts() -> list[SubtaskSpec]:
    return [SubtaskSpec.model_validate(p) for p in PARTS]


def delegation() -> Delegation:
    return Delegation(parts=parts(), note=NOTE)


def test_master_delegates_and_resumes_without_spending_a_quality_attempt(
    store: Store, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = TaskSpec(
        goal="Research a topic",
        subtasks=[
            SubtaskSpec(description="background", acceptance_criteria="given"),
            SubtaskSpec(
                description="Research a topic",
                acceptance_criteria=MASTER_CRITERIA,
                kind=Kind.ANALYSIS,
                master=True,
                inputs={"topic": "given"},
                knowledge=["Manual/topic"],
                depends_on=[0],
            ),
        ],
    )
    task = store.create_task(spec)
    seed, master = store.list_subtasks(task)
    llm = (
        FakeLLM()
        .on(
            EXECUTE,
            '{"text": "Background"}',
            REPLY,
            '{"text": "Fact A"}',
            '{"text": "Fact B"}',
            ANSWER,
        )
        .on(REVIEW, ACCEPT, repeat_last=True)
    )
    coordinator = Coordinator(store, llm_executor(llm), llm_reviewer(llm))
    events: list[str] = []
    assert coordinator.tick(task, events)  # execute the dependency
    assert coordinator.tick(task, events)  # review the dependency and delegate
    first, second = store.list_subtasks(task)[2:]
    assert store.get_subtask(master.id).status is S.PENDING
    assert store.get_subtask(master.id).attempts.quality == 0
    assert len(calls(llm, REVIEW)) == 1
    assert f"delegated {master.id} into 2" in events
    assert store.delegations(master.id) == store.executions(master.id) == 1
    [entry] = [e for e in history(store.conn, task) if e.action == "delegated"]
    assert entry.details["note"] == NOTE and entry.details["parts"] == [first.id, second.id]
    assert [first.kind, second.kind] == [Kind.ANALYSIS, Kind.TEXT]
    for part in (first, second):
        assert part.parent_id == master.id and not part.master
        assert part.inputs == master.inputs and part.knowledge == master.knowledge
        assert part.priority == master.priority
    assert store.dependencies(first.id) == [seed.id]
    assert set(store.dependencies(second.id)) == {seed.id, first.id}
    assert [s.part_of for s in task_report(store, task).subtasks] == [
        None,
        None,
        master.seq,
        master.seq,
    ]

    assert coordinator.run(task).status is TaskStatus.DONE
    _, initial, part_call, _, resumed = calls(llm, EXECUTE)
    assert "Delegations left: 2" in initial.messages[0].content
    assert "Delegations left" not in part_call.messages[0].content
    assert "Delegations left: 1" in resumed.messages[0].content
    resume_context = payload(resumed)
    assert resume_context["your_earlier_delegations"] == store.delegation_notes(master.id)
    assert "Fact A" in resumed.prompt and "Fact B" in resumed.prompt
    starts = [
        e.details["strategy"]
        for e in history(store.conn, task)
        if e.subtask_id == master.id and e.action == "attempt_started"
    ]
    assert starts == ["first", "first"]
    assert len(calls(llm, REVIEW)) == 4
    assert store.final_result(task) == {"text": "Done"}
    report = stats.collect(store.conn)
    counts = report.subtasks
    assert counts.split == 0 and counts.delegated == 1
    assert counts.retried == 0 and counts.first_try == 4
    assert report.strategies == [stats.StrategyUse("first", started=4, accepted=4)]
    assert report.as_json()["subtasks"]["delegated"] == 1
    actions.print_stats(report, "all")
    assert "Delegated parts to the cluster: 1." in capsys.readouterr().out
    assert check_history(store.conn, task) == []


def test_delegation_notes_reduce_the_budget_for_dependency_results(store: Store) -> None:
    task = store.create_task(master_plan("Write an outline"))
    [master] = store.list_subtasks(task)
    note = "Outline: " + "x" * 200
    budget = 110
    resumed: list[Work] = []

    def execute(work: Work) -> Success | Delegation:
        if not work.subtask.master:
            return Success(content={"text": "Fact " * 100})
        if not work.context.delegations:
            return Delegation(parts=parts(), note=note)
        resumed.append(work)
        return Success(content={"text": "Done"})

    reviewer = llm_reviewer(FakeLLM().on(REVIEW, ACCEPT, repeat_last=True))
    coordinator = Coordinator(store, execute, reviewer, context_budget_tokens=budget)
    assert coordinator.run(task).status is TaskStatus.DONE
    [work] = resumed
    notes = store.delegation_notes(master.id)
    assert work.context.delegations == notes
    reserved = estimate_tokens(json.dumps(notes, ensure_ascii=False))
    assert 0 < reserved < budget
    dependencies = work.context.dependencies
    assert all(dep.truncated for dep in dependencies)
    assert sum(len(dep.content) for dep in dependencies) == (budget - reserved) * CHARS_PER_TOKEN


@pytest.mark.parametrize("mode", ["available", "exhausted", "combine", "ordinary"])
def test_delegate_reply_and_prompt_eligibility(store: Store, tmp_path: Path, mode: str) -> None:
    spec = master_plan("Produce the answer")
    if mode == "ordinary":
        spec.subtasks[0].master = False
    work = work_of(store, tmp_path, spec)
    context = work.context.model_copy(update={"delegations_left": int(mode != "exhausted")})
    if mode == "combine":
        context = context.model_copy(update={"strategy": Strategy.COMBINE})
    malformed = '{"type":"delegate","parts":[]}'
    if mode == "available":
        malformed = json.dumps(
            {
                "type": "delegate",
                "note": NOTE,
                "parts": [{**PARTS[0], "depends_on": [9]}, PARTS[1]],
            }
        )
    llm = FakeLLM().on(EXECUTE, malformed, REPLY)
    outcome = llm_executor(llm)(work.model_copy(update={"context": context}))
    assert len(llm.calls) == 2
    assert ("Delegations left: 1" in llm.calls[0].messages[0].content) == (mode == "available")
    if mode == "available":
        assert outcome == delegation()
    else:
        assert isinstance(outcome, Failure) and "delegation is not available" in outcome.reason


def test_delegation_is_never_cached(store: Store, tmp_path: Path) -> None:
    work = work_of(store, tmp_path, master_plan("Produce the answer"))
    context = work.context.model_copy(update={"delegations_left": MAX_DELEGATIONS})
    work = work.model_copy(update={"context": context})
    llm = FakeLLM().on(EXECUTE, REPLY, ANSWER)
    execute = llm_executor(Cached(llm, store.conn, "fake"), clock=store.clock)
    assert execute(work) == delegation()
    assert execute(work) == Success(content={"text": "Done"})
    assert len(llm.calls) == 2


def test_invalid_delegation_graph_fails_atomically(store: Store) -> None:
    task = store.create_task(master_plan("Big goal"))
    [master] = store.list_subtasks(task)
    bad_parts = parts()
    bad_parts[0].depends_on = [1]  # the second part already depends on the first
    outcome = Delegation(parts=bad_parts, note=NOTE)
    coordinator = Coordinator(store, lambda _: outcome, llm_reviewer(FakeLLM()))
    assert coordinator.tick(task, [])
    assert store.get_subtask(master.id).attempts.quality == 1
    assert store.parts(master.id) == []
    assert store.delegations(master.id) == 0
    assert check_history(store.conn, task) == []


def test_delegation_limit_and_a_delegated_master_cannot_split(store: Store) -> None:
    task = store.create_task(master_plan("Big goal"))
    master = store.list_subtasks(task)[0]

    def execute(work: Work) -> Delegation | Success:
        if work.subtask.master:
            if store.delegations(master.id):
                with pytest.raises(ValueError, match="cannot be split"):
                    store.decompose(master.id, work.subtask.lease_token or "", parts())
            if store.delegations(master.id) == MAX_DELEGATIONS:
                with pytest.raises(ValueError, match="delegations left"):
                    store.delegate(master.id, work.subtask.lease_token or "", parts(), NOTE)
            return delegation()
        return Success(content={"text": "Fact"})

    def no_split(work: Work) -> list[SubtaskSpec]:
        pytest.fail("already delegated, so decompose must never run")

    coordinator = Coordinator(
        store,
        execute,
        llm_reviewer(FakeLLM().on(REVIEW, ACCEPT, repeat_last=True)),
        decomposer=no_split,
    )
    assert coordinator.run(task).status is TaskStatus.NEEDS_HUMAN
    assert len(store.parts(master.id)) == 2 * MAX_DELEGATIONS
    assert set(store.dependencies(master.id)) == set(store.parts(master.id))
    first_batch, next_batch = store.delegation_notes(master.id)
    for part in next_batch["parts"]:
        assert set(first_batch["parts"]) <= set(store.dependencies(part))
    assert any(
        "reached its delegation limit" in e.details.get("reason", "")
        for e in history(store.conn, task)
    )
    assert check_history(store.conn, task) == []


def test_split_master_cannot_delegate(store: Store) -> None:
    task = store.create_task(master_plan("Big goal"))
    master = store.claim("worker", task_id=task)
    assert master is not None and master.lease_token is not None
    store.decompose(master.id, master.lease_token, parts())

    def execute(work: Work) -> Delegation | Success:
        if work.subtask.master:
            assert work.context.strategy is Strategy.COMBINE
            assert work.context.delegations_left == MAX_DELEGATIONS
            with pytest.raises(ValueError, match="unsplit master"):
                store.delegate(master.id, work.subtask.lease_token or "", parts(), NOTE)
            return delegation()
        return Success(content={"text": "Fact"})

    coordinator = Coordinator(
        store, execute, llm_reviewer(FakeLLM().on(REVIEW, ACCEPT, repeat_last=True))
    )
    events: list[str] = []
    while not any(e.startswith("quality") for e in events):
        assert coordinator.tick(task, events)
    assert store.delegations(master.id) == 0
    assert any("already split" in e.details.get("reason", "") for e in history(store.conn, task))
    counts = stats.collect(store.conn).subtasks
    assert counts.split == 1 and counts.delegated == 0


@pytest.mark.parametrize("origin", [Origin.LOCAL, Origin.NETWORK])
def test_ordinary_executors_cannot_delegate(store: Store, origin: Origin) -> None:
    task = store.create_task(TaskSpec(goal="g", subtasks=parts()), origin=origin)
    sub = store.list_subtasks(task)[0]
    reviewer = llm_reviewer(FakeLLM())
    coordinator = Coordinator(
        store,
        lambda _: delegation(),
        reviewer,
        network=Crew(lambda _: delegation(), reviewer, None),
    )
    events: list[str] = []
    assert coordinator.tick(task, events)
    assert any(e.startswith(f"quality {sub.id}") for e in events)
    assert not store.parts(sub.id)


def test_lost_lease_discards_delegation(store: Store) -> None:
    task = store.create_task(master_plan("Big goal"))
    [master] = store.list_subtasks(task)

    def release_then_delegate(_work: Work) -> Delegation:
        store.cancel_task(task, "cancelled while executing")
        return delegation()

    events: list[str] = []
    coordinator = Coordinator(store, release_then_delegate, llm_reviewer(FakeLLM()))
    assert coordinator.tick(task, events)
    assert f"lost lease {master.id}" in events
    [discarded] = [e for e in history(store.conn, task) if e.action == "work_discarded"]
    assert discarded.details["reason"] == "lease lost before delegation"
    assert store.parts(master.id) == []
    assert store.delegations(master.id) == 0
    with pytest.raises(StaleStateError):
        store.delegate(master.id, "stale lease", parts(), NOTE)


@pytest.mark.parametrize("selection", ["turn", "switch", "temporary", "catalog"])
def test_chat_master_bypasses_planner(
    store: Store,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selection: str,
) -> None:
    turn = {
        "type": "task",
        "goal": "Write an outline",
        "title": "Outline",
        "master": selection == "turn",
    }
    llm = FakeLLM().on(CHAT, json.dumps(turn))
    config = make_config(configured=True, language="en")
    open_config().save(config)
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    chat = Chat(Sessions(store.conn), background_tasks=True, temporary=selection == "temporary")
    if selection in ("switch", "temporary"):
        assert chat._slash("/master")
        assert "Master mode is off" in capsys.readouterr().out
        assert chat._slash("/master on")
        assert chat._slash("/master")
        assert "Master mode is on for this chat until it ends." in capsys.readouterr().out
        assert "until this chat ends" in _commands()["/master"]
    if selection == "catalog":
        parsed = commands.parse_line('run --master "Write an outline"', config)
        assert parsed is not None
        _, args = parsed
        assert isinstance(args, commands.RunArgs) and args.master
        chat._typed('run --master "Write an outline"')
    else:
        chat._message("Write an outline")
    [row] = store.conn.execute("SELECT id FROM tasks")
    [sub] = store.list_subtasks(row["id"])
    assert sub.master and sub.acceptance_criteria == MASTER_CRITERIA
    assert not calls(llm, "Plan task")
    if selection in ("switch", "temporary"):
        assert chat._slash("/master off") and not chat.master_mode


@pytest.mark.parametrize("exit_by", ["delete", "delete_all", "new", "terminal"])
def test_pending_master_mode_does_not_follow_session_navigation(
    store: Store, monkeypatch: pytest.MonkeyPatch, exit_by: str
) -> None:
    sessions = Sessions(store.conn)
    existing = sessions.add(None, Role.USER, "Existing chat")
    selected = sessions.get(existing.session_id)
    assert selected is not None
    open_config().save(make_config(configured=True, language="en"))
    llm = FakeLLM().on(CHAT, '{"type":"reply","text":"Hello"}')
    monkeypatch.setattr(runs, "models", lambda *a, **kw: runs.Models(llm, None))
    monkeypatch.setattr(ui, "confirm", lambda *a, **kw: True)
    chat = Chat(sessions, background_tasks=True)
    assert chat._slash("/master on") and chat.master_mode
    assert not chat.master_sessions
    if exit_by == "new":
        chat._new()
    elif exit_by == "terminal":
        chat._terminal_open()
        chat._back()
    else:
        chat._open_session(selected)
        if exit_by == "delete":
            chat._delete()
        else:
            chat._delete_all()
    assert chat.session_id is None
    chat._message("New message")
    assert chat.session_id is not None and not chat.master_mode
    assert not chat.master_sessions


def test_chat_without_a_master_keeps_files_pending(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_config(configured=True, language="en", api_keys={"GROQ_API_KEY": GROQ})
    open_config().save(config)
    real = runs.models(config, store.conn)
    llm = FakeLLM().on(CHAT, '{"type":"task","goal":"Read input","title":"Read","master":true}')
    monkeypatch.setattr(
        runs,
        "models",
        lambda *a, **kw: runs.Models(llm, None, choice=real.choice, no_master=real.no_master),
    )
    chat = Chat(Sessions(store.conn), background_tasks=True)
    chat.attach_file(Given("facts.txt", "facts", 5, "facts.txt"))
    request = "Read the attached file"
    chat.session_id = chat._add_user(request)
    seq = store.conn.execute("SELECT MAX(seq) FROM messages").fetchone()[0]
    chat.sessions.register_turn(seq)
    chat.sessions.reply_to = seq
    chat._message(request, added=True)
    assert not chat.sessions.turn_settled(seq)
    output = capsys.readouterr()
    assert (
        real.no_master is not None
        and "No model ready for a master module" in output.out + output.err
    )
    assert not list(store.conn.execute("SELECT id FROM tasks"))
    assert chat.pending_files[chat.session_id or ""].snapshot()[0].name == "facts.txt"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("mode", ["extend", "redo", "fallback", "cluster"])
def test_continuations_keep_master_when_ready(
    store: Store, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    spec = master_plan("Original goal")
    spec.subtasks[0].master = mode != "cluster"
    task = store.create_task(spec)
    coordinator = Coordinator(
        store,
        lambda _: Success(content={"text": "Original result"}),
        llm_reviewer(FakeLLM().on(REVIEW, ACCEPT)),
    )
    assert coordinator.run(task).status is TaskStatus.DONE
    [old] = store.list_subtasks(task)
    pending = actions.prepare_continuation(
        store,
        None if mode == "redo" else task,
        "Update the details",
        old.id if mode == "redo" else None,
    )
    config = make_config(language="en", api_keys={"GROQ_API_KEY": GROQ})
    llm = FakeLLM().on(CONTINUE, plan_of("Update the details", "Give complete answer"))
    if mode == "fallback":
        ready = runs.models(config, store.conn)
        models = runs.Models(llm, None, choice=ready.choice, no_master=ready.no_master)
    else:
        models = runs.Models(llm, None)
    _, count = continued(store, pending, models, config)
    new = store.list_subtasks(task)[1:]
    [entry] = [e for e in history(store.conn, task) if e.action == "task_extended"]
    assert check_history(store.conn, task) == []
    if mode in ("extend", "redo"):
        assert count == 1 and not llm.calls
        [sub] = new
        assert sub.master and sub.acceptance_criteria == MASTER_CRITERIA
        assert sub.description == (
            f"Redo step {old.seq} ({old.description}): Update the details"
            if mode == "redo"
            else "Update the details"
        )
        assert entry.details["plan"] == "master" and entry.details["master"] == [sub.id]
    else:
        assert count == 2 and len(llm.calls) == 1
        assert all(not sub.master for sub in new)
        output = capsys.readouterr()
        assert ("continuation is planned for the cluster" in output.out + output.err) == (
            mode == "fallback"
        )
