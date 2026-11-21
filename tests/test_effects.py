"""Effect journal (ROADMAP 3.4, D6): rerunning an interrupted execution does not repeat effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from conftest import FakeClock
from desman.agents import llm_executor
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.effects import INTERRUPTED, Effects, evidence
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.policy import Policy, Role, RolePolicy
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import StaleStateError, Store
from desman.core.tools import (
    LIST_FILES,
    READ_FILE,
    WRITE_FILE,
    Access,
    Permission,
    Registry,
    Tool,
    Toolbox,
    ToolCall,
    ToolResult,
    WriteFileArgs,
)
from desman.core.workspace import Workspace
from desman.llm.cache import Cached
from desman.llm.fake import FakeLLM, ServerError

S = SubtaskStatus
TOOLS = ["bump", "read_file", "write_file", "list_files"]
ACCESS = Access(Policy(roles={Role.EXECUTOR: RolePolicy(tools=TOOLS)}), Role.EXECUTOR)


class Killed(BaseException):
    """The process dies here (like SIGKILL: nothing in DESMAN catches it)."""


class BumpArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = 1


def bump_tool(ran: list[int], *, killed_at: int | None = None) -> Tool[BumpArgs]:
    """An effect that is visibly not idempotent: every run is counted."""

    def run(args: BumpArgs, workspace: Workspace) -> dict[str, Any]:
        ran.append(args.n)
        if len(ran) == killed_at:
            raise Killed  # after the effect, before its result is recorded
        return {"runs": len(ran)}

    return Tool("bump", "Count a run.", BumpArgs, Permission.WORKSPACE, run)


def registry(bump: Tool[BumpArgs]) -> Registry:
    return Registry([bump, READ_FILE, WRITE_FILE, LIST_FILES])


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="act", acceptance_criteria="ok")])


@pytest.fixture
def claimed(store: Store) -> Subtask:
    store.create_task(one_step())
    subtask = store.claim("w1")
    assert subtask is not None
    return subtask


def toolbox(store: Store, subtask: Subtask, root: Path, tools: Registry) -> Toolbox:
    assert subtask.lease_token is not None

    def record(action: str, details: dict[str, Any]) -> None:
        store.note(subtask.id, Actor.EXECUTOR, action, details)

    return Toolbox(
        Workspace(root, subtask.task_id),
        tools,
        access=ACCESS,
        record=record,
        journal=Effects(store, subtask.id, subtask.lease_token),
    )


def crash_and_reclaim(store: Store) -> Subtask:
    """The executor's process is gone: its lease is taken back and the subtask claimed again."""
    store.recover_expired(abandoned=lambda owner: True)
    subtask = store.claim("w2")
    assert subtask is not None
    return subtask


def bump(n: int) -> ToolCall:
    return ToolCall(tool="bump", args={"n": n})


def tool_entries(store: Store, task_id: str) -> list[tuple[str, dict[str, Any]]]:
    return [
        (row["action"], json.loads(row["details_json"]))
        for row in store.audit_log(task_id)
        if row["action"].startswith("tool_")
    ]


def test_a_rerun_replays_effects_instead_of_repeating_them(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    ran: list[int] = []
    tools = registry(bump_tool(ran))
    box = toolbox(store, claimed, tmp_path, tools)
    first = [box.call(bump(1)), box.call(bump(2))]

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    assert [box.call(bump(1)), box.call(bump(2))] == first  # what the model saw the first time
    assert ran == [1, 2]
    assert box.call(bump(3)).output == {"runs": 3}  # past the journal: runs for real
    assert ran == [1, 2, 3]
    assert [(action, d["args"], d["ok"]) for action, d in tool_entries(store, claimed.task_id)] == [
        ("tool_called", {"n": 1}, True),
        ("tool_called", {"n": 2}, True),
        ("tool_replayed", {"n": 1}, True),
        ("tool_replayed", {"n": 2}, True),
        ("tool_called", {"n": 3}, True),
    ]
    assert check_history(store.conn, claimed.task_id) == []


def test_the_first_different_call_ends_the_replay(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    ran: list[int] = []
    tools = registry(bump_tool(ran))
    box = toolbox(store, claimed, tmp_path, tools)
    box.call(bump(1))
    box.call(bump(2))

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    box.call(bump(1))
    box.call(bump(5))  # the run went another way from here
    box.call(bump(2))  # same call as before, but the workspace is not where it was then
    assert ran == [1, 2, 5, 2]

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    for n in (1, 5, 2):
        box.call(bump(n))
    assert ran == [1, 2, 5, 2]  # the journal now follows the latest run
    assert [row["tool"] for row in store.effects(claimed.id, 0)] == ["bump"] * 3


def test_an_interrupted_call_is_not_run_again(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    ran: list[int] = []
    tools = registry(bump_tool(ran, killed_at=2))
    box = toolbox(store, claimed, tmp_path, tools)
    box.call(bump(1))
    with pytest.raises(Killed):
        box.call(bump(2))

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    box.call(bump(1))
    interrupted = box.call(bump(2))
    assert interrupted == ToolResult(ok=False, error=INTERRUPTED, at=interrupted.at)
    assert interrupted.at is not None
    assert ran == [1, 2]  # its effects may have happened: not risked twice
    action, details = tool_entries(store, claimed.task_id)[-1]
    assert action == "tool_replayed" and details["interrupted"] is True
    # The model checked and asks again: a new call, so it runs.
    assert box.call(bump(2)).ok
    assert ran == [1, 2, 2]

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    assert [box.call(bump(n)) for n in (1, 2, 2)][1] == interrupted  # the same answer each time
    assert ran == [1, 2, 2]


def test_a_repeatable_call_is_run_again_after_an_interruption(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    assert claimed.lease_token is not None
    args = WriteFileArgs(path="a.txt", content="hello")
    # Intent recorded, then the process died before the file was written.
    assert Effects(store, claimed.id, claimed.lease_token).begin(WRITE_FILE, args) is None

    box = toolbox(store, crash_and_reclaim(store), tmp_path, registry(bump_tool([])))
    result = box.call(ToolCall(tool="write_file", args=args.model_dump()))
    assert (
        result.ok and (tmp_path / claimed.task_id / "a.txt").read_text(encoding="utf-8") == "hello"
    )
    [row] = store.effects(claimed.id, 0)
    assert row["state"] == "done"


def write(path: str, content: str) -> ToolCall:
    return ToolCall(tool="write_file", args={"path": path, "content": content})


def read(path: str) -> ToolCall:
    return ToolCall(tool="read_file", args={"path": path})


def test_reads_are_replayed_as_they_were(store: Store, claimed: Subtask, tmp_path: Path) -> None:
    tools = registry(bump_tool([]))
    box = toolbox(store, claimed, tmp_path, tools)
    box.call(write("a.txt", "draft"))
    seen = box.call(read("a.txt"))
    box.call(write("a.txt", "final"))

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    box.call(write("a.txt", "draft"))
    # Read now, the file would say "final": the replay shows what the run saw at this point.
    assert box.call(read("a.txt")) == seen and seen.output["content"] == "draft"
    box.call(write("a.txt", "final"))
    assert (tmp_path / claimed.task_id / "a.txt").read_text(encoding="utf-8") == "final"


def test_a_resume_from_a_checkpoint_replays_from_there(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    assert claimed.lease_token is not None
    ran: list[int] = []
    tools = registry(bump_tool(ran))
    box = toolbox(store, claimed, tmp_path, tools)
    box.call(bump(1))
    assert isinstance(box.journal, Effects)
    store.heartbeat(
        claimed.id, claimed.lease_token, checkpoint={"step": 1}, effects=box.journal.position
    )
    box.call(bump(2))

    # The executor resumes after step 1 and makes the same call for step 2: it is not step 1's.
    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    box.call(bump(1))
    assert ran == [1, 2, 1]
    assert [row["ordinal"] for row in store.effects(claimed.id, 0)] == [0, 1]


def test_a_new_attempt_acts_again(store: Store, claimed: Subtask, tmp_path: Path) -> None:
    assert claimed.lease_token is not None
    ran: list[int] = []
    tools = registry(bump_tool(ran))
    toolbox(store, claimed, tmp_path, tools).call(bump(1))
    store.submit(claimed.id, claimed.lease_token, {"done": True})
    store.record_failure(
        claimed.id, S.REVIEW, Actor.REVIEWER, FailureKind.QUALITY, reason="do it again"
    )

    again = store.claim("w1")
    assert again is not None and store.executions(again.id) == 1
    toolbox(store, again, tmp_path, tools).call(bump(1))
    assert ran == [1, 1]  # rejected work is redone, not replayed


def test_a_late_result_from_a_lost_lease_is_ignored(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    assert claimed.lease_token is not None
    ran: list[int] = []
    tools = registry(bump_tool(ran))
    stalled = Effects(store, claimed.id, claimed.lease_token)
    assert stalled.begin(tools.get("bump"), BumpArgs(n=1)) is None  # type: ignore[arg-type]

    box = toolbox(store, crash_and_reclaim(store), tmp_path, tools)
    assert box.call(bump(1)).error == INTERRUPTED
    stalled.finish(ToolResult(ok=True, output={"late": True}))  # its process woke up
    [row] = store.effects(claimed.id, 0)
    assert row["state"] == "interrupted"
    assert json.loads(row["result_json"])["error"] == INTERRUPTED


def test_nothing_runs_without_the_lease(store: Store, claimed: Subtask, tmp_path: Path) -> None:
    ran: list[int] = []
    box = toolbox(store, claimed, tmp_path, registry(bump_tool(ran)))
    store.cancel_task(claimed.task_id, "operator stopped it")
    with pytest.raises(StaleStateError):
        box.call(bump(1))
    assert ran == [] and store.effects(claimed.id, 0) == []


# ── Through the coordinator, with a real executor ───────────────────────────


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def test_a_provider_failure_mid_execution_does_not_repeat_the_effect(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    ran: list[int] = []
    fake = FakeLLM().on(
        "Execute",
        json.dumps({"tool_calls": [{"tool": "bump", "args": {"n": 7}}]}),
        ServerError(),  # the attempt is interrupted after the effect
        '{"summary": "bumped once"}',
    )
    task_id = store.create_task(one_step())
    coordinator = Coordinator(
        store,
        llm_executor(Cached(fake, store.conn, "m", clock=clock)),
        accept,
        workspaces=tmp_path,
        tools=registry(bump_tool(ran)),
        access=ACCESS,
        sleep=clock.sleep,
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert ran == [7]  # the rerun got the same call from the cache and replayed it
    assert len(fake.calls) == 3
    assert [action for action, _ in tool_entries(store, task_id)] == [
        "tool_called",
        "tool_replayed",
    ]
    assert check_history(store.conn, task_id) == []


def test_the_coordinator_saves_the_journal_position_with_a_checkpoint(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    ran: list[int] = []

    def executor(work: Work) -> Success:
        if work.context.checkpoint is None:
            work.tools.call(bump(1))  # step 1
            work.heartbeat({"step": 1})
            raise Killed
        work.tools.call(bump(1))  # step 2 happens to make the same call
        return Success(content={"steps": 2})

    task_id = store.create_task(one_step())

    def coordinator() -> Coordinator:
        return Coordinator(
            store,
            executor,
            accept,
            workspaces=tmp_path,
            tools=registry(bump_tool(ran)),
            access=ACCESS,
            abandoned=lambda owner: True,
            sleep=clock.sleep,
        )

    with pytest.raises(Killed):
        coordinator().run(task_id)
    assert coordinator().run(task_id).status is TaskStatus.DONE
    assert ran == [1, 1]
    assert check_history(store.conn, task_id) == []


# ── Evidence for the reviewer ────────────────────────────────────────────────


def test_the_reviewer_sees_what_the_tools_gave_the_attempt_under_review(
    store: Store, claimed: Subtask, tmp_path: Path
) -> None:
    assert claimed.lease_token is not None
    tools = registry(bump_tool([]))
    toolbox(store, claimed, tmp_path, tools).call(bump(1))
    store.submit(claimed.id, claimed.lease_token, {"done": True})
    [call] = evidence(store, claimed)["tool_results"]
    assert call == {
        "tool": "bump",
        "at": call["at"],
        "result": {"ok": True, "output": {"runs": 1}, "error": None},
        "truncated": False,
    }

    store.record_failure(
        claimed.id, S.REVIEW, Actor.REVIEWER, FailureKind.QUALITY, reason="do it again"
    )
    again = store.claim("w1")
    assert again is not None and again.lease_token is not None
    box = toolbox(store, again, tmp_path, tools)
    box.call(bump(2))
    box.call(bump(3))
    store.submit(again.id, again.lease_token, {"done": True})
    # Only this attempt's calls: the rejected attempt's results are not what this one rests on.
    shown = evidence(store, again)["tool_results"]
    assert [c["result"]["output"]["runs"] for c in shown] == [2, 3]


def test_evidence_holds_inputs_and_dependencies_and_fits_its_budget(store: Store) -> None:
    store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="find", acceptance_criteria="ok"),
                SubtaskSpec(
                    description="pick",
                    acceptance_criteria="ok",
                    depends_on=[0],
                    inputs={"city": "Lisbon"},
                ),
            ],
        )
    )
    find = store.claim("w")
    assert find is not None and find.lease_token is not None
    store.submit(find.id, find.lease_token, {"flights": ["TP 1351, 61 EUR"] * 400})
    store.transition(find.id, S.REVIEW, S.DONE, Actor.REVIEWER, reason="ok")
    pick = store.claim("w")
    assert pick is not None and pick.description == "pick"

    shown = evidence(store, pick, budget_tokens=100)
    assert shown["inputs"] == {"city": "Lisbon"}  # small: whole
    [dependency] = shown["results_of_dependencies"]
    assert dependency["subtask"] == "find" and dependency["truncated"]
    assert "TP 1351" in dependency["result"] and len(json.dumps(shown)) < 1000
    assert shown["tool_results"] == []


def test_retry_sees_stale_files_sizes_and_writer_attempt_within_budget(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    from desman.core.context import build_context
    from desman.core.effects import workspace_files

    fake = FakeLLM().on(
        "Execute",
        '{"tool_calls":[{"tool":"write_file","args":{"path":"tasks.json","content":"[1,2]"}}]}',
        '{"done":true}',
        '{"done":true}',
    )
    reviews = iter(
        [Review(verdict=Verdict.RETRY, feedback="reset demo"), Review(verdict=Verdict.ACCEPT)]
    )
    task = store.create_task(one_step())
    coordinator = Coordinator(
        store,
        llm_executor(fake, format_final=False),
        lambda sub, result: next(reviews),
        tools=Registry([WRITE_FILE]),
        access=ACCESS,
        workspaces=tmp_path,
        sleep=clock.sleep,
    )
    assert coordinator.run(task).status is TaskStatus.DONE
    retry = json.loads(fake.calls[-1].prompt.split(":\n", 1)[1])
    assert retry["workspace_files"] == [{"path": "tasks.json", "size": 5, "earlier_attempt": 1}]
    assert "reset or rewrite" in retry["workspace_files_note"]
    sub = store.get_subtask(
        store.conn.execute("SELECT id FROM subtasks WHERE task_id = ?", (task,)).fetchone()[0]
    )
    ws = Workspace(tmp_path, task)
    ws.write("unknown.txt", "other")
    files = workspace_files(store, sub, ws, 4000)
    assert files[-1] == {"path": "unknown.txt", "size": 5}
    assert workspace_files(store, sub, ws, 1) == []
    called: list[int] = []

    def inventory(chars: int) -> list[dict[str, Any]]:
        called.append(chars)
        return workspace_files(store, sub, ws, chars)

    build_context(
        sub,
        [],
        None,
        budget_tokens=40,
        files=inventory,
    )
    assert called == [40]


def test_quote_sources_include_the_documents_a_planned_subtask_works_from(store: Store) -> None:
    from desman.core.effects import quote_sources
    from desman.core.workspace import workspace_root_for

    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(
                    description="answer",
                    acceptance_criteria="c",
                    inputs={"read_in_parts": ["input/book.txt", "input/gone.txt", "../x"]},
                )
            ],
        )
    )
    Workspace(workspace_root_for(store.conn), task_id).write("input/book.txt", "Вдали был лес.")
    [subtask] = store.list_subtasks(task_id)
    sources = quote_sources(store, subtask, knowledge=None, context_budget_tokens=1000)
    assert sources == ["Вдали был лес."]


def test_quote_sources_of_a_combine_include_its_document(store: Store) -> None:
    from desman.core.effects import quote_sources
    from desman.core.workspace import workspace_root_for

    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(
                    description="combine",
                    acceptance_criteria="c",
                    inputs={"document": "input/b.txt"},
                )
            ],
        )
    )
    Workspace(workspace_root_for(store.conn), task_id).write("input/b.txt", "Текст книги.")
    [subtask] = store.list_subtasks(task_id)
    assert quote_sources(store, subtask, knowledge=None, context_budget_tokens=1000) == [
        "Текст книги."
    ]


def test_quote_sources_use_normalized_paths_and_task_wide_write_history(store: Store) -> None:
    from desman.core.effects import quote_sources

    store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="write", acceptance_criteria="c"),
                SubtaskSpec(description="read", acceptance_criteria="c", depends_on=[0]),
            ],
        )
    )

    def journal(subtask: Subtask, ordinal: int, tool: str, output: dict[str, str]) -> None:
        assert subtask.lease_token is not None
        store.start_effect(subtask.id, subtask.lease_token, 0, ordinal, tool, str(ordinal))
        assert store.finish_effect(
            subtask.id,
            subtask.lease_token,
            0,
            ordinal,
            ToolResult(ok=True, output=output).model_dump(),
        )

    writer = store.claim("w")
    assert writer is not None and writer.lease_token is not None
    journal(writer, 0, "write_file", {"path": "input/generated.txt"})
    store.submit(writer.id, writer.lease_token, {"done": True})
    store.transition(writer.id, S.REVIEW, S.DONE, Actor.REVIEWER)
    reader = store.claim("w")
    assert reader is not None and reader.lease_token is not None
    attached = {"path": "./input//attached.txt", "content": "The person's words."}
    fetched = {"path": "./web/page.txt", "content": "The page's words."}
    journal(reader, 0, "read_file", {"path": "./input//generated.txt", "content": "Invented."})
    journal(reader, 1, "read_file", attached)
    journal(reader, 2, "read_file", fetched)
    journal(reader, 3, "run_command", {"stdout": "Invented output."})
    page = {"text": "An external source."}
    journal(reader, 4, "web_fetch", page)
    store.submit(reader.id, reader.lease_token, {"done": True})
    assert quote_sources(store, reader, knowledge=None, context_budget_tokens=1000) == [
        attached,
        fetched,
        page,
    ]
