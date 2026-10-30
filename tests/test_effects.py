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
    assert interrupted == ToolResult(ok=False, error=INTERRUPTED)
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
    assert result.ok and (tmp_path / claimed.task_id / "a.txt").read_text() == "hello"
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
    assert (tmp_path / claimed.task_id / "a.txt").read_text() == "final"


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
