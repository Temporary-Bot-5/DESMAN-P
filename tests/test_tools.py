"""Tool registry (ROADMAP 3.1): executors act only through checked, guarded, audited tool calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from conftest import FakeClock
from desman.agents import MAX_TOOL_ROUNDS, TOOL_RESULTS, llm_executor, llm_reviewer
from desman.core import tools as tools_module
from desman.core.audit import check_history
from desman.core.context import build_context
from desman.core.coordinator import Coordinator, Failure, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.policy import Policy, Role, RolePolicy
from desman.core.retry import FailureKind
from desman.core.states import TaskStatus
from desman.core.store import StaleStateError, Store
from desman.core.tools import (
    AUDIT_TEXT,
    Access,
    Permission,
    Registry,
    Tool,
    Toolbox,
    ToolCall,
    builtin_registry,
)
from desman.core.workspace import Workspace
from desman.llm.fake import FakeLLM


class _NoArgs(BaseModel):
    pass


def fetch_tool(ran: list[str]) -> Tool[_NoArgs]:
    def run(args: _NoArgs, workspace: Workspace) -> dict[str, Any]:
        ran.append("fetch")
        return {}

    return Tool(
        name="fetch",
        description="Fetch a page.",
        args=_NoArgs,
        permission=Permission.NETWORK,
        run=run,
    )


class Recorder:
    def __init__(self) -> None:
        self.entries: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, action: str, details: dict[str, Any]) -> None:
        self.entries.append((action, details))


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path, "t1")


def call(tool: str, **args: Any) -> ToolCall:
    return ToolCall(tool=tool, args=args)


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_refuses_duplicate_and_bad_names() -> None:
    registry = builtin_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tools_module.READ_FILE)
    with pytest.raises(ValueError, match="bad tool name"):
        registry.register(Tool("Read File", "", _NoArgs, Permission.WORKSPACE, lambda a, w: {}))
    assert registry.names == ["read_file", "write_file", "list_files"]
    assert builtin_registry() is not registry  # registering is never global


def test_a_spec_declares_the_argument_schema() -> None:
    spec = tools_module.WRITE_FILE.spec()
    assert spec["name"] == "write_file"
    assert spec["args"]["required"] == ["path", "content"]
    assert spec["args"]["additionalProperties"] is False


# ── Toolbox ──────────────────────────────────────────────────────────────────


def test_only_granted_tools_are_offered_and_run(ws: Workspace) -> None:
    ran: list[str] = []
    registry = Registry([*builtin_registry(), fetch_tool(ran)])
    policy = Policy(roles={Role.EXECUTOR: RolePolicy(tools=[*registry.names])})
    record = Recorder()
    box = Toolbox(ws, registry, access=Access(policy, Role.EXECUTOR), record=record)
    assert [s["name"] for s in box.specs()] == ["read_file", "write_file", "list_files"]

    refused = box.call(call("fetch"))
    assert not refused.ok and refused.error is not None and "network_access" in refused.error
    assert ran == []
    [(action, details)] = record.entries
    assert action == "tool_refused"
    assert details["permission"] == "network_access" and "not turned on" in details["reason"]

    access = Access(policy, Role.EXECUTOR, [Permission.NETWORK])
    granted = Toolbox(ws, registry, access=access)
    assert "fetch" in [s["name"] for s in granted.specs()]
    assert granted.call(call("fetch")).ok and ran == ["fetch"]


def test_mistakes_come_back_to_the_model_and_are_audited(ws: Workspace) -> None:
    record = Recorder()
    box = Toolbox(ws, record=record)
    unknown = box.call(call("delete_everything"))
    assert unknown.error is not None and "available: read_file, write_file" in unknown.error
    extra = box.call(call("write_file", path="a.txt", content="x", mode="w"))
    assert extra.error is not None and "mode: Extra inputs are not permitted" in extra.error
    missing = box.call(call("write_file", path="a.txt"))
    assert missing.error is not None and "content: Field required" in missing.error
    escape = box.call(call("write_file", path="/etc/passwd", content="x"))
    assert escape.error is not None and "PathEscapeError" in escape.error
    absent = box.call(call("read_file", path="nope.txt"))
    assert absent.error is not None and "FileNotFoundError" in absent.error
    assert ws.files() == []
    assert [(a, d["tool"], d["ok"]) for a, d in record.entries] == [
        ("tool_called", "delete_everything", False),
        ("tool_called", "write_file", False),
        ("tool_called", "write_file", False),
        ("tool_called", "write_file", False),
        ("tool_called", "read_file", False),
    ]


def test_files_round_trip_through_the_tools(ws: Workspace) -> None:
    box = Toolbox(ws)
    assert box.call(call("write_file", path="./out//a.md", content="# hi")).output == {
        "path": "out/a.md",
        "chars": 4,
    }
    assert box.call(call("read_file", path="out/a.md")).output == {
        "path": "out/a.md",
        "content": "# hi",
    }
    assert box.call(call("list_files")).output == {"files": ["out/a.md"]}


def test_long_and_binary_files_are_handled(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools_module, "MAX_READ_CHARS", 5)
    monkeypatch.setattr(tools_module, "MAX_LISTED_FILES", 1)
    box = Toolbox(ws)
    ws.write("long.txt", "0123456789")
    ws.write("blob.bin", b"\xff\xfe")
    assert box.call(call("read_file", path="long.txt")).output == {
        "path": "long.txt",
        "content": "01234",
        "truncated": True,
        "chars": 10,
    }
    binary = box.call(call("read_file", path="blob.bin"))
    assert binary.error is not None and "not a UTF-8 text file" in binary.error
    assert box.call(call("list_files")).output == {"files": ["blob.bin"], "more": 1}


def test_the_audit_keeps_arguments_short(ws: Workspace) -> None:
    record = Recorder()
    Toolbox(ws, record=record).call(call("write_file", path="big.txt", content="x" * 5000))
    [(_, details)] = record.entries
    content = details["args"]["content"]
    assert content.startswith("x" * AUDIT_TEXT) and content.endswith("…[5000 chars]")
    assert details == {"tool": "write_file", "args": details["args"], "ok": True}


def test_nothing_runs_once_the_lease_is_lost(ws: Workspace) -> None:
    def lost() -> None:
        raise StaleStateError("lease lost")

    box = Toolbox(ws, guard=lost)
    with pytest.raises(StaleStateError):
        box.call(call("write_file", path="a.txt", content="x"))
    assert ws.files() == []


# ── The LLM executor's tool loop ─────────────────────────────────────────────


def one_step() -> TaskSpec:
    return TaskSpec(
        goal="g", subtasks=[SubtaskSpec(description="write a note", acceptance_criteria="ok")]
    )


def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
    return Review(verdict=Verdict.ACCEPT)


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> str:
    return json.dumps({"tool_calls": [{"tool": t, "args": a} for t, a in calls]})


def test_executor_writes_files_through_tool_calls(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    llm = FakeLLM().on(
        "Execute",
        tool_calls(("write_file", {"path": "note.md", "content": "hello"})),
        tool_calls(("read_file", {"path": "note.md"}), ("list_files", {})),
        '{"summary": "wrote note.md", "files": ["note.md"]}',
    )
    task_id = store.create_task(one_step())
    root = tmp_path / "ws"
    coordinator = Coordinator(store, llm_executor(llm), accept, workspaces=root, sleep=clock.sleep)
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert (root / task_id / "note.md").read_text() == "hello"
    [sub] = store.list_subtasks(task_id)
    assert store.latest_artifact(sub.id) == {"summary": "wrote note.md", "files": ["note.md"]}

    first, second, third = llm.calls
    offered = json.loads(first.prompt.split("\n", 1)[1])["tools"]
    assert [t["name"] for t in offered] == ["read_file", "write_file", "list_files"]
    assert '"tool_calls"' in first.messages[0].content
    assert second.prompt.startswith(f"{TOOL_RESULTS}:\n")
    assert json.loads(second.prompt.split("\n", 1)[1]) == [
        {"tool": "write_file", "ok": True, "output": {"path": "note.md", "chars": 5}}
    ]
    results = json.loads(third.prompt.split("\n", 1)[1])
    assert results[0]["output"]["content"] == "hello"
    assert results[1]["output"] == {"files": ["note.md"]}
    assert [m.role for m in third.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    tools_used = [
        json.loads(r["details_json"])["tool"]
        for r in store.audit_log(task_id)
        if r["action"] == "tool_called"
    ]
    assert tools_used == ["write_file", "read_file", "list_files"]
    assert check_history(store.conn, task_id) == []


def test_malformed_tool_calls_get_the_repair_request(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    llm = FakeLLM().on(
        "Execute",
        '{"tool_calls": [{"name": "write_file"}]}',  # "tool" missing, "name" not allowed
        '{"done": true}',
    )
    task_id = store.create_task(one_step())
    coordinator = Coordinator(
        store, llm_executor(llm), accept, workspaces=tmp_path, sleep=clock.sleep
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert llm.calls[1].prompt.startswith("Execute subtask (repair):")
    assert "tool_calls.0.tool" in llm.calls[1].prompt


def no_heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
    pass


def test_endless_tool_calls_fail_the_attempt(store: Store, tmp_path: Path) -> None:
    llm = FakeLLM().on("Execute", tool_calls(("list_files", {})), repeat_last=True)
    store.create_task(one_step())
    claimed = store.claim("w")
    assert claimed is not None
    work = Work(
        subtask=claimed,
        context=build_context(claimed, [], None),
        heartbeat=no_heartbeat,
        tools=Toolbox(Workspace(tmp_path, claimed.task_id)),
    )
    outcome = llm_executor(llm)(work)
    assert isinstance(outcome, Failure) and outcome.kind is FailureKind.QUALITY
    assert "still calling tools" in outcome.reason
    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1


def test_a_cancelled_task_writes_nothing_more(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    def executor(work: Work) -> Success:
        store.cancel_task(work.subtask.task_id, "operator stopped it")
        work.tools.call(call("write_file", path="late.txt", content="x"))
        return Success(content={})

    task_id = store.create_task(one_step())
    root = tmp_path / "ws"
    result = Coordinator(store, executor, accept, workspaces=root, sleep=clock.sleep).run(task_id)
    assert result.status is TaskStatus.CANCELLED
    assert not (root / task_id / "late.txt").exists()
    assert "work_discarded" in [r["action"] for r in store.audit_log(task_id)]


def test_reviewer_gets_no_tools(store: Store, clock: FakeClock, tmp_path: Path) -> None:
    llm = FakeLLM().on("Execute", '{"ok": 1}').on("Review", '{"verdict": "accept"}')
    task_id = store.create_task(one_step())
    Coordinator(
        store, llm_executor(llm), llm_reviewer(llm), workspaces=tmp_path, sleep=clock.sleep
    ).run(task_id)
    review = next(r for r in llm.calls if r.prompt.startswith("Review"))
    assert "tool" not in review.prompt and "tool_calls" not in review.messages[0].content
