"""Review regressions for T.6e round 5."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from desman import actions, providers, task_files
from desman.agents import llm_executor
from desman.chat import Chat
from desman.core.coordinator import Coordinator, Review, Verdict
from desman.core.effects import workspace_files
from desman.core.models import Kind, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.sessions import Role, Sessions
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import READ_FILE, Registry
from desman.core.workspace import Workspace
from desman.interlocutor import Interlocutor
from desman.llm import router
from desman.llm.base import Completion, Message, Request, RequestTooLarge
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import CallRecord, OpenAICompatible
from desman.planner import PlannedSubtask, needs_run

REQUEST = Request(messages=[Message(role="user", content="hello")])


def task() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="work", acceptance_criteria="c")])


def test_provider_413_fails_the_attempt_and_falls_back_without_repeating(store: Store) -> None:
    seen: list[Request] = []

    class Oversized:
        name = "groq"

        def complete(self, request: Request) -> Completion:
            seen.append(request)
            raise RequestTooLarge(self.name, 413, "too large")

    backed = router.Backoff(Oversized(), sleep=lambda _: pytest.fail("backoff on 413"))
    chain = router.Fallback([(backed, None)])
    with pytest.raises(RequestTooLarge) as caught:
        chain.complete(REQUEST)
    # L.2 live run: a too-small last model fails the attempt, never the subtask at once.
    assert caught.value.kind is FailureKind.QUALITY and caught.value.unfit_model
    assert len(seen) == 1
    assert chain.last_call.skipped[0].reason == "context"
    assert router.Fallback([(backed, None), (FakeLLM("ok"), None)]).complete(REQUEST).text == "ok"
    assert len(seen) == 2
    # With no model big enough, each attempt fails once (no backoff), then the person decides.
    ref = store.create_task(task())
    report = Coordinator(
        store, llm_executor(backed), lambda *_: Review(verdict=Verdict.ACCEPT)
    ).run(ref)
    assert report.status is TaskStatus.NEEDS_HUMAN
    attempts = len(seen) - 2
    assert attempts == task().retry.max_attempts


@pytest.mark.parametrize(
    "description",
    [
        "Run hello.py with python3 and report what it prints",
        "Execute the example",
        "python3 demo",
        "bash demo",
        "shell invocation",
        "pytest -q",
        "Запусти пример",
        "Выполни пример",
        "Исполни пример",
        "Check hello.py",
        "Check hello.sh",
        "Check hello.js",
    ],
)
@pytest.mark.parametrize("marked", [True, False])
def test_real_run_hints_preserve_only_explicit_run_marks(description: str, marked: bool) -> None:
    sub = PlannedSubtask(description=description, acceptance_criteria="c", runs_code=marked)
    assert needs_run(sub, Kind.TEXT) is marked


@pytest.mark.parametrize(
    "description", ["Find news", "Summarise the contest results", "Найди новости"]
)
def test_spurious_run_marks_on_lookup_are_removed(description: str) -> None:
    assert not needs_run(
        PlannedSubtask(description=description, acceptance_criteria="c", runs_code=True), Kind.TEXT
    )


def test_new_chat_does_not_swallow_waiting_items(store: Store) -> None:
    sessions = Sessions(store.conn)
    chat = Chat(sessions)
    chat.session_id = sessions.add(None, Role.USER, "hello").session_id
    ref = store.create_task(task())
    sub = store.claim("worker")
    assert sub is not None
    store.transition(
        sub.id,
        SubtaskStatus.RUNNING,
        SubtaskStatus.NEEDS_HUMAN,
        Actor.EXECUTOR,
        lease=sub.lease_token,
        action="test_wait",
    )
    chat._new()
    fresh = chat.waiting.new()
    assert len(fresh) == 1 and fresh[0].task_id == ref


def test_completed_body_is_accepted_after_deadline_on_iterator_exit() -> None:
    elapsed = [0.0]

    class Body(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield b'{"choices":[{"message":{"content":"ok"}}]}'
            elapsed[0] = 61

    llm = OpenAICompatible(
        "groq",
        "https://example.test/v1",
        ["key"],
        "m",
        timeout=60,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Body()))
        ),
        monotonic=lambda: elapsed[0],
    )
    assert llm.complete(REQUEST).text == "ok"


@pytest.mark.parametrize(
    "items", [["1. a", "2. b", "3. c"], ["1) a", "12) b"], ["- a", "* b", "• c"]]
)
def test_existing_list_markers_are_preserved(items: list[str]) -> None:
    assert actions.result_text(items) == "\n".join(items)


@pytest.mark.parametrize("latest", ["ok", "/status", "Покажи результат"])
def test_retell_prompt_defines_language_fallback_order(latest: str) -> None:
    fake = FakeLLM('{"text":"Готово"}')
    model = Interlocutor(fake, "en")
    model.retell({"goal": "Найди новости", "latest_message": latest, "status": "done"})
    prompt = fake.calls[0].messages[0].content
    assert "ok or /status" in prompt
    assert (
        prompt.index("latest message")
        < prompt.index("GOAL's language")
        < prompt.index("English (en)")
    )
    assert (
        json.loads(fake.calls[0].messages[1].content.split(":\n", 1)[1])["latest_message"] == latest
    )


def test_ceiling_query_is_bounded_and_once_per_member(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.core.calls import recorder

    choice = router.Choice(providers.find("groq"), "qwen/qwen3.8-27b")
    record = recorder(store.conn)
    for limits in [{"x-ratelimit-limit-tokens": "7500"}, {}, {"x-ratelimit-limit-requests": "10"}]:
        record(
            CallRecord(
                provider="groq",
                model=choice.model,
                key_id="k",
                prompt_hash="p",
                response_hash=None,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                http_status=200,
                error_kind=None,
                error=None,
                limits=limits,
            )
        )
    statements: list[str] = []
    store.conn.set_trace_callback(statements.append)
    assert router.request_ceiling(choice, store.conn) == 7500
    queries = [s for s in statements if "SELECT" in s and "model_calls" in s]
    assert len(queries) == 1 and "LIMIT 1" in queries[0] and "json_extract" in queries[0]
    queried: list[str] = []
    original = router.request_ceiling

    def ceiling(c: router.Choice, conn: Any = None) -> int | None:
        queried.append(c.label)
        return original(c, conn)

    monkeypatch.setattr(router, "request_ceiling", ceiling)
    chain = router.Fallback([(FakeLLM("ok"), choice)], conn=store.conn)
    with pytest.raises(Exception, match="largest available request ceiling"):
        chain.complete(Request(messages=[Message(role="user", content="x" * 40000)]))
    assert queried == [choice.label]
    store.conn.set_trace_callback(None)


def test_tool_contract_is_reexported_without_loading_runtime() -> None:
    import subprocess
    import sys

    from desman.core.tools import ToolCall as old
    from desman.llm.tool_calls import ToolCall

    assert old is ToolCall
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import desman.llm.openai_compat; assert 'desman.core.tools' not in sys.modules; assert 'desman.core.sandbox' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr


def test_inventory_uses_sizes_filters_caches_and_stops_at_budget(
    store: Store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = store.create_task(task())
    sub = store.claim("worker")
    assert sub is not None
    ws = Workspace(tmp_path, ref)
    for path in ["a", "b", "c", "__pycache__/x.pyc", ".pytest_cache/v/x"]:
        ws.write(path, "data")
    monkeypatch.setattr(Workspace, "read", lambda *_: pytest.fail("inventory read file contents"))
    assert [f.path for f in task_files.listing(tmp_path, ref)] == ["a", "b", "c"]
    assert [f["path"] for f in workspace_files(store, sub, ws, 1000)] == ["a", "b", "c"]
    sizes: list[str] = []
    original = ws.size

    def size(path: str) -> int:
        sizes.append(path)
        return original(path)

    monkeypatch.setattr(ws, "size", size)
    assert workspace_files(store, sub, ws, 1) == []
    assert sizes == ["a"]
    sizes.clear()
    assert workspace_files(store, sub, ws, 0) == []
    assert sizes == []


@pytest.mark.parametrize("file_tools", [False, True])
def test_executor_only_sees_inventory_with_file_tools(
    store: Store, tmp_path: Path, file_tools: bool
) -> None:
    from desman.core.policy import executor_access

    ref = store.create_task(task())
    Workspace(tmp_path, ref).write("hello.py", "print(1)")
    fake = FakeLLM('{"text":"done"}', repeat_last=True)
    reviews = iter(
        [Review(verdict=Verdict.RETRY, feedback="retry"), Review(verdict=Verdict.ACCEPT)]
    )
    report = Coordinator(
        store,
        llm_executor(fake),
        lambda *_: next(reviews),
        tools=Registry([READ_FILE] if file_tools else []),
        access=executor_access(),
        workspaces=tmp_path,
    ).run(ref)
    assert report.status is TaskStatus.DONE
    payload = json.loads(fake.calls[-1].prompt.split(":\n", 1)[1])
    assert ("workspace_files" in payload) is file_tools


@pytest.mark.parametrize("secure", [False, True])
def test_stat_size_preserves_workspace_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, secure: bool
) -> None:
    from desman.core import workspace as module
    from desman.core.workspace import PathEscapeError

    if secure and not module.SECURE:
        pytest.skip("descriptor-relative stat unavailable")
    monkeypatch.setattr(module, "SECURE", secure)
    ws = Workspace(tmp_path, "task")
    ws.write("nested/work", "hello")
    assert ws.size("nested/work") == 5
    with pytest.raises(PathEscapeError):
        ws.size("../outside")
    with pytest.raises(PathEscapeError):
        ws.size("nested")
    try:
        (ws.root / "link").symlink_to(ws.root / "nested/work")
        (ws.root / "hard").hardlink_to(ws.root / "nested/work")
    except OSError:
        pytest.skip("links unavailable")
    for path in ["link", "hard", "nested/work"]:
        with pytest.raises(PathEscapeError):
            ws.size(path)
