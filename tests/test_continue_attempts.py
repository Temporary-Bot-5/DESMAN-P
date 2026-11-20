"""Bounded executor tool history, continuation and routing across tool rounds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from conftest import FakeClock
from desman import agents, providers, runs
from desman.agents import MAX_TOOL_ROUNDS, TOOL_RESULT_BUDGET, TOOL_RESULTS, llm_executor
from desman.core import db
from desman.core.coordinator import MODEL_CALL_MARGIN, Coordinator, Review, Success, Verdict, Work
from desman.core.effects import archive_result, evidence, quote_sources
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.policy import Permission, executor_access
from desman.core.states import TaskStatus
from desman.core.store import PreemptedError, Store
from desman.core.tools import (
    MAX_READ_CHARS,
    ListFilesArgs,
    Registry,
    Tool,
    Toolbox,
    builtin_registry,
)
from desman.core.web import UNTRUSTED
from desman.core.workspace import Workspace
from desman.llm import catalog, router
from desman.llm.base import Message, ProviderRejected, Request
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import OpenAICompatible
from desman.llm.tool_calls import ToolCall
from desman.task_files import for_retelling, listing


class Killed(BaseException):
    pass


def reading(*offsets: int) -> str:
    return json.dumps(
        {
            "tool_calls": [
                {"tool": "read_file", "args": {"path": "input/book.txt", "offset": offset}}
                for offset in offsets
            ]
        }
    )


def task_with_book(store: Store, root: Path) -> tuple[str, Workspace]:
    task = store.create_task(
        TaskSpec(
            goal="Read", subtasks=[SubtaskSpec(description="Read", acceptance_criteria="Done")]
        )
    )
    workspace = Workspace(root, task)
    workspace.write("input/book.txt", "".join(str(i) * MAX_READ_CHARS for i in range(9)))
    return task, workspace


def run_with(
    model: FakeLLM | router.Fallback, store: Store, root: Path, clock: FakeClock
) -> Coordinator:
    return Coordinator(
        store,
        llm_executor(model, clock=clock),
        lambda *_: Review(verdict=Verdict.ACCEPT),
        workspaces=root,
        sleep=clock.sleep,
        abandoned=lambda _: True,
    )


def tool_messages(request: Request) -> list[str]:
    return [m.content for m in request.messages if m.content.startswith(f"{TOOL_RESULTS}:")]


def body(message: str) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.JSONDecoder().raw_decode(message.split(":\n", 1)[1])[0])


@pytest.mark.parametrize("crash", [False, True])
def test_continuation_after_exhausted_rounds_or_crash(
    store: Store, tmp_path: Path, clock: FakeClock, crash: bool
) -> None:
    task, workspace = task_with_book(store, tmp_path)
    offsets = [i * MAX_READ_CHARS for i in range(MAX_TOOL_ROUNDS)]
    reads = [reading(offset) for offset in offsets]
    if crash:

        def killed(_request: Request) -> str:
            raise Killed

        model = FakeLLM().on("Execute", *reads, killed, '{"text":"Finished"}')
    else:
        model = FakeLLM().on("Execute", *reads, reading(offsets[-1]), '{"text":"Finished"}')

    coordinator = run_with(model, store, tmp_path, clock)
    if crash:
        with pytest.raises(Killed):
            coordinator.run(task)
        assert not workspace.exists(f".tool-results/0-e0-r{MAX_TOOL_ROUNDS}-1.json")
    assert coordinator.run(task).status is TaskStatus.DONE

    last, retry = model.calls[-2:]
    results = tool_messages(last)
    assert len(results) == MAX_TOOL_ROUNDS
    assert sum(map(len, results)) <= TOOL_RESULT_BUDGET
    [first] = body(results[0])
    assert first["tool"] == "read_file" and first["untrusted"] is True
    assert "input/book.txt" in first["args"]
    assert first["result"] == "saved in .tool-results/0-e0-r1-1.json; read_file to see it again"
    assert json.loads(workspace.read_text(".tool-results/0-e0-r1-1.json"))["output"]["content"] == (
        "0" * MAX_READ_CHARS
    )
    assert body(results[-1])[0]["output"]["content"] == str(MAX_TOOL_ROUNDS - 1) * MAX_READ_CHARS
    assert [file.path for file in listing(tmp_path, task)] == ["input/book.txt"]
    assert [file["path"] for file in for_retelling(tmp_path, task)] == ["input/book.txt"]

    context = json.loads(retry.prompt.split(":\n", 1)[1])
    earlier = context["Already done by earlier attempts"]
    assert len(earlier) == MAX_TOOL_ROUNDS
    assert len(json.dumps(earlier, ensure_ascii=False)) <= 3000
    assert earlier[0]["outcome"] == (
        f"ok; read characters [{offsets[-1]}, {offsets[-1] + MAX_READ_CHARS}); "
        f"continue at offset {offsets[-1] + MAX_READ_CHARS}"
    )
    assert earlier[-1]["file"] == ".tool-results/0-e0-r1-1.json"
    assert all(workspace.exists(entry["file"]) and entry["untrusted"] for entry in earlier)
    subtask = store.list_subtasks(task)[0]
    reviewed = evidence(store, subtask, workspace=workspace)
    prior = [entry for entry in reviewed["tool_results"] if entry.get("earlier_attempt") or crash]
    assert len(prior) == len(earlier)
    assert all(entry["tool"] == "read_file" for entry in prior)
    assert any(entry["truncated"] for entry in prior)
    assert "repeat a call only if its result is missing or stale" in context["earlier_results_note"]
    assert context["attempt"] == (1 if crash else 2)
    assert router.estimated_tokens(retry) < router.estimated_tokens(last)


def test_oversized_latest_round_keeps_previews_and_deterministic_archives(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    messages = []
    for _ in range(2):
        task, workspace = task_with_book(store, tmp_path)
        model = FakeLLM().on(
            "Execute", reading(*(i * MAX_READ_CHARS for i in range(7))), '{"text":"Done"}'
        )
        assert run_with(model, store, tmp_path, clock).run(task).status is TaskStatus.DONE
        [latest] = tool_messages(model.calls[-1])
        assert len(latest) <= TOOL_RESULT_BUDGET
        entries = body(latest)
        assert len(entries) == 7
        assert any(entry.get("preview") and "content" in entry["preview"] for entry in entries)
        assert any("output" in entry for entry in entries)
        for index, entry in enumerate(entries, 1):
            if "result" in entry:
                archive = json.loads(workspace.read_text(f".tool-results/0-e0-r1-{index}.json"))
                assert archive["output"]["content"] == str(index - 1) * MAX_READ_CHARS
        # The completion timestamp is journal data, and FakeClock is identical for both tasks.
        messages.append(latest)
    assert messages[0] == messages[1]


def test_latest_round_that_fits_alone_stays_whole(
    store: Store, tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    cap = 8700
    monkeypatch.setattr(agents, "TOOL_RESULT_BUDGET", cap)
    task, workspace = task_with_book(store, tmp_path)
    model = FakeLLM().on(
        "Execute", *(reading(i * MAX_READ_CHARS) for i in range(6)), '{"text":"Done"}'
    )
    assert run_with(model, store, tmp_path, clock).run(task).status is TaskStatus.DONE
    messages = tool_messages(model.calls[-1])
    assert sum(map(len, messages)) <= cap
    assert body(messages[-1])[0]["output"]["content"] == "5" * MAX_READ_CHARS
    assert len(messages) < 6
    assert workspace.exists(".tool-results/0-e0-r1-1.json")


def test_a_tool_that_saved_its_payload_needs_no_duplicate(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path, "a" * 32)
    workspace.write("page.txt", "page content")
    payload = {"tool": "web_fetch", "ok": True, "output": {"saved": "page.txt"}}
    assert archive_result(
        workspace, ".tool-results/0-e0-r1-1.json", "web_fetch", payload, "{}"
    ) == ("page.txt")
    assert workspace.files() == ["page.txt"]


@pytest.mark.parametrize("constraint", ["context", "request_tokens"])
def test_growing_tool_round_skips_a_model_that_was_fit_initially(
    store: Store, tmp_path: Path, clock: FakeClock, constraint: str
) -> None:
    task, _ = task_with_book(store, tmp_path)
    small, large = FakeLLM(reading(0)), FakeLLM('{"text":"Done"}')
    provider = providers.find("groq")
    spec = (
        catalog.Model("test-small", context=3500)
        if constraint == "context"
        else catalog.Model("test-small", request_tokens=3500)
    )
    choice = router.Choice(provider, spec.id, spec)
    fallback = router.Fallback([(small, choice), (large, None)])
    assert run_with(fallback, store, tmp_path, clock).run(task).status is TaskStatus.DONE
    assert len(small.calls) == len(large.calls) == 1
    assert router.estimated_tokens(small.calls[0]) <= 3500 < router.estimated_tokens(large.calls[0])
    [skipped] = fallback.last_call.skipped
    assert (skipped.model, skipped.reason, skipped.context) == (choice.label, "context", 3500)
    assert skipped.needed == router.estimated_tokens(large.calls[0])

    only_small = router.Fallback([(small, choice)])
    with pytest.raises(ProviderRejected, match="largest available request ceiling is 3500"):
        only_small.complete(large.calls[0])
    assert len(small.calls) == 1
    assert only_small.last_call.skipped[0].needed == skipped.needed


def test_effect_continuation_migration_preserves_legacy_rows(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "legacy.db")
    try:
        assert db.migrate(conn, db.MIGRATIONS[:-1]) == db.SCHEMA_VERSION - 1
        task = "1" * 32
        subtask = "2" * 32
        conn.execute("INSERT INTO tasks (id, goal) VALUES (?, 'goal')", (task,))
        conn.execute(
            "INSERT INTO subtasks (id, task_id, seq, description, acceptance_criteria) "
            "VALUES (?, ?, 0, 'read', 'read')",
            (subtask, task),
        )
        conn.execute(
            "INSERT INTO effects (subtask_id, execution, ordinal, tool, key, lease_token, "
            "state, started_at, finished_at, result_json) "
            "VALUES (?, 0, 0, 'read_file', 'key', 'token', 'done', 't', 't', '{}')",
            (subtask,),
        )
        assert db.migrate(conn) == 1
        [row] = Store(conn).effects(subtask, 0)
        assert row["result_json"] == "{}" and row["continuation_json"] is None
        assert conn.execute("SELECT COUNT(*) FROM discarded_effects").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_second_resume_keeps_results_from_a_diverged_journal(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task, workspace = task_with_book(store, tmp_path)

    def killed(_request: Request) -> str:
        raise Killed

    model = FakeLLM().on(
        "Execute", reading(0), killed, reading(MAX_READ_CHARS), killed, '{"text":"Done"}'
    )
    coordinator = run_with(model, store, tmp_path, clock)
    for _ in range(2):
        with pytest.raises(Killed):
            coordinator.run(task)
    assert coordinator.run(task).status is TaskStatus.DONE
    context = json.loads(model.calls[-1].prompt.split(":\n", 1)[1])
    newest, oldest = context["Already done by earlier attempts"]
    assert [newest["file"], oldest["file"]] == [
        ".tool-results/0-e0-r9-1.json",
        ".tool-results/0-e0-r1-1.json",
    ]
    assert all(workspace.exists(entry["file"]) for entry in (newest, oldest))
    subtask = store.list_subtasks(task)[0]
    reviewed = evidence(store, subtask, workspace=workspace)["tool_results"]
    assert len(reviewed) == 2
    assert any(item.get("earlier_attempt") for item in reviewed)
    assert any("0" * 20 in json.dumps(item["result"]) for item in reviewed)
    assert any("1" * 20 in json.dumps(item["result"]) for item in reviewed)
    sources = quote_sources(store, subtask, knowledge=None, context_budget_tokens=4000)
    assert {output["content"] for output in cast(list[dict[str, Any]], sources)} == {
        "0" * MAX_READ_CHARS,
        "1" * MAX_READ_CHARS,
    }


@pytest.mark.parametrize("long", [False, True])
@pytest.mark.parametrize("review", [False, True])
def test_model_call_extends_only_the_lease_it_needs(
    store: Store, tmp_path: Path, clock: FakeClock, long: bool, review: bool
) -> None:
    task, _ = task_with_book(store, tmp_path)
    initial = runs.step_seconds(60)
    expected_timeout = 300 if long else 60
    seen = []

    def response(request: httpx.Request) -> httpx.Response:
        subtask = store.list_subtasks(task)[0]
        expiry = subtask.review_until if review else subtask.lease_until
        assert expiry is not None
        remaining = (db.parse_iso(expiry) - clock()).total_seconds()
        assert remaining == max(initial, expected_timeout + MODEL_CALL_MARGIN)
        assert request.extensions["timeout"]["read"] == expected_timeout
        seen.append(remaining)
        clock.advance(expected_timeout - 1)
        if review:
            assert store.reserve_review(subtask.id, "another reviewer") is None
        else:
            assert store.recover_expired() == []
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    request = Request(messages=[Message(role="user", content="x" * (100_000 if long else 1))])
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        model = OpenAICompatible(
            "test",
            "https://example.invalid",
            ["key"],
            "model",
            client=client,
            monotonic=lambda: clock().timestamp(),
        )

        def execute(_: Work) -> Success:
            return Success(content={"text": "done" if review else model.complete(request).text})

        def judge(*_: object) -> Review:
            if review:
                model.complete(request)
            return Review(verdict=Verdict.ACCEPT)

        coordinator = Coordinator(
            store,
            execute,
            judge,
            workspaces=tmp_path,
            sleep=clock.sleep,
            lease_seconds=initial,
            review_seconds=initial,
        )
        assert coordinator.run(task).status is TaskStatus.DONE
    assert len(seen) == 1


def test_worker_that_dies_before_a_call_keeps_the_base_lease(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task, _ = task_with_book(store, tmp_path)
    initial = runs.step_seconds(60)

    def killed(_: Work) -> Success:
        raise Killed

    coordinator = Coordinator(
        store,
        killed,
        lambda *_: Review(verdict=Verdict.ACCEPT),
        workspaces=tmp_path,
        sleep=clock.sleep,
        lease_seconds=initial,
        abandoned=lambda _: False,
    )
    with pytest.raises(Killed):
        coordinator.run(task)
    clock.advance(initial - 1)
    assert store.recover_expired() == []
    clock.advance(2)
    assert len(store.recover_expired()) == 1


def test_large_write_arguments_are_folded_even_in_the_latest_round(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task, workspace = task_with_book(store, tmp_path)
    text = "private draft " * 65_000
    write = json.dumps(
        {"tool_calls": [{"tool": "write_file", "args": {"path": "draft.txt", "content": text}}]}
    )
    model = FakeLLM().on("Execute", write, reading(0), '{"text":"Done"}')
    assert run_with(model, store, tmp_path, clock).run(task).status is TaskStatus.DONE
    assert workspace.read_text("draft.txt") == text
    for request in model.calls[1:]:
        history = request.messages[2:]
        assert sum(len(message.content) for message in history) <= TOOL_RESULT_BUDGET
        saved = json.loads(history[0].content)["tool_calls"][0]
        assert saved["file"] == "draft.txt"
        assert len(saved["args"]) <= 120


@pytest.mark.parametrize("path", ["./.tool-results/result.json", "web/page.txt", "input/web-x"])
@pytest.mark.parametrize("offset", [0, 1])
def test_reading_saved_tool_data_marks_it_untrusted(tmp_path: Path, path: str, offset: int) -> None:
    workspace = Workspace(tmp_path, "a" * 32)
    workspace.write(path, "source text")
    result = Toolbox(workspace, builtin_registry()).call(
        ToolCall(tool="read_file", args={"path": path, "offset": offset})
    )
    assert result.ok
    assert result.output["untrusted"] == UNTRUSTED


def test_preempted_formatting_keeps_the_completed_result(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task, _ = task_with_book(store, tmp_path)
    subtask = store.list_subtasks(task)[0]

    def preempted(_: Request) -> str:
        raise PreemptedError(subtask.id, "critical")

    model = FakeLLM().on("Execute", '{"value": 42}').on("Format final answer", preempted)
    assert run_with(model, store, tmp_path, clock).run(task).status is TaskStatus.DONE
    assert store.latest_artifact(subtask.id) == {"value": 42}
    assert len(model.calls) == 2


def test_runs_code_retries_run_again_in_the_current_execution(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task = store.create_task(
        TaskSpec(
            goal="Run the program",
            subtasks=[SubtaskSpec(description="Run", acceptance_criteria="Ran", runs_code=True)],
        )
    )
    calls = []
    registry = Registry(tool for tool in builtin_registry() if tool.name != "run_command")

    def run(_: ListFilesArgs, ws: Workspace) -> dict[str, int]:
        calls.append(ws)
        return {"exit_code": 0}

    registry.register(
        Tool(
            name="run_command",
            description="Run the program",
            args=ListFilesArgs,
            permission=Permission.SHELL,
            run=run,
        )
    )
    run_call = '{"tool_calls":[{"tool":"run_command","args":{}}]}'
    model = FakeLLM().on(
        "Execute",
        run_call,
        reading(0),
        '{"type":"error","reason":"Try again"}',
        run_call,
        '{"text":"Done"}',
    )
    Workspace(tmp_path, task).write("input/book.txt", "data")
    coordinator = Coordinator(
        store,
        llm_executor(model),
        lambda *_: Review(verdict=Verdict.ACCEPT),
        workspaces=tmp_path,
        tools=registry,
        access=executor_access([Permission.SHELL]),
        sleep=clock.sleep,
    )
    assert coordinator.run(task).status is TaskStatus.DONE
    assert len(calls) == 2
    context = json.loads(model.calls[3].prompt.split(":\n", 1)[1])
    assert [entry["tool"] for entry in context["Already done by earlier attempts"]] == ["read_file"]
    assert "Run the code again" in context["earlier_results_note"]
