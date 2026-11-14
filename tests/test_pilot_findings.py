"""Regressions for the first free-model pilot (T.6d)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import FakeClock
from desman import providers, runs
from desman.agents import MAX_TOOL_ROUNDS, _ToolCalls, llm_executor, llm_reviewer
from desman.core.context import build_context
from desman.core.coordinator import Coordinator, Crew, Failure, Review, Success, Verdict, Work
from desman.core.effects import Effects, earlier_lookups, evidence
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind, Strategy, strategy_for
from desman.core.sandbox import Sandbox
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store
from desman.core.tools import LIST_FILES, ListFilesArgs, Toolbox, ToolCall
from desman.core.workspace import Workspace
from desman.llm import router
from desman.llm.base import Message, Request
from desman.llm.cache import Cached
from desman.llm.fake import FakeLLM, RateLimit


def claim(store: Store) -> Subtask:
    store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="c")])
    )
    subtask = store.claim("worker")
    assert subtask is not None
    return subtask


def work(store: Store, tmp_path: Path) -> Work:
    subtask = claim(store)

    def heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
        pass

    return Work(
        subtask=subtask,
        context=build_context(subtask, [], None),
        heartbeat=heartbeat,
        tools=Toolbox(Workspace(tmp_path, subtask.task_id)),
    )


def journal_result(
    store: Store, subtask: Subtask, execution: int, ordinal: int, tool: str, text: str
) -> None:
    assert subtask.lease_token is not None
    store.start_effect(subtask.id, subtask.lease_token, execution, ordinal, tool, "key")
    store.finish_effect(
        subtask.id,
        subtask.lease_token,
        execution,
        ordinal,
        {
            "ok": True,
            "output": {"untrusted": True, "text": text, "url": "https://example.org"},
            "error": None,
        },
    )


def reject(store: Store, subtask: Subtask) -> Subtask:
    assert subtask.lease_token is not None
    store.submit(subtask.id, subtask.lease_token, {"done": True})
    store.record_failure(
        subtask.id, SubtaskStatus.REVIEW, Actor.REVIEWER, FailureKind.QUALITY, reason="retry"
    )
    again = store.claim("worker")
    assert again is not None
    return again


def test_journal_completion_time_is_given_to_executor_replay_and_reviewer(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    subtask = claim(store)
    assert subtask.lease_token is not None
    journal = Effects(store, subtask.id, subtask.lease_token)
    box = Toolbox(Workspace(tmp_path, subtask.task_id), journal=journal)
    result = box.call(ToolCall(tool="list_files", args={}))
    assert result.at == clock.now.isoformat(timespec="seconds").replace("+00:00", "Z")
    replay = Effects(store, subtask.id, subtask.lease_token).begin(LIST_FILES, ListFilesArgs())
    assert replay is not None and replay.result.at == result.at
    store.submit(subtask.id, subtask.lease_token, {"done": True})
    shown = evidence(store, subtask)["tool_results"]
    assert shown[0]["at"] == result.at
    llm = FakeLLM('{"verdict":"accept"}')
    llm_reviewer(llm, lambda _: evidence(store, subtask))(subtask, {"retrieved": result.at})
    assert "matching its at" in llm.calls[0].messages[0].content


def test_evidence_prioritizes_latest_and_only_reuses_earlier_web(store: Store) -> None:
    subtask = claim(store)
    journal_result(store, subtask, 0, 0, "web_search", "old " * 1000)
    journal_result(store, subtask, 0, 1, "run_command", "stale command")
    subtask = reject(store, subtask)
    journal_result(store, subtask, 1, 0, "web_fetch", "newer " * 1000)
    journal_result(store, subtask, 1, 1, "read_file", "stale file")
    subtask = reject(store, subtask)
    journal_result(store, subtask, 2, 0, "write_file", "latest file")
    assert subtask.lease_token is not None
    store.submit(subtask.id, subtask.lease_token, {"done": True})
    shown = evidence(store, subtask, budget_tokens=100)["tool_results"]
    assert [item["tool"] for item in shown] == ["write_file", "web_fetch", "web_search"]
    assert shown[0]["truncated"] is False
    assert all(item["earlier_attempt"] and item["untrusted"] for item in shown[1:])
    assert all(item["truncated"] for item in shown[1:])
    assert (
        sum(
            len(json.dumps(item["result"])) if not item["truncated"] else len(item["result"])
            for item in shown
        )
        <= 400
    )
    assert earlier_lookups(store, subtask, 2, 0) == []


def test_retry_receives_earlier_lookups_within_remaining_budget(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    seen: list[Work] = []
    llm = FakeLLM('{"answer":"from source"}')
    execute = llm_executor(llm)

    def executor(item: Work) -> Success | Failure:
        seen.append(item)
        if len(seen) == 1:
            journal_result(store, item.subtask, 0, 0, "web_search", "facts " * 1000)
            return Failure(kind=FailureKind.QUALITY, reason="retry")
        return execute(item)

    subtask = claim(store)
    # Give the coordinator the initial claim back without spending an attempt.
    store.recover_expired(abandoned=lambda _: True)
    Coordinator(
        store,
        executor,
        lambda *_: Review(verdict=Verdict.ACCEPT),
        context_budget_tokens=100,
        sleep=clock.sleep,
    ).run(subtask.task_id)
    assert len(seen) == 2 and not seen[0].context.earlier_lookups
    lookups = seen[1].context.earlier_lookups
    assert lookups[0]["untrusted"] and len(lookups[0]["result"]) <= 400
    assert '"earlier_lookups"' in llm.calls[0].prompt
    assert "only for what they lack" in llm.calls[0].messages[0].content
    combined = seen[1].model_copy(
        update={"context": seen[1].context.model_copy(update={"strategy": Strategy.COMBINE})}
    )
    other = FakeLLM('{"answer":"combined"}')
    llm_executor(other)(combined)
    assert '"earlier_lookups"' not in other.calls[0].prompt


@pytest.mark.parametrize("arguments", [{"path": "."}, '{"path":"."}'])
def test_openai_tool_aliases(arguments: Any) -> None:
    calls = _ToolCalls.model_validate(
        {"tool_calls": [{"name": "list_files", "arguments": arguments}]}
    )
    assert calls.tool_calls[0] == ToolCall(tool="list_files", args={"path": "."})


@pytest.mark.parametrize(
    "item",
    [
        {"name": "x", "tool": "x", "args": {}},
        {"tool": "x", "arguments": "[]"},
        {"tool": "x", "arguments": "bad"},
        {"name": "x", "arguments": {}, "extra": True},
        {"tool": "x", "args": {}, "arguments": {}},
    ],
)
def test_tool_aliases_keep_validation_strict(item: Any) -> None:
    with pytest.raises(ValidationError):
        _ToolCalls.model_validate({"tool_calls": [item]})


def test_error_with_unused_tools_explains_call_syntax(store: Store, tmp_path: Path) -> None:
    llm = FakeLLM(
        '{"type":"error","reason":"executor reported an error: executor reported an error: no tools"}'
    )
    result = llm_executor(llm)(work(store, tmp_path))
    assert isinstance(result, Failure)
    assert "list_files" in result.reason and '{"tool_calls": [...]}' in result.reason
    assert result.reason.count("executor reported an error:") == 1


def test_last_tool_round_requires_final_result(store: Store, tmp_path: Path) -> None:
    llm = FakeLLM(
        *(['{"tool_calls":[{"name":"list_files","arguments":"{}"}]}'] * MAX_TOOL_ROUNDS),
        '{"done":true}',
    )
    result = llm_executor(llm)(work(store, tmp_path))
    assert isinstance(result, Success)
    assert "last allowed tool round" in llm.calls[-1].prompt
    assert "last allowed tool round" not in llm.calls[-2].prompt


def test_plain_prose_is_reviewed_without_repair(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    llm = FakeLLM().on("Execute", "A complete story.").on("Review", '{"verdict":"accept"}')
    subtask = claim(store)
    store.recover_expired(abandoned=lambda _: True)
    Coordinator(store, llm_executor(llm), llm_reviewer(llm), sleep=clock.sleep).run(subtask.task_id)
    assert len(llm.calls) == 2
    assert json.loads(llm.calls[-1].prompt.split(":\n", 1)[1])["result"] == {
        "text": "A complete story."
    }


@pytest.mark.parametrize("text", ["{broken", "```bad", 'bad "tool_calls"', " "])
def test_attempted_json_still_uses_repair(text: str, store: Store, tmp_path: Path) -> None:
    llm = FakeLLM(text, '{"fixed":true}')
    assert llm_executor(llm)(work(store, tmp_path)) == Success(content={"fixed": True})
    assert len(llm.calls) == 2 and "(repair)" in llm.calls[-1].prompt


def test_all_other_models_must_be_unfit_to_lose_strategy(store: Store) -> None:
    subtask = claim(store)

    def executor(_: Work) -> Success:
        return Success(content={})

    def reviewer(*_: Any) -> Review:
        return Review(verdict=Verdict.ACCEPT)

    crew = Crew(
        executor,
        reviewer,
        None,
        providers=("cohere/command-r7b", ("groq/gpt-oss-20b", "cohere/command-a")),
    )
    coordinator = Coordinator(store, executor, reviewer)
    avoid = coordinator._avoid(crew, subtask, frozenset({"groq/gpt-oss-20b"}))
    assert not avoid["avoid_other"]
    assert strategy_for(4, **avoid) is Strategy.OTHER_MODEL
    avoid = coordinator._avoid(crew, subtask, frozenset({"groq", "cohere/command-a"}))
    assert avoid["avoid_other"]


def test_sandbox_binds_only_pytest_runtime_imports(tmp_path: Path) -> None:
    argv = Sandbox(bwrap="bwrap", prlimit="prlimit")._argv(["true"], tmp_path)
    index = argv.index("PYTHONPATH")
    assert argv[index - 1 : index + 2] == ["--setenv", "PYTHONPATH", "/opt/desman/python"]
    targets = []
    for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments", "py"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            continue
        package = spec.submodule_search_locations is not None
        source = Path(spec.origin).parent if package else Path(spec.origin)
        target = "/opt/desman/python/" + name + ("" if package else ".py")
        pos = argv.index(target)
        assert argv[pos - 2 : pos] == ["--ro-bind", str(source)]
        targets.append(target)
    assert len([a for a in argv if a.startswith("/opt/desman/python/")]) == len(targets)


def test_other_lineup_skips_unfit_first_and_reaches_same_provider(store: Store) -> None:
    def choice(provider: str, model: str) -> router.Choice:
        p = providers.find(provider)
        return router.Choice(p, model)

    main = choice("cohere", "command-r7b-12-2024")
    groq = choice("groq", "openai/gpt-oss-20b")
    google = choice("google", "gemini-2.5-flash")
    stronger = choice("cohere", "command-a-03-2025")
    alternatives = router.other_lineup([main, groq, google, stronger])
    assert alternatives[0] == groq
    bad = FakeLLM(name="unfit")
    down = FakeLLM(RateLimit(retry_after=600), name="down")
    good = FakeLLM("stronger result", name="good")
    by_label = {groq.label: bad, google.label: down, stronger.label: good}
    chain = router.Fallback([(by_label[c.label], c) for c in alternatives])
    models = runs.Models(FakeLLM(name="main"), chain)
    subtask = claim(store)
    crew = Crew(
        lambda _: Success(content={}),
        lambda *_: Review(verdict=Verdict.ACCEPT),
        None,
        providers=models.labels,
    )
    avoid = Coordinator(store, crew.executor, crew.reviewer)._avoid(
        crew, subtask, frozenset({groq.label})
    )
    assert not avoid["avoid_other"]
    request = Request(
        messages=[Message(role="user", content="execute")], avoid=frozenset({groq.label})
    )
    assert chain.complete(request).text == "stronger result"
    assert not bad.calls and len(down.calls) == 1 and len(good.calls) == 1


def test_plain_result_is_cached_as_success(store: Store, tmp_path: Path) -> None:
    fake = FakeLLM("A story.")
    execute = llm_executor(Cached(fake, store.conn, "pilot"))
    item = work(store, tmp_path)
    assert execute(item) == Success(content={"text": "A story."})
    assert execute(item) == Success(content={"text": "A story."})
    assert len(fake.calls) == 1


def test_dependencies_take_context_budget_before_earlier_lookups(
    store: Store, clock: FakeClock
) -> None:
    seen: list[Work] = []

    def execute(item: Work) -> Success | Failure:
        if item.subtask.seq == 0:
            return Success(content={"text": "dependency " * 1000})
        seen.append(item)
        if item.context.attempt == 1:
            journal_result(store, item.subtask, 0, 0, "web_fetch", "earlier lookup")
            return Failure(kind=FailureKind.QUALITY, reason="retry")
        return Success(content={"done": True})

    task = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="first", acceptance_criteria="c"),
                SubtaskSpec(description="second", acceptance_criteria="c", depends_on=[0]),
            ],
        )
    )
    Coordinator(
        store,
        execute,
        lambda *_: Review(verdict=Verdict.ACCEPT),
        context_budget_tokens=100,
        sleep=clock.sleep,
    ).run(task)
    assert len(seen) == 2
    assert seen[-1].context.dependencies[0].truncated
    assert not seen[-1].context.earlier_lookups


def test_latest_evidence_exhausts_budget_before_earlier_web(store: Store) -> None:
    subtask = claim(store)
    journal_result(store, subtask, 0, 0, "web_search", "earlier")
    subtask = reject(store, subtask)
    journal_result(store, subtask, 1, 0, "read_file", "latest " * 1000)
    assert subtask.lease_token is not None
    store.submit(subtask.id, subtask.lease_token, {"done": True})
    shown = evidence(store, subtask, budget_tokens=10)["tool_results"]
    assert len(shown) == 1 and shown[0]["tool"] == "read_file"
    assert shown[0]["truncated"]


def test_sandbox_silently_skips_missing_runtime_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    argv = Sandbox(bwrap="bwrap", prlimit="prlimit")._argv(["true"], tmp_path)
    assert "/opt/desman/python" in argv
    assert not any(arg.startswith("/opt/desman/python/") for arg in argv)
