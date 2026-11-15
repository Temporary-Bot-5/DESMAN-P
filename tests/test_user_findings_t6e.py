"""Regressions from using run/resume with the default free models (T.6e)."""

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import actions, providers, runs
from desman.cli import cli
from desman.core import calls, db
from desman.core.coordinator import Coordinator
from desman.core.effects import Effects, evidence
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.sandbox import Sandbox
from desman.core.store import Store
from desman.core.tools import WRITE_FILE, ToolResult, WriteFileArgs
from desman.core.workspace import Workspace
from desman.llm import router
from desman.llm.base import LLM, GenerationFailed, Message, ProviderTimeout, Request
from desman.llm.fake import FakeLLM, RateLimit, Timeout
from desman.llm.openai_compat import CallRecord, OpenAICompatible
from desman.planner import LANGUAGE_RULES

REQUEST = Request(messages=[Message(role="user", content="Execute subtask:\n{}")])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bubblewrap is Linux only")
def test_sandbox_binds_only_the_selected_distribution_metadata(tmp_path: Path) -> None:
    argv = Sandbox(bwrap="bwrap", prlimit="/usr/bin/prlimit")._argv(["python3"], tmp_path)
    binds = {(argv[i + 1], argv[i + 2]) for i, word in enumerate(argv) if word == "--ro-bind"}
    for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments"):
        dist = importlib.metadata.distribution(name)
        file = next(f for f in dist.files or () if f.parts[0].endswith(".dist-info"))
        source = Path(str(dist.locate_file(file.parts[0])))
        assert (str(source), "/opt/desman/python/" + source.name) in binds
        assert (str(source.parent), "/opt/desman/python") not in binds


@pytest.mark.parametrize("ordered", [False, True])
def test_slow_failure_does_not_mask_untried_models_with_an_earlier_limit(
    clock: FakeClock, ordered: bool
) -> None:
    limited = FakeLLM(RateLimit(retry_after=60), name="limited")
    slow_llm = FakeLLM(Timeout(after=60), name="slow", sleep=clock.sleep)
    slow = router.Backoff(
        slow_llm,
        sleep=clock.sleep,
        monotonic=lambda: clock().timestamp(),
    )
    untried = FakeLLM("done", name="untried")
    models: list[LLM] = [limited, slow, untried]
    choices = [
        router.Choice(providers.find("groq"), model)
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "llama-3.3-70b-versatile")
    ]
    chain = router.Fallback(
        [(model, choices[i] if ordered else None) for i, model in enumerate(models)],
        order=(lambda cs: cs) if ordered else None,
    )
    with calls.for_(calls.Purpose(role=calls.Role.EXECUTOR, task_id="task")):
        with pytest.raises(ProviderTimeout):
            chain.complete(REQUEST)
        assert not untried.calls
        assert chain.complete(REQUEST).text == "done"
    assert len(limited.calls) == len(slow_llm.calls) == 1


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("command", ["run", "resume"])
@pytest.mark.parametrize("seconds, expected", [(18, 0), (600, 0), (900, 0), (901, 4)])
def test_cli_waits_only_for_short_provider_pauses(
    command: str,
    as_json: bool,
    seconds: int,
    expected: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
) -> None:
    original = runs.coordinator

    def coordinator(*args: Any, **kwargs: Any) -> Coordinator:
        co = original(*args, **kwargs)
        co.store.clock = clock
        co.sleep = clock.sleep
        return co

    monkeypatch.setattr(runs, "coordinator", coordinator)
    path = tmp_path / "task.db"
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {
                "routes": [
                    {
                        "match": "Execute",
                        "steps": [
                            {"type": "rate_limit", "retry_after": seconds},
                            '{"output":"done"}',
                        ],
                    },
                    {"match": "Review", "steps": ['{"verdict":"accept"}']},
                ]
            }
        ),
        encoding="utf-8",
    )
    args = [command, "goal", "--db", str(path), "--fake-llm", str(script)]
    if command == "resume":
        conn = db.open_db(path)
        task = Store(conn, clock=clock).create_task(
            TaskSpec(
                goal="goal", subtasks=[SubtaskSpec(description="work", acceptance_criteria="done")]
            )
        )
        conn.close()
        args[1] = task
    if as_json:
        args.append("--json")
    result = CliRunner().invoke(cli, args)
    if as_json:
        assert json.loads(result.stdout)["run"]["status"] == ("done" if expected == 0 else "active")
    assert result.exit_code == expected, result.output
    if expected == 0:
        assert clock.slept == seconds
        assert result.output.count(f"Waiting {seconds} s for a provider limit") == 1
        assert "Continue with:" not in result.output
    else:
        assert clock.slept == 0
        if not as_json:
            assert "Continue with:" in result.output
            assert "`desman run --wait` or `desman worker`" in result.output


@pytest.mark.parametrize(
    "native",
    [
        {"name": "write_file", "arguments": {"path": "x", "content": "hello"}},
        [{"name": "write_file", "arguments": '{"path":"x","content":"hello"}'}],
        [
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path":"x","content":"hello"}'},
            }
        ],
        {"tool_calls": [{"tool": "write_file", "args": {"path": "x", "content": "hello"}}]},
    ],
)
@pytest.mark.parametrize("status", [200, 400])
def test_native_calls_become_executor_calls(native: Any, status: int) -> None:
    payload = native if isinstance(native, list) else [native]
    if isinstance(native, dict) and "tool_calls" in native:
        payload = native["tool_calls"]
    body = (
        {"choices": [{"message": {"content": None, "tool_calls": payload}}]}
        if status == 200
        else {
            "error": {
                "code": "tool_use_failed",
                "failed_generation": json.dumps(native),
                "unrelated": "must not be kept",
            }
        }
    )
    records: list[CallRecord] = []
    llm = OpenAICompatible(
        "groq",
        "https://example.test/v1",
        ["key"],
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body))
        ),
        on_call=records.append,
    )
    reply = llm.complete(REQUEST)
    assert json.loads(reply.text) == {
        "tool_calls": [{"tool": "write_file", "args": {"path": "x", "content": "hello"}}]
    }
    assert len(records) == 1 and records[0].error is None
    assert "must not be kept" not in repr(records)


@pytest.mark.parametrize(
    "failed",
    [
        "not JSON",
        '{"answer":"not a tool"}',
        '{"name":"write_file","arguments":"not JSON"}',
        '{"name":"write_file","arguments":{},"extra":"private"}',
    ],
)
def test_unrecoverable_generation_stays_quality_without_body_leaks(failed: str) -> None:
    body = {
        "error": {
            "code": "tool_use_failed",
            "message": "generation failed",
            "failed_generation": failed,
        }
    }
    llm = OpenAICompatible(
        "groq",
        "https://example.test/v1",
        ["key"],
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(400, json=body))
        ),
    )
    with pytest.raises(GenerationFailed) as error:
        llm.complete(REQUEST)
    assert failed not in str(error.value)


def test_google_malformed_function_call_is_quality() -> None:
    body = {
        "choices": [
            {
                "message": {"content": None},
                "finish_reason": "function_call_filter: MALFORMED_FUNCTION_CALL",
            }
        ]
    }
    llm = OpenAICompatible(
        "google",
        "https://example.test/v1",
        ["key"],
        "m",
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    )
    with pytest.raises(GenerationFailed):
        router.Backoff(llm).complete(REQUEST)


@pytest.mark.parametrize(
    "value, text",
    [
        ({"type": "result", "output": "one\ntwo"}, "one\ntwo"),
        ({"letter": "Dear…\nRegards"}, "Dear…\nRegards"),
        ({"plan": "Week 1\nWeek 2"}, "Week 1\nWeek 2"),
        ({"a": "one\ntwo", "b": {"c": ["x", "y"]}}, "a: one\n  two\nb:\n  c:\n    - x\n    - y"),
        ([{"a": "x", "b": [1, 2]}], "- a: x\n  b:\n    - 1\n    - 2"),
    ],
)
def test_results_are_readable(value: Any, text: str) -> None:
    assert actions.result_text(value) == text


def test_reviewer_sees_current_written_files_with_fair_budget(store: Store, tmp_path: Path) -> None:
    store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="work", acceptance_criteria="c")])
    )
    sub = store.claim("worker")
    assert sub is not None and sub.lease_token is not None
    ws = Workspace(tmp_path, sub.task_id)
    effects = Effects(store, sub.id, sub.lease_token)
    for path, text in [
        ("small.txt", "small"),
        ("large.txt", "x" * 9000),
        ("binary", "initial"),
        ("gone", "initial"),
    ]:
        args = WriteFileArgs(path=path, content=text)
        effects.begin(WRITE_FILE, args)
        effects.finish(ToolResult(ok=True, output=WRITE_FILE.run(args, ws)))
    ws.write("small.txt", "current")
    ws.write("binary", b"\xff\x00")
    (ws.root / "gone").unlink()
    store.submit(sub.id, sub.lease_token, {"done": True})
    shown = evidence(store, sub, budget_tokens=200, workspace=ws)
    files = {file["path"]: file for file in shown["written_files"]}
    assert files["small.txt"] == {"path": "small.txt", "content": "current", "truncated": False}
    assert files["large.txt"]["truncated"] and "x" in files["large.txt"]["content"]
    assert "unavailable" in files["binary"]["content"]
    assert "unavailable" in files["gone"]["content"]
    assert len(json.dumps(shown)) < 2000


def test_language_rule_distinguishes_requested_text_from_prose() -> None:
    assert "prose addressed to the person" in LANGUAGE_RULES
    assert "END of its description" in LANGUAGE_RULES
    assert "translation, a letter in" in LANGUAGE_RULES
    assert "must stay in that requested language" in LANGUAGE_RULES
    assert "when splitting a subtask" in LANGUAGE_RULES


@pytest.mark.parametrize("step, times_out", [(20.0, True), (5.0, False)])
def test_whole_call_deadline_stops_dribbling_body(step: float, times_out: bool) -> None:
    from collections.abc import Iterator

    elapsed = [0.0]
    closed = []

    class Body(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            for chunk in [b'{"choices":', b'[{"message":', b'{"content":"ok"}}]}']:
                elapsed[0] += step
                yield chunk

        def close(self) -> None:
            closed.append(True)

    records: list[CallRecord] = []
    llm = OpenAICompatible(
        "openrouter",
        "https://example.test/v1",
        ["secret"],
        "m",
        timeout=60,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, stream=Body(), headers={"x-ratelimit-limit": "10"}
                )
            )
        ),
        monotonic=lambda: elapsed[0],
        on_call=records.append,
    )
    if times_out:
        with pytest.raises(ProviderTimeout):
            llm.complete(REQUEST)
        assert records[0].error_type == "ProviderTimeout"
        assert records[0].response_hash is None
    else:
        assert llm.complete(REQUEST).text == "ok"
        assert records[0].error_type is None
    assert closed == [True]
    assert len(records) == 1
    assert records[0].latency_ms == int(step * 3000)
    assert records[0].http_status == 200
    assert records[0].limits == {"x-ratelimit-limit": "10"}


def test_simple_goals_and_readable_results_prompt_contracts() -> None:
    from desman.agents import EXECUTOR_SYSTEM, REVIEWER_EVIDENCE, REVIEWER_SYSTEM
    from desman.planner import MAX_SUBTASKS, PLANNER_RULES, template_plan

    assert "ONE subtask" in PLANNER_RULES
    assert "a translation" in PLANNER_RULES
    assert "different tools/kinds" in PLANNER_RULES
    assert MAX_SUBTASKS == 8
    assert len(template_plan("Translate a phrase").subtasks) == 1
    assert "put that text in the result itself" in EXECUTOR_SYSTEM
    assert "put the text in the result" in REVIEWER_SYSTEM
    assert "README.md" in REVIEWER_SYSTEM
    assert "Never decide from memory" in REVIEWER_EVIDENCE
    assert "search the web first" in REVIEWER_EVIDENCE
    assert "Never escalate for that omission" in REVIEWER_EVIDENCE


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="bubblewrap is Linux only")
def test_sandbox_disables_generated_python_caches(tmp_path: Path) -> None:
    argv = Sandbox(bwrap="bwrap", prlimit="prlimit")._argv(["python3"], tmp_path)
    env = {argv[i + 1]: argv[i + 2] for i, word in enumerate(argv) if word == "--setenv"}
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTEST_ADDOPTS"] == "-p no:cacheprovider"


def test_task_listing_omits_nested_cache_directories(tmp_path: Path) -> None:
    from desman import task_files

    ws = Workspace(tmp_path, "task")
    for path in (
        ".pytest_cache/a",
        "nested/__pycache__/a.pyc",
        "nested/.pytest_cache/b",
        "answer.py",
        "__pycache__.txt",
    ):
        ws.write(path, "x")
    assert [f.path for f in task_files.listing(tmp_path, "task")] == [
        "__pycache__.txt",
        "answer.py",
    ]


@pytest.mark.parametrize("bullet", ["- ", "* ", "• ", "1. "])
def test_result_summary_preserves_existing_bullets(bullet: str) -> None:
    assert (
        actions.result_text({"type": "result", "summary": [bullet + "**Период**", "plain"]})
        == "summary:\n  " + bullet + "**Период**\n  - plain"
    )
    assert actions.result_text({"type": "error", "summary": ["plain"]}).startswith("type: error\n")
