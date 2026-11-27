"""Independent review tests decide whether submitted code is accepted."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest

from conftest import FakeClock
from desman.actions import print_report
from desman.agents import llm_executor, llm_reviewer
from desman.core import db
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work, review_test_file
from desman.core.coordinator import TestRun as ReviewTestResult
from desman.core.effects import evidence, has_written_files, last_review_tests
from desman.core.models import Kind, Subtask, SubtaskSpec, TaskSpec, Tools
from desman.core.policy import Permission, executor_access
from desman.core.report import task_report
from desman.core.retry import RetryPolicy
from desman.core.sandbox import Outcome, Sandbox
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import builtin_registry
from desman.core.workspace import Workspace
from desman.llm.fake import FakeLLM
from desman.runs import Models, _review_test_context, _run_review_tests, coordinator
from helpers import make_config
from test_sandbox import confined

MODULE = "def test_answer():\n    assert True\n"
ACCEPT = json.dumps({"verdict": "accept", "tests": MODULE})


def submitted(store: Store, *, runs_code: bool = True, tools: Tools = Tools.GRANTED) -> str:
    task_id = store.create_task(
        TaskSpec(
            goal="For input 2, the answer is 4.",
            tools=tools,
            retry=RetryPolicy(max_attempts=1, max_transient=2),
            subtasks=[
                SubtaskSpec(
                    description="Implement square",
                    acceptance_criteria="Square returns an integer",
                    runs_code=runs_code,
                )
            ],
        )
    )
    sub = store.claim("executor", task_id=task_id)
    assert sub is not None and sub.lease_token is not None
    store.submit(sub.id, sub.lease_token, {"summary": "Ready to review"})
    return task_id


def unread(work: Work) -> Success:
    pytest.fail("the submitted artifact should be reviewed without re-execution")


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (ReviewTestResult(ran=True, exit_code=0, passed=3), None),
        (
            ReviewTestResult(
                ran=True, exit_code=1, failed=1, failing=["test_answer"], tail="assert 2 == 4"
            ),
            "test_answer\nassert 2 == 4",
        ),
        (ReviewTestResult(ran=True, timed_out=True), "timed out"),
        (
            ReviewTestResult(ran=True, exit_code=2, errors=1, failing=[".review/test_review_1.py"]),
            ".review/test_review_1.py",
        ),
        (ReviewTestResult(ran=True, exit_code=5), "pytest exited 5"),
        (ReviewTestResult(ran=True, exit_code=0, failed=1, failing=["test_answer"]), "test_answer"),
        (ReviewTestResult(ran=True, exit_code=0, passed=1, errors=1), "pytest exited 0"),
        (
            ReviewTestResult(ran=True, exit_code=0, passed=1, stopped="watch stopped"),
            "watch stopped",
        ),
        (
            ReviewTestResult(ran=True, exit_code=0, passed=1, workspace_full="quota exceeded"),
            "quota exceeded",
        ),
        (
            ReviewTestResult(ran=True, exit_code=0, tail="1 skipped"),
            "verified nothing: 0 passed\n1 skipped",
        ),
    ],
)
def test_review_result(
    store: Store, clock: FakeClock, result: ReviewTestResult, reason: str | None
) -> None:
    task_id = submitted(store)

    def runner(sub: Subtask, workspace: Workspace, module: str, attempt: int) -> ReviewTestResult:
        assert module == MODULE and attempt == 1
        current = store.get_subtask(sub.id)
        assert current.review_until is not None
        assert db.parse_iso(current.review_until) >= clock() + timedelta(seconds=90)
        return result

    llm = FakeLLM().on("Review", ACCEPT)
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        review_seconds=1,
        access=executor_access([Permission.SHELL]),
        tools=builtin_registry(),
        review_tests=runner,
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is (
        TaskStatus.DONE if reason is None else TaskStatus.NEEDS_HUMAN
    )
    rows = store.audit_log(task_id)
    [note] = [row for row in rows if row["action"] == "review_tests"]
    assert note["actor"] == "reviewer" and note["to_status"] is None
    [sub] = store.list_subtasks(task_id)
    assert json.loads(note["details_json"]) == {
        "file": f".review/{sub.id}/test_review_1.py",
        **result.model_dump(),
    }
    if reason is not None:
        [verdict] = [row for row in rows if row["action"] == "verdict"]
        message = reason if reason.startswith("verified nothing") else f"failed: {reason}"
        assert json.loads(verdict["details_json"])["feedback"] == f"reviewer's tests {message}"


@pytest.mark.parametrize("mode", ["non-code", "permission", "tools", "sandbox", "missing"])
def test_skips_and_missing_module(
    store: Store, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    sandbox = Sandbox()
    monkeypatch.setattr(sandbox, "unavailable", lambda: "no sandbox here")
    task_id = submitted(
        store,
        runs_code=mode != "non-code",
        tools=Tools.NONE if mode == "tools" else Tools.GRANTED,
    )
    replies = ('{"verdict":"accept"}',) * 2 if mode == "missing" else (ACCEPT,)
    llm = FakeLLM().on("Review", *replies)
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        tools=builtin_registry(sandbox),
        access=executor_access([] if mode == "permission" else [Permission.SHELL]),
        review_tests=partial(_run_review_tests, sandbox),
        sleep=clock.sleep,
    )
    if mode == "missing":
        assert coord.tick(task_id, [])
    else:
        assert coord.run(task_id).status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    note = last_review_tests(store, sub.id)
    if mode in ("non-code", "missing"):
        assert note is None
        assert ("top-level test_ functions" in llm.calls[0].messages[0].content) == (
            mode == "missing"
        )
        if mode == "missing":
            assert sub.status is SubtaskStatus.REVIEW
            assert sub.attempts.transient == 1
            [interruption] = [
                row for row in store.audit_log(task_id) if row["action"] == "review_interrupted"
            ]
            assert "reviewer gave no tests" in json.loads(interruption["details_json"])["reason"]
            assert coord.run(task_id).status is TaskStatus.NEEDS_HUMAN
            assert len(llm.calls) == 2
            [failure] = [
                row for row in store.audit_log(task_id) if row["action"] == "review_failed"
            ]
            assert json.loads(failure["details_json"])["cause"] == "reviewer gave no tests"
    else:
        assert note is not None and not note["ran"]
        reason = {
            "permission": "shell_execution",
            "tools": "does not grant",
            "sandbox": "no sandbox here",
        }
        assert reason[mode] in note["reason_not_run"]


@pytest.mark.parametrize(
    ("bad", "problem"),
    [
        ("def test_broken(", "valid Python"),
        ("def helper(): pass", "top-level test_"),
        (MODULE + "#" * 20_000, "at most 20000"),
    ],
)
def test_invalid_module_gets_one_repair(
    store: Store, clock: FakeClock, bad: str, problem: str
) -> None:
    reply = json.dumps({"verdict": "accept", "tests": bad})
    llm = FakeLLM().on("Review", reply, reply, ACCEPT)
    task_id = submitted(store)
    coord = Coordinator(store, unread, llm_reviewer(llm), sleep=clock.sleep)
    assert coord.run(task_id).status is TaskStatus.DONE
    assert len(llm.calls) == 3
    assert any(problem in message.content for message in llm.calls[1].messages)
    [entry] = [row for row in store.audit_log(task_id) if row["action"] == "review_interrupted"]
    assert "StructuredOutputError" in entry["details_json"]


def test_runner_writes_module_and_parses_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Store
) -> None:
    sandbox = Sandbox()
    workspace = Workspace(tmp_path, "task")
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)

    [sub] = store.list_subtasks(submitted(store))
    path = f".review/{sub.id}/test_review_2.py"

    def run(command: list[str], cwd: Path, *, timeout: float) -> Outcome:
        assert command == [
            "python3",
            "-I",
            "-m",
            "pytest",
            "-c",
            ".review/pytest.ini",
            "--rootdir",
            ".review",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            path,
        ]
        assert cwd == workspace.root and timeout == 60
        assert workspace.read_text(path) == MODULE
        assert workspace.read_text(".review/pytest.ini") == "[pytest]\npythonpath = ..\n"
        return Outcome(
            1,
            f"FAILED {path}::test_answer - assert 1\n1 failed, 2 passed, 1 error in 0.02s\n",
            "",
            workspace_full="over quota",
            stopped="watch stopped",
        )

    monkeypatch.setattr(sandbox, "run", run)
    result = _run_review_tests(sandbox, sub, workspace, MODULE, 2)
    assert (result.exit_code, result.passed, result.failed, result.errors) == (1, 2, 1, 1)
    assert result.failing == [f"{path}::test_answer"]
    assert result.workspace_full == "over quota"
    assert result.stopped == "watch stopped"


def test_code_written_with_tools_gets_reviews_and_feedback(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a module",
            subtasks=[
                SubtaskSpec(description="write code", acceptance_criteria="works", kind=Kind.CODE)
            ],
        )
    )
    write = json.dumps(
        {
            "tool_calls": [
                {"tool": "write_file", "args": {"path": "code.py", "content": "answer = 4"}}
            ]
        }
    )
    llm = (
        FakeLLM()
        .on("Execute", write, '{"summary":"ready"}', '{"summary":"fixed"}')
        .on("Review", ACCEPT, ACCEPT)
    )
    seen: list[int] = []

    def runner(sub: Subtask, workspace: Workspace, module: str, attempt: int) -> ReviewTestResult:
        seen.append(attempt)
        workspace.write(review_test_file(sub.id, attempt), module)
        if attempt == 1:
            return ReviewTestResult(
                ran=True,
                exit_code=1,
                failed=1,
                failing=["test_answer"],
                tail="x" * 3_000 + "assertion details",
            )
        return ReviewTestResult(ran=True, exit_code=0, passed=1)

    coord = Coordinator(
        store,
        llm_executor(llm, format_final=False),
        llm_reviewer(
            llm,
            evidence=partial(evidence, store),
            test_context=partial(_review_test_context, store),
        ),
        tools=builtin_registry(),
        access=executor_access([Permission.SHELL]),
        review_tests=runner,
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is TaskStatus.DONE
    assert seen == [1, 2]
    execute = [call.prompt for call in llm.calls if call.prompt.startswith("Execute subtask")]
    assert "reviewer's tests failed: test_answer" in execute[-1]
    review = [call.prompt for call in llm.calls if call.prompt.startswith("Review")]
    previous = json.loads(review[-1].partition("\n")[2])["previous_review_tests"]
    assert previous["source"] == MODULE
    assert len(previous["tail"]) == 2_000 and previous["tail"].endswith("assertion details")
    assert "inputs" not in json.loads(review[-1].partition("\n")[2])
    [first, _] = [
        json.loads(row["details_json"])
        for row in store.audit_log(task_id)
        if row["action"] == "review_tests"
    ]
    assert "source" not in first and first["tail"] == previous["tail"]


def test_normal_run_wires_review_to_command_sandbox(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox = Sandbox()
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)
    commands: list[list[str]] = []

    def run(command: list[str], cwd: Path, *, stdin: str = "", timeout: float) -> Outcome:
        commands.append(command)
        if command[-1] == "main.py":
            return Outcome(0, "4", "")
        assert command[-1] == review_test_file(sub.id, 1)
        return Outcome(0, "1 passed in 0.01s", "")

    monkeypatch.setattr(sandbox, "run", run)
    llm = (
        FakeLLM()
        .on(
            "Execute",
            json.dumps(
                {
                    "tool_calls": [
                        {"tool": "write_file", "args": {"path": "main.py", "content": "print(4)"}},
                        {"tool": "run_command", "args": {"command": ["python3", "main.py"]}},
                    ]
                }
            ),
            '{"summary":"ready"}',
        )
        .on("Review", ACCEPT)
    )
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a running Python module",
            subtasks=[
                SubtaskSpec(
                    description="write code", acceptance_criteria="prints 4", runs_code=True
                )
            ],
        )
    )
    [sub] = store.list_subtasks(task_id)
    coord = coordinator(
        store,
        Models(llm, None),
        executor_access([Permission.SHELL]),
        make_config(),
        sandbox=sandbox,
    )
    assert coord.run(task_id).status is TaskStatus.DONE
    assert [command[-1] for command in commands] == ["main.py", review_test_file(sub.id, 1)]
    report = task_report(store, task_id)
    print_report(report, store)
    output = capsys.readouterr().out
    assert "Reviewer's tests: 1 passed, 0 failed, 0 errors" in output
    assert review_test_file(sub.id, 1) not in output


# This small fixture reproduces the four live probe failures while its smoke tests pass.
CALC = """
import math

def calculate(expression):
    if FIXED:
        expression = expression.strip()
    elif expression != expression.rstrip():
        raise ValueError("unexpected trailing space")
    if not FIXED and "^-" in expression:
        raise ValueError("unexpected minus")
    value = eval(expression.replace("^", "**"), {"__builtins__": {}})
    if FIXED and isinstance(value, complex):
        raise ValueError("expected a real result")
    if not FIXED and not isinstance(value, complex):
        math.isfinite(value)
    return value
"""
EXECUTOR_TESTS = """
from calc import calculate

def test_smoke():
    assert calculate("2^3") == 8
    assert calculate("1+1") == 2
"""
REVIEWER_TESTS = """
import pytest
from calc import calculate

def test_trailing_space():
    assert calculate("2^3 ") == 8

def test_negative_power():
    assert calculate("2^-1") == 0.5

def test_complex_rejected():
    with pytest.raises(ValueError):
        calculate("(-8)^0.5")

def test_huge_integer():
    assert calculate("10^1000") == 10**1000
"""


@confined
@pytest.mark.parametrize("fixed", [False, True])
def test_calculator_in_real_sandbox(
    store: Store, tmp_path: Path, clock: FakeClock, fixed: bool
) -> None:
    task_id = submitted(store)
    sandbox = Sandbox()
    workspace = Workspace(tmp_path, task_id)
    workspace.write("calc.py", CALC.replace("FIXED", str(fixed)))
    workspace.write("test_executor.py", EXECUTOR_TESTS)
    assert (
        sandbox.run(["python3", "-m", "pytest", "-q", "test_executor.py"], workspace.root).exit_code
        == 0
    )
    llm = FakeLLM().on("Review", json.dumps({"verdict": "accept", "tests": REVIEWER_TESTS}))
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        tools=builtin_registry(sandbox),
        workspaces=tmp_path,
        access=executor_access([Permission.SHELL]),
        review_tests=partial(_run_review_tests, sandbox),
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is (TaskStatus.DONE if fixed else TaskStatus.NEEDS_HUMAN)
    [sub] = store.list_subtasks(task_id)
    note = last_review_tests(store, sub.id)
    assert note is not None and workspace.read_text(note["file"]) == REVIEWER_TESTS
    if fixed:
        assert note["passed"] == 4
    else:
        assert note["failed"] == 4
        [verdict] = [row for row in store.audit_log(task_id) if row["action"] == "verdict"]
        feedback = json.loads(verdict["details_json"])["feedback"]
        for case in ("trailing_space", "negative_power", "complex_rejected", "huge_integer"):
            assert f"::test_{case}" in feedback


def test_error_artifact_overrules_accept_before_tests(store: Store, clock: FakeClock) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            retry=RetryPolicy(max_attempts=1),
            subtasks=[SubtaskSpec(description="code", acceptance_criteria="works", runs_code=True)],
        )
    )
    sub = store.claim("executor", task_id=task_id)
    assert sub is not None and sub.lease_token is not None
    store.submit(sub.id, sub.lease_token, {"type": "error", "message": "gave up"})
    llm = FakeLLM().on("Review", ACCEPT)

    def runner(sub: Subtask, workspace: Workspace, module: str, attempt: int) -> ReviewTestResult:
        pytest.fail("an error artifact must never be tested for acceptance")

    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        access=executor_access([Permission.SHELL]),
        review_tests=runner,
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is TaskStatus.NEEDS_HUMAN
    assert last_review_tests(store, sub.id) is None
    [verdict] = [row for row in store.audit_log(task_id) if row["action"] == "verdict"]
    assert "gave up" in json.loads(verdict["details_json"])["feedback"]


def test_code_with_unjournaled_files_is_reviewed(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a module",
            subtasks=[
                SubtaskSpec(description="write code", acceptance_criteria="works", kind=Kind.CODE)
            ],
        )
    )
    workspace = Workspace(tmp_path, task_id)

    def execute(work: Work) -> Success:
        workspace.write("script.py", "answer = 4")
        return Success(content={"summary": "ready"})

    llm = FakeLLM().on("Review", ACCEPT)

    def runner(sub: Subtask, ws: Workspace, module: str, attempt: int) -> ReviewTestResult:
        assert ws.read_text("script.py") == "answer = 4"
        return ReviewTestResult(ran=True, exit_code=0, passed=1)

    coord = Coordinator(
        store,
        execute,
        llm_reviewer(llm, evidence=partial(evidence, store, workspace=workspace)),
        access=executor_access([Permission.SHELL]),
        workspaces=tmp_path,
        review_tests=runner,
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert last_review_tests(store, sub.id) is not None
    [request] = llm.calls
    assert "answer = 4" in request.prompt
    assert "top-level test_ functions" in request.messages[0].content


@pytest.mark.parametrize("verdict", [Verdict.RETRY, Verdict.ESCALATE])
def test_nonaccept_verdict_discards_invalid_tests(store: Store, verdict: Verdict) -> None:
    [sub] = store.list_subtasks(submitted(store))
    reply = json.dumps({"verdict": verdict, "feedback": "Make it square", "tests": {"bad": True}})
    llm = FakeLLM().on("Review", reply)
    assert llm_reviewer(llm)(sub, {"summary": "Ready"}) == Review(
        verdict=verdict, feedback="Make it square"
    )
    assert len(llm.calls) == 1


@confined
@pytest.mark.parametrize("skipped", [False, True])
def test_reviews_ignore_workspace_pytest_configuration(
    store: Store, tmp_path: Path, skipped: bool
) -> None:
    task_id = submitted(store)
    [sub] = store.list_subtasks(task_id)
    workspace = Workspace(tmp_path, task_id)
    workspace.write("answer.py", "answer = 4")
    workspace.write("pytest.py", "raise RuntimeError('workspace pytest imported')")
    workspace.write("pytest.ini", "[pytest]\naddopts = --not-a-real-option\n")
    workspace.write(
        "conftest.py",
        "import pytest\ndef pytest_collection_modifyitems(items):\n"
        "    for item in items:\n        item.add_marker(pytest.mark.skip)\n",
    )
    module = "from answer import answer\ndef test_answer():\n    assert answer == 4\n"
    if skipped:
        module = "import pytest\n" + module.replace(
            "    assert answer == 4", "    pytest.importorskip('missing_answer_module')"
        )
    result = _run_review_tests(Sandbox(), sub, workspace, module, 1)
    assert result.ran and result.exit_code == 0, result.tail
    assert result.passed == (0 if skipped else 1), result.tail
    assert ("1 skipped" in result.tail) == skipped


def test_code_fallback_ignores_earlier_subtask_and_copied_inputs(
    store: Store, tmp_path: Path, clock: FakeClock
) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Write a report, then describe the code in prose",
            subtasks=[
                SubtaskSpec(description="report", acceptance_criteria="file"),
                SubtaskSpec(
                    description="describe",
                    acceptance_criteria="prose",
                    kind=Kind.CODE,
                    depends_on=[0],
                ),
            ],
        )
    )
    workspace = Workspace(tmp_path, task_id)

    def execute(work: Work) -> Success:
        if work.subtask.description == "report":
            path = workspace.root / workspace.write("report.md", "# Report")
            at = clock().timestamp()
            os.utime(path, (at, at))
            clock.advance(1)
        else:
            workspace.write("from-abc/data.csv", "copied input")
            assert not has_written_files(store, work.subtask, root=tmp_path)
        return Success(content={"summary": "ready"})

    def runner(sub: Subtask, ws: Workspace, module: str, attempt: int) -> ReviewTestResult:
        pytest.fail("the prose-only subtask did not write output files")

    llm = FakeLLM().on("Review", '{"verdict":"accept"}', '{"verdict":"accept"}')
    coord = Coordinator(
        store,
        execute,
        llm_reviewer(llm, evidence=partial(evidence, store, workspace=workspace)),
        access=executor_access([Permission.SHELL]),
        tools=builtin_registry(),
        workspaces=tmp_path,
        review_tests=runner,
        sleep=clock.sleep,
    )
    assert coord.run(task_id).status is TaskStatus.DONE
    assert "written_files" not in llm.calls[-1].prompt
    assert "top-level test_ functions" not in llm.calls[-1].messages[0].content
