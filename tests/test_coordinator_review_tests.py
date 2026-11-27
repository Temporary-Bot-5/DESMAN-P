"""Independent review tests decide whether submitted code is accepted."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest

from conftest import FakeClock
from desman.actions import print_report
from desman.agents import llm_executor, llm_reviewer
from desman.core import db
from desman.core.coordinator import (
    Coordinator,
    Failure,
    Review,
    ReviewTestRunner,
    Success,
    Verdict,
    Work,
    review_tests_enabled,
)
from desman.core.coordinator import TestRun as ReviewTestResult
from desman.core.effects import evidence, has_written_python_files, last_review_tests
from desman.core.models import Kind, Subtask, SubtaskSpec, TaskSpec, Tools
from desman.core.policy import Permission, executor_access
from desman.core.report import task_report
from desman.core.retry import RetryPolicy
from desman.core.sandbox import Limits, Outcome, Sandbox, SandboxUnavailable
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import builtin_registry
from desman.core.workspace import PathEscapeError, Workspace, workspace_root_for
from desman.llm.cache import Cached
from desman.llm.fake import FakeLLM
from desman.llm.structured import StructuredOutputError
from desman.runs import Models, _review_test_context, _run_review_tests, coordinator
from helpers import make_config
from test_sandbox import confined

MODULE = "def test_answer():\n    assert True\n"
ACCEPT = json.dumps({"verdict": "accept", "tests": MODULE})


def write_report(
    command: list[str], cwd: Path, cases: str = '<testcase name="test_answer"/>'
) -> None:
    path = command[command.index("--junitxml") + 1]
    review_dir = Path(command[-1]).parent
    assert review_dir.name.startswith(".desman-review-")
    assert Path(path).parent == review_dir and Path(path).suffix == ".xml"
    (cwd / path).write_text(
        f"<testsuites><testsuite>{cases}</testsuite></testsuites>", encoding="utf-8"
    )


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

    def runner(
        sub: Subtask,
        workspace: Workspace,
        module: str,
        attempt: int,
        cover: Callable[[float], None],
    ) -> ReviewTestResult:
        assert module == MODULE and attempt == 1
        cover(60)
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
    assert json.loads(note["details_json"]) == {
        "source": MODULE,
        "attempt": 1,
        **result.model_dump(),
    }
    if reason is not None:
        [verdict] = [row for row in rows if row["action"] == "verdict"]
        message = reason if reason.startswith("verified nothing") else f"failed: {reason}"
        assert json.loads(verdict["details_json"])["feedback"] == f"reviewer's tests {message}"


@pytest.mark.parametrize(
    "mode", ["non-code", "permission", "tools", "sandbox", "missing", "no-runner"]
)
def test_skips_and_missing_module(
    store: Store, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    sandbox = Sandbox()
    monkeypatch.setattr(
        sandbox, "unavailable", lambda: "no sandbox here" if mode == "sandbox" else None
    )
    task_id = submitted(
        store,
        runs_code=mode != "non-code",
        tools=Tools.NONE if mode == "tools" else Tools.GRANTED,
    )
    replies = ('{"verdict":"accept"}',) * 4
    llm = FakeLLM().on("Review", *replies)
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        tools=builtin_registry(sandbox),
        access=executor_access([] if mode == "permission" else [Permission.SHELL]),
        review_tests=None
        if mode == "no-runner"
        else ReviewTestRunner(partial(_run_review_tests, sandbox), sandbox.unavailable),
        sleep=clock.sleep,
    )
    if mode in ("missing", "sandbox"):
        assert coord.tick(task_id, [])
    else:
        assert coord.run(task_id).status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    note = last_review_tests(store, sub.id)
    if mode == "sandbox":
        assert not llm.calls
    else:
        assert ("top-level test_ functions" in llm.calls[0].messages[0].content) == (
            mode == "missing"
        )
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
            assert len(llm.calls) == 4
            [failure] = [
                row for row in store.audit_log(task_id) if row["action"] == "review_failed"
            ]
            assert "reviewer gave no tests" in json.loads(failure["details_json"])["cause"]
    else:
        assert note is not None and not note["ran"]
        reason = {
            "permission": "shell_execution",
            "tools": "does not grant",
            "sandbox": "no sandbox here",
            "no-runner": "runner unavailable",
        }
        assert reason[mode] in note["reason_not_run"]
        if mode == "sandbox":
            assert sub.status is SubtaskStatus.REVIEW
            assert sub.attempts.transient == 1


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
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        sleep=clock.sleep,
        access=executor_access([Permission.SHELL]),
        review_tests=lambda *_: ReviewTestResult(ran=True, exit_code=0, passed=1),
    )
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

    workspace.write("answer.py", "answer = 4")
    workspace.write("delete.txt", "keep me")
    included = [".review/old.py", "web/page.html", ".tool-results/result"]
    for path in included:
        workspace.write(path, "keep")
    for path in ("__pycache__/x.pyc", ".pytest_cache/cache"):
        workspace.write(path, "skip")
    stale = tmp_path / "review-stale"
    stale.mkdir()
    os.utime(stale, (1, 1))
    fresh = tmp_path / "review-fresh"
    fresh.mkdir()
    for changing_file in ("gone.py", "forbidden.py", "linked.py"):
        workspace.write(changing_file, "temporary")
    original_read = workspace.read

    def read(path: str, limit: int = -1) -> bytes:
        problems = {
            "gone.py": FileNotFoundError,
            "forbidden.py": PermissionError,
            "linked.py": PathEscapeError,
        }
        if path in problems:
            raise problems[path](path)
        return original_read(path, limit=limit)

    monkeypatch.setattr(workspace, "read", read)
    locations: list[Path] = []
    covered: list[float] = []

    def run(command: list[str], cwd: Path, *, timeout: float, limits: Limits) -> Outcome:
        path = command[-1]
        review_dir = Path(path).parent.as_posix()
        ini = f"{review_dir}/pytest.ini"
        report = command[-2]
        assert command == [
            "python3",
            "-I",
            "-B",
            "-m",
            "pytest",
            "-c",
            ini,
            "--rootdir",
            review_dir,
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "-q",
            "--junitxml",
            report,
            path,
        ]
        assert cwd.parent == workspace.root.parent and cwd != workspace.root and timeout == 60
        assert covered[0] == 30 and 30 < covered[1] < 31 and covered[2:] == [timeout]
        assert not stale.exists() and fresh.exists()
        locations.append(cwd)
        copy = Workspace(cwd.parent, cwd.name)
        assert copy.files() == sorted([path, ini, "answer.py", "delete.txt", *included])
        assert copy.read_text(path) == MODULE
        assert copy.read_text(ini) == "[pytest]\npythonpath = ..\n"
        copy.write("answer.py", "changed")
        (cwd / "delete.txt").unlink()
        write_report(
            command,
            cwd,
            '<testcase classname="test_review" name="test_answer"><failure/></testcase>'
            '<testcase name="test_cleanup"><error/></testcase>'
            '<testcase name="test_one"/><testcase name="test_two"/>'
            '<testcase name="test_skip"><skipped/></testcase>',
        )
        return Outcome(
            1,
            "99 passed in 0.02s\n",
            "",
            workspace_full="over quota",
            stopped="watch stopped",
        )

    monkeypatch.setattr(sandbox, "run", run)
    result = _run_review_tests(sandbox, sub, workspace, MODULE, 2, covered.append)
    assert result.skipped_files == 3
    assert result.skipped_names == ["forbidden.py", "gone.py", "linked.py"]
    assert (result.exit_code, result.passed, result.failed, result.errors) == (1, 2, 1, 1)
    assert result.failing == ["test_review::test_answer", "test_cleanup"]
    assert result.skipped == 1
    assert result.workspace_full == "over quota"
    assert result.stopped == "watch stopped"
    assert not list(workspace.root.glob(".desman-review-*"))
    assert workspace.read_text("answer.py") == "answer = 4"
    assert workspace.read_text("delete.txt") == "keep me"
    assert not locations[0].exists()


def test_copy_refreshes_review_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: Store
) -> None:
    sandbox = Sandbox()
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)

    def run(command: list[str], cwd: Path, **kwargs: object) -> Outcome:
        write_report(command, cwd)
        return Outcome(0, "", "")

    monkeypatch.setattr(sandbox, "run", run)
    workspace = Workspace(tmp_path, "task")
    for index in range(501):
        workspace.write(f"{index}.txt", b"")

    [sub] = store.list_subtasks(submitted(store))
    covered: list[float] = []
    result = _run_review_tests(sandbox, sub, workspace, MODULE, 1, covered.append)

    assert result.passed == 1
    assert covered == pytest.approx([30, 55.05, 30.05, 60])


@pytest.mark.parametrize(("path", "content"), [("code.py", "answer = 4"), ("code.sh", "echo 4")])
def test_code_written_with_tools_gets_reviews_and_feedback(
    store: Store, clock: FakeClock, path: str, content: str
) -> None:
    python_output = path.endswith(".py")
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a module",
            subtasks=[
                SubtaskSpec(description="write code", acceptance_criteria="works", kind=Kind.CODE)
            ],
        )
    )
    write = json.dumps(
        {"tool_calls": [{"tool": "write_file", "args": {"path": path, "content": content}}]}
    )
    llm = (
        FakeLLM()
        .on("Execute", write, '{"summary":"ready"}', '{"summary":"fixed"}')
        .on("Review", ACCEPT if python_output else '{"verdict":"accept"}', ACCEPT)
    )
    seen: list[int] = []

    def runner(
        sub: Subtask,
        workspace: Workspace,
        module: str,
        attempt: int,
        cover: Callable[[float], None],
    ) -> ReviewTestResult:
        seen.append(attempt)
        if attempt == 1:
            return ReviewTestResult(
                ran=True,
                exit_code=1,
                failed=1,
                failing=["test_answer"],
                skipped_files=6,
                skipped_names=["a.py", "b.py", "c.py", "d.py", "e.py"],
                tail="x" * 3_000 + "assertion details",
            )
        return ReviewTestResult(ran=True, exit_code=0, passed=1)

    executor = llm_executor(llm, format_final=False)

    def execute(work: Work) -> Success | Failure:
        result = executor(work)
        assert isinstance(result, (Success, Failure))
        return result

    coord = Coordinator(
        store,
        execute,
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
    reviews = [request for request in llm.calls if request.prompt.startswith("Review")]
    assert reviews and all(
        ("top-level test_ functions" in request.messages[0].content) == python_output
        for request in reviews
    )
    if not python_output:
        assert seen == [] and len(reviews) == 1
        return
    assert seen == [1, 2]
    executions = [call.prompt for call in llm.calls if call.prompt.startswith("Execute subtask")]
    assert "reviewer's tests failed: test_answer" in executions[-1]
    assert "6 files could not be copied: 'a.py', 'b.py', 'c.py', 'd.py', 'e.py'" in executions[-1]
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
    assert first["source"] == MODULE and first["tail"] == previous["tail"]


def test_normal_run_wires_review_to_command_sandbox(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sandbox = Sandbox()
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)
    commands: list[list[str]] = []

    def run(
        command: list[str],
        cwd: Path,
        *,
        stdin: str = "",
        timeout: float,
        limits: Limits | None = None,
    ) -> Outcome:
        commands.append(command)
        if command[-1] == "main.py":
            return Outcome(0, "4", "")
        write_report(command, cwd)
        return Outcome(0, "1 passed in 0.01s\natexit: closing", "")

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
    coord = coordinator(
        store,
        Models(llm, None),
        executor_access([Permission.SHELL]),
        make_config(),
        sandbox=sandbox,
    )
    assert coord.run(task_id).status is TaskStatus.DONE
    assert len(commands) == 2 and commands[0][-1] == "main.py"
    assert Path(commands[1][-1]).name == "test_review.py"
    report = task_report(store, task_id)
    print_report(report, store)
    output = capsys.readouterr().out
    assert "Reviewer's tests (attempt 1) — passed: 1, failed: 0, errors: 0" in output
    assert not (Workspace(workspace_root_for(store.conn), task_id).root / ".review").exists()
    assert commands[1][-1] not in output
    [subreport] = report.subtasks
    assert subreport.model_dump()["review_tests"] == {
        "attempt": 1,
        "ran": True,
        "reason_not_run": None,
        "missing_packages": [],
        "exit_code": 0,
        "passed": 1,
        "failed": 0,
        "errors": 0,
        "timed_out": False,
        "report_missing": False,
        "stopped": "",
    }
    attempts = subreport.attempts.model_copy(update={"quality": 1})
    report.subtasks[0] = subreport.model_copy(update={"attempts": attempts})
    print_report(report, store)
    assert "Reviewer's tests (attempt 1)" in capsys.readouterr().out


def test_status_recovers_attempt_from_legacy_review_note(store: Store) -> None:
    task_id = submitted(store)
    [sub] = store.list_subtasks(task_id)
    store.note(sub.id, Actor.EXECUTOR, "attempt_started", {"attempt": 2})
    store.note(sub.id, Actor.REVIEWER, "review_tests", {"ran": False})
    store.note(sub.id, Actor.EXECUTOR, "attempt_started", {"attempt": 3})
    [reported] = task_report(store, task_id).subtasks
    assert reported.review_tests is not None and reported.review_tests.attempt == 2


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
    with pytest.raises(Exception):
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
    assert note is not None and note["source"] == REVIEWER_TESTS
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

    def runner(
        sub: Subtask,
        workspace: Workspace,
        module: str,
        attempt: int,
        cover: Callable[[float], None],
    ) -> ReviewTestResult:
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


@pytest.mark.parametrize("mode", ["pass", "missing", "outage"])
def test_code_with_command_outputs_is_reviewed(
    store: Store, tmp_path: Path, clock: FakeClock, mode: str
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
        assert work.subtask.lease_token is not None
        lease = work.subtask.lease_token
        store.start_effect(work.subtask.id, lease, 0, 0, "run_command", "key")
        clock.advance(1)
        for name, content in {
            "script.py": "answer = 4",
            "helper.sh": "echo 4",
            "older.md": "# Notes",
            **{f"a{index}.csv": "1,2" for index in range(12)},
        }.items():
            path = workspace.root / workspace.write(name, content)
            at = clock().timestamp() - (0.5 if name == "older.md" else 0)
            os.utime(path, (at, at))
        clock.advance(1)
        store.finish_effect(work.subtask.id, lease, 0, 0, {"ok": True})
        clock.advance(3)
        late = workspace.root / workspace.write("late.py", "different subtask")
        at = clock().timestamp()
        os.utime(late, (at, at))
        return Success(content={"summary": "ready"})

    llm = FakeLLM().on("Review", ACCEPT)

    def runner(
        sub: Subtask, ws: Workspace, module: str, attempt: int, cover: Callable[[float], None]
    ) -> ReviewTestResult:
        assert ws.read_text("script.py") == "answer = 4"
        assert mode != "missing"
        if mode == "outage":
            raise SandboxUnavailable("backend went away")
        return ReviewTestResult(ran=True, exit_code=0, passed=1)

    reviewer = llm_reviewer(llm, evidence=partial(evidence, store, workspace=workspace))
    coord = Coordinator(
        store,
        execute,
        (lambda *_: Review(verdict=Verdict.ACCEPT)) if mode == "missing" else reviewer,
        access=executor_access([Permission.SHELL]),
        workspaces=tmp_path,
        review_tests=runner,
        sleep=clock.sleep,
    )
    if mode == "missing":
        assert coord.tick(task_id, [])  # Execute and submit.
        assert coord.tick(task_id, [])  # Review the submitted work.
        [sub] = store.list_subtasks(task_id)
        assert sub.status is SubtaskStatus.REVIEW and sub.attempts.transient == 1
        [entry] = [row for row in store.audit_log(task_id) if row["action"] == "review_interrupted"]
        assert json.loads(entry["details_json"])["reason"].startswith("reviewer gave no tests:")
        return
    assert coord.run(task_id).status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    note = last_review_tests(store, sub.id)
    assert note is not None and note["ran"] is (mode == "pass")
    if mode == "outage":
        assert note["reason_not_run"] == "backend went away"
    [request] = llm.calls
    assert "answer = 4" in request.prompt
    assert "late.py" not in request.prompt
    shown = json.loads(request.prompt.partition("\n")[2])["evidence"]["written_files"]
    assert [file["path"] for file in shown[:3]] == ["script.py", "helper.sh", "older.md"]
    assert len(shown) == 12
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
    assert result.skipped == int(skipped), result.tail


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
            unrelated = workspace.root / workspace.write("unrelated.py", "not from a command")
            at = clock().timestamp()
            os.utime(unrelated, (at, at))
            assert not has_written_python_files(store, work.subtask, root=tmp_path)
        return Success(content={"summary": "ready"})

    def runner(
        sub: Subtask, ws: Workspace, module: str, attempt: int, cover: Callable[[float], None]
    ) -> ReviewTestResult:
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


@pytest.mark.parametrize("runs_code", [False, True])
def test_missing_tests_are_repaired_without_caching(store: Store, runs_code: bool) -> None:
    [sub] = store.list_subtasks(submitted(store, runs_code=runs_code))
    if not runs_code:
        sub = sub.model_copy(update={"kind": Kind.CODE})
    missing = '{"verdict":"accept"}'
    llm = FakeLLM().on("Review", missing, missing, missing, ACCEPT)
    cached = Cached(llm, store.conn, "reviewer")
    reviewer = llm_reviewer(cached, evidence=lambda *_: {"written_files": [{"path": "main.py"}]})

    setting = review_tests_enabled.set(True)
    try:
        with pytest.raises(StructuredOutputError, match="reviewer gave no tests"):
            reviewer(sub, {"summary": "Ready"})
        assert len(llm.calls) == 2
        assert store.conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 0
        repaired = reviewer(sub, {"summary": "Ready"})
    finally:
        review_tests_enabled.reset(setting)
    assert isinstance(repaired, Review) and repaired.tests == MODULE
    assert len(llm.calls) == 4 and cached.hits == 0
    assert [row[0] for row in store.conn.execute("SELECT text FROM response_cache")] == [ACCEPT]


@confined
def test_review_mutations_stay_in_sandbox_copy(store: Store, tmp_path: Path) -> None:
    task_id = submitted(store)
    [sub] = store.list_subtasks(task_id)
    workspace = Workspace(tmp_path, task_id)
    workspace.write("answer.py", "answer = 4")
    workspace.write("delete.txt", "keep me")
    module = """
import atexit
from pathlib import Path

atexit.register(print, "atexit: review finished")

def test_changes():
    answer = Path("answer.py")
    assert answer.read_text() == "answer = 4"
    answer.write_text("changed")
    Path("delete.txt").unlink()
"""
    result = _run_review_tests(Sandbox(), sub, workspace, module, 1)
    assert result.passed == 1, result.tail
    assert result.tail.endswith("atexit: review finished\n")
    assert not (workspace.root / ".review").exists()
    assert workspace.read_text("answer.py") == "answer = 4"
    assert workspace.read_text("delete.txt") == "keep me"
    assert list(tmp_path.glob("review-*")) == []


@pytest.mark.parametrize("reason", ["disk", "outage", "quota", "write-error", "fits"])
def test_review_infrastructure_failure(
    store: Store, clock: FakeClock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    task_id = submitted(store)
    sandbox = Sandbox(limits=Limits(workspace_bytes=100, workspace_files=1))
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)
    workspace = Workspace(tmp_path, task_id)
    workspace.write("answer.py", "x" * (101 if reason == "quota" else 100))
    if reason == "write-error":

        def disk_full(self: Workspace, path: str, data: bytes | str) -> str:
            raise OSError("disk full")

        monkeypatch.setattr(Workspace, "write", disk_full)

    def run(command: list[str], cwd: Path, *, timeout: float, limits: Limits) -> Outcome:
        assert reason not in {"quota", "write-error"}
        assert limits.workspace_files == 5
        assert limits.workspace_bytes == (
            100 + len(MODULE.encode()) + len("[pytest]\npythonpath = ..\n") + 1024 * 1024
        )
        if reason == "fits":
            write_report(command, cwd)
            return Outcome(0, "===== 1 passed in 0.01s =====\natexit: closing", "")
        if reason == "outage":
            raise SandboxUnavailable("backend went away")
        return Outcome(None, "", "", stopped="disk space too low", disk_low=True)

    monkeypatch.setattr(sandbox, "run", run)
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(FakeLLM().on("Review", ACCEPT, ACCEPT, ACCEPT)),
        access=executor_access([Permission.SHELL]),
        workspaces=tmp_path,
        review_tests=partial(_run_review_tests, sandbox),
        sleep=clock.sleep,
    )
    assert coord.tick(task_id, [])
    [sub] = store.list_subtasks(task_id)
    if reason == "fits":
        assert sub.status is SubtaskStatus.DONE
    elif reason == "quota":
        assert sub.status is SubtaskStatus.NEEDS_HUMAN
        assert sub.attempts.transient == 0
    else:
        assert sub.status is SubtaskStatus.REVIEW and sub.attempts.transient == 1
    note = last_review_tests(store, sub.id)
    if reason == "write-error":
        assert note is None
        [entry] = [row for row in store.audit_log(task_id) if row["action"] == "review_interrupted"]
        assert "disk full" in json.loads(entry["details_json"])["reason"]
        assert list(tmp_path.glob("review-*")) == []
        return
    assert note is not None
    assert note["disk_low"] is (reason == "disk")
    assert note["ran"] is (reason in {"disk", "fits"})
    if reason == "quota":
        assert "quota: " in note["copy_over_quota"]
        assert sub.attempts.quality == 0
        [entry] = [row for row in store.audit_log(task_id) if row["action"] == "review_failed"]
        details = json.loads(entry["details_json"])
        assert details["failure"] == "permanent"
        assert "at least 1 entries, 101 bytes" in details["reason"]
    assert list(tmp_path.glob("review-*")) == []


def test_lost_reservation_keeps_final_verdict_without_test_note(
    store: Store, clock: FakeClock
) -> None:
    task_id = submitted(store)

    def runner(
        sub: Subtask, ws: Workspace, module: str, attempt: int, cover: Callable[[float], None]
    ) -> ReviewTestResult:
        clock.advance(120)
        assert store.reserve_review(sub.id, "other reviewer", seconds=60)
        return ReviewTestResult(ran=True, exit_code=1, failed=1, failing=["test_answer"])

    coord = Coordinator(
        store,
        unread,
        lambda *_: Review(verdict=Verdict.ACCEPT, feedback="looks good", tests=MODULE),
        access=executor_access([Permission.SHELL]),
        review_tests=runner,
        review_seconds=1,
        sleep=clock.sleep,
    )
    coord.tick(task_id, [])
    [sub] = store.list_subtasks(task_id)
    assert last_review_tests(store, sub.id) is None
    [entry] = [row for row in store.audit_log(task_id) if row["action"] == "review_discarded"]
    details = json.loads(entry["details_json"])
    assert (details["verdict"], details["feedback"]) == (
        "retry",
        "reviewer's tests failed: test_answer",
    )


def test_command_fallback_ends_at_submit(store: Store, tmp_path: Path, clock: FakeClock) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a module",
            subtasks=[
                SubtaskSpec(description="write code", acceptance_criteria="works", kind=Kind.CODE)
            ],
        )
    )
    sub = store.claim("executor", task_id=task_id)
    assert sub is not None and sub.lease_token is not None
    store.note(sub.id, Actor.EXECUTOR, "attempt_started", {})
    store.start_effect(sub.id, sub.lease_token, 0, 0, "run_command", "key")
    workspace = Workspace(tmp_path, task_id)
    path = workspace.root / workspace.write("during.py", "included")
    at = clock().timestamp()
    os.utime(path, (at, at))
    clock.advance(1)
    store.submit(sub.id, sub.lease_token, {"summary": "ready"})
    clock.advance(3)
    late = workspace.root / workspace.write("after.py", "excluded")
    at = clock().timestamp()
    os.utime(late, (at, at))
    result = evidence(store, sub, workspace=workspace)
    assert [file["path"] for file in result["written_files"]] == ["during.py"]
    assert "submit times" in result["written_files_timing"]
    path.unlink()
    assert not has_written_python_files(store, sub, root=tmp_path)
    assert "written_files_timing" not in evidence(store, sub, workspace=workspace)


@pytest.mark.parametrize("report", [None, "<bad xml", "x" * (1024 * 1024 + 1)])
def test_invalid_report_cannot_verify_fake_summary(
    store: Store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    report: str | None,
) -> None:
    task_id = submitted(store)
    sandbox = Sandbox()
    monkeypatch.setattr(sandbox, "unavailable", lambda: None)

    def run(command: list[str], cwd: Path, **kwargs: object) -> Outcome:
        if report is not None:
            (cwd / command[-2]).write_text(report, encoding="utf-8")
        return Outcome(0, "1 passed in 0.01s", "")

    monkeypatch.setattr(sandbox, "run", run)
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(FakeLLM().on("Review", ACCEPT)),
        workspaces=tmp_path,
        access=executor_access([Permission.SHELL]),
        review_tests=partial(_run_review_tests, sandbox),
    )
    assert coord.run(task_id).status is TaskStatus.NEEDS_HUMAN
    [verdict] = [row for row in store.audit_log(task_id) if row["action"] == "verdict"]
    assert json.loads(verdict["details_json"])["feedback"] == (
        "reviewer's tests produced no report\n1 passed in 0.01s"
    )


@confined
@pytest.mark.parametrize("fake_summary", [False, True])
def test_review_requires_report_and_copies_web_page(
    store: Store, tmp_path: Path, clock: FakeClock, fake_summary: bool
) -> None:
    task_id = submitted(store)
    sandbox = Sandbox()
    workspace = Workspace(tmp_path, task_id)
    workspace.write("web/page.html", "<p>4</p>")
    source = "from pathlib import Path\nanswer = Path('web/page.html').read_text()\n"
    if fake_summary:
        source += "import os\nprint('1 passed in 0.01s', flush=True)\nos._exit(0)\n"
    workspace.write("answer.py", source)
    module = "from answer import answer\ndef test_answer():\n    assert answer == '<p>4</p>'\n"
    coord = Coordinator(
        store,
        unread,
        lambda *_: Review(verdict=Verdict.ACCEPT, tests=module),
        tools=builtin_registry(sandbox),
        workspaces=tmp_path,
        access=executor_access([Permission.SHELL]),
        review_tests=partial(_run_review_tests, sandbox),
        sleep=clock.sleep,
    )
    expected = TaskStatus.NEEDS_HUMAN if fake_summary else TaskStatus.DONE
    assert coord.run(task_id).status is expected
    [sub] = store.list_subtasks(task_id)
    note = last_review_tests(store, sub.id)
    assert note is not None and note["exit_code"] == 0
    assert note["passed"] == (0 if fake_summary else 1)
    if fake_summary:
        [verdict] = [row for row in store.audit_log(task_id) if row["action"] == "verdict"]
        assert json.loads(verdict["details_json"])["feedback"].startswith(
            "reviewer's tests produced no report"
        )


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ({"ran": True, "timed_out": True, "report_missing": True}, "timed out."),
        ({"ran": True, "report_missing": True}, "produced no report."),
        ({"ran": True, "exit_code": 5}, "— pytest exited 5"),
        (
            {"ran": False, "missing_packages": ["requests"]},
            "did not run. needs package requests, not installed in the sandbox",
        ),
        ({"ran": True, "stopped": "disk space too low"}, "stopped: disk space too low"),
        ({"ran": True, "workspace_full": "quota exceeded"}, "stopped: quota exceeded"),
        ({"ran": False, "reason_not_run": "no sandbox here"}, "did not run. no sandbox here"),
    ],
)
def test_status_shows_review_test_outcome(
    store: Store, capsys: pytest.CaptureFixture[str], note: dict[str, object], expected: str
) -> None:
    task_id = submitted(store)
    [sub] = store.list_subtasks(task_id)
    store.note(sub.id, Actor.REVIEWER, "review_tests", {"attempt": 1, **note})
    print_report(task_report(store, task_id), store)
    assert f"Reviewer's tests (attempt 1) {expected}" in capsys.readouterr().out


@pytest.mark.parametrize("runs_code", [False, True])
@pytest.mark.parametrize(
    "mode", ["missing", "call", "local", "mixed", "nested", pytest.param("real", marks=confined)]
)
def test_missing_requests_in_review(
    store: Store,
    tmp_path: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    runs_code: bool,
    mode: str,
) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a fetch module",
            retry=RetryPolicy(max_attempts=1),
            subtasks=[
                SubtaskSpec(
                    description="write fetch.py",
                    acceptance_criteria="produce the module",
                    kind=Kind.CODE,
                    runs_code=runs_code,
                )
            ],
        )
    )
    sub = store.claim("executor", task_id=task_id)
    assert sub is not None and sub.lease_token is not None
    workspace = Workspace(tmp_path, task_id)
    # The real run needs a package the host's /usr surely lacks (requests may be there).
    package = "desman_absent_package" if mode == "real" else "requests"
    workspace.write("fetch.py", f"import {package}\n")
    if not runs_code:
        store.start_effect(sub.id, sub.lease_token, 0, 0, "write_file", "key")
        store.finish_effect(
            sub.id, sub.lease_token, 0, 0, {"ok": True, "output": {"path": "fetch.py"}}
        )
    store.submit(sub.id, sub.lease_token, {"summary": "ready"})
    sandbox = Sandbox()
    if mode == "local":
        workspace.write("requests/__init__.py", "")
    if mode != "real":
        monkeypatch.setattr(sandbox, "unavailable", lambda: None)

        def run(command: list[str], cwd: Path, **_: object) -> Outcome:
            assert (cwd / "fetch.py").read_text() == "import requests\n"
            name = "requests.api" if mode == "nested" else "requests"
            cases = (
                '<testcase name="test_review"><error message="collection failure">'
                "fetch.py:1: in &lt;module&gt;\n    import requests\n"
                f"E   ModuleNotFoundError: No module named '{name}'</error></testcase>"
            )
            if mode == "call":
                cases = (
                    '<testcase name="test_fetch"><failure '
                    'message="ModuleNotFoundError: No module named &apos;requests&apos;">'
                    "E   ModuleNotFoundError: No module named 'requests'\n\n"
                    "fetch.py:1: ModuleNotFoundError</failure></testcase>"
                )
            if mode == "mixed":
                cases += '<testcase name="test_other"><failure>AssertionError</failure></testcase>'
            write_report(command, cwd, cases)
            return Outcome(2, "", "")

        monkeypatch.setattr(sandbox, "run", run)
    module = f"import fetch\n\ndef test_fetch():\n    assert fetch.{package}\n"
    llm = FakeLLM().on("Review", json.dumps({"verdict": "accept", "tests": module}))
    coord = Coordinator(
        store,
        unread,
        llm_reviewer(llm),
        access=executor_access([Permission.SHELL]),
        workspaces=tmp_path,
        review_tests=ReviewTestRunner(partial(_run_review_tests, sandbox), sandbox.unavailable),
        sleep=clock.sleep,
    )
    assert coord.tick(task_id, [])
    note = last_review_tests(store, sub.id)
    assert note is not None and note["source"] == module
    if mode in {"missing", "call", "real"}:
        assert store.get_subtask(sub.id).status is SubtaskStatus.DONE
        assert not note["ran"]
        assert note["reason_not_run"] == f"needs package {package}, not installed in the sandbox"
    else:
        assert store.get_subtask(sub.id).status is not SubtaskStatus.DONE
        assert note["ran"] and note["errors"] == 1
    assert len(llm.calls) == 1


def test_command_evidence_uses_earlier_execution_and_reads_only_top_candidates(
    store: Store, tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="Produce a module",
            subtasks=[
                SubtaskSpec(description="write code", acceptance_criteria="works", kind=Kind.CODE)
            ],
        )
    )
    sub = store.claim("executor", task_id=task_id)
    assert sub is not None and sub.lease_token is not None
    workspace = Workspace(tmp_path, task_id)
    store.start_effect(sub.id, sub.lease_token, 0, 0, "run_command", "key")
    started = clock().timestamp()
    clock.advance(1)
    store.finish_effect(sub.id, sub.lease_token, 0, 0, {"ok": True})
    for name in ("script.py", *(f"extra-{i:02}.txt" for i in range(13))):
        path = workspace.root / workspace.write(name, "answer")
        at = started - 2 if name == "script.py" else started + 3
        os.utime(path, (at, at))
    store.submit(sub.id, sub.lease_token, {"summary": "first"})
    store.transition(sub.id, SubtaskStatus.REVIEW, SubtaskStatus.READY, Actor.REVIEWER)
    clock.advance(10)
    again = store.claim("executor", task_id=task_id)
    assert again is not None and again.lease_token is not None
    store.submit(again.id, again.lease_token, {"summary": "second"})

    read = Workspace.read
    reads: list[str] = []

    def track_read(ws: Workspace, path: str, *, limit: int = -1) -> bytes:
        reads.append(path)
        return read(ws, path, limit=limit)

    monkeypatch.setattr(Workspace, "read", track_read)
    assert has_written_python_files(store, again, root=tmp_path)
    assert reads == []
    shown = evidence(store, again, workspace=workspace)["written_files"]
    assert reads == ["script.py", *(f"extra-{i:02}.txt" for i in range(11))]
    assert [file["path"] for file in shown] == reads
