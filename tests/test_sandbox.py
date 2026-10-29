"""Sandbox for generated code (ROADMAP 3.3): commands run only under bwrap, confined.

The tests marked `confined` start real sandboxes (real processes and, for the timeout, real
time); they are skipped where bubblewrap cannot run. CI installs it and sets
DESMAN_REQUIRE_SANDBOX, so there they never skip.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import cli as cli_module
from desman.agents import llm_executor
from desman.cli import EXIT_DONE, cli
from desman.config import ConfigManager, default_config_path
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Verdict
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.policy import Permission, executor_access
from desman.core.sandbox import Limits, Outcome, Sandbox, SandboxUnavailable
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.core.tools import MAX_RUN_SECONDS, Toolbox, ToolCall, builtin_registry
from desman.core.workspace import PathEscapeError, Workspace
from desman.llm.fake import FakeLLM
from helpers import make_config

PROBLEM = Sandbox().unavailable()
if PROBLEM is not None and os.environ.get("DESMAN_REQUIRE_SANDBOX"):  # set in CI
    raise RuntimeError(f"DESMAN_REQUIRE_SANDBOX is set, but there is no sandbox: {PROBLEM}")
confined = pytest.mark.skipif(PROBLEM is not None, reason=f"no sandbox here: {PROBLEM}")


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return Workspace(tmp_path / "workspaces", "t1")


def run(ws: Workspace, *command: str, stdin: str = "", **limits: Any) -> Outcome:
    return Sandbox(Limits(**limits)).run(command, ws.root, stdin=stdin)


def py(ws: Workspace, code: str, **limits: Any) -> Outcome:
    return run(ws, "python3", "-c", code, **limits)


# ── Confinement ─────────────────────────────────────────────────────────────


@confined
def test_code_works_in_the_workspace_and_its_files_stay(ws: Workspace) -> None:
    ws.write("data.txt", "21")
    outcome = py(
        ws,
        "import os; n = int(open('data.txt').read()); os.makedirs('out'); "
        "open('out/answer.txt', 'w').write(str(n * 2)); print(os.getcwd(), os.environ['HOME'])",
    )
    assert outcome == Outcome(exit_code=0, stdout="/workspace /workspace\n", stderr="")
    assert ws.read_text("out/answer.txt") == "42"
    assert run(ws, "sh", "-c", "read x; echo got $x", stdin="hi\n").stdout == "got hi\n"


@confined
def test_there_is_no_network(ws: Workspace) -> None:
    outcome = py(
        ws,
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "except OSError as e:\n"
        "    print('refused', e.errno)\n"
        "print([n for _, n in socket.if_nameindex()])",
    )
    assert outcome.exit_code == 0, outcome.stderr
    assert outcome.stdout.startswith("refused")
    assert "['lo']" in outcome.stdout  # a loopback of its own, nothing else


@confined
def test_nothing_of_the_person_is_visible(ws: Workspace, tmp_path: Path) -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"GROQ_API_KEY": "gsk_x"}))
    paths = [str(Path.home()), str(tmp_path), str(default_config_path()), "/etc/passwd", "/root"]
    outcome = py(ws, f"import os; print([p for p in {paths!r} if os.path.exists(p)])")
    assert outcome.stdout == "[]\n", outcome.stderr
    top = py(ws, "import os; print(sorted(os.listdir('/')))").stdout
    assert "home" not in top and "workspace" in top


@confined
def test_the_environment_is_cleared(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    outcome = py(ws, "import os, json; print(json.dumps(dict(os.environ)))")
    env = json.loads(outcome.stdout)
    assert "GROQ_API_KEY" not in env and "gsk_secret" not in outcome.stdout
    assert set(env) <= {"PATH", "HOME", "TMPDIR", "LANG", "PWD"}


@confined
def test_the_system_is_read_only_and_tmp_is_private(ws: Workspace) -> None:
    name = f"/tmp/desman-test-{os.getpid()}-{time.monotonic_ns()}"
    outcome = run(
        ws, "sh", "-c", f"touch /usr/x 2>/dev/null; echo $?; echo hi > {name}; cat {name}"
    )
    assert outcome.stdout.splitlines() == ["1", "hi"]
    assert not Path(name).exists()


@confined
def test_it_cannot_make_namespaces_or_escape_through_links(ws: Workspace) -> None:
    nested = run(ws, "unshare", "--user", "--map-root-user", "true")
    assert nested.exit_code != 0
    run(ws, "ln", "-s", "/etc/hostname", "link")
    with pytest.raises(PathEscapeError):
        ws.read("link")  # a link made inside is never followed outside
    assert ws.files() == []


# ── Limits ──────────────────────────────────────────────────────────────────


@confined
def test_a_timeout_kills_everything_it_started(ws: Workspace) -> None:
    started = time.monotonic()
    outcome = run(ws, "sh", "-c", "(sleep 30; touch late) & sleep 30", timeout=1.0)
    assert outcome.timed_out and outcome.exit_code is None
    assert time.monotonic() - started < 10
    time.sleep(0.5)
    assert not ws.exists("late")


@confined
def test_processes_files_memory_and_cpu_are_limited(ws: Workspace) -> None:
    forks = py(
        ws,
        "import os, time\n"
        "n = 0\n"
        "try:\n"
        "    for _ in range(100):\n"
        "        if os.fork() == 0:\n"
        "            time.sleep(1); os._exit(0)\n"
        "        n += 1\n"
        "except OSError:\n"
        "    print(n)",
        processes=8,
    )
    assert forks.stdout.strip() and int(forks.stdout) < 8, forks
    big = py(ws, "open('big', 'w').write('a' * 2_000_000)", file_bytes=1_000_000)
    assert big.exit_code != 0 and "File too large" in big.stderr
    memory = py(ws, "x = bytearray(400 * 2**20)", memory_bytes=200 * 2**20)
    assert memory.exit_code != 0 and "MemoryError" in memory.stderr
    cpu = py(ws, "while True: pass", cpu_seconds=1, timeout=20.0)
    assert not cpu.timed_out and cpu.exit_code is not None and cpu.exit_code > 128


@confined
def test_output_is_cut_keeping_the_start_and_the_end(ws: Workspace) -> None:
    outcome = py(
        ws,
        "import sys\n"
        "for i in range(100_000): print(i)\n"
        "for i in range(100_000): print(i, file=sys.stderr)",
        stdout_chars=20,
        stderr_chars=20,
    )
    assert outcome.exit_code == 0
    assert (
        outcome.stdout_cut and outcome.stdout.startswith("0\n1\n2\n") and len(outcome.stdout) == 20
    )
    assert outcome.stderr_cut and outcome.stderr.endswith("99998\n99999\n")


# ── No sandbox: nothing runs ────────────────────────────────────────────────


def missing() -> Sandbox:
    return Sandbox(bwrap="/nonexistent/bwrap")


def test_without_a_working_bwrap_nothing_runs(ws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(SandboxUnavailable, match="cannot be started"):
        missing().run(["touch", "x"], ws.root)
    refusing = tmp_path / "bwrap"
    refusing.write_text(
        "#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\nexit 1\n"
    )
    refusing.chmod(0o755)
    sandbox = Sandbox(bwrap=str(refusing))
    assert "No permissions to create new namespace" in (sandbox.unavailable() or "")
    with pytest.raises(SandboxUnavailable):
        sandbox.run(["touch", "x"], ws.root)
    assert ws.files() == []


def test_without_bwrap_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("desman.core.sandbox.shutil.which", lambda name: None)
    assert Sandbox().unavailable() == "bubblewrap (bwrap) is not installed"


# ── The run_command tool ────────────────────────────────────────────────────


class Scripted(Sandbox):
    """A sandbox that works and answers without running anything."""

    def __init__(self, outcome: Outcome) -> None:
        super().__init__(bwrap="bwrap", prlimit="prlimit")
        self.outcome = outcome
        self.calls: list[tuple[list[str], str, float | None]] = []

    def unavailable(self) -> str | None:
        return None

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        stdin: str = "",
        timeout: float | None = None,
    ) -> Outcome:
        self.calls.append((list(command), stdin, timeout))
        return self.outcome


def box(ws: Workspace, sandbox: Sandbox, *switches: Permission) -> tuple[Toolbox, list[Any]]:
    entries: list[Any] = []
    toolbox = Toolbox(
        ws,
        builtin_registry(sandbox),
        access=executor_access(switches),
        record=lambda action, details: entries.append((action, details)),
    )
    return toolbox, entries


def names(toolbox: Toolbox) -> list[str]:
    return [spec["name"] for spec in toolbox.specs()]


def test_run_command_needs_the_human_switch(ws: Workspace) -> None:
    sandbox = Scripted(Outcome(exit_code=0, stdout="", stderr=""))
    toolbox, entries = box(ws, sandbox)
    assert "run_command" not in names(toolbox)
    result = toolbox.call(ToolCall(tool="run_command", args={"command": ["true"]}))
    assert not result.ok and "shell_execution" in (result.error or "")
    assert entries[0][0] == "tool_refused" and sandbox.calls == []


def test_run_command_needs_a_working_sandbox(ws: Workspace) -> None:
    toolbox, entries = box(ws, missing(), Permission.SHELL)
    assert names(toolbox) == ["read_file", "write_file", "list_files"]
    result = toolbox.call(ToolCall(tool="run_command", args={"command": ["true"]}))
    assert not result.ok and "cannot run here" in (result.error or "")
    [(action, details)] = entries
    assert action == "tool_refused" and "cannot be started" in details["reason"]


def test_the_result_the_model_gets_and_the_audit(ws: Workspace) -> None:
    outcome = Outcome(exit_code=None, stdout="partial", stderr="", timed_out=True, stdout_cut=True)
    sandbox = Scripted(outcome)
    toolbox, entries = box(ws, sandbox, Permission.SHELL)
    assert "run_command" in names(toolbox)
    args = {"command": ["python3", "main.py"], "stdin": "secret input", "timeout": 5}
    result = toolbox.call(ToolCall(tool="run_command", args=args))
    assert result.ok and result.output == {
        "exit_code": None,
        "stdout": "partial",
        "stderr": "",
        "timed_out": True,
        "stdout_cut": True,
    }
    assert sandbox.calls == [(["python3", "main.py"], "secret input", 5)]
    [(action, details)] = entries
    assert action == "tool_called" and details["permission"] == "shell_execution"
    assert details["args"]["stdin"] == "[12 chars]"  # what was typed in is not kept


@pytest.mark.parametrize(
    "args",
    [
        {"command": []},
        {"command": ["python3", "a\0b"]},
        {"command": [""]},
        {"command": "python3 main.py"},
        {"command": ["true"], "timeout": MAX_RUN_SECONDS + 1},
        {"command": ["true"], "cwd": "/"},
    ],
)
def test_bad_arguments_come_back_to_the_model(ws: Workspace, args: dict[str, Any]) -> None:
    sandbox = Scripted(Outcome(exit_code=0, stdout="", stderr=""))
    toolbox, _ = box(ws, sandbox, Permission.SHELL)
    result = toolbox.call(ToolCall(tool="run_command", args=args))
    assert not result.ok and "bad arguments" in (result.error or "")
    assert sandbox.calls == []


@confined
def test_an_executor_runs_code_it_wrote(store: Store, clock: FakeClock, tmp_path: Path) -> None:
    def tool_calls(*calls: tuple[str, dict[str, Any]]) -> str:
        return json.dumps({"tool_calls": [{"tool": t, "args": a} for t, a in calls]})

    def accept(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        return Review(verdict=Verdict.ACCEPT)

    llm = FakeLLM().on(
        "Execute",
        tool_calls(
            ("write_file", {"path": "main.py", "content": "print(sum(range(10)))"}),
            ("run_command", {"command": ["python3", "main.py"]}),
        ),
        '{"summary": "45"}',
    )
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="sum", acceptance_criteria="ok")])
    task_id = store.create_task(spec)
    coordinator = Coordinator(
        store,
        llm_executor(llm),
        accept,
        workspaces=tmp_path / "ws",
        access=executor_access([Permission.SHELL]),
        sleep=clock.sleep,
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    results = json.loads(llm.calls[1].prompt.split("\n", 1)[1])
    assert results[1]["output"] == {"exit_code": 0, "stdout": "45\n", "stderr": ""}
    assert check_history(store.conn, task_id) == []


# ── desman run ──────────────────────────────────────────────────────────────


def test_run_warns_when_the_switch_is_on_but_there_is_no_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    routes = [
        {"match": "Execute", "steps": ['{"ok": 1}']},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    script = tmp_path / "fake.json"
    script.write_text(json.dumps({"routes": routes}))
    monkeypatch.setattr(cli_module, "Sandbox", missing)
    warning = "shell_execution is on, but there is no sandbox to run commands in"

    def desman_run() -> str:
        result = CliRunner().invoke(cli, ["run", "g", "--fake-llm", str(script)])
        assert result.exit_code == EXIT_DONE, result.output
        return result.output

    assert warning not in desman_run()  # the switch is off: nothing to warn about
    config = make_config()
    config.permissions.shell_execution = True
    ConfigManager(default_config_path()).save(config)
    output = desman_run()
    assert warning in output and "cannot be started" in output
