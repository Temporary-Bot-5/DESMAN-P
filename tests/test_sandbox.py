"""Sandbox for generated code (ROADMAP 3.3): commands run only under bwrap, confined.

The tests marked `confined` start real sandboxes (real processes and, for the timeout, real
time); they are skipped where bubblewrap cannot run. CI installs it and sets
DESMAN_REQUIRE_SANDBOX, so there they never skip.
"""

from __future__ import annotations

import json
import os
import platform
import sys
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
from desman.core.sandbox import Limits, Outcome, Sandbox, SandboxUnavailable, _Tail
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.core.tools import MAX_RUN_SECONDS, Toolbox, ToolCall, builtin_registry
from desman.core.workspace import PathEscapeError, Workspace
from desman.llm.fake import FakeLLM
from helpers import make_config

PROBLEM = Sandbox().unavailable() if sys.platform.startswith("linux") else "Linux only"
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
def test_generated_tests_run_with_pytest(ws: Workspace) -> None:
    ws.write("test_x.py", "def test_one():\n    assert 1 + 1 == 2\n")
    outcome = run(ws, "python3", "-m", "pytest", "-q", timeout=60)
    assert outcome.exit_code == 0, outcome.stdout + outcome.stderr
    assert "1 passed" in outcome.stdout
    assert not (ws.root / ".pytest_cache").exists()
    assert not (ws.root / "__pycache__").exists()


@confined
def test_pytest_header_reads_plugin_metadata(ws: Workspace) -> None:
    ws.write("test_x.py", "def test_one():\n    assert True\n")
    outcome = run(ws, "python3", "-m", "pytest", "test_x.py", timeout=60)
    assert outcome.exit_code == 0, outcome.stdout + outcome.stderr
    assert "1 passed" in outcome.stdout


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
    assert set(env) <= {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "PWD",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTEST_ADDOPTS",
    }
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTEST_ADDOPTS"] == "-p no:cacheprovider"
    assert env.get("PYTHONPATH", "/opt/desman/python") == "/opt/desman/python"


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
def test_system_calls_a_program_never_needs_are_refused(ws: Workspace) -> None:
    code = (
        "import ctypes, errno, os\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "for nr in (101, 321, 323, 298, 425, 250, 165, 272):\n"  # x86-64 numbers
        "    ctypes.set_errno(0)\n"
        "    print(libc.syscall(nr, 0, 0, 0, 0), errno.errorcode.get(ctypes.get_errno()))\n"
        "print(os.getpid() > 0)\n"
    )
    if platform.machine() != "x86_64":
        pytest.skip("the numbers in this test are x86-64's")
    outcome = py(ws, code)
    assert outcome.exit_code == 0, outcome.stderr
    assert outcome.stdout.splitlines() == ["-1 EPERM"] * 8 + ["True"]


@confined
def test_a_run_that_fills_the_workspace_is_stopped(ws: Workspace) -> None:
    ws.write("kept.txt", "x")
    endless = "i=0; while true; do i=$((i+1)); echo x > f$i; done"
    outcome = Sandbox(Limits(workspace_files=200)).run(["sh", "-c", endless], ws.root, timeout=30)
    assert not outcome.timed_out and outcome.exit_code == 128 + 9
    assert outcome.stopped == "the workspace grew past 200 files and directories or 512.0 MiB"
    assert outcome.workspace_full.startswith("the workspace holds at least 201 files")
    assert ws.read_text("kept.txt") == "x"  # stopped, nothing removed


@confined
def test_a_run_that_closes_its_output_is_still_watched(ws: Workspace) -> None:
    endless = "exec >&- 2>&-; i=0; while true; do i=$((i+1)); echo x > f$i; done"
    outcome = Sandbox(Limits(workspace_files=200)).run(["sh", "-c", endless], ws.root, timeout=30)
    assert not outcome.timed_out and outcome.exit_code == 128 + 9
    assert outcome.stopped and "at least 201 files" in outcome.workspace_full


@confined
def test_a_folder_made_unreadable_hides_nothing_from_the_quota(ws: Workspace) -> None:
    hide = "mkdir d && echo x > d/f && chmod 0 d && sleep 1.5 && echo done"
    outcome = Sandbox(Limits()).run(["sh", "-c", hide], ws.root, timeout=30)
    assert outcome.exit_code == 0 and not outcome.stopped  # an unfinished count stops nothing
    assert outcome.workspace_full.startswith("part of the workspace cannot be read")
    # The next run is still watched, by the disk's free space.
    write = "head -c 3000000 /dev/zero > d2; sleep 3; echo done"
    stopped = Sandbox(Limits(workspace_bytes=10**6)).run(["sh", "-c", write], ws.root, timeout=30)
    assert stopped.exit_code == 128 + 9
    assert stopped.stopped.startswith("the disk holding the workspace lost")
    cleaned = Sandbox(Limits()).run(["sh", "-c", "chmod 700 d && rm -r d d2"], ws.root)
    assert cleaned.exit_code == 0 and cleaned.workspace_full == ""


@confined
def test_a_workspace_already_over_its_quota_can_still_be_cleaned(ws: Workspace) -> None:
    for i in range(5):
        ws.write(f"f{i}", "x" * 100)
    limits = Limits(workspace_files=3, workspace_bytes=10**6)
    full = Sandbox(limits).run(["sh", "-c", "sleep 1.5; echo ok"], ws.root)
    assert full.exit_code == 0 and full.stdout == "ok\n"  # not grown: not stopped
    assert "holds at least 4 files" in full.workspace_full
    cleaned = Sandbox(limits).run(["sh", "-c", "rm f1 f2 f3"], ws.root)
    assert cleaned.exit_code == 0 and cleaned.workspace_full == ""


def test_the_quota_is_measured_after_every_backend(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fill(*args: Any) -> Outcome:
        ws.write("big", "x" * 2048)
        return Outcome(exit_code=0, stdout="", stderr="")

    box = Sandbox(Limits(workspace_bytes=1024), docker="docker", platform="win32")
    monkeypatch.setattr(box, "unavailable", lambda: None)
    monkeypatch.setattr(box, "_run", fill)
    outcome = box.run(["true"], ws.root)
    assert outcome.workspace_full == (
        "the workspace holds at least 1 files and directories and at least 2.0 KiB; "
        "the quota is 10000 files and directories and 1.0 KiB"
    )


def test_the_watch_stops_a_run_the_disk_shows_but_a_count_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.core import sandbox as module

    free = [10**9]
    monkeypatch.setattr("shutil.disk_usage", lambda path: shutil_usage(free[-1]))
    check = module._watch(tmp_path, Limits(workspace_files=10, workspace_bytes=4096))
    assert check() is None
    free.append(10**9 - 8192)  # twice the quota
    assert check() is None
    free.append(10**9 - 8193)
    assert check() == (
        "the disk holding the workspace lost 8.0 KiB of free space during the run, "
        "more than 8.0 KiB (writes of other programs count too)"
    )
    free.append(10**9)
    for i in range(11):
        (tmp_path / f"f{i}").write_text("x")
    assert check() == "the workspace grew past 10 files and directories or 4.0 KiB"


def shutil_usage(free: int) -> Any:
    import shutil

    return shutil._ntuple_diskusage(10**12, 10**12 - free, free)


@pytest.mark.skipif(sys.platform == "win32", reason="counted by path on Windows")
def test_a_folder_swapped_for_a_link_leads_the_count_nowhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.core.sandbox import Usage, workspace_usage

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big").write_bytes(b"x" * 1000)
    root = tmp_path / "ws"
    (root / "d").mkdir(parents=True)
    real_open = os.open

    def swap(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "d" and (root / "d").is_dir():  # swapped between listing and opening
            (root / "d").rmdir()
            (root / "d").symlink_to(outside, target_is_directory=True)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("os.open", swap)
    usage = workspace_usage(root, Limits())
    assert usage == Usage(1, 0, complete=False)  # refused to follow it, never counted outside


def test_a_tree_deeper_than_the_count_goes_is_unfinished(tmp_path: Path) -> None:
    from desman.core.sandbox import Usage, workspace_usage

    deep = tmp_path / "ws"
    for _ in range(100):
        deep = deep / "d"
    deep.mkdir(parents=True)
    assert not workspace_usage(tmp_path / "ws", Limits()).complete
    assert workspace_usage(tmp_path / "missing", Limits()) == Usage(0, 0)
    assert not module_usage_by_path(tmp_path / "ws").complete


def module_usage_by_path(root: Path) -> Any:
    from desman.core import sandbox as module

    return module._usage_by_path(root, Limits())


def test_workspace_usage_counts_without_following_links(tmp_path: Path) -> None:
    from desman.core.sandbox import workspace_usage

    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"x" * 10)
    (tmp_path / "b").write_bytes(b"y" * 5)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "huge").write_bytes(b"z" * 1000)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    usage = workspace_usage(tmp_path, Limits())
    assert (usage.entries, usage.size) == (4, 15)
    assert workspace_usage(tmp_path, Limits(workspace_files=2)).over(Limits(workspace_files=2))


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
def test_output_is_cut_keeping_the_tails(ws: Workspace) -> None:
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
        outcome.stdout_cut
        and outcome.stdout.endswith("99998\n99999\n")
        and len(outcome.stdout) == 20
    )
    assert outcome.stderr_cut and outcome.stderr.endswith("99998\n99999\n")


# ── No sandbox: nothing runs ────────────────────────────────────────────────


def missing() -> Sandbox:
    return Sandbox(bwrap="/nonexistent/bwrap", docker="")


def test_without_a_working_bwrap_nothing_runs(ws: Workspace, tmp_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        with pytest.raises(SandboxUnavailable, match="Docker is not installed"):
            missing().run(["touch", "x"], ws.root)
        assert ws.files() == []
        return
    with pytest.raises(SandboxUnavailable, match="cannot be started"):
        missing().run(["touch", "x"], ws.root)
    refusing = tmp_path / "bwrap"
    refusing.write_text(
        "#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\nexit 1\n",
        encoding="utf-8",
    )
    refusing.chmod(0o755)
    sandbox = Sandbox(bwrap=str(refusing))
    assert "No permissions to create new namespace" in (sandbox.unavailable() or "")
    with pytest.raises(SandboxUnavailable):
        sandbox.run(["touch", "x"], ws.root)
    assert ws.files() == []


def test_the_longest_timeout_the_tool_allows_is_not_cut_short() -> None:
    assert Sandbox().limits.timeout == MAX_RUN_SECONDS
    assert Sandbox().limits.cpu_seconds >= MAX_RUN_SECONDS


@confined
def test_a_timeout_asked_for_is_never_longer_than_the_limit(ws: Workspace) -> None:
    started = time.monotonic()
    outcome = Sandbox(Limits(timeout=1.0)).run(["sleep", "30"], ws.root, timeout=60)
    assert outcome.timed_out and time.monotonic() - started < 10


def test_without_bwrap_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("desman.core.sandbox.shutil.which", lambda name: None)
    sandbox = Sandbox(docker="")
    reason = sandbox.unavailable() or ""
    if sys.platform.startswith("linux"):
        assert reason == "bubblewrap (bwrap) is not installed"
    else:
        assert "Docker is not installed" in reason


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
    assert names(toolbox) == ["read_file", "write_file", "edit_file", "list_files", "read_node"]
    result = toolbox.call(ToolCall(tool="run_command", args={"command": ["true"]}))
    assert not result.ok and "cannot run here" in (result.error or "")
    [(action, details)] = entries
    assert (
        action == "tool_refused"
        and ("cannot be started" if sys.platform.startswith("linux") else "Docker is not installed")
        in details["reason"]
    )


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


def test_the_model_is_told_when_the_workspace_is_over_its_quota(ws: Workspace) -> None:
    sandbox = Scripted(Outcome(exit_code=0, stdout="", stderr="", workspace_full="it holds 9"))
    toolbox, _ = box(ws, sandbox, Permission.SHELL)
    result = toolbox.call(ToolCall(tool="run_command", args={"command": ["true"]}))
    assert result.ok and isinstance(result.output, dict)
    assert result.output["workspace_full"].startswith("it holds 9: delete files")
    stopped = Scripted(Outcome(exit_code=137, stdout="", stderr="", stopped="it grew"))
    toolbox, _ = box(ws, stopped, Permission.SHELL)
    result = toolbox.call(ToolCall(tool="run_command", args={"command": ["true"]}))
    assert result.ok and isinstance(result.output, dict)
    assert result.output["stopped"] == "it grew" and "workspace_full" not in result.output


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
        {"match": "Execute", "steps": ['{"text": "done"}']},
        {"match": "Review", "steps": ['{"verdict": "accept"}'], "repeat_last": True},
    ]
    script = tmp_path / "fake.json"
    script.write_text(json.dumps({"routes": routes}), encoding="utf-8")
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
    assert (
        warning in output
        and ("cannot be started" if sys.platform.startswith("linux") else "Docker is not installed")
        in output
    )


def test_output_tail_skips_continuation_bytes_at_the_cut() -> None:
    tail = _Tail(4)
    tail.add(b"x" + bytes([0x80]) * 18)
    tail.add("éé".encode())
    assert tail.cut
    assert tail.text() == "éé"
