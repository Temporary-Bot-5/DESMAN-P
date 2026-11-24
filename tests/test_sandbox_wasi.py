"""WebAssembly setup and CLI behavior, plus real runs with DESMAN_WASI_TEST_DIR."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from desman import docker_cli, sandbox_cli, ui, wizard
from desman.cli import cli
from desman.commands import Context, Proposal, SandboxSetupArgs, find
from desman.config import Permissions, open_config
from desman.core import sandbox
from desman.core.policy import Permission, executor_access
from desman.core.sandbox import Limits, Sandbox, SandboxUnavailable, install_wasi
from desman.core.store import Store
from desman.core.tools import Toolbox, ToolCall, builtin_registry
from desman.core.workspace import Workspace


def _box(path: Path, *, platform: str = "darwin", limits: Limits | None = None) -> Sandbox:
    return Sandbox(limits, platform=platform, docker="", wsl="", wasi_dir=path)


def test_wasi_uses_the_database_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESMAN_DATA_DIR", str(tmp_path))
    assert Sandbox(platform="win32", docker="").wasi_dir.parent == tmp_path / "wasi"


def _zip(files: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    monkeypatch.setattr(sandbox, "_warm_wasi_cache", lambda _: None)
    archives = {
        "python.zip": _zip({"python.wasm": "wasm", "lib/os.py": "stdlib"}),
        "pytest.whl": _zip(
            {
                name + "/__init__.py": ""
                for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments")
            }
        ),
    }
    monkeypatch.setattr(
        sandbox,
        "_WASI_ARCHIVES",
        tuple(
            (url, hashlib.sha256(data).hexdigest(), folder)
            for (url, data), folder in zip(archives.items(), ("py", "site"), strict=True)
        ),
    )
    return archives


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_selection(
    platform: str, tmp_path: Path, downloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "wasi_runtime", lambda: True)
    root = tmp_path / "runtime"
    box = _box(root, platform=platform)
    reason = box.unavailable() or ""
    assert box.backend == "wasi" and not box.wasi_present
    assert "desman docker install" in reason and "desman sandbox setup" in reason
    assert install_wasi(root, download=downloads.__getitem__)
    assert not install_wasi(root, download=downloads.__getitem__)
    assert box.reprobe() is None and box.wasi_present

    def docker_run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if argv[1] == "info":
            return subprocess.CompletedProcess(argv, 0, "linux", "")
        return subprocess.CompletedProcess(argv, 1, "", "No such image")

    docker = Sandbox(
        platform=platform, docker="docker", docker_runner=docker_run, wsl="", wasi_dir=root
    )
    assert docker.unavailable() is None
    assert docker.backend == "wasi"
    assert Sandbox(platform="linux", docker="docker", docker_runner=docker_run).backend == "bwrap"

    monkeypatch.setattr(sandbox, "wasi_runtime", lambda: False)
    assert box.reprobe() is not None
    assert box.wasi_present


@pytest.mark.parametrize(
    "command", [["sh", "-c", "true"], ["python", "-m", "pip"], ["python"], ["python", "-c"]]
)
def test_unsupported_command(
    command: list[str], tmp_path: Path, downloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sandbox, "wasi_runtime", lambda: True)
    root = tmp_path / "runtime"
    install_wasi(root, download=downloads.__getitem__)
    outcome = _box(root).run(command, tmp_path)
    assert outcome.exit_code and not outcome.timed_out
    assert "pytest and unittest" in outcome.stderr
    assert "Packages with C code are not available" in outcome.stderr


def test_hash_mismatch_discards_staging(tmp_path: Path, downloads: dict[str, bytes]) -> None:
    downloads["pytest.whl"] = b"modified"
    with pytest.raises(SandboxUnavailable, match="SHA-256 mismatch"):
        install_wasi(tmp_path / "runtime", download=downloads.__getitem__)
    assert list(tmp_path.iterdir()) == []


def test_streaming_download_stops_at_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Archive(httpx.SyncByteStream):
        read = 0
        closed = False

        def __iter__(self) -> Iterator[bytes]:
            for _ in range(3):
                self.read += 1
                yield bytes(64 * 1024)

        def close(self) -> None:
            self.closed = True

    archive = Archive()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=archive))
    with httpx.Client(transport=transport) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        monkeypatch.setattr(sandbox, "_WASI_DOWNLOAD_LIMIT", 64 * 1024)
        with pytest.raises(SandboxUnavailable, match="exceeds the size limit"):
            sandbox._download_wasi("https://example.test/archive")
    assert archive.read == 2 and archive.closed


def test_isolated_runner_command_for_installed_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sysconfig, "get_path", lambda _: tmp_path)
    monkeypatch.setattr(sandbox, "__file__", str(tmp_path / "desman/core/sandbox.py"))
    spec = tmp_path / "spec.json"
    assert sandbox._wasi_runner_command(spec) == [
        sys.executable,
        "-I",
        "-m",
        "desman.core.wasi_runner",
        str(spec),
    ]


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", r"a\..\outside", "f:x"])
def test_refuses_unsafe_archive(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _zip({"lib/ok": "", name: ""})
    monkeypatch.setattr(
        sandbox, "_WASI_ARCHIVES", (("test.zip", hashlib.sha256(data).hexdigest(), "py"),)
    )
    with pytest.raises(SandboxUnavailable, match="Unsafe sandbox archive"):
        install_wasi(tmp_path / "runtime", download=lambda _: data)
    assert list(tmp_path.iterdir()) == []


def test_setup_consent_and_status(
    tmp_path: Path, downloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, store: Store
) -> None:
    root = tmp_path / "runtime"
    monkeypatch.setattr(sandbox, "Sandbox", lambda: _box(root))
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: _box(root))
    monkeypatch.setattr(sandbox, "wasi_runtime", lambda: True)
    monkeypatch.setattr(sandbox_cli, "wasi_runtime", lambda: True)
    monkeypatch.setattr(sandbox, "_download_wasi", downloads.__getitem__)
    runner = CliRunner()
    status = runner.invoke(cli, ["sandbox", "status", "--json"])
    assert status.exit_code == 0, status.output
    result = json.loads(status.output)
    assert result["backend"] == "wasi" and not result["wasi_present"] and not result["available"]
    assert "files are missing" in runner.invoke(cli, ["sandbox", "status"]).output
    decline = runner.invoke(cli, ["sandbox", "setup"])
    assert decline.exit_code == 0 and "24 MB download, 52 MB on disk" in decline.output
    assert not root.exists()

    questions: list[str] = []
    monkeypatch.setattr(ui, "checkbox", lambda *_args, **_kwargs: {"shell_execution"})

    def decline_wizard(question: str) -> bool:
        questions.append(question)
        return False

    monkeypatch.setattr(ui, "confirm", decline_wizard)
    assert wizard._permissions(Permissions()).shell_execution
    assert len(questions) == 1 and "WebAssembly" in questions[0]
    assert not root.exists()

    entry = find("sandbox setup")
    assert entry is not None
    proposal = entry.handler(
        Context(store, open_config(), lambda *_: None, lambda _: False), SandboxSetupArgs()
    )
    assert isinstance(proposal, Proposal) and "WebAssembly" in proposal.lines[0]
    assert not root.exists()

    status = runner.invoke(cli, ["sandbox", "setup", "--yes"])
    assert status.exit_code == 0 and "files are present" in status.output
    result = json.loads(runner.invoke(cli, ["sandbox", "status", "--json"]).output)
    assert result["backend"] == "wasi" and result["wasi_present"] and result["available"]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = os.environ.get("DESMAN_WASI_TEST_DIR")
    if not source:
        pytest.skip("set DESMAN_WASI_TEST_DIR to the unpacked CPython and pytest fixture")
    pytest.importorskip("wasmtime")
    root = tmp_path_factory.mktemp("wasm-runtime")
    for name in ("py", "site"):
        shutil.copytree(Path(source) / name, root / name)
    (root / "ok").touch()
    assert _box(root).wasi_present
    return root


def test_command_mapping_and_stdin(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime)
    cold = box.run(["python", "-c", "print(42)"], tmp_path, timeout=1)
    assert cold.exit_code == 0 and cold.stdout == "42\n", cold
    assert list(runtime.glob("wasmtime-*/python.cwasm"))
    (tmp_path / "test_script.py").write_text(
        "import os, sys\nprint(os.getcwd(), sys.argv[1], input())\n", encoding="utf-8"
    )
    for file in ("test_script.py", "/workspace/test_script.py"):
        outcome = box.run(["python3", file, "arg"], tmp_path, stdin="in\n")
        assert outcome.exit_code == 0, outcome
        assert outcome.stdout == "/workspace arg in\n"
    outcome = box.run(
        ["python", "-c", "from __future__ import annotations\nimport sys\nprint(sys.argv)", "arg"],
        tmp_path,
    )
    assert outcome.exit_code == 0 and outcome.stdout == "['-c', 'arg']\n"


def test_real_limits_and_isolation(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime, limits=Limits(stdout_chars=5, stderr_chars=7, memory_bytes=64 * 2**20))
    timeout = box.run(
        ["python", "-c", "print('loop', flush=True)\nwhile True: pass"], tmp_path, timeout=1.5
    )
    assert timeout.timed_out and timeout.exit_code is None and timeout.stdout == "loop\n"
    output = box.run(
        [
            "python",
            "-c",
            "import sys; print('start' * 100); print('ending' * 100, file=sys.stderr)",
        ],
        tmp_path,
    )
    assert output.exit_code == 0, output
    assert output.stdout == "tart\n" and output.stdout_cut
    assert output.stderr == "ending\n" and output.stderr_cut
    memory = _box(runtime, limits=Limits(memory_bytes=64 * 2**20)).run(
        ["python", "-c", "bytearray(128 * 2**20)"], tmp_path
    )
    assert memory.exit_code == 1 and "MemoryError" in memory.stderr
    for code in (
        "import socket; socket.create_connection(('127.0.0.1', 80))",
        "open('/lib/write-test', 'w')",
        "open('/site/write-test', 'w')",
        "open('/etc/passwd').read()",
    ):
        assert box.run(["python", "-c", code], tmp_path).exit_code == 1
    write = box.run(["python", "-c", "open('ok', 'w').write('yes')"], tmp_path)
    assert write.exit_code == 0
    assert (tmp_path / "ok").read_text(encoding="utf-8") == "yes"


def test_blocking_guest_times_out_and_child_is_reaped(
    runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.Popen
    children: list[subprocess.Popen[bytes]] = []

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        assert "PYTHONPATH" not in kwargs["env"]
        child = original(*args, **kwargs)
        children.append(child)
        return child

    box = _box(runtime)
    assert box.run(["python", "-c", "pass"], tmp_path).exit_code == 0  # warm the cache
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted"))
    monkeypatch.setattr(subprocess, "Popen", launch)
    start = time.monotonic()
    outcome = box.run(
        ["python", "-c", "import time; print('sleeping', flush=True); time.sleep(1000)"],
        tmp_path,
        timeout=1,
    )
    assert time.monotonic() - start < 5
    assert outcome.timed_out and outcome.exit_code is None and outcome.stdout == "sleeping\n"
    (child,) = children
    assert child.returncode is not None


@pytest.mark.parametrize("exit_code", [0, 7])
def test_verbose_wasi_run_keeps_both_tails_and_exit_code(
    runtime: Path, tmp_path: Path, exit_code: int
) -> None:
    box = _box(runtime, limits=Limits(stdout_chars=8, stderr_chars=8))
    outcome = box.run(
        [
            "python",
            "-c",
            "import sys, time\n"
            "print('x' * (128 * 1024), flush=True)\n"
            "print('y' * (128 * 1024), file=sys.stderr, flush=True)\n"
            "time.sleep(1)\n"
            "print('end')\n"
            "print('err', file=sys.stderr)\n"
            f"sys.exit({exit_code})\n",
        ],
        tmp_path,
        timeout=10,
    )
    assert outcome.exit_code == exit_code and not outcome.timed_out, outcome
    assert outcome.stdout == "xxx\nend\n" and outcome.stdout_cut
    assert outcome.stderr == "yyy\nerr\n" and outcome.stderr_cut


def test_real_pytest_and_tool(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime)
    work = Workspace(tmp_path, "wasi-test")
    command = ["python", "-m", "pytest", "-q", "test_example.py"]
    tools = Toolbox(work, builtin_registry(box), access=executor_access((Permission.SHELL,)))
    for code, label, expression in [(0, "passed", "True"), (1, "failed", "False")]:
        (work.root / "test_example.py").write_text(
            f"def test_example():\n    assert {expression}\n", encoding="utf-8"
        )
        result = box.run(command, work.root)
        assert result.exit_code == code and f"1 {label}" in result.stdout, result
        call = tools.call(ToolCall(tool="run_command", args={"command": command}))
        assert call.ok, call
        assert call.output["exit_code"] == code and f"1 {label}" in call.output["stdout"], call
    assert not (work.root / ".pytest_cache").exists()


def test_real_cache_rebuilt_on_deserialize_failure(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime)
    assert box.run(["python", "-c", "pass"], tmp_path).exit_code == 0
    (cache,) = runtime.glob("wasmtime-*/python.cwasm")
    cache.write_bytes(b"broken")
    assert box.run(["python", "-c", "pass"], tmp_path).exit_code == 0
    assert cache.stat().st_size > len(b"broken")


def test_setup_repairs_partial_install(
    tmp_path: Path, downloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    (root / "stale").write_text("partial", encoding="utf-8")

    def warm(stage: Path) -> None:
        assert not (stage / "ok").exists()
        assert (stage / "py/python.wasm").is_file()
        (stage / "compile-record").touch()

    monkeypatch.setattr(sandbox, "_warm_wasi_cache", warm)
    assert install_wasi(root, download=downloads.__getitem__)
    assert (root / "compile-record").exists()
    assert (root / "ok").is_file()
    assert not (root / "stale").exists()
    assert not list(tmp_path.glob(".setup-*"))
    assert not list(tmp_path.glob(".runtime.broken-*"))


def test_docker_reprobe_keeps_wasi_until_image_works(
    tmp_path: Path, downloads: dict[str, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    install_wasi(root, download=downloads.__getitem__)
    monkeypatch.setattr(sandbox, "wasi_runtime", lambda: True)
    answers = {"info": (1, "", "daemon is starting"), "image": (1, "", "No such image")}
    calls: list[str] = []
    now = 0

    def run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(argv[1])
        return subprocess.CompletedProcess(argv, *answers.get(argv[1], (0, "", "")))

    box = Sandbox(
        platform="darwin", docker="docker", docker_runner=run, wasi_dir=root, monotonic=lambda: now
    )
    assert box.unavailable() is None and box.backend == "wasi"
    answers["info"] = (0, "linux 2020-01-02T12:00:00Z", "")
    now = 59
    assert box.unavailable() is None and calls == ["info"]
    now = 60
    assert box.unavailable() is None and box.backend == "wasi"
    assert calls == ["info", "info", "image"]
    now = 119
    assert box.unavailable() is None and calls.count("info") == 2
    answers["image"] = (0, "", "")
    now = 120
    assert box.unavailable() is None and box.backend == "docker"
    assert calls.count("info") == 3


def test_wizard_keeps_wasi_offer_in_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def stopped(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", "starting")

    def box() -> Sandbox:
        return Sandbox(
            platform="win32", docker="docker", docker_runner=stopped, wsl="", wasi_dir=tmp_path
        )

    monkeypatch.setattr(sandbox, "Sandbox", box)
    monkeypatch.setattr(sandbox_cli, "wasi_runtime", lambda: True)
    monkeypatch.setattr(wizard, "sys", SimpleNamespace(platform="win32"))
    state = docker_cli.Status(
        installed=True,
        available=False,
        problem="starting",
        version=None,
        bwrap=False,
    )
    monkeypatch.setattr(docker_cli, "docker_status", lambda **_: state)
    monkeypatch.setattr(
        docker_cli,
        "current_plan",
        lambda _: docker_cli.plan("win32", lambda _: None, {}, "", state),
    )
    monkeypatch.setattr(docker_cli, "apply_plan", lambda _: {**state, "available": True})
    monkeypatch.setattr(wizard, "_offer_searxng", lambda _: None)
    monkeypatch.setattr(ui, "checkbox", lambda *_args, **_kwargs: {"shell_execution"})
    questions: list[str] = []

    def confirm(question: str, **_: object) -> bool:
        questions.append(question)
        return "Install Docker" in question

    monkeypatch.setattr(ui, "confirm", confirm)
    wizard._docker(wizard._Session(open_config().load()))
    assert len(questions) == 1 and "Install Docker" in questions[0]
    assert "desman sandbox setup" in capsys.readouterr().out
    assert wizard._permissions(Permissions()).shell_execution
    assert len(questions) == 2 and "WebAssembly" in questions[1]
    message = capsys.readouterr().out
    assert "daemon did not answer" in message and "desman sandbox setup" in message


@pytest.mark.parametrize(
    "flags",
    [
        ["-u"],
        ["-B"],
        ["-E"],
        ["-s"],
        ["-I"],
        ["-X", "dev"],
        ["-Xdev"],
        ["-W", "ignore"],
        ["-u", "-E", "-X", "utf8", "-W", "default"],
    ],
)
def test_wasi_ignores_interpreter_flags(flags: list[str]) -> None:
    assert sandbox._wasi_argv(["python", *flags, "-c", "pass", "-u"]) == sandbox._wasi_argv(
        ["python", "-c", "pass", "-u"]
    )


def test_real_unittest(runtime: Path, tmp_path: Path) -> None:
    (tmp_path / "test_sample.py").write_text(
        "import unittest\nclass Sample(unittest.TestCase):\n    def test_it(self): pass\n",
        encoding="utf-8",
    )
    result = _box(runtime).run(["python", "-B", "-m", "unittest", "-v", "test_sample"], tmp_path)
    assert result.exit_code == 0 and "Ran 1 test" in result.stderr, result


def test_completed_large_output_preserves_exit_code(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime, limits=Limits(file_bytes=1024, stdout_chars=64))
    outcome = box.run(
        ["python", "-c", "import sys; sys.stdout.write('x' * 2048); sys.exit(7)"], tmp_path
    )
    assert outcome.exit_code == 7 and outcome.stdout_cut, outcome
    assert outcome.stdout == "x" * 64
    assert "output limit exceeded" not in outcome.stderr


@pytest.mark.parametrize("files", [1, 9])
def test_wasi_workspace_watchdog(runtime: Path, tmp_path: Path, files: int) -> None:
    box = _box(runtime, limits=Limits(file_bytes=64 * 1024))
    (tmp_path / "existing").write_bytes(bytes(128 * 1024))
    outcome = box.run(
        [
            "python",
            "-c",
            "import time\n"
            f"for i in range({files}):\n"
            f"    with open(str(i), 'wb') as f: f.write(bytes({128 if files == 1 else 32} * 1024))\n"
            "time.sleep(1000)",
        ],
        tmp_path,
        timeout=6,
    )
    assert outcome.exit_code == 1 and not outcome.timed_out, outcome
    assert "file size limit exceeded" in outcome.stderr


@pytest.mark.parametrize(("returncode", "expected"), [(23, 1), (-9, 137)])
def test_runner_crash_is_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, expected: int
) -> None:
    code = f"raise SystemExit({returncode})"
    if returncode < 0:
        if sys.platform == "win32":
            pytest.skip("negative subprocess exit status is POSIX-specific")
        code = f"import os; os.kill(os.getpid(), {-returncode})"
    monkeypatch.setattr(
        sandbox, "_wasi_runner_command", lambda _: [sys.executable, "-I", "-c", code]
    )
    outcome = sandbox._execute_wasi(["python", "-c", "pass"], tmp_path, "", Limits(), tmp_path)
    assert outcome.exit_code == expected and not outcome.timed_out
    assert "the WebAssembly runner stopped" in outcome.stderr


def test_setup_warms_pinned_runtime(
    runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use the same pinned Python bytes as setup, with just package stubs for this smoke test.
    python = io.BytesIO()
    with zipfile.ZipFile(python, "w") as archive:
        for path in (runtime / "py").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(runtime / "py"))
    site = _zip(
        {f"{path.name}/__init__.py": "" for path in (runtime / "site").iterdir() if path.is_dir()}
    )
    artifacts = {"py": python.getvalue(), "site": site}
    monkeypatch.setattr(
        sandbox,
        "_WASI_ARCHIVES",
        tuple((name, hashlib.sha256(data).hexdigest(), name) for name, data in artifacts.items()),
    )
    root = tmp_path / "runtime"
    assert install_wasi(root, download=artifacts.__getitem__)
    assert list(root.glob("wasmtime-*/python.cwasm"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _box(root).run(["python", "-c", "print(42)"], workspace, timeout=1)
    assert result.exit_code == 0 and result.stdout == "42\n", result
