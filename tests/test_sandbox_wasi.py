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
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from desman import sandbox_cli, ui, wizard
from desman.cli import cli
from desman.commands import Context, NoArgs, Proposal, find
from desman.config import Permissions, open_config
from desman.core import sandbox
from desman.core.policy import Permission, executor_access
from desman.core.sandbox import Limits, Sandbox, SandboxUnavailable, install_wasi
from desman.core.store import Store
from desman.core.tools import Toolbox, ToolCall, builtin_registry
from desman.core.workspace import Workspace


def _box(path: Path, *, platform: str = "darwin", limits: Limits | None = None) -> Sandbox:
    return Sandbox(limits, platform=platform, docker="", wasi_dir=path)


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

    docker = Sandbox(platform=platform, docker="docker", docker_runner=docker_run, wasi_dir=root)
    assert docker.backend == "docker"
    assert "image is missing" in (docker.unavailable() or "")
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
    assert "Only python and python -m pytest" in outcome.stderr
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
        Context(store, open_config(), lambda *_: None, lambda _: False), NoArgs()
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
    assert output.stdout == "start" and output.stdout_cut
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


def test_output_watchdog_kills_writer(runtime: Path, tmp_path: Path) -> None:
    box = _box(runtime, limits=Limits(file_bytes=64 * 1024))
    assert box.run(["python", "-c", "pass"], tmp_path).exit_code == 0
    outcome = box.run(
        [
            "python",
            "-c",
            "import sys, time; print('x' * 128 * 1024, flush=True); time.sleep(1000)",
        ],
        tmp_path,
        timeout=10,
    )
    assert outcome.exit_code == 1 and not outcome.timed_out and outcome.stdout_cut
    assert outcome.stdout == "x" * box.limits.stdout_chars
    assert "output limit exceeded" in outcome.stderr


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
