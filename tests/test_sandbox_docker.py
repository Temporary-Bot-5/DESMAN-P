"""Docker sandbox contract tests and opt-in real daemon checks."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import IO

import pytest
from click.testing import CliRunner

from desman import runs, sandbox_cli, ui, wizard
from desman.cli import cli
from desman.commands import Context, Kind, NoArgs, Proposal, find
from desman.commands import Outcome as CommandOutcome
from desman.config import Config, ConfigManager, Permissions, default_config_path
from desman.core import sandbox
from desman.core.sandbox import (
    DOCKER_IMAGE,
    DOCKER_LABEL,
    Limits,
    Outcome,
    Sandbox,
    SandboxUnavailable,
)
from desman.core.store import Store
from desman.llm.fake import FakeLLM


class Docker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []
        self.answers: dict[str, tuple[int, str, str]] = {
            "info": (0, "linux\n", ""),
            "{{.SystemTime}}": (0, "2020-01-02T14:00:00.123456789+02:00\n", ""),
            "manifest": (1, "", "registry unavailable"),
        }

    def run(self, argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "docker-test"
        assert timeout > 0
        self.calls.append(argv[1:])
        self.timeouts.append(timeout)
        if argv[1] == "pull":
            self.answers["image"] = (0, "", "")
        key = argv[-1] if argv[-1] == "{{.SystemTime}}" else argv[1]
        return subprocess.CompletedProcess(argv, *self.answers.get(key, (0, "", "")))

    def sandbox(self, **limits: int) -> Sandbox:
        return Sandbox(
            Limits(**limits), platform="win32", docker="docker-test", docker_runner=self.run
        )


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_docker_argv(platform: str, tmp_path: Path) -> None:
    limits = Limits(
        cpu_seconds=7,
        processes=11,
        memory_bytes=512_000_000,
        file_bytes=42,
        open_files=100,
        tmp_bytes=65_536,
    )
    box = Sandbox(limits, platform=platform, docker="docker-test")
    work = tmp_path / "workspace, with spaces"
    argv = box._docker_argv(["python3", "-c", "print('a; b')"], work, "desman-sbx-abcd")

    def values(flag: str) -> list[str]:
        return [argv[i + 1] for i, arg in enumerate(argv) if arg == flag]

    assert argv[:2] == ["docker-test", "run"]
    assert {"--init", "--read-only", "-i"} <= set(argv)
    assert "--rm" not in argv
    assert values("--pull") == ["never"]
    assert values("--network") == ["none"]
    assert values("--cap-drop") == ["ALL"]
    assert values("--security-opt") == ["no-new-privileges"]
    assert values("--name") == ["desman-sbx-abcd"]
    assert values("--label") == [DOCKER_LABEL]
    uid = "65534:65534"
    if sys.platform != "win32" and platform == "darwin":
        uid = f"{os.getuid()}:{os.getgid()}"
    assert values("--user") == [uid]
    assert values("--pids-limit") == ["11"]
    assert values("--memory") == values("--memory-swap") == ["512000000"]
    assert values("--cpus") == ["1"]
    assert values("--ulimit") == ["cpu=7:8", "fsize=42", "nofile=100", "core=0"]
    assert values("--tmpfs") == ["/tmp:size=65536"]
    assert values("-w") == ["/workspace"]
    assert set(values("-e")) == {
        f"PATH={sandbox.SANDBOX_PATH}",
        "HOME=/workspace",
        "TMPDIR=/tmp",
        "LANG=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTEST_ADDOPTS=-p no:cacheprovider",
        "PYTHONPATH=/opt/desman/python",
    }
    workspace, *mounts = list(csv.reader(values("--mount")))
    assert workspace == ["type=bind", f"source={work}", "target=/workspace"]
    assert all(fields[-1] == "readonly" for fields in mounts)
    targets = {fields[2] for fields in mounts}
    for package in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments"):
        assert f"target=/opt/desman/python/{package}" in targets
    assert any(
        target.startswith("target=/opt/desman/python/pytest-") and target.endswith(".dist-info")
        for target in targets
    )
    assert values("--entrypoint") == ["/usr/local/bin/python3"]
    assert argv[argv.index(DOCKER_IMAGE) + 1 :] == [
        "-I",
        "-B",
        "-c",
        sandbox._DOCKER_EXEC,
        "python3",
        "-c",
        "print('a; b')",
    ]


def test_backend_selection_and_docker_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert Sandbox(platform="linux", docker="docker-test").backend == "bwrap"
    assert Sandbox(platform="darwin").unavailable() == "Docker is not installed or is not on PATH."


@pytest.mark.parametrize(
    ("command", "answer", "reason"),
    [
        ("info", (1, "", "cannot connect"), "Docker daemon did not answer: cannot connect"),
        ("info", (0, "windows\n", ""), "switch Docker Desktop to Linux containers"),
        ("image", (1, "", "Error: No such image"), "run `desman sandbox setup`"),
        ("image", (1, "", "daemon went away"), "daemon went away"),
        ("run", (125, "", "bind failed"), "Docker cannot create a sandbox here: bind failed"),
    ],
)
def test_unavailable_reason(command: str, answer: tuple[int, str, str], reason: str) -> None:
    docker = Docker()
    docker.answers[command] = answer
    box = docker.sandbox()
    assert reason.lower() in (box.unavailable() or "").lower()
    calls = docker.calls[:]
    with pytest.raises(SandboxUnavailable):
        box.run(["true"], Path.cwd())
    assert docker.calls == calls  # checked once


def test_unresponsive_docker_daemon() -> None:
    def timeout(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, timeout)

    box = Sandbox(platform="darwin", docker="docker-test", docker_runner=timeout)
    assert (box.unavailable() or "").startswith("Docker daemon did not answer")


@pytest.mark.parametrize("timeout", [5, 120])
def test_sweep_and_empty_probe(timeout: int) -> None:
    docker = Docker()
    now = dt.datetime(2020, 1, 2, 12, tzinfo=dt.UTC)
    age = max(Limits().timeout, timeout) + 600
    old = (now - dt.timedelta(seconds=age + 1)).strftime("%Y-%m-%d %H:%M:%S %z UTC")
    young = (now - dt.timedelta(seconds=age - 1)).strftime("%Y-%m-%d %H:%M:%S %z UTC")
    docker.answers["ps"] = (
        0,
        f"111111111111\t{old}\n222222222222\t{young}\n--bad\t{old}\n333333333333\tunknown\n",
        "",
    )
    assert docker.sandbox(timeout=timeout).unavailable() is None
    _, image, clock, listing, remove, run, cleanup = docker.calls
    assert image == ["image", "inspect", DOCKER_IMAGE]
    assert clock == ["info", "--format", "{{.SystemTime}}"]
    assert listing[:5] == ["ps", "-a", "--filter", "label=" + DOCKER_LABEL, "--format"]
    assert remove == ["rm", "-f", "111111111111"]
    name = run[run.index("--name") + 1]
    assert re.fullmatch(r"desman-sbx-[a-f0-9]{32}", name)
    assert DOCKER_IMAGE in run and run[-1] == "true"
    assert docker.timeouts[-2] == 60.0
    assert cleanup == ["rm", "-f", name]


def test_sweep_ignores_host_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    class HostClock(dt.datetime):
        @classmethod
        def now(cls, tz: dt.tzinfo | None = None) -> HostClock:
            return cls(2099, 1, 2, 12, tzinfo=tz)

    monkeypatch.setattr(dt, "datetime", HostClock)
    docker = Docker()
    docker.answers["ps"] = (0, "111111111111\t2020-01-02 11:59:30 +0000 UTC\n", "")
    assert docker.sandbox().unavailable() is None
    assert ["rm", "-f", "111111111111"] not in docker.calls


@pytest.mark.parametrize("failing_command", ["info", "run"])
def test_probe_retries_transient_failure(failing_command: str) -> None:
    docker = Docker()
    now = 0.0
    failing = True

    def run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        if failing and argv[1] == failing_command:
            if failing_command == "run":
                assert timeout == 60.0
                raise subprocess.TimeoutExpired(argv, timeout)
            return subprocess.CompletedProcess(argv, 1, "", "cannot connect")
        return docker.run(argv, timeout)

    box = Sandbox(platform="darwin", docker="docker-test", docker_runner=run, monotonic=lambda: now)
    reason = box.unavailable()
    assert reason is not None
    failing = False
    now = 59
    assert box.unavailable() == reason
    now = 60
    assert box.unavailable() is None
    checked = docker.calls[:]
    now = 180
    assert box.unavailable() is None
    assert docker.calls == checked


@pytest.mark.parametrize(
    ("client_code", "message", "state", "exit_code"),
    [
        (125, "daemon is unavailable", (1, "", "no such container"), None),
        (1, "bind failed", (0, "0001-01-01T00:00:00Z 0", ""), None),
        (125, "program error", (0, "2020-01-02T12:00:00Z 125", ""), 125),
        (1, "docker: x", (0, "2020-01-02T12:00:00Z 1", ""), 1),
        (125, "client detached", (0, "2020-01-02T12:00:00Z 137", ""), 137),
    ],
)
def test_docker_uses_container_state(
    client_code: int,
    message: str,
    state: tuple[int, str, str],
    exit_code: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    popen = subprocess.Popen
    docker = Docker()
    docker.answers["inspect"] = state

    def fake_client(
        argv: list[str],
        *,
        stdin: IO[bytes],
        stdout: int,
        stderr: int,
    ) -> subprocess.Popen[bytes]:
        assert argv[0] == "docker-test"
        script = (  # bytes: a Linux container never writes \r\n, whatever the host
            f"import sys; sys.stderr.buffer.write({(message + chr(10)).encode()!r}); "
            f"sys.exit({client_code})"
        )
        return popen([sys.executable, "-c", script], stdin=stdin, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "Popen", fake_client)
    box = docker.sandbox()
    if exit_code is None:
        with pytest.raises(SandboxUnavailable, match=message):
            box.run(["nothing"], tmp_path)
    else:
        assert box.run(["nothing"], tmp_path) == Outcome(exit_code, "", message + "\n")
    inspect, cleanup = docker.calls[-2:]
    assert inspect[:3] == ["inspect", "--format", "{{.State.StartedAt}} {{.State.ExitCode}}"]
    assert cleanup == ["rm", "-f", inspect[-1]]


@pytest.mark.parametrize("interrupted", [False, True])
def test_docker_stops_container_before_client(
    interrupted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = Docker()
    events: list[str] = []

    class Client:
        def __init__(self, argv: list[str], *, stdin: IO[bytes], stdout: int, stderr: int) -> None:
            self.stdout = io.BytesIO(b"hello world")
            self.stderr = io.BytesIO(b"warning")
            assert stdin.read() == b"input\n"
            assert stdout == stderr == subprocess.PIPE
            self.name = argv[argv.index("--name") + 1]

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                assert timeout == 0.25
                if interrupted:
                    raise KeyboardInterrupt
                raise subprocess.TimeoutExpired(self.name, timeout)
            return 137

        def kill(self) -> None:
            assert docker.calls[-1] == ["kill", self.name]
            events.append(self.name)

    monkeypatch.setattr(subprocess, "Popen", Client)
    box = docker.sandbox(stdout_chars=5, stderr_chars=3)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            box.run(["nothing"], tmp_path, stdin="input\n", timeout=0.25)
    else:
        assert box.run(["nothing"], tmp_path, stdin="input\n", timeout=0.25) == Outcome(
            None,
            "hello",
            "ing",
            timed_out=True,
            stdout_cut=True,
            stderr_cut=True,
        )
    (name,) = events
    assert docker.calls[-1] == ["rm", "-f", name]


def test_cli_status_setup_and_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    docker = Docker()
    monkeypatch.setattr(sandbox_cli, "Sandbox", docker.sandbox)
    status = CliRunner().invoke(cli, ["sandbox", "status", "--json"])
    assert status.exit_code == 0, status.output
    assert json.loads(status.output) == {
        "backend": "docker",
        "available": True,
        "reason": None,
        "image": DOCKER_IMAGE,
        "image_present": True,
    }
    docker.calls.clear()
    installed = CliRunner().invoke(cli, ["sandbox", "setup", "--yes"])
    assert installed.exit_code == 0 and "Image present" in installed.output
    assert [call[0] for call in docker.calls] == ["info", "image"]

    docker.answers["image"] = (1, "", "No such image")
    decline = CliRunner().invoke(cli, ["sandbox", "setup"])
    assert decline.exit_code == 0, decline.output
    assert "about 45 MB" in decline.output
    assert "Nothing changed. Confirm with: desman sandbox setup --yes" in decline.output
    assert not any(call[0] == "pull" for call in docker.calls)
    manifest = {"OCIManifest": {"layers": [{"size": 36_000_000}, {"size": 3_000_000}]}}
    docker.answers["manifest"] = (0, json.dumps([manifest]), "")
    confirm = CliRunner().invoke(cli, ["sandbox", "setup", "--yes"])
    assert confirm.exit_code == 0, confirm.output
    assert "about 39 MB" in confirm.output
    assert ["pull", DOCKER_IMAGE] in docker.calls
    assert DOCKER_IMAGE in docker.calls[-2] and docker.calls[-2][-1] == "true"
    status_entry = find("sandbox status")
    setup_entry = find("sandbox setup")
    assert status_entry is not None and status_entry.kind is Kind.READ
    assert setup_entry is not None and setup_entry.kind is Kind.CHANGE and setup_entry.typed_confirm


def test_cli_linux_setup_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sandbox_cli, "Sandbox", lambda: Sandbox(platform="linux", bwrap="/nonexistent/bwrap")
    )
    monkeypatch.setattr(Sandbox, "_probe", lambda _: "unavailable in this test")
    runner = CliRunner()
    setup = runner.invoke(cli, ["sandbox", "setup"])
    assert setup.exit_code == 0 and "No image is needed" in setup.output
    status = runner.invoke(cli, ["sandbox", "status", "--json"])
    assert status.exit_code == 0, status.output
    data = json.loads(status.output)
    assert data["backend"] == "bwrap" and not data["available"]
    assert data["image"] is None and data["image_present"] is None


def test_catalog_waits_for_confirmation(
    store: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    docker = Docker()
    monkeypatch.setattr(sandbox_cli, "Sandbox", docker.sandbox)
    monkeypatch.setattr(sandbox, "Sandbox", docker.sandbox)
    context = Context(store, ConfigManager(default_config_path()), lambda *_: None, lambda _: False)
    status = find("sandbox status")
    assert status is not None
    result = status.handler(context, NoArgs())
    assert isinstance(result, CommandOutcome) and result.data["image_present"] is True
    assert capsys.readouterr().out == ""
    docker.answers["image"] = (1, "", "No such image")
    entry = find("sandbox setup")
    assert entry is not None
    proposal = entry.handler(context, NoArgs())
    assert isinstance(proposal, Proposal) and "45 MB" in proposal.lines[0]
    assert not any(call[0] == "pull" for call in docker.calls)
    assert proposal.apply().data == {"installed": True}
    assert ["pull", DOCKER_IMAGE] in docker.calls


def test_wizard_offers_setup_when_granting_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    docker = Docker()
    monkeypatch.setattr(sandbox, "Sandbox", docker.sandbox)
    monkeypatch.setattr(ui, "checkbox", lambda *_args, **_kwargs: {"shell_execution"})
    offered: list[str] = []

    def decline(question: str) -> bool:
        offered.append(question)
        return False

    monkeypatch.setattr(ui, "confirm", decline)
    assert wizard._permissions(Permissions()).shell_execution
    assert not offered
    assert [call[0] for call in docker.calls] == ["info", "image"]
    docker.answers["image"] = (1, "", "No such image")
    saved = wizard._permissions(Permissions())
    assert saved.shell_execution
    assert len(offered) == 1 and "45 MB" in offered[0]
    assert not any(call[0] == "pull" for call in docker.calls)
    assert wizard._permissions(saved).shell_execution
    assert len(offered) == 1


def test_run_shares_sandbox_probe(store: Store) -> None:
    docker = Docker()
    box = docker.sandbox()
    config = Config(permissions=Permissions(shell_execution=True))
    access = runs.executor_access(config, sandbox=box)
    coordinator = runs.coordinator(store, runs.Models(FakeLLM(), None), access, config, sandbox=box)
    tool = coordinator.tools.get("run_command") if coordinator.tools else None
    assert tool is not None and tool.ready() is None
    assert sum(call[0] == "run" for call in docker.calls) == 1


@pytest.mark.parametrize(
    ("failure", "expected", "setup_hint", "install_link"),
    [
        ("absent", "Docker is not installed", False, True),
        ("windows", "switch Docker Desktop to Linux containers", False, False),
        ("image", "The sandbox image is missing", True, False),
    ],
)
def test_executor_access_hints(
    failure: str,
    expected: str,
    setup_hint: bool,
    install_link: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docker = Docker()
    if failure == "windows":
        docker.answers["info"] = (0, "windows", "")
    if failure == "image":
        docker.answers["image"] = (1, "", "No such image")
    box = Sandbox(
        platform="darwin",
        docker="" if failure == "absent" else "docker-test",
        docker_runner=docker.run,
    )
    runs.executor_access(Config(permissions=Permissions(shell_execution=True)), sandbox=box)
    out = capsys.readouterr().out.replace("\n", " ")
    assert expected in out
    assert ("desman sandbox setup" in out) is setup_hint
    assert ("desman docker install" in out) is install_link


@pytest.fixture(scope="module")
def real_docker() -> Sandbox:
    # Exercise the Docker backend even on Linux, whose default is bubblewrap.
    box = Sandbox(platform="win32" if sys.platform == "win32" else "darwin")
    reason = box.unavailable()
    if reason is not None:
        if os.environ.get("DESMAN_DOCKER_TESTS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    return box


def test_real_docker_hello_and_workspace(real_docker: Sandbox, tmp_path: Path) -> None:
    result = real_docker.run(
        [
            "python3",
            "-c",
            "from pathlib import Path; Path('answer').write_text(input()); print('hello')",
        ],
        tmp_path,
        stdin="42\n",
    )
    assert result == Outcome(0, "hello\n", "")
    assert (tmp_path / "answer").read_text() == "42"


@pytest.mark.parametrize(("code", "message"), [(125, "program error"), (1, "docker: x")])
def test_real_docker_program_exit(
    real_docker: Sandbox, tmp_path: Path, code: int, message: str
) -> None:
    result = real_docker.run(
        ["python3", "-c", f"import sys; print({message!r}, file=sys.stderr); sys.exit({code})"],
        tmp_path,
    )
    assert result == Outcome(code, "", message + "\n")


def test_real_docker_ignores_environment_injected_by_docker(
    real_docker: Sandbox,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_argv = real_docker._docker_argv

    def with_proxy(command: list[str], workspace: Path, name: str) -> list[str]:
        argv = docker_argv(command, workspace, name)
        argv[2:2] = ["-e", "HTTPS_PROXY=mock-secret"]
        return argv

    monkeypatch.setattr(real_docker, "_docker_argv", with_proxy)
    result = real_docker.run(["env"], tmp_path)
    assert result.exit_code == 0, result
    assert dict(line.split("=", 1) for line in result.stdout.splitlines()) == sandbox._DOCKER_ENV


def test_real_docker_has_no_network_and_readonly_root(real_docker: Sandbox, tmp_path: Path) -> None:
    result = real_docker.run(
        [
            "python3",
            "-c",
            "import errno, os, socket\n"
            "assert [name for _, name in socket.if_nameindex()] == ['lo']\n"
            "try: socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
            "except OSError as exc: assert exc.errno == errno.ENETUNREACH\n"
            "else: raise AssertionError('network is available')\n"
            "assert os.statvfs('/').f_flag & os.ST_RDONLY\n"
            "open('/tmp/writable', 'w').close()",
        ],
        tmp_path,
    )
    assert result == Outcome(0, "", "")


def test_real_docker_pytest_and_output_limits(real_docker: Sandbox, tmp_path: Path) -> None:
    (tmp_path / "test_x.py").write_text("def test_x():\n    assert 21 * 2 == 42\n")
    test = real_docker.run(["python3", "-m", "pytest", "-q"], tmp_path)
    assert test.exit_code == 0 and "1 passed" in test.stdout, test
    assert not (tmp_path / ".pytest_cache").exists()
    result = real_docker.run(
        [
            "python3",
            "-c",
            "import sys; print('first' + 'x' * 100_000); "
            "print('x' * 100_000 + 'last', file=sys.stderr)",
        ],
        tmp_path,
    )
    assert result.exit_code == 0
    assert result.stdout_cut and result.stdout.startswith("first")
    assert len(result.stdout) == real_docker.limits.stdout_chars
    assert result.stderr_cut and result.stderr.endswith("last\n")
    assert len(result.stderr) == real_docker.limits.stderr_chars


def test_real_docker_timeout_leaves_no_container(
    real_docker: Sandbox,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = uuid.uuid4()
    monkeypatch.setattr(uuid, "uuid4", lambda: identity)
    result = real_docker.run(
        ["python3", "-c", "print('started', flush=True); import time; time.sleep(30)"],
        tmp_path,
        timeout=5,
    )
    assert result.timed_out and result.exit_code is None and result.stdout == "started\n", result
    listing = real_docker.docker_command(
        [
            "ps",
            "-a",
            "--filter",
            "label=" + DOCKER_LABEL,
            "--filter",
            f"name=desman-sbx-{identity.hex}",
        ]
    )
    assert listing.returncode == 0 and identity.hex not in listing.stdout, listing
