"""WSL selection, setup, and command construction on any host."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from desman import sandbox_cli, ui, wizard
from desman.cli import cli
from desman.config import Permissions
from desman.core import sandbox
from desman.core.sandbox import WSL_DISTRO, WSL_PIN, Limits, Sandbox, SandboxUnavailable

WINDOWS_PATH = Path(r"D:\Task folders\one")
TAR = b"pinned archive"
_REAL_WSL_WORKSPACE = sandbox._wsl_workspace


class Wsl:
    def __init__(self, *, usable: bool = True, present: bool = False, fail: str = "") -> None:
        self.usable = usable
        self.present = present
        self.fail = fail
        self.calls: list[tuple[list[str], float, bool]] = []

    def __call__(
        self, argv: list[str], timeout: float, terminal: bool
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, timeout, terminal))
        action = argv[1]
        out = b""
        if action == "--status":
            return subprocess.CompletedProcess(argv, int(not self.usable), out, b"")
        if action == "--list":
            if self.present:
                out = f"{WSL_DISTRO}\r\n".encode("utf-16-le")
        elif action == "--import":
            assert Path(argv[4]).read_bytes() == TAR
            assert argv[5:] == ["--version", "2"]
            self.present = True
        elif action == "--unregister":
            self.present = False
        elif action == "-d":
            action = argv[argv.index("--exec") + 1]
        failed = action == self.fail
        return subprocess.CompletedProcess(
            argv, int(failed), out, "failed step".encode("utf-16-le") if failed else b""
        )

    def box(
        self, directory: Path, *, docker: bool = False, limits: Limits | None = None
    ) -> Sandbox:
        def run_docker(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
            stdout = "linux 2026-01-01T00:00:00+00:00" if argv[1] == "info" else ""
            return subprocess.CompletedProcess(argv, int(not docker), stdout, "daemon stopped")

        return Sandbox(
            limits,
            platform="win32",
            docker="docker.exe",
            docker_runner=run_docker,
            wsl="wsl.exe",
            wsl_runner=self,
            wsl_dir=directory / "wsl",
            wasi_dir=directory / "wasi",
        )


@pytest.fixture(autouse=True)
def probe_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_wsl_workspace", lambda _: "/mnt/d/empty")


@pytest.mark.parametrize(
    ("docker", "present", "failure", "expected"),
    [
        (True, True, "", "docker"),
        (False, True, "", "wsl"),
        (False, False, "", "wasi"),
        (False, True, "/bin/busybox", "wasi"),
    ],
)
def test_backend_order(
    tmp_path: Path, docker: bool, present: bool, failure: str, expected: str
) -> None:
    wsl = Wsl(present=present, fail=failure)
    box = wsl.box(tmp_path, docker=docker)
    problem = box.unavailable()
    assert box.backend == expected
    if expected == "docker":
        assert not wsl.calls
    elif expected == "wsl":
        assert problem is None
        assert wsl.calls[-1][0][-1] == "true" and wsl.calls[-1][1:] == (60.0, False)
    else:
        assert problem is not None and "desman sandbox setup" in problem
        assert ("failed step" if failure else "distribution is missing") in problem


def test_workspace_and_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "_wsl_workspace", _REAL_WSL_WORKSPACE)
    limits = Limits(cpu_seconds=9, memory_bytes=100, file_bytes=200, open_files=16, tmp_bytes=300)
    box = Wsl().box(tmp_path, limits=limits)
    code = "print('one; two $(three)')"
    argv = box._wsl_argv(["python3", "-c", code], WINDOWS_PATH, 2.2)
    assert argv[:8] == ["wsl.exe", "-d", WSL_DISTRO, "-u", "sandbox", "--cd", "/", "--exec"]
    assert argv[8:13] == ["/bin/busybox", "timeout", "-s", "KILL", "8"]
    assert argv[argv.index("--bind") + 1 : argv.index("--bind") + 3] == [
        "/mnt/d/Task folders/one",
        "/workspace",
    ]
    assert argv[argv.index("--size") + 1] == "300"
    assert argv[-12:] == [
        "--",
        "/usr/bin/prlimit",
        "--nproc=32",
        "--as=100",
        "--fsize=200",
        "--nofile=16",
        "--cpu=9",
        "--core=0",
        "--",
        "python3",
        "-c",
        code,
    ]
    for path in (r"\\server\share\task", r"C:relative", "relative", r"\rooted"):
        with pytest.raises(SandboxUnavailable, match="local Windows drive"):
            box._wsl_argv(["true"], Path(path))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16"])
def test_wsl_decodes_stderr(encoding: str) -> None:
    message = "WSL error: distribution indisponível\r\n"
    data = message.encode(encoding)
    assert sandbox._decode_wsl(data) == message
    tail = sandbox._WslTail(14)
    tail.add(data)
    assert tail.text() == message[-14:]
    assert tail.cut


def test_wsl_decodes_localized_stderr_with_few_nuls() -> None:
    message = "ディストリビューションがありません。\r\n"
    assert sandbox._decode_wsl(message.encode("utf-16-le")) == message
    assert sandbox._decode_wsl(b"abc\0def") == "abc\0def"


def test_environment_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSLENV", "SECRET")
    monkeypatch.setenv("SECRET", "a key")
    monkeypatch.setenv("USERPROFILE", r"C:\User")
    assert sandbox._wsl_env()["USERPROFILE"] == r"C:\User"
    assert not {"SECRET", "WSLENV"} & sandbox._wsl_env().keys()


def test_install_verifies_pin_and_runs_ordered_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsl = Wsl()
    box = wsl.box(tmp_path)
    assert sandbox_cli.setup_backend(box) == "wsl"
    assert sandbox_cli.prepare_setup(box, "wsl") is not None
    with pytest.raises(SandboxUnavailable, match="SHA-256 mismatch"):
        sandbox.install_wsl(box, download=lambda _: b"bad")
    assert not any(argv[1] == "--import" for argv, _, _ in wsl.calls)

    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())

    def fetch(url: str) -> bytes:
        assert url.endswith(f"/{WSL_PIN}.tar.gz")
        return TAR

    wsl.calls.clear()
    assert sandbox.install_wsl(box, download=fetch)
    calls = [argv[1:] for argv, _, _ in wsl.calls]
    install, packages, user, marker = calls[1:5]
    assert install[:3] == ["--import", WSL_DISTRO, str(box.wsl_dir / "root")]
    assert packages[7:] == [
        "/sbin/apk",
        "add",
        "--no-cache",
        "python3",
        "py3-pytest",
        "bubblewrap",
        "util-linux-misc",
    ]
    assert user[7:] == ["/usr/sbin/adduser", "-D", "-H", "-s", "/bin/sh", "sandbox"]
    assert "/etc/desman-sandbox" in marker[9] and marker[-1].startswith(f"{WSL_PIN} ")
    probe = calls[-1]
    assert probe[:4] == ["-d", WSL_DISTRO, "-u", "sandbox"] and probe[-1] == "true"
    assert not sandbox.install_wsl(box, download=lambda _: pytest.fail("downloaded twice"))


@pytest.mark.parametrize("step", ["/sbin/apk", "/usr/sbin/adduser", "/bin/sh", "/bin/busybox"])
def test_failed_setup_reports_step_and_unregisters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    wsl = Wsl(fail=step)
    box = wsl.box(tmp_path)
    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())
    label = "could not start" if step == "/bin/busybox" else f"{step.rsplit('/', 1)[-1]} failed"
    with pytest.raises(SandboxUnavailable, match=f"{label}.*failed step"):
        sandbox.install_wsl(box, download=lambda _: TAR)
    assert wsl.calls[-1][0][1:] == ["--unregister", WSL_DISTRO]


def test_force_wsl_plan_and_terminal_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsl = Wsl(usable=False)
    box = wsl.box(tmp_path, docker=True)
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: box)
    runner = CliRunner()
    plan = runner.invoke(cli, ["sandbox", "setup", "--wsl"])
    assert plan.exit_code == 0 and "desman sandbox setup --wsl --yes" in plan.output
    assert [argv[1] for argv, _, _ in wsl.calls] == ["--status"]
    result = runner.invoke(cli, ["sandbox", "setup", "--wsl", "--yes"])
    assert result.exit_code == 0 and "Restart Windows" in result.output
    assert wsl.calls[-1] == (["wsl.exe", "--install", "--no-distribution"], 1800.0, True)


def test_setup_automatically_installs_wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wsl = Wsl()
    box = wsl.box(tmp_path)
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: box)
    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())
    monkeypatch.setattr(sandbox, "_download_wsl", lambda _: TAR)
    runner = CliRunner()
    plan = runner.invoke(cli, ["sandbox", "setup"])
    assert plan.exit_code == 0 and "60 MB" in plan.output
    assert not box.wsl_dir.exists()
    result = runner.invoke(cli, ["sandbox", "setup", "--yes"])
    assert result.exit_code == 0, result.output
    assert wsl.present and box.wsl_dir.exists()


def test_remove_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    box.wsl_dir.mkdir()
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: wsl.box(tmp_path))
    runner = CliRunner()

    status = runner.invoke(cli, ["sandbox", "status", "--json"])
    assert status.exit_code == 0, status.output
    data = json.loads(status.output)
    assert data["backend"] == "wsl" and data["available"]
    assert data["wsl_present"] and data["wsl_pin"] == WSL_PIN

    refused = runner.invoke(cli, ["sandbox", "remove", "--yes"])
    assert refused.exit_code == 2 and wsl.present
    plan = runner.invoke(cli, ["sandbox", "remove", "--wsl"])
    assert plan.exit_code == 0 and wsl.present and box.wsl_dir.exists()
    removed = runner.invoke(cli, ["sandbox", "remove", "--wsl", "--yes"])
    assert removed.exit_code == 0, removed.output
    assert not wsl.present and not box.wsl_dir.exists()
    assert wsl.calls[-1][0][1:] == ["--unregister", WSL_DISTRO]
    status = runner.invoke(cli, ["sandbox", "status", "--json"])
    assert status.exit_code == 0 and not json.loads(status.output)["wsl_present"]


def test_wizard_offers_wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wsl = Wsl()
    box = wsl.box(tmp_path)
    offered: list[str] = []

    def confirm(question: str) -> bool:
        offered.append(question)
        return False

    monkeypatch.setattr(sandbox, "Sandbox", lambda: box)
    monkeypatch.setattr(ui, "checkbox", lambda *_, **_kw: {"shell_execution"})
    monkeypatch.setattr(ui, "confirm", confirm)
    assert wizard._permissions(Permissions()).shell_execution
    assert len(offered) == 1 and "WSL sandbox" in offered[0] and "60 MB" in offered[0]
