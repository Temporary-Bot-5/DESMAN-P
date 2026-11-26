"""WSL selection, setup, and command construction on any host."""

import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from desman import docker_cli, sandbox_cli, ui, wizard
from desman.cli import cli
from desman.config import Permissions, open_config
from desman.core import sandbox, seccomp
from desman.core.sandbox import WSL_DISTRO, WSL_PIN, Limits, Sandbox, SandboxUnavailable

WINDOWS_PATH = Path(r"D:\Task folders\one")
TAR = b"pinned archive"
_REAL_WSL_WORKSPACE = sandbox._wsl_workspace


class Wsl:
    def __init__(self, *, usable: bool = True, present: bool = False, fail: str = "") -> None:
        self.usable = usable
        self.present = present
        self.fail = fail
        self.conf: bytes | None = b"[interop]\nenabled=false\nappendWindowsPath=false\n"
        self.marker = WSL_PIN.encode()
        self.status = b""
        self.calls: list[tuple[list[str], float, bool]] = []

    def __call__(
        self, argv: list[str], timeout: float, terminal: bool
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((argv, timeout, terminal))
        action = argv[1]
        out = b""
        if action == "--status":
            return subprocess.CompletedProcess(argv, int(not self.usable), self.status, b"")
        if action == "--list":
            if self.present:
                out = f"{WSL_DISTRO}\r\n".encode("utf-16-le")
        elif action == "--import":
            assert Path(argv[4]).read_bytes() == TAR
            assert argv[5:] == ["--version", "2"]
            self.present = self.fail != "--import"
        elif action == "--unregister":
            self.present = False
        elif action == "-d":
            action = argv[argv.index("--exec") + 1]
            if action == "/bin/cat":
                if argv[-1] == "/etc/wsl.conf":
                    if self.conf is None:
                        return subprocess.CompletedProcess(argv, 1, b"", b"missing")
                    out = self.conf
                else:
                    assert argv[-1] == "/etc/desman-sandbox"
                    out = self.marker
            elif action == "/bin/sh":
                if "/etc/wsl.conf" in argv[10]:
                    action = "write /etc/wsl.conf"
                    self.conf = "".join(f"{line}\n" for line in argv[12:]).encode()
                else:
                    self.marker = argv[-1].encode()
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
    assert argv[8:13] == ["/bin/busybox", "timeout", "-s", "KILL", "5"]
    assert argv[13:17] == ["/bin/busybox", "sh", "-c", sandbox._WSL_SECCOMP]
    assert argv[17] == "sh" and argv[19:22] == ["/usr/bin/bwrap", "--seccomp", "3"]
    escaped = argv[18].split("\\")[1:]
    assert bytes(int(byte, 8) for byte in escaped) == seccomp.program("x86_64")
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
    install, conf, packages, user, marker, terminate, status, read_conf, read_marker, probe = calls[
        1:
    ]
    assert install[:3] == ["--import", WSL_DISTRO, str(box.wsl_dir / "root")]
    assert packages[7:] == [
        "/sbin/apk",
        "add",
        "--no-cache",
        "python3",
        "py3-pytest",
        "bubblewrap",
        "util-linux-misc",
        "procps-ng",
    ]
    assert user[7:] == ["/usr/sbin/adduser", "-D", "-H", "-s", "/bin/sh", "sandbox"]
    assert conf[7:10] == ["/bin/sh", "-c", """printf '%s\\n' "$@" > /etc/wsl.conf"""]
    assert wsl.conf == b"[interop]\nenabled=false\nappendWindowsPath=false\n"
    assert "/etc/desman-sandbox" in marker[9] and marker[-1].startswith(f"{WSL_PIN} ")
    assert terminate == ["--terminate", WSL_DISTRO]
    assert status == ["--list", "--quiet"]
    assert read_conf[-2:] == ["/bin/cat", "/etc/wsl.conf"]
    assert read_marker[-2:] == ["/bin/cat", "/etc/desman-sandbox"]
    assert probe[:4] == ["-d", WSL_DISTRO, "-u", "sandbox"] and probe[-1] == "true"
    assert not sandbox.install_wsl(box, download=lambda _: pytest.fail("downloaded twice"))


@pytest.mark.parametrize(
    "step",
    [
        "write /etc/wsl.conf",
        "/sbin/apk",
        "/usr/sbin/adduser",
        "/bin/sh",
        "--terminate",
        "/bin/busybox",
    ],
)
def test_failed_setup_reports_step_and_unregisters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    wsl = Wsl(fail=step)
    box = wsl.box(tmp_path)
    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())
    label = (
        "could not start"
        if step == "/bin/busybox"
        else (
            f"{step.removeprefix('/sbin/').removeprefix('/usr/sbin/').removeprefix('/bin/')} failed"
        )
    )
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


@pytest.mark.parametrize(
    "conf",
    [
        None,
        b"",
        b"[interop]\nenabled=true\n",
        b"[interop]\nenabled=nope\n",
        b"[other]\nenabled=false\n",
        b"[interop]\nenabled=false\nenabled=true\n",
    ],
)
def test_probe_refuses_unknown_or_enabled_interop(tmp_path: Path, conf: bytes | None) -> None:
    wsl = Wsl(present=True)
    wsl.conf = conf
    assert wsl.box(tmp_path).wsl_problem() == (
        "The WSL sandbox is not hardened (interop is on); "
        "run `desman sandbox setup --wsl --yes` to repair it"
    )
    assert wsl.calls[-1][0][-2:] == ["/bin/cat", "/etc/wsl.conf"]


@pytest.mark.parametrize("problem", ["pin", "interop", "bwrap"])
def test_repair_plan_and_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    wsl = Wsl(present=True, fail="/bin/busybox" if problem == "bwrap" else "")
    if problem == "pin":
        wsl.marker = b"old-pin"
    elif problem == "interop":
        wsl.conf = b"[interop]\nenabled=true\n"
    box = wsl.box(tmp_path)
    assert "setup --wsl --yes" in (box.wsl_problem() or "")
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: box)
    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())

    def download(_: str) -> bytes:
        wsl.fail = ""
        return TAR

    monkeypatch.setattr(sandbox, "_download_wsl", download)
    runner = CliRunner()
    plan = runner.invoke(cli, ["sandbox", "setup", "--wsl"])
    assert plan.exit_code == 0 and "Unregister" in plan.output and "reinstall" in plan.output
    assert not any(argv[1] == "--unregister" for argv, _, _ in wsl.calls)
    result = runner.invoke(cli, ["sandbox", "setup", "--wsl", "--yes"])
    assert result.exit_code == 0, result.output
    mutations = [argv[1] for argv, _, _ in wsl.calls if argv[1] in ("--unregister", "--import")]
    assert mutations == ["--unregister", "--import"]
    assert box.wsl_problem() is None


def test_setup_skips_wsl_when_status_says_wsl2_is_unavailable(tmp_path: Path) -> None:
    wsl = Wsl()
    wsl.status = "Default Version: 2\r\nWSL2 is not supported with your current machine configuration.\r\n".encode(
        "utf-16-le"
    )
    assert sandbox_cli.setup_backend(wsl.box(tmp_path)) == "wasi"


@pytest.mark.parametrize("option", ["", "--wsl", "--wasi"])
def test_setup_wasi_override_and_auto_import_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    wsl = Wsl(fail="--import")
    box = wsl.box(tmp_path, docker=option == "--wasi")
    monkeypatch.setattr(sandbox_cli, "Sandbox", lambda: box)
    monkeypatch.setattr(sandbox_cli, "wasi_runtime", lambda: True)
    monkeypatch.setattr(sandbox, "_WSL_SHA256", hashlib.sha256(TAR).hexdigest())
    monkeypatch.setattr(sandbox, "_download_wsl", lambda _: TAR)
    installed: list[Path] = []

    def install(path: Path) -> bool:
        installed.append(path)
        return True

    monkeypatch.setattr(sandbox_cli, "install_wasi", install)
    runner = CliRunner()
    if option == "--wasi":
        plan = runner.invoke(cli, ["sandbox", "setup", option])
        assert plan.exit_code == 0 and "desman sandbox setup --wasi --yes" in plan.output
        assert not installed
    result = runner.invoke(cli, ["sandbox", "setup", "--yes", *([option] if option else [])])
    if option == "--wsl":
        assert result.exit_code == 1 and "--import failed" in result.output
        assert not installed
    else:
        assert result.exit_code == 0, result.output
        assert installed == [box.wasi_dir]
        assert ("WSL import failed" in result.output) == (not option)
    assert any(argv[1] == "--import" for argv, _, _ in wsl.calls) == (option != "--wasi")


def test_setup_rejects_both_backend_flags() -> None:
    result = CliRunner().invoke(cli, ["sandbox", "setup", "--wsl", "--wasi", "--yes"])
    assert result.exit_code == 2 and "Choose either" in result.output


@pytest.mark.parametrize("daemon_reported_ready", [False, True])
def test_wizard_does_not_offer_wsl_in_docker_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    daemon_reported_ready: bool,
) -> None:
    wsl = Wsl()
    monkeypatch.setattr(sandbox, "Sandbox", lambda: wsl.box(tmp_path))
    monkeypatch.setattr(wizard, "sys", SimpleNamespace(platform="win32"))
    state = docker_cli.Status(
        installed=False, available=False, problem="missing", version=None, bwrap=False
    )
    monkeypatch.setattr(docker_cli, "docker_status", lambda **_: state)
    monkeypatch.setattr(
        docker_cli,
        "current_plan",
        lambda _: docker_cli.plan(
            "win32", lambda program: program if program == "winget" else None, {}, "", state
        ),
    )
    monkeypatch.setattr(
        docker_cli,
        "apply_plan",
        lambda _: {**state, "installed": True, "available": daemon_reported_ready},
    )
    monkeypatch.setattr(wizard, "_offer_searxng", lambda _: None)
    monkeypatch.setattr(ui, "confirm", lambda question, **_: "Install Docker" in question)
    wizard._docker(wizard._Session(open_config().load()))
    assert not wsl.calls
    assert "Once Docker Desktop is running" in capsys.readouterr().out


class Process:
    def __init__(self, code: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = code
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode


@pytest.mark.parametrize("interrupted", [False, True])
def test_wsl_kills_only_timed_out_or_interrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    names: list[str] = []
    events: list[str] = []

    class Running(Process):
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                assert timeout == 2.2
                if interrupted:
                    raise KeyboardInterrupt
                raise subprocess.TimeoutExpired("wsl.exe", timeout)
            return super().wait()

        def poll(self) -> int | None:
            return self.returncode if events[-1:] == ["kill"] else None

        def kill(self) -> None:
            events.append("kill")

    def start(argv: list[str], **_: object) -> Running:
        assert argv[12] == "5"
        name = argv[argv.index("--hostname") + 1]
        assert re.fullmatch(r"desman-[a-f0-9]{32}", name)
        names.append(name)
        return Running(-1)

    def run_wsl(
        argv: list[str], timeout: float, terminal: bool
    ) -> subprocess.CompletedProcess[bytes]:
        if "/usr/bin/pkill" in argv:
            assert events[-1] == "kill"
            assert argv == [
                "wsl.exe",
                "-d",
                WSL_DISTRO,
                "-u",
                "root",
                "--cd",
                "/",
                "--exec",
                "/usr/bin/pkill",
                "-KILL",
                "-f",
                names[-1],
            ]
            assert timeout == 5 and not terminal
            events.append("cleanup")
            raise subprocess.TimeoutExpired(argv, timeout)
        return wsl(argv, timeout, terminal)

    monkeypatch.setattr(box, "_wsl_runner", run_wsl)
    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(sandbox, "_watch", lambda *_: None)
    for _ in range(2):
        if interrupted:
            with pytest.raises(KeyboardInterrupt):
                box.run(["true"], tmp_path, timeout=2.2)
        else:
            outcome = box.run(["true"], tmp_path, timeout=2.2)
            assert outcome.timed_out and outcome.exit_code is None
    assert events == ["kill", "cleanup"] * 2
    assert len(set(names)) == 2


def test_a_wsl_run_that_fills_the_workspace_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    events: list[str] = []

    class Running(Process):
        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and events[-1:] != ["kill"]:
                assert timeout <= 1.0
                raise subprocess.TimeoutExpired("wsl.exe", timeout)
            return super().wait()

        def poll(self) -> int | None:
            return self.returncode if events[-1:] == ["kill"] else None

        def kill(self) -> None:
            events.append("kill")

    answers = iter([None, sandbox.Stop("the workspace grew past 10 files")])
    monkeypatch.setattr(sandbox, "_watch", lambda *_: lambda: next(answers))
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: Running(1))
    outcome = box.run(["true"], tmp_path, timeout=30)
    assert outcome.stopped == "the workspace grew past 10 files"
    assert outcome.exit_code == 128 + 9 and not outcome.timed_out
    assert events == ["kill"]
    assert any("/usr/bin/pkill" in argv for argv, _, _ in wsl.calls)  # the instance stopped


@pytest.mark.parametrize(
    ("code", "stdout", "message", "unavailable"),
    [
        (-1, b"", "", True),
        (4294967295, b"", "", True),
        (1, b"", "There is no distribution with the supplied name.", True),
        (1, b"", "WSL (7) ERROR", True),
        (1, b"", "Error code: Wsl/Service/Start", True),
        (1, b"generated output", "WSL (7) ERROR", False),
        (255, b"", "Python exception", False),
    ],
)
def test_wsl_process_failure_is_distinct_from_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    stdout: bytes,
    message: str,
    unavailable: bool,
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path, limits=Limits(stderr_chars=3))
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_, **__: Process(code, stdout, message.encode("utf-16-le")),
    )
    if unavailable:
        # Only a failing probe makes it the sandbox's failure.
        assert box.unavailable() is None
        wsl.fail = "/bin/busybox"
        with pytest.raises(SandboxUnavailable, match="WSL failed"):
            box.run(["true"], tmp_path)
        wsl.fail = ""
        wsl.calls.clear()
        assert box.unavailable() is None
        assert wsl.calls[-1][0][-1] == "true"
        assert not any("/usr/bin/pkill" in argv for argv, _, _ in wsl.calls)
    else:
        assert box.run(["true"], tmp_path).exit_code == code


def test_code_printing_a_wsl_error_is_its_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_, **__: Process(1, b"", "There is no distribution".encode("utf-16-le")),
    )
    outcome = box.run(["true"], tmp_path)
    assert outcome.exit_code == 1 and not outcome.timed_out
    assert box.unavailable() is None


@pytest.mark.parametrize("code", [-1, 4294967295])
def test_an_exit_code_only_wsl_makes_is_unavailable_even_with_a_healthy_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *_, **__: Process(code, b"", b""))
    with pytest.raises(SandboxUnavailable, match="WSL failed"):
        box.run(["true"], tmp_path)


def test_a_failing_probe_names_its_own_problem_not_the_codes_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wsl = Wsl(present=True)
    box = wsl.box(tmp_path)
    assert box.unavailable() is None
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_, **__: Process(1, b"", b"WSL (1) ERROR: run this")
    )
    wsl.conf = None
    with pytest.raises(SandboxUnavailable) as raised:
        box.run(["true"], tmp_path)
    assert "run this" not in str(raised.value) and "interop" in str(raised.value)
