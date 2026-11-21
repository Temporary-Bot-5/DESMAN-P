"""Docker plans and operator commands, with a fake subprocess boundary."""

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import NamedTuple

import click
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import docker_cli, providers, sandbox_cli, searxng, ui, wizard
from desman.cli import cli
from desman.config import Config, ConfigManager, open_config
from desman.core import sandbox
from desman.core.sandbox import Sandbox
from desman.docker_cli import Plan, Status, plan
from test_wizard import KEEP_PERMISSIONS, KEEP_PERSONAL

WINDOWS_START = [r"C:\Program Files\Docker\Docker\Docker Desktop.exe"]
WINGET = [
    "winget",
    "install",
    "--exact",
    "--id",
    "Docker.DockerDesktop",
    "--source",
    "winget",
    "--accept-package-agreements",
    "--accept-source-agreements",
]
MAC_INSTALL = ["brew", "install", "--cask", "docker-desktop"]
MAC_START = ["open", "-a", "Docker"]


def state(
    *,
    installed: bool = False,
    ready: bool = False,
    bwrap: bool | None = None,
    problem: str | None = None,
) -> Status:
    return Status(installed=installed, available=ready, version=None, bwrap=bwrap, problem=problem)


def available_tool(name: str) -> str:
    return name


def argv(installation: Plan) -> list[list[str]]:
    return [step.argv for step in installation.steps]


@pytest.mark.parametrize(
    ("platform", "install", "start", "url"),
    [
        ("win32", WINGET, WINDOWS_START, "windows-install/"),
        ("darwin", MAC_INSTALL, MAC_START, "mac-install/"),
    ],
)
def test_desktop_plans(platform: str, install: list[str], start: list[str], url: str) -> None:
    installation = plan(platform, available_tool, {}, "person", state())
    assert argv(installation) == [install, start]
    assert installation.steps[-1].detached is (platform == "win32")
    assert installation.needs_admin
    assert installation.may_reboot is (platform == "win32")
    assert installation.download and "600 MB" in installation.download
    assert installation.license and "250 employees" in installation.license
    assert installation.license and "$10 million" in installation.license
    assert any("desman sandbox setup --yes" in line for line in installation.after)
    assert any("desman search setup --yes" in line for line in installation.after)

    manual = plan(platform, lambda _: None, {}, "person", state())
    assert argv(manual) == []
    assert manual.link == "https://docs.docker.com/desktop/setup/install/" + url


@pytest.mark.parametrize(
    ("release", "package", "update"),
    [
        ({"ID": "debian"}, "apt-get install -y docker.io", ["sudo apt-get update"]),
        (
            {"ID": "mint", "ID_LIKE": "ubuntu debian"},
            "apt-get install -y docker.io",
            ["sudo apt-get update"],
        ),
        ({"ID": "fedora"}, "dnf install -y moby-engine", []),
        ({"ID": "nobara", "ID_LIKE": "fedora"}, "dnf install -y moby-engine", []),
        ({"ID": "arch"}, "pacman -S --needed --noconfirm docker", []),
        (
            {"ID": "opensuse-tumbleweed", "ID_LIKE": "opensuse suse"},
            "zypper --non-interactive install docker",
            [],
        ),
    ],
)
@pytest.mark.parametrize("bwrap", [True, False])
def test_linux_packages(
    release: dict[str, str], package: str, update: list[str], bwrap: bool
) -> None:
    installation = plan("linux", available_tool, release, "person", state(bwrap=bwrap))
    assert argv(installation) == [
        *(command.split() for command in update),
        ["sudo", *package.split(), *([] if bwrap else ["bubblewrap"])],
        ["sudo", "systemctl", "enable", "--now", "docker"],
        ["sudo", "usermod", "-aG", "docker", "person"],
    ]
    lines = " ".join(installation.lines())
    assert "docker group gives administrator rights" in lines
    assert "Log out and back in" in lines
    assert "desman search setup --yes" in lines
    assert "desman sandbox setup" not in lines


@pytest.mark.parametrize("distro", ["rhel", "rocky", "almalinux", "centos"])
@pytest.mark.parametrize("bwrap", [True, False])
def test_rhel_family_uses_manual_docker_install(distro: str, bwrap: bool) -> None:
    release = {"ID": distro, "ID_LIKE": "rhel centos fedora"}
    installation = plan("linux", available_tool, release, "person", state(bwrap=bwrap))
    assert installation.link == docker_cli.ENGINE_LINK
    assert argv(installation) == ([] if bwrap else [["sudo", "dnf", "install", "-y", "bubblewrap"]])


def test_unknown_linux_has_link_and_no_steps() -> None:
    installation = plan("linux", available_tool, {"ID": "other"}, "person", state(bwrap=False))
    assert argv(installation) == []
    assert installation.link == "https://docs.docker.com/engine/install/"


def test_root_does_not_use_sudo() -> None:
    installation = plan("linux", available_tool, {"ID": "ubuntu"}, "root", state(bwrap=True))
    assert argv(installation) == [
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "docker.io"],
        ["systemctl", "enable", "--now", "docker"],
    ]
    assert not installation.notices
    assert all("Log out" not in note for note in installation.after)


@pytest.mark.parametrize(
    ("platform", "start"),
    [
        ("win32", WINDOWS_START),
        ("darwin", MAC_START),
        ("linux", ["sudo", "systemctl", "start", "docker"]),
    ],
)
def test_stopped_docker_only_starts(platform: str, start: list[str]) -> None:
    stopped = state(installed=True, bwrap=True)
    assert argv(plan(platform, lambda _: None, {}, "person", stopped)) == [start]


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_ready_docker_needs_no_steps(platform: str) -> None:
    ready = state(installed=True, ready=True, bwrap=True)
    installation = plan(platform, lambda _: None, {}, "person", ready)
    assert installation.lines() == ["Docker is ready."]
    assert not installation.steps


def test_ready_linux_can_install_only_bubblewrap() -> None:
    ready = state(installed=True, ready=True, bwrap=False)
    installation = plan("linux", available_tool, {"ID": "fedora"}, "person", ready)
    assert argv(installation) == [["sudo", "dnf", "install", "-y", "bubblewrap"]]
    assert "Docker is ready." in installation.lines()


class Call(NamedTuple):
    argv: list[str]
    timeout: float | None
    detached: bool


class Machine:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[Call] = []
        self.installed = False
        self.ready = False
        self.fail = 0
        self.desktop_started = False
        self.answer_on_poll = 1
        self.polls = 0
        monkeypatch.setattr(docker_cli, "sys", SimpleNamespace(platform="darwin"))
        monkeypatch.setattr(shutil, "which", self.which)
        monkeypatch.setattr(docker_cli, "docker_executable", lambda: self.which("docker"))
        monkeypatch.setattr(docker_cli, "_process", self.process)

    def which(self, name: str) -> str | None:
        if name == "brew" or (name == "docker" and self.installed):
            return name
        return None

    def process(
        self, argv: list[str], *, timeout: float | None = None, detached: bool = False
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(Call(argv, timeout, detached))
        if argv == MAC_INSTALL:
            self.installed = self.fail == 0
            return subprocess.CompletedProcess(argv, self.fail)
        if argv == MAC_START:
            self.desktop_started = True
        if argv == ["docker", "info"]:
            if self.desktop_started:
                self.polls += 1
                self.ready = self.polls >= self.answer_on_poll
            return subprocess.CompletedProcess(
                argv,
                0 if self.ready else 1,
                "info" if self.ready else "",
                None if self.ready else "daemon unavailable é",
            )
        if argv == ["docker", "--version"]:
            return subprocess.CompletedProcess(argv, 0, "Docker version 29.0.0")
        return subprocess.CompletedProcess(argv, 0)


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Machine:
    return Machine(monkeypatch)


def test_preview_prints_all_commands_without_running_them(machine: Machine) -> None:
    result = CliRunner().invoke(cli, ["docker", "install"])
    assert result.exit_code == 0, result.output
    assert " ".join(MAC_INSTALL) in result.output
    assert " ".join(MAC_START) in result.output
    assert "desman docker install --yes" in result.output
    assert machine.calls == []


@pytest.mark.parametrize("fail", [0, 17])
def test_confirmed_cli_install(machine: Machine, fail: int) -> None:
    machine.fail = fail
    result = CliRunner().invoke(cli, ["docker", "install", "--yes"])
    assert machine.calls[0] == Call(MAC_INSTALL, None, False)
    if fail:
        assert result.exit_code == 1
        assert "Install Docker Desktop failed (exit 17)" in result.output
        assert len(machine.calls) == 1
    else:
        assert result.exit_code == 0, result.output
        assert machine.calls[1:] == [
            Call(MAC_START, None, False),
            Call(["docker", "info"], 5, False),
            Call(["docker", "--version"], 5, False),
        ]
        assert "Docker is ready." in result.output
        assert "desman sandbox setup --yes" in result.output
        assert "desman search setup --yes" in result.output


@pytest.mark.parametrize("ready", [True, False])
def test_status_json(machine: Machine, ready: bool) -> None:
    machine.installed = True
    machine.ready = ready
    result = CliRunner().invoke(cli, ["docker", "status", "--json"])
    assert result.exit_code == 0, result.output
    if not ready:
        assert "daemon unavailable é" in result.output
    assert json.loads(result.output) == {
        "installed": True,
        "available": ready,
        "version": "Docker version 29.0.0",
        "bwrap": None,
        "problem": None if ready else "daemon unavailable é",
    }


def test_bare_docker_is_status(machine: Machine) -> None:
    result = CliRunner().invoke(cli, ["docker"])
    assert result.exit_code == 0, result.output
    assert "Docker is not installed." in result.output
    assert machine.calls == []


@pytest.mark.parametrize("response", ["no", "yes", "timeout", "ready", "failure", "link"])
def test_docker_step_in_configuration(
    monkeypatch: pytest.MonkeyPatch, machine: Machine, response: str, clock: FakeClock
) -> None:
    machine.installed = machine.ready = response == "ready"
    machine.answer_on_poll = 100 if response == "timeout" else 3
    monkeypatch.setattr(
        docker_cli,
        "time",
        SimpleNamespace(sleep=clock.sleep, monotonic=lambda: clock().timestamp()),
    )
    # The provider menu checks PATH before the Docker step.
    done = str(len(providers.PROVIDERS + providers.SERVICES) + 1 + machine.installed)
    answers = ["1", done]
    if response in ("yes", "no", "failure", "timeout"):
        answers.append("n" if response == "no" else "y")
    if response == "yes":
        answers.append("y")  # SearXNG
    if response == "failure":
        machine.fail = 19
    if response == "link":
        monkeypatch.setattr(shutil, "which", lambda _: None)

    offers: list[str] = []

    def setup(manager: ConfigManager, *, save_url: bool, staged: Config, force_native: bool) -> str:
        assert not force_native
        assert not save_url
        assert not manager.load().settings.get("SEARXNG_URL")
        offers.append("search")
        staged.settings["SEARXNG_URL"] = "http://127.0.0.1:8888"
        return "http://127.0.0.1:8888"

    def probe(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        assert args[1] == "info" and machine.ready
        return subprocess.CompletedProcess(args, 0, "linux 2020-01-02T12:00:00Z", "")

    monkeypatch.setattr(wizard, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(
        sandbox,
        "Sandbox",
        lambda: Sandbox(platform="darwin", docker="docker", docker_runner=probe),
    )
    monkeypatch.setattr(sandbox_cli, "offer_setup", lambda _: offers.append("sandbox"))
    monkeypatch.setattr(searxng, "setup", setup)
    answers.extend([*KEEP_PERMISSIONS, *KEEP_PERSONAL, ""])
    result = CliRunner().invoke(cli, ["config"], input="\n".join(answers) + "\n")
    assert result.exit_code == 0, result.output
    assert ("Install Docker now?" in result.output) is (
        response in ("yes", "no", "failure", "timeout")
    )
    assert ("Docker lets DESMAN search" in result.output) is (response != "ready")
    assert ("https://docs.docker.com/desktop/setup/install/mac-install/" in result.output) is (
        response == "link"
    )
    assert [call.argv for call in machine.calls if call.timeout is None] == {
        "yes": [MAC_INSTALL, MAC_START],
        "timeout": [MAC_INSTALL, MAC_START],
        "failure": [MAC_INSTALL],
    }.get(response, [])
    assert ("Install Docker Desktop failed (exit 19)" in result.output) is (response == "failure")
    if response in ("yes", "timeout"):
        assert result.output.count("Waiting for Docker to start…") == 1
        assert clock.slept == (6 if response == "yes" else 120)
    if response == "timeout":
        assert "Once Docker Desktop is running, follow these steps." in result.output
        assert "desman sandbox setup --yes" in result.output
        assert "desman search setup --yes" in result.output
        assert "daemon did not answer" not in result.output
    assert offers == (["sandbox", "search"] if response == "yes" else [])
    assert open_config().load().settings.get("SEARXNG_URL") == (
        "http://127.0.0.1:8888" if response == "yes" else None
    )


@pytest.mark.parametrize(
    ("platform", "bundled"),
    [
        ("win32", r"D:\Apps\Docker\Docker\resources\bin\docker.exe"),
        ("darwin", "/Applications/Docker.app/Contents/Resources/bin/docker"),
    ],
)
def test_desktop_binary_before_path_refresh(
    monkeypatch: pytest.MonkeyPatch, platform: str, bundled: str
) -> None:
    monkeypatch.setattr(sandbox, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr(sandbox, "windows_program_files", lambda: r"D:\Apps")
    monkeypatch.setattr(sandbox, "_find_prlimit", lambda: None)
    monkeypatch.setattr(os.path, "isfile", lambda path: path == bundled)
    assert docker_cli.docker_executable() == bundled
    assert sandbox.Sandbox().docker == bundled


def test_linux_status_reports_bwrap_and_probe_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(docker_cli, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(shutil, "which", available_tool)

    def process(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        if argv == ["docker", "info"]:
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess(argv, 0, "Docker version 29.0.0")

    monkeypatch.setattr(docker_cli, "_process", process)
    assert docker_cli.docker_status(display=False) == Status(
        installed=True,
        available=False,
        version="Docker version 29.0.0",
        bwrap=True,
        problem="Command '['docker', 'info']' timed out after 5 seconds",
    )


def test_declining_standalone_docker_section_keeps_config(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = open_config()
    monkeypatch.setattr(ui, "confirm", lambda _question, default: default)
    wizard.run_section(manager, "docker")
    assert not manager.config_path.exists()
    assert machine.calls == []


@pytest.mark.parametrize("ready", [True, False])
def test_debian_with_docker_installs_bubblewrap_after_update(ready: bool) -> None:
    installation = plan(
        "linux",
        available_tool,
        {"ID": "debian"},
        "person",
        state(installed=True, ready=ready, bwrap=False),
    )
    expected = [
        ["sudo", "apt-get", "update"],
        ["sudo", "apt-get", "install", "-y", "bubblewrap"],
    ]
    if not ready:
        expected.append(["sudo", "systemctl", "start", "docker"])
    assert argv(installation) == expected


@pytest.mark.parametrize("user", ["person", "root"])
@pytest.mark.parametrize("bwrap", [True, False])
def test_linux_permission_denied_repairs_group_membership(
    monkeypatch: pytest.MonkeyPatch, user: str, bwrap: bool
) -> None:
    monkeypatch.setattr(docker_cli, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(shutil, "which", lambda name: name if bwrap or name != "bwrap" else None)
    error = "permission denied while trying to connect to the Docker daemon socket"

    def process(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        if argv == ["docker", "info"]:
            return subprocess.CompletedProcess(argv, 1, "", error)
        return subprocess.CompletedProcess(argv, 0, "Docker version")

    monkeypatch.setattr(docker_cli, "_process", process)
    status = docker_cli.docker_status(display=False)
    assert status["problem"] == error
    installation = plan("linux", available_tool, {"ID": "debian"}, user, status)
    sudo = [] if user == "root" else ["sudo"]
    expected = (
        []
        if bwrap
        else [[*sudo, "apt-get", "update"], [*sudo, "apt-get", "install", "-y", "bubblewrap"]]
    )
    if user != "root":
        expected.append([*sudo, "usermod", "-aG", "docker", user])
    assert argv(installation) == expected
    assert any("administrator rights" in note for note in installation.notices) == (user != "root")
    assert any("Log out and back in" in note for note in installation.after) == (user != "root")


def test_linux_wizard_offers_bubblewrap_when_docker_is_ready(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(docker_cli, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(wizard, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        docker_cli, "docker_status", lambda **_: state(installed=True, ready=True, bwrap=False)
    )
    monkeypatch.setattr(docker_cli, "freedesktop_os_release", lambda: {"ID": "debian"})
    confirmed: list[str] = []
    applied: list[Plan] = []

    def confirm(question: str, *, default: bool) -> bool:
        assert not default
        confirmed.append(question)
        return True

    def apply(installation: Plan) -> Status:
        applied.append(installation)
        return state(installed=True, ready=True, bwrap=True)

    monkeypatch.setattr(ui, "confirm", confirm)
    monkeypatch.setattr(docker_cli, "apply_plan", apply)
    wizard.run_section(open_config(), "docker")
    assert confirmed == ["Install bubblewrap now?"]
    assert len(applied) == 1
    assert argv(applied[0])[-1][-4:] == ["apt-get", "install", "-y", "bubblewrap"]
    assert "On Linux, bubblewrap runs generated code" in capsys.readouterr().out


@pytest.mark.parametrize("desktop_exists", [True, False])
def test_nonzero_winget_exit_checks_for_desktop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], desktop_exists: bool
) -> None:
    installation = plan("win32", available_tool, {}, "", state())
    calls: list[Call] = []

    def process(
        argv: list[str], *, timeout: float | None = None, detached: bool = False
    ) -> subprocess.CompletedProcess[str]:
        calls.append(Call(argv, timeout, detached))
        return subprocess.CompletedProcess(argv, 17 if argv == WINGET else 0)

    monkeypatch.setattr(docker_cli, "_process", process)
    monkeypatch.setattr(
        docker_cli,
        "os",
        SimpleNamespace(
            path=SimpleNamespace(isfile=lambda path: desktop_exists and path == WINDOWS_START[0])
        ),
    )
    monkeypatch.setattr(docker_cli, "_desktop_status", lambda *_: state(installed=True, ready=True))
    if desktop_exists:
        assert docker_cli.apply_plan(installation)["available"]
        assert calls == [Call(WINGET, None, False), Call(WINDOWS_START, None, True)]
        assert "Windows may ask you to restart." in capsys.readouterr().out
    else:
        with pytest.raises(click.ClickException, match="Install Docker Desktop failed"):
            docker_cli.apply_plan(installation)
        assert calls == [Call(WINGET, None, False)]


def test_waiting_for_desktop_reports_version_when_cli_appears(
    monkeypatch: pytest.MonkeyPatch, machine: Machine, clock: FakeClock
) -> None:
    installation = plan("darwin", available_tool, {}, "", state())
    installation.steps = installation.steps[1:]

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        machine.installed = True

    result = docker_cli.apply_plan(installation, sleep=sleep, clock=lambda: clock().timestamp())
    assert result["available"]
    assert result["version"] == "Docker version 29.0.0"
