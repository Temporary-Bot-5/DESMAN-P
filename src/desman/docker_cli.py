"""Inspect Docker and plan or perform installation with the operator's consent."""

import contextlib
import json
import ntpath
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from platform import freedesktop_os_release
from typing import TypedDict

import click

from desman import ui
from desman.config_cli import json_option
from desman.core.sandbox import find_docker, windows_program_files
from desman.i18n import t

docker_executable = find_docker

INSTALL_COMMAND = "desman docker install --yes"
DESKTOP_ROOT = "https://docs.docker.com/desktop/setup/install/"
ENGINE_LINK = "https://docs.docker.com/engine/install/"


class Status(TypedDict):
    installed: bool
    available: bool
    version: str | None
    bwrap: bool | None
    problem: str | None


@dataclass
class Step:
    argv: list[str]
    description: str
    detached: bool = False


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    needs_admin: bool = False
    may_reboot: bool = False
    download: str | None = None
    license: str | None = None
    link: str | None = None
    notices: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    ready: bool = False
    windows: bool = False
    starts_desktop: bool = False

    def lines(self, *, include_after: bool = True) -> list[str]:
        lines = [t("Docker is ready.")] if self.ready else []
        if self.steps:
            # Render arguments with the quoting conventions of the target platform.
            quote = subprocess.list2cmdline if self.windows else shlex.join
            for step in self.steps:
                lines.extend([step.description, quote(step.argv)])
        if self.needs_admin:
            lines.append(t("Administrator rights are required."))
        if self.download:
            lines.append(t("Download: {size}", size=self.download))
        if self.may_reboot:
            lines.append(t("Windows may ask you to restart."))
        if self.license:
            lines.append(self.license)
        if self.link:
            lines.append(self.link)
        lines.extend(self.notices)
        return [*lines, *self.after] if include_after else lines


def _desktop_license() -> str:
    return t(
        "Docker Desktop is free for personal use, education, non-commercial open source, "
        "and small businesses with fewer than 250 employees and less than $10 million in "
        "annual revenue. Other use requires a paid subscription."
    )


def _linux_packages(
    os_release: Mapping[str, str],
) -> tuple[list[str], str | None, list[str]] | None:
    distro = os_release.get("ID", "")
    families = {distro, *os_release.get("ID_LIKE", "").split()}
    if distro == "fedora":
        return ["dnf", "install", "-y"], "moby-engine", []
    if distro in {"rhel", "rocky", "almalinux", "centos"} or families & {"rhel", "centos"}:
        return ["dnf", "install", "-y"], None, []
    if families & {"debian", "ubuntu"}:
        return ["apt-get", "install", "-y"], "docker.io", ["apt-get", "update"]
    if "fedora" in families:
        return ["dnf", "install", "-y"], "moby-engine", []
    if "arch" in families:
        return ["pacman", "-S", "--needed", "--noconfirm"], "docker", []
    if families & {"opensuse", "suse"}:
        return ["zypper", "--non-interactive", "install"], "docker", []
    return None


def plan(
    platform: str,
    which: Callable[[str], str | None],
    os_release: Mapping[str, str],
    user: str,
    docker_state: Status,
    *,
    program_files: str = r"C:\Program Files",
) -> Plan:
    """Build an installation plan using only the supplied machine state."""
    ready = docker_state["available"]
    installed = docker_state["installed"]
    linux = platform.startswith("linux")
    windows = platform == "win32"
    result = Plan(ready=ready, windows=windows)
    if ready and (not linux or docker_state["bwrap"]):
        return result

    next_steps = [t("Next: `desman search setup --yes`.")]
    if windows or platform == "darwin":
        next_steps.insert(0, t("For the sandbox, run `desman sandbox setup --yes`."))
        start = (
            [ntpath.join(program_files, "Docker", "Docker", "Docker Desktop.exe")]
            if windows
            else ["open", "-a", "Docker"]
        )
        if not installed:
            manager = "winget" if windows else "brew"
            if which(manager) is None:
                result.link = DESKTOP_ROOT + ("windows-install/" if windows else "mac-install/")
                return result
            install = (
                [
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
                if windows
                else ["brew", "install", "--cask", "docker-desktop"]
            )
            result.steps.append(Step(install, t("Install Docker Desktop")))
            result.needs_admin = True
            result.may_reboot = windows
            result.download = t("about 600 MB")
            result.license = _desktop_license()
        result.steps.append(Step(start, t("Start Docker Desktop"), detached=windows))
        result.starts_desktop = True
        result.after = [
            t("Accept the Docker Desktop agreement in its window if prompted."),
            *next_steps,
        ]
        return result

    if not linux:
        result.link = ENGINE_LINK
        return result

    sudo = [] if user == "root" else ["sudo"]
    packages = _linux_packages(os_release)
    needs_engine = not installed and not ready
    needs_bwrap = not docker_state["bwrap"]
    engine = packages[1] if packages is not None else None
    if needs_engine and engine is None:
        result.link = ENGINE_LINK

    names = [engine] if needs_engine and engine is not None else []
    if needs_bwrap and packages is not None:
        names.append("bubblewrap")
    if packages is not None and names:
        install, _, update = packages
        if update:
            result.steps.append(Step([*sudo, *update], t("Update the package list")))
        description = (
            t("Install the Docker packages") if needs_engine and engine else t("Install bubblewrap")
        )
        result.steps.append(Step([*sudo, *install, *names], description))
        result.download = t("varies by distribution")

    if needs_engine and engine is not None:
        result.steps.append(
            Step([*sudo, "systemctl", "enable", "--now", "docker"], t("Enable and start Docker"))
        )
        _add_docker_group(result, sudo, user)
    elif _group_access_needed(docker_state):
        _add_docker_group(result, sudo, user)
    elif installed and not ready:
        result.steps.append(Step([*sudo, "systemctl", "start", "docker"], t("Start Docker")))
    if result.steps:
        result.needs_admin = True
        result.after.extend(next_steps)
    return result


def _group_access_needed(state: Status) -> bool:
    return state["installed"] and "permission denied" in (state["problem"] or "").casefold()


def _add_docker_group(installation: Plan, sudo: list[str], user: str) -> None:
    if user == "root":
        return
    installation.steps.append(
        Step([*sudo, "usermod", "-aG", "docker", user], t("Add the user to the docker group"))
    )
    installation.needs_admin = True
    installation.notices.append(
        t("Membership in the docker group gives administrator rights on this computer.")
    )
    installation.after.append(t("Log out and back in for the group membership to take effect."))


def _process(
    argv: list[str], *, timeout: float | None = None, detached: bool = False
) -> subprocess.CompletedProcess[str]:
    if detached and sys.platform == "win32":
        subprocess.Popen(
            argv, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        return subprocess.CompletedProcess(argv, 0)
    if timeout is not None:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    return subprocess.run(argv, check=False, text=True)


def _probe(argv: list[str], *, timeout: float = 5) -> tuple[str | None, str | None]:
    try:
        result = _process(argv, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode == 0:
        return (result.stdout or "").strip(), None
    return None, (result.stderr or result.stdout or "").strip() or None


def docker_status(*, display: bool = True) -> Status:
    executable = docker_executable()
    info, problem = _probe([executable, "info"]) if executable is not None else (None, None)
    result = Status(
        installed=executable is not None,
        available=info is not None,
        version=_probe([executable, "--version"])[0] if executable is not None else None,
        bwrap=shutil.which("bwrap") is not None if sys.platform.startswith("linux") else None,
        problem=problem,
    )
    if display:
        _show_status(result)
    return result


def _show_status(state: Status) -> None:
    if not state["installed"]:
        ui.note(t("Docker is not installed."))
    elif state["available"]:
        ui.ok(t("Docker is ready."))
    else:
        ui.warn(t("Docker is installed, but the daemon did not answer."))
    if state["version"]:
        ui.note(state["version"])
    if state["bwrap"] is not None:
        ui.note(t("bubblewrap is installed.") if state["bwrap"] else t("bubblewrap is missing."))


def current_plan(state: Status | None = None) -> Plan:
    state = docker_status(display=False) if state is None else state
    release: Mapping[str, str] = {}
    user = ""
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError):
            release = freedesktop_os_release()
        # Read the account from the OS before granting docker group membership.
        if os.name == "posix":
            import pwd

            uid = os.getuid()
            packages = _linux_packages(release)
            if uid == 0:
                user = "root"
            elif _group_access_needed(state) or (
                not state["installed"] and not state["available"] and packages and packages[1]
            ):
                with contextlib.suppress(KeyError):
                    user = pwd.getpwuid(uid).pw_name
                if not user or user.startswith("-"):
                    raise click.ClickException(
                        t("Cannot determine a valid Linux user for the docker group.")
                    )
    return plan(
        sys.platform,
        shutil.which,
        release,
        user,
        state,
        program_files=windows_program_files(),
    )


def show_plan(installation: Plan, *, include_after: bool = True) -> None:
    for line in installation.lines(include_after=include_after):
        ui.note(line)


def apply_plan(
    installation: Plan,
    *,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Status:
    """Run the confirmed steps in order, then check whether the daemon answers."""
    for step in installation.steps:
        try:
            result = _process(step.argv, detached=step.detached)
        except OSError as exc:
            raise click.ClickException(
                t("{description} failed: {error}", description=step.description, error=str(exc))
            ) from exc
        if result.returncode:
            if (
                installation.windows
                and step.argv[:2] == ["winget", "install"]
                and os.path.isfile(installation.steps[-1].argv[0])
            ):
                ui.note(t("Windows may ask you to restart."))
                continue
            raise click.ClickException(
                t(
                    "{description} failed (exit {code})",
                    description=step.description,
                    code=result.returncode,
                )
            )
    if installation.starts_desktop:
        state = _desktop_status(sleep or time.sleep, clock or time.monotonic)
        if state["available"]:
            _show_status(state)
        else:
            ui.note(t("Once Docker Desktop is running, follow these steps."))
    else:
        state = docker_status()
    if installation.may_reboot:
        ui.note(t("Windows may ask you to restart."))
    for line in installation.after:
        ui.note(line)
    return state


def _desktop_status(sleep: Callable[[float], None], clock: Callable[[], float]) -> Status:
    deadline = clock() + 120
    state = docker_status(display=False)
    if state["available"]:
        return state
    ui.note(t("Waiting for Docker to start…"))
    while (remaining := deadline - clock()) > 0:
        sleep(min(3, remaining))
        executable = docker_executable()
        if executable is None:
            continue
        # Allow the last check at the deadline without adding another full probe timeout.
        info, problem = _probe([executable, "info"], timeout=max(0.001, min(5, deadline - clock())))
        state.update({"installed": True, "available": info is not None, "problem": problem})
        if state["available"]:
            state["version"] = _probe([executable, "--version"])[0]
            break
    return state


@click.group(invoke_without_command=True)
@click.pass_context
def docker(ctx: click.Context) -> None:
    """Manage the Docker installation."""
    if ctx.invoked_subcommand is None:
        docker_status()


@docker.command()
@json_option
def status(as_json: bool) -> None:
    """Show Docker installation and daemon status."""
    result = docker_status(display=not as_json)
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))


@docker.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def install(yes: bool) -> None:
    """Plan or install Docker."""
    installation = current_plan()
    if yes and installation.steps:
        show_plan(installation, include_after=False)
        apply_plan(installation)
    else:
        show_plan(installation)
        if installation.steps:
            ui.note(t("Nothing changed. Confirm with: {command}", command=INSTALL_COMMAND))
