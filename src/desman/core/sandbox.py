"""Sandbox for generated code (ROADMAP 3.3; B5, second part).

The old prototype ran generated code with the network and the whole file system in reach. Here
a command runs inside bubblewrap (`bwrap`) on Linux, or Docker or WebAssembly on Windows
and macOS, with bubblewrap in WSL2 also available on Windows (ADRs 0012 and 0013).

Bubblewrap provides:

- every namespace is new (`--unshare-all`): no network (only a loopback of its own), no other
  processes, no IPC; it cannot make namespaces of its own (`--disable-userns`) and has no
  capabilities;
- the file system is the system's programs and libraries (`/usr` and the links into it),
  read-only, a small private `/tmp`, and the task's workspace at `/workspace` (the working
  directory, and `HOME`). The person's home, the configuration, the database and the policy
  are not there at all;
- the environment is cleared (`--clearenv`): no API key or anything else of the caller's
  reaches the code;
- resource limits: CPU time, address space, size of a written file, open files, no core dumps,
  set before `bwrap` starts; the number of processes is limited inside the sandbox's own user
  namespace (`prlimit`), since there it counts only the sandbox's processes;
- a wall-clock timeout: the whole process group is killed, and the sandbox's processes die with
  it (`--die-with-parent`, the PID namespace goes with its first process);
- only so much output is kept (the start of stdout, the end of stderr, where a traceback ends);
  the rest is read and dropped, so a chatty program cannot fill memory.

When `bwrap` (or `prlimit`) is missing or cannot create namespaces (some distributions forbid
it to unprivileged users), the sandbox is *unavailable*: `run` refuses, and the tool that needs
it is not offered (`Tool.ready`). There is no fallback to running code unconfined.

Only pytest and its runtime imports (pytest, _pytest, pluggy, iniconfig, packaging, pygments,
and py.py when present) are bound read-only from the running interpreter under
/opt/desman/python, so generated tests can run without exposing other site-packages.

A standalone Python tree holds only the interpreter and its standard library. When
DESMAN uses one outside /usr, it is bound at /opt/desman/runtime so pytest runs with
the Python version it was installed for.

Known limits: the workspace is the disk the code may fill (each file is capped, the number of
files is not); `/usr` is readable (system files only, nothing of the person's).
"""

from __future__ import annotations

import configparser
import contextlib
import csv
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import io
import math
import ntpath
import os
import posixpath
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import IO, Final

import httpx
from pydantic import BaseModel

from desman.core import filelock
from desman.core.db import default_db_path

_WINDOWS_PROCESS_GROUP: Final = 0x00000200  # CREATE_NEW_PROCESS_GROUP
DOCKER_IMAGE: Final = (
    "python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
)
DOCKER_LABEL: Final = "desman.sandbox=1"
_DOCKER_TIMEOUT: Final = 5.0
_DOCKER_NAME: Final = "desman-sbx-"
_PYTHON_PATH: Final = "/opt/desman/python"
DockerRunner = Callable[[list[str], float], subprocess.CompletedProcess[str]]

WSL_DISTRO: Final = "desman-sandbox"
WSL_PIN: Final = "alpine-minirootfs-3.22.8-x86_64"
_WSL_URL: Final = f"https://dl-cdn.alpinelinux.org/alpine/v3.22/releases/x86_64/{WSL_PIN}.tar.gz"
_WSL_SHA256: Final = "2d7f46af9b7e86ad64ebeba3fe9b75b62dd5b259e708afb2ab21268f627696e5"
_WSL_MARKER: Final = """printf '%s\\n' "$1" > /etc/desman-sandbox"""
_WSL_SETUP: Final = "Run `desman sandbox setup`."
_WSL_REPAIR: Final = "run `desman sandbox setup --wsl --yes` to repair it"
_WSL_CONF: Final = ("[interop]", "enabled=false", "appendWindowsPath=false")
_WSL_ROOT: Final = ("-d", WSL_DISTRO, "-u", "root", "--cd", "/", "--exec")
WslRunner = Callable[[list[str], float, bool], subprocess.CompletedProcess[bytes]]

MOUNT_POINT: Final = "/workspace"
SANDBOX_PATH: Final = "/usr/local/bin:/usr/bin:/bin"
_DOCKER_ENV: Final = {
    "PATH": SANDBOX_PATH,
    "HOME": MOUNT_POINT,
    "TMPDIR": "/tmp",
    "LANG": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_ADDOPTS": "-p no:cacheprovider",
    "PYTHONPATH": _PYTHON_PATH,
}
# Docker also supplies image defaults and can inject client proxy credentials.
# Start with isolated Python, then replace it with the command and just the fixed environment.
_DOCKER_EXEC: Final = f"""\
import os
import sys

command = sys.argv[1:]
try:
    os.execvpe(command[0], command, {{key: os.environ[key] for key in {tuple(_DOCKER_ENV)!r}}})
except OSError as error:
    print(error, file=sys.stderr)
    sys.exit(127 if isinstance(error, FileNotFoundError) else 126)
"""
_SYSTEM_LINKS: Final = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
"""Top-level directories that are links into /usr on a merged-/usr system, or real
directories (bound read-only) on an older one."""
_SYSTEM_FILES: Final = (
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/alternatives",
    "/etc/localtime",
)
"""What the dynamic loader and common programs need from /etc; nothing else of it is shown."""
_READ_CHUNK: Final = 65_536


class SandboxUnavailable(RuntimeError):
    """The sandbox cannot run here: no command runs at all."""


class WslImportUnavailable(SandboxUnavailable):
    """WSL could not import its sandbox distribution."""


class _DockerDaemonUnavailable(SandboxUnavailable):
    """A Docker daemon that may recover after startup or a restart."""


@dataclass(frozen=True)
class Limits:
    timeout: float = 60.0
    """Wall-clock seconds for the whole run; the most a caller may ask for in `run`."""
    cpu_seconds: int = 60
    memory_bytes: int = 1024 * 2**20
    """Address space of each process."""
    file_bytes: int = 16 * 2**20
    """Largest file the code may write."""
    tmp_bytes: int = 64 * 2**20
    """Size of the private /tmp (kept in memory)."""
    processes: int = 32
    open_files: int = 256
    stdout_chars: int = 6_000
    stderr_chars: int = 2_000


@dataclass(frozen=True)
class Outcome:
    exit_code: int | None
    """None after a timeout. 128+N: killed by signal N (e.g. a CPU limit)."""
    stdout: str
    stderr: str
    timed_out: bool = False
    stdout_cut: bool = False
    stderr_cut: bool = False


def windows_program_files() -> str:
    return os.environ.get("PROGRAMW6432") or os.environ.get("PROGRAMFILES", r"C:\Program Files")


def find_docker() -> str | None:
    """Find the CLI, including Docker Desktop before the current PATH is refreshed."""
    executable = shutil.which("docker")
    if executable is not None:
        return executable
    bundled: str | None = None
    if sys.platform == "win32":
        bundled = ntpath.join(
            windows_program_files(), "Docker", "Docker", "resources", "bin", "docker.exe"
        )
    elif sys.platform == "darwin":
        bundled = "/Applications/Docker.app/Contents/Resources/bin/docker"
    return bundled if bundled is not None and os.path.isfile(bundled) else None


def docker_env(executable: str) -> dict[str, str]:
    """Let the Docker CLI find credential helpers shipped beside it."""
    windows = sys.platform == "win32"
    paths = ntpath if windows else posixpath
    separator = ";" if windows else ":"
    env = dict(os.environ)
    directory = paths.dirname(executable)
    if directory:
        entries = [entry for entry in env.get("PATH", paths.defpath).split(separator) if entry]
        normalized = paths.normcase(paths.abspath(directory))
        if not any(paths.normcase(paths.abspath(entry)) == normalized for entry in entries):
            entries.insert(0, directory)
        env["PATH"] = separator.join(entries)
    return env


def _system_wsl() -> str:
    system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return ntpath.join(system_root, "System32", "wsl.exe")


def find_wsl() -> str | None:
    executable = _system_wsl()
    return executable if os.path.isfile(executable) else shutil.which("wsl.exe")


def _wsl_workspace(workdir: Path) -> str:
    path = PureWindowsPath(workdir)
    if not re.fullmatch("[A-Za-z]:", path.drive) or not path.root or ".." in path.parts:
        raise SandboxUnavailable("WSL needs a workspace on a local Windows drive.")
    return "/".join(("/mnt", path.drive[0].lower(), *path.parts[1:]))


def _wsl_env() -> dict[str, str]:
    keys = ("SystemRoot", "PATH", "USERPROFILE", "LOCALAPPDATA", "TEMP", "TMP")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _decode_wsl(data: bytes | bytearray) -> str:
    if data.startswith(b"\xff\xfe"):
        return data[2:].decode("utf-16-le", errors="replace")
    odd_nuls = data[1::2].count(0)
    even_nuls = data[::2].count(0)
    if odd_nuls >= 2 and odd_nuls > 2 * even_nuls:
        return data.decode("utf-16-le", errors="replace")
    return data.decode("utf-8", errors="replace")


def _wsl_detail(result: subprocess.CompletedProcess[bytes]) -> str:
    return _decode_wsl(result.stderr or result.stdout or b"").strip() or str(result.returncode)


def _wsl_interop_disabled(text: str) -> bool:
    conf = configparser.ConfigParser(interpolation=None, delimiters=("=",))
    conf.optionxform = lambda optionstr: optionstr  # type: ignore[method-assign]
    try:
        conf.read_string(text)
    except configparser.Error:
        return False
    return not conf.defaults() and conf.get("interop", "enabled", fallback=None) == "false"


def _wsl_run(argv: list[str], timeout: float, terminal: bool) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        argv,
        timeout=timeout,
        env=_wsl_env(),
        capture_output=not terminal,
        stdin=None if terminal else subprocess.DEVNULL,
        creationflags=0 if terminal else _WINDOWS_PROCESS_GROUP,
    )


class Sandbox:
    """Runs confined commands. Caches availability and retries transient Docker probe failures."""

    def __init__(
        self,
        limits: Limits | None = None,
        *,
        bwrap: str | None = None,
        prlimit: str | None = None,
        docker: str | None = None,
        platform: str | None = None,
        docker_runner: DockerRunner | None = None,
        wasi_dir: Path | None = None,
        wsl: str | None = None,
        wsl_runner: WslRunner | None = None,
        wsl_dir: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits if limits is not None else Limits()
        self.bwrap = bwrap if bwrap is not None else shutil.which("bwrap")
        self.prlimit = prlimit if prlimit is not None else _find_prlimit()
        self.platform = sys.platform if platform is None else platform
        self.docker = docker
        if self.docker is None and not self.platform.startswith("linux"):
            self.docker = find_docker()
        self.wasi_dir = (
            wasi_dir if wasi_dir is not None else default_db_path().parent / "wasi" / _WASI_PIN
        )
        self.wsl = find_wsl() if self.platform == "win32" and wsl is None else wsl
        self.wsl_dir = wsl_dir if wsl_dir is not None else default_db_path().parent / "wsl"
        self._wsl_runner = _wsl_run if wsl_runner is None else wsl_runner
        self.wsl_present: bool | None = None
        self._wsl_issue: str | None = None
        self._backend: str | None = None
        self._docker_issue: str | None = None
        self._docker_checked_at: float | None = None
        self._docker_time = ""
        self._docker_runner = docker_runner if docker_runner is not None else _docker_run
        self.image_present: bool | None = None
        self._problem: str | None = None
        self._checked = False
        self._monotonic = monotonic
        self._retry_at: float | None = None

    @property
    def backend(self) -> str:
        if self._backend is None:
            if self.platform.startswith("linux"):
                self._backend = "bwrap"
            else:
                problem = self.docker_problem()
                if problem is None or self.platform not in ("win32", "darwin"):
                    self._backend = "docker"
                elif self.platform == "win32" and self._probe_wsl():
                    self._backend = "wsl"
                else:
                    self._backend = "wasi"
        return self._backend

    @property
    def wasi_present(self) -> bool:
        return _wasi_files(self.wasi_dir) and (self.wasi_dir / "ok").is_file()

    def _probe_wasi(self) -> str | None:
        if self.wasi_present and wasi_runtime():
            return None
        details = " ".join(issue for issue in (self._docker_issue, self._wsl_issue) if issue)
        return (
            f"{details} "
            "Run `desman docker install` or set up WebAssembly with `desman sandbox setup`."
        )

    def unavailable(self) -> str | None:
        """Why no command can run here, or None if the sandbox works."""
        if self._retry_at is not None and self._monotonic() >= self._retry_at:
            return self.reprobe()
        if not self._checked:
            try:
                if self.backend == "bwrap":
                    self._problem = self._probe()
                elif self.backend == "wasi":
                    self._problem = self._probe_wasi()
                    if self.docker or self._problem is not None:
                        self._retry_at = self._monotonic() + 60.0
                elif self.backend == "wsl":
                    self._retry_at = self._monotonic() + 60.0
                else:
                    self._problem = self._probe_docker()
            except _DockerDaemonUnavailable as exc:
                self._problem = str(exc)
                self._retry_at = self._monotonic() + 60.0
            if self._problem is not None and self.backend == "docker":
                if self.platform == "win32" and self._probe_wsl():
                    self._backend = "wsl"
                elif self.platform in ("win32", "darwin") and self.wasi_present and wasi_runtime():
                    self._backend = "wasi"
                if self._backend != "docker":
                    self._problem = None
                    self.image_present = None
                    self._retry_at = self._monotonic() + 60.0
                elif self._wsl_issue is not None:
                    self._problem += " " + self._wsl_issue
            self._checked = True
        return self._problem

    def reprobe(self) -> str | None:
        """Refresh availability after the operator changes the sandbox setup."""
        self._checked = False
        self._backend = None
        self._docker_checked_at = None
        self.image_present = None
        self.wsl_present = None
        self._wsl_issue = None
        self._problem = None
        self._retry_at = None
        return self.unavailable()

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        stdin: str = "",
        timeout: float | None = None,
    ) -> Outcome:
        """Run `command` with `workdir` as /workspace. Raises SandboxUnavailable."""
        problem = self.unavailable()
        if problem is not None:
            raise SandboxUnavailable(problem)
        limits = self.limits
        if timeout is not None:
            limits = _with(limits, min(timeout, limits.timeout))
        if self.backend == "docker":
            name = _DOCKER_NAME + uuid.uuid4().hex
            try:
                return _execute_docker(
                    self._docker_argv(command, workdir, name),
                    stdin,
                    limits,
                    lambda: self._stop_docker(name),
                    lambda: self._docker_exit(name),
                )
            finally:
                self._remove_docker(name)
        if self.backend == "wasi":
            return _execute_wasi(command, workdir, stdin, limits, self.wasi_dir)
        if self.backend == "wsl":
            name = f"desman-{uuid.uuid4().hex}"
            argv = self._wsl_argv(command, workdir, limits.timeout, hostname=name)
            try:
                return _execute_wsl(argv, stdin, limits, lambda: self._stop_wsl(name))
            except _WslSuspect as suspect:
                # The code itself may print what looks like a WSL error: only a failing probe
                # makes it the sandbox's failure.
                problem = self.wsl_problem()
                if problem is None:
                    return suspect.outcome
                self._retry_at = self._monotonic()
                raise SandboxUnavailable(f"WSL failed: {problem}") from None
            except SandboxUnavailable:
                self._retry_at = self._monotonic()
                raise
        return _execute(self._argv(command, workdir), stdin, limits)

    def _wsl_argv(
        self,
        command: Sequence[str],
        workdir: Path,
        timeout: float | None = None,
        *,
        hostname: str | None = None,
    ) -> list[str]:
        assert self.wsl
        mount = _wsl_workspace(workdir)
        limits = self.limits
        seconds = limits.timeout if timeout is None else timeout
        argv = [
            self.wsl,
            "-d",
            WSL_DISTRO,
            "-u",
            "sandbox",
            "--cd",
            "/",
            "--exec",
            "/bin/busybox",
            "timeout",
            "-s",
            "KILL",
            str(math.ceil(seconds) + 2),
            "/usr/bin/bwrap",
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        if hostname is not None:
            argv += ["--hostname", hostname]
        for key, value in _DOCKER_ENV.items():
            if key != "PYTHONPATH":
                argv += ["--setenv", key, "/usr/bin:/bin" if key == "PATH" else value]
        for path in ("/usr", "/bin", "/lib"):
            argv += ["--ro-bind", path, path]
        return [
            *argv,
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--size",
            str(limits.tmp_bytes),
            "--tmpfs",
            "/tmp",
            "--bind",
            mount,
            MOUNT_POINT,
            "--chdir",
            MOUNT_POINT,
            "--",
            "/usr/bin/prlimit",
            f"--nproc={limits.processes}",
            f"--as={limits.memory_bytes}",
            f"--fsize={limits.file_bytes}",
            f"--nofile={limits.open_files}",
            f"--cpu={limits.cpu_seconds}",
            "--core=0",
            "--",
            *command,
        ]

    def wsl_command(
        self, args: list[str], timeout: float = 60.0, *, terminal: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        executable = self.wsl
        if not executable and args[0] == "--install":
            executable = _system_wsl()
        if not executable:
            raise SandboxUnavailable(f"WSL is missing. {_WSL_SETUP}")
        try:
            return self._wsl_runner([executable, *args], timeout, terminal)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(str(exc)) from exc

    def _stop_wsl(self, name: str) -> None:
        with contextlib.suppress(SandboxUnavailable):
            self.wsl_command([*_WSL_ROOT, "/usr/bin/pkill", "-KILL", "-f", name], 5.0)

    def wsl_usable(self) -> bool:
        if self.platform != "win32" or not self.wsl:
            return False
        try:
            result = self.wsl_command(["--status"])
            status = "\n".join(_decode_wsl(data or b"") for data in (result.stdout, result.stderr))
            unavailable = re.search(
                r"(?:WSL\s*2|WSL version 2)\s+is\s+"
                r"(?:not (?:supported|available|enabled)|unavailable|disabled)",
                status,
                re.IGNORECASE,
            )
            return result.returncode == 0 and unavailable is None
        except SandboxUnavailable:
            return False

    def wsl_distribution_present(self) -> bool:
        if not self.wsl:
            self.wsl_present = False
            return False
        result = self.wsl_command(["--list", "--quiet"])
        if result.returncode and not self.wsl_usable():
            raise SandboxUnavailable(f"WSL is unavailable. {_WSL_SETUP}")
        self.wsl_present = result.returncode == 0 and WSL_DISTRO in {
            name.strip().casefold() for name in _decode_wsl(result.stdout).splitlines()
        }
        return self.wsl_present

    def _probe_wsl(self) -> bool:
        self._wsl_issue = self.wsl_problem()
        return self._wsl_issue is None

    def wsl_problem(self) -> str | None:
        try:
            if not self.wsl:
                return f"WSL is missing. {_WSL_SETUP}"
            if not self.wsl_distribution_present():
                return f"The WSL sandbox distribution is missing. {_WSL_SETUP}"
            try:
                conf = self.wsl_command([*_WSL_ROOT, "/bin/cat", "/etc/wsl.conf"])
                hardened = conf.returncode == 0 and _wsl_interop_disabled(
                    _decode_wsl(conf.stdout or b"")
                )
            except SandboxUnavailable:
                hardened = False
            if not hardened:
                return f"The WSL sandbox is not hardened (interop is on); {_WSL_REPAIR}"
            marker = self.wsl_command([*_WSL_ROOT, "/bin/cat", "/etc/desman-sandbox"])
            fields = _decode_wsl(marker.stdout or b"").split()
            if marker.returncode or not fields or fields[0] != WSL_PIN:
                return f"The WSL sandbox distribution pin is missing or outdated; {_WSL_REPAIR}"
            with tempfile.TemporaryDirectory(prefix="desman-probe-") as empty:
                _, *args = self._wsl_argv(["true"], Path(empty), 60.0)
                result = self.wsl_command(args)
            if result.returncode == 0:
                return None
            detail = _wsl_detail(result)
        except (SandboxUnavailable, OSError) as exc:
            detail = str(exc)
        return f"The WSL sandbox could not start: {detail}; {_WSL_REPAIR}"

    def _argv(self, command: Sequence[str], workdir: Path) -> list[str]:
        assert self.bwrap is not None and self.prlimit is not None
        argv = [
            self.bwrap,
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTEST_ADDOPTS",
            "-p no:cacheprovider",
            "--setenv",
            "PATH",
            SANDBOX_PATH,
            "--setenv",
            "HOME",
            MOUNT_POINT,
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        runtime = Path(sys.base_prefix).resolve()
        if not runtime.is_relative_to("/usr"):
            target = "/opt/desman/runtime"
            argv += ["--ro-bind", str(runtime), target]
            argv += ["--setenv", "PATH", target + "/bin:" + SANDBOX_PATH]
            # An interpreter linked to its own libpython (setup-python's builds) finds it by
            # the path it was built at, which is not there: point the loader at the bind.
            argv += ["--setenv", "LD_LIBRARY_PATH", target + "/lib"]
            if not (runtime / "bin/python3").exists():
                # A new bin mount lets bwrap add the alias without writing to the
                # read-only interpreter tree. Keep every original entry read-only.
                argv += ["--tmpfs", target + "/bin"]
                for executable in sorted((runtime / "bin").iterdir()):
                    argv += ["--ro-bind", str(executable), target + "/bin/" + executable.name]
                argv += [
                    "--symlink",
                    f"python{sys.version_info.major}.{sys.version_info.minor}",
                    target + "/bin/python3",
                ]
        for name in _SYSTEM_LINKS:
            path = Path(name)
            if path.is_symlink():
                argv += ["--symlink", os.readlink(path), name]
            elif path.is_dir():
                argv += ["--ro-bind", name, name]
        for name in _SYSTEM_FILES:
            argv += ["--ro-bind-try", name, name]
        argv += ["--setenv", "PYTHONPATH", "/opt/desman/python"]
        for source, target in _pytest_mounts():
            argv += ["--ro-bind", str(source), target]
        argv += [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--size",
            str(self.limits.tmp_bytes),
            "--tmpfs",
            "/tmp",
            "--bind",
            str(workdir),
            MOUNT_POINT,
            "--chdir",
            MOUNT_POINT,
            "--",
            self.prlimit,
            f"--nproc={self.limits.processes}",
            "--",
            *command,
        ]
        return argv

    def _probe(self) -> str | None:
        if self.bwrap is None:
            return "bubblewrap (bwrap) is not installed"
        if self.prlimit is None:
            return "prlimit (util-linux) is not installed"
        with tempfile.TemporaryDirectory(prefix="desman-probe-") as empty:
            try:
                outcome = _execute(self._argv(["true"], Path(empty)), "", _with(self.limits, 10.0))
            except OSError as e:
                return f"bwrap cannot be started: {e}"
        if outcome.exit_code != 0:
            detail = outcome.stderr.strip() or f"exit code {outcome.exit_code}"
            return f"bwrap cannot create a sandbox here: {detail}"
        return None

    def docker_command(
        self, args: list[str], timeout: float = _DOCKER_TIMEOUT
    ) -> subprocess.CompletedProcess[str]:
        """Run a DESMAN-owned Docker command with fixed arguments."""
        if not self.docker:
            raise SandboxUnavailable("Docker is not installed or is not on PATH.")
        try:
            return self._docker_runner([self.docker, *args], timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(str(exc)) from exc

    def docker_problem(self) -> str | None:
        if self._docker_checked_at is None or self._monotonic() - self._docker_checked_at >= 60:
            try:
                self._docker_issue = self._docker_problem()
            except _DockerDaemonUnavailable as exc:
                self._docker_issue = str(exc)
            self._docker_checked_at = self._monotonic()
        return self._docker_issue

    def _docker_problem(self) -> str | None:
        if not self.docker:
            return "Docker is not installed or is not on PATH."
        try:
            result = self.docker_command(["info", "--format", "{{.OSType}} {{.SystemTime}}"])
        except SandboxUnavailable as exc:
            raise _DockerDaemonUnavailable(f"Docker daemon did not answer: {exc}") from exc
        if result.returncode:
            raise _DockerDaemonUnavailable(
                f"Docker daemon did not answer: {_docker_detail(result)}"
            )
        os_type, _, self._docker_time = result.stdout.strip().partition(" ")
        if os_type != "linux":
            return "Docker runs Windows containers: switch Docker Desktop to Linux containers."
        return None

    def _docker_argv(self, command: Sequence[str], workdir: Path, name: str) -> list[str]:
        assert self.docker
        limits = self.limits
        cpu = max(1, limits.cpu_seconds)
        if self.platform == "win32" or sys.platform == "win32":
            user = "65534:65534"
        else:
            user = f"{os.getuid()}:{os.getgid()}"
        argv = [
            self.docker,
            "run",
            "--init",
            "--pull",
            "never",
            "--name",
            name,
            "--label",
            DOCKER_LABEL,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            user,
            "--pids-limit",
            str(limits.processes),
            "--memory",
            str(limits.memory_bytes),
            "--memory-swap",
            str(limits.memory_bytes),
            "--cpus",
            "1",
            "--ulimit",
            f"cpu={cpu}:{cpu + 1}",
            "--ulimit",
            f"fsize={limits.file_bytes}",
            "--ulimit",
            f"nofile={limits.open_files}",
            "--ulimit",
            "core=0",
            "--tmpfs",
            f"/tmp:size={limits.tmp_bytes}",
            "--mount",
            _docker_mount(workdir, MOUNT_POINT),
            "-w",
            MOUNT_POINT,
            "-i",
        ]
        for key, value in _DOCKER_ENV.items():
            argv += ["-e", f"{key}={value}"]
        for source, target in _pytest_mounts():
            argv += ["--mount", _docker_mount(source, target, readonly=True)]
        return [
            *argv,
            "--entrypoint",
            "/usr/local/bin/python3",
            DOCKER_IMAGE,
            "-I",
            "-B",
            "-c",
            _DOCKER_EXEC,
            *command,
        ]

    def docker_image_present(self) -> bool:
        image = self.docker_command(["image", "inspect", DOCKER_IMAGE])
        if image.returncode and "no such image" not in image.stderr.lower():
            raise _DockerDaemonUnavailable(_docker_detail(image))
        return image.returncode == 0

    def _probe_docker(self) -> str | None:
        if self._docker_issue is not None:
            return self._docker_issue
        try:
            self.image_present = self.docker_image_present()
            if not self.image_present:
                return "The sandbox image is missing. Run `desman sandbox setup`."
            self._sweep_docker()
            with tempfile.TemporaryDirectory(prefix="desman-probe-") as empty:
                name = _DOCKER_NAME + uuid.uuid4().hex
                try:
                    result = self.docker_command(
                        self._docker_argv(["true"], Path(empty), name)[1:], timeout=60.0
                    )
                except (SandboxUnavailable, KeyboardInterrupt):
                    self._stop_docker(name)
                    raise
                finally:
                    self._remove_docker(name)
            if result.returncode:
                return f"Docker cannot create a sandbox here: {_docker_detail(result)}"
        except _DockerDaemonUnavailable:
            raise
        except SandboxUnavailable as exc:
            message = f"Docker cannot create a sandbox here: {exc}"
            if isinstance(exc.__cause__, subprocess.TimeoutExpired):
                raise _DockerDaemonUnavailable(message) from exc
            return message
        return None

    def _sweep_docker(self) -> None:
        # datetime uses microseconds. Truncate Docker's fractional seconds before parsing.
        timestamp = re.sub(r"(\.\d{6})\d+", r"\1", self._docker_time)
        try:
            now = dt.datetime.fromisoformat(timestamp)
            if now.tzinfo is None:
                raise ValueError
        except ValueError as exc:
            raise SandboxUnavailable(
                f"Docker reported an invalid system time: {timestamp}"
            ) from exc
        cutoff = now - dt.timedelta(seconds=max(Limits().timeout, self.limits.timeout) + 600)
        result = self.docker_command(
            [
                "ps",
                "-a",
                "--filter",
                "label=" + DOCKER_LABEL,
                "--format",
                "{{.ID}}	{{.CreatedAt}}",
            ]
        )
        if result.returncode:
            raise _DockerDaemonUnavailable(_docker_detail(result))
        for line in result.stdout.splitlines():
            container, _, created = line.partition("	")
            if not re.fullmatch(r"[a-f0-9]{12,64}", container):
                continue
            try:
                # Docker's CreatedAt includes a numeric offset and a trailing zone name.
                created_at = dt.datetime.strptime(
                    " ".join(created.split()[:3]), "%Y-%m-%d %H:%M:%S %z"
                )
            except ValueError:
                continue
            if created_at < cutoff:
                # A worker or the daemon may have removed it since the listing.
                self.docker_command(["rm", "-f", container])

    def _docker_exit(self, name: str) -> int | None:
        result = self.docker_command(
            ["inspect", "--format", "{{.State.StartedAt}} {{.State.ExitCode}}", name]
        )
        if result.returncode:
            return None
        started, _, code = result.stdout.strip().partition(" ")
        if started == "0001-01-01T00:00:00Z":
            return None
        try:
            if started:
                return int(code)
        except ValueError:
            pass
        raise SandboxUnavailable("Docker returned an invalid container state.")

    def _stop_docker(self, name: str) -> None:
        with contextlib.suppress(SandboxUnavailable):
            self.docker_command(["kill", name])

    def _remove_docker(self, name: str) -> None:
        with contextlib.suppress(SandboxUnavailable):
            self.docker_command(["rm", "-f", name])


def _docker_run(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        env=docker_env(argv[0]),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _docker_detail(result: subprocess.CompletedProcess[str]) -> str:
    return result.stderr.strip() or f"exit code {result.returncode}"


def _docker_mount(source: Path, target: str, *, readonly: bool = False) -> str:
    # --mount parses CSV, including Windows drive letters and commas in host paths.
    fields = ["type=bind", f"source={source.absolute()}", f"target={target}"]
    if readonly:
        fields.append("readonly")
    output = io.StringIO()
    csv.writer(output).writerow(fields)
    return output.getvalue().removesuffix("\r\n")


def _pytest_mounts() -> Iterator[tuple[Path, str]]:
    for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments", "py"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            continue
        package = spec.submodule_search_locations is not None
        source = Path(spec.origin).parent if package else Path(spec.origin)
        target = name if package else name + ".py"
        yield source, _PYTHON_PATH + "/" + target
    for name in ("pytest", "pluggy", "iniconfig", "packaging", "pygments", "py"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for file in distribution.files or ():
            if file.parts[0].endswith(".dist-info"):
                source = Path(str(distribution.locate_file(file.parts[0])))
                yield source, _PYTHON_PATH + "/" + source.name
                break


def _with(limits: Limits, timeout: float) -> Limits:
    return dataclasses.replace(limits, timeout=timeout)


def _find_prlimit() -> str | None:
    """prlimit runs inside the sandbox, so it must be one of the system's (under /usr)."""
    for path in ("/usr/bin/prlimit", "/bin/prlimit"):
        if os.access(path, os.X_OK):
            return path
    return None


def _set_limits(limits: Limits) -> Callable[[], None]:
    if sys.platform == "win32":
        raise RuntimeError("the sandbox needs Linux (bubblewrap)")
    import resource  # POSIX only: importing this module must still work elsewhere

    def apply() -> None:  # in the child, between fork and exec
        cpu = max(1, limits.cpu_seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_bytes, limits.file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (limits.open_files, limits.open_files))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return apply


def _execute(argv: list[str], stdin: str, limits: Limits) -> Outcome:
    with tempfile.TemporaryFile() as given:
        # From a file, not a pipe: writing a large stdin while reading output could deadlock.
        given.write(stdin.encode("utf-8"))
        given.seek(0)
        process = subprocess.Popen(
            argv,
            stdin=given,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={},
            close_fds=True,
            start_new_session=True,  # its own process group, killed as a whole on timeout
            preexec_fn=_set_limits(limits),
        )
        assert process.stdout is not None and process.stderr is not None
        stdout = _Head(limits.stdout_chars)
        stderr = _Tail(limits.stderr_chars)
        try:
            timed_out = _collect(process, {process.stdout: stdout, process.stderr: stderr}, limits)
        finally:
            if process.poll() is None:
                _kill(process)  # interrupted (Ctrl+C) or timed out: nothing is left running
            process.stdout.close()
            process.stderr.close()
        code = process.wait()
    return Outcome(
        exit_code=None if timed_out else _exit_code(code),
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
        stdout_cut=stdout.cut,
        stderr_cut=stderr.cut,
    )


@contextlib.contextmanager
def _wsl_install_lock(root: Path) -> Iterator[None]:
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent / ".wsl.lock").open("a+b") as lock:
            filelock.lock(lock.fileno())
            try:
                yield
            finally:
                filelock.unlock(lock.fileno())
    except (OSError, httpx.HTTPError) as exc:
        raise SandboxUnavailable(str(exc)) from exc


def _check_wsl(
    box: Sandbox,
    args: list[str],
    timeout: float,
    *,
    terminal: bool = False,
    step: str | None = None,
) -> None:
    label = step or args[0]
    try:
        result = box.wsl_command(args, timeout, terminal=terminal)
    except SandboxUnavailable as exc:
        raise SandboxUnavailable(f"{label} failed: {exc}") from exc
    if result.returncode:
        raise SandboxUnavailable(f"{label} failed: {_wsl_detail(result)}")


def enable_wsl(box: Sandbox) -> None:
    _check_wsl(box, ["--install", "--no-distribution"], 1800.0, terminal=True)


def _download_wsl(url: str) -> bytes:
    data = bytearray()
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=_READ_CHUNK):
            if len(data) + len(chunk) > 8 * 2**20:
                raise SandboxUnavailable("The WSL sandbox download exceeds the size limit.")
            data.extend(chunk)
    return bytes(data)


def install_wsl(box: Sandbox, *, download: Callable[[str], bytes] | None = None) -> bool:
    fetch = _download_wsl if download is None else download
    with _wsl_install_lock(box.wsl_dir):
        present = box.wsl_distribution_present()
        if present and box.wsl_problem() is None:
            return False
        data = fetch(_WSL_URL)
        if hashlib.sha256(data).hexdigest() != _WSL_SHA256:
            raise SandboxUnavailable(f"SHA-256 mismatch for {WSL_PIN}.tar.gz")
        if present:
            _check_wsl(box, ["--unregister", WSL_DISTRO], 120)
        root = box.wsl_dir / "root"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".setup-", dir=box.wsl_dir) as stage:
            archive = Path(stage) / f"{WSL_PIN}.tar.gz"
            archive.write_bytes(data)
            try:
                _check_wsl(
                    box, ["--import", WSL_DISTRO, str(root), str(archive), "--version", "2"], 120
                )
            except SandboxUnavailable as exc:
                raise WslImportUnavailable(str(exc)) from exc
            try:
                _check_wsl(
                    box,
                    [
                        *_WSL_ROOT,
                        "/bin/sh",
                        "-c",
                        """printf '%s\\n' "$@" > /etc/wsl.conf""",
                        "sh",
                        *_WSL_CONF,
                    ],
                    60,
                    step="write /etc/wsl.conf",
                )
                for step in (
                    [
                        "/sbin/apk",
                        "add",
                        "--no-cache",
                        "python3",
                        "py3-pytest",
                        "bubblewrap",
                        "util-linux-misc",
                        "procps-ng",
                    ],
                    ["/usr/sbin/adduser", "-D", "-H", "-s", "/bin/sh", "sandbox"],
                    [
                        "/bin/sh",
                        "-c",
                        _WSL_MARKER,
                        "sh",
                        f"{WSL_PIN} {dt.date.today().isoformat()}",
                    ],
                ):
                    _check_wsl(box, [*_WSL_ROOT, *step], 600, step=step[0].rsplit("/", 1)[-1])
                _check_wsl(box, ["--terminate", WSL_DISTRO], 60)
                if problem := box.wsl_problem():
                    raise SandboxUnavailable(problem)
            except (SandboxUnavailable, KeyboardInterrupt):
                with contextlib.suppress(SandboxUnavailable):
                    _check_wsl(box, ["--unregister", WSL_DISTRO], 120)
                raise
    return True


def remove_wsl(box: Sandbox) -> bool:
    with _wsl_install_lock(box.wsl_dir):
        present = box.wsl_distribution_present()
        if present:
            _check_wsl(box, ["--unregister", WSL_DISTRO], 120)
        if box.wsl_dir.exists():
            shutil.rmtree(box.wsl_dir)
        return present


class _WslSuspect(Exception):
    """wsl.exe's own failure, or code printing what looks like one: the caller probes."""

    def __init__(self, detail: str, outcome: Outcome) -> None:
        super().__init__(detail)
        self.detail = detail
        self.outcome = outcome


def _execute_wsl(argv: list[str], stdin: str, limits: Limits, stop: Callable[[], None]) -> Outcome:
    stdout = _Head(limits.stdout_chars)
    stderr = _WslTail(limits.stderr_chars)
    diagnostic = _WslTail(2_000)
    with tempfile.TemporaryFile() as given:
        given.write(stdin.encode("utf-8"))
        given.seek(0)
        try:
            process = subprocess.Popen(
                argv,
                stdin=given,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_wsl_env(),
                creationflags=_WINDOWS_PROCESS_GROUP,
            )
        except OSError as exc:
            raise SandboxUnavailable(f"WSL cannot be started: {exc}") from exc
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=_read_output, args=(process.stdout, stdout)),
            threading.Thread(target=_read_output, args=(process.stderr, stderr, diagnostic)),
        ]
        timed_out = interrupted = False
        try:
            for reader in readers:
                reader.start()
            process.wait(timeout=limits.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if timed_out or interrupted:
                stop()
            for reader in readers:
                if reader.ident is not None:
                    reader.join()
            process.stdout.close()
            process.stderr.close()
    outcome = Outcome(
        exit_code=None if timed_out else process.returncode,
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
        stdout_cut=stdout.cut,
        stderr_cut=stderr.cut,
    )
    if not timed_out:
        detail = diagnostic.text().strip()
        if not 0 <= process.returncode <= 255:
            # No code inside bwrap can exit so: wsl.exe itself failed.
            raise SandboxUnavailable(f"WSL failed: {detail or process.returncode}")
        if (
            not stdout.data
            and not stdout.cut
            and any(prefix in detail for prefix in ("There is no distribution", "WSL (", "Wsl/"))
        ):
            raise _WslSuspect(detail, outcome)
    return outcome


def _execute_docker(
    argv: list[str],
    stdin: str,
    limits: Limits,
    stop: Callable[[], None],
    inspect_exit: Callable[[], int | None],
) -> Outcome:
    stdout = _Head(limits.stdout_chars)
    stderr = _Tail(limits.stderr_chars)
    diagnostic = _Tail(2_000)
    with tempfile.TemporaryFile() as given:
        given.write(stdin.encode("utf-8"))
        given.seek(0)
        try:
            process = subprocess.Popen(
                argv,
                env=docker_env(argv[0]),
                stdin=given,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise SandboxUnavailable(f"Docker cannot be started: {exc}") from exc
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=_read_output, args=(process.stdout, stdout)),
            threading.Thread(target=_read_output, args=(process.stderr, stderr, diagnostic)),
        ]
        complete = timed_out = False
        try:
            for reader in readers:
                reader.start()
            try:
                client_code = process.wait(timeout=limits.timeout)
                for reader in readers:
                    reader.join()
                complete = True
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if not complete:
                try:
                    stop()
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    process.wait()
                for reader in readers:
                    if reader.ident is not None:
                        reader.join()
            process.stdout.close()
            process.stderr.close()
    code = None
    if not timed_out:
        code = inspect_exit()
        if code is None:
            raise SandboxUnavailable(
                diagnostic.text().strip()
                or f"Docker exited with code {client_code} before starting"
            )
    return Outcome(
        exit_code=None if code is None else _exit_code(code),
        stdout=stdout.text(),
        stderr=stderr.text(),
        timed_out=timed_out,
        stdout_cut=stdout.cut,
        stderr_cut=stderr.cut,
    )


def _read_output(stream: IO[bytes], *buffers: _Head) -> None:
    while chunk := stream.read(_READ_CHUNK):
        for buffer in buffers:
            buffer.add(chunk)


def _collect(
    process: subprocess.Popen[bytes], streams: dict[IO[bytes], _Head | _Tail], limits: Limits
) -> bool:
    """Read both streams until they close or time runs out; True if it ran out."""
    keep = {stream.fileno(): kept for stream, kept in streams.items()}
    deadline = time.monotonic() + limits.timeout
    timed_out = False
    with selectors.DefaultSelector() as selector:
        for stream in streams:
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            left = deadline - time.monotonic()
            if left <= 0 and not timed_out:
                timed_out = True
                _kill(process)
                deadline = time.monotonic() + 5.0  # the pipes close once the group is gone
                continue
            if left <= 0:
                break  # something outlived the kill: stop reading, never hang
            for key, _ in selector.select(timeout=left):
                chunk = os.read(key.fd, _READ_CHUNK)
                if chunk:
                    keep[key.fd].add(chunk)
                else:
                    selector.unregister(key.fileobj)
    if timed_out:
        return True
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:  # closed its output but still runs
        _kill(process)
        return True
    return False


def _kill(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        if sys.platform == "win32":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)


def _exit_code(code: int) -> int:
    return 128 - code if code < 0 else code  # killed outside bwrap: same convention as inside


class _Head:
    """The first `limit` characters (decoded as UTF-8) of a stream."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.cut = False

    def add(self, chunk: bytes) -> None:
        room = 4 * self.limit - len(self.data)  # a character is at most four bytes
        if room > 0:
            self.data += chunk[:room]
        if len(chunk) > max(room, 0):
            self.cut = True

    def text(self) -> str:
        text = self.data.decode("utf-8", errors="replace")
        if len(text) > self.limit:
            self.cut = True
            text = text[: self.limit]
        return text


class _Tail(_Head):
    """The last `limit` characters of a stream."""

    def add(self, chunk: bytes) -> None:
        self.data += chunk
        excess = len(self.data) - 4 * self.limit
        if excess > 0:
            del self.data[:excess]
            self.cut = True

    def text(self) -> str:
        text = self.data.decode("utf-8", errors="replace")
        if len(text) > self.limit:
            self.cut = True
            text = text[-self.limit :]
        return text


# ADR 0013. Change the install ID whenever the runtime or any wheel changes.
class _WslTail(_Tail):
    def text(self) -> str:
        # The tail keeps an even number of bytes, so UTF-16 code units stay aligned.
        text = _decode_wsl(self.data)
        if len(text) > self.limit:
            self.cut = True
            return text[-self.limit :]
        return text


_WASI_PIN = "python-3.15.2-sdk33-pytest-9.2.0"
_WASI_ARCHIVES = (
    (
        "https://github.com/brettcannon/cpython-wasi-build/releases/download/v3.15.2/python-3.15.2-wasi_sdk-33.zip",
        "b8a63559c56760bbdc310dedb74fd69d10bcbafae26b705bea0ad353ad256888",
        "py",
    ),
    (
        "https://files.pythonhosted.org/packages/51/39/1f5d8c8877a0fa327db959c19341d4be61660de1ab6b7391f9eb805bcd67/pytest-9.2.0-py3-none-any.whl",
        "07acdcf88ec174f907dc01af70bf52d49b014bbab03faeeade31c96b18c2e588",
        "site",
    ),
    (
        "https://files.pythonhosted.org/packages/40/9e/2b38731e0fc536806f16490e1a12d7f0dc2a1235aa8cc07bcc75416a7daa/pluggy-1.7.0-py3-none-any.whl",
        "7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec",
        "site",
    ),
    (
        "https://files.pythonhosted.org/packages/56/43/4ca9e49d27a1fcf6bece6f6aec0ea46bb9112489b93d4b688fb415457bdb/iniconfig-2.3.1-py3-none-any.whl",
        "9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7",
        "site",
    ),
    (
        "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
        "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
        "site",
    ),
    (
        "https://files.pythonhosted.org/packages/71/46/17f022dd3e953bf20a04a028a21ec746d942f8d2af30fa0f124fa0e6a684/pygments-2.21.0-py3-none-any.whl",
        "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9",
        "site",
    ),
)


def wasi_runtime() -> bool:
    try:
        import wasmtime  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def _wasi_files(root: Path) -> bool:
    return (
        (root / "py/python.wasm").is_file()
        and (root / "py/lib").is_dir()
        and all(
            (root / "site" / name).is_dir()
            for name in ("pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments")
        )
    )


_WASI_DOWNLOAD_LIMIT: Final = 64 * 2**20


def _download_wasi(url: str) -> bytes:
    data = bytearray()
    with httpx.stream(
        "GET", url, headers={"Accept-Encoding": "identity"}, follow_redirects=True, timeout=60.0
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_raw(chunk_size=64 * 1024):
            if len(data) + len(chunk) > _WASI_DOWNLOAD_LIMIT:
                raise SandboxUnavailable("The WebAssembly sandbox download exceeds the size limit.")
            data.extend(chunk)
    return bytes(data)


def _unpack_wasi(data: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            name = entry.orig_filename
            parts = name.removesuffix("/").split("/")
            if (
                "\\" in name
                or "\x00" in name
                or any(
                    part in ("", ".", "..")
                    or ":" in part
                    or part.endswith((" ", "."))
                    or PureWindowsPath(part).is_reserved()
                    for part in parts
                )
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise SandboxUnavailable(f"Unsafe sandbox archive entry: {name!r}")
            target = destination.joinpath(*parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def install_wasi(root: Path, *, download: Callable[[str], bytes] | None = None) -> bool:
    """Verify each download and publish the entire installation in a single rename."""
    if _wasi_files(root) and (root / "ok").is_file():
        return False
    fetch = _download_wasi if download is None else download
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".setup-", dir=root.parent) as scratch:
            stage = Path(scratch) / "install"
            for url, digest, folder in _WASI_ARCHIVES:
                data = fetch(url)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise SandboxUnavailable(f"SHA-256 mismatch for {url.rsplit('/', 1)[-1]}")
                _unpack_wasi(data, stage / folder)
            if not _wasi_files(stage):
                raise SandboxUnavailable("The WebAssembly sandbox download is incomplete.")
            _warm_wasi_cache(stage)
            (stage / "ok").touch()
            # Serialize publication so simultaneous repairs cannot displace a completed install.
            with (root.parent / f".{root.name}.lock").open("a+b") as lock:
                filelock.lock(lock.fileno())
                try:
                    if _wasi_files(root) and (root / "ok").is_file():
                        return False
                    broken = root.with_name(f".{root.name}.broken-{uuid.uuid4().hex}")
                    if os.path.lexists(root):
                        root.rename(broken)
                    try:
                        stage.rename(root)
                    except OSError:
                        if os.path.lexists(broken):
                            broken.rename(root)
                        raise
                finally:
                    filelock.unlock(lock.fileno())
            if broken.is_dir() and not broken.is_symlink():
                shutil.rmtree(broken)
            else:
                broken.unlink(missing_ok=True)
    except (OSError, httpx.HTTPError, zipfile.BadZipFile) as exc:
        raise SandboxUnavailable(str(exc)) from exc
    return True


def _warm_wasi_cache(root: Path) -> None:
    try:
        from desman.core.wasi_runner import warm_cache

        warm_cache(root)
    except (ImportError, RuntimeError) as exc:
        raise SandboxUnavailable(f"Cannot prepare the WebAssembly sandbox: {exc}") from exc


_WASI_PYTEST_ARGS = (
    "-p",
    "no:cacheprovider",
    "--capture=sys",
    "-p",
    "no:faulthandler",
    "-p",
    "no:logging",
)
_WASI_UNSUPPORTED = (
    "Only Python scripts, snippets, pytest and unittest run in this sandbox (WebAssembly); "
    "use python FILE.py, python -c CODE, python -m pytest, or python -m unittest. "
    "Packages with C code are not available."
)


def _wasi_argv(command: Sequence[str]) -> list[str] | None:
    if len(command) < 2 or command[0] not in ("python", "python3"):
        return None
    start = f"import os; os.chdir({MOUNT_POINT!r}); "
    index = 1
    while index < len(command):
        flag = command[index]
        if flag in ("-u", "-B", "-E", "-s", "-I") or (flag.startswith("-X") and flag != "-X"):
            index += 1
        elif flag in ("-X", "-W"):
            index += 2
        else:
            break
    match list(command[index:]):
        case ["-m", "pytest", *args]:
            source = (
                f"{start}import sys, pytest; sys.exit(pytest.main({[*args, *_WASI_PYTEST_ARGS]!r}))"
            )
            return ["python", "-c", source]
        case ["-m", "unittest", *args]:
            source = (
                f"{start}import runpy; "
                "runpy.run_module('unittest', run_name='__main__', alter_sys=True)"
            )
            return ["python", "-c", source, *args]
        case ["-c", code, *args]:
            # Compile the original source on its own, preserving future imports and line numbers.
            return ["python", "-c", f"{start}exec(compile({code!r}, '<string>', 'exec'))", *args]
        case [file, *args] if file.endswith(".py") and not file.startswith("-"):
            source = (
                f"{start}import runpy, sys; "
                f"sys.argv = {[file, *args]!r}; "
                "sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
                "runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            return ["python", "-c", source]
        case _:
            return None


class _WasiSpec(BaseModel):
    argv: list[str]
    workdir: Path
    root: Path
    timeout: float
    memory_bytes: int
    stdin: Path
    stdout: Path
    stderr: Path
    result: Path
    ready: Path


class _WasiResult(BaseModel):
    exit_code: int | None = None
    trap: str = ""
    error: str | None = None


def _wasi_runner_command(spec: Path) -> list[str]:
    source = Path(__file__).resolve().parents[2]
    installed_paths = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    runner = "desman.core.wasi_runner"
    if source in installed_paths:
        return [sys.executable, "-I", "-m", runner, str(spec)]
    # Isolated mode ignores PYTHONPATH. Explicitly load this checkout in source/editable runs.
    bootstrap = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name='__main__')"
    return [sys.executable, "-I", "-c", bootstrap, str(source), runner, str(spec)]


def _wasi_output_over_limit(path: Path, chars: int, limits: Limits) -> bool:
    # Leave headroom for ordinary output truncation before terminating a writer.
    cap = min(limits.file_bytes, 4 * chars + 64 * 1024)
    return path.stat().st_size > cap


def _workspace_files(root: Path) -> Iterator[tuple[Path, os.stat_result]]:
    pending = [root]
    while pending:
        directory = pending.pop()
        with (
            contextlib.suppress(FileNotFoundError, NotADirectoryError, PermissionError),
            os.scandir(directory) as entries,
        ):
            for entry in entries:
                with contextlib.suppress(FileNotFoundError):
                    info = entry.stat(follow_symlinks=False)
                    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                        continue
                    path = Path(entry.path)
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(path)
                    elif stat.S_ISREG(info.st_mode):
                        yield path, info


def _file_version(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _wasi_files_over_limit(
    workdir: Path, before: dict[Path, tuple[int, int, int, int]], cap: int
) -> bool:
    total = 0
    for path, info in _workspace_files(workdir):
        if before.get(path) != _file_version(info):
            total += info.st_size
            if info.st_size > cap or total > 4 * cap:
                return True
    return False


def _wait_wasi(
    process: subprocess.Popen[bytes],
    spec: _WasiSpec,
    limits: Limits,
    before: dict[Path, tuple[int, int, int, int]],
) -> _WasiResult:
    deadline = time.monotonic() + 180.0
    ready = False
    next_scan = time.monotonic()
    while True:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=min(0.2, max(0.0, deadline - time.monotonic())))
        now = time.monotonic()
        if not ready and spec.ready.is_file():
            ready = True
            deadline = now + limits.timeout + 2.0
        if process.returncode is not None or (ready and now >= next_scan):
            if _wasi_files_over_limit(spec.workdir, before, limits.file_bytes):
                return _WasiResult(
                    exit_code=1, trap="WebAssembly sandbox file size limit exceeded.\n"
                )
            next_scan = now + 0.5
        if process.returncode is not None:
            return _wasi_exit_result(process.returncode, spec)
        if now >= deadline:
            note = "" if ready else "Timed out loading the WebAssembly sandbox.\n"
            return _WasiResult(trap=note)
        if _wasi_output_over_limit(
            spec.stdout, limits.stdout_chars, limits
        ) or _wasi_output_over_limit(spec.stderr, limits.stderr_chars, limits):
            # A large final write can reach the files just before the runner exits.
            # Give teardown one polling interval before terminating an active writer.
            try:
                process.wait(timeout=min(0.2, deadline - now))
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    _kill(process)
                    code = process.wait()
                    if sys.platform == "win32":
                        killed = code == 1
                    else:
                        killed = code == -signal.SIGKILL
                    if killed:
                        return _WasiResult(
                            exit_code=1, trap="WebAssembly sandbox output limit exceeded.\n"
                        )


def _wasi_exit_result(code: int, spec: _WasiSpec) -> _WasiResult:
    detail = f"exit code {code}"
    if code == 0:
        try:
            result = _WasiResult.model_validate_json(spec.result.read_bytes())
        except (OSError, ValueError) as exc:
            detail = str(exc)
        else:
            if result.error is None:
                return result
            detail = result.error
    return _WasiResult(
        exit_code=_exit_code(code) if code < 0 else 1,
        trap=f"the WebAssembly runner stopped: {detail}\n",
    )


def _execute_wasi(
    command: Sequence[str], workdir: Path, stdin: str, limits: Limits, root: Path
) -> Outcome:
    argv = _wasi_argv(command)
    if argv is None:
        return Outcome(2, "", _WASI_UNSUPPORTED)

    try:
        with tempfile.TemporaryDirectory(
            prefix="desman-wasi-", ignore_cleanup_errors=True
        ) as directory:
            scratch = Path(directory)
            spec = _WasiSpec(
                argv=argv,
                workdir=workdir.resolve(),
                root=root.resolve(),
                timeout=limits.timeout,
                memory_bytes=limits.memory_bytes,
                stdin=scratch / "stdin",
                stdout=scratch / "stdout",
                stderr=scratch / "stderr",
                result=scratch / "result.json",
                ready=scratch / "ready",
            )
            spec.stdin.write_bytes(stdin.encode("utf-8"))
            spec.stdout.touch()
            spec.stderr.touch()
            config = scratch / "spec.json"
            config.write_text(spec.model_dump_json(), encoding="utf-8")
            return _run_wasi_child(config, spec, limits)
    except (OSError, ValueError) as exc:
        raise SandboxUnavailable(f"Cannot start the WebAssembly sandbox: {exc}") from exc


def _run_wasi_child(config: Path, spec: _WasiSpec, limits: Limits) -> Outcome:
    windows = sys.platform == "win32"
    env = (
        {
            name: os.environ[name]
            for name in ("SYSTEMROOT", "TEMP", "TMP", "PATH")
            if name in os.environ
        }
        if windows
        else {}
    )
    before = {path: _file_version(info) for path, info in _workspace_files(spec.workdir)}
    # This child runs DESMAN's trusted runner. Generated code executes only inside Wasmtime.
    with subprocess.Popen(
        _wasi_runner_command(config),
        cwd=config.parent,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=not windows,
        creationflags=_WINDOWS_PROCESS_GROUP if windows else 0,
    ) as process:
        try:
            result = _wait_wasi(process, spec, limits, before)
        finally:
            if process.returncode is None:
                _kill(process)
                process.wait()

    stdout_cut = _wasi_output_over_limit(spec.stdout, limits.stdout_chars, limits)
    stderr_cut = _wasi_output_over_limit(spec.stderr, limits.stderr_chars, limits)
    stdout, stderr = _Head(limits.stdout_chars), _Tail(limits.stderr_chars)
    with spec.stdout.open("rb") as stream:
        stdout.add(stream.read(4 * limits.stdout_chars + 1))
    with spec.stderr.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - 4 * limits.stderr_chars - 1))
        stderr.add(stream.read())
    if result.trap:
        stderr.add(result.trap.encode("utf-8"))
    return Outcome(
        result.exit_code,
        stdout.text(),
        stderr.text(),
        result.exit_code is None,
        stdout_cut or stdout.cut,
        stderr_cut or stderr.cut,
    )
