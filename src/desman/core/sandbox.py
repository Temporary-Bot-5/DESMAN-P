"""Sandbox for generated code (ROADMAP 3.3; B5, second part).

The old prototype ran generated code with the network and the whole file system in reach. Here
a command runs inside bubblewrap (`bwrap`) on Linux, or Docker elsewhere (ADR 0012).

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

import contextlib
import csv
import dataclasses
import datetime as dt
import importlib.metadata
import importlib.util
import io
import os
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

DOCKER_IMAGE: Final = (
    "python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
)
DOCKER_LABEL: Final = "desman.sandbox=1"
_DOCKER_TIMEOUT: Final = 5.0
_DOCKER_NAME: Final = "desman-sbx-"
_PYTHON_PATH: Final = "/opt/desman/python"
DockerRunner = Callable[[list[str], float], subprocess.CompletedProcess[str]]

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
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits if limits is not None else Limits()
        self.bwrap = bwrap if bwrap is not None else shutil.which("bwrap")
        self.prlimit = prlimit if prlimit is not None else _find_prlimit()
        self.platform = sys.platform if platform is None else platform
        self.backend = "bwrap" if self.platform.startswith("linux") else "docker"
        self.docker = (
            docker
            if docker is not None
            else (shutil.which("docker") if self.backend == "docker" else None)
        )
        self._docker_runner = docker_runner if docker_runner is not None else _docker_run
        self.image_present: bool | None = None
        self._problem: str | None = None
        self._checked = False
        self._monotonic = monotonic
        self._retry_at: float | None = None

    def unavailable(self) -> str | None:
        """Why no command can run here, or None if the sandbox works."""
        if self._retry_at is not None and self._monotonic() >= self._retry_at:
            return self.reprobe()
        if not self._checked:
            try:
                self._problem = self._probe() if self.backend == "bwrap" else self._probe_docker()
            except _DockerDaemonUnavailable as exc:
                self._problem = str(exc)
                self._retry_at = self._monotonic() + 60.0
            self._checked = True
        return self._problem

    def reprobe(self) -> str | None:
        """Refresh availability after the operator changes the sandbox setup."""
        self._checked = False
        self.image_present = None
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
        return _execute(self._argv(command, workdir), stdin, limits)

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
        try:
            return self._docker_problem()
        except _DockerDaemonUnavailable as exc:
            return str(exc)

    def _docker_problem(self) -> str | None:
        if not self.docker:
            return "Docker is not installed or is not on PATH."
        try:
            result = self.docker_command(["info", "--format", "{{.OSType}}"])
        except SandboxUnavailable as exc:
            raise _DockerDaemonUnavailable(f"Docker daemon did not answer: {exc}") from exc
        if result.returncode:
            raise _DockerDaemonUnavailable(
                f"Docker daemon did not answer: {_docker_detail(result)}"
            )
        if result.stdout.strip() != "linux":
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
        problem = self._docker_problem()
        if problem is not None:
            return problem
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
                problem = self._docker_problem()
                return problem or f"Docker cannot create a sandbox here: {_docker_detail(result)}"
        except _DockerDaemonUnavailable:
            raise
        except SandboxUnavailable as exc:
            message = f"Docker cannot create a sandbox here: {exc}"
            if isinstance(exc.__cause__, subprocess.TimeoutExpired):
                raise _DockerDaemonUnavailable(message) from exc
            return message
        return None

    def _sweep_docker(self) -> None:
        clock = self.docker_command(["info", "--format", "{{.SystemTime}}"])
        if clock.returncode:
            raise _DockerDaemonUnavailable(_docker_detail(clock))
        # datetime uses microseconds. Truncate Docker's fractional seconds before parsing.
        timestamp = re.sub(r"(\.\d{6})\d+", r"\1", clock.stdout.strip())
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
                stdin=given,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise SandboxUnavailable(f"Docker cannot be started: {exc}") from exc
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=_read_docker, args=(process.stdout, stdout)),
            threading.Thread(target=_read_docker, args=(process.stderr, stderr, diagnostic)),
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


def _read_docker(stream: IO[bytes], *buffers: _Head) -> None:
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
