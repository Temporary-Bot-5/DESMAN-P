"""Sandbox for generated code (ROADMAP 3.3; B5, second part).

The old prototype ran generated code with the network and the whole file system in reach. Here
a command runs only inside bubblewrap (`bwrap`), and never without it:

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

Known limits: the workspace is the disk the code may fill (each file is capped, the number of
files is not); `/usr` is readable (system files only, nothing of the person's).
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

MOUNT_POINT: Final = "/workspace"
SANDBOX_PATH: Final = "/usr/local/bin:/usr/bin:/bin"
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
    """Runs commands confined by bubblewrap. Whether it works here is checked once, on first
    use, by starting an empty sandbox."""

    def __init__(
        self,
        limits: Limits | None = None,
        *,
        bwrap: str | None = None,
        prlimit: str | None = None,
    ) -> None:
        self.limits = limits if limits is not None else Limits()
        self.bwrap = bwrap if bwrap is not None else shutil.which("bwrap")
        self.prlimit = prlimit if prlimit is not None else _find_prlimit()
        self._problem: str | None = None
        self._checked = False

    def unavailable(self) -> str | None:
        """Why no command can run here, or None if the sandbox works."""
        if not self._checked:
            self._problem = self._probe()
            self._checked = True
        return self._problem

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
        for name in _SYSTEM_LINKS:
            path = Path(name)
            if path.is_symlink():
                argv += ["--symlink", os.readlink(path), name]
            elif path.is_dir():
                argv += ["--ro-bind", name, name]
        for name in _SYSTEM_FILES:
            argv += ["--ro-bind-try", name, name]
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
        if not sys.platform.startswith("linux"):
            return "the sandbox needs Linux (bubblewrap)"
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


def _with(limits: Limits, timeout: float) -> Limits:
    return dataclasses.replace(limits, timeout=timeout)


def _find_prlimit() -> str | None:
    """prlimit runs inside the sandbox, so it must be one of the system's (under /usr)."""
    for path in ("/usr/bin/prlimit", "/bin/prlimit"):
        if os.access(path, os.X_OK):
            return path
    return None


def _set_limits(limits: Limits) -> Callable[[], None]:
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
