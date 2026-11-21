"""Windows AppContainer experiment. Importing this module has no side effects."""
import argparse
from contextlib import ExitStack, contextmanager
import ctypes as C
from ctypes import wintypes as W
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import tempfile
import time

PROFILE = "desman.research.sandbox"
JOURNAL = Path(__file__).resolve().with_name("acl-journal.json")
OUTPUT_LIMIT = 64 * 1024
PTR = C.c_void_p
SIZE = C.c_size_t
DWORD = W.DWORD
HANDLE = W.HANDLE
BOOL = W.BOOL
P = C.POINTER

# UpdateProcThreadAttribute keys (input attributes).
HANDLE_LIST = 0x20002
SECURITY_CAPABILITIES = 0x20009
ALL_APPLICATION_PACKAGES_POLICY = 0x2000F
DACL_SECURITY_INFORMATION = 4
SE_FILE_OBJECT = 1
INHERITED_ACE = 0x10


class SecurityAttributes(C.Structure):
    _fields_ = [("length", DWORD), ("descriptor", PTR), ("inherit", BOOL)]


class SecurityCapabilities(C.Structure):
    _fields_ = [("sid", PTR), ("capabilities", PTR), ("count", DWORD), ("reserved", DWORD)]


class StartupInfo(C.Structure):
    _fields_ = [
        ("cb", DWORD), ("reserved", W.LPWSTR), ("desktop", W.LPWSTR),
        ("title", W.LPWSTR), ("x", DWORD), ("y", DWORD), ("xsize", DWORD),
        ("ysize", DWORD), ("xchars", DWORD), ("ychars", DWORD), ("fill", DWORD),
        ("flags", DWORD), ("show", W.WORD), ("reserved_size", W.WORD),
        ("reserved_bytes", PTR), ("stdin", HANDLE), ("stdout", HANDLE), ("stderr", HANDLE),
    ]


class StartupInfoEx(C.Structure):
    _fields_ = [("startup", StartupInfo), ("attributes", PTR)]


class ProcessInfo(C.Structure):
    _fields_ = [("process", HANDLE), ("thread", HANDLE), ("pid", DWORD), ("tid", DWORD)]


class BasicLimits(C.Structure):
    _fields_ = [
        ("process_time", C.c_int64), ("job_time", C.c_int64), ("flags", DWORD),
        ("min_working_set", SIZE), ("max_working_set", SIZE), ("active_processes", DWORD),
        ("affinity", SIZE), ("priority", DWORD), ("scheduling", DWORD),
    ]


class ExtendedLimits(C.Structure):
    _fields_ = [
        ("basic", BasicLimits), ("io_counters", C.c_uint64 * 6),
        ("process_memory", SIZE), ("job_memory", SIZE),
        ("peak_process_memory", SIZE), ("peak_job_memory", SIZE),
    ]


class Accounting(C.Structure):
    _fields_ = [
        ("times", C.c_int64 * 4), ("page_faults", DWORD), ("total", DWORD),
        ("active", DWORD), ("terminated", DWORD),
    ]


def bind(dll, name, result, *args):
    function = getattr(dll, name)
    function.restype = result
    function.argtypes = args
    return function


def win_error(name, code=None):
    code = C.get_last_error() if code is None else code
    return OSError(f"{name}: [WinError {code}] {C.FormatError(code).strip()}")


def checked(function, *args):
    result = function(*args)
    if not result:
        raise win_error(function.__name__)
    return result


def status(function, *args, accepted=(0,)):
    result = function(*args) & 0xFFFFFFFF
    if result not in accepted:
        code = result & 0xFFFF if result & 0xFFFF0000 == 0x80070000 else result
        raise win_error(function.__name__, code)
    return result


class Windows:
    def __init__(self):
        kernel = C.WinDLL("kernel32", use_last_error=True)
        security = C.WinDLL("advapi32", use_last_error=True)
        profiles = C.WinDLL("userenv", use_last_error=True)
        self.create_profile = bind(
            profiles, "CreateAppContainerProfile", C.c_long,
            W.LPCWSTR, W.LPCWSTR, W.LPCWSTR, PTR, DWORD, P(PTR),
        )
        self.derive_sid = bind(
            profiles, "DeriveAppContainerSidFromAppContainerName", C.c_long, W.LPCWSTR, P(PTR),
        )
        self.delete_profile = bind(profiles, "DeleteAppContainerProfile", C.c_long, W.LPCWSTR)
        self.free_sid = bind(security, "FreeSid", PTR, PTR)
        self.sid_size = bind(security, "GetLengthSid", DWORD, PTR)
        self.get_acl = bind(
            security, "GetNamedSecurityInfoW", DWORD,
            W.LPCWSTR, DWORD, DWORD, PTR, PTR, P(PTR), PTR, P(PTR),
        )
        self.set_acl = bind(
            security, "SetNamedSecurityInfoW", DWORD, W.LPWSTR, DWORD, DWORD, PTR, PTR, PTR, PTR,
        )
        self.local_free = bind(kernel, "LocalFree", PTR, PTR)
        self.close = bind(kernel, "CloseHandle", BOOL, HANDLE)
        self.open_token = bind(security, "OpenProcessToken", BOOL, HANDLE, DWORD, P(HANDLE))
        self.token_info = bind(
            security, "GetTokenInformation", BOOL, HANDLE, DWORD, PTR, DWORD, P(DWORD),
        )
        self.equal_sid = bind(security, "EqualSid", BOOL, PTR, PTR)
        self.init_attributes = bind(
            kernel, "InitializeProcThreadAttributeList", BOOL, PTR, DWORD, DWORD, P(SIZE),
        )
        self.update_attribute = bind(
            kernel, "UpdateProcThreadAttribute", BOOL, PTR, DWORD, SIZE, PTR, SIZE, PTR, PTR,
        )
        self.delete_attributes = bind(kernel, "DeleteProcThreadAttributeList", None, PTR)
        self.create_pipe = bind(
            kernel, "CreatePipe", BOOL, P(HANDLE), P(HANDLE), P(SecurityAttributes), DWORD,
        )
        self.handle_flags = bind(kernel, "SetHandleInformation", BOOL, HANDLE, DWORD, DWORD)
        self.peek = bind(kernel, "PeekNamedPipe", BOOL, HANDLE, PTR, DWORD, PTR, P(DWORD), PTR)
        self.read = bind(kernel, "ReadFile", BOOL, HANDLE, PTR, DWORD, P(DWORD), PTR)
        self.create_job = bind(kernel, "CreateJobObjectW", HANDLE, PTR, W.LPCWSTR)
        self.set_job = bind(kernel, "SetInformationJobObject", BOOL, HANDLE, C.c_int, PTR, DWORD)
        self.query_job = bind(
            kernel, "QueryInformationJobObject", BOOL, HANDLE, C.c_int, PTR, DWORD, PTR,
        )
        self.assign = bind(kernel, "AssignProcessToJobObject", BOOL, HANDLE, HANDLE)
        self.kill_job = bind(kernel, "TerminateJobObject", BOOL, HANDLE, W.UINT)
        self.kill_process = bind(kernel, "TerminateProcess", BOOL, HANDLE, W.UINT)
        self.create_process = bind(
            kernel, "CreateProcessW", BOOL, W.LPCWSTR, W.LPWSTR, PTR, PTR, BOOL, DWORD,
            PTR, W.LPCWSTR, P(StartupInfo), P(ProcessInfo),
        )
        self.resume = bind(kernel, "ResumeThread", DWORD, HANDLE)
        self.exit_code = bind(kernel, "GetExitCodeProcess", BOOL, HANDLE, P(DWORD))

    def acl(self, path):
        acl, descriptor = PTR(), PTR()
        status(self.get_acl, str(path), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
               None, None, C.byref(acl), None, C.byref(descriptor))
        try:
            if not acl:
                raise ValueError(f"Refusing to change a null DACL at {path}")
            revision, _, size, count, _ = struct.unpack("<BBHHH", C.string_at(acl, 8))
            data = C.string_at(acl, size)
            entries = []
            offset = 8
            for _ in range(count):
                length = struct.unpack_from("<H", data, offset + 2)[0]
                entries.append(data[offset:offset + length])
                offset += length
            return revision, entries
        finally:
            self.local_free(descriptor)

    def set_entries(self, path, revision, entries):
        data = b"".join(entries)
        header = struct.pack("<BBHHH", revision, 0, 8 + len(data), len(entries), 0)
        acl = C.create_string_buffer(header + data)
        status(self.set_acl, str(path), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
               None, None, acl, None)

    def verify_container(self, process, sid):
        token, size, inside = HANDLE(), DWORD(), DWORD()
        checked(self.open_token, process, 0x0008, C.byref(token))  # TOKEN_QUERY
        try:
            checked(self.token_info, token, 29, C.byref(inside), C.sizeof(inside), C.byref(size))
            if not inside.value:
                raise RuntimeError("Suspended process has no AppContainer token")
            self.token_info(token, 31, None, 0, C.byref(size))  # TokenAppContainerSid
            if C.get_last_error() != 122 or not size.value:
                raise win_error("GetTokenInformation sizing")
            buffer = C.create_string_buffer(size.value)
            checked(self.token_info, token, 31, buffer, len(buffer), C.byref(size))
            actual_sid = PTR.from_buffer(buffer).value
            if not actual_sid or not self.equal_sid(actual_sid, sid):
                raise RuntimeError("Suspended process has a different AppContainer SID")
        finally:
            checked(self.close, token)


@contextmanager
def journal_lock():
    import msvcrt

    lock_path = JOURNAL.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def save_journal(journal):
    fd, name = tempfile.mkstemp(dir=JOURNAL.parent, prefix="acl-journal-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(journal, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, JOURNAL)
    finally:
        Path(name).unlink(missing_ok=True)


def load_journal():
    if JOURNAL.exists():
        return json.loads(JOURNAL.read_text(encoding="utf-8"))
    return {"profile": False, "grants": []}


def check_tree(path):
    """Keep recursive DACL propagation away from junctions and other reparse points."""
    for part in (path, *path.parents):
        if part.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError(f"Reparse point in ACL path: {part}")
    pending = [path]
    while pending:
        for child in pending.pop().iterdir():
            info = child.lstat()
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f"Reparse point in ACL tree: {child}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(child)
            elif info.st_nlink != 1:
                raise ValueError(f"Hard-linked file in ACL tree: {child}")


def allow_ace(sid, flags, mask):
    return struct.pack("<BBHI", 0, flags, 8 + len(sid), mask) + sid


def grant(api, journal, path, requested):
    identity = path.stat()
    revision, entries = api.acl(path)
    for record in journal["grants"]:
        if Path(record["path"]) == path:
            if (record["identity"] != [identity.st_dev, identity.st_ino]
                    or record["aces"] != [ace.hex() for ace in requested]
                    or any(entries.count(ace) != 1 for ace in requested)):
                raise ValueError(f"Journal and DACL differ at {path}; run --cleanup first")
            return
    if any(ace in entries for ace in requested):
        raise ValueError(f"An identical unowned ACE already exists at {path}")
    position = next((i for i, ace in enumerate(entries) if ace[1] & INHERITED_ACE), len(entries))
    journal["grants"].append({
        "path": str(path), "identity": [identity.st_dev, identity.st_ino],
        "aces": [ace.hex() for ace in requested],
    })
    save_journal(journal)  # Record intent before changing the directory.
    api.set_entries(path, revision, entries[:position] + requested + entries[position:])


def cleanup(api, journal):
    for record in reversed(journal["grants"][:]):
        path = Path(record["path"])
        check_tree(path)
        identity = path.stat()
        if record["identity"] != [identity.st_dev, identity.st_ino]:
            raise ValueError(f"Refusing to clean a replacement directory: {path}")
        revision, entries = api.acl(path)
        remaining = entries[:]
        for encoded in record["aces"]:
            ace = bytes.fromhex(encoded)
            if remaining.count(ace) > 1:
                raise ValueError(f"Ambiguous duplicate ACE at {path}")
            if ace in remaining:
                remaining.remove(ace)
        if remaining != entries:
            api.set_entries(path, revision, remaining)
        journal["grants"].remove(record)
        save_journal(journal)
    if journal["profile"]:
        status(api.delete_profile, PROFILE, accepted=(0, 0x80070002, 0x80070490))
    JOURNAL.unlink(missing_ok=True)


def drain(api, reader, captured):
    available, count = DWORD(), DWORD()
    if not api.peek(reader, None, 0, None, C.byref(available), None):
        if C.get_last_error() in (109, 232):  # Broken pipe or no data after writer closes.
            return False
        raise win_error("PeekNamedPipe")
    if not available.value:
        return False
    buffer = C.create_string_buffer(min(available.value, 16384))
    checked(api.read, reader, buffer, len(buffer), C.byref(count), None)
    captured.extend(buffer.raw[:min(count.value, OUTPUT_LIMIT - len(captured))])
    return True


def monitor(api, job, process, readers, timeout, started):
    streams = [bytearray(), bytearray()]
    accounting = Accounting()
    while True:
        checked(api.query_job, job, 1, C.byref(accounting), C.sizeof(accounting), None)
        if not accounting.active:
            timed_out = False
            break
        if time.monotonic() - started >= timeout:
            timed_out = True
            checked(api.kill_job, job, 124)
            deadline = time.monotonic() + 5
            while accounting.active:
                checked(api.query_job, job, 1, C.byref(accounting), C.sizeof(accounting), None)
                if accounting.active and time.monotonic() >= deadline:
                    raise TimeoutError("Job did not terminate within five seconds")
                time.sleep(0.01)
            break
        activity = [drain(api, reader, stream) for reader, stream in zip(readers, streams)]
        if not any(activity):
            time.sleep(0.01)
    for reader, stream in zip(readers, streams):
        while drain(api, reader, stream):
            pass
    code = DWORD()
    checked(api.exit_code, process, C.byref(code))
    return dict(exit_code=code.value, timed_out=timed_out,
                stdout=streams[0].decode("utf-8", errors="replace"),
                stderr=streams[1].decode("utf-8", errors="replace"),
                seconds=round(time.monotonic() - started, 3))


def launch(api, sid, workspace, runtime, executable, command, timeout, memory, lpac):
    with ExitStack() as stack:
        # The job stays open through collection. Closing it kills any remaining members.
        job = checked(api.create_job, None, None)
        stack.callback(checked, api.close, job)
        limits = ExtendedLimits()
        # Active process, die on unhandled exception, process/job memory, kill on close.
        limits.basic.flags = 0x8 | 0x400 | 0x100 | 0x200 | 0x2000
        limits.basic.active_processes = 32
        limits.process_memory = limits.job_memory = memory
        checked(api.set_job, job, 9, C.byref(limits), C.sizeof(limits))
        # Desktop operations, clipboard reads/writes, global atoms.
        if not os.environ.get("DESMAN_RESEARCH_NO_UI"):  # research toggle: UI limits off
            ui = DWORD(0x40 | 0x2 | 0x4 | 0x20)
            checked(api.set_job, job, 4, C.byref(ui), C.sizeof(ui))

        with ExitStack() as child_ends:
            child_stdin, stdin_writer = HANDLE(), HANDLE()
            attrs = SecurityAttributes(C.sizeof(SecurityAttributes), None, True)
            checked(api.create_pipe, C.byref(child_stdin), C.byref(stdin_writer), C.byref(attrs), 0)
            child_ends.callback(checked, api.close, child_stdin)
            checked(api.close, stdin_writer)  # Child reads EOF.
            out_reader, out_writer = make_output_pipe(api, stack, child_ends)
            err_reader, err_writer = make_output_pipe(api, stack, child_ends)

            inherited = (HANDLE * 3)(child_stdin.value, out_writer.value, err_writer.value)
            values = {
                SECURITY_CAPABILITIES: SecurityCapabilities(sid, None, 0, 0),
                HANDLE_LIST: inherited,
            }
            if lpac:
                values[ALL_APPLICATION_PACKAGES_POLICY] = DWORD(1)
            size = SIZE()
            api.init_attributes(None, len(values), 0, C.byref(size))
            if C.get_last_error() != 122 or not size.value:
                raise win_error("InitializeProcThreadAttributeList sizing")
            attributes = C.create_string_buffer(size.value)
            checked(api.init_attributes, attributes, len(values), 0, C.byref(size))
            child_ends.callback(api.delete_attributes, attributes)
            for key, value in values.items():
                checked(api.update_attribute, attributes, 0, key,
                        C.byref(value), C.sizeof(value), None, None)

            startup = StartupInfoEx()
            startup.startup.cb = C.sizeof(startup)
            startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = inherited
            startup.attributes = C.cast(attributes, PTR)
            extra = "".join(
                f"{name}={os.environ[name]}\0"
                for name in os.environ.get("DESMAN_RESEARCH_ENV", "").split(",")
                if name and name in os.environ
            )
            env = C.create_unicode_buffer(
                f"PATH={runtime}\0SystemRoot={os.environ['SystemRoot']}\0"
                f"TEMP={workspace}\0TMP={workspace}\0"
                # CreateProcessW fails with ERROR_ENVVAR_NOT_FOUND (203) for an AppContainer
                # without it: Windows derives the container's own folder from it (desman-test).
                f"LOCALAPPDATA={os.environ['LOCALAPPDATA']}\0" + extra
            )
            cmdline = C.create_unicode_buffer(subprocess.list2cmdline([str(executable), *command]))
            process = ProcessInfo()
            # Extended startup info, Unicode environment, suspended, no console window.
            flags = 0x80000 | 0x400 | 0x4 | 0x08000000
            checked(api.create_process, str(executable), cmdline, None, None, True, flags,
                    env, str(workspace), C.byref(startup.startup), C.byref(process))
            stack.callback(checked, api.close, process.process)
            stack.callback(checked, api.close, process.thread)
            try:
                checked(api.assign, job, process.process)
            except BaseException:
                checked(api.kill_process, process.process, 125)
                raise
            api.verify_container(process.process, sid)

        started = time.monotonic()
        if api.resume(process.thread) == 0xFFFFFFFF:
            raise win_error("ResumeThread")
        return monitor(api, job, process.process, (out_reader, err_reader), timeout, started)


def make_output_pipe(api, readers, writers):
    reader, writer = HANDLE(), HANDLE()
    attrs = SecurityAttributes(C.sizeof(SecurityAttributes), None, True)
    checked(api.create_pipe, C.byref(reader), C.byref(writer), C.byref(attrs), 0)
    readers.callback(checked, api.close, reader)
    writers.callback(checked, api.close, writer)
    checked(api.handle_flags, reader, 1, 0)  # Only the writer may be inherited.
    return reader, writer


def directory(arg):
    path = Path(os.path.abspath(arg))
    if not path.is_dir() or path == path.parent or str(path).startswith("\\\\"):
        raise ValueError(f"Expected a dedicated local directory: {path}")
    return path


def run(api, journal, args, command):
    workspace, runtime = directory(args.workspace), directory(args.runtime)
    if workspace == runtime or workspace in runtime.parents or runtime in workspace.parents:
        raise ValueError("Workspace and runtime must be disjoint")
    for root in (workspace, runtime):
        if JOURNAL.is_relative_to(root):
            raise ValueError("Keep the launcher and journal outside the ACL trees")
        check_tree(root)
    if any(Path(item["path"]) not in (workspace, runtime) for item in journal["grants"]):
        raise ValueError("Run --cleanup before using different directories")
    executable = Path(command[0])
    if executable.parent == Path("."):
        # Bare command names search the private runtime only.
        executable = runtime / executable
    if not executable.is_absolute():
        raise ValueError("Use a bare runtime command or an absolute .exe path")
    if not executable.suffix:
        executable = executable.with_suffix(".exe")
    if executable.suffix.lower() != ".exe" or not executable.is_file():
        raise ValueError(f"Expected an existing .exe: {executable}")
    journal["profile"] = True
    save_journal(journal)
    sid = PTR()
    result = status(api.create_profile, PROFILE, PROFILE, "DESMAN research", None, 0,
                    C.byref(sid), accepted=(0, 0x800700B7))
    if result:
        status(api.derive_sid, PROFILE, C.byref(sid))
    try:
        sid_bytes = C.string_at(sid, api.sid_size(sid))
        # Directories get traverse; files inherit read/write/delete without execute.
        grant(api, journal, workspace, [
            allow_ace(sid_bytes, 0x02, 0x1301BF),
            allow_ace(sid_bytes, 0x09, 0x13019F),
        ])
        grant(api, journal, runtime, [allow_ace(sid_bytes, 0x03, 0x1200A9)])
        return launch(api, sid, workspace, runtime, executable, command[1:], args.timeout,
                      args.memory * 1024 * 1024, args.lpac)
    finally:
        api.free_sid(sid)


def main():
    parser = argparse.ArgumentParser(description="Run a Windows AppContainer research probe.")
    parser.add_argument("--workspace", metavar="DIR")
    parser.add_argument("--runtime", metavar="PYDIR")
    parser.add_argument("--timeout", type=float, default=30, metavar="S")
    parser.add_argument("--memory", type=int, default=256, metavar="MB")
    parser.add_argument("--lpac", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Remove recorded ACEs and profile.")
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD ARGS...")
    args = parser.parse_args()
    if os.name != "nt":
        parser.exit(2, "AppContainer requires Windows.\n")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.cleanup:
        if args.workspace or args.runtime or args.lpac or command:
            parser.error("--cleanup is a standalone operation")
    elif not (args.workspace and args.runtime and command):
        parser.error("--workspace, --runtime and a command are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not 0 < args.memory * 1024 * 1024 < 1 << (8 * C.sizeof(SIZE)):
        parser.error("--memory is outside the supported range")

    try:
        api = Windows()
        with journal_lock():
            journal = load_journal()
            if args.cleanup:
                cleanup(api, journal)
            else:
                print(json.dumps(run(api, journal, args, command)))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
