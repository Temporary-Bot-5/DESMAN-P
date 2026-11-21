"""Run harmless access checks from inside the research AppContainer."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile


def bind(dll, name, result, *args):
    function = getattr(dll, name)
    function.restype = result
    function.argtypes = args
    return function


def write_probe(directory):
    with tempfile.NamedTemporaryFile(dir=directory, prefix="desman-probe-") as file:
        file.write(b"AppContainer research")
        file.flush()
    return "Created, wrote and removed a temporary file"


def listing(directory):
    return f"Listed {sum(1 for _ in directory.iterdir())} entries"


def registry_probe():
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Software", access=winreg.KEY_READ) as key:
        children, values, _ = winreg.QueryInfoKey(key)
    return f"Opened HKCU Software view ({children} subkeys, {values} values)"


def read_hosts(path):
    with path.open("rb") as file:
        return f"Read {len(file.read(64))} bytes"


def connect(host, port):
    with socket.create_connection((host, port), timeout=2):
        return "TCP connection opened"


def launch(path, *args):
    # An exit failure after a successful create still counts as a successful start.
    with subprocess.Popen(
        [str(path), *args], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ) as process:
        try:
            process.communicate(timeout=6)
            return f"Started; exit code {process.returncode}"
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return "Started; stopped after six seconds"


def pipe_probe(name):
    try:
        import ctypes as C
        from ctypes import wintypes as W
    except ImportError:
        with open(name, "rb"):
            return "Opened host pipe for reading (plain open)"
    kernel = C.WinDLL("kernel32", use_last_error=True)
    create_file = bind(kernel, "CreateFileW", W.HANDLE, W.LPCWSTR, W.DWORD, W.DWORD,
                       C.c_void_p, W.DWORD, W.DWORD, W.HANDLE)
    close = bind(kernel, "CloseHandle", W.BOOL, W.HANDLE)
    # SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION limits server impersonation.
    handle = create_file(name, 0x80000000, 0, None, 3, 0x110000, None)
    if handle == C.c_void_p(-1).value:
        raise C.WinError(C.get_last_error())
    if not close(handle):
        raise C.WinError(C.get_last_error())
    return "Opened host pipe for reading"


def enum_processes():
    import ctypes as C  # ImportError → the check reports it
    from ctypes import wintypes as W

    kernel = C.WinDLL("kernel32", use_last_error=True)
    psapi = C.WinDLL("psapi", use_last_error=True)
    enumerate_pids = bind(psapi, "EnumProcesses", W.BOOL, C.POINTER(W.DWORD), W.DWORD,
                          C.POINTER(W.DWORD))
    open_process = bind(kernel, "OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    close = bind(kernel, "CloseHandle", W.BOOL, W.HANDLE)
    size, used = 1024, W.DWORD()
    while True:
        pids = (W.DWORD * size)()
        if not enumerate_pids(pids, C.sizeof(pids), C.byref(used)):
            raise C.WinError(C.get_last_error())
        if used.value < C.sizeof(pids):
            break
        size *= 2
    others = {pid for pid in pids[:used.value // C.sizeof(W.DWORD)] if pid not in (0, os.getpid())}
    opened = 0
    for pid in others:
        handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            opened += 1
            if not close(handle):
                raise C.WinError(C.get_last_error())
    return f"Enumerated {len(others)} other PIDs; opened {opened} for limited query"


def pytest_probe():
    if importlib.util.find_spec("pytest") is None:
        return "Skipped; pytest is unavailable"
    import pytest

    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="pytest-probe-") as directory:
        path = Path(directory) / "test_smoke.py"
        path.write_text("def test_smoke():\n    assert 1 + 1 == 2\n", encoding="utf-8")
        # Keep pytest output off the JSON-lines stream.
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = pytest.main(["-q", "-p", "no:cacheprovider", str(path)])
    if code:
        raise RuntimeError(f"pytest returned {int(code)}")
    return "Workspace smoke test passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-profile", type=Path, required=True)
    parser.add_argument("--user-temp", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--loopback-port", type=int, required=True)
    parser.add_argument("--pipe", required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.exit(2, "The access probe requires Windows.\n")

    windows = Path(os.environ["SystemRoot"])
    system32 = windows / "System32"
    checks = [
        ("documents_list", lambda: listing(args.user_profile / "Documents")),
        ("ssh_list", lambda: listing(args.user_profile / ".ssh")),
        ("hkcu_software", registry_probe),
        ("user_profile_write", lambda: write_probe(args.user_profile)),
        ("user_temp_write", lambda: write_probe(args.user_temp)),
        ("public_write", lambda: write_probe(args.public)),
        ("hosts_read", lambda: read_hosts(system32 / "drivers/etc/hosts")),
        ("drive_root_list", lambda: listing(Path(windows.anchor))),
        ("outbound_tcp", lambda: connect("1.1.1.1", 80)),
        ("dns", lambda: f"Resolved {len(socket.getaddrinfo('example.com', None))} records"),
        ("loopback", lambda: connect("127.0.0.1", args.loopback_port)),
        ("cmd_start", lambda: launch(system32 / "cmd.exe", "/d", "/c", "echo probe")),
        ("powershell_start", lambda: launch(
            system32 / "WindowsPowerShell/v1.0/powershell.exe",
            "-NoProfile", "-NonInteractive", "-Command", "exit 0",
        )),
        ("host_pipe", lambda: pipe_probe(args.pipe)),
        ("process_enumeration", enum_processes),
        ("workspace_write", lambda: write_probe(Path.cwd())),
        ("pytest", pytest_probe),
    ] + [
        (f"import_{module}", lambda module=module: f"Imported {__import__(module).__name__}")
        for module in ("ctypes", "ssl", "sqlite3", "lzma", "decimal", "_multiprocessing")
    ]
    for name, check in checks:
        try:
            detail = check()
            allowed = not detail.startswith("Skipped;")
        except Exception as error:
            allowed = False
            detail = f"{type(error).__name__}: {error}"
        print(json.dumps(dict(check=name, allowed=allowed, detail=detail)), flush=True)


if __name__ == "__main__":
    main()
