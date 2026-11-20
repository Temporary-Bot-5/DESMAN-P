"""Pinned native SearXNG install and local service lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import click
import httpx

from desman.core import filelock
from desman.core.atomic import replace_file, retry_sharing
from desman.core.db import default_db_path
from desman.i18n import t

COMMIT = "2d500cb6b8e98b5f57fb446786e91f8de03be89f"
ARCHIVE = f"https://github.com/searxng/searxng/archive/{COMMIT}.tar.gz"
SHA256 = "d214b4d411ebecbc913ab9a1548d43ce818dde49e6676a88bd547f46665fb487"
SIZE = 6_742_172
CONTACT = "https://github.com/Temporary-Bot-5/DESMAN-P"

PWD_SHIM = '''"""Subset of pwd used by SearXNG on Windows."""
from collections import namedtuple
from getpass import getuser
from pathlib import Path

struct_passwd = namedtuple(
    "struct_passwd", "pw_name pw_passwd pw_uid pw_gid pw_gecos pw_dir pw_shell"
)

def getpwuid(uid):
    home = Path.home()
    try:
        user = getuser()
    except (OSError, AttributeError, ImportError):
        # The service environment deliberately omits USERNAME.
        user = home.name
    return struct_passwd(user, "", uid, 0, "", str(home), "")

def getpwnam(name):
    entry = getpwuid(0)
    if name != entry.pw_name:
        raise KeyError(name)
    return entry
'''


def directory() -> Path:
    return default_db_path().parent.resolve() / "searxng"


def _platform() -> str:
    return sys.platform


def python(folder: Path) -> Path:
    return folder / ("venv/Scripts/python.exe" if _platform() == "win32" else "venv/bin/python")


def environment(folder: Path) -> dict[str, str]:
    allowed = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "LANG",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    }
    if _platform() == "win32":
        allowed.update(("SYSTEMROOT", "TEMP", "TMP"))
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    paths = [str(folder / "src")]
    if _platform() == "win32":
        paths.insert(0, str(folder / "shim"))
    return {
        **env,
        "PYTHONPATH": os.pathsep.join(paths),
        "SEARXNG_SETTINGS_PATH": str(folder / "settings.yml"),
    }


def download(target: Path, http: httpx.Client) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with http.stream("GET", ARCHIVE) as response, target.open("wb") as output:
            response.raise_for_status()
            for chunk in response.iter_bytes(64 * 1024):
                size += len(chunk)
                if size > SIZE:
                    raise click.ClickException(t("SearXNG archive exceeds the expected size."))
                digest.update(chunk)
                output.write(chunk)
        if size != SIZE or digest.hexdigest() != SHA256:
            raise click.ClickException(t("SearXNG archive verification failed."))
    except (OSError, httpx.HTTPError, click.ClickException):
        target.unlink(missing_ok=True)
        raise


def extract(archive: Path, target: Path) -> None:
    if not hasattr(tarfile, "data_filter"):
        raise click.ClickException(
            t("Native SearXNG needs Python 3.11.4 or newer for safe extraction.")
        )
    root = f"searxng-{COMMIT}/"
    target.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            if not member.name.startswith(root) or ":" in member.name or "\\" in member.name:
                continue
            member.name = member.name.removeprefix(root)
            if not member.name or member.name.startswith("/"):
                continue
            if not (member.isfile() or member.isdir()) or ".." in Path(member.name).parts:
                continue
            try:
                source.extract(member, target, filter="data")
            except tarfile.FilterError:
                continue


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as log:
            log.seek(max(0, path.stat().st_size - 8192))
            return "\n".join(log.read(8192).decode("utf-8", errors="replace").splitlines()[-15:])
    except OSError:
        return ""


def _install_command(args: list[str], log: Path) -> None:
    with log.open("wb") as output:
        result = subprocess.run(
            args, stdout=output, stderr=subprocess.STDOUT, timeout=900, check=False
        )
    if result.returncode:
        raise click.ClickException(t("Native SearXNG failed: {error}", error=_tail(log)))


def commit(folder: Path) -> str | None:
    try:
        return (folder / "commit").read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None


def _settings(folder: Path) -> tuple[int | None, str | None]:
    try:
        server = json.loads((folder / "settings.yml").read_text(encoding="utf-8"))["server"]
        port, secret = server["port"], server["secret_key"]
        if type(port) is int and 1 <= port <= 65535 and isinstance(secret, str) and secret:
            return port, secret
    except (FileNotFoundError, ValueError, TypeError, KeyError):
        pass
    return None, None


def url(folder: Path) -> str:
    port, _ = _settings(folder)
    return f"http://127.0.0.1:{port}" if port else ""


@contextmanager
def locked(folder: Path, *, timeout: float = 0) -> Iterator[None]:
    folder.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Keep the command lock outside the removable install directory.
    with folder.with_suffix(".lock").open("a+b") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                filelock.lock(handle.fileno(), blocking=False)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise click.ClickException(
                        t("Another native SearXNG operation is in progress.")
                    ) from exc
                time.sleep(min(0.05, remaining))
        try:
            yield
        finally:
            filelock.unlock(handle.fileno())


def running(folder: Path) -> bool:
    try:
        with (folder / "service.lock").open("r+b") as handle:
            try:
                filelock.lock(handle.fileno(), blocking=False)
            except BlockingIOError:
                return True
            filelock.unlock(handle.fileno())
    except FileNotFoundError:
        pass
    return False


def stop(folder: Path, *, sleep: Callable[[float], None] = time.sleep) -> None:
    # Only the process holding this install's service lock observes the activation token.
    retry_sharing(lambda: (folder / "activation").unlink(missing_ok=True), sleep=sleep)
    for _ in range(50):
        if not running(folder):
            return
        sleep(0.1)
    raise click.ClickException(t("SearXNG did not stop in time."))


def _launch(folder: Path, token: str) -> None:
    flags = 0
    windows = _platform() == "win32"
    if windows:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200) | getattr(
            subprocess, "DETACHED_PROCESS", 8
        )
    args = [
        str(python(folder)),
        "-u",
        str(Path(__file__).with_name("_searxng_service.py").resolve()),
        str(folder),
        token,
    ]
    with (folder / "searxng.log").open("wb") as log:
        subprocess.Popen(
            args,
            cwd=folder / "src",
            env=environment(folder),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=not windows,
            creationflags=flags,
        )


def _probe(url: str, http: httpx.Client) -> bool:
    try:
        response = http.get(f"{url}/search", params={"q": "test", "format": "json"})
        return response.is_success and isinstance(response.json().get("results"), list)
    except (httpx.HTTPError, ValueError, AttributeError):
        return False


def listening(address: str) -> bool:
    # Called only for the exact loopback URL from the native settings.
    try:
        with socket.create_connection(("127.0.0.1", int(address.rsplit(":", 1)[1])), timeout=0.2):
            return True
    except OSError:
        return False


def start(
    folder: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    address = url(folder)
    if not address or not commit(folder):
        raise click.ClickException(t("SearXNG URL is not configured. Run `desman search setup`."))
    if not running(folder):
        if listening(address):
            raise click.ClickException(t("No free local port for SearXNG."))
        token = secrets.token_hex(32)
        activation = folder / "activation"
        activation.write_text(token, encoding="ascii")
        try:
            _launch(folder, token)
        except OSError:
            activation.unlink(missing_ok=True)
            raise
    deadline = clock() + 30
    with httpx.Client(timeout=5, trust_env=False) as http:
        for _ in range(60):
            if running(folder) and listening(address) and _probe(address, http):
                return
            if clock() >= deadline:
                break
            sleep(0.5)
    stop(folder, sleep=sleep)
    raise click.ClickException(
        t("SearXNG did not answer: {error}", error=_tail(folder / "searxng.log"))
    )


def _write(path: Path, content: str) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def setup(
    folder: Path,
    port: int | None,
    free_port: Callable[[int | None], int],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    with locked(folder):
        current = commit(folder)
        old_port, secret = _settings(folder)
        reinstall = current != COMMIT or not python(folder).is_file()
        if reinstall:
            with tempfile.TemporaryDirectory(dir=folder.parent) as scratch:
                archive = Path(scratch) / "source.tar.gz"
                source = Path(scratch) / "src"
                with httpx.Client(follow_redirects=True, timeout=30) as http:
                    download(archive, http)
                extract(archive, source)
                stop(folder, sleep=sleep)
                folder.mkdir(mode=0o700, exist_ok=True)
                marker = folder / "commit"
                marker.unlink(missing_ok=True)
                for name in ("src", "venv"):
                    target = folder / name
                    if target.exists():
                        shutil.rmtree(target)
                source.rename(folder / "src")
                log = folder / "searxng.log"
                _install_command([sys.executable, "-m", "venv", str(folder / "venv")], log)
                _install_command(
                    [
                        str(python(folder)),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(folder / "src/requirements.txt"),
                    ],
                    log,
                )
        if _platform() == "win32":
            shim = folder / "shim"
            shim.mkdir(exist_ok=True)
            (shim / "pwd.py").write_text(PWD_SHIM, encoding="utf-8")
        if port is not None and old_port != port:
            stop(folder, sleep=sleep)
        if old_port and (port is None or port == old_port) and running(folder):
            chosen = old_port
        else:
            chosen = free_port(port)
        _write(
            folder / "settings.yml",
            json.dumps(
                {
                    "use_default_settings": True,
                    "general": {"contact_url": CONTACT},
                    "server": {
                        "secret_key": secret or secrets.token_hex(32),
                        "bind_address": "127.0.0.1",
                        "port": chosen,
                        "limiter": False,
                        "public_instance": False,
                    },
                    "search": {"formats": ["html", "json"]},
                },
                indent=2,
            ),
        )
        _write(folder / "commit", COMMIT)
        start(folder, sleep=sleep)
        return url(folder)
