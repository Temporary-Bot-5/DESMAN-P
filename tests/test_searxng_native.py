"""Native SearXNG with simulated HTTP, process creation and readiness."""

from __future__ import annotations

import getpass
import hashlib
import io
import json
import os
import runpy
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import _searxng_service, cli, manual, runs, searxng
from desman import searxng_native as native
from desman.config import ConfigManager
from desman.core import db, filelock, search
from desman.core.store import Store

URL = "http://127.0.0.1:8888"


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DESMAN_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data/searxng"


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path / "config.json")


@pytest.mark.parametrize("mode", ["valid", "digest", "size"])
def test_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    payload = b"simulated archive"
    monkeypatch.setattr(native, "SIZE", len(payload) - (mode == "size"))
    monkeypatch.setattr(
        native, "SHA256", "0" * 64 if mode == "digest" else hashlib.sha256(payload).hexdigest()
    )
    file = tmp_path / "archive"
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=payload))
    ) as http:
        if mode == "valid":
            native.download(file, http)
            assert file.read_bytes() == payload
        else:
            with pytest.raises(click.ClickException, match="verification failed|expected size"):
                native.download(file, http)
            assert not file.exists()


def make_archive(archive: Path) -> None:
    root = f"searxng-{native.COMMIT}"
    with tarfile.open(archive, "w:gz") as output:
        for name in (
            f"{root}/requirements.txt",
            f"{root}/src:skip",
            f"{root}/../escape",
            f"{root}//absolute",
            f"{root}/..\\escape-win",
            "/outside",
        ):
            info = tarfile.TarInfo(name)
            info.size = 2
            output.addfile(info, io.BytesIO(b"ok"))
        link = tarfile.TarInfo(f"{root}/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../escape"
        output.addfile(link)


def test_extract(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    make_archive(archive)
    dest = tmp_path / "src"
    native.extract(archive, dest)
    assert [path.name for path in dest.iterdir()] == ["requirements.txt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["archive", "src"]


def configure(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "commit").write_text(native.COMMIT, encoding="ascii")
    (folder / "settings.yml").write_text(
        json.dumps({"server": {"port": 8888, "secret_key": "test"}}), encoding="utf-8"
    )


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_setup_preserves_secret_and_reinstalls_pin(
    folder: Path, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    monkeypatch.setattr(native, "_platform", lambda: platform)
    monkeypatch.setattr(searxng, "find_docker", lambda: None)
    monkeypatch.setattr(searxng, "free_port", lambda port: port or 8888)
    monkeypatch.setattr(native, "download", lambda path, http: make_archive(path))
    monkeypatch.setattr(native, "start", lambda *args, **kwargs: None)
    commands: list[list[str]] = []

    def install(args: list[str], log: Path) -> None:
        commands.append(args)
        executable = native.python(folder)
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()

    monkeypatch.setattr(native, "_install_command", install)
    assert searxng.setup(manager) == URL
    assert native.directory() == folder
    settings = folder / "settings.yml"
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["use_default_settings"]
    assert data["server"] == {
        "port": 8888,
        "secret_key": data["server"]["secret_key"],
        "bind_address": "127.0.0.1",
        "limiter": False,
        "public_instance": False,
    }
    assert len(data["server"]["secret_key"]) == 64
    assert data["search"]["formats"] == ["html", "json"]
    searxng.setup(manager)
    assert len(commands) == 2
    assert commands[0][1:3] == ["-m", "venv"]
    assert commands[1] == [
        str(native.python(folder)),
        "-m",
        "pip",
        "install",
        "-r",
        str(folder / "src/requirements.txt"),
    ]
    (folder / "commit").write_text("old", encoding="ascii")
    searxng.setup(manager)
    assert len(commands) == 4
    assert native.commit(folder) == native.COMMIT
    assert json.loads(settings.read_text(encoding="utf-8")) == data
    monkeypatch.setenv("SECRET_FOR_TEST", "should not be passed")
    for key in (
        "HTTP_PROXY",
        "https_proxy",
        "no_proxy",
        "ALL_PROXY",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
    ):
        monkeypatch.setenv(key, "test-value")
        # Windows keeps environment names in upper case.
        passed = {name.upper(): value for name, value in native.environment(folder).items()}
        assert passed[key.upper()] == "test-value"
    env = native.environment(folder)
    assert "SECRET_FOR_TEST" not in env
    assert env["SEARXNG_SETTINGS_PATH"] == str(settings)
    paths = env["PYTHONPATH"].split(os.pathsep)
    if platform == "win32":
        shim = folder / "shim/pwd.py"
        assert paths == [str(shim.parent), str(folder / "src")]
        pwd = runpy.run_path(str(shim))
        entry = pwd["getpwuid"](17)
        assert isinstance(entry, pwd["struct_passwd"])
        assert pwd["getpwnam"](entry.pw_name).pw_name == entry.pw_name
        with pytest.raises(KeyError):
            pwd["getpwnam"](entry.pw_name + " other")

        def no_username() -> str:
            raise OSError("username is absent from the environment")

        monkeypatch.setattr(getpass, "getuser", no_username)
        assert runpy.run_path(str(shim))["getpwuid"](17).pw_name == Path.home().name
    else:
        assert paths == [str(folder / "src")]
        assert not (folder / "shim").exists()


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_launch_flags(folder: Path, monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    configure(folder)
    monkeypatch.setattr(native, "_platform", lambda: platform)
    calls: list[dict[str, Any]] = []

    def spawn(args: list[str], **kwargs: Any) -> None:
        assert args[0] == str(native.python(folder))
        assert args[1] == "-u"
        assert args[3:] == [str(folder), "token"]
        assert Path(kwargs["stdout"].name) == folder / "searxng.log"
        calls.append(kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    native._launch(folder, "token")
    assert calls[0]["start_new_session"] is (platform == "linux")
    assert calls[0]["creationflags"] == (0x208 if platform == "win32" else 0)


def test_pip_reports_log_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        kwargs["stdout"].write(b"old\n" * 30 + b"dependency failed\n")
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(click.ClickException, match="dependency failed") as caught:
        native._install_command(["python", "-m", "pip"], tmp_path / "log")
    assert str(caught.value).count("old") == 14


@pytest.mark.parametrize("success", [True, False])
def test_start_waits_and_reports_log(
    folder: Path, monkeypatch: pytest.MonkeyPatch, success: bool
) -> None:
    configure(folder)
    launches: list[str] = []
    probes: list[str] = []
    elapsed = 0.0

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    def launch(path: Path, token: str) -> None:
        launches.append(token)
        (path / "searxng.log").write_text("startup detail", encoding="utf-8")

    def answer(req: httpx.Request) -> httpx.Response:
        probes.append(str(req.url))
        return httpx.Response(200 if success and len(probes) == 2 else 503, json={"results": []})

    client = httpx.Client(transport=httpx.MockTransport(answer))
    monkeypatch.setattr(native, "_launch", launch)
    monkeypatch.setattr(
        native, "running", lambda path: bool(launches) and (path / "activation").exists()
    )
    monkeypatch.setattr(native, "listening", lambda address: bool(launches))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    if success:
        native.start(folder, sleep=sleep, clock=lambda: elapsed)
        assert elapsed == 0.5
    else:
        with pytest.raises(click.ClickException, match="startup detail"):
            native.start(folder, sleep=sleep, clock=lambda: elapsed)
        assert not (folder / "activation").exists()
        assert elapsed <= 30
    assert len(launches) == 1
    assert probes[0] == URL + "/search?q=test&format=json"


def test_stop_ignores_foreign_pid(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(folder)
    pidfile = folder / "searxng.pid"
    pidfile.write_text(str(os.getpid()), encoding="ascii")
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("must never signal a PID"))
    native.stop(folder)
    assert pidfile.read_text(encoding="ascii") == str(os.getpid())


@pytest.mark.parametrize("fail", [True, False])
def test_runs_starts_once_or_keeps_other_engines(
    folder: Path,
    manager: ConfigManager,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    clock: FakeClock,
    fail: bool,
) -> None:
    configure(folder)
    config = manager.load()
    config.settings["SEARXNG_URL"] = URL
    config.add_key("TAVILY_API_KEY", "fake")
    starts: list[Path] = []

    def start(path: Path) -> None:
        starts.append(path)
        if fail and len(starts) == 1:
            raise click.ClickException("service offline")

    def ready() -> bool:
        return len(starts) >= (2 if fail else 1)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            assert ready()
        else:
            assert fail and request.url.host == "api.tavily.com"
        return httpx.Response(200, json={"results": []})

    monkeypatch.setattr(native, "start", start)
    monkeypatch.setattr(native, "running", lambda path: ready())
    monkeypatch.setattr(native, "listening", lambda url: ready())
    with (
        db.open_db(":memory:") as conn,
        httpx.Client(transport=httpx.MockTransport(respond)) as http,
    ):
        store = Store(conn, clock=clock)
        web = runs.web(config, store, http)
        assert [engine.name for engine in web.engines] == ["searxng", "tavily"]
        assert starts == []
        assert web.search("first", 1)["engine"] == ("tavily" if fail else "searxng")
        assert starts == [folder]
        if fail:
            assert "service offline" in caplog.text
            marker = folder / "start-failed"
            assert float(marker.read_text(encoding="ascii")) == clock().timestamp()
            clock.advance(599)
            rebuilt = runs.web(config, store, http)
            assert rebuilt.search("during cooldown", 1)["engine"] == "tavily"
            assert starts == [folder]
            clock.advance(1)
            assert rebuilt.search("after cooldown", 1)["engine"] == "searxng"
            assert starts == [folder, folder]
            assert not marker.exists()
        else:
            assert runs.web(config, store, http).search("second", 1)["engine"] == "searxng"
            assert starts == [folder]


def test_native_cli_status_start_stop_remove(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman.config_cli import load_config

    configure(folder)
    with load_config().edit() as config:
        config.settings["SEARXNG_URL"] = URL
    calls: list[str] = []
    monkeypatch.setattr(native, "start", lambda path: calls.append("start"))
    monkeypatch.setattr(native, "stop", lambda path: calls.append("stop"))
    monkeypatch.setattr(native, "running", lambda path: calls[-1:] == ["start"])
    monkeypatch.setattr(searxng, "find_docker", lambda: None)
    monkeypatch.setattr(searxng, "available", lambda: pytest.fail("Docker is not needed"))
    monkeypatch.setattr(search, "check_searxng", lambda url, http: None)
    runner = CliRunner()
    for action in ["start", "stop", "start"]:
        result = runner.invoke(cli.cli, ["search", action])
        assert result.exit_code == 0, result.output
    result = runner.invoke(cli.cli, ["search", "status", "--json"])
    assert result.exit_code == 0, result.output
    status = json.loads(result.output)
    assert (status["kind"], status["running"], status["url"], status["commit"]) == (
        "native",
        True,
        URL,
        native.COMMIT,
    )
    result = runner.invoke(cli.cli, ["search", "remove"])
    assert result.exit_code == 0 and folder.exists()
    result = runner.invoke(cli.cli, ["search", "remove", "--yes"])
    assert result.exit_code == 0, result.output
    assert not folder.exists()
    assert "SEARXNG_URL" not in load_config().load().settings
    assert calls == ["start", "stop", "start", "stop"]


def test_search_manual_describes_native_setup() -> None:
    topic = next(topic for topic in manual.TOPICS if topic.name == "search")
    assert "Python virtual environment" in topic.text


def test_service_refuses_stale_activation(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configure(folder)
    (folder / "activation").write_text("different-token", encoding="ascii")
    monkeypatch.setattr(runpy, "run_module", lambda *a, **kw: pytest.fail("stale child ran"))
    _searxng_service.main(folder, "old-token")
    assert not native.running(folder)


def test_configured_docker_url_takes_precedence(
    folder: Path, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(folder)
    with manager.edit() as config:
        config.settings.update({"SEARXNG_URL": URL, searxng.DOCKER_URL: URL})
    monkeypatch.setattr(searxng, "available", lambda: None)
    monkeypatch.setattr(searxng, "inspect", lambda: {"Config": {"Image": searxng.IMAGE}})
    monkeypatch.setattr(search, "check_searxng", lambda *args: None)
    monkeypatch.setattr(native, "start", lambda *args: pytest.fail("selected Docker, not native"))
    status = searxng.status(manager, display=False)
    assert (status["kind"], status["commit"], status["url"]) == ("docker", native.COMMIT, URL)
    with db.open_db(":memory:") as conn, search.client() as http:
        assert [engine.name for engine in runs.web(manager.load(), Store(conn), http).engines] == [
            "searxng"
        ]


def test_extract_requires_safe_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    target = tmp_path / "src"
    with pytest.raises(click.ClickException, match="Python 3.11.4"):
        native.extract(tmp_path / "archive", target)
    assert not target.exists()


def test_concurrent_searches_wait_for_start_and_restart_after_stop(
    folder: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock
) -> None:
    configure(folder)
    starting = threading.Event()
    contended = threading.Event()
    ready = threading.Event()
    starts: list[Path] = []
    lock = filelock.lock

    def observed_lock(fd: int, *, blocking: bool) -> None:
        try:
            lock(fd, blocking=blocking)
        except BlockingIOError:
            contended.set()
            raise

    def start(path: Path) -> None:
        starts.append(path)
        ready.set()
        starting.set()
        assert contended.wait(5)

    monkeypatch.setattr(filelock, "lock", observed_lock)
    monkeypatch.setattr(native, "start", start)
    monkeypatch.setattr(native, "running", lambda path: ready.is_set())
    monkeypatch.setattr(native, "listening", lambda address: ready.is_set())
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(searxng.ensure_native, URL, clock=clock)
        assert starting.wait(5)
        second = workers.submit(searxng.ensure_native, URL, clock=clock)
        assert first.result(timeout=5)
        assert second.result(timeout=5)
    assert starts == [folder]
    ready.clear()
    assert searxng.ensure_native(URL, clock=clock)
    assert starts == [folder, folder]


@pytest.mark.parametrize("unexpected", [False, True])
def test_watcher_exits_if_activation_becomes_unreadable(
    folder: Path, monkeypatch: pytest.MonkeyPatch, unexpected: bool
) -> None:
    configure(folder)
    reads = []

    def read(path: Path, *, encoding: str) -> str:
        reads.append(path)
        if len(reads) == 1:
            return "token"
        if unexpected:
            raise ValueError("bad activation")
        raise PermissionError("pending delete")

    class ExitService(Exception):
        pass

    def exit_service(code: int) -> None:
        assert code == 0
        raise ExitService

    class InlineThread:
        def __init__(self, *, target: Any, daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(threading, "Thread", InlineThread)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(os, "_exit", exit_service)
    with pytest.raises(ExitService):
        _searxng_service.main(folder, "token")
    assert reads == [folder / "activation"] * (2 if unexpected else 4)


def test_docker_takeover_stops_native_and_remove_cleans_both(
    folder: Path, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(folder)
    events: list[str] = []
    monkeypatch.setattr(searxng, "find_docker", lambda: "docker")
    monkeypatch.setattr(searxng, "available", lambda **kwargs: None)
    monkeypatch.setattr(native, "stop", lambda path, **kw: events.append("native stop"))

    def setup_docker(*args: Any) -> str:
        events.append("docker setup")
        return URL

    monkeypatch.setattr(searxng, "_setup_docker", setup_docker)
    monkeypatch.setattr(searxng, "inspect", lambda: {"State": {"Status": "running"}})
    monkeypatch.setattr(searxng, "docker", lambda args: events.append(args[0]))
    searxng.setup(manager)
    assert searxng.uses_docker(manager.load())
    searxng.operate(manager, "stop")
    assert manager.load().settings["SEARXNG_URL"] == URL
    searxng.operate(manager, "remove")
    assert events == ["native stop", "docker setup", "native stop", "stop", "native stop", "rm"]
    assert not folder.exists()
    assert not manager.load().settings


def test_status_with_native_install_and_external_url(
    folder: Path, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure(folder)
    with manager.edit() as config:
        config.settings["SEARXNG_URL"] = "https://search.example.com"
    monkeypatch.setattr(searxng, "find_docker", lambda: None)
    monkeypatch.setattr(search, "check_searxng", lambda url, http: None)
    result = searxng.status(manager, display=False)
    assert result["kind"] == "external"
    assert result["url"] == "https://search.example.com"
    assert result["commit"] is None
    assert result["answers"]
