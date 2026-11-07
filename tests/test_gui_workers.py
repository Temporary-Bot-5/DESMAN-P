"""Page pool lifecycle, deterministic supervision and isolated process integration."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import click
import pytest

from conftest import FakeClock
from desman.config import Config, Tuning, upgrade
from desman.core import db
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.store import Store
from desman.gui.workers import Pool, Process


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO()
        self.code: int | None = None
        self.signals: list[int] = []
        self.waits: list[float | None] = []
        self.terminated = False
        self.killed = False
        self.hang = False

    def poll(self) -> int | None:
        return self.code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self.hang and not self.killed:
            raise subprocess.TimeoutExpired("worker", timeout or 0)
        self.code = 0
        return 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


class Launcher:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, str], Path]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, argv: list[str], env: dict[str, str], path: Path) -> Process:
        self.calls.append((argv, env, path))
        process = FakeProcess()
        self.processes.append(process)
        return process


def test_pool_launches_same_python_database_and_config_and_stops(tmp_path: Path) -> None:
    launcher = Launcher()
    pool = Pool(tmp_path / "custom.db", tmp_path / "config", launcher=launcher, guard=lambda: None)
    pool.tick()
    assert len(launcher.calls) == 3
    for n, (argv, env, log) in enumerate(launcher.calls, 1):
        assert argv == [
            sys.executable,
            "-m",
            "desman",
            "worker",
            "--db",
            str(tmp_path / "custom.db"),
        ]
        assert env["DESMAN_DATA_DIR"] == str(tmp_path)
        assert env["DESMAN_CONFIG_DIR"] == str(tmp_path / "config")
        assert log == tmp_path / "workers" / f"worker-{n}.log"
    pool.stop()
    for process in launcher.processes:
        assert process.signals
        assert process.waits and process.waits[0] is not None and process.waits[0] <= 10
    pool.tick()
    assert len(launcher.calls) == 3


def test_backoff_and_five_quick_deaths_stop_only_that_slot(
    tmp_path: Path, clock: FakeClock
) -> None:
    launcher = Launcher()
    pool = Pool(
        tmp_path / "db",
        tmp_path,
        size=2,
        launcher=launcher,
        clock=lambda: clock().timestamp(),
        guard=lambda: None,
    )
    pool.tick()
    for death in range(1, 6):
        launcher.processes[-1 if death > 1 else 0].code = 1
        pool.tick()
        count = len(launcher.calls)
        if death < 5:
            pool.tick()
            assert len(launcher.calls) == count
            clock.advance(2 ** (death - 1))
            pool.tick()
            assert len(launcher.calls) == count + 1
    assert pool.slots[0].stopped and not pool.slots[1].stopped
    assert len(pool.messages) == 1 and "worker-1.log" in pool.messages[0]
    clock.advance(1000)
    pool.tick()
    assert len(launcher.calls) == 6
    pool.stop()


def test_long_lived_worker_resets_death_streak(tmp_path: Path, clock: FakeClock) -> None:
    launcher = Launcher()
    pool = Pool(
        tmp_path / "db",
        tmp_path,
        1,
        launcher=launcher,
        clock=lambda: clock().timestamp(),
        guard=lambda: None,
    )
    pool.tick()
    pool.slots[0].deaths = 4
    clock.advance(61)
    launcher.processes[0].code = 1
    pool.tick()
    assert pool.slots[0].deaths == 1 and not pool.slots[0].stopped


def test_protected_guard_warning_is_once_and_never_restarted(tmp_path: Path) -> None:
    launcher = Launcher()
    blocked = False

    def guard() -> None:
        if blocked:
            raise click.ClickException("Protected files changed: approve them")

    pool = Pool(tmp_path / "db", tmp_path, 1, launcher=launcher, guard=guard)
    pool.tick()
    blocked = True
    launcher.processes[0].code = 1
    pool.tick()
    pool.tick()
    assert list(pool.messages) == ["Protected files changed: approve them"]
    assert len(launcher.calls) == 1 and pool.blocked


def test_hung_workers_are_terminated_then_killed(tmp_path: Path) -> None:
    launcher = Launcher()
    pool = Pool(tmp_path / "db", tmp_path, 1, launcher=launcher, guard=lambda: None)
    pool.tick()
    launcher.processes[0].hang = True
    pool.stop()
    assert launcher.processes[0].terminated and launcher.processes[0].killed


def test_schema_six_upgrades_without_changing_tuning() -> None:
    config = Config.model_validate(upgrade({"schema_version": 6, "tuning": {"max_subtasks": 4}}))
    assert config.schema_version == 8 and config.tuning.workers is None
    assert config.tuning.max_subtasks == 4
    assert Tuning(workers=8).workers == 8
    with pytest.raises(ValueError):
        Tuning(workers=9)


@pytest.mark.integration
def test_one_real_worker_finishes_a_task(tmp_path: Path) -> None:
    path = tmp_path / "desman.db"
    conn = db.open_db(path)
    store = Store(conn)
    task = store.create_task(
        TaskSpec(
            goal="example",
            subtasks=[SubtaskSpec(description="say hello", acceptance_criteria="hello")],
        )
    )
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {
                "routes": [
                    {"match": "Execute", "steps": ['{"hello": "world"}']},
                    {"match": "Review", "steps": ['{"verdict": "accept"}']},
                ]
            }
        )
    )
    pool = Pool(path, tmp_path, 1)
    pool.argv.extend(["--fake-llm", str(script)])
    try:
        pool.start()
        deadline = time.monotonic() + 30
        while store.task_status(task).value != "done":
            assert time.monotonic() < deadline, list(pool.messages)
            time.sleep(0.05)
    finally:
        pool.stop()
        conn.close()
    assert (tmp_path / "workers" / "worker-1.log").exists()


def test_idle_line_checks_every_two_seconds_without_replacing_prompt(
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman.gui.front import WebFront

    front = WebFront(clock=lambda: clock().timestamp())
    checked: list[float] = []

    def idle() -> None:
        checked.append(clock().timestamp())
        assert front.prompt is not None
        assert front.answer({"id": front.prompt["id"], "value": "hello"})

    def wait(timeout: float | None = None) -> bool:
        assert timeout is not None and timeout > 0
        clock.advance(timeout)
        return False

    monkeypatch.setattr(front.condition, "wait", wait)
    front.idle = idle
    assert front.line() == "hello"
    assert checked == [clock().timestamp()]
    assert clock().second == 2


def test_outcome_marker_migration_from_previous_schema() -> None:
    conn = db.connect(":memory:")
    try:
        db.migrate(conn, db.MIGRATIONS[:-1])
        assert "outcome_audit" not in {
            r["name"] for r in conn.execute("PRAGMA table_info(messages)")
        }
        db.migrate(conn)
        assert "outcome_audit" in {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    finally:
        conn.close()


def test_missing_executable_gives_up_without_spinning(tmp_path: Path, clock: FakeClock) -> None:
    count = 0

    def unavailable(argv: list[str], env: dict[str, str], path: Path) -> Process:
        nonlocal count
        count += 1
        raise OSError("cannot launch")

    pool = Pool(
        tmp_path / "db",
        tmp_path,
        1,
        launcher=unavailable,
        clock=lambda: clock().timestamp(),
        guard=lambda: None,
    )
    for _ in range(5):
        pool.tick()
        pool.tick()
        clock.advance(60)
    assert count == 5 and pool.slots[0].stopped and len(pool.messages) == 1


def test_task_queue_requires_busy_workers_and_clears_when_claimed(tmp_path: Path) -> None:
    from desman.core.sessions import Role, Sessions
    from desman.gui.front import WebFront
    from desman.gui.panels import snapshot

    path = tmp_path / "db"
    conn = db.open_db(path)
    try:
        store = Store(conn)
        sid = Sessions(conn).add(None, Role.USER, "go").session_id
        spec = TaskSpec(
            goal="work", subtasks=[SubtaskSpec(description="step", acceptance_criteria="ok")]
        )
        first = store.create_task(spec, session_id=sid)
        second = store.create_task(spec, session_id=sid)
        front = WebFront()
        front.worker_count = 1
        front.session(sid, False)
        assert not any(t["queued"] for t in snapshot(path, "tasks", front)["tasks"])
        claimed = store.claim("worker")
        assert claimed is not None
        data = {t["id"]: t for t in snapshot(path, "tasks", front)["tasks"]}
        assert not data[claimed.task_id]["queued"]
        assert data[second if claimed.task_id == first else first]["queued"]
    finally:
        conn.close()


def test_workers_cli_and_catalog_use_shared_bounds() -> None:
    from click.testing import CliRunner

    from desman.cli import cli
    from desman.commands import TuningArgs
    from desman.config import open_config
    from desman.settings_actions import Limits

    assert CliRunner().invoke(cli, ["tuning", "--workers", "8"]).exit_code == 0
    assert open_config().load().tuning.workers == 8
    assert TuningArgs(workers=2, models=None, keys=None).change().applied(Tuning()).workers == 2
    assert Limits(section="limits", name="workers", value=3).value == 3
    assert CliRunner().invoke(cli, ["tuning", "--workers", "9"]).exit_code != 0
    assert open_config().load().tuning.workers == 8


def test_worker_notices_go_to_general_terminal_once(tmp_path: Path) -> None:
    from desman import ui
    from desman.chat import Chat
    from desman.core.sessions import Sessions
    from desman.gui.front import WebFront

    conn = db.open_db(tmp_path / "db")
    try:
        sessions = Sessions(conn)
        page = Chat(sessions, background_tasks=True)
        front = WebFront()
        with ui.use(front):
            page.worker_notice("Protected files changed", blocked=True)
            page.worker_notice("Protected files changed", blocked=True)
        assert not front.events
        assert len(sessions.messages(sessions.terminal())) == 1
        page.session_id = sessions.terminal()
        with ui.use(front):
            page.worker_notice("Worker failed: worker-1.log")
        assert any("worker-1.log" in e.get("html", "") for e in front.events)
    finally:
        conn.close()


def test_launcher_rotates_output_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Callable
    from logging.handlers import RotatingFileHandler
    from typing import Any

    from desman.gui import workers

    observed: dict[str, Any] = {}
    process = FakeProcess()
    process.stdout = io.BytesIO(b"worker output line\n" * 30)

    def popen(argv: list[str], **kwargs: Any) -> FakeProcess:
        observed.update(kwargs)
        observed["argv"] = argv
        return process

    class Thread:
        def __init__(self, target: Callable[[], None], *, daemon: bool) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

    def handler(path: Path, **kwargs: Any) -> RotatingFileHandler:
        return RotatingFileHandler(path, maxBytes=80, backupCount=2, encoding="utf-8")

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(threading, "Thread", Thread)
    monkeypatch.setattr(workers, "RotatingFileHandler", handler)
    path = tmp_path / "worker.log"
    workers.launch([sys.executable, "-m", "desman", "worker"], {"EXAMPLE": "value"}, path)
    assert observed["env"] == {"EXAMPLE": "value"}
    assert observed.get("shell", False) is False
    assert observed["start_new_session"] is (sys.platform != "win32")
    assert observed["stdout"] == subprocess.PIPE and observed["stderr"] == subprocess.STDOUT
    assert path.exists() and path.with_suffix(".log.1").exists()
    assert all(p.stat().st_size <= 80 for p in tmp_path.glob("worker.log*"))
