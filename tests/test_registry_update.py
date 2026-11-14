"""Release selection, schema consent, rollback and printed-key assignments."""

from __future__ import annotations

import configparser
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.core import filelock
from desman.net import sign
from desman.registry import admin, store, update
from desman.registry.cli import cli
from test_registry_ops import ready as ready  # shared isolated signing fixture


class FakeRunner:
    def __init__(self, app: Path, directory: Path) -> None:
        self.app = app
        self.directory = directory
        self.calls: list[list[str]] = []
        self.head = "old"
        self.tags = "v2\nv1"
        self.ancestor = True
        self.dirty = False
        self.schema: int | None = len(store.MIGRATIONS)
        self.health_fails = False
        self.bump = False
        self.fail: str | None = None

    def __call__(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        out = ""
        code = 0
        if args[:3] == ["git", "rev-parse", "--show-toplevel"]:
            out = str(self.app)
        elif args == [
            "git",
            "-c",
            "versionsort.suffix=-",
            "tag",
            "--merged",
            "origin/master",
            "--sort=-v:refname",
            "v*",
        ]:
            out = self.tags
        elif args == ["git", "rev-parse", "HEAD"]:
            out = self.head
        elif args[:2] == ["git", "rev-parse"]:
            out = "old" if "v1" in args[2] else "new"
        elif args[:2] == ["git", "merge-base"]:
            code = 0 if self.ancestor else 1
        elif args == ["git", "status", "--porcelain"]:
            out = " M file" if self.dirty else ""
        elif args[:3] == [sys.executable, "-I", "-c"]:
            out = str(self.schema)
            code = 1 if self.schema is None else 0
        elif Path(args[0]).name == "desman-registry":
            code = 1 if self.health_fails else 0
        elif args[:3] == ["systemctl", "--user", "restart"] and self.bump:
            self.bump = False
            with sqlite3.connect(self.directory / store.DB_NAME) as conn:
                conn.execute(f"PRAGMA user_version={len(store.MIGRATIONS) + 1}")
                conn.execute("CREATE TABLE migration_marker (n)")
        if self.fail is not None and self.fail in args:
            self.fail = None
            code = 1
        return subprocess.CompletedProcess(args, code, out, "private output must not be logged")


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, FakeRunner]:
    directory = tmp_path / "data"
    store.open_registry(directory / store.DB_NAME).close()
    app = tmp_path / "app"
    app.mkdir()
    return directory, FakeRunner(app, directory)


def entries(directory: Path) -> list[store.LogEntry]:
    with store.connection(directory / store.DB_NAME) as conn:
        return [e for e in store.entries(conn) if e.action.startswith("update_")]


def execute(
    directory: Path, runner: FakeRunner, clock: FakeClock, **kwargs: object
) -> tuple[str, bool]:
    # Explicit typed options keep the fake runner narrow and test call sites readable.
    return update.update(
        directory,
        app=runner.app,
        runner=runner,
        sleep=clock.sleep,
        now=clock(),
        check=bool(kwargs.get("check", False)),
        allow_schema=bool(kwargs.get("allow_schema", False)),
        to=str(kwargs["to"]) if "to" in kwargs else None,
    )


@pytest.mark.parametrize("mode", ["same", "downgrade", "no tags"])
def test_up_to_date(setup: tuple[Path, FakeRunner], clock: FakeClock, mode: str) -> None:
    directory, runner = setup
    if mode == "same":
        runner.head = "new"
    elif mode == "downgrade":
        runner.ancestor = False
    else:
        runner.tags = ""
    assert execute(directory, runner, clock) == ("Registry is up to date.", True)
    assert not entries(directory)
    assert not any(c[0] == "systemctl" for c in runner.calls)


def test_apply_order_and_journal(
    setup: tuple[Path, FakeRunner], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, runner = setup
    original = admin.backup

    def backup(source: Path, dest: Path) -> None:
        runner.calls.append(["backup"])
        original(source, dest)

    monkeypatch.setattr(admin, "backup", backup)
    assert execute(directory, runner, clock)[1]
    start = runner.calls.index(["systemctl", "--user", "stop", *update.TIMERS])
    assert runner.calls[start:] == [
        ["systemctl", "--user", "stop", *update.TIMERS],
        ["backup"],
        ["git", "checkout", "--detach", "new"],
        [str(runner.app / ".venv/bin/python"), "-m", "pip", "install", "-e", "."],
        ["systemctl", "--user", "restart", "desman-registry.service"],
        [str(runner.app / ".venv/bin/desman-registry"), "--dir", str(directory), "health"],
        ["systemctl", "--user", "start", *update.TIMERS],
    ]
    journal = entries(directory)
    assert journal[-1].action == "update_applied"
    assert journal[-1].details["from"] == "old"
    assert journal[-1].details["to"] == "v2"
    backup_path = directory / "backups" / journal[-1].details["backup"]
    assert backup_path.is_file()
    if sys.platform != "win32":
        assert backup_path.stat().st_mode & 0o777 == 0o600
    with store.connection(directory / store.DB_NAME) as conn:
        assert update.update_health(conn).level == "ok"


def test_explicit_tag_must_be_merged(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, runner = setup
    with pytest.raises(admin.AdminError, match="origin/master"):
        execute(directory, runner, clock, to="v99")
    assert not any("checkout" in c for c in runner.calls)


@pytest.mark.parametrize("mode", ["dirty", "older schema"])
def test_refusals(setup: tuple[Path, FakeRunner], clock: FakeClock, mode: str) -> None:
    directory, runner = setup
    runner.dirty = mode == "dirty"
    if mode == "older schema":
        runner.schema = len(store.MIGRATIONS) - 1
    with pytest.raises(admin.AdminError):
        execute(directory, runner, clock)
    assert not entries(directory)
    assert not any(c[0] == "systemctl" for c in runner.calls)


@pytest.mark.parametrize("probe", ["bump", "failure", "cleanup failure"])
def test_schema_waits(setup: tuple[Path, FakeRunner], clock: FakeClock, probe: str) -> None:
    directory, runner = setup
    runner.schema = None if probe == "failure" else len(store.MIGRATIONS) + 1
    if probe == "cleanup failure":
        runner.fail = "remove"
    message, ok = execute(directory, runner, clock)
    assert ok and "--allow-schema" in message
    assert entries(directory)[-1].action == "update_waiting"
    assert not any(c[0] == "systemctl" for c in runner.calls)
    assert any(c[:3] == ["git", "worktree", "remove"] for c in runner.calls)
    with store.connection(directory / store.DB_NAME) as conn:
        check = update.update_health(conn)
        assert check.level == "warn" and "v2" in check.message


def test_allow_schema(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, runner = setup
    runner.schema = len(store.MIGRATIONS) + 1
    runner.bump = True
    assert execute(directory, runner, clock, allow_schema=True)[1]
    assert update._schema(directory / store.DB_NAME) == len(store.MIGRATIONS) + 1
    assert entries(directory)[-1].action == "update_applied"


@pytest.mark.parametrize("schema", [False, True])
def test_health_failure_rolls_back(
    setup: tuple[Path, FakeRunner], clock: FakeClock, schema: bool
) -> None:
    directory, runner = setup
    runner.health_fails = True
    runner.bump = schema
    runner.schema = len(store.MIGRATIONS) + int(schema)
    message, ok = execute(directory, runner, clock, allow_schema=schema)
    assert not ok and "health" in message
    assert ["git", "checkout", "--detach", "old"] in runner.calls
    assert runner.calls[-1] == ["systemctl", "--user", "start", *update.TIMERS]
    failed = entries(directory)[-1]
    assert failed.action == "update_failed" and failed.details["rolled_back"]
    assert failed.details["step"] == "health"
    assert len([c for c in runner.calls if Path(c[0]).name == "desman-registry"]) == 30
    assert update._schema(directory / store.DB_NAME) == len(store.MIGRATIONS)
    with store.connection(directory / store.DB_NAME) as conn:
        assert update.update_health(conn).level == "warn"
        assert not conn.execute(
            "SELECT name FROM sqlite_master WHERE name='migration_marker'"
        ).fetchall()
    if schema:
        assert ["systemctl", "--user", "stop", "desman-registry.service"] in runner.calls


@pytest.mark.parametrize("step", ["stop", "checkout", "install", "restart", "start"])
def test_apply_failures_restart_timers(
    setup: tuple[Path, FakeRunner], clock: FakeClock, step: str
) -> None:
    directory, runner = setup
    runner.fail = step
    assert not execute(directory, runner, clock)[1]
    assert runner.calls[-1] == ["systemctl", "--user", "start", *update.TIMERS]
    assert entries(directory)[-1].details["rolled_back"]


def test_check_does_not_apply_or_journal(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, runner = setup
    runner.schema = len(store.MIGRATIONS) + 1
    before = (directory / store.DB_NAME).read_bytes()
    assert execute(directory, runner, clock, check=True)[1]
    assert not entries(directory)
    assert (directory / store.DB_NAME).read_bytes() == before
    assert not (directory / "backups").exists()
    assert not any(c[0] == "systemctl" for c in runner.calls)


def test_concurrent_lock(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, runner = setup
    with (directory / "update.lock").open("a+b") as lock:
        filelock.lock(lock.fileno(), blocking=False)
        try:
            assert "Another" in execute(directory, runner, clock)[0]
            assert not runner.calls
        finally:
            filelock.unlock(lock.fileno())


def test_assign_printed_command_and_pem(ready: Path, tmp_path: Path) -> None:
    cli_runner = CliRunner()
    made = cli_runner.invoke(cli, ["--dir", str(ready), "appointer", "create"])
    assert made.exit_code == 0, made.output
    # Rich may wrap the command on narrow terminals; key itself is also printed alone.
    key = sign.encode_public(sign.load_node_key(ready / store.APPOINTER_KEY).public)
    printed = " ".join(made.output.split())
    command = f"desman-registry assign {key} --scope appoint"
    assert command in printed
    args = shlex.split(command)[1:]
    result = cli_runner.invoke(cli, ["--dir", str(ready), *args])
    assert result.exit_code == 0, result.output
    pem = tmp_path / "operator.pem"
    pem.write_bytes(sign.public_pem(sign.NodeKey.generate().public))
    result = cli_runner.invoke(cli, ["--dir", str(ready), "assign", str(pem)])
    assert result.exit_code == 0, result.output
    bad = cli_runner.invoke(cli, ["--dir", str(ready), "assign", "garbage"])
    assert bad.exit_code == 1 and "PEM" in bad.output and "base64" in bad.output


def test_health_update_is_read_only(ready: Path, clock: FakeClock) -> None:
    with store.connection(ready / store.DB_NAME) as conn:
        store.log(conn, "update_waiting", details={"tag": "v9"})
    before = (ready / store.DB_NAME).read_bytes()
    checks = admin.health(ready, now=clock(), url="", root_pub=sign.ROOT_PUB)
    assert next(c for c in checks if c.name == "update").level == "warn"
    result = CliRunner().invoke(cli, ["--dir", str(ready), "health", "--url", ""])
    assert "v9" in result.output and "--allow-schema" in result.output
    assert (ready / store.DB_NAME).read_bytes() == before


def test_update_units() -> None:
    directory = Path(__file__).resolve().parents[1] / "docs/systemd"
    config = configparser.ConfigParser(interpolation=None)
    config.read(directory / "desman-registry-update.timer")
    assert config["Timer"]["OnCalendar"] == "*-*-* 03:30"
    assert config["Timer"].getboolean("Persistent")
    assert config["Timer"]["RandomizedDelaySec"] == "10m"
    config.read(directory / "desman-registry-update.service")
    assert config["Service"]["Type"] == "oneshot"
    assert "NoNewPrivileges" not in config["Service"]


def test_local_git_selects_only_merged_releases(tmp_path: Path, clock: FakeClock) -> None:
    origin = tmp_path / "origin.git"
    app = tmp_path / "checkout"
    directory = tmp_path / "data"
    store.open_registry(directory / store.DB_NAME).close()

    def git(*args: str, cwd: Path = tmp_path) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=30
        ).stdout.strip()

    git("init", "--bare", str(origin))
    git("clone", str(origin), str(app))
    git("config", "user.email", "test@example.invalid", cwd=app)
    git("config", "user.name", "Test", cwd=app)
    git("checkout", "-b", "master", cwd=app)
    (app / "file").write_text("one")
    git("add", ".", cwd=app)
    git("commit", "-m", "one", cwd=app)
    git("tag", "v0.2.0-test1", cwd=app)
    git("tag", "v0.2.0", cwd=app)
    old = git("rev-parse", "HEAD", cwd=app)
    (app / "file").write_text("two")
    git("commit", "-am", "two", cwd=app)
    git("tag", "v0.2.1-test1", cwd=app)
    prerelease = git("rev-parse", "HEAD", cwd=app)
    (app / "file").write_text("release")
    git("commit", "-am", "release", cwd=app)
    git("tag", "-a", "v0.2.1", "-m", "release", cwd=app)
    new = git("rev-parse", "HEAD", cwd=app)
    git("push", "origin", "master", "--tags", cwd=app)
    git("checkout", "-b", "side", cwd=app)
    (app / "file").write_text("side")
    git("commit", "-am", "side", cwd=app)
    git("tag", "v99", cwd=app)
    git("push", "origin", "side", "--tags", cwd=app)
    git("checkout", "--detach", old, cwd=app)
    calls: list[list[str]] = []

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "git":
            return update.run(args, cwd=cwd)
        if args[0] == sys.executable:
            return subprocess.CompletedProcess(args, 0, str(len(store.MIGRATIONS)), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    message, ok = update.update(directory, app=app, runner=runner, sleep=clock.sleep, now=clock())
    assert ok and "v0.2.1" in message
    assert git("rev-parse", "HEAD", cwd=app) == new
    assert not git("worktree", "list", "--porcelain", cwd=app).count("worktree ") > 1
    assert entries(directory)[-1].details["to"] == "v0.2.1"
    git("checkout", "--detach", prerelease, cwd=app)
    clock.advance(1)
    message, ok = update.update(directory, app=app, runner=runner, sleep=clock.sleep, now=clock())
    assert ok and "v0.2.1" in message
    assert git("rev-parse", "HEAD", cwd=app) == new


def test_backup_failure_still_restarts_timers(
    setup: tuple[Path, FakeRunner], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, runner = setup

    def broken(source: Path, dest: Path) -> None:
        raise OSError("backup failed")

    monkeypatch.setattr(admin, "backup", broken)
    assert not execute(directory, runner, clock)[1]
    assert not any(c[:2] == ["git", "checkout"] for c in runner.calls)
    assert entries(directory)[-1].details["step"] == "backup"
    assert runner.calls[-1] == ["systemctl", "--user", "start", *update.TIMERS]


def test_rollback_failure_is_journaled(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, fake = setup
    fake.health_fails = True

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = fake(args, cwd=cwd)
        if args == ["git", "checkout", "--detach", "old"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        return result

    assert not update.update(
        directory, app=fake.app, runner=runner, sleep=clock.sleep, now=clock()
    )[1]
    assert entries(directory)[-1].details["rolled_back"] is False
    assert fake.calls[-1] == ["systemctl", "--user", "start", *update.TIMERS]


def test_preupdate_backup_rotation(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, runner = setup
    folder = directory / "backups"
    folder.mkdir()
    for day in range(1, 17):
        (folder / f"registry-preupdate-202501{day:02}-000000Z.db").write_text("old")
    retained = [
        "registry-preupdate-manual.db",
        "registry-20250101-000000Z.db",
        "registry-preupdate-29990101-000000Z.db",
    ]
    for name in retained:
        (folder / name).write_text("keep")
    assert execute(directory, runner, clock)[1]
    assert all((folder / name).read_text() == "keep" for name in retained)
    assert not (folder / "registry-preupdate-20250101-000000Z.db").exists()
    assert len(list(folder.iterdir())) == 17


def test_restore_removes_wal_and_shm(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, _ = setup
    backup = update._backup(directory, clock())
    path = directory / store.DB_NAME
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).write_bytes(b"stale")
    update._restore(directory, backup)
    assert not Path(str(path) + "-wal").exists()
    assert not Path(str(path) + "-shm").exists()
    assert update._schema(path) == len(store.MIGRATIONS)
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600


def test_cli_passes_options_and_failure_exit_code(
    setup: tuple[Path, FakeRunner], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, runner = setup
    seen: dict[str, object] = {}

    def apply(
        directory: Path, *, app: Path | None, check: bool, allow_schema: bool, to: str | None
    ) -> tuple[str, bool]:
        seen.update(directory=directory, app=app, check=check, allow_schema=allow_schema, to=to)
        return "failed", False

    monkeypatch.setattr(update, "update", apply)
    result = CliRunner().invoke(
        cli,
        [
            "--dir",
            str(directory),
            "update",
            "--app",
            str(runner.app),
            "--check",
            "--allow-schema",
            "--to",
            "v2",
        ],
    )
    assert result.exit_code == 1
    assert seen == dict(directory=directory, app=runner.app, check=True, allow_schema=True, to="v2")


@pytest.mark.parametrize("uv", [None, "/test/bin/uv"])
def test_installer_preflight(
    setup: tuple[Path, FakeRunner], clock: FakeClock, uv: str | None
) -> None:
    directory, fake = setup
    fake.health_fails = True

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = fake(args, cwd=cwd)
        if args[-3:] == ["-m", "pip", "--version"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        return result

    if uv is None:
        with pytest.raises(admin.AdminError, match="No installer available"):
            update.update(directory, app=fake.app, runner=runner, which=lambda _: uv)
        assert not any(c[0] == "systemctl" for c in fake.calls)
        assert not entries(directory)
        assert not (directory / "backups").exists()
    else:
        assert not update.update(
            directory,
            app=fake.app,
            runner=runner,
            which=lambda _: uv,
            sleep=clock.sleep,
            now=clock(),
        )[1]
        command = [uv, "pip", "install", "--python", str(fake.app / ".venv/bin/python"), "-e", "."]
        assert fake.calls.count(command) == 2
        assert entries(directory)[-1].details["rolled_back"] is True


@pytest.mark.parametrize("schema", [False, True])
def test_failed_rollback_install_still_restarts_service(
    setup: tuple[Path, FakeRunner], clock: FakeClock, schema: bool
) -> None:
    directory, fake = setup
    fake.health_fails = True
    fake.bump = schema
    installs = 0

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal installs
        result = fake(args, cwd=cwd)
        if "install" in args:
            installs += 1
            if installs == 2:
                return subprocess.CompletedProcess(args, 1, "", "")
        return result

    assert not update.update(
        directory, app=fake.app, runner=runner, sleep=clock.sleep, now=clock(), allow_schema=schema
    )[1]
    rollback = fake.calls.index(["git", "checkout", "--detach", "old"])
    assert fake.calls[rollback + 2 :] == (
        [["systemctl", "--user", "stop", "desman-registry.service"]] if schema else []
    ) + [
        ["systemctl", "--user", "restart", "desman-registry.service"],
        ["systemctl", "--user", "start", *update.TIMERS],
    ]
    assert entries(directory)[-1].details["rolled_back"] is False

    assert update._schema(directory / store.DB_NAME) == len(store.MIGRATIONS)


@pytest.mark.parametrize("action", ["update_waiting", "update_failed"])
@pytest.mark.parametrize("ancestor", [False, True])
def test_current_clears_warning(
    setup: tuple[Path, FakeRunner], clock: FakeClock, action: str, ancestor: bool
) -> None:
    directory, runner = setup
    with store.connection(directory / store.DB_NAME) as conn:
        store.log(conn, action, details={"tag": "v2"})
    runner.head = "new" if ancestor else "old"
    runner.ancestor = ancestor
    execute(directory, runner, clock, check=True)
    assert entries(directory)[-1].action == action
    execute(directory, runner, clock)
    execute(directory, runner, clock)
    assert [e.action for e in entries(directory)] == [action, "update_current"]
    with store.connection(directory / store.DB_NAME) as conn:
        assert update.update_health(conn).message == "current"


@pytest.mark.parametrize("target", [None, len(store.MIGRATIONS) + 1])
def test_waiting_deduplicated(
    setup: tuple[Path, FakeRunner], clock: FakeClock, target: int | None
) -> None:
    directory, runner = setup
    runner.schema = target
    for _ in range(2):
        message, ok = execute(directory, runner, clock)
        assert ok
        assert ("could not be checked" in message) == (target is None)
    assert len(entries(directory)) == 1
    assert entries(directory)[0].details["to_schema"] == target
    assert sum(c[:3] == ["git", "worktree", "add"] for c in runner.calls) == 2
    runner.schema = len(store.MIGRATIONS) + 2
    execute(directory, runner, clock)
    assert len(entries(directory)) == 2


def test_failed_add_and_cleanup_waits(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, fake = setup

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        result = fake(args, cwd=cwd)
        if args[:2] == ["git", "worktree"]:
            return subprocess.CompletedProcess(args, 1, "", "")
        return result

    message, ok = update.update(directory, app=fake.app, runner=runner, sleep=clock.sleep)
    assert ok and "could not be checked" in message
    assert entries(directory)[-1].details["to_schema"] is None


@pytest.mark.parametrize(
    "target,expected",
    [
        (None, "schema unknown"),
        (len(store.MIGRATIONS) + 1, "would wait"),
        (len(store.MIGRATIONS), "would apply"),
    ],
)
def test_check_explains_action(
    setup: tuple[Path, FakeRunner], clock: FakeClock, target: int | None, expected: str
) -> None:
    directory, runner = setup
    runner.schema = target
    message, ok = execute(directory, runner, clock, check=True)
    assert ok and expected in message
    if target != len(store.MIGRATIONS):
        assert "--allow-schema" in message
        assert "would apply" in execute(directory, runner, clock, check=True, allow_schema=True)[0]
    assert not entries(directory)


@pytest.mark.parametrize("same", [False, True])
def test_explicit_non_new_tag(setup: tuple[Path, FakeRunner], clock: FakeClock, same: bool) -> None:
    directory, runner = setup
    runner.head = "new" if same else "old"
    runner.ancestor = same
    message, ok = execute(directory, runner, clock, to="v2")
    assert ok == same
    assert ("already at" if same else "refused") in message
    assert not any(c[0] == "systemctl" for c in runner.calls)


@pytest.mark.parametrize("raw", [None, "garbage", "null", "[]", "42", "{}"])
@pytest.mark.parametrize("action", ["update_waiting", "update_failed"])
def test_health_bad_details(raw: str | None, action: str) -> None:
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE registry_log (seq INTEGER, action TEXT, details_json TEXT)")
        conn.execute("INSERT INTO registry_log VALUES (1, ?, ?)", (action, raw))
        check = update.update_health(conn)
        assert check.level == "warn" and "?" in check.message


def test_health_schema_failure_skips_update(
    setup: tuple[Path, FakeRunner], clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, _ = setup
    with sqlite3.connect(directory / store.DB_NAME) as conn:
        conn.execute("PRAGMA user_version=1")

    def forbidden(conn: sqlite3.Connection) -> admin.HealthCheck:
        pytest.fail("update health must not run on an incompatible database")

    monkeypatch.setattr(update, "update_health", forbidden)
    found = admin.health(directory, now=clock(), url="", root_pub=sign.ROOT_PUB)
    assert next(c for c in found if c.name == "database").level == "fail"


def test_health_retries_for_a_minute(setup: tuple[Path, FakeRunner], clock: FakeClock) -> None:
    directory, fake = setup
    tries = 0
    sleeps: list[float] = []

    def runner(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        nonlocal tries
        result = fake(args, cwd=cwd)
        if Path(args[0]).name == "desman-registry":
            tries += 1
            return subprocess.CompletedProcess(args, int(tries < 30), "", "")
        return result

    assert update.update(directory, app=fake.app, runner=runner, sleep=sleeps.append)[1]
    assert tries == 30 and sleeps == [2] * 29


def test_health_json_is_untranslated(ready: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from desman.registry import cli as cli_module

    with store.connection(ready / store.DB_NAME) as conn:
        store.log(conn, "update_waiting", details={"tag": "v9", "to_schema": None})
    monkeypatch.setattr(cli_module, "t", lambda text, **kwargs: "TRANSLATED")
    monkeypatch.setattr(update, "t", lambda text, **kwargs: "TRANSLATED")
    result = CliRunner().invoke(cli, ["--dir", str(ready), "health", "--url", "", "--json"])
    checks = json.loads(result.output)["checks"]
    assert (
        next(c for c in checks if c["name"] == "update")["message"] == "waiting unknown schema v9"
    )
