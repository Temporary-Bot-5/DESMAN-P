"""Unattended registry maintenance, read-only diagnostics and public liveness."""

from __future__ import annotations

import configparser
import json
import shlex
import sqlite3
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.net import protocol, sign
from desman.registry import admin, store
from desman.registry.cli import cli
from desman.registry.server import Registry
from test_registry import _write_key


@pytest.fixture
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> Path:
    directory = tmp_path / "registry"
    path = directory / store.DB_NAME
    conn = store.open_registry(path)
    root = sign.NodeKey.generate()
    public = tmp_path / "root.pub"
    public.write_bytes(sign.public_pem(root.public))
    monkeypatch.setattr(sign, "ROOT_PUB", public)
    monkeypatch.setattr("desman.registry.cli._now", clock)
    _write_key(directory / store.ROOT_KEY, root)
    for name, scopes in [
        (store.OPERATOR_KEY, set(sign.DEFAULT_SCOPES)),
        ("appointer.key", {sign.Scope.APPOINT}),
    ]:
        key = sign.node_key(directory / name)
        admin.assign(conn, root, key.public, scopes, days=90, now=clock())
    conn.close()
    return directory


def levels(directory: Path, clock: FakeClock) -> dict[str, str]:
    result: dict[str, str] = {}
    rank = {"ok": 0, "warn": 1, "fail": 2}
    for c in admin.health(directory, now=clock(), url="", root_pub=sign.ROOT_PUB):
        if rank[c.level] >= rank[result.get(c.name, "ok")]:
            result[c.name] = c.level
    return result


def test_daily_rotation(ready: Path, clock: FakeClock) -> None:
    folder = ready / "backups"
    folder.mkdir(mode=0o700)
    manual = folder / "registry-2026-11-05-preupdate.db"
    manual.write_text("manual")
    old = folder / "registry-20250101-000000Z.db"
    old.write_text("old")
    link = folder / "registry-20240101-000000Z.db"
    link.symlink_to(manual)
    done = admin.daily(ready, days=7, keep=1, now=clock())
    assert done.trust is not None and done.error is None and done.pruned == 1
    assert not old.exists() and link.is_symlink() and manual.read_text() == "manual"
    assert done.backup.stat().st_mode & 0o777 == 0o600
    assert folder.stat().st_mode & 0o777 == 0o700
    with sqlite3.connect(done.backup) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert levels(ready, clock)["backup"] == "ok"


def test_daily_no_root_still_backs_up(ready: Path) -> None:
    (ready / store.ROOT_KEY).unlink()
    result = CliRunner().invoke(cli, ["--dir", str(ready), "daily"])
    assert result.exit_code == 1, result.output
    assert "Backup written" in result.output
    assert len(admin.backup_files(ready / "backups")) == 1


def test_health_fresh_and_http(ready: Path, clock: FakeClock) -> None:
    admin.daily(ready, days=7, keep=14, now=clock())
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    ) as client:
        found = admin.health(ready, root_pub=sign.ROOT_PUB, now=clock(), client=client)
    assert all(c.level == "ok" for c in found), found
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as client:
        assert (
            admin.health(ready, root_pub=sign.ROOT_PUB, now=clock(), client=client)[-1].level
            == "fail"
        )
    result = CliRunner().invoke(cli, ["--dir", str(ready), "health", "--json", "--url", ""])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True


def test_health_age_boundaries(ready: Path, clock: FakeClock) -> None:
    admin.daily(ready, days=7, keep=14, now=clock())
    clock.advance(26 * 3600 + 10)
    assert levels(ready, clock)["trust_list"] == "warn"
    assert levels(ready, clock)["backup"] == "warn"
    clock.advance(22 * 3600)
    assert levels(ready, clock)["trust_list"] == "fail"
    clock.advance(24 * 3600)
    assert levels(ready, clock)["backup"] == "fail"
    for p in admin.backup_files(ready / "backups"):
        p.unlink()
    assert levels(ready, clock)["backup"] == "fail"


@pytest.mark.parametrize("version", [0, len(store.MIGRATIONS) + 1])
def test_schema_read_only(ready: Path, clock: FakeClock, version: int) -> None:
    path = ready / store.DB_NAME
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version={version}")
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    found = admin.health(ready, root_pub=sign.ROOT_PUB, now=clock(), url="")
    assert found[0].level == "fail"
    assert ("upgrade" if version else "restart") in found[0].message
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified


def test_missing_database_not_created(tmp_path: Path, clock: FakeClock) -> None:
    directory = tmp_path / "missing"
    assert levels(directory, clock)["database"] == "fail"
    assert not directory.exists()


def test_expiring_assignment_and_missing_appointer(ready: Path, clock: FakeClock) -> None:
    clock.advance(77 * 86400)
    assert levels(ready, clock)["assignments"] == "warn"
    clock.advance(14 * 86400)
    assert levels(ready, clock)["assignments"] == "fail"
    clock.advance(-91 * 86400)
    (ready / "appointer.key").unlink()
    assert levels(ready, clock)["assignments"] == "warn"


def test_health_inbox(ready: Path, clock: FakeClock) -> None:
    conn = store.open_registry(ready / store.DB_NAME)
    try:
        key = admin.load_operator_key(ready / store.OPERATOR_KEY)
        job = admin.add_job(
            conn,
            key,
            protocol.Job(description="Say hello", acceptance_criteria="A greeting"),
            days=7,
            now=clock(),
        )
        conn.execute("UPDATE jobs SET status='operator' WHERE id=?", (job.id,))
        conn.commit()
    finally:
        conn.close()
    assert levels(ready, clock)["inbox"] == "warn"


def test_healthz(ready: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = Registry(ready / store.DB_NAME, clock=clock)
    with store.connection(registry.path) as conn:
        before = conn.execute("SELECT count(*) FROM registry_log").fetchone()[0]
    for target in ("/healthz", "/healthz?anything"):
        reply = registry.handle("GET", target, [], b"")
        assert reply.status == 200 and reply.body == {"status": "ok"}
    assert registry.handle("POST", "/healthz", [], b"").status == 405
    assert registry.handle("GET", "/healthz", [], b"x").status == 400
    with store.connection(registry.path) as conn:
        assert conn.execute("SELECT count(*) FROM registry_log").fetchone()[0] == before
    assert registry.handle("GET", protocol.TRUST, [], b"").status == 400

    (ready / store.DB_NAME).write_bytes(b"broken")
    reply = registry.handle("GET", "/healthz", [], b"")
    assert reply.status == 503 and reply.body == {"status": "unavailable"}


def test_systemd_units() -> None:
    directory = Path(__file__).resolve().parents[1] / "docs" / "systemd"
    for path in directory.glob("*.service"):
        config = configparser.ConfigParser(strict=False, interpolation=None)
        config.read(path)
        command = shlex.split(config["Service"]["ExecStart"])
        assert command[0] == "%h/desman-registry-app/.venv/bin/desman-registry"
        result = CliRunner().invoke(cli, command[1:] + ["--help"])
        assert result.exit_code == 0, result.output
        assert config["Service"]["UMask"] == "0077"
    for name, calendar in [("daily", "*-*-* 04:00"), ("health", "hourly")]:
        config = configparser.ConfigParser(strict=False, interpolation=None)
        config.read(directory / f"desman-registry-{name}.timer")
        assert config["Timer"]["OnCalendar"] == calendar
        assert config["Timer"].getboolean("Persistent")


@pytest.mark.parametrize(
    "free, expected", [(99 * 1024**2, "fail"), (500 * 1024**2, "warn"), (2 * 1024**3, "ok")]
)
def test_disk_thresholds(
    ready: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, free: int, expected: str
) -> None:
    import shutil

    usage = shutil.disk_usage(ready)
    monkeypatch.setattr(
        "desman.registry.admin.shutil.disk_usage",
        lambda path: type(usage)(usage.total, usage.used, free),
    )
    assert levels(ready, clock)["disk"] == expected


def test_health_missing_trust_and_invalid_backup(ready: Path, clock: FakeClock) -> None:
    clock.advance(8 * 86400)
    assert levels(ready, clock)["trust_list"] == "fail"
    folder = ready / "backups"
    folder.mkdir()
    (folder / "registry-99999999-999999Z.db").write_text("invalid timestamp")
    assert levels(ready, clock)["backup"] == "fail"


def test_latest_expired_assignment(ready: Path, clock: FakeClock) -> None:
    clock.advance(10)
    root = admin.load_root_key(ready / store.ROOT_KEY)
    key = admin.load_operator_key(ready / store.OPERATOR_KEY)
    with store.connection(ready / store.DB_NAME) as conn:
        admin.assign(conn, root, key.public, set(sign.DEFAULT_SCOPES), days=1, now=clock())
    clock.advance(2 * 86400)
    assert levels(ready, clock)["assignments"] == "fail"


@pytest.mark.parametrize("signing", [False, True])
def test_daily_backup_sqlite_error(
    ready: Path, monkeypatch: pytest.MonkeyPatch, signing: bool
) -> None:
    def broken_backup(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("backup broken")

    if signing:
        (ready / store.ROOT_KEY).unlink()
    monkeypatch.setattr(admin, "backup", broken_backup)
    result = CliRunner().invoke(cli, ["--dir", str(ready), "daily"])
    assert result.exit_code == 1
    assert "backup broken" in result.output
    if signing:
        assert "key" in result.output.lower()


def test_daily_unexpected_signing_error(
    ready: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("signing broken")

    monkeypatch.setattr(admin, "sign_trust_list", broken)
    done = admin.daily(ready, days=7, keep=1, now=clock())
    assert done.error == "signing broken" and done.backup.is_file()


@pytest.mark.parametrize("version", [None, 0, len(store.MIGRATIONS) + 1])
def test_healthz_never_creates_or_migrates(
    ready: Path, clock: FakeClock, version: int | None
) -> None:
    registry = Registry(ready / store.DB_NAME, clock=clock)
    path = ready / "probe.db"
    registry.path = path
    if version is not None:
        with sqlite3.connect(path) as conn:
            conn.execute(f"PRAGMA user_version={version}")
    before = path.read_bytes() if path.exists() else None
    reply = registry.handle("GET", "/healthz", [], b"")
    assert reply.status == 503 and reply.body == {"status": "unavailable"}
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("check", ["backup", "disk", "server"])
def test_health_os_errors(
    ready: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise PermissionError("access denied")

    target = {"backup": "backup_files", "disk": "shutil.disk_usage", "server": "httpx.Client.get"}[
        check
    ]
    monkeypatch.setattr("desman.registry.admin." + target, broken)
    found = admin.health(ready, now=clock(), root_pub=sign.ROOT_PUB)
    assert any(
        c.name == check and c.level == "fail" and "access denied" in c.message for c in found
    )


def test_future_backups(ready: Path, clock: FakeClock) -> None:
    folder = ready / "backups"
    folder.mkdir()
    future = folder / "registry-20990101-000000Z.db"
    future.write_text("future")
    found = admin.health(ready, now=clock(), root_pub=sign.ROOT_PUB, url="")
    assert any(c.name == "backup" and c.level == "fail" for c in found)
    assert any(c.message == "backup dated in the future" and c.level == "warn" for c in found)
    first = admin.daily(ready, days=7, keep=1, now=clock())
    assert first.backup.exists() and future.exists() and first.pruned == 0
    clock.advance(86400)
    second = admin.daily(ready, days=7, keep=1, now=clock())
    assert second.backup.exists() and future.exists() and not first.backup.exists()
    assert levels(ready, clock)["backup"] == "ok"


def test_health_appointer_helper_and_root(
    ready: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_pub = sign.ROOT_PUB
    monkeypatch.setattr(store, "APPOINTER_KEY", "custom.key")
    (ready / "appointer.key").rename(ready / store.APPOINTER_KEY)
    original = admin.load_appointer
    seen: list[Path] = []

    def load(path: Path) -> sign.NodeKey | None:
        seen.append(path)
        return original(path)

    monkeypatch.setattr(admin, "load_appointer", load)
    monkeypatch.setattr(sign, "ROOT_PUB", ready / "nonexistent.pub")
    found = admin.health(ready, now=clock(), root_pub=root_pub, url="")
    assert seen == [ready / store.APPOINTER_KEY]
    assert all(c.level == "ok" for c in found if c.name == "assignments")


def test_assignment_messages(ready: Path, clock: FakeClock) -> None:
    def messages() -> list[str]:
        return [
            c.message
            for c in admin.health(ready, now=clock(), root_pub=sign.ROOT_PUB, url="")
            if c.name == "assignments"
        ]

    clock.advance(77 * 86400)
    assert "operator expires in 13 days" in messages()
    clock.advance(14 * 86400)
    assert "operator unassigned or expired" in messages()
    assert "appointer unassigned or expired" in messages()
    (ready / store.OPERATOR_KEY).unlink()
    (ready / store.APPOINTER_KEY).unlink()
    assert "operator missing" in messages()
    assert "appointer missing: clusters will not form" in messages()
    result = CliRunner().invoke(cli, ["--dir", str(ready), "health", "--url", ""])
    assert "Operator key missing" in result.output
    assert "clusters will not form" in result.output


def test_update_instructions() -> None:
    text = (Path(__file__).resolve().parents[1] / "docs/registry-deploy.md").read_text()
    section = text.split("## Updating the server", 1)[1].split("## Accepting", 1)[0]
    assert section.index("stop desman-registry-daily.timer") < section.index(
        "backup ~/registry-preupdate-$(date +%F-%H%M)"
    )
    assert section.index("desman-registry health") < section.index(
        "start desman-registry-daily.timer"
    )


def test_assignment_json_stays_english(ready: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (ready / store.APPOINTER_KEY).unlink()
    monkeypatch.setenv("DESMAN_REGISTRY_LANGUAGE", "ru")
    args = ["--dir", str(ready), "health", "--url", ""]
    plain = CliRunner().invoke(cli, args)
    assert "кластеры" in plain.output
    raw = CliRunner().invoke(cli, args + ["--json"])
    assert any(
        c["message"] == "appointer missing: clusters will not form"
        for c in json.loads(raw.output)["checks"]
    )
