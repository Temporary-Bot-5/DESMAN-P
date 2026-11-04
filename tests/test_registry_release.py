"""G.7c: operator confirmation, exact commits, root signatures and final decisions."""

from __future__ import annotations

import fcntl
import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import applying, improve
from desman.core import db
from desman.net import protocol, sign
from desman.registry import admin, checks, releases, store
from desman.registry.cli import cli
from desman.registry.server import Registry
from test_evaluation import checkout, git, origin, package
from test_proposal_checks import (
    ask,
    checker,
    fresh,
    give,
    record,
    signed_submission,
    submission,
    submit,
)
from test_registry import _write_key, db_path, registry, registry_dir, root, root_pub

__all__ = ["checkout", "origin", "fresh", "db_path", "registry", "registry_dir", "root", "root_pub"]


@pytest.fixture
def ready(registry: Registry, clock: FakeClock, fresh: None, checkout: Path) -> str:
    a, b = checker(registry, clock), checker(registry, clock)
    pid = submit(a, submission(a.key, checkout))
    lent = ask(b)
    assert lent is not None
    assert give(b, lent, record(b, lent, "checked")).status_code == 200
    return pid


@pytest.fixture
def key_path(root: sign.NodeKey, registry_dir: Path) -> Path:
    path = registry_dir / store.ROOT_KEY
    _write_key(path, root)
    return path


def run_accept(
    conn: sqlite3.Connection,
    pid: str,
    repo: Path,
    directory: Path,
    clock: FakeClock,
    confirmation: str | None = None,
) -> dict[str, Any]:
    confirmation = confirmation or releases.plan(conn, pid, repo)["confirmation"]
    return releases.accept(conn, pid, repo, confirmation, directory=directory, now=clock())


def test_preview_accept_and_second_release(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    root: sign.NodeKey,
    root_pub: Path,
    key_path: Path,
    registry: Registry,
    clock: FakeClock,
) -> None:
    with store.connection(db_path) as conn:
        before = git(checkout, "rev-parse", "HEAD").strip()
        old = improve.current(improve.BY_ID["planner.rules"], package(checkout))
        snapshot = list(conn.iterdump())
        preview = releases.plan(conn, ready, checkout)
        assert len(preview["confirmation"]) == 64
        assert list(conn.iterdump()) == snapshot
        assert git(checkout, "rev-parse", "HEAD").strip() == before
        done = run_accept(conn, ready, checkout, registry_dir, clock, preview["confirmation"])
        assert done["number"] == 1 and done["parent"] == before
        assert done["commit"] == git(checkout, "rev-parse", "HEAD").strip()
        assert git(checkout, "rev-parse", "HEAD^").strip() == before
        assert (
            improve.current(improve.BY_ID["planner.rules"], package(checkout))
            == old + " Prefer one subtask."
        )
        assert store.proposal(conn, ready).status == "accepted"  # type: ignore[union-attr]
        signed = sign.Signed.model_validate(done["signed"])
        sign.check_signature(signed, root.public)
        store.trust(conn, root.public, clock()).receive(done["signed"])
        protocol.Release.model_validate(signed.record.body)
        assert signed.record.expires_at == signed.record.issued_at + timedelta(days=365)
        message = git(checkout, "log", "-1", "--format=%B")
        assert f"Desman-Submission: {ready}" in message and "Desman-Release: 1" in message
    a, b = checker(registry, clock), checker(registry, clock)
    second = submit(a, submission(a.key, checkout))
    lent = ask(b)
    assert lent is not None
    assert give(b, lent, record(b, lent, "checked")).status_code == 200
    with store.connection(db_path) as conn:
        assert run_accept(conn, second, checkout, registry_dir, clock)["number"] == 2


@pytest.mark.parametrize(
    "case",
    [
        "hash",
        "stale",
        "submitted",
        "reviewing",
        "rejected",
        "accepted",
        "no_checked",
        "refused",
        "detached",
        "dirty",
        "base",
        "wrong_key",
        "lock",
        "diff",
    ],
)
def test_refusals_change_nothing(
    case: str,
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    root_pub: Path,
    key_path: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with store.connection(db_path) as conn:
        confirmation = releases.plan(conn, ready, checkout)["confirmation"]
        if case in ("submitted", "reviewing", "rejected", "accepted"):
            conn.execute("UPDATE proposals SET status = ? WHERE id = ?", (case, ready))
        elif case == "hash":
            confirmation = "0" * 64
        elif case == "stale":
            git(checkout, "commit", "--allow-empty", "-m", "moved")
        elif case == "detached":
            git(checkout, "checkout", "--detach")
        elif case in ("dirty", "base"):
            path = package(checkout) / "planner.py"
            path.write_text(path.read_text() + "\n# changed\n")
            if case == "base":
                git(checkout, "add", ".")
                git(checkout, "commit", "-m", "changed base")
        elif case == "wrong_key":
            _write_key(key_path, sign.NodeKey.generate())
        elif case in ("no_checked", "refused", "diff"):
            original = releases.find
            found = original(conn, ready)
            if case == "diff":
                item = found.submission.model_copy(update={"diff": found.submission.diff + "evil"})
                replacement = found.signed.model_copy(update={"submission": item})
                import dataclasses

                monkeypatch.setattr(
                    releases, "find", lambda *args: dataclasses.replace(found, signed=replacement)
                )
                confirmation = releases.plan(conn, ready, checkout)["confirmation"]
            else:
                original_results = checks.results

                def results(c: sqlite3.Connection, pid: str) -> list[dict[str, Any]]:
                    return [
                        r | {"outcome": "unable" if case == "no_checked" else "refused"}
                        for r in original_results(c, pid)
                    ]

                monkeypatch.setattr(checks, "results", results)
        lock = (registry_dir / "releases.lock").open("a")
        try:
            if case == "lock":
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = git(checkout, "rev-parse", "HEAD")
            snapshot = list(conn.iterdump())
            with pytest.raises(admin.AdminError):
                run_accept(conn, ready, checkout, registry_dir, clock, confirmation)
            assert list(conn.iterdump()) == snapshot
            assert git(checkout, "rev-parse", "HEAD") == before
            assert len(git(checkout, "worktree", "list").splitlines()) == 1
        finally:
            lock.close()


def test_rejection_final_and_late_requests(
    ready: str, checkout: Path, registry: Registry, db_path: Path, clock: FakeClock, fresh: None
) -> None:
    with store.connection(db_path) as conn:
        releases.reject(conn, ready, "Not useful")
        assert store.proposal(conn, ready).reason == "Not useful"  # type: ignore[union-attr]
        with pytest.raises(admin.AdminError):
            releases.reject(conn, ready, "again")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE proposals SET status = 'submitted' WHERE id = ?", (ready,))
        assert checks._after(conn, ready) == "rejected"
    a, b = checker(registry, clock), checker(registry, clock)
    pid = submit(a, submission(a.key, checkout))
    lent = ask(b)
    assert lent is not None
    with store.connection(db_path) as conn:
        with pytest.raises(admin.AdminError, match="checking"):
            releases.reject(conn, pid, "later")
        # Simulate a decided legacy row with an outstanding lease: clean refusal, no trigger abort.
        conn.execute("UPDATE proposals SET status = 'rejected' WHERE id = ?", (pid,))
    assert give(b, lent, record(b, lent, "checked")).status_code != 200
    clock.sleep(10000)
    with store.connection(db_path) as conn:
        with db.transaction(conn):
            checks.expire(conn, clock(), 3)
            checks.withdraw(conn, b.fingerprint, clock())
        assert store.proposal(conn, pid).status == "rejected"  # type: ignore[union-attr]
    c = checker(registry, clock)
    pid = submit(c, submission(c.key, checkout))
    with store.connection(db_path) as conn:
        releases.reject(conn, pid, "No")
        assert store.proposal(conn, pid).status == "rejected"  # type: ignore[union-attr]


def test_operator_cannot_sign_release(root: sign.NodeKey, clock: FakeClock) -> None:
    operator = sign.NodeKey.generate()
    trust = sign.Trust(root.public, clock=clock)
    assigned = sign.assign(
        root, operator.public, set(sign.Scope), issued_at=clock(), lifetime=timedelta(days=7)
    )
    trust.receive(assigned.model_dump(mode="json"))
    body = protocol.Release(
        number=1,
        commit="a" * 40,
        parent="b" * 40,
        branch="main",
        proposal="c" * 32,
        submission="d" * 64,
        checks=["e" * 32],
        summary="Release",
        files=["src/desman/planner.py"],
    )
    signed = sign.sign(
        operator,
        sign.RecordKind.RELEASE,
        body.model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    with pytest.raises(sign.Refused) as refused:
        trust.receive(signed.model_dump(mode="json"))
    assert refused.value.reason == sign.Reason.NOT_ALLOWED
    root_signed = sign.sign(
        root,
        sign.RecordKind.RELEASE,
        body.model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    trust.receive(root_signed.model_dump(mode="json"))


def test_migration_seven_preserves_submissions_and_checks(
    ready: str, db_path: Path, tmp_path: Path
) -> None:
    # Recreate the actual previous schema, including existing proposal/check data.
    conn = db.connect(tmp_path / "old.db")
    try:
        db.migrate(conn, store.MIGRATIONS[:6])
        with store.connection(db_path) as current:
            for table in ("nodes", "proposals", "proposal_leases", "proposal_checks"):
                for row in current.execute(f"SELECT * FROM {table}"):
                    conn.execute(
                        f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})", tuple(row)
                    )
        db.migrate(conn, store.MIGRATIONS)
        assert checks.results(conn, ready)[0]["outcome"] == "checked"
        conn.execute("UPDATE proposals SET status = 'accepted' WHERE id = ?", (ready,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE proposals SET status = 'reviewed' WHERE id = ?", (ready,))
        conn.execute(
            "INSERT INTO releases VALUES (1, ?, ?, ?, ?, 'main', '{}', 'now')",
            ("a" * 32, ready, "a" * 40, "b" * 40),
        )
        for sql in ("UPDATE releases SET branch = 'other'", "DELETE FROM releases"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    finally:
        conn.close()


def test_cli(
    ready: str,
    checkout: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("desman.registry.cli._now", clock)
    base = git(checkout, "rev-parse", "HEAD").strip()
    git(checkout, "commit", "--allow-empty", "-m", "new HEAD")
    head = git(checkout, "rev-parse", "HEAD").strip()
    runner = CliRunner()
    args = ["proposal", "accept", ready, "--repo", str(checkout)]
    preview = runner.invoke(cli, [*args, "--json"])
    assert preview.exit_code == 0, preview.output
    planned = json.loads(preview.output)
    assert planned["base_commit"] == base
    assert planned["head"] == head
    confirmation = planned["confirmation"]
    plain = runner.invoke(cli, args)
    assert plain.exit_code == 0 and confirmation in plain.output
    assert f"Base commit: {base}" in plain.output
    assert f"HEAD: {head}" in plain.output
    done = runner.invoke(cli, [*args, "--confirm", confirmation, "--json"])
    assert done.exit_code == 0, done.output
    assert json.loads(done.output)["number"] == 1
    for command in (["release", "list"], ["release", "show", "1"], ["proposal", "show", ready]):
        for suffix in ([], ["--json"]):
            result = runner.invoke(cli, command + suffix)
            assert result.exit_code == 0, result.output
            if suffix:
                json.loads(result.output)
    rejected = runner.invoke(cli, ["proposal", "reject", ready, "--reason", "No"])
    assert rejected.exit_code != 0


def test_db_failure_names_unrecorded_commit(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
) -> None:
    with store.connection(db_path) as conn:
        conn.execute(
            "CREATE TRIGGER fail_release BEFORE INSERT ON releases BEGIN SELECT RAISE(ABORT, 'fault'); END"
        )
        with pytest.raises(admin.AdminError, match="unrecorded") as error:
            run_accept(conn, ready, checkout, registry_dir, clock)
        assert git(checkout, "rev-parse", "HEAD").strip() in str(error.value)
        assert releases.listing(conn) == []
        assert store.proposal(conn, ready).status == "reviewed"  # type: ignore[union-attr]


def test_no_hooks_run(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
    tmp_path: Path,
) -> None:
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    touched = tmp_path / "ran-hook"
    for name in (
        "pre-commit",
        "prepare-commit-msg",
        "commit-msg",
        "post-commit",
        "post-checkout",
        "post-merge",
    ):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\ntouch '{touched}'\nexit 1\n")
        hook.chmod(0o700)
    git(checkout, "config", "core.hooksPath", str(hooks))
    with store.connection(db_path) as conn:
        run_accept(conn, ready, checkout, registry_dir, clock)
        with pytest.raises(admin.AdminError):
            releases.reject(conn, ready, "No")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE proposals SET status = 'reviewed' WHERE id = ?", (ready,))
    assert not touched.exists()


def test_checkout_moves_during_build(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = applying.commit_texts

    def moved(*args: Any, **kwargs: Any) -> str:
        commit = build(*args, **kwargs)
        git(checkout, "checkout", "-b", "other")
        return commit

    monkeypatch.setattr(applying, "commit_texts", moved)
    with store.connection(db_path) as conn:
        before = git(checkout, "rev-parse", "HEAD")
        with pytest.raises(admin.AdminError, match="branch or HEAD"):
            run_accept(conn, ready, checkout, registry_dir, clock)
        assert git(checkout, "rev-parse", "HEAD") == before
        assert releases.listing(conn) == []


def test_wrong_key_is_refused_before_git(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_key(key_path, sign.NodeKey.generate())

    def no_git(*args: Any, **kwargs: Any) -> str:
        pytest.fail("git was called before rejecting a wrong root key")

    monkeypatch.setattr(applying, "_git", no_git)
    with (
        store.connection(db_path) as conn,
        pytest.raises(admin.AdminError, match="not the root key"),
    ):
        run_accept(conn, ready, checkout, registry_dir, clock, "0" * 64)


def test_cli_reject_and_accept_human_output(
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    registry: Registry,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("desman.registry.cli._now", clock)
    runner = CliRunner()
    with store.connection(db_path) as conn:
        confirmation = releases.plan(conn, ready, checkout)["confirmation"]
    result = runner.invoke(
        cli, ["proposal", "accept", ready, "--repo", str(checkout), "--confirm", confirmation]
    )
    assert result.exit_code == 0, result.output
    assert "push origin main" in result.output
    a = checker(registry, clock)
    pid = submit(a, submission(a.key, checkout))
    result = runner.invoke(cli, ["proposal", "reject", pid, "--reason", "Not wanted"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["proposal", "show", pid, "--json"])
    assert json.loads(result.output)["reason"] == "Not wanted"
    for reason in ("", "x" * 1001):
        with store.connection(db_path) as conn, pytest.raises(admin.AdminError):
            releases.reject(conn, pid, reason)


@pytest.mark.parametrize(
    "field,value",
    [
        ("number", 0),
        ("commit", "a" * 39),
        ("parent", "x" * 40),
        ("branch", "bad branch"),
        ("branch", "bad\x00branch"),
        ("branch", "x" * 201),
        ("proposal", "a" * 31),
        ("submission", "bad"),
        ("checks", []),
        ("checks", ["a" * 32] * 17),
        ("summary", "x" * 2001),
        ("files", []),
    ],
)
def test_release_contract(field: str, value: Any) -> None:
    from pydantic import ValidationError

    data = dict(
        number=1,
        commit="a" * 40,
        parent="b" * 64,
        branch="main",
        proposal="c" * 32,
        submission="d" * 64,
        checks=["e" * 32],
        summary="summary",
        files=["src/desman/planner.py"],
    )
    with pytest.raises(ValidationError):
        protocol.Release.model_validate(data | {field: value})


def test_registry_rechecks_a_checked_malicious_diff(
    registry: Registry,
    fresh: None,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    root_pub: Path,
    key_path: Path,
    clock: FakeClock,
) -> None:
    a, b = checker(registry, clock), checker(registry, clock)
    item = submission(a.key, checkout).submission
    item = item.model_copy(
        update={"diff": item.diff + "\n--- a/other.py\n+++ b/other.py\n@@ -1 +1 @@\n-old\n+evil\n"}
    )
    pid = submit(a, signed_submission(a.key, item))
    lent = ask(b)
    assert lent is not None
    assert give(b, lent, record(b, lent, "checked")).status_code == 200
    with store.connection(db_path) as conn:
        before = git(checkout, "rev-parse", "HEAD")
        snapshot = list(conn.iterdump())
        with pytest.raises(admin.AdminError, match="diff is not"):
            run_accept(conn, pid, checkout, registry_dir, clock)
        assert list(conn.iterdump()) == snapshot
        assert git(checkout, "rev-parse", "HEAD") == before


@pytest.mark.parametrize("reject_during_build", [False, True])
def test_registry_writes_during_release_build(
    reject_during_build: bool,
    ready: str,
    checkout: Path,
    db_path: Path,
    registry_dir: Path,
    key_path: Path,
    root_pub: Path,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = applying.commit_texts

    def concurrent_write(*args: Any, **kwargs: Any) -> str:
        with store.connection(db_path) as other:
            other.execute("PRAGMA busy_timeout = 50")
            if reject_during_build:
                releases.reject(other, ready, "Rejected during build")
            else:
                with db.transaction(other):
                    store.log(other, "concurrent_write", None, {})
        return build(*args, **kwargs)

    monkeypatch.setattr(applying, "commit_texts", concurrent_write)
    with store.connection(db_path) as conn:
        if reject_during_build:
            with pytest.raises(admin.AdminError, match="unrecorded") as error:
                run_accept(conn, ready, checkout, registry_dir, clock)
            assert git(checkout, "rev-parse", "HEAD").strip() in str(error.value)
            assert releases.listing(conn) == []
            assert store.proposal(conn, ready).status == "rejected"  # type: ignore[union-attr]
        else:
            assert run_accept(conn, ready, checkout, registry_dir, clock)["number"] == 1
            assert conn.execute(
                "SELECT 1 FROM registry_log WHERE action = 'concurrent_write'"
            ).fetchone()
