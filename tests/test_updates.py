"""G.8: local bare remotes, signed releases, manual installation and safe rollback."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from desman import applying, commands, evaluation, global_cli
from desman.cli import cli
from desman.core import db, netlog, operator, proposals
from desman.net import sign, updates
from test_evaluation import git


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    c = db.open_db(":memory:")
    yield c
    c.close()


@pytest.fixture
def release_repo(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.name", "Test")
    git(source, "config", "user.email", "test@example.com")
    (source / "src/desman").mkdir(parents=True)
    (source / "src/desman/protected.toml").write_text('[package]\n"agents.py" = "hash"\n')
    (source / "src/desman/agents.py").write_text("old\n")
    (source / "pyproject.toml").write_text("old\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "base")
    base = git(source, "rev-parse", "HEAD").strip()
    remote = tmp_path / "remote.git"
    git(source, "clone", "--bare", str(source), str(remote))
    node = tmp_path / "node"
    git(source, "clone", str(remote), str(node))
    git(source, "remote", "add", "origin", str(remote))
    (source / "src/desman/agents.py").write_text("new\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "intermediate")
    parent = git(source, "rev-parse", "HEAD").strip()
    (source / "pyproject.toml").write_text("new\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "release")
    commit = git(source, "rev-parse", "HEAD").strip()
    git(source, "push", "origin", "main")
    assert base != parent
    return node, source, parent, commit


def envelope(root: sign.NodeKey, parent: str, commit: str, /, **changes: object) -> sign.Signed:
    body = dict(
        number=1,
        commit=commit,
        parent=parent,
        branch="main",
        proposal="a" * 32,
        submission="b" * 64,
        checks=["c" * 32],
        summary="A release",
        files=["pyproject.toml"],
    )
    body.update(changes)
    return sign.sign(
        root,
        sign.RecordKind.RELEASE,
        body,
        issued_at=datetime.now(UTC),
        lifetime=timedelta(days=10),
    )


def receive(
    conn: sqlite3.Connection, tmp_path: Path, parent: str, commit: str, /, **changes: object
) -> tuple[Path, sign.NodeKey]:
    root = sign.NodeKey.generate()
    pub = tmp_path / "root.pub"
    pub.write_bytes(sign.public_pem(root.public))
    updates.keep(conn, sign.Trust(root.public), envelope(root, parent, commit, **changes))
    return pub, root


def test_apply_revert_reapply(
    conn: sqlite3.Connection, tmp_path: Path, release_repo: tuple[Path, Path, str, str]
) -> None:
    repo, source, parent, commit = release_repo
    pub, root = receive(conn, tmp_path, parent, commit)
    base = git(repo, "rev-parse", "HEAD").strip()
    plan = applying.plan_update(conn, repo=repo, root=pub)
    assert plan.commit_count == 2
    assert len(plan.commits) == 2
    assert plan.files == ["pyproject.toml", "src/desman/agents.py"]
    assert plan.protected == ["src/desman/agents.py"]
    assert plan.dependencies
    assert "A release" in "\n".join(global_cli.update_lines(plan))
    applying.carry_out_update(conn, plan)
    assert git(repo, "rev-parse", "HEAD").strip() == commit
    assert proposals.last_update(conn).details["schema"] == db.SCHEMA_VERSION  # type: ignore[union-attr]
    with pytest.raises(applying.ApplyError, match="already installed"):
        applying.plan_update(conn, repo=repo, root=pub)
    applying.carry_out_update(conn, applying.plan_update_revert(conn, repo=repo))
    assert git(repo, "rev-parse", "HEAD").strip() == base
    applying.carry_out_update(conn, applying.plan_update(conn, 1, repo=repo, root=pub))
    assert [u.action for u in proposals.updates(conn)] == ["applied", "reverted", "applied"]
    assert [e.action for e in netlog.entries(conn)] == [
        "release_received",
        "update_applied",
        "update_reverted",
        "update_applied",
    ]
    assert len(operator.entries(conn)) == 3
    signed = envelope(root, parent, commit, number=2)
    assert updates.keep(conn, sign.Trust(root.public), signed)
    with pytest.raises(sqlite3.IntegrityError, match="revert"):
        proposals.record_update(
            conn, 2, "applied", commit=commit, base=base, branch="main", details={}
        )


@pytest.mark.parametrize(
    "case",
    ["parent", "files", "root", "forged", "dirty", "diverged", "detached", "installed", "expired"],
)
def test_apply_refusals(
    case: str, conn: sqlite3.Connection, tmp_path: Path, release_repo: tuple[Path, Path, str, str]
) -> None:
    repo, source, parent, commit = release_repo
    extra: dict[str, object] = {}
    if case == "parent":
        extra["parent"] = "0" * 40
    if case == "files":
        extra["files"] = ["other.py"]
    pub, root = receive(conn, tmp_path, parent, commit, **extra)
    if case == "root":
        pub.write_bytes(sign.public_pem(sign.NodeKey.generate().public))
    if case in ("forged", "expired"):
        signed = envelope(root, parent, commit)
        if case == "forged":
            signed = signed.model_copy(update={"signature": "A" * 86 + "=="})
        else:
            signed = sign.sign(
                root,
                sign.RecordKind.RELEASE,
                signed.record.body,
                issued_at=datetime.now(UTC) - timedelta(days=5),
                lifetime=timedelta(days=1),
            )
        # Deliberately bypass append-only protection to model tampered on-disk storage.
        conn.execute("DROP TRIGGER releases_no_update")
        conn.execute("UPDATE releases SET signed_json = ?", (signed.model_dump_json(),))
    if case == "dirty":
        (repo / "src/desman/agents.py").write_text("mine")
    if case == "diverged":
        git(
            repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=t@x",
            "commit",
            "--allow-empty",
            "-m",
            "own",
        )
    if case == "detached":
        git(repo, "checkout", "--detach")
    if case == "installed":
        git(repo, "fetch")
        git(repo, "merge", "--ff-only", commit)
    before = git(repo, "rev-parse", "HEAD")
    with pytest.raises(applying.ApplyError):
        applying.plan_update(conn, repo=repo, root=pub)
    assert git(repo, "rev-parse", "HEAD") == before
    assert proposals.updates(conn) == []


@pytest.mark.parametrize(
    "case", ["head", "schema", "dirty", "plan_head", "plan_schema", "plan_record"]
)
def test_revert_refusals(
    case: str, conn: sqlite3.Connection, tmp_path: Path, release_repo: tuple[Path, Path, str, str]
) -> None:
    repo, _, parent, commit = release_repo
    pub, _ = receive(conn, tmp_path, parent, commit)
    applying.carry_out_update(conn, applying.plan_update(conn, repo=repo, root=pub))
    plan = applying.plan_update_revert(conn, repo=repo)
    if case in ("head", "plan_head"):
        git(
            repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=t@x",
            "commit",
            "--allow-empty",
            "-m",
            "own",
        )
    if case in ("schema", "plan_schema"):
        conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    if case == "dirty":
        (repo / "pyproject.toml").write_text("mine")
    if case == "plan_record":
        applying.carry_out_update(conn, plan)
    with pytest.raises(applying.ApplyError):
        if case.startswith("plan_"):
            applying.carry_out_update(conn, plan)
        else:
            applying.plan_update_revert(conn, repo=repo)


def test_keep_and_conflict(conn: sqlite3.Connection) -> None:
    root = sign.NodeKey.generate()
    trust = sign.Trust(root.public)
    signed = envelope(root, "a" * 40, "b" * 40)
    assert updates.keep(conn, trust, signed)
    assert not updates.keep(conn, trust, signed)
    assert not updates.keep(conn, sign.Trust(root.public), signed)
    with pytest.raises(sign.Refused, match="another release"):
        updates.keep(conn, trust, envelope(root, "a" * 40, "c" * 40))
    assert [e.action for e in netlog.entries(conn)] == ["release_received", "release_conflict"]
    for sql in ("UPDATE releases SET branch = 'evil'", "DELETE FROM releases"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql)


def test_cli_and_catalog(
    conn: sqlite3.Connection,
    tmp_path: Path,
    release_repo: tuple[Path, Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _, parent, commit = release_repo
    pub, _ = receive(conn, tmp_path, parent, commit)
    original = applying.plan_update
    monkeypatch.setattr(evaluation, "repository", lambda: repo)
    monkeypatch.setattr(
        applying, "plan_update", lambda c, number=None: original(c, number, repo=repo, root=pub)
    )
    from contextlib import contextmanager

    @contextmanager
    def journal(c: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
        yield conn

    monkeypatch.setattr(global_cli, "journal", journal)
    runner = CliRunner()
    for args in ([], ["--show"], ["--show", "1"], ["--show", "--json"]):
        result = runner.invoke(cli, ["global", "update", *args])
        assert result.exit_code == 0, result.output
    for args in (["--show", "--revert"], ["--apply", "1", "--show"], ["--show", "0"]):
        assert runner.invoke(cli, ["global", "update", *args]).exit_code == 2
    for args in (["--apply", "1", "--json"], ["--revert", "--json"]):
        result = runner.invoke(cli, ["global", "update", *args])
        assert result.exit_code == 0, result.output
    entry = commands.find("global update")
    assert entry is not None
    assert entry.kind_of(commands.UpdateArgs()) is commands.Kind.READ
    assert entry.kind_of(commands.UpdateArgs(action="apply", release=1)) is commands.Kind.CHANGE


@pytest.mark.parametrize("answer", ["n", "y"])
def test_chat_update_window(
    answer: str,
    conn: sqlite3.Connection,
    tmp_path: Path,
    release_repo: tuple[Path, Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from desman import runs
    from desman.config import open_config
    from desman.interlocutor import CHAT
    from desman.llm.fake import FakeLLM
    from helpers import make_config
    from test_commands import asks, chat

    repo, _, parent, commit = release_repo
    pub, _ = receive(conn, tmp_path, parent, commit)
    opened = db.open_db()
    conn.backup(opened)
    opened.close()
    llm = FakeLLM()
    llm.on(CHAT, asks("global update", action="apply", release=1))
    monkeypatch.setattr(runs, "models", lambda config, c, **kwargs: runs.Models(llm, None))
    open_config().save(make_config(language="en", configured=True))
    original = applying.plan_update
    monkeypatch.setattr(
        applying, "plan_update", lambda c, number=None: original(c, number, repo=repo, root=pub)
    )
    result = chat("install release 1", answer)
    assert "Release 1: A release" in result.output
    assert "Protected files" in result.output
    assert "Restart running workers" in result.output
    assert (git(repo, "rev-parse", "HEAD").strip() == commit) is (answer == "y")
    opened = db.open_db()
    try:
        assert len(proposals.updates(opened)) == (1 if answer == "y" else 0)
        actions = [e.action for e in operator.entries(opened)]
        assert ("operator_update_applied" in actions) is (answer == "y")
        assert ("operator_declined" in actions) is (answer == "n")
    finally:
        opened.close()


def test_apply_older_stored_release(
    conn: sqlite3.Connection,
    tmp_path: Path,
    release_repo: tuple[Path, Path, str, str],
) -> None:
    repo, _, parent, commit = release_repo
    pub, root = receive(conn, tmp_path, parent, commit)
    assert updates.keep(conn, sign.Trust(root.public), envelope(root, parent, "a" * 40, number=2))
    assert updates.latest(conn).record.body["number"] == 2  # type: ignore[union-attr]
    plan = applying.plan_update(conn, 1, repo=repo, root=pub)
    applying.carry_out_update(conn, plan)
    assert git(repo, "rev-parse", "HEAD").strip() == commit


def test_release_state_does_not_use_replay_journal() -> None:
    root = sign.NodeKey.generate()
    signed = envelope(root, "a" * 40, "b" * 40)

    class NoSeen:
        def add(self, id: str, expires_at: datetime, now: datetime) -> bool:
            pytest.fail("release is state, not a one-shot record")

    for _ in range(2):
        trust = sign.Trust(root.public, seen=NoSeen())
        trust.receive(signed.model_dump(mode="json"))
        trust.receive(signed.model_dump(mode="json"))
