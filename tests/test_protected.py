"""Tests for protected files (ROADMAP 5.3): a change nobody approved blocks every run."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner, Result

from desman import runs
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db, operator, protected
from desman.core.protected import Blocked, Change, Kind, Layout, ProtectedError
from desman.interlocutor import CHAT
from desman.llm.fake import FakeLLM
from helpers import make_config

ROOT = Path(__file__).parent.parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


# ── The shipped list ─────────────────────────────────────────────────────────


def test_the_shipped_hashes_match_the_files() -> None:
    """A commit that changes a protected file updates its hash in protected.toml too: the
    change is then visible in review, and a fresh database takes the release as approved."""
    layout = protected.here()
    assert layout.checkout == ROOT  # the tests run from a source checkout
    manifest = protected.load_manifest(layout)
    stale = [
        f'"{path.removeprefix(protected.PREFIX)}" = "{now}"'
        for path, shipped in manifest.expected(layout).items()
        if (now := protected.sha256(protected.content(layout.file(path)))) != shipped
    ]
    assert stale == [], "update src/desman/protected.toml:\n" + "\n".join(stale)


def test_the_list_holds_what_the_roadmap_names() -> None:
    manifest = protected.load_manifest(protected.here())
    for path in ("policy.toml", "core/policy.py", "agents.py", "core/protected.py"):
        assert path in manifest.package
    assert "tests/conftest.py" in manifest.checkout


# ── A package of our own ─────────────────────────────────────────────────────


class Package:
    """A package and a checkout in tmp_path, with a manifest that matches them."""

    def __init__(self, root: Path, *, checkout: bool = True) -> None:
        self.package = root / "src" / "desman"
        self.root = root
        self.package.mkdir(parents=True)
        (root / "tests").mkdir()
        self.files = {"policy.toml": b"[roles]\n", "agents.py": b"REVIEW = 1\n"}
        self.tests = {"tests/conftest.py": b"# harness\n"}
        for name, data in self.files.items():
            (self.package / name).write_bytes(data)
        for name, data in self.tests.items():
            (root / name).write_bytes(data)
        self.write_manifest()
        self.layout = Layout(self.package, root if checkout else None)

    def write_manifest(self, extra: str = "") -> None:
        lines = ["[package]"]
        lines += [f'"{n}" = "{sha(d)}"' for n, d in self.files.items()]
        lines += ["[checkout]"]
        lines += [f'"{n}" = "{sha(d)}"' for n, d in self.tests.items()]
        (self.package / "protected.toml").write_text("\n".join(lines) + "\n" + extra)

    def edit(self, name: str, data: bytes) -> None:
        (self.package / name).write_bytes(data)


@pytest.fixture
def pkg(tmp_path: Path) -> Package:
    return Package(tmp_path / "install")


@pytest.fixture
def conn() -> sqlite3.Connection:
    return db.open_db(":memory:")


def approvals(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    rows = conn.execute("SELECT source, note FROM protected_approvals ORDER BY id").fetchall()
    return [(r["source"], r["note"]) for r in rows]


def test_a_fresh_database_takes_the_release_as_approved(
    pkg: Package, conn: sqlite3.Connection
) -> None:
    assert protected.changes(conn, pkg.layout) == []
    assert approvals(conn) == [("release", None)]
    paths = {r["path"] for r in conn.execute("SELECT path FROM protected_files")}
    assert paths == {
        "desman/protected.toml",
        "desman/policy.toml",
        "desman/agents.py",
        "tests/conftest.py",
    }
    protected.changes(conn, pkg.layout)
    assert len(approvals(conn)) == 1  # recorded once


def test_a_release_that_does_not_match_its_hashes_is_not_taken(
    pkg: Package, conn: sqlite3.Connection
) -> None:
    pkg.edit("agents.py", b"REVIEW = 0\n")
    [change] = protected.changes(conn, pkg.layout)
    assert (change.path, change.kind) == ("desman/agents.py", Kind.CHANGED)
    assert change.approved == sha(b"REVIEW = 1\n")
    assert approvals(conn) == []
    assert protected.diff(conn, change, pkg.layout) is None  # no approved content to show
    with pytest.raises(Blocked, match="desman/agents.py"):
        protected.require(conn, pkg.layout)


def test_a_change_blocks_until_the_person_approves_it(
    pkg: Package, conn: sqlite3.Connection
) -> None:
    protected.require(conn, pkg.layout)
    pkg.edit("policy.toml", b"[roles]\nexecutor = ['everything']\n")
    with pytest.raises(Blocked) as blocked:
        protected.require(conn, pkg.layout)
    [change] = blocked.value.changes
    assert change.kind is Kind.CHANGED
    shown = protected.diff(conn, change, pkg.layout)
    assert shown is not None and "+executor = ['everything']" in shown
    assert "desman/policy.toml (approved)" in shown

    done = protected.approve(conn, "reviewed it", layout=pkg.layout)
    assert done == [change]
    protected.require(conn, pkg.layout)
    assert approvals(conn) == [("release", None), ("operator", "reviewed it")]
    [entry] = operator.entries(conn)
    assert entry.action == "operator_protected_approved"
    assert entry.details["note"] == "reviewed it"
    assert entry.details["changes"][0]["path"] == "desman/policy.toml"
    assert protected.approve(conn, layout=pkg.layout) == []  # nothing more to approve
    assert len(operator.entries(conn)) == 1


def test_a_missing_file_and_a_shortened_list_block_too(
    pkg: Package, conn: sqlite3.Connection
) -> None:
    protected.require(conn, pkg.layout)
    (pkg.package / "agents.py").unlink()
    assert [(c.path, c.kind) for c in protected.changes(conn, pkg.layout)] == [
        ("desman/agents.py", Kind.MISSING)
    ]
    pkg.edit("agents.py", b"REVIEW = 1\n")
    del pkg.files["agents.py"]  # taken off the list: the manifest changes as well
    pkg.write_manifest()
    assert {(c.path, c.kind) for c in protected.changes(conn, pkg.layout)} == {
        ("desman/agents.py", Kind.REMOVED),
        ("desman/protected.toml", Kind.CHANGED),
    }


def test_a_file_put_on_the_list_needs_approval(pkg: Package, conn: sqlite3.Connection) -> None:
    protected.require(conn, pkg.layout)
    pkg.files["sandbox.py"] = b"NETWORK = False\n"
    pkg.edit("sandbox.py", b"NETWORK = False\n")
    pkg.write_manifest()
    kinds = {c.path: c.kind for c in protected.changes(conn, pkg.layout)}
    assert kinds == {"desman/sandbox.py": Kind.ADDED, "desman/protected.toml": Kind.CHANGED}


def test_line_endings_are_not_a_change(pkg: Package, conn: sqlite3.Connection) -> None:
    protected.require(conn, pkg.layout)
    pkg.edit("agents.py", b"REVIEW = 1\r\n")
    assert protected.changes(conn, pkg.layout) == []


def test_what_was_shown_is_what_gets_approved(pkg: Package, conn: sqlite3.Connection) -> None:
    protected.require(conn, pkg.layout)
    pkg.edit("agents.py", b"REVIEW = 2\n")
    shown = protected.changes(conn, pkg.layout)
    pkg.edit("agents.py", b"REVIEW = 3\n")  # changed again after the person looked
    with pytest.raises(ProtectedError, match="changed since they were shown"):
        protected.approve(conn, expected=shown, layout=pkg.layout)
    assert approvals(conn) == [("release", None)]
    assert operator.entries(conn) == []


def test_an_installed_package_does_not_check_the_harness(
    tmp_path: Path, conn: sqlite3.Connection
) -> None:
    installed = Package(tmp_path / "site", checkout=False)
    (installed.root / "tests" / "conftest.py").unlink()
    assert protected.changes(conn, installed.layout) == []
    # A checkout's approval, harness included, holds for the package without it as well.
    with_tests = Layout(installed.package, installed.root)
    (installed.root / "tests" / "conftest.py").write_bytes(b"# harness\n")
    assert protected.changes(conn, with_tests) == [
        Change(
            path="tests/conftest.py",
            kind=Kind.ADDED,
            approved=None,
            current=sha(b"# harness\n"),
        )
    ]


@pytest.mark.parametrize(
    "manifest",
    [
        "",
        "[package]\n",
        '[package]\n"../outside.py" = "' + "0" * 64 + '"\n',
        '[package]\n"protected.toml" = "' + "0" * 64 + '"\n',
        '[package]\n"agents.py" = "not a hash"\n',
        '[package]\n"agents.py" = "' + "0" * 64 + '"\n[elsewhere]\n',
        "not toml at all [",
    ],
)
def test_a_manifest_that_cannot_be_used_blocks_everything(
    pkg: Package, conn: sqlite3.Connection, manifest: str
) -> None:
    (pkg.package / "protected.toml").write_text(manifest)
    with pytest.raises(ProtectedError):
        protected.require(conn, pkg.layout)
    with pytest.raises(ProtectedError):
        protected.approve(conn, layout=pkg.layout)
    assert approvals(conn) == []


def test_a_missing_manifest_blocks_everything(pkg: Package, conn: sqlite3.Connection) -> None:
    (pkg.package / "protected.toml").unlink()
    with pytest.raises(ProtectedError, match="missing"):
        protected.require(conn, pkg.layout)


def test_approvals_are_append_only(pkg: Package, conn: sqlite3.Connection) -> None:
    protected.require(conn, pkg.layout)
    for statement in (
        "UPDATE protected_approvals SET note = 'x'",
        "DELETE FROM protected_approvals",
        "UPDATE protected_files SET sha256 = NULL, content = NULL",
        "DELETE FROM protected_files",
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(statement)


# ── The commands ─────────────────────────────────────────────────────────────


@pytest.fixture
def here(pkg: Package, monkeypatch: pytest.MonkeyPatch) -> Package:
    """The installation is `pkg`: what the commands check."""
    monkeypatch.setattr(protected, "here", lambda: pkg.layout)
    return pkg


def desman(*args: str | Path, input: str | None = None) -> Result:
    return CliRunner().invoke(cli, [str(a) for a in args], input=input)


def fake_script(tmp_path: Path) -> Path:
    path = tmp_path / "script.json"
    routes = [
        {"match": "Execute", "steps": ['{"x": 1}']},
        {"match": "Review", "steps": ['{"verdict": "accept"}']},
    ]
    path.write_text(json.dumps({"routes": routes}))
    return path


def tasks() -> int:
    conn = db.open_db()
    try:
        return int(conn.execute("SELECT count(*) FROM tasks").fetchone()[0])
    finally:
        conn.close()


def test_run_resume_and_worker_refuse_until_approved(here: Package, tmp_path: Path) -> None:
    fake = fake_script(tmp_path)
    assert desman("protected").exit_code == 0  # the release is approved
    here.edit("agents.py", b"REVIEW = 'always accept'\n")
    for args in (
        ("run", "g", "--fake-llm", fake),
        ("worker", "--until-idle", "--fake-llm", fake),
        ("resume", "abc", "--fake-llm", fake),
    ):
        result = desman(*args)
        assert result.exit_code == 1, result.output
        assert "Protected files changed without your approval: desman/agents.py" in result.output
        assert "desman protected show --diff" in result.output
    assert tasks() == 0  # refused before a task was made

    shown = desman("protected", "show", "--diff")
    assert shown.exit_code == 0, shown.output
    assert "1 protected file changed without your approval" in shown.output
    assert "desman/agents.py: changed" in shown.output
    assert "+REVIEW = 'always accept'" in shown.output

    approved = desman("protected", "approve", "--note", "I wrote it")
    assert approved.exit_code == 0, approved.output
    assert "Approved 1 change: tasks run again." in approved.output
    assert desman("run", "g", "--fake-llm", fake).exit_code == 0
    assert tasks() == 1


def test_protected_json(here: Package) -> None:
    out = json.loads(desman("protected", "show", "--json").output)
    assert out["approved"]["source"] == "release" and out["changes"] == []
    here.edit("policy.toml", b"")
    out = json.loads(desman("protected", "show", "--json").output)
    [change] = out["changes"]
    assert change["path"] == "desman/policy.toml" and change["kind"] == "changed"
    out = json.loads(desman("protected", "approve", "--json").output)
    assert [c["path"] for c in out["approved"]] == ["desman/policy.toml"]
    assert json.loads(desman("protected", "approve", "--json").output) == {"approved": []}


def test_a_broken_manifest_is_said_plainly(here: Package, tmp_path: Path) -> None:
    (here.package / "protected.toml").write_text("[")
    for args in (
        ("protected",),
        ("protected", "approve"),
        ("run", "g", "--fake-llm", fake_script(tmp_path)),
    ):
        result = desman(*args)
        assert result.exit_code == 1
        assert "Cannot check the protected files" in result.output


def test_a_coordinator_cannot_be_put_together_while_blocked(here: Package) -> None:
    from desman.core.store import Store

    store = Store(db.open_db(":memory:"))
    here.edit("agents.py", b"")
    with pytest.raises(Exception, match="Protected files changed"):
        runs.summarizer(store, runs.Models(FakeLLM(), None))


def test_a_running_worker_stops_at_a_later_change(here: Package) -> None:
    """A change made while a worker runs stops it before its next step, holding nothing: the
    rest of the work waits for the approval."""
    import functools

    from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
    from desman.core.models import SubtaskSpec, TaskSpec
    from desman.core.states import SubtaskStatus
    from desman.core.store import Store

    store = Store(db.open_db(":memory:"))
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="first", acceptance_criteria="ok"),
                SubtaskSpec(description="second", acceptance_criteria="ok"),
            ],
        )
    )

    def execute(work: Work) -> Success:
        here.edit("agents.py", b"REVIEW = 'changed while running'\n")
        return Success(content={"done": work.subtask.description})

    coordinator = Coordinator(
        store,
        execute,
        lambda subtask, artifact: Review(verdict=Verdict.ACCEPT),
        gate=functools.partial(runs.guard, store),
    )
    with pytest.raises(Exception, match="Protected files changed"):
        coordinator.work(until_idle=True)
    statuses = sorted(s.status for s in store.list_subtasks(task_id))
    assert coordinator.executed == 1
    assert SubtaskStatus.READY in statuses  # the second was never claimed
    assert all(s.lease_owner is None for s in store.list_subtasks(task_id))


# ── The chat ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, here: Package) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def chat(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def asks(name: str, **args: Any) -> str:
    return json.dumps({"type": "command", "name": name, "args": args})


def test_the_chat_says_tasks_are_blocked_and_runs_none(fake: FakeLLM, here: Package) -> None:
    desman("protected")
    here.edit("agents.py", b"")
    fake.on(CHAT, json.dumps({"type": "task", "title": "Haiku", "goal": "Write a haiku"}))
    result = chat("write me a haiku")
    assert result.output.count("Protected files changed without your approval") == 2
    assert "Ask here to show the change or approve it." in result.output  # told once
    assert "Planning…" not in result.output
    assert tasks() == 0


def test_the_chat_approves_only_after_the_window(fake: FakeLLM, here: Package) -> None:
    desman("protected")
    here.edit("agents.py", b"")
    fake.on(CHAT, asks("protected approve"), asks("protected approve"))
    declined = chat("approve it", "n")
    assert "desman/agents.py: changed" in declined.output
    assert "approve only a change you have looked at" in declined.output
    assert protected.changes(db.open_db()) != []

    accepted = chat("approve it", "y")
    assert "Approved 1 change: tasks run again." in accepted.output
    conn = db.open_db()
    assert protected.changes(conn) == []
    [approval] = [e for e in operator.entries(conn) if e.action == "operator_protected_approved"]
    assert approval.session_id is not None  # the session the person said yes in
    actions = [e.action for e in operator.entries(conn)]
    assert actions == [
        "operator_declined",
        "operator_protected_approved",
        "operator_command",
    ]
