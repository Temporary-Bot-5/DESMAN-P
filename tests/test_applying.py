"""Tests for applying and reverting a proposal (ROADMAP 5.6, D8): only by the person's command,
as a git commit of the checkout (built by code, never a patch) and the person's settings;
reverted the same way. The checkout is a copy of this one, a git repository of its own."""

# ruff: noqa: F811  (the fixtures of test_evaluation, imported, are the tests' arguments)

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from desman import applying, commands, evaluation, improve
from desman.config import Tuning, open_config
from desman.core import db, operator, proposals
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.proposals import Proposal
from desman.core.sessions import Role, Sessions
from desman.core.store import Store
from desman.improve import BY_ID, Change
from test_evaluation import (  # noqa: F401
    a_proposal,
    checkout,
    desman,
    edited_diff,
    git,
    origin,
    package,
    proposal,
    rules_change,
    store,
)

PLANNER = improve.SOURCE_PREFIX + "planner.py"


@pytest.fixture(autouse=True)
def no_git_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """DESMAN's own git calls see no identity of the person running the tests."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.delenv(f"GIT_{name}_NAME", raising=False)
        monkeypatch.delenv(f"GIT_{name}_EMAIL", raising=False)


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def apply(s: Store, p: Proposal, repo: Path) -> proposals.Application:
    plan = applying.plan_apply(s.conn, p, open_config().load().tuning, repo=repo)
    return applying.carry_out(s.conn, plan, open_config())


def revert(s: Store, p: Proposal, repo: Path) -> proposals.Application:
    plan = applying.plan_revert(s.conn, p, open_config().load().tuning, repo=repo)
    return applying.carry_out(s.conn, plan, open_config())


def body(diff: str) -> list[str]:
    """A diff's changed and context lines (git adds the enclosing statement to hunk headers)."""
    return [line for line in diff.splitlines() if line[:1] in " +-" and line[:3] not in "---+++"]


def planner_rules(repo: Path) -> str:
    return improve.current(BY_ID["planner.rules"], package(repo))


# ── Applying ─────────────────────────────────────────────────────────────────


def test_a_proposal_is_applied_as_a_commit_on_the_branch(checkout: Path) -> None:
    s = store()
    before, old = head(checkout), planner_rules(checkout)
    p = a_proposal(s, checkout)
    done = apply(s, p, checkout)
    assert done.action == "applied" and done.base == before and done.branch == "main"
    assert done.commit == head(checkout) and git(checkout, "rev-parse", "HEAD^").strip() == before
    assert done.details["files"] == [PLANNER]
    # The checkout's file holds the proposed text, and the commit is exactly the diff.
    assert planner_rules(checkout) == old + " Prefer one subtask."
    assert p.diff is not None
    assert body(git(checkout, "show", "--format=", done.commit)) == body(p.diff)
    message = git(checkout, "log", "-1", "--format=%an <%ae>%n%B")
    assert message.startswith("DESMAN <desman@localhost>\nDESMAN proposal 1: planner.rules")
    assert f"{applying.TRAILER}: {p.id}" in message
    assert git(checkout, "status", "--porcelain") == ""
    assert len(git(checkout, "worktree", "list").splitlines()) == 1
    [entry] = operator.entries(s.conn)
    assert entry.action == "operator_proposal_applied"
    assert entry.details == {"proposal": p.id, "commit": done.commit, "settings": []}


def test_the_checkouts_own_identity_commits(checkout: Path) -> None:
    git(checkout, "config", "user.name", "Person")
    git(checkout, "config", "user.email", "person@example.invalid")
    s = store()
    apply(s, a_proposal(s, checkout), checkout)
    assert git(checkout, "log", "-1", "--format=%an <%ae>").strip() == (
        "Person <person@example.invalid>"
    )


def test_a_proposal_is_applied_once(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    apply(s, p, checkout)
    with pytest.raises(applying.ApplyError, match="already applied"):
        applying.plan_apply(s.conn, p, Tuning(), repo=checkout)


def test_uncommitted_changes_in_its_files_stop_it(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    planner = package(checkout) / "planner.py"
    planner.write_text(planner.read_text() + "\n# mine\n")
    before = head(checkout)
    with pytest.raises(applying.ApplyError, match="uncommitted changes in src/desman/planner.py"):
        apply(s, p, checkout)
    assert head(checkout) == before and planner.read_text().endswith("# mine\n")
    assert proposals.applications(s.conn, p.id) == []


def test_changes_in_other_files_stay_as_they_are(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    other = checkout / "pyproject.toml"
    other.write_text(other.read_text() + "\n# mine\n")
    done = apply(s, p, checkout)
    assert done.commit is not None
    assert git(checkout, "show", "--name-only", "--format=", done.commit).split() == [PLANNER]
    assert git(checkout, "status", "--porcelain").strip() == "M pyproject.toml"


def test_a_head_that_moved_after_the_plan_stops_it(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    plan = applying.plan_apply(s.conn, p, Tuning(), repo=checkout)
    git(checkout, "commit", "-q", "--allow-empty", "-m", "meanwhile")
    with pytest.raises(applying.ApplyError, match="branch or HEAD changed"):
        applying.carry_out(s.conn, plan, open_config())
    assert proposals.applications(s.conn, p.id) == []


def test_a_proposal_made_from_other_files_is_not_applied(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    planner = package(checkout) / "planner.py"
    planner.write_text(planner.read_text() + "\n# changed since\n")
    git(checkout, "commit", "-q", "-am", "later")
    before = head(checkout)
    with pytest.raises(applying.ApplyError, match="other versions of src/desman/planner"):
        apply(s, p, checkout)  # refused when planned, before any window
    assert head(checkout) == before


def test_a_diff_that_is_not_the_texts_is_never_applied(checkout: Path) -> None:
    s = store()
    diff, details = edited_diff(
        checkout, lambda text: text.replace("MAX_SUBTASKS = 8", "MAX_SUBTASKS = 9")
    )
    p = proposal(s, diff, details)
    before = head(checkout)
    with pytest.raises(applying.ApplyError, match="not the one the proposal's texts make"):
        apply(s, p, checkout)
    assert head(checkout) == before and git(checkout, "status", "--porcelain") == ""


def test_a_detached_head_is_moved_too(checkout: Path) -> None:
    git(checkout, "checkout", "-q", "--detach")
    s = store()
    done = apply(s, a_proposal(s, checkout), checkout)
    assert done.branch is None and done.commit == head(checkout)


def test_only_a_proposed_change_is_applied(checkout: Path) -> None:
    s = store()
    spec = TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    started = proposals.start(s, spec, None)
    with pytest.raises(applying.ApplyError, match="proposes nothing"):
        applying.plan_apply(s.conn, started, Tuning(), repo=checkout)
    with pytest.raises(sqlite3.IntegrityError, match="only a proposed change"):
        proposals.record_application(
            s.conn, started.id, "applied", commit=None, base=None, branch=None, details={}
        )


def test_outside_a_checkout_a_diff_is_not_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkout: Path
) -> None:
    s = store()
    p = a_proposal(s, checkout)
    monkeypatch.setattr(improve, "PACKAGE", tmp_path)
    with pytest.raises(applying.ApplyError, match="git checkout"):
        applying.plan_apply(s.conn, p, Tuning())


# ── Reverting ────────────────────────────────────────────────────────────────


def test_a_proposal_is_reverted_by_a_commit_and_applied_again(checkout: Path) -> None:
    s = store()
    old = planner_rules(checkout)
    p = a_proposal(s, checkout)
    applied = apply(s, p, checkout)
    reverted = revert(s, p, checkout)
    assert reverted.action == "reverted" and reverted.base == applied.commit
    assert reverted.commit == head(checkout) and reverted.details["files"] == [PLANNER]
    assert planner_rules(checkout) == old
    message = git(checkout, "log", "-1", "--format=%B")
    assert message.startswith(f"Revert DESMAN proposal {p.id}")
    assert f"This reverts commit {applied.commit}." in message
    assert git(checkout, "status", "--porcelain") == ""
    with pytest.raises(applying.ApplyError, match="is not applied"):
        applying.plan_revert(s.conn, p, Tuning(), repo=checkout)
    again = apply(s, p, checkout)  # the files are the proposal's base again
    assert again.base == reverted.commit
    assert [a.action for a in proposals.applications(s.conn, p.id)] == [
        "applied",
        "reverted",
        "applied",
    ]
    assert [e.action for e in operator.entries(s.conn)] == [
        "operator_proposal_applied",
        "operator_proposal_reverted",
        "operator_proposal_applied",
    ]


def test_a_revert_over_later_changes_to_the_same_lines_is_refused(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    apply(s, p, checkout)
    planner = package(checkout) / "planner.py"
    rules = BY_ID["planner.rules"]
    mine = improve.replace(planner.read_text(), rules, planner_rules(checkout) + " Mine.")
    planner.write_text(mine)
    git(checkout, "commit", "-q", "-am", "the person's edit")
    before = head(checkout)
    with pytest.raises(applying.ApplyError, match="later commits changed the same lines"):
        revert(s, p, checkout)
    assert head(checkout) == before and git(checkout, "status", "--porcelain") == ""
    assert len(git(checkout, "worktree", "list").splitlines()) == 1


def test_a_commit_not_in_the_history_is_not_reverted(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    applied = apply(s, p, checkout)
    git(checkout, "reset", "-q", "--hard", "HEAD^")
    with pytest.raises(applying.ApplyError, match=f"{applied.commit[:12]} .* not in the history"):  # type: ignore[index]
        revert(s, p, checkout)


# ── Settings ─────────────────────────────────────────────────────────────────


def test_settings_are_applied_and_reverted_without_a_commit(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout, [Change(target="tuning.max_subtasks", value=3)])
    before = head(checkout)
    done = apply(s, p, checkout)
    assert done.commit is None and head(checkout) == before
    assert done.details["settings"] == [{"setting": "tuning.max_subtasks", "old": None, "new": 3}]
    assert open_config().load().tuning.max_subtasks == 3
    back = revert(s, p, checkout)
    assert back.details["settings"] == [{"setting": "tuning.max_subtasks", "old": 3, "new": None}]
    assert open_config().load().tuning.max_subtasks is None


def test_a_setting_changed_since_is_left_by_the_revert(checkout: Path) -> None:
    s = store()
    p = a_proposal(
        s,
        checkout,
        [
            Change(target="tuning.max_subtasks", value=3),
            Change(target="tuning.max_parts", value=2),
        ],
    )
    apply(s, p, checkout)
    with open_config().edit() as config:
        config.tuning.max_subtasks = 5
    plan = applying.plan_revert(s.conn, p, open_config().load().tuning, repo=checkout)
    assert plan.left == ["tuning.max_subtasks"]
    done = applying.carry_out(s.conn, plan, open_config())
    assert done.details["left"] == ["tuning.max_subtasks"]
    tuning = open_config().load().tuning
    assert tuning.max_subtasks == 5 and tuning.max_parts is None


def test_a_setting_changed_after_the_plan_stops_it(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout, [Change(target="tuning.max_subtasks", value=3)])
    plan = applying.plan_apply(s.conn, p, open_config().load().tuning, repo=checkout)
    with open_config().edit() as config:
        config.tuning.max_subtasks = 5
    with pytest.raises(applying.ApplyError, match="tuning.max_subtasks changed since"):
        applying.carry_out(s.conn, plan, open_config())
    assert open_config().load().tuning.max_subtasks == 5


def test_a_prompt_and_a_setting_together(checkout: Path) -> None:
    s = store()
    p = a_proposal(
        s, checkout, [rules_change(checkout), Change(target="tuning.max_subtasks", value=3)]
    )
    done = apply(s, p, checkout)
    assert done.commit == head(checkout) and open_config().load().tuning.max_subtasks == 3


# ── The record ───────────────────────────────────────────────────────────────


def test_applications_come_in_turn_and_are_kept(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)

    def record(action: str) -> None:
        proposals.record_application(
            s.conn, p.id, action, commit=None, base=None, branch=None, details={}
        )

    with pytest.raises(sqlite3.IntegrityError, match="in turn"):
        record("reverted")
    record("applied")
    with pytest.raises(sqlite3.IntegrityError, match="in turn"):
        record("applied")
    record("reverted")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        s.conn.execute("UPDATE applications SET action = 'applied'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        s.conn.execute("DELETE FROM applications")


def test_a_v24_database_upgrades_to_applications(tmp_path: Path) -> None:
    path = tmp_path / "v24.db"
    old = db.connect(path)
    for v, migration in enumerate(db.MIGRATIONS[:24], start=1):
        with db.transaction(old):
            for statement in db._split_sql(migration):
                old.execute(statement)
            old.execute(f"PRAGMA user_version = {v}")
    old.execute("INSERT INTO tasks (id, goal) VALUES ('t1', 'goal')")
    old.execute("INSERT INTO proposals (task_id) VALUES ('t1')")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert proposals.applications(conn, 1) == []


# ── The commands ─────────────────────────────────────────────────────────────


def test_apply_and_revert_commands(checkout: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    s = store()
    p = a_proposal(
        s, checkout, [rules_change(checkout), Change(target="tuning.max_parts", value=3)]
    )
    shown = desman("proposals", "show", str(p.id))
    assert f"Not applied. If you agree, apply it: desman proposals apply {p.id}" in shown.output
    result = desman("proposals", "apply", str(p.id))
    assert result.exit_code == 0, result.output
    assert "was not evaluated" in result.output and "keep the old prompts" in result.output
    assert f"Proposal {p.id} applied in commit {head(checkout)[:12]}" in result.output
    assert "tuning.max_parts: default → 3" in result.output
    assert "applied" in desman("proposals", "list").output
    assert "Applied in commit" in desman("proposals", "show", str(p.id)).output
    data = json.loads(desman("proposals", "show", str(p.id), "--json").output)
    assert [a["action"] for a in data["applications"]] == ["applied"]
    assert json.loads(desman("proposals", "list", "--json").output)[0]["applied"] is True
    twice = desman("proposals", "apply", str(p.id))
    assert twice.exit_code == 1 and "Cannot apply proposal" in twice.output
    back = json.loads(desman("proposals", "revert", str(p.id), "--json").output)
    assert back["action"] == "reverted" and back["commit"] == head(checkout)
    assert "Reverted. Apply it again" in desman("proposals", "show", str(p.id)).output
    none = desman("proposals", "revert", str(p.id))
    assert none.exit_code == 1 and "Cannot revert proposal" in none.output


def test_a_worse_evaluation_is_said_before_applying(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    s = store()
    p = a_proposal(s, checkout)
    proposals.record_evaluation(s.conn, p.id, mode="fake", commit="c", verdict="worse")
    result = desman("proposals", "apply", str(p.id))
    assert "Its last evaluation: the proposed version did worse." in result.output


def test_the_chat_applies_only_after_the_window(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation, "repository", lambda: checkout)
    s = store()
    p = a_proposal(s, checkout)
    before = head(checkout)
    session = Sessions(s.conn).add(None, Role.USER, "apply proposal 1").session_id
    ctx = commands.Context(
        s, open_config(), lambda title, goal, runs_code: None, lambda task: None, lambda: session
    )
    entry = commands.find("proposals apply")
    assert entry is not None
    args = entry.args.model_validate({"proposal": p.id})
    assert entry.kind_of(args) is commands.Kind.CHANGE
    window = entry.handler(ctx, args)
    assert isinstance(window, commands.Proposal)
    assert window.lines[0] == f"Apply proposal {p.id}:"
    assert any("on branch main" in line for line in window.lines)
    assert head(checkout) == before  # nothing until the "yes"
    outcome = window.apply()
    assert outcome.data["commit"] == head(checkout) != before
    [entry_log] = operator.entries(s.conn)
    assert entry_log.session_id == session
    reverting = commands.find("proposals revert")
    assert reverting is not None
    revert_window = reverting.handler(ctx, args)
    assert isinstance(revert_window, commands.Proposal)
    assert revert_window.lines[0] == f"Revert proposal {p.id}:"


# ── What happens meanwhile (review of #71) ───────────────────────────────────


def test_no_hook_of_the_checkout_runs(checkout: Path) -> None:
    """`--no-verify` leaves prepare-commit-msg and others running; none may change what was
    checked, or run at all."""
    hooks = checkout / ".git" / "hooks"
    marker = checkout.parent / "hook-ran"
    for name in ("prepare-commit-msg", "post-commit", "post-checkout", "post-merge"):
        hook = hooks / name
        hook.write_text(
            "#!/bin/sh\n"
            f"echo {name} >> {marker}\n"
            'echo "# sneaked" >> src/desman/agents.py && git add src/desman/agents.py\n'
        )
        hook.chmod(0o755)
    s = store()
    p = a_proposal(s, checkout)
    done = apply(s, p, checkout)
    assert not marker.exists()
    assert done.commit is not None
    assert git(checkout, "show", "--name-only", "--format=", done.commit).split() == [PLANNER]
    revert(s, p, checkout)
    assert not marker.exists()


def test_another_branch_at_the_same_commit_is_not_moved(checkout: Path) -> None:
    s = store()
    p = a_proposal(s, checkout)
    plan = applying.plan_apply(s.conn, p, Tuning(), repo=checkout)
    git(checkout, "checkout", "-q", "-b", "other")
    with pytest.raises(applying.ApplyError, match="branch or HEAD changed"):
        applying.carry_out(s.conn, plan, open_config())
    assert git(checkout, "rev-parse", "main").strip() == git(checkout, "rev-parse", "other").strip()


def test_the_settings_are_put_back_when_the_commit_fails(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = store()
    p = a_proposal(
        s, checkout, [rules_change(checkout), Change(target="tuning.max_subtasks", value=3)]
    )
    plan = applying.plan_apply(s.conn, p, open_config().load().tuning, repo=checkout)
    before = head(checkout)

    def fail(repo: Path, head: str, commit: str) -> None:
        assert open_config().load().tuning.max_subtasks == 3  # set before the commit
        raise applying.ApplyError("no")

    monkeypatch.setattr(applying, "forward", fail)
    with pytest.raises(applying.ApplyError, match="no"):
        applying.carry_out(s.conn, plan, open_config())
    assert open_config().load().tuning.max_subtasks is None
    assert head(checkout) == before and proposals.applications(s.conn, p.id) == []
