"""Applying a proposal of the Improvement Analyst, and reverting it (ROADMAP 5.6, D8). Only the
person does either: `desman proposals apply N` and `desman proposals revert N`, or the same
in the chat after the person's "yes" in the window. Nothing applies a proposal on its own.

Applying makes the proposal's diff a git commit of the checkout DESMAN runs from, on the
branch it is on. The commit is made in a git worktree of HEAD, its files built by code exactly
as the evaluation builds the proposed version (`desman.evaluation`): the diff is never applied
as a patch; the files must make the proposal's diff to the byte, and differ from HEAD only in
the targets' texts. Then the checkout is fast-forwarded to that commit, so nothing is merged:
if HEAD moved, or the person has uncommitted changes in those files, nothing changes. The
proposal's settings are written to the person's `tuning`.

Reverting goes the other way: `git revert` of that commit, made in a worktree and checked to
touch only the files the commit changed, then a fast-forward. Each setting goes back to the
value it had, unless the person changed it since (it is then left as it is, and said).

Each is recorded once (`core.proposals.record_application`), with its entry in the operator's
journal; the database takes applied and reverted only in turn. A process killed after the
commit and before the record leaves the commit unrecorded: applying again is then refused (its
files are no longer the proposal's base), and the person reverts that commit with git. Processes already running (a
worker, the chat) keep the prompts they loaded until they start again; settings are read anew
by every run.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import sqlite3
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from desman import evaluation, improve
from desman.config import ConfigError, ConfigManager, Tuning
from desman.core import proposals
from desman.core.proposals import Application, Proposal, Status

Action = Literal["applied", "reverted"]

IDENTITY = ("-c", "user.name=DESMAN", "-c", "user.email=desman@localhost")
"""Who commits when the checkout has no git identity of its own."""
TRAILER = "Desman-Proposal"
"""The commit message's trailer naming the proposal."""


class ApplyError(Exception):
    """The proposal cannot be applied or reverted now (nothing changed): why, for the person."""


@dataclass(frozen=True)
class Plan:
    """What applying or reverting a proposal would do, checked: shown to the person, then
    carried out only if nothing it rests on changed in between."""

    proposal: Proposal
    action: Action
    last: Application | None
    """What was done before (the application a revert undoes)."""
    repo: Path | None = None
    """The checkout; None when no commit is made (only settings change)."""
    head: str | None = None
    branch: str | None = None
    files: list[str] = field(default_factory=list)
    """Relative to the checkout: the files the commit changes."""
    settings: list[dict[str, Any]] = field(default_factory=list)
    """{setting, old, new}: what the settings would become."""
    left: list[str] = field(default_factory=list)
    """Settings a revert leaves: the person changed them since they were applied."""


# ── Planning ─────────────────────────────────────────────────────────────────


def plan_apply(
    conn: sqlite3.Connection, proposal: Proposal, tuning: Tuning, *, repo: Path | None = None
) -> Plan:
    """What applying `proposal` would do; raises `ApplyError` when it cannot be applied."""
    if proposal.status is not Status.PROPOSED:
        raise ApplyError(f"proposal {proposal.id} proposes nothing ({proposal.status})")
    last = proposals.last_application(conn, proposal.id)
    if last is not None and last.action == "applied":
        raise ApplyError(f"proposal {proposal.id} is already applied{_in(last.commit)}")
    try:
        evaluation.proposed_tuning(tuning, proposal)  # the values are still valid
    except evaluation.EvaluationError as e:
        raise ApplyError(str(e)) from e
    settings = [
        {"setting": item["setting"], "old": _get(tuning, item["setting"]), "new": item["new"]}
        for item in proposal.details.get("settings", [])
    ]
    if not proposal.diff:
        return Plan(proposal, "applied", last, settings=settings)
    repo, head, branch = _checkout(repo)
    files = sorted({improve.SOURCE_PREFIX + improve.BY_ID[t].path for t in _targets(proposal)})
    _clean(repo, files)
    differ = [
        name
        for name, digest in proposal.details.get("base", {}).items()
        if hashlib.sha256(_git(repo, "show", f"HEAD:{name}").encode()).hexdigest() != digest
    ]
    if differ:  # checked again on the commit's own files (`_commit_apply`)
        raise ApplyError(
            f"the proposal was made from other versions of {', '.join(differ)} than HEAD has: "
            "ask for a new analysis"
        )
    return Plan(proposal, "applied", last, repo, head, branch, files, settings)


def plan_revert(
    conn: sqlite3.Connection, proposal: Proposal, tuning: Tuning, *, repo: Path | None = None
) -> Plan:
    """What reverting `proposal` would do; raises `ApplyError` when it cannot be reverted."""
    last = proposals.last_application(conn, proposal.id)
    if last is None or last.action != "applied":
        raise ApplyError(f"proposal {proposal.id} is not applied")
    settings, left = [], []
    for item in last.details.get("settings", []):
        now = _get(tuning, item["setting"])
        if now == item["new"]:
            settings.append({"setting": item["setting"], "old": now, "new": item["old"]})
        else:
            left.append(str(item["setting"]))
    if last.commit is None:
        return Plan(proposal, "reverted", last, settings=settings, left=left)
    repo, head, branch = _checkout(repo)
    try:
        _git(repo, "merge-base", "--is-ancestor", last.commit, head)
    except ApplyError:
        raise ApplyError(
            f"commit {last.commit[:12]} of proposal {proposal.id} is not in the history of the "
            "checkout's HEAD: revert it with git by hand"
        ) from None
    files = sorted(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", last.commit).split()
    )
    _clean(repo, files)
    return Plan(proposal, "reverted", last, repo, head, branch, files, settings, left)


def _in(commit: str | None) -> str:
    return f" (commit {commit[:12]})" if commit else ""


def _targets(proposal: Proposal) -> list[str]:
    found = [str(c.get("target")) for c in proposal.details.get("changes", [])]
    if not found or any(t not in improve.BY_ID for t in found):
        raise ApplyError(f"proposal {proposal.id} names no known targets for its diff")
    return found


def _checkout(repo: Path | None) -> tuple[Path, str, str | None]:
    """The checkout, its HEAD and the branch it is on (None: detached)."""
    try:
        repo = evaluation.repository() if repo is None else repo
    except evaluation.EvaluationError as e:
        raise ApplyError(str(e)) from e
    head = _git(repo, "rev-parse", "HEAD").strip()
    try:
        branch: str | None = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").strip()
    except ApplyError:
        branch = None
    return repo, head, branch


def _clean(repo: Path, files: list[str]) -> None:
    """The person has nothing uncommitted (staged or not) in `files`: a commit over them
    would be refused by git, or would mix their work into DESMAN's."""
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *files)
    if status.strip():
        raise ApplyError(
            f"there are uncommitted changes in {', '.join(files)}: commit them or put them "
            "aside first"
        )


# ── Carrying out ─────────────────────────────────────────────────────────────


def carry_out(
    conn: sqlite3.Connection,
    plan: Plan,
    manager: ConfigManager,
    *,
    session_id: str | None = None,
) -> Application:
    """Do what `plan` says, if nothing it rests on changed: the commit (made in a worktree,
    then the checkout fast-forwarded to it), then the settings, then the record."""
    if proposals.last_application(conn, plan.proposal.id) != plan.last:
        raise ApplyError(f"proposal {plan.proposal.id} was applied or reverted meanwhile")
    try:  # a broken config, or a setting changed since, is found before any commit
        tuning = manager.load().tuning
    except ConfigError as e:
        raise ApplyError(str(e)) from e
    for item in plan.settings:
        if _get(tuning, item["setting"]) != item["old"]:
            raise ApplyError(f"the setting {item['setting']} changed since: look again")
    commit = None
    if plan.repo is not None:
        assert plan.head is not None
        if _git(plan.repo, "rev-parse", "HEAD").strip() != plan.head:
            raise ApplyError("the checkout's HEAD moved since: look again")
        _clean(plan.repo, plan.files)
        with _worktrees(plan.repo, plan.head, 2 if plan.action == "applied" else 1) as trees:
            if plan.action == "applied":
                commit = _commit_apply(plan, trees[0], trees[1])
            else:
                commit = _commit_revert(plan, trees[0])
            _forward(plan.repo, plan.head, commit)
    settings = _set(manager, plan.settings)
    details: dict[str, Any] = {"files": plan.files, "settings": settings}
    if plan.left:
        details["left"] = plan.left
    return proposals.record_application(
        conn,
        plan.proposal.id,
        plan.action,
        commit=commit,
        base=plan.head if commit else None,
        branch=plan.branch if commit else None,
        details=details,
        session_id=session_id,
    )


def _commit_apply(plan: Plan, base: Path, tree: Path) -> str:
    """The proposal's commit, made in `tree` (a worktree of HEAD) by code, as the evaluation
    makes the proposed version; `base` is HEAD as it is, to compare with."""
    p = plan.proposal
    try:
        evaluation._check_base(tree, p, plan.head or "")
        evaluation._build(tree, p)
        evaluation._check_changes(base, tree, p)
    except evaluation.EvaluationError as e:
        raise ApplyError(str(e)) from e
    _git(tree, "add", "--", *plan.files)
    message = (
        f"DESMAN proposal {p.id}: {', '.join(_targets(p))}\n\n"
        f"Applied by the person: desman proposals apply {p.id}\n"
        f"Why: desman proposals show {p.id}\n\n"
        f"{TRAILER}: {p.id}\n"
    )
    return _commit(tree, message, plan.files)


def _commit_revert(plan: Plan, tree: Path) -> str:
    """The revert of the proposal's commit, made in `tree` (a worktree of HEAD)."""
    assert plan.last is not None and plan.last.commit is not None
    applied = plan.last.commit
    try:
        _git(tree, "revert", "--no-commit", applied)
    except ApplyError:
        raise ApplyError(
            f"later commits changed the same lines as commit {applied[:12]}: revert it with "
            "git by hand"
        ) from None
    p = plan.proposal
    message = (
        f"Revert DESMAN proposal {p.id}\n\n"
        f"This reverts commit {applied}.\n"
        f"Reverted by the person: desman proposals revert {p.id}\n\n"
        f"{TRAILER}: {p.id}\n"
    )
    return _commit(tree, message, plan.files)


def _commit(tree: Path, message: str, files: list[str]) -> str:
    """Commit what is staged in `tree`, after checking it is exactly a change to `files`.
    Hooks are not run: what is committed was built and checked by code, and the person's
    hooks could change it after the check, or take any time."""
    staged = set(_git(tree, "diff", "--cached", "--name-only").split())
    if staged != set(files):
        raise ApplyError("the commit would change other files than the proposal's")
    rest = _git(tree, "status", "--porcelain", "--untracked-files=all", "--ignored")
    if any(not line.startswith("M  ") for line in rest.splitlines()):
        raise ApplyError("the worktree holds changes that would not be committed")
    _git(tree, *_identity(tree), "commit", "--quiet", "--no-verify", "--file", "-", stdin=message)
    return _git(tree, "rev-parse", "HEAD").strip()


def _identity(tree: Path) -> tuple[str, ...]:
    """The checkout's own git identity if it has one, otherwise DESMAN's."""
    try:
        _git(tree, "config", "user.name")
        _git(tree, "config", "user.email")
    except ApplyError:
        return IDENTITY
    return ()


def _forward(repo: Path, head: str, commit: str) -> None:
    """Move the checkout from `head` to `commit`, its child, without merging anything."""
    if _git(repo, "rev-parse", f"{commit}^").strip() != head:
        raise ApplyError("the commit was not made on the checkout's HEAD")
    try:
        _git(repo, "merge", "--ff-only", "--quiet", commit)
    except ApplyError as e:
        raise ApplyError(f"the checkout could not be moved to the new commit: {e}") from e


@contextlib.contextmanager
def _worktrees(repo: Path, commit: str, count: int) -> Iterator[list[Path]]:
    """`count` detached worktrees of `commit`, removed afterwards."""
    folder = Path(tempfile.mkdtemp(prefix="desman-apply-"))
    trees: list[Path] = []
    try:
        for n in range(count):
            tree = folder / f"tree{n}"
            _git(repo, "worktree", "add", "--quiet", "--detach", str(tree), commit)
            trees.append(tree)
        yield trees
    finally:
        for tree in trees:
            with contextlib.suppress(ApplyError):
                _git(repo, "worktree", "remove", "--force", str(tree))
        shutil.rmtree(folder, ignore_errors=True)
        with contextlib.suppress(ApplyError):
            _git(repo, "worktree", "prune")


def _git(where: Path, *args: str, stdin: str | None = None) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(where), *args],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=evaluation.GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ApplyError(f"git {args[0]}: {e}") from e
    if done.returncode != 0:
        raise ApplyError(f"git {args[0]}: {done.stderr.strip() or done.returncode}")
    return done.stdout


# ── Settings ─────────────────────────────────────────────────────────────────


def _get(tuning: Tuning, setting_id: str) -> Any:
    setting = improve.SETTINGS_BY_ID.get(setting_id)
    if setting is None:
        raise ApplyError(f"unknown setting {setting_id!r}")
    if setting.is_model:
        return getattr(tuning.models, setting.name)
    return getattr(tuning, setting.name)


def _set(manager: ConfigManager, settings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give the settings their new values, under the config's lock; what each was before it is
    recorded as it was then (`old`), which a revert restores."""
    if not settings:
        return []
    done = []
    with manager.edit() as config:
        data = config.tuning.model_dump()
        for item in settings:
            setting = improve.SETTINGS_BY_ID[item["setting"]]
            done.append(
                {"setting": setting.id, "old": _get(config.tuning, setting.id), "new": item["new"]}
            )
            if setting.is_model:
                data["models"][setting.name] = item["new"]
            else:
                data[setting.name] = item["new"]
        try:
            config.tuning = Tuning.model_validate(data)
        except ValidationError as e:
            raise ApplyError(f"the settings are not valid: {e}") from e
    return done
