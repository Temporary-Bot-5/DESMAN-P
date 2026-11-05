"""Applying a proposal of the Improvement Analyst, and reverting it (ROADMAP 5.6, D8). Only the
person does either: `desman proposals apply N` and `desman proposals revert N`, or the same
in the chat after the person's "yes" in the window. Nothing applies a proposal on its own.
G.7b shares plain-data validation with isolated proposal checks.

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
journal; the database takes applied and reverted only in turn. No hook of the checkout runs. A process killed after the
commit and before the record leaves the commit unrecorded: applying again is then refused (its
files are no longer the proposal's base), and the person reverts that commit with git. Processes already running (a
worker, the chat) keep the prompts they loaded until they start again; settings are read anew
by every run.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import sqlite3
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from desman import evaluation, improve
from desman.config import ConfigError, ConfigManager, Tuning
from desman.core import db, proposals
from desman.core.proposals import Application, Proposal, Status
from desman.i18n import t
from desman.net import sign

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
        raise ApplyError(
            t("proposal {id} proposes nothing ({status})", id=proposal.id, status=proposal.status)
        )
    last = proposals.last_application(conn, proposal.id)
    if last is not None and last.action == "applied" and last.commit is not None:
        raise ApplyError(
            t(
                "proposal {id} is already applied (commit {commit})",
                id=proposal.id,
                commit=last.commit[:12],
            )
        )
    if last is not None and last.action == "applied":
        raise ApplyError(t("proposal {id} is already applied", id=proposal.id))
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
    repo, head, branch = checkout(repo)
    files = sorted({improve.SOURCE_PREFIX + improve.BY_ID[t].path for t in _targets(proposal)})
    clean(repo, files)
    differ = [
        name
        for name, digest in proposal.details.get("base", {}).items()
        if hashlib.sha256(_git(repo, "show", f"HEAD:{name}").encode()).hexdigest() != digest
    ]
    if differ:  # checked again on the commit's own files (`_commit_apply`)
        raise ApplyError(
            t(
                "the proposal was made from other versions of {files} than HEAD has: ask for a "
                "new analysis",
                files=", ".join(differ),
            )
        )
    return Plan(proposal, "applied", last, repo, head, branch, files, settings)


def plan_revert(
    conn: sqlite3.Connection, proposal: Proposal, tuning: Tuning, *, repo: Path | None = None
) -> Plan:
    """What reverting `proposal` would do; raises `ApplyError` when it cannot be reverted."""
    last = proposals.last_application(conn, proposal.id)
    if last is None or last.action != "applied":
        raise ApplyError(t("proposal {id} is not applied", id=proposal.id))
    settings, left = [], []
    for item in last.details.get("settings", []):
        now = _get(tuning, item["setting"])
        if now == item["new"]:
            settings.append({"setting": item["setting"], "old": now, "new": item["old"]})
        else:
            left.append(str(item["setting"]))
    if last.commit is None:
        return Plan(proposal, "reverted", last, settings=settings, left=left)
    repo, head, branch = checkout(repo)
    try:
        _git(repo, "merge-base", "--is-ancestor", last.commit, head)
    except ApplyError:
        raise ApplyError(
            t(
                "commit {commit} of proposal {id} is not in the history of the checkout's HEAD: "
                "revert it with git by hand",
                commit=last.commit[:12],
                id=proposal.id,
            )
        ) from None
    files = sorted(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", last.commit).split()
    )
    clean(repo, files)
    return Plan(proposal, "reverted", last, repo, head, branch, files, settings, left)


def _targets(proposal: Proposal) -> list[str]:
    found = [str(c.get("target")) for c in proposal.details.get("changes", [])]
    if not found or any(t not in improve.BY_ID for t in found):
        raise ApplyError(t("proposal {id} names no known targets for its diff", id=proposal.id))
    return found


def checkout(repo: Path | None) -> tuple[Path, str, str | None]:
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


def clean(repo: Path, files: list[str]) -> None:
    """The person has nothing uncommitted (staged or not) in `files`: a commit over them
    would be refused by git, or would mix their work into DESMAN's."""
    status = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *files)
    if status.strip():
        raise ApplyError(
            t(
                "there are uncommitted changes in {files}: commit them or put them aside first",
                files=", ".join(files),
            )
        )


# ── Carrying out ─────────────────────────────────────────────────────────────


def carry_out(
    conn: sqlite3.Connection,
    plan: Plan,
    manager: ConfigManager,
    *,
    session_id: str | None = None,
) -> Application:
    """Do what `plan` says, if nothing it rests on changed: the settings (each only if it
    still has the value the plan showed), then the commit (made in a worktree, then the
    checkout fast-forwarded to it), then the record. If the commit fails, the settings are
    put back."""
    if proposals.last_application(conn, plan.proposal.id) != plan.last:
        raise ApplyError(t("proposal {id} was applied or reverted meanwhile", id=plan.proposal.id))
    if plan.repo is not None:
        _unmoved(plan)
        clean(plan.repo, plan.files)
    settings = _set(manager, plan.settings)
    commit = None
    try:
        if plan.repo is not None:
            assert plan.head is not None
            if plan.action == "applied":
                commit = _commit_apply(plan)
            else:
                with _worktrees(plan.repo, plan.head, 1) as trees:
                    commit = _commit_revert(plan, trees[0])
            _unmoved(plan)
            forward(plan.repo, plan.head, commit)
    except BaseException:
        back = [{"setting": i["setting"], "old": i["new"], "new": i["old"]} for i in settings]
        with contextlib.suppress(ApplyError, ConfigError):
            _set(manager, back)
        raise
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


def _unmoved(plan: Plan) -> None:
    """The checkout is on the HEAD and the branch the plan showed."""
    assert plan.repo is not None
    _, head, branch = checkout(plan.repo)
    if (head, branch) != (plan.head, plan.branch):
        raise ApplyError(t("the checkout's branch or HEAD changed since: look again"))


def _commit_apply(plan: Plan) -> str:
    """Build the proposal's commit using the shared text-only builder."""
    p = plan.proposal
    message = (
        f"DESMAN proposal {p.id}: {', '.join(_targets(p))}\n\n"
        f"Applied by the person: desman proposals apply {p.id}\n"
        f"Why: desman proposals show {p.id}\n\n"
        f"{TRAILER}: {p.id}\n"
    )
    assert plan.repo is not None and plan.head is not None
    return commit_texts(
        plan.repo,
        plan.head,
        changes=p.details.get("changes", []),
        base=p.details.get("base", {}),
        diff=p.diff or "",
        files=plan.files,
        message=message,
    )


def commit_texts(
    repo: Path,
    head: str,
    *,
    changes: list[dict[str, Any]],
    base: dict[str, str],
    diff: str,
    files: list[str],
    message: str,
) -> str:
    """Build and verify a text-only commit in two detached worktrees; leave HEAD alone."""
    with _worktrees(repo, head, 2) as trees:
        original, tree = trees
        try:
            evaluation._check_base(tree, base, head)
            evaluation._build(tree, changes, diff)
            evaluation._check_changes(original, tree, changes)
        except evaluation.EvaluationError as e:
            raise ApplyError(str(e)) from e
        _git(tree, "add", "--", *files)
        return _commit(tree, message, files)


def _commit_revert(plan: Plan, tree: Path) -> str:
    """The revert of the proposal's commit, made in `tree` (a worktree of HEAD)."""
    assert plan.last is not None and plan.last.commit is not None
    applied = plan.last.commit
    try:
        _git(tree, "revert", "--no-commit", applied)
    except ApplyError:
        raise ApplyError(
            t(
                "later commits changed the same lines as commit {commit}: revert it with git by "
                "hand",
                commit=applied[:12],
            )
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
    """Commit what is staged in `tree`, after checking it is exactly a change to `files`; the
    commit must hold that very tree. No hook runs (`_git`): what is committed was built and
    checked by code, and a hook could change it after the check, or take any time."""
    staged = set(_git(tree, "diff", "--cached", "--name-only").split())
    if staged != set(files):
        raise ApplyError(t("the commit would change other files than the proposal's"))
    rest = _git(tree, "status", "--porcelain", "--untracked-files=all", "--ignored")
    if any(not line.startswith("M  ") for line in rest.splitlines()):
        raise ApplyError(t("the worktree holds changes that would not be committed"))
    checked = _git(tree, "write-tree").strip()
    parent = _git(tree, "rev-parse", "HEAD").strip()
    _git(tree, *_identity(tree), "commit", "--quiet", "--file", "-", stdin=message)
    commit = _git(tree, "rev-parse", "HEAD").strip()
    made = _git(tree, "rev-parse", f"{commit}^{{tree}}", f"{commit}^").split()
    if made != [checked, parent]:
        raise ApplyError(t("the commit is not what was checked"))
    return commit


def _identity(tree: Path) -> tuple[str, ...]:
    """The checkout's own git identity if it has one, otherwise DESMAN's."""
    try:
        _git(tree, "config", "user.name")
        _git(tree, "config", "user.email")
    except ApplyError:
        return IDENTITY
    return ()


def forward(repo: Path, head: str, commit: str) -> None:
    """Move the checkout from `head` to `commit`, its child, without merging anything."""
    if _git(repo, "rev-parse", f"{commit}^").strip() != head:
        raise ApplyError(t("the commit was not made on the checkout's HEAD"))
    try:
        _git(repo, "merge", "--ff-only", "--quiet", commit)
    except ApplyError as e:
        raise ApplyError(
            t("the checkout could not be moved to the new commit: {error}", error=e)
        ) from e


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


HOOKS_OFF = ("-c", f"core.hooksPath={os.devnull}")
"""Every git call runs without the checkout's hooks: `--no-verify` leaves some running
(prepare-commit-msg, post-commit, post-checkout, post-merge), and any could change what was
checked."""


def _git(where: Path, *args: str, stdin: str | None = None) -> str:
    try:
        done = subprocess.run(
            ["git", "-c", "core.autocrlf=false", "-C", str(where), *HOOKS_OFF, *args],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
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
        raise ApplyError(t("unknown setting {name}", name=setting_id))
    if setting.is_model:
        return getattr(tuning.models, setting.name)
    return getattr(tuning, setting.name)


def _set(manager: ConfigManager, settings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each setting its `new` value, under the config's lock, only if every one still has
    its `old` value (the one the person was shown): otherwise nothing is written. Returns what
    was changed."""
    if not settings:
        return []
    try:
        with manager.edit() as config:
            data = config.tuning.model_dump()
            for item in settings:
                setting = improve.SETTINGS_BY_ID[item["setting"]]
                if _get(config.tuning, setting.id) != item["old"]:
                    raise ApplyError(
                        t("the setting {name} changed since: look again", name=setting.id)
                    )
                if setting.is_model:
                    data["models"][setting.name] = item["new"]
                else:
                    data[setting.name] = item["new"]
            try:
                config.tuning = Tuning.model_validate(data)
            except ValidationError as e:
                raise ApplyError(t("the settings are not valid: {error}", error=e)) from e
    except ConfigError as e:
        raise ApplyError(str(e)) from e
    return [{"setting": i["setting"], "old": i["old"], "new": i["new"]} for i in settings]


@dataclass(frozen=True)
class UpdatePlan:
    """A verified release and the exact checkout move shown to the person."""

    release: sign.Release
    action: Action
    repo: Path
    head: str
    branch: str
    target: str
    last: proposals.Update | None
    files: list[str]
    commits: list[str]
    commit_count: int
    protected: list[str]
    dependencies: bool
    schema: int
    root: Path | None = None


def _ancestor(repo: Path, older: str, newer: str) -> bool:
    try:
        _git(repo, "merge-base", "--is-ancestor", older, newer)
    except ApplyError:
        return False
    return True


def plan_update(
    conn: sqlite3.Connection,
    number: int | None = None,
    *,
    repo: Path | None = None,
    root: Path | None = None,
) -> UpdatePlan:
    from desman.net import updates

    try:
        release = updates.verified(conn, number, root=root)
    except (ValueError, OSError) as e:
        raise ApplyError(t("The release cannot be verified: {error}", error=e)) from e
    if release is None:
        if number is not None:
            raise ApplyError(t("No release {number}.", number=number))
        raise ApplyError(
            t(
                "No release received yet; releases arrive while `desman worker` runs with desman-global on."
            )
        )
    last = proposals.installed_update(conn)
    repo, head, branch = checkout(repo)
    if branch is None:
        raise ApplyError(t("The checkout must be on a branch."))
    _git(repo, "fetch", "--quiet")
    commit = release.commit
    try:
        matches = (
            _git(repo, "cat-file", "-t", commit).strip() == "commit"
            and _git(repo, "rev-parse", f"{commit}^").strip() == release.parent
            and sorted(
                _git(
                    repo, "diff-tree", "--no-commit-id", "--name-only", "-r", release.parent, commit
                ).splitlines()
            )
            == sorted(release.files)
        )
    except ApplyError:
        matches = False
    if not matches:
        raise ApplyError(t("The fetched commit is not the one the root signed."))
    if _ancestor(repo, commit, head):
        raise ApplyError(t("This release is already installed."))
    if not _ancestor(repo, head, commit):
        raise ApplyError(
            t(
                "The checkout has commits of its own or is on another line; update it with git by hand."
            )
        )
    files = _git(repo, "diff", "--name-only", head, commit).splitlines()
    clean(repo, files)
    commits = _git(repo, "log", "--format=%H%x09%s", f"{head}..{commit}").splitlines()

    protected_files: set[str] = set()
    for revision in (head, commit):
        manifest_path = "src/desman/protected.toml"
        if (
            manifest_path
            not in _git(repo, "ls-tree", "--name-only", revision, manifest_path).splitlines()
        ):
            continue
        manifest = _git(repo, "show", f"{revision}:{manifest_path}")
        protected_files.update(
            "src/desman/" + name for name in tomllib.loads(manifest).get("package", {})
        )
    protected_files.add("src/desman/protected.toml")
    return UpdatePlan(
        release,
        "applied",
        repo,
        head,
        branch,
        commit,
        last,
        files,
        commits[:50],
        len(commits),
        sorted(set(files) & protected_files),
        "pyproject.toml" in files,
        db.SCHEMA_VERSION,
        root,
    )


def plan_update_revert(conn: sqlite3.Connection, *, repo: Path | None = None) -> UpdatePlan:

    last = proposals.installed_update(conn)
    if last is None or last.action != "applied":
        raise ApplyError(t("No release is installed to revert."))
    repo, head, branch = checkout(repo)
    if (head, branch) != (last.commit, last.branch):
        raise ApplyError(t("Commits were made since; go back with git by hand."))
    schema = int(last.details["schema"])
    if conn.execute("PRAGMA user_version").fetchone()[0] > schema:
        raise ApplyError(
            t("The database schema grew since this update; the older version could not open it.")
        )
    files = _git(repo, "diff", "--name-only", last.base, last.commit).splitlines()
    clean(repo, files)
    # The applied record supplies the original signed body for the window, not installation.

    row = conn.execute(
        "SELECT signed_json FROM releases WHERE number = ?", (last.release,)
    ).fetchone()
    if row is None:
        raise ApplyError(t("No release {number}.", number=last.release))
    release = sign.Release.model_validate(sign.Signed.model_validate_json(row[0]).record.body)
    return UpdatePlan(
        release,
        "reverted",
        repo,
        head,
        last.branch,
        last.base,
        last,
        files,
        [],
        int(last.details.get("commit_count", 0)),
        last.details.get("protected", []),
        "pyproject.toml" in files,
        db.SCHEMA_VERSION,
    )


def forward_update(repo: Path, head: str, commit: str) -> None:
    """Fast-forward through any number of signed history's commits, without hooks."""
    if not _ancestor(repo, head, commit):
        raise ApplyError(
            t(
                "The checkout has commits of its own or is on another line; update it with git by hand."
            )
        )
    _git(repo, "merge", "--ff-only", "--quiet", commit)


def carry_out_update(
    conn: sqlite3.Connection,
    plan: UpdatePlan,
    *,
    session_id: str | None = None,
) -> proposals.Update:
    """Move then record. A crash between them requires the person to restore the old HEAD
    with git before trying again. Revert uses reset --keep: release commits remain in the
    remote and no release history is lost. Running workers/chats must be restarted."""
    from desman.net import updates

    if proposals.installed_update(conn) != plan.last:
        raise ApplyError(t("The installed release changed since: look again."))
    if checkout(plan.repo)[1:] != (plan.head, plan.branch):
        raise ApplyError(t("the checkout's branch or HEAD changed since: look again"))
    clean(plan.repo, plan.files)
    if plan.action == "applied":
        try:
            if updates.verified(conn, plan.release.number, root=plan.root) != plan.release:
                raise ValueError("release changed")
        except (ValueError, OSError) as e:
            raise ApplyError(t("The release cannot be verified: {error}", error=e)) from e
        forward_update(plan.repo, plan.head, plan.target)
    else:
        assert plan.last is not None
        if conn.execute("PRAGMA user_version").fetchone()[0] > int(plan.last.details["schema"]):
            raise ApplyError(
                t(
                    "The database schema grew since this update; the older version could not open it."
                )
            )
        _git(plan.repo, "reset", "--keep", plan.target)
    return proposals.record_update(
        conn,
        plan.release.number,
        plan.action,
        commit=plan.release.commit,
        base=plan.head if plan.action == "applied" else plan.target,
        branch=plan.branch,
        details={
            "files": plan.files,
            "commit_count": plan.commit_count,
            "protected": plan.protected,
            "schema": plan.schema,
        },
        session_id=session_id,
    )
