"""Operator decisions and root-signed releases (ADR 0004 decision 8; ROADMAP G.7c).

Only the person's explicit confirmation builds a commit, in detached worktrees using the
same checked text builder as local proposals. No hook runs and DESMAN never pushes. The
owner supplies a checkout on a branch and pushes it themselves afterwards.

An exclusive releases.lock serialises accepts and stabilises the planned release number.
Git runs without a database transaction; one short write transaction rechecks eligibility
and records the decision. Git and SQLite cannot commit together:
a crash or database failure after the fast-forward leaves an unrecorded commit. The person
reverts it with git before trying again; the error names that commit. Releases are retained
here only: distributing or installing them belongs to G.8.
"""

from __future__ import annotations

import fcntl
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

from desman.core import db
from desman.i18n import t
from desman.net import protocol, sign
from desman.registry import admin, checks, store

RELEASE_DAYS: Final = 365


def find(conn: sqlite3.Connection, prefix: str) -> store.ProposalSubmission:
    matches = store.find_proposal(conn, prefix)
    if len(matches) != 1:
        raise admin.AdminError(t("No unique proposal matches {id}.", id=prefix))
    found = store.proposal(conn, matches[0])
    assert found is not None
    return found


def _eligible(conn: sqlite3.Connection, prefix: str) -> store.ProposalSubmission:
    found = find(conn, prefix)
    outcomes = [c["outcome"] for c in checks.results(conn, found.id)]
    active = conn.execute(
        "SELECT 1 FROM proposal_leases WHERE proposal = ? AND ended IS NULL", (found.id,)
    ).fetchone()
    released = conn.execute("SELECT 1 FROM releases WHERE proposal = ?", (found.id,)).fetchone()
    if (
        found.status != "reviewed"
        or active
        or released
        or "checked" not in outcomes
        or "refused" in outcomes
    ):
        raise admin.AdminError(t("Proposal {id} is not eligible for release.", id=found.id))
    return found


def plan(conn: sqlite3.Connection, prefix: str, repo: Path) -> dict[str, Any]:
    """Read-only preview, bound to the checkout and the exact submission and checks."""
    from desman import applying, improve

    found = _eligible(conn, prefix)
    item = found.submission
    targets = [c.target for c in item.details.changes]
    if any(target not in improve.BY_ID for target in targets):
        raise admin.AdminError(t("proposal {id} names no known targets for its diff", id=found.id))
    files = sorted({improve.SOURCE_PREFIX + improve.BY_ID[target].path for target in targets})
    try:
        repo, head, branch = applying.checkout(repo.resolve())
        if branch is None:
            raise admin.AdminError(t("The release checkout must be on a branch."))
        applying.clean(repo, files)
        if set(item.details.base) != set(files):
            raise admin.AdminError(t("The submission base must name exactly the target files."))
        for name, digest in item.details.base.items():
            text = applying._git(repo, "show", f"{head}:{name}")
            if hashlib.sha256(text.encode()).hexdigest() != digest:
                raise admin.AdminError(
                    t("The submission base differs from HEAD: {file}.", file=name)
                )
    except applying.ApplyError as e:
        raise admin.AdminError(str(e)) from e
    checked = checks.results(conn, found.id)
    ids = [c["record"]["record"]["id"] for c in checked]
    if not 1 <= len(ids) <= 16:
        raise admin.AdminError(t("Proposal {id} is not eligible for release.", id=found.id))
    number = int(conn.execute("SELECT coalesce(max(number), 0) + 1 FROM releases").fetchone()[0])
    bound = dict(
        number=number,
        proposal=found.id,
        submission=protocol.submission_hash(item),
        repo=str(repo),
        branch=branch,
        head=head,
        files=files,
        checks=ids,
    )
    return bound | dict(
        confirmation=hashlib.sha256(sign.canonical(bound)).hexdigest(),
        base_commit=item.base_commit,
        node=found.node,
        summary=item.summary,
        diff=item.diff,
        check_results=checked,
    )


def accept(
    conn: sqlite3.Connection,
    prefix: str,
    repo: Path,
    confirmation: str,
    *,
    directory: Path,
    root_key: Path | None = None,
    root_pub: Path | None = None,
    now: datetime,
) -> dict[str, Any]:
    """Load the root first, then lock, replan, build, sign, forward and record once."""
    from desman import applying

    root = admin.load_root_key(
        directory / store.ROOT_KEY if root_key is None else root_key, root_pub
    )
    with (directory / "releases.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise admin.AdminError(t("Another release is being made.")) from e
        commit: str | None = None
        forwarded = False
        try:
            preview = plan(conn, prefix, repo)
            if confirmation != preview["confirmation"]:
                raise admin.AdminError(t("The release confirmation changed: preview it again."))
            found = _eligible(conn, prefix)
            number = preview["number"]
            item = found.submission
            targets = ", ".join(c.target for c in item.details.changes)
            message = (
                f"DESMAN release {number}: {targets}\n\n"
                f"Accepted by the operator: desman-registry proposal accept {found.id}\n"
                f"Submitted by node {found.node}\n\n"
                f"Desman-Submission: {found.id}\nDesman-Release: {number}\n"
            )
            path = Path(preview["repo"])
            commit = applying.commit_texts(
                path,
                preview["head"],
                changes=[c.model_dump() for c in item.details.changes],
                base=item.details.base,
                diff=item.diff,
                files=preview["files"],
                message=message,
            )
            _, head, branch = applying.checkout(path)
            if (head, branch) != (preview["head"], preview["branch"]):
                raise admin.AdminError(t("the checkout's branch or HEAD changed since: look again"))
            applying.clean(path, preview["files"])
            # Validate the body before advancing the checkout.
            body = protocol.Release(
                number=number,
                commit=commit,
                parent=head,
                branch=branch or "",
                proposal=found.id,
                submission=preview["submission"],
                checks=preview["checks"],
                summary=item.summary,
                files=preview["files"],
            )
            signed = sign.sign(
                root,
                sign.RecordKind.RELEASE,
                body.model_dump(mode="json"),
                issued_at=now,
                lifetime=timedelta(days=RELEASE_DAYS),
            )
            store.trust(conn, root.public, now).receive(signed.model_dump(mode="json"))
            applying.forward(path, head, commit)
            forwarded = True
            with db.transaction(conn):
                _eligible(conn, prefix)
                current_ids = [c["record"]["record"]["id"] for c in checks.results(conn, found.id)]
                next_number = int(
                    conn.execute("SELECT coalesce(max(number), 0) + 1 FROM releases").fetchone()[0]
                )
                if current_ids != preview["checks"] or next_number != number:
                    raise admin.AdminError(t("The release confirmation changed: preview it again."))
                conn.execute(
                    "INSERT INTO releases (number, id, proposal, commit_sha, parent, branch, signed_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        number,
                        signed.record.id,
                        found.id,
                        commit,
                        head,
                        branch,
                        signed.model_dump_json(),
                        db.to_iso(now),
                    ),
                )
                conn.execute(
                    "UPDATE proposals SET status = 'accepted', reason = NULL WHERE id = ?",
                    (found.id,),
                )
                store.log(
                    conn,
                    "release_signed",
                    None,
                    dict(number=number, proposal=found.id, commit=commit),
                )
        except Exception as e:
            if forwarded:
                raise admin.AdminError(
                    t(
                        "Commit {commit} is unrecorded: revert it with git before trying again. {error}",
                        commit=commit,
                        error=str(e),
                    )
                ) from e
            if isinstance(e, applying.ApplyError):
                raise admin.AdminError(str(e)) from e
            raise
    return show(conn, number)


def reject(conn: sqlite3.Connection, prefix: str, reason: str) -> str:
    if not 1 <= len(reason) <= 1000:
        raise admin.AdminError(t("A rejection reason must contain 1 to 1000 characters."))
    with db.transaction(conn):
        found = find(conn, prefix)
        active = conn.execute(
            "SELECT 1 FROM proposal_leases WHERE proposal = ? AND ended IS NULL", (found.id,)
        ).fetchone()
        if found.status == "reviewing" or active:
            raise admin.AdminError(t("A node is checking it, try later."))
        if found.status not in ("submitted", "reviewed"):
            raise admin.AdminError(t("Proposal {id} is already decided.", id=found.id))
        conn.execute(
            "UPDATE proposals SET status = 'rejected', reason = ? WHERE id = ?", (reason, found.id)
        )
        store.log(conn, "proposal_rejected", None, dict(proposal=found.id, reason=reason))
    return found.id


def listing(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT number, commit_sha AS 'commit', branch, proposal, created_at FROM releases ORDER BY number"
        ).fetchall()
    ]


def show(conn: sqlite3.Connection, number: int) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM releases WHERE number = ?", (number,)).fetchone()
    if row is None:
        raise admin.AdminError(t("No release {number}.", number=number))
    signed = sign.Signed.model_validate_json(row["signed_json"])
    return dict(signed.record.body) | dict(
        id=row["id"],
        created_at=row["created_at"],
        signer=signed.record.signer,
        signed=signed.model_dump(mode="json"),
    )


def handed_out(conn: sqlite3.Connection, limit: int = 5) -> list[sign.Signed]:
    """Newest releases, oldest first; caller holds a read snapshot."""
    rows = conn.execute(
        "SELECT signed_json FROM releases ORDER BY number DESC LIMIT ?", (limit,)
    ).fetchall()
    return [sign.Signed.model_validate_json(r["signed_json"]) for r in reversed(rows)]
