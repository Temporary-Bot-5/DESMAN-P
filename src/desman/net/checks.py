"""Independent fake proposal benchmarks (G.7b), checked before running and durably signed.

No models, config or fetch: the base commit must already be in this checkout. Signed results
survive interrupted hand-ins and are never recomputed for the same registry and proposal.
"""

from __future__ import annotations

import base64
import binascii
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from pydantic import ValidationError

from desman import bench, evaluation
from desman.config import Tuning
from desman.core import db
from desman.net import protocol, sign

RUN_TIMEOUT = 120.0
CHECK_TIMEOUT = 2 * RUN_TIMEOUT
State = Literal["signed", "handed_in", "lost"]
submission_hash = protocol.submission_hash


def can_check() -> bool:
    try:
        evaluation.repository()
    except evaluation.EvaluationError:
        return False
    return True


def verify(
    work: protocol.ProposalWork,
    trust: sign.Trust,
    now: datetime,
    *,
    conn: sqlite3.Connection | None = None,
    registry: str | None = None,
) -> None:
    """Authenticate the submission independently of the registry before any run."""
    signed = work.submission
    try:
        protocol.SignedProposalSubmission.model_validate_json(signed.model_dump_json())
        key = sign.decode_public(work.submitter_key)
        if sign.fingerprint(key) != signed.signer:
            raise sign.Refused(sign.Reason.UNKNOWN_SIGNER, "submission key does not match signer")
        if trust.is_revoked(signed.signer):
            raise sign.Refused(sign.Reason.REVOKED, "submitter revoked")
        key.verify(
            base64.b64decode(signed.signature, validate=True),
            sign.canonical(signed.submission.model_dump(mode="json")),
        )
    except (InvalidSignature, binascii.Error) as e:
        raise sign.Refused(sign.Reason.FORGED, "bad submission signature") from e
    except (ValidationError, sign.KeyFileError) as e:
        raise sign.Refused(sign.Reason.MALFORMED, "invalid submission") from e
    if work.expires_at.tzinfo is None:
        raise sign.Refused(sign.Reason.MALFORMED, "proposal lease has no time zone")
    if work.expires_at <= now:
        raise sign.Refused(sign.Reason.EXPIRED, "proposal lease expired")
    if (
        conn is not None
        and conn.execute(
            "SELECT 1 FROM network_checks WHERE registry = ? AND proposal = ?",
            (registry, work.proposal),
        ).fetchone()
    ):
        raise sign.Refused(sign.Reason.REPLAY, "proposal already checked here")


def run(work: protocol.ProposalWork, repo: Path, now: datetime) -> protocol.ProposalCheck:
    item = work.submission.submission
    common: dict[str, Any] = dict(
        proposal=work.proposal,
        lease=work.lease,
        submission=submission_hash(item),
        commit=item.base_commit,
    )
    try:
        evaluation._git(repo, "cat-file", "-e", item.base_commit + "^{commit}")
        result = evaluation.check(
            repo,
            item.base_commit,
            changes=[c.model_dump() for c in item.details.changes],
            base=item.details.base,
            diff=item.diff,
            mode="fake",
            base_tuning=Tuning(),
            candidate_tuning=Tuning(),
            timeout=RUN_TIMEOUT,
        )
    except evaluation.EvaluationError:
        return protocol.ProposalCheck(
            **common,
            outcome="unable",
            reason="The base commit or its git worktrees are unavailable.",
        )
    if result.outcome == "refused":
        return protocol.ProposalCheck(
            **common,
            outcome="refused",
            reason="The submission does not change only the allowed target texts at the base commit.",
        )
    if result.outcome == "failed":
        return protocol.ProposalCheck(
            **common,
            outcome="failed",
            reason="The fake benchmark did not complete successfully.",
            base=None if result.base is None else protocol.benchmark_snapshot(result.base),
        )
    assert result.base is not None and result.candidate is not None
    matches = None
    if item.mode == "fake":
        try:
            matches = bench.metrics(result.base) == bench.metrics(
                bench.Result.model_validate(item.base.model_dump())
            ) and bench.metrics(result.candidate) == bench.metrics(
                bench.Result.model_validate(item.candidate.model_dump())
            )
        except (ValueError, TypeError, AttributeError):
            # Snapshot statistics are untrusted JSON, not instructions or executable code.
            matches = False

    return protocol.ProposalCheck(
        **common,
        outcome="checked",
        verdict=result.verdict,
        base=protocol.benchmark_snapshot(result.base),
        candidate=protocol.benchmark_snapshot(result.candidate),
        matches_submission=matches,
    )


@dataclass(frozen=True)
class Pending:
    lease: str
    registry: str
    proposal: str
    submission_sha256: str
    record: sign.Signed


def keep(
    conn: sqlite3.Connection,
    registry: str,
    work: protocol.ProposalWork,
    record: sign.Signed,
    now: datetime,
) -> None:
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO network_checks (lease, registry, proposal, submission_sha256, record_json, state, created_at) VALUES (?, ?, ?, ?, ?, 'signed', ?)",
            (
                work.lease,
                registry,
                work.proposal,
                submission_hash(work.submission.submission),
                record.model_dump_json(),
                db.to_iso(now),
            ),
        )


def pending(conn: sqlite3.Connection) -> list[Pending]:
    with db.transaction(conn):
        rows = conn.execute(
            "SELECT * FROM network_checks WHERE state = 'signed' ORDER BY created_at, lease"
        ).fetchall()
        return [
            Pending(
                row["lease"],
                row["registry"],
                row["proposal"],
                row["submission_sha256"],
                sign.Signed.model_validate_json(row["record_json"]),
            )
            for row in rows
        ]


def close(conn: sqlite3.Connection, lease: str, state: State) -> None:
    with db.transaction(conn):
        conn.execute(
            "UPDATE network_checks SET state = ? WHERE lease = ? AND state = 'signed'",
            (state, lease),
        )
