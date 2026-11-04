"""Independent proposal checks (G.7b): leases, signed hand-ins and operator-visible results.

All mutations run in the caller's transaction; work.expire is the single expiry sweep.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from typing import Any, Final

from pydantic import ValidationError

from desman.core import db
from desman.net import protocol, sign
from desman.net.protocol import submission_hash
from desman.registry import store

MAX_UNABLE: Final = 3


def _after(conn: sqlite3.Connection, proposal: str, reason: str | None = None) -> str:
    status = "reviewed" if reason else "submitted"
    conn.execute(
        "UPDATE proposals SET status = ?, reason = ? WHERE id = ?", (status, reason, proposal)
    )
    if reason:
        store.log(conn, "proposal_unchecked", None, {"proposal": proposal, "reason": reason})
    return status


def expire(conn: sqlite3.Connection, now: datetime, max_lost: int) -> None:
    moment = db.to_iso(now)
    rows = conn.execute(
        "SELECT * FROM proposal_leases WHERE ended IS NULL AND expires_at <= ?", (moment,)
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE proposal_leases SET ended = 'expired', ended_at = ? WHERE token = ?",
            (moment, row["token"]),
        )
        lost = conn.execute(
            "SELECT count(*) FROM proposal_leases WHERE proposal = ? AND ended = 'expired'",
            (row["proposal"],),
        ).fetchone()[0]
        store.log(conn, "proposal_lease_expired", row["node"], {"proposal": row["proposal"]})
        _after(conn, row["proposal"], "leases_lost" if lost >= max_lost else None)


def lend(
    conn: sqlite3.Connection, caller: store.Node, now: datetime
) -> protocol.ProposalWork | None:
    from desman.registry.work import LEASE

    for found in store.proposals(conn, "submitted"):
        submitter = store.node(conn, found.node)
        if (
            submitter is None
            or found.node == caller.fingerprint
            or submitter.owned_by == caller.owned_by
        ):
            continue
        if (
            conn.execute(
                "SELECT 1 FROM proposal_leases WHERE proposal = ? AND node = ?",
                (found.id, caller.fingerprint),
            ).fetchone()
            is not None
        ):
            continue
        token = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO proposal_leases (token, proposal, node, granted_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token, found.id, caller.fingerprint, db.to_iso(now), db.to_iso(now + LEASE)),
        )
        conn.execute("UPDATE proposals SET status = 'reviewing' WHERE id = ?", (found.id,))
        store.log(conn, "proposal_lent", caller.fingerprint, {"proposal": found.id})
        return protocol.ProposalWork(
            lease=token,
            expires_at=now + LEASE,
            proposal=found.id,
            submission=found.signed,
            submitter_key=submitter.public_key,
        )
    return None


def withdraw(conn: sqlite3.Connection, node: str, now: datetime) -> int:
    rows = conn.execute(
        "SELECT * FROM proposal_leases WHERE node = ? AND ended IS NULL", (node,)
    ).fetchall()
    for row in rows:
        conn.execute(
            "UPDATE proposal_leases SET ended = 'withdrawn', ended_at = ? WHERE token = ?",
            (db.to_iso(now), row["token"]),
        )
        _after(conn, row["proposal"])
        store.log(
            conn, "proposal_unchecked", node, {"proposal": row["proposal"], "reason": "withdrawn"}
        )
    return len(rows)


def hand_in(
    conn: sqlite3.Connection, caller: store.Node, given: protocol.HandInCheck, now: datetime
) -> protocol.CheckReceived:
    from desman.registry import work

    work.expire(conn, now)
    signed = given.record
    record = signed.record

    def refuse(reason: sign.Reason, detail: str) -> sign.Refused:
        return sign.Refused(reason, detail, id=record.id, signer=caller.fingerprint)

    with db.transaction(conn):
        if conn.execute("SELECT 1 FROM proposal_checks WHERE id = ?", (record.id,)).fetchone():
            raise refuse(sign.Reason.REPLAY, "this check was already handed in")
        lease = conn.execute(
            "SELECT * FROM proposal_leases WHERE token = ?", (given.lease,)
        ).fetchone()
        if (
            lease is None
            or lease["node"] != caller.fingerprint
            or lease["ended"] is not None
            or db.parse_iso(lease["expires_at"]) <= now
        ):
            raise work.LeaseLost
        if record.signer != caller.fingerprint:
            raise refuse(sign.Reason.UNKNOWN_SIGNER, "not the lease holder")
        sign.check_signature(signed, sign.decode_public(caller.public_key))
        if record.issued_at > now + sign.SKEW:
            raise refuse(sign.Reason.NOT_YET, "check issued in the future")
        if now >= record.expires_at:
            raise refuse(sign.Reason.EXPIRED, "check expired")
        if record.kind is not sign.RecordKind.PROPOSAL_CHECK:
            raise refuse(sign.Reason.MALFORMED, "not a proposal check")
        try:
            body = protocol.ProposalCheck.model_validate(record.body)
        except ValidationError as e:
            raise refuse(sign.Reason.MALFORMED, "invalid proposal check") from e
        found = store.proposal(conn, lease["proposal"])
        assert found is not None
        if (
            body.proposal != found.id
            or body.lease != given.lease
            or body.submission != submission_hash(found.signed.submission)
            or body.commit != found.submission.base_commit
        ):
            raise refuse(sign.Reason.MALFORMED, "another proposal, lease or submission")
        if found.submission.mode != "fake" and body.matches_submission is not None:
            raise refuse(sign.Reason.MALFORMED, "a real submission cannot be compared with fake")
        conn.execute(
            "INSERT INTO proposal_checks (id, proposal, node, owner, lease, outcome, signed_json, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                found.id,
                caller.fingerprint,
                caller.owned_by,
                given.lease,
                body.outcome,
                signed.model_dump_json(),
                db.to_iso(now),
            ),
        )
        conn.execute(
            "UPDATE proposal_leases SET ended = 'handed_in', ended_at = ? WHERE token = ?",
            (db.to_iso(now), given.lease),
        )
        if body.outcome == "unable":
            unable = conn.execute(
                "SELECT count(*) FROM proposal_checks WHERE proposal = ? AND outcome = 'unable'",
                (found.id,),
            ).fetchone()[0]
            status = _after(conn, found.id, "unable" if unable >= MAX_UNABLE else None)
            if unable < MAX_UNABLE:
                store.log(
                    conn,
                    "proposal_unchecked",
                    caller.fingerprint,
                    {"proposal": found.id, "reason": "unable"},
                )
        else:
            status = "reviewed"
            conn.execute(
                "UPDATE proposals SET status = 'reviewed', reason = NULL WHERE id = ?", (found.id,)
            )
        store.log(
            conn,
            "proposal_checked",
            caller.fingerprint,
            {
                "proposal": found.id,
                "outcome": body.outcome,
                "verdict": body.verdict,
                "record": record.id,
            },
        )
    return protocol.CheckReceived(proposal=found.id, status=status)


def results(conn: sqlite3.Connection, proposal: str) -> list[dict[str, Any]]:
    return [
        dict(
            node=row["node"],
            owner=row["owner"],
            record=sign.Signed.model_validate_json(row["signed_json"]).model_dump(mode="json"),
            **sign.Signed.model_validate_json(row["signed_json"]).record.body,
        )
        for row in conn.execute(
            "SELECT * FROM proposal_checks WHERE proposal = ? ORDER BY received_at, id", (proposal,)
        ).fetchall()
    ]
