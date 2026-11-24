"""Operator decisions about correspondence; signed locally, never sent as trust records."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError

from desman.core import db
from desman.i18n import t
from desman.net import operator_records, sign
from desman.net import protocol as p
from desman.net.protocol import ComplaintDetail, ComplaintRow
from desman.net.protocol import ModerationRow as ModerationRow
from desman.registry import admin, store

_COMPLAINTS = """
    SELECT c.id,c.from_node,c.about_node,f.name AS from_name,a.name AS about_name,
           c.message_id,c.created_at,c.status,
           (SELECT count(*) FROM complaints total WHERE total.about_node=c.about_node) AS count
    FROM complaints c
    LEFT JOIN messenger_names f ON f.node=c.from_node
    LEFT JOIN messenger_names a ON a.node=c.about_node
"""


def complaints(conn: sqlite3.Connection, *, all: bool = False) -> list[ComplaintRow]:
    return [
        ComplaintRow.model_validate(dict(row))
        for row in conn.execute(
            _COMPLAINTS + " WHERE (? OR c.status='open') ORDER BY c.created_at DESC,c.id DESC",
            (all,),
        )
    ]


def find_complaint(conn: sqlite3.Connection, prefix: str) -> str:
    # substr avoids treating a prefix containing SQL wildcard characters specially.
    rows = conn.execute(
        "SELECT id FROM complaints WHERE substr(id,1,?)=?", (len(prefix), prefix)
    ).fetchall()
    if not rows:
        raise admin.Missing(t("No complaint {id}.", id=prefix))
    if len(rows) > 1:
        raise admin.Ambiguous(
            t("More than one complaint starts with {id}; give more of it.", id=prefix)
        )
    return str(rows[0][0])


def complaint(conn: sqlite3.Connection, id: str) -> ComplaintDetail:
    id = find_complaint(conn, id)
    row = conn.execute(_COMPLAINTS + " WHERE c.id=?", (id,)).fetchone()
    texts = conn.execute("SELECT text,message_text FROM complaints WHERE id=?", (id,)).fetchone()
    decisions = [
        dict(r) for r in conn.execute("SELECT * FROM complaint_decisions WHERE complaint=?", (id,))
    ]
    return ComplaintDetail.model_validate({**dict(row), **dict(texts), "decisions": decisions})


def _settle(
    conn: sqlite3.Connection,
    id: str,
    decision: str,
    record: str | None,
    note: str | None,
    now: datetime,
) -> None:
    changed = conn.execute(
        "UPDATE complaints SET status='settled' WHERE id=? AND status='open'", (id,)
    )
    if changed.rowcount != 1:
        raise admin.AlreadySettled(t("Complaint {id} is already settled.", id=id))
    conn.execute(
        "INSERT INTO complaint_decisions VALUES (?,?,?,?,?)",
        (id, decision, record, note, db.to_iso(now)),
    )


def dismiss(conn: sqlite3.Connection, id: str, note: str | None, now: datetime) -> str:
    with store.write_transaction(conn):
        id = find_complaint(conn, id)
        _settle(conn, id, "dismissed", None, note, now)
        store.log(conn, "complaint_dismissed", None, {"complaint": id, "note": note})
        return id


def _node(conn: sqlite3.Connection, ref: str) -> str:
    row = conn.execute("SELECT node FROM messenger_names WHERE name=?", (ref,)).fetchone()
    return str(row[0]) if row is not None else admin.find_node(conn, ref)


def _valid(row: sqlite3.Row, believed: sign.Trust, now: datetime) -> p.Suspension | None:
    """The record as a node would take it now (as `Trust.receive` checks it): signed by the
    root, or by an operator with scope `moderation`, not revoked, issued within its assignment
    and not ahead of the clock, both still in term. Uses `Trust`'s public view only (receiving
    a moderation record would mark its id as seen)."""
    try:
        signed = sign.Signed.model_validate_json(row["signed_json"])
        record = signed.record
        if record.kind is not sign.RecordKind.MODERATION or believed.is_revoked(record.signer):
            return None
        if record.issued_at > now + sign.SKEW or now >= record.expires_at:
            return None
        if record.signer == believed.root_fingerprint:
            key = believed.root
        else:
            found = believed.operator(record.signer)
            if found is None or sign.Scope.MODERATION not in found.scopes:
                return None
            if not found.issued_at <= record.issued_at < found.expires_at:
                return None
            if now >= found.expires_at:
                return None
            key = found.key
        sign.check_signature(signed, key)
        body = p.Moderation.model_validate(record.body)
    except (sign.Refused, ValidationError):
        return None
    if body.action != row["action"] or body.node != row["node"] or body.lifts != row["lifts"]:
        return None
    return p.Suspension(
        until=signed.record.expires_at,
        reason=body.reason,
        record=signed.record.id,
        operator=signed.record.signer,
    )


def suspension(
    conn: sqlite3.Connection, node: str, now: datetime, *, root: Ed25519PublicKey | None = None
) -> p.Suspension | None:
    rows = conn.execute(
        "SELECT s.* FROM moderations s WHERE s.node=? AND s.action='suspend' AND s.expires_at>? "
        "ORDER BY s.issued_at DESC,s.rowid DESC",
        (node, db.to_iso(now)),
    ).fetchall()
    if not rows:
        return None
    root = sign.load_root(sign.ROOT_PUB) if root is None else root
    believed = store.trust(conn, root, now)
    for row in rows:
        found = _valid(row, believed, now)
        if found is not None and not _lifted(conn, row["id"], believed, now):
            return found
    return None


def _lifted(conn: sqlite3.Connection, id: str, believed: sign.Trust, now: datetime) -> bool:
    return any(
        _valid(row, believed, now) is not None
        for row in conn.execute("SELECT * FROM moderations WHERE action='lift' AND lifts=?", (id,))
    )


def moderations(
    conn: sqlite3.Connection, *, now: datetime | None = None, root: Ed25519PublicKey | None = None
) -> list[ModerationRow]:
    now = datetime.now(UTC) if now is None else now
    root = sign.load_root(sign.ROOT_PUB) if root is None else root
    believed = store.trust(conn, root, now)
    lifted = {
        row["lifts"]
        for row in conn.execute("SELECT * FROM moderations WHERE action='lift'")
        if _valid(row, believed, now) is not None
    }
    return [
        ModerationRow.model_validate(
            {**dict(row), "lifted": row["action"] == "suspend" and row["id"] in lifted}
        )
        for row in conn.execute(
            "SELECT id,node,action,lifts,signer,issued_at,expires_at "
            "FROM moderations ORDER BY issued_at DESC,rowid DESC"
        )
    ]


def _keep(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    fields: dict[str, Any],
    lifetime: timedelta,
    now: datetime,
    root_pub: Path | None,
) -> ModerationRow:
    try:
        body = p.Moderation.model_validate(fields)
    except ValidationError:
        raise admin.AdminError(t("Invalid moderation request; nothing was signed.")) from None
    root = sign.load_root(sign.ROOT_PUB if root_pub is None else root_pub)
    believed = store.trust(conn, root, now)
    assigned = believed.operator(operator.fingerprint)
    try:
        signed = operator_records.moderation(
            operator,
            body,
            now=now,
            lifetime=lifetime,
            until=None if assigned is None else assigned.expires_at,
        )
        return add(conn, signed, root, now)
    except operator_records.AssignmentEnded:
        raise admin.AdminError(t("A node would refuse it: the assignment has ended.")) from None
    except sign.Refused as e:
        raise admin.AdminError(t("A node would refuse it: {error}", error=str(e))) from e


def add(
    conn: sqlite3.Connection,
    signed: sign.Signed,
    root: Ed25519PublicKey,
    now: datetime,
) -> ModerationRow:
    """Verify and keep a signed suspension or lift, with its complaint decisions."""
    with store.write_transaction(conn):
        return _add(conn, signed, root, now)


def _add(
    conn: sqlite3.Connection,
    signed: sign.Signed,
    root: Ed25519PublicKey,
    now: datetime,
) -> ModerationRow:
    record = signed.record
    if record.kind is not sign.RecordKind.MODERATION:
        raise sign.Refused(sign.Reason.MALFORMED, "expected a moderation record")
    believed = store.trust(conn, root, now)
    believed.receive(signed.model_dump(mode="json"))
    if conn.execute("SELECT 1 FROM moderations WHERE id=?", (record.id,)).fetchone():
        raise sign.Refused(sign.Reason.REPLAY, "record already stored", id=record.id)
    try:
        body = p.Moderation.model_validate(record.body)
    except ValidationError as e:
        raise sign.Refused(sign.Reason.MALFORMED, "invalid moderation body") from e
    if store.node(conn, body.node) is None:
        raise admin.Missing(t("No node {key}.", key=body.node))
    if len(set(body.complaints)) != len(body.complaints):
        raise admin.AdminError(t("Cite each complaint only once."))
    for id in body.complaints:
        row = conn.execute("SELECT about_node,status FROM complaints WHERE id=?", (id,)).fetchone()
        if row is None:
            raise admin.Missing(t("No complaint {id}.", id=id))
        if body.action != "suspend" or row["about_node"] != body.node or row["status"] != "open":
            raise admin.AdminError(t("Complaint {id} must be open and about this node.", id=id))
    if body.lifts is not None:
        suspension = conn.execute(
            "SELECT * FROM moderations WHERE id=? AND node=? AND action='suspend'",
            (body.lifts, body.node),
        ).fetchone()
        if suspension is None or _valid(suspension, believed, now) is None:
            raise admin.Missing(t("No suspension is in effect for {id}.", id=body.lifts))
        if _lifted(conn, body.lifts, believed, now):
            raise admin.AlreadySettled(t("No suspension is in effect for {id}.", id=body.lifts))
    conn.execute(
        "INSERT INTO moderations VALUES (?,?,?,?,?,?,?,?)",
        (
            record.id,
            body.node,
            body.action,
            body.lifts,
            record.signer,
            db.to_iso(record.issued_at),
            db.to_iso(record.expires_at),
            signed.model_dump_json(),
        ),
    )
    for id in body.complaints:
        _settle(conn, id, "suspended", record.id, None, now)
    store.log(
        conn,
        "moderation",
        body.node,
        {
            "record": record.id,
            "action": body.action,
            "expires_at": db.to_iso(record.expires_at),
            "complaints": body.complaints,
        },
    )
    return ModerationRow(
        id=record.id,
        node=body.node,
        action=body.action,
        lifts=body.lifts,
        signer=record.signer,
        issued_at=db.to_iso(record.issued_at),
        expires_at=db.to_iso(record.expires_at),
        lifted=False,
    )


def suspend(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    node_prefix_or_name: str,
    reason: str,
    *,
    days: int = p.MODERATION_DAYS,
    complaints: list[str],
    now: datetime,
    root_pub: Path | None = None,
) -> ModerationRow:
    if not 1 <= days <= p.MAX_MODERATION_DAYS:
        raise admin.AdminError(
            t("A suspension lasts from 1 to {days} days.", days=p.MAX_MODERATION_DAYS)
        )
    with db.transaction(conn):
        node = _node(conn, node_prefix_or_name)
        ids = [find_complaint(conn, id) for id in complaints]
        return _keep(
            conn,
            operator,
            {"node": node, "action": "suspend", "reason": reason, "complaints": ids},
            timedelta(days=days),
            now,
            root_pub,
        )


def lift(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    ref: str,
    reason: str,
    *,
    now: datetime,
    root_pub: Path | None = None,
) -> ModerationRow:
    root = sign.load_root(sign.ROOT_PUB if root_pub is None else root_pub)
    with db.transaction(conn):
        rows = conn.execute(
            "SELECT * FROM moderations WHERE substr(id,1,?)=? AND action='suspend'", (len(ref), ref)
        ).fetchall()
        if len(rows) > 1:
            raise admin.AdminError(
                t("More than one moderation starts with {id}; give more of it.", id=ref)
            )
        if rows:
            row = rows[0]
            believed = store.trust(conn, root, now)
            found = _valid(row, believed, now)
            if _lifted(conn, row["id"], believed, now):
                found = None
        else:
            found = suspension(conn, _node(conn, ref), now, root=root)
        if found is None:
            raise admin.AdminError(t("No suspension is in effect for {id}.", id=ref))
        return _keep(
            conn,
            operator,
            {
                "node": rows[0]["node"] if rows else _node(conn, ref),
                "action": "lift",
                "reason": reason,
                "lifts": found.record,
            },
            max(timedelta(seconds=1), found.until - now),
            now,
            root_pub,
        )
