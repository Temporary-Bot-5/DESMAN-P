"""Operator decisions about correspondence; signed locally, never sent as trust records."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, ValidationError

from desman.core import db
from desman.i18n import t
from desman.net import protocol as p
from desman.net import sign
from desman.registry import admin, store


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModerationRow(Row):
    id: str
    node: str
    action: Literal["suspend", "lift"]
    lifts: str | None
    signer: str
    issued_at: str
    expires_at: str
    lifted: bool


class ComplaintRow(Row):
    id: str
    from_node: str
    about_node: str
    from_name: str | None
    about_name: str | None
    message_id: int | None
    created_at: str
    status: str
    count: int


class ComplaintDecision(Row):
    complaint: str
    decision: str
    record: str | None
    note: str | None
    decided_at: str


class ComplaintDetail(ComplaintRow):
    text: str
    message_text: str | None
    decisions: list[ComplaintDecision]


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
        raise admin.AdminError(t("No complaint {id}.", id=prefix))
    if len(rows) > 1:
        raise admin.AdminError(
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
        raise admin.AdminError(t("Complaint {id} is already settled.", id=id))
    conn.execute(
        "INSERT INTO complaint_decisions VALUES (?,?,?,?,?)",
        (id, decision, record, note, db.to_iso(now)),
    )


def dismiss(conn: sqlite3.Connection, id: str, note: str | None, now: datetime) -> None:
    with db.transaction(conn):
        id = find_complaint(conn, id)
        _settle(conn, id, "dismissed", None, note, now)
        store.log(conn, "complaint_dismissed", None, {"complaint": id, "note": note})


def _node(conn: sqlite3.Connection, ref: str) -> str:
    row = conn.execute("SELECT node FROM messenger_names WHERE name=?", (ref,)).fetchone()
    return str(row[0]) if row is not None else admin.find_node(conn, ref)


def _valid(
    conn: sqlite3.Connection, row: sqlite3.Row, root: Ed25519PublicKey, now: datetime
) -> p.Suspension | None:
    try:
        signed = sign.Signed.model_validate_json(row["signed_json"])
        store.trust(conn, root, now).receive(signed.model_dump(mode="json"))
        body = p.Moderation.model_validate(signed.record.body)
    except (sign.Refused, ValidationError):
        return None
    if body.action != "suspend" or body.node != row["node"]:
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
        "AND NOT EXISTS (SELECT 1 FROM moderations l WHERE l.action='lift' AND l.lifts=s.id) "
        "ORDER BY s.issued_at DESC,s.rowid DESC",
        (node, db.to_iso(now)),
    ).fetchall()
    if not rows:
        return None
    root = sign.load_root(sign.ROOT_PUB) if root is None else root
    for row in rows:
        found = _valid(conn, row, root, now)
        if found is not None:
            return found
    return None


def moderations(conn: sqlite3.Connection) -> list[ModerationRow]:
    return [
        ModerationRow.model_validate(dict(row))
        for row in conn.execute(
            "SELECT s.id,s.node,s.action,s.lifts,s.signer,s.issued_at,s.expires_at,"
            "EXISTS(SELECT 1 FROM moderations l WHERE l.action='lift' AND l.lifts=s.id) AS lifted "
            "FROM moderations s ORDER BY s.issued_at DESC,s.rowid DESC"
        )
    ]


def _keep(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    fields: dict[str, Any],
    days: int,
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
    lifetime = timedelta(days=days)
    if assigned is not None:
        lifetime = min(lifetime, assigned.expires_at - now.replace(microsecond=0))
    if lifetime <= timedelta(0):
        raise admin.AdminError(t("A node would refuse it: the assignment has ended."))
    signed = sign.sign(
        operator,
        sign.RecordKind.MODERATION,
        body.model_dump(mode="json"),
        issued_at=now,
        lifetime=lifetime,
    )
    try:
        believed.receive(signed.model_dump(mode="json"))
    except sign.Refused as e:
        raise admin.AdminError(t("A node would refuse it: {error}", error=str(e))) from e
    record = signed.record
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
    return next(row for row in moderations(conn) if row.id == record.id)


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
        if len(set(ids)) != len(ids):
            raise admin.AdminError(t("Cite each complaint only once."))
        for id in ids:
            row = conn.execute(
                "SELECT about_node,status FROM complaints WHERE id=?", (id,)
            ).fetchone()
            if row["about_node"] != node or row["status"] != "open":
                raise admin.AdminError(t("Complaint {id} must be open and about this node.", id=id))
        return _keep(
            conn,
            operator,
            {"node": node, "action": "suspend", "reason": reason, "complaints": ids},
            days,
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
            found = _valid(conn, row, root, now)
            if conn.execute(
                "SELECT 1 FROM moderations WHERE action='lift' AND lifts=?", (row["id"],)
            ).fetchone():
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
            1,
            now,
            root_pub,
        )
