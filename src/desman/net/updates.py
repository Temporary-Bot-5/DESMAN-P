"""Root-signed releases received by the node; receiving never installs anything (G.8)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from desman.core import db, netlog
from desman.net import sign


def keep(conn: sqlite3.Connection, trust: sign.Trust, signed: sign.Signed) -> bool:
    if signed.record.kind is not sign.RecordKind.RELEASE:
        raise sign.Refused(sign.Reason.NOT_ALLOWED, "not a release")
    record = trust.receive(signed.model_dump(mode="json"))
    body = sign.Release.model_validate(record.body)
    conflict = False
    with db.transaction(conn):
        if conn.execute("SELECT 1 FROM releases WHERE id = ?", (record.id,)).fetchone():
            return False
        if conn.execute("SELECT 1 FROM releases WHERE number = ?", (body.number,)).fetchone():
            netlog.write(conn, "release_conflict", {"number": body.number, "id": record.id})
            conflict = True
        else:
            conn.execute(
                "INSERT INTO releases (number, id, commit_sha, parent, branch, signed_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    body.number,
                    record.id,
                    body.commit,
                    body.parent,
                    body.branch,
                    signed.model_dump_json(),
                ),
            )
            netlog.write(
                conn,
                "release_received",
                {
                    "number": body.number,
                    "id": record.id,
                    "commit": body.commit,
                },
            )
    if conflict:
        raise sign.Refused(sign.Reason.CONFLICT, "another release has this number", id=record.id)
    return True


def stored(conn: sqlite3.Connection) -> list[sign.Signed]:
    """Stored envelopes, newest first. Installation must use `verified`."""
    return [
        sign.Signed.model_validate_json(r[0])
        for r in conn.execute("SELECT signed_json FROM releases ORDER BY number DESC")
    ]


def latest(conn: sqlite3.Connection) -> sign.Signed | None:
    found = stored(conn)
    return found[0] if found else None


def verified(
    conn: sqlite3.Connection, number: int | None, *, root: Path | None = None
) -> sign.Release | None:
    """Recheck the signed envelope with a fresh root trust, never trusting indexed columns."""
    row = conn.execute(
        "SELECT number, signed_json FROM releases "
        + ("ORDER BY number DESC LIMIT 1" if number is None else "WHERE number = ?"),
        () if number is None else (number,),
    ).fetchone()
    if row is None:
        return None
    signed = sign.Signed.model_validate_json(row["signed_json"])
    trust = sign.Trust(sign.load_root(sign.ROOT_PUB if root is None else root))
    if signed.record.kind is not sign.RecordKind.RELEASE:
        raise sign.Refused(sign.Reason.NOT_ALLOWED, "not a release")
    record = trust.receive(signed.model_dump(mode="json"))
    release = sign.Release.model_validate(record.body)
    if release.number != row["number"]:
        raise sign.Refused(sign.Reason.CONFLICT, "release number differs from stored number")
    return release
