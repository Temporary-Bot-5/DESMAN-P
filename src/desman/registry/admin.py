"""What the owner does on the registry's server (ADR 0004; ROADMAP G.4): assign an operator,
revoke a key, back the database up. The bodies of `desman-registry assign|revoke|backup`.

Assignments and revocations are signed by the root key, which lives on this server only
(`root.key` in the registry's directory, readable by its owner only). A key that is not the
root of `net/root.pub` is refused before anything is signed: nodes would refuse what it signs.
Every new record is first checked the way a node will check it (`store.trust`), then kept and
journaled in one transaction.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.core import db
from desman.i18n import t
from desman.net import sign
from desman.registry import store

ASSIGNMENT_DAYS: Final = 90
REVOCATION_LIFETIME: Final = timedelta(days=3650)
"""A revocation is honoured past its term anyway; the term only has to be valid."""


class AdminError(ValueError):
    """Something the owner asked for that cannot be done; the message (translated) says why."""


def load_root_key(path: Path, root_pub: Path | None = None) -> sign.NodeKey:
    root_pub = sign.ROOT_PUB if root_pub is None else root_pub
    try:
        key = sign.load_node_key(path)
    except sign.KeyFileError as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e
    if key.fingerprint != sign.fingerprint(sign.load_root(root_pub)):
        raise AdminError(
            t("{path} is not the root key of {name}.", path=str(path), name=root_pub.name)
        )
    return key


def operator_key(path: Path) -> Ed25519PublicKey:
    """The registry's own operator key (created on first use), or a public key from a file."""
    try:
        return sign.node_key(path).public
    except sign.KeyFileError as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e


def public_key(path: Path) -> Ed25519PublicKey:
    try:
        return sign.load_public(path)
    except sign.KeyFileError as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e


def _keep(conn: sqlite3.Connection, root: sign.NodeKey, signed: sign.Signed, now: datetime) -> str:
    with db.transaction(conn):
        believed = store.trust(conn, root.public, now)
        try:
            believed.receive(signed.model_dump(mode="json"))
        except sign.Refused as e:
            raise AdminError(t("A node would refuse it: {error}", error=str(e))) from e
        subject = store.add_record(conn, signed)
        record = signed.record
        store.log(
            conn,
            record.kind.value,
            subject,
            {
                "record": record.id,
                "expires_at": db.to_iso(record.expires_at),
                **{k: v for k, v in record.body.items() if k != "key"},
            },
        )
    return subject


@dataclass(frozen=True)
class Assigned:
    operator: str
    scopes: list[str]
    expires_at: datetime


def assign(
    conn: sqlite3.Connection,
    root: sign.NodeKey,
    operator: Ed25519PublicKey,
    scopes: set[sign.Scope],
    *,
    days: int,
    now: datetime,
) -> Assigned:
    signed = sign.assign(root, operator, scopes, issued_at=now, lifetime=timedelta(days=days))
    subject = _keep(conn, root, signed, now)
    return Assigned(subject, sorted(scopes), signed.record.expires_at)


def revoke(
    conn: sqlite3.Connection, root: sign.NodeKey, key: str, reason: str, *, now: datetime
) -> sign.Signed:
    signed = sign.revoke(root, key, reason, issued_at=now, lifetime=REVOCATION_LIFETIME)
    _keep(conn, root, signed, now)
    return signed


def backup(source: Path, dest: Path) -> None:
    """A consistent copy of the database, taken while the server runs (SQLite backup API);
    readable by its owner only, never over an existing file."""
    if not source.exists():
        raise AdminError(
            t("No database at {path}: the registry has not run yet.", path=str(source))
        )
    try:
        os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError as e:
        raise AdminError(t("{path} exists already; choose another name.", path=str(dest))) from e
    except OSError as e:
        raise AdminError(
            t("{path} cannot be written: {error}", path=str(dest), error=str(e))
        ) from e
    try:
        with store.connection(source) as conn:
            target = sqlite3.connect(dest)
            try:
                conn.backup(target)
                verdict = target.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                target.close()
        if verdict != "ok":
            raise AdminError(t("The copy is damaged ({verdict}); try again.", verdict=verdict))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
