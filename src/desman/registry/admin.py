"""What the owner does on the registry's server (ADR 0004; ROADMAP G.4, G.6): assign an
operator, revoke a key, sign the trust list, back the database up; as the operator, add jobs
and settle those that wait for a person, and say whose node is whose. The bodies of
`desman-registry assign|revoke|trust|backup|job|owner`.

Assignments and revocations are signed by the root key, which lives on this server only
(`root.key` in the registry's directory, readable by its owner only). A key that is not the
root of `net/root.pub` is refused before anything is signed: nodes would refuse what it signs.
Every new record is first checked the way a node will check it (`store.trust`), then kept and
journaled in one transaction. An assignment or a revocation also signs a new trust list (G.6c),
so nodes learn of it at their next poll; the person signs one again before the last expires
(`desman-registry trust sign`, e.g. from a timer), or nodes take no work.
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
from desman.net import protocol, sign
from desman.registry import store, work

ASSIGNMENT_DAYS: Final = 90
JOB_DAYS: Final = 7
TRUST_DAYS: Final = 7
"""How long a trust list lasts unless the person says otherwise."""
MAX_TRUST_DAYS: Final = 30
"""The longest a trust list may last: a hidden revocation is hidden at most this long."""
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
        current = store.latest_trust_list(conn, now)
        left = timedelta(0) if current is None else current.record.expires_at - now
        _sign_list(conn, root, max(timedelta(days=TRUST_DAYS), left), now)  # never shortened
    return subject


def _sign_list(
    conn: sqlite3.Connection, root: sign.NodeKey, lifetime: timedelta, now: datetime
) -> sign.Signed:
    """Inside the caller's transaction: a trust list naming every record in effect, checked
    the way a node will check it, kept and journaled."""
    listed = [
        sign.Listed(
            id=s.record.id,
            until=None if s.record.kind is sign.RecordKind.REVOCATION else s.record.expires_at,
        )
        for s in store.in_effect(conn, now)
    ]
    # Each list later than the last: a node keeps a list only for one issued after its own.
    last = conn.execute("SELECT max(issued_at) FROM trust_lists").fetchone()[0]
    issued = now if last is None else max(now, db.parse_iso(last) + timedelta(seconds=1))
    signed = sign.trust_list(root, listed, issued_at=issued, lifetime=now + lifetime - issued)
    believed = store.trust(conn, root.public, now)
    try:
        believed.receive(signed.model_dump(mode="json"))
    except sign.Refused as e:
        raise AdminError(t("A node would refuse it: {error}", error=str(e))) from e
    why = believed.fresh()
    if why is not None:  # a record kept here that a node would refuse: never listed as needed
        raise AdminError(t("A node would refuse it: {error}", error=why))
    store.add_trust_list(conn, signed)
    store.log(
        conn,
        "trust_list",
        root.fingerprint,
        {
            "record": signed.record.id,
            "records": len(listed),
            "expires_at": db.to_iso(signed.record.expires_at),
        },
    )
    return signed


@dataclass(frozen=True)
class TrustListed:
    id: str
    records: int
    issued_at: datetime
    expires_at: datetime


def sign_trust_list(
    conn: sqlite3.Connection, root: sign.NodeKey, *, days: int, now: datetime
) -> TrustListed:
    """`desman-registry trust sign`: a new trust list lasting `days`."""
    if not 1 <= days <= MAX_TRUST_DAYS:
        raise AdminError(t("A trust list lasts from 1 to {days} days.", days=MAX_TRUST_DAYS))
    with db.transaction(conn):
        signed = _sign_list(conn, root, timedelta(days=days), now)
    return _listed(signed)


def _listed(signed: sign.Signed) -> TrustListed:
    record = signed.record
    body = sign.TrustList.model_validate(record.body)
    return TrustListed(record.id, len(body.records), record.issued_at, record.expires_at)


def trust_list(conn: sqlite3.Connection, now: datetime) -> TrustListed | None:
    """The trust list handed out now: the newest in its term."""
    found = store.latest_trust_list(conn, now)
    return None if found is None else _listed(found)


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


# ── Work (G.6) ───────────────────────────────────────────────────────────────


def load_operator_key(path: Path) -> sign.NodeKey:
    """The operator's private key (the registry's own, `operator.key`, unless told otherwise);
    never created here: an operator signs only once the root assigned it."""
    try:
        return sign.load_node_key(path)
    except (sign.KeyFileError, OSError) as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e


def add_job(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    job: protocol.Job,
    *,
    days: int,
    now: datetime,
    root_pub: Path | None = None,
) -> work.JobRow:
    """Sign the job as a `task` record and keep it, if a node would accept it. Its term ends
    with the operator's assignment at the latest: after that nodes refuse what it signed."""
    root = sign.load_root(sign.ROOT_PUB if root_pub is None else root_pub)
    lifetime = timedelta(days=days)
    assigned = store.trust(conn, root, now).operator(operator.fingerprint)
    if assigned is not None:
        lifetime = min(lifetime, assigned.expires_at - now.replace(microsecond=0))
    if lifetime <= timedelta(0):
        raise AdminError(t("A node would refuse it: the assignment has ended."))
    signed = sign.sign(
        operator,
        sign.RecordKind.TASK,
        job.model_dump(mode="json"),
        issued_at=now,
        lifetime=lifetime,
    )
    try:
        return work.add(conn, signed, root, now)
    except sign.Refused as e:
        raise AdminError(t("A node would refuse it: {error}", error=str(e))) from e


def find_job(conn: sqlite3.Connection, prefix: str) -> str:
    found = work.find(conn, prefix)
    if not found:
        raise AdminError(t("No job {id}.", id=prefix))
    if len(found) > 1:
        raise AdminError(t("More than one job starts with {id}; give more of it.", id=prefix))
    return found[0]


def settle(
    conn: sqlite3.Connection, prefix: str, decision: work.Decision, *, now: datetime
) -> work.JobRow:
    job_id = find_job(conn, prefix)
    work.expire(conn, now)  # decided on where the job is now, not where nodes last left it
    try:
        return work.settle(conn, job_id, decision, now)
    except work.SettleError as e:
        raise AdminError(
            t(
                "Job {id} is {status}: it cannot be settled with {decision}.",
                id=job_id,
                status=e.status,
                decision=decision,
            )
        ) from e


def find_node(conn: sqlite3.Connection, prefix: str) -> str:
    """A registered node by its fingerprint or the start of it (as `nodes` shows it)."""
    if not prefix or any(c not in "0123456789abcdef" for c in prefix):
        raise AdminError(t("No node {key}.", key=prefix))
    found = [n.fingerprint for n in store.nodes(conn) if n.fingerprint.startswith(prefix)]
    if not found:
        raise AdminError(t("No node {key}.", key=prefix))
    if len(found) > 1:
        raise AdminError(t("More than one node starts with {key}; give more of it.", key=prefix))
    return found[0]


def set_owner(conn: sqlite3.Connection, prefix: str, owner: str | None) -> str:
    fingerprint = find_node(conn, prefix)
    work.set_owner(conn, fingerprint, owner)
    return fingerprint
