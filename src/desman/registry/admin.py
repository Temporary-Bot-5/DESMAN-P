"""What the owner does on the registry's server (ADR 0004; ROADMAP G.4, G.6): assign an
operator, revoke a key, sign the trust list, back the database up; as the operator, add jobs
and settle those that wait for a person, and say whose node is whose. The bodies of
`desman-registry assign|revoke|trust|backup|job|owner`, and (ADR 0006, T.4b) `appointer create`
and `cluster add|settle`.

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
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.core import db
from desman.i18n import t
from desman.net import protocol, sign
from desman.registry import clusters, store, work

_BINARY: Final = getattr(os, "O_BINARY", 0)

ASSIGNMENT_DAYS: Final = 90
JOB_DAYS: Final = 7
TRUST_DAYS: Final = 7
"""How long a trust list lasts unless the person says otherwise."""
MAX_TRUST_DAYS: Final = 30
"""The longest a trust list may last: a hidden revocation is hidden at most this long."""
CLUSTER_DAYS: Final = 30
"""The longest a cluster task lasts: its appointments (renewed every quarter hour) stay
within what a node keeps of one cluster (`sign.MAX_APPOINTMENTS`)."""
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


def assignment_key(value: str) -> Ed25519PublicKey:
    """Accept the printed appointer key as well as the historical PEM file path."""
    try:
        path = Path(value)
        return sign.load_public(path) if path.is_file() else sign.decode_public(value)
    except (sign.KeyFileError, OSError, ValueError) as e:
        raise AdminError(t("Use an existing PEM public-key file or a base64 public key.")) from e


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
    if sign.Scope.APPOINT in scopes and len(scopes) > 1:
        raise AdminError(t("The scope appoint is never assigned together with another scope."))
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
        os.close(os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY, 0o600))
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
    found = work.job(conn, job_id)
    if found is not None and found.cluster is not None:
        raise AdminError(
            t(
                "Job {id} belongs to cluster task {task}: its cluster decides it"
                " (desman-registry cluster settle).",
                id=job_id[:8],
                task=found.cluster[:8],
            )
        )
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


# ── Clusters (ADR 0006, T.4b) ────────────────────────────────────────────────


@dataclass(frozen=True)
class Appointer:
    fingerprint: str
    public_key: str
    created: bool


def appointer_create(path: Path) -> Appointer:
    """The registry's appointer key (`appointer.key`, 0600), made if missing. The root assigns
    it `--scope appoint` alone; until then the registry appoints nobody."""
    existed = path.exists()
    try:
        key = sign.node_key(path)
    except (sign.KeyFileError, OSError) as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e
    return Appointer(key.fingerprint, sign.encode_public(key.public), not existed)


def load_appointer(path: Path) -> sign.NodeKey | None:
    """The appointer key if the person made it; None otherwise."""
    if not path.exists():
        return None
    try:
        return sign.load_node_key(path)
    except (sign.KeyFileError, OSError) as e:
        raise AdminError(t("The key cannot be used: {error}", error=str(e))) from e


def add_cluster_task(
    conn: sqlite3.Connection,
    operator: sign.NodeKey,
    task: protocol.ClusterTask,
    *,
    days: int,
    now: datetime,
    root_pub: Path | None = None,
) -> clusters.ClusterTaskRow:
    """Sign a goal as a `cluster_task` record and keep it, if a node would accept it; the
    registry forms its cluster when enough nodes are free."""
    if not 1 <= days <= CLUSTER_DAYS:
        raise AdminError(t("A cluster task lasts from 1 to {days} days.", days=CLUSTER_DAYS))
    root = sign.load_root(sign.ROOT_PUB if root_pub is None else root_pub)
    lifetime = timedelta(days=days)
    assigned = store.trust(conn, root, now).operator(operator.fingerprint)
    if assigned is not None:
        lifetime = min(lifetime, assigned.expires_at - now.replace(microsecond=0))
    if lifetime <= timedelta(0):
        raise AdminError(t("A node would refuse it: the assignment has ended."))
    signed = sign.sign(
        operator,
        sign.RecordKind.CLUSTER_TASK,
        task.model_dump(mode="json"),
        issued_at=now,
        lifetime=lifetime,
    )
    try:
        return clusters.add(conn, signed, root, now)
    except sign.Refused as e:
        raise AdminError(t("A node would refuse it: {error}", error=str(e))) from e


def find_cluster(conn: sqlite3.Connection, prefix: str) -> str:
    found = clusters.find(conn, prefix)
    if not found:
        raise AdminError(t("No cluster task {id}.", id=prefix))
    if len(found) > 1:
        raise AdminError(
            t("More than one cluster task starts with {id}; give more of it.", id=prefix)
        )
    return found[0]


def settle_cluster(
    conn: sqlite3.Connection,
    prefix: str,
    decision: clusters.Decision,
    *,
    note: str,
    now: datetime,
    appointer: sign.NodeKey | None,
    online: timedelta,
    root_pub: Path | None = None,
) -> clusters.ClusterTaskRow:
    task_id = find_cluster(conn, prefix)
    root = sign.load_root(sign.ROOT_PUB if root_pub is None else root_pub)
    clusters.tick(conn, appointer, root, now, online)  # decided on where it is now
    try:
        return clusters.settle(conn, task_id, decision, now, note=note)
    except clusters.SettleError as e:
        raise AdminError(
            t(
                "Cluster task {id} is {status}: it cannot be settled with {decision}.",
                id=task_id[:8],
                status=e.status,
                decision=decision,
            )
        ) from e


_BACKUP_NAME = re.compile(r"registry-\d{8}-\d{6}Z\.db", re.ASCII)


def backup_files(directory: Path) -> list[Path]:
    """Only our timestamped regular backups; never follow a symlink."""
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if _BACKUP_NAME.fullmatch(p.name) and not p.is_symlink() and p.is_file()
    )


@dataclass(frozen=True)
class Daily:
    trust: TrustListed | None
    error: str | None
    backup: Path
    pruned: int


def daily(
    directory: Path, *, days: int, keep: int, now: datetime, root_key: Path | None = None
) -> Daily:
    """Signing failure must not prevent the independent backup."""
    if not 1 <= keep <= 365:
        raise ValueError("keep must be between 1 and 365")
    signed = None
    error = None
    try:
        root = load_root_key(directory / store.ROOT_KEY if root_key is None else root_key)
        conn = store.open_registry(directory / store.DB_NAME)
        try:
            signed = sign_trust_list(conn, root, days=days, now=now)
        finally:
            conn.close()
    except Exception as e:
        error = str(e)
    try:
        return _daily_backup(directory, keep=keep, now=now, signed=signed, error=error)
    except (AdminError, OSError, sqlite3.Error) as e:
        if error is not None:
            raise AdminError(f"{error}; {e}") from e
        raise


def _backup_stamp(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.name, "registry-%Y%m%d-%H%M%SZ.db").replace(tzinfo=UTC)
    except ValueError:
        return None


def _daily_backup(
    directory: Path, *, keep: int, now: datetime, signed: TrustListed | None, error: str | None
) -> Daily:
    folder = directory / "backups"
    folder.mkdir(mode=0o700, exist_ok=True)
    dest = folder / now.astimezone(UTC).strftime("registry-%Y%m%d-%H%M%SZ.db")
    backup(directory / store.DB_NAME, dest)
    removed = [
        p
        for p in backup_files(folder)
        if (stamp := _backup_stamp(p)) is not None and stamp <= now + timedelta(minutes=5)
    ][:-keep]
    for path in removed:
        path.unlink()
    return Daily(signed, error, dest, len(removed))


@dataclass(frozen=True)
class HealthCheck:
    name: str
    level: str
    message: str


def _age_check(name: str, age: timedelta | None, limit: timedelta) -> HealthCheck:
    if age is None or age >= limit:
        return HealthCheck(name, "fail", "missing or stale")
    return HealthCheck(
        name,
        "ok" if age < timedelta(hours=26) else "warn",
        "current" if age < timedelta(hours=26) else "refresh soon",
    )


def health(
    directory: Path,
    *,
    now: datetime,
    url: str = "http://127.0.0.1:8470",
    client: httpx.Client | None = None,
    root_pub: Path = sign.ROOT_PUB,
) -> list[HealthCheck]:
    """Inspect only: no schema migration, key creation, expiry sweep or journal writes."""
    result: list[HealthCheck] = []
    path = directory / store.DB_NAME
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version != len(store.MIGRATIONS):
                message = (
                    "upgrade desman-registry"
                    if version > len(store.MIGRATIONS)
                    else "run the new version once / restart the service"
                )
                result.append(HealthCheck("database", "fail", message))
            else:
                verdict = conn.execute("PRAGMA quick_check").fetchall()
                if len(verdict) != 1 or verdict[0][0] != "ok":
                    result.append(HealthCheck("database", "fail", "quick_check failed"))
            from desman.registry.update import update_health

            if not result:
                result.append(HealthCheck("database", "ok", "ready"))
                listed = trust_list(conn, now)
                result.append(
                    _age_check(
                        "trust_list",
                        None if listed is None else now - listed.issued_at,
                        sign.TRUST_LIST_MAX_AGE,
                    )
                )
                believed = store.trust(conn, sign.load_root(root_pub), now)
                for role, filename in (
                    ("operator", store.OPERATOR_KEY),
                    ("appointer", store.APPOINTER_KEY),
                ):
                    try:
                        key = (
                            load_appointer(directory / filename)
                            if role == "appointer"
                            else load_operator_key(directory / filename)
                        )
                    except (AdminError, OSError) as e:
                        result.append(
                            HealthCheck(
                                "assignments",
                                "fail",
                                "operator missing"
                                if role == "operator" and not (directory / filename).exists()
                                else f"{role}: {e}",
                            )
                        )
                        continue
                    if key is None:
                        result.append(
                            HealthCheck(
                                "assignments", "warn", "appointer missing: clusters will not form"
                            )
                        )
                        continue
                    assigned = believed.operator(key.fingerprint)
                    newest = conn.execute(
                        "SELECT expires_at FROM records WHERE kind='assignment' AND subject=?"
                        " ORDER BY issued_at DESC, seq DESC LIMIT 1",
                        (key.fingerprint,),
                    ).fetchone()
                    if (
                        assigned is None
                        or newest is None
                        or db.parse_iso(newest["expires_at"]) <= now
                        or assigned.expires_at <= now
                        or (role == "appointer" and sign.Scope.APPOINT not in assigned.scopes)
                    ):
                        result.append(
                            HealthCheck("assignments", "fail", f"{role} unassigned or expired")
                        )
                    elif assigned.expires_at - now < timedelta(days=14):
                        days_left = (assigned.expires_at - now).days
                        result.append(
                            HealthCheck(
                                "assignments", "warn", f"{role} expires in {days_left} days"
                            )
                        )
                    else:
                        result.append(HealthCheck("assignments", "ok", f"{role} current"))
                count = len(work.jobs(conn, "operator", cluster=None)) + len(
                    clusters.cluster_tasks(conn, "operator")
                )
                result.append(HealthCheck("inbox", "warn" if count else "ok", str(count)))
                result.append(update_health(conn))
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError) as e:
        result.append(HealthCheck("database", "fail", str(e)))
    try:
        stamps = [
            stamp
            for p in backup_files(directory / "backups")
            if (stamp := _backup_stamp(p)) is not None
        ]
        valid = [stamp for stamp in stamps if stamp <= now + timedelta(minutes=5)]
        stamp = max(valid, default=None)
        result.append(
            _age_check("backup", None if stamp is None else now - stamp, timedelta(hours=72))
        )
        if stamps and not valid:
            result.append(HealthCheck("backup", "warn", "backup dated in the future"))
    except OSError as e:
        result.append(HealthCheck("backup", "fail", str(e)))
    try:
        location = directory
        while not location.exists():
            location = location.parent
        free = shutil.disk_usage(location).free
        result.append(
            HealthCheck(
                "disk",
                "fail" if free < 100 * 1024**2 else "warn" if free < 1024**3 else "ok",
                str(free),
            )
        )
    except OSError as e:
        result.append(HealthCheck("disk", "fail", str(e)))
    if not url:
        result.append(HealthCheck("server", "warn", "not checked"))
    else:
        try:

            def fetch(http: httpx.Client) -> bool:
                response = http.get(url.rstrip("/") + "/healthz", timeout=5)
                return response.status_code == 200 and response.json() == {"status": "ok"}

            if client is None:
                with httpx.Client() as http:
                    ready = fetch(http)
            else:
                ready = fetch(client)
            result.append(
                HealthCheck(
                    "server", "ok" if ready else "fail", "ready" if ready else "unavailable"
                )
            )
        except (httpx.HTTPError, ValueError, OSError) as e:
            result.append(HealthCheck("server", "fail", str(e)))
    return result
