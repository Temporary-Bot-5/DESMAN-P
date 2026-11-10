"""Work in the registry (ADR 0004, decision 7; ROADMAP G.6): jobs lent to nodes, as in phase 4.

A **job** is a `task` record an operator signed (`protocol.Job`: one self-contained subtask).
It passes two stages, each lent to one node at a time with a **lease** and its token, like
`Store.claim()`:

1. `execute`: a node does the work and hands in a `result` signed by its own key.
2. `recheck`: the result is lent to a node of **another owner** (`store.Node.owned_by`), whose
   reviewer hands in a `verdict`. Accepted → the job is `accepted`. Rejected → it waits for the
   operator (`operator`, reason `disputed`): a disagreement is never settled silently.

A lease not extended in time (`LEASE`) expires: the job is lent again, and whatever the old
holder hands in later is refused by its token (`lease_lost`). A job that lost `MAX_LOST` leases
in one stage, or whose nodes said `MAX_FAILED` times that they could not do it, waits for the
operator too (`leases_lost`, `failed`). The operator settles it (`settle`): accept the result,
lend it again from the start (a new *round*), or drop it.

Only a job a node would accept is lent: its record still passes `sign.Trust` (the operator
assigned, not revoked, in its term). A job past its record's term is `expired`.

G.7b shares the lease limit and expiry sweep with independent proposal checks.

**Cluster jobs** (ADR 0006, T.4b) are jobs too: a cluster operator's `cluster_job` record,
kept by `registry/clusters.py` with its cluster task in `jobs.cluster`, lent by the same
machinery with two differences. `execute` goes only to a member of the cluster's newest
appointment. `recheck` goes to any node that is neither the node that executed it nor one that
operated the cluster (the one rule of ADR 0006, decision 3: another device, no owner rule).
A cluster job that gives up waits for the *cluster's* operator (`operator`), not the global one;
`registry/clusters.py` decides when it goes further.

Every step is journaled in `registry_log` in the transaction that makes it; the lease token is
not (whoever reads the journal could not use it, but it is nobody's business either).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError

from desman.core import db
from desman.net import protocol, sign
from desman.net.sign import Reason, Refused
from desman.registry import checks, store

LEASE: Final = timedelta(minutes=10)
"""How long a lease runs without being extended."""
MAX_LOST: Final = 3
MAX_FAILED: Final = 2
MAX_HELD: Final = 1
"""Leases one node may hold at once."""

LEASE_LOST: Final = "lease_lost"

Status = Literal["execute", "recheck", "accepted", "operator", "dropped", "expired"]
LENT: Final = ("execute", "recheck")
Decision = Literal["accept", "again", "drop"]


class LeaseLost(Exception):
    """The token names no lease this node holds now (expired, ended, someone else's)."""


class SettleError(ValueError):
    """The operator's decision does not fit the job's state."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass(frozen=True)
class JobRow:
    id: str
    signer: str
    signed: sign.Signed
    expires_at: str
    status: Status
    result: str | None
    """The `result` record being rechecked, accepted or disputed."""
    reason: str | None
    """Why it waits for the operator."""
    round: int
    """Lent again from the start by the operator: a new round (counts start over)."""
    created_at: str
    cluster: str | None = None
    """The cluster task a cluster job belongs to; None for an ordinary job."""

    @property
    def job(self) -> protocol.Job:
        if self.signed.record.kind is sign.RecordKind.CLUSTER_JOB:
            return protocol.ClusterJob.model_validate(self.signed.record.body).job
        return protocol.Job.model_validate(self.signed.record.body)

    @property
    def cluster_job(self) -> protocol.ClusterJob | None:
        if self.signed.record.kind is not sign.RecordKind.CLUSTER_JOB:
            return None
        return protocol.ClusterJob.model_validate(self.signed.record.body)

    def as_json(self) -> dict[str, Any]:
        return {
            "job": self.id,
            "status": self.status,
            "reason": self.reason,
            "signer": self.signer,
            "result": self.result,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "task": self.job.model_dump(mode="json"),
            "cluster": self.cluster,
        }


@dataclass(frozen=True)
class Handed:
    """Something a node handed in."""

    id: str
    job: str
    node: str
    kind: str
    outcome: str
    signed: sign.Signed
    received_at: str

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node": self.node,
            "kind": self.kind,
            "outcome": self.outcome,
            "received_at": self.received_at,
            "body": self.signed.record.body,
        }


def _job(row: sqlite3.Row) -> JobRow:
    return JobRow(
        id=row["id"],
        signer=row["signer"],
        signed=sign.Signed.model_validate_json(row["signed_json"]),
        expires_at=row["expires_at"],
        status=row["status"],
        result=row["result"],
        reason=row["reason"],
        round=row["round"],
        created_at=row["created_at"],
        cluster=row["cluster"],
    )


def job(conn: sqlite3.Connection, id: str) -> JobRow | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (id,)).fetchone()
    return None if row is None else _job(row)


def jobs(
    conn: sqlite3.Connection, status: str | None = None, *, cluster: str | None | bool = True
) -> list[JobRow]:
    """Jobs (in `status`, if given): every one (`cluster=True`), only ordinary ones (`None`),
    or only those of one cluster task (its id)."""
    where: list[str] = []
    params: list[str] = []
    if status is not None:
        where.append("status = ?")
        params.append(status)
    if cluster is None:
        where.append("cluster IS NULL")
    elif isinstance(cluster, str):
        where.append("cluster = ?")
        params.append(cluster)
    sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
    rows = conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
    return [_job(r) for r in rows]


def handed_in(conn: sqlite3.Connection, job_id: str) -> list[Handed]:
    rows = conn.execute("SELECT * FROM results WHERE job = ? ORDER BY seq", (job_id,)).fetchall()
    return [
        Handed(
            id=r["id"],
            job=r["job"],
            node=r["node"],
            kind=r["kind"],
            outcome=r["outcome"],
            signed=sign.Signed.model_validate_json(r["signed_json"]),
            received_at=r["received_at"],
        )
        for r in rows
    ]


def find(conn: sqlite3.Connection, prefix: str) -> list[str]:
    """Ids of jobs starting with `prefix` (a person types eight characters, not 32)."""
    if not prefix or any(c not in "0123456789abcdef" for c in prefix):
        return []
    rows = conn.execute(
        "SELECT id FROM jobs WHERE substr(id, 1, ?) = ? ORDER BY id LIMIT 2",
        (len(prefix), prefix),
    ).fetchall()
    return [r["id"] for r in rows]


# ── State changes (each inside the caller's transaction) ────────────────────


def _to_operator(
    conn: sqlite3.Connection, found: JobRow, reason: str, details: dict[str, Any]
) -> None:
    conn.execute("UPDATE jobs SET status = 'operator', reason = ? WHERE id = ?", (reason, found.id))
    store.log(conn, "job_to_operator", None, {"job": found.id, "reason": reason, **details})


def _end_lease(
    conn: sqlite3.Connection,
    token: str,
    how: Literal["handed_in", "expired", "withdrawn"],
    now: str,
) -> None:
    conn.execute(
        "UPDATE leases SET ended = ?, ended_at = ? WHERE token = ? AND ended IS NULL",
        (how, now, token),
    )


def expire(conn: sqlite3.Connection, now: datetime) -> None:
    """Leases past their time end; a job that lost too many goes to the operator; a job past its
    record's term (and lent to nobody) is `expired`. Its own transaction. Run before anything
    that lends or hands in, and before the operator reads or decides (nothing else sweeps)."""
    moment = db.to_iso(now)
    with db.transaction(conn):
        checks.expire(conn, now, MAX_LOST)
        lapsed = conn.execute(
            "SELECT token, job, node, stage FROM leases WHERE ended IS NULL AND expires_at <= ?",
            (moment,),
        ).fetchall()
        for lease in lapsed:
            _end_lease(conn, lease["token"], "expired", moment)
            store.log(
                conn, "lease_expired", lease["node"], {"job": lease["job"], "stage": lease["stage"]}
            )
        for job_id in dict.fromkeys(lease["job"] for lease in lapsed):
            found = job(conn, job_id)
            if found is None or found.status not in LENT:
                continue
            lost = conn.execute(
                "SELECT count(*) FROM leases WHERE job = ? AND stage = ? AND ended = 'expired'"
                " AND round = ?",
                (found.id, found.status, found.round),
            ).fetchone()[0]
            if lost >= MAX_LOST:
                _to_operator(conn, found, "leases_lost", {"stage": found.status, "lost": lost})
        stale = conn.execute(
            "SELECT * FROM jobs WHERE status IN ('execute', 'recheck') AND expires_at <= ?"
            " AND NOT EXISTS (SELECT 1 FROM leases WHERE leases.job = jobs.id"
            " AND ended IS NULL)",
            (moment,),
        ).fetchall()
        for row in stale:
            conn.execute("UPDATE jobs SET status = 'expired' WHERE id = ?", (row["id"],))
            store.log(conn, "job_expired", None, {"job": row["id"]})


def _lendable(believed: sign.Trust, found: JobRow) -> bool:
    """Would a node accept the job's record now? (Each record once per `Trust`.)"""
    try:
        believed.receive(found.signed.model_dump(mode="json"))
    except Refused:
        return False
    return True


def _owners_of_result(conn: sqlite3.Connection, result_id: str | None) -> set[str] | None:
    """Whose node did the work: its owner when the result came and its owner now (the operator
    may have named one since). None if there is no such result."""
    if result_id is None:
        return None
    row = conn.execute("SELECT node, owner FROM results WHERE id = ?", (result_id,)).fetchone()
    if row is None:
        return None
    owners = {row["owner"] or row["node"]}
    done_by = store.node(conn, row["node"])
    if done_by is not None:
        owners.add(done_by.owned_by)
    return owners


def claim(
    conn: sqlite3.Connection,
    caller: store.Node,
    root: Ed25519PublicKey,
    now: datetime,
    *,
    jobs: bool = True,
    proposals: bool = False,
) -> protocol.Work | protocol.ProposalWork | None:
    """Lend `caller` the job waiting longest (rechecks first: they finish work), or nothing.

    Jobs need opt-in and lent models; proposal checks need opt-in and only a checkout. `MAX_HELD` counts both. A
    node never gets a job it was lent before, in any stage or round (it refuses a record it
    has seen as a replay, G.6b), and never rechecks the work of its own owner. Nothing is
    lent while the registry holds no fresh trust list (G.6c): every node would refuse it."""
    expire(conn, now)
    if not (jobs and caller.capabilities) and not proposals:
        return None
    moment = db.to_iso(now)
    with db.transaction(conn):
        held = conn.execute(
            "SELECT count(*) FROM leases WHERE node = ? AND ended IS NULL", (caller.fingerprint,)
        ).fetchone()[0]
        held += conn.execute(
            "SELECT count(*) FROM proposal_leases WHERE node = ? AND ended IS NULL",
            (caller.fingerprint,),
        ).fetchone()[0]
        if held >= MAX_HELD:
            return None
        believed = store.trust(conn, root, now)
        if believed.fresh() is not None:  # every node would refuse what is lent
            return None
        rows = (
            conn.execute(
                "SELECT * FROM jobs WHERE status IN ('execute', 'recheck') AND expires_at > ?"
                " AND NOT EXISTS (SELECT 1 FROM leases WHERE leases.job = jobs.id"
                " AND ended IS NULL)"
                " ORDER BY status = 'recheck' DESC, created_at, id",
                (moment,),
            ).fetchall()
            if jobs
            else []
        )
        if proposals:
            # Rechecks finish jobs already done; then proposals; then fresh executions.
            rechecks = [row for row in rows if row["status"] == "recheck"]
            executions = [row for row in rows if row["status"] == "execute"]
            rows = [*rechecks, None, *executions]
        for row in rows:
            if row is None:
                offered = checks.lend(conn, caller, now)
                if offered is not None:
                    return offered
                continue
            if not caller.capabilities:
                continue
            found = _job(row)
            lent_before = conn.execute(
                "SELECT 1 FROM leases WHERE job = ? AND node = ?", (found.id, caller.fingerprint)
            ).fetchone()
            if lent_before is not None:
                continue
            if found.cluster is not None:
                if not _cluster_lends_to(conn, found, caller):
                    continue
            elif found.status == "recheck":
                owners = _owners_of_result(conn, found.result)
                if owners is None or caller.owned_by in owners:
                    continue
            if not _lendable(believed, found):
                continue
            return _lend(conn, caller, found, now)
    return None


def cluster_operators(conn: sqlite3.Connection, cluster_task: str) -> set[str]:
    """Every node an appointment named as the operator of this cluster task's cluster."""
    rows = conn.execute(
        "SELECT a.operator FROM appointments a JOIN cluster_tasks c ON c.cluster = a.cluster"
        " WHERE c.id = ?",
        (cluster_task,),
    ).fetchall()
    return {r["operator"] for r in rows}


def cluster_members(conn: sqlite3.Connection, cluster_task: str) -> set[str]:
    """The members of the cluster's newest appointment (none before the first)."""
    row = conn.execute(
        "SELECT a.members_json FROM appointments a JOIN cluster_tasks c ON c.cluster = a.cluster"
        " WHERE c.id = ? ORDER BY a.issued_at DESC, a.seq DESC LIMIT 1",
        (cluster_task,),
    ).fetchone()
    return set() if row is None else set(json.loads(row["members_json"]))


def _cluster_lends_to(conn: sqlite3.Connection, found: JobRow, caller: store.Node) -> bool:
    """ADR 0006: a member executes; a node that neither executed it nor operated the cluster
    rechecks; nothing is lent once the cluster task is settled."""
    assert found.cluster is not None
    row = conn.execute("SELECT status FROM cluster_tasks WHERE id = ?", (found.cluster,)).fetchone()
    if row is None or row["status"] not in ("waiting", "running"):
        return False
    if found.status == "execute":
        return caller.fingerprint in cluster_members(conn, found.cluster)
    done_by = conn.execute("SELECT node FROM results WHERE id = ?", (found.result,)).fetchone()
    if done_by is None or done_by["node"] == caller.fingerprint:
        return False
    return caller.fingerprint not in cluster_operators(conn, found.cluster) | {found.signer}


def _lend(
    conn: sqlite3.Connection, caller: store.Node, found: JobRow, now: datetime
) -> protocol.Work:
    token = uuid.uuid4().hex
    until = now + LEASE
    conn.execute(
        "INSERT INTO leases (token, job, node, stage, round, granted_at, expires_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            token,
            found.id,
            caller.fingerprint,
            found.status,
            found.round,
            db.to_iso(now),
            db.to_iso(until),
        ),
    )
    result: sign.Signed | None = None
    result_key: str | None = None
    if found.status == "recheck":
        row = conn.execute(
            "SELECT r.signed_json, n.public_key FROM results r JOIN nodes n ON n.fingerprint = r.node"
            " WHERE r.id = ?",
            (found.result,),
        ).fetchone()
        result = sign.Signed.model_validate_json(row["signed_json"])
        result_key = row["public_key"]
    store.log(conn, "lent", caller.fingerprint, {"job": found.id, "stage": found.status})
    stage: protocol.Stage = "execute" if found.status == "execute" else "recheck"
    return protocol.Work(
        lease=token,
        expires_at=until,
        stage=stage,
        task=found.signed,
        result=result,
        result_key=result_key,
    )


def _held(conn: sqlite3.Connection, caller: store.Node, token: str, now: datetime) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM leases WHERE token = ?", (token,)).fetchone()
    if (
        row is None
        or row["node"] != caller.fingerprint
        or row["ended"] is not None
        or db.parse_iso(row["expires_at"]) <= now
    ):
        raise LeaseLost
    held: sqlite3.Row = row
    return held


def extend(
    conn: sqlite3.Connection, caller: store.Node, token: str, now: datetime
) -> protocol.Lease:
    """The holder keeps its lease for another `LEASE`."""
    expire(conn, now)
    with db.transaction(conn):
        _held(conn, caller, token, now)
        until = now + LEASE
        conn.execute("UPDATE leases SET expires_at = ? WHERE token = ?", (db.to_iso(until), token))
    return protocol.Lease(lease=token, expires_at=until)


def hand_in(
    conn: sqlite3.Connection, caller: store.Node, given: protocol.HandIn, now: datetime
) -> protocol.Received:
    """Keep a signed result or verdict from the lease's holder and move the job on.

    Checked before anything is kept: the lease is the caller's and still runs, the record is
    signed by the caller's key, is the kind its stage hands in, names this job and lease (and,
    for a verdict, the result that was lent), was not handed in before, and (a done result)
    holds for as long as the job does."""
    expire(conn, now)
    moment = db.to_iso(now)
    signed = given.record
    record = signed.record

    def refuse(reason: Reason, detail: str) -> Refused:
        return Refused(reason, detail, id=record.id, signer=caller.fingerprint)

    with db.transaction(conn):
        lease = _held(conn, caller, given.lease, now)
        if record.signer != caller.fingerprint:
            raise refuse(Reason.UNKNOWN_SIGNER, "the record is not signed by the lease's holder")
        sign.check_signature(signed, sign.decode_public(caller.public_key))
        if record.issued_at > now + sign.SKEW:
            raise refuse(Reason.NOT_YET, f"issued at {record.issued_at.isoformat()}")
        if now >= record.expires_at:
            raise refuse(Reason.EXPIRED, f"expired at {record.expires_at.isoformat()}")
        found = job(conn, lease["job"])
        assert found is not None  # a lease always names a job (foreign key)
        body: protocol.Result | protocol.Verdict
        outcome: str
        try:
            if lease["stage"] == "execute":
                if record.kind is not sign.RecordKind.RESULT:
                    raise refuse(Reason.MALFORMED, "this stage hands in a result")
                body = protocol.Result.model_validate(record.body)
                outcome = body.outcome
                if outcome == "done" and record.expires_at < db.parse_iso(found.expires_at):
                    # Its recheck may come as late as the job's own term: a result expiring
                    # first could not be judged by then, and the job would wait for nothing.
                    raise refuse(Reason.MALFORMED, "the result expires before the job")
            else:
                if record.kind is not sign.RecordKind.VERDICT:
                    raise refuse(Reason.MALFORMED, "this stage hands in a verdict")
                body = protocol.Verdict.model_validate(record.body)
                outcome = "accept" if body.accept else "reject"
                if body.result != found.result:
                    raise refuse(Reason.MALFORMED, "not the result that was lent")
        except ValidationError as e:
            raise refuse(Reason.MALFORMED, f"{e.error_count()} problem(s) in the body") from e
        if body.job != found.id or body.lease != given.lease:
            raise refuse(Reason.MALFORMED, "another job or lease")
        try:
            conn.execute(
                "INSERT INTO results (id, job, node, owner, lease, kind, round, outcome,"
                " signed_json, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    found.id,
                    caller.fingerprint,
                    caller.owned_by,
                    given.lease,
                    record.kind.value,
                    lease["round"],
                    outcome,
                    json.dumps(signed.model_dump(mode="json"), sort_keys=True),
                    moment,
                ),
            )
        except sqlite3.IntegrityError as e:
            raise refuse(Reason.REPLAY, "this record was already handed in") from e
        _end_lease(conn, given.lease, "handed_in", moment)
        store.log(
            conn,
            "handed_in",
            caller.fingerprint,
            {"job": found.id, "kind": record.kind.value, "record": record.id, "outcome": outcome},
        )
        status = _after(conn, found, record.id, outcome)
    return protocol.Received(job=found.id, status=status)


def _after(conn: sqlite3.Connection, found: JobRow, record_id: str, outcome: str) -> str:
    if outcome == "done":
        conn.execute(
            "UPDATE jobs SET status = 'recheck', result = ? WHERE id = ?", (record_id, found.id)
        )
        return "recheck"
    if outcome == "failed":
        failed = conn.execute(
            "SELECT count(*) FROM results WHERE job = ? AND outcome = 'failed' AND round = ?",
            (found.id, found.round),
        ).fetchone()[0]
        if failed >= MAX_FAILED:
            _to_operator(conn, found, "failed", {"failed": failed})
            return "operator"
        return "execute"
    if outcome == "accept":
        conn.execute("UPDATE jobs SET status = 'accepted' WHERE id = ?", (found.id,))
        store.log(conn, "job_accepted", None, {"job": found.id, "result": found.result})
        return "accepted"
    _to_operator(conn, found, "disputed", {"verdict": record_id})
    return "operator"


def withdraw(conn: sqlite3.Connection, fingerprint: str, now: datetime) -> int:
    """A node that left gives back what it held (inside the caller's transaction)."""
    rows = conn.execute(
        "SELECT token, job FROM leases WHERE node = ? AND ended IS NULL", (fingerprint,)
    ).fetchall()
    for row in rows:
        _end_lease(conn, row["token"], "withdrawn", db.to_iso(now))
        store.log(conn, "lease_withdrawn", fingerprint, {"job": row["job"]})
    return len(rows) + checks.withdraw(conn, fingerprint, now)


# ── The operator's side ──────────────────────────────────────────────────────


def add(
    conn: sqlite3.Connection,
    signed: sign.Signed,
    root: Ed25519PublicKey,
    now: datetime,
) -> JobRow:
    """Keep a job whose record a node would accept now (else `Refused`)."""
    record = signed.record
    if record.kind is not sign.RecordKind.TASK:
        raise Refused(Reason.MALFORMED, "a job is a task record", id=record.id)
    try:
        protocol.Job.model_validate(record.body)
    except ValidationError as e:
        raise Refused(Reason.MALFORMED, str(e), id=record.id) from e
    moment = db.to_iso(now)
    with db.transaction(conn):
        store.trust(conn, root, now).receive(signed.model_dump(mode="json"))
        conn.execute(
            "INSERT INTO jobs (id, signer, signed_json, expires_at, status, created_at)"
            " VALUES (?, ?, ?, ?, 'execute', ?)",
            (
                record.id,
                record.signer,
                json.dumps(signed.model_dump(mode="json"), sort_keys=True),
                db.to_iso(record.expires_at),
                moment,
            ),
        )
        store.log(
            conn,
            "job_added",
            record.signer,
            {"job": record.id, "expires_at": db.to_iso(record.expires_at)},
        )
        added = job(conn, record.id)
    assert added is not None
    return added


SETTLES_FROM: Final[dict[Decision, tuple[str, ...]]] = {
    "accept": ("operator",),
    "again": ("operator",),
    "drop": ("execute", "recheck", "operator"),
}


def settle(conn: sqlite3.Connection, job_id: str, decision: Decision, now: datetime) -> JobRow:
    """The operator's word on a job: `accept` its result (one must be in hand), lend it `again`
    from the start (a new round: counts of lost leases and failures start over), or `drop` it
    (a lease held on it is withdrawn)."""
    moment = db.to_iso(now)
    with db.transaction(conn):
        found = job(conn, job_id)
        if found is None:
            raise SettleError("missing")
        if found.status not in SETTLES_FROM[decision] or (
            decision == "accept" and found.result is None
        ):
            raise SettleError(found.status)
        if decision == "again" and db.parse_iso(found.expires_at) <= now:
            raise SettleError("expired")  # nodes would refuse its record: sign a new job
        if decision == "accept":
            conn.execute("UPDATE jobs SET status = 'accepted' WHERE id = ?", (found.id,))
        elif decision == "again":
            conn.execute(
                "UPDATE jobs SET status = 'execute', result = NULL, reason = NULL,"
                " round = round + 1 WHERE id = ?",
                (found.id,),
            )
        else:
            for row in conn.execute(
                "SELECT token FROM leases WHERE job = ? AND ended IS NULL", (found.id,)
            ).fetchall():
                _end_lease(conn, row["token"], "withdrawn", moment)
            conn.execute("UPDATE jobs SET status = 'dropped' WHERE id = ?", (found.id,))
        store.log(
            conn,
            "job_settled",
            None,
            {"job": found.id, "decision": decision, "was": found.status, "reason": found.reason},
        )
        settled = job(conn, found.id)
    assert settled is not None
    return settled


def set_owner(conn: sqlite3.Connection, fingerprint: str, owner: str | None) -> bool:
    """Name whose node it is (None: its own). False if no such node registered."""
    with db.transaction(conn):
        if store.node(conn, fingerprint) is None:
            return False
        conn.execute("UPDATE nodes SET owner = ? WHERE fingerprint = ?", (owner, fingerprint))
        store.log(conn, "owner", fingerprint, {"owner": owner})
    return True
