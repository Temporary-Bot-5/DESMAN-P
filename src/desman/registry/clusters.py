"""Clusters in the registry (ADR 0006; ROADMAP T.4b): DESMAN appoints a node to lead a cluster.

A **cluster task** is a goal the global operator signed (`cluster_task`, scope `tasks`;
`desman-registry cluster add`). It waits until there are enough free nodes: one that may lead
(`Claim.lead`) and two others, one to execute and one to recheck (`NEEDED`). Then the registry
signs an **appointment** by itself with its appointer key (`appointer.key`, assigned by the root
with the scope `appoint` alone): the leader is the free node with the most results accepted on
recheck (then the oldest), the members the next ones in the same order, at most
`sign.MAX_MEMBERS`. No thresholds (the owner, 2026-11-10): what keeps a malicious node from going
unnoticed is that a result is never rechecked on the device that made it (`work.py`).

**Free** = taking part, not revoked, heartbeat within the window (`online`), its last claim took
jobs and it lends a module, and it is neither the operator nor a member of a running cluster.

The operator polls its cluster (`poll`, `GET /v1/clusters/mine`); each poll is its heartbeat,
and past half its term it gets a fresh appointment (gone members replaced by free nodes). It
hands in its plan, its jobs and the final result (`hand_in`, `POST /v1/clusters/records`), each
checked the way a node will (`store.trust`) and against what the cluster holds: a plan version
follows the one before and keeps every accepted node as it is; a job is a node of the current
plan whose dependencies are accepted and that is not lent yet; the result names the accepted
result of the last node. Only the operator in force hands in, and nothing dated before its
appointments began: the registry is the guard against a record dated back into an old term.

The registry gives up nothing silently (`tick`): an operator not seen for one term (or gone,
revoked, no longer leading) is replaced by another free node that may lead and never led this
task; after `MAX_OPERATORS_LOST` the cluster task goes to the global operator's inbox, as it does
when a job gives up on its last plan version or on a node that runs code, and when its result
comes. The person settles it (`settle`): accept, send it back (`again`, with a note), or drop it.

Every step is in `registry_log` (counts and fingerprints, never content), in the transaction
that makes it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final, Literal

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import ValidationError

from desman.core import db
from desman.net import protocol, sign
from desman.net.sign import Reason, Refused
from desman.registry import store, work

TERM: Final = sign.APPOINTMENT_TERM
ONLINE: Final = timedelta(seconds=2 * protocol.HEARTBEAT_SECONDS)
"""How recent a free node's heartbeat is, by default (the server: two of its intervals)."""
NEEDED: Final = 3
"""Free nodes before a cluster is formed: one leads, one executes, another rechecks."""
MAX_OPERATORS_LOST: Final = 3

Status = Literal["waiting", "running", "operator", "accepted", "dropped", "expired"]
Decision = Literal["accept", "again", "drop"]
_Refuse = Callable[[Reason, str], Refused]
Waiting = Literal["no_appointer", "appointer_not_assigned", "trust_list", "no_leader", "free"]


class SettleError(ValueError):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass(frozen=True)
class ClusterTaskRow:
    id: str
    cluster: str
    signer: str
    signed: sign.Signed
    expires_at: str
    status: Status
    reason: str | None
    note: str | None
    replan: bool
    operator_seen: str | None
    lost: int
    created_at: str

    @property
    def task(self) -> protocol.ClusterTask:
        return protocol.ClusterTask.model_validate(self.signed.record.body)

    def as_json(self) -> dict[str, Any]:
        return {
            "task": self.id,
            "cluster": self.cluster,
            "status": self.status,
            "reason": self.reason,
            "note": self.note,
            "signer": self.signer,
            "lost": self.lost,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "goal": self.task.model_dump(mode="json"),
        }


@dataclass(frozen=True)
class AppointmentRow:
    id: str
    cluster: str
    operator: str
    members: list[str]
    issued_at: datetime
    expires_at: datetime
    signed: sign.Signed


@dataclass(frozen=True)
class Plan:
    id: str
    signed: sign.Signed
    body: protocol.ClusterPlan


def _row(row: sqlite3.Row) -> ClusterTaskRow:
    return ClusterTaskRow(
        id=row["id"],
        cluster=row["cluster"],
        signer=row["signer"],
        signed=sign.Signed.model_validate_json(row["signed_json"]),
        expires_at=row["expires_at"],
        status=row["status"],
        reason=row["reason"],
        note=row["note"],
        replan=bool(row["replan"]),
        operator_seen=row["operator_seen"],
        lost=row["lost"],
        created_at=row["created_at"],
    )


def cluster_task(conn: sqlite3.Connection, id: str) -> ClusterTaskRow | None:
    row = conn.execute("SELECT * FROM cluster_tasks WHERE id = ?", (id,)).fetchone()
    return None if row is None else _row(row)


def by_cluster(conn: sqlite3.Connection, cluster: str) -> ClusterTaskRow | None:
    row = conn.execute("SELECT * FROM cluster_tasks WHERE cluster = ?", (cluster,)).fetchone()
    return None if row is None else _row(row)


def cluster_tasks(conn: sqlite3.Connection, status: str | None = None) -> list[ClusterTaskRow]:
    if status is None:
        rows = conn.execute("SELECT * FROM cluster_tasks ORDER BY created_at, id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM cluster_tasks WHERE status = ? ORDER BY created_at, id", (status,)
        ).fetchall()
    return [_row(r) for r in rows]


def find(conn: sqlite3.Connection, prefix: str) -> list[str]:
    """Cluster tasks whose id (or cluster id) starts with `prefix`."""
    if not prefix or any(c not in "0123456789abcdef" for c in prefix):
        return []
    rows = conn.execute(
        "SELECT id FROM cluster_tasks WHERE substr(id, 1, ?) = ? OR substr(cluster, 1, ?) = ?"
        " ORDER BY id LIMIT 2",
        (len(prefix), prefix, len(prefix), prefix),
    ).fetchall()
    return [r["id"] for r in rows]


def _appointment(r: sqlite3.Row) -> AppointmentRow:
    return AppointmentRow(
        id=r["id"],
        cluster=r["cluster"],
        operator=r["operator"],
        members=json.loads(r["members_json"]),
        issued_at=db.parse_iso(r["issued_at"]),
        expires_at=db.parse_iso(r["expires_at"]),
        signed=sign.Signed.model_validate_json(r["signed_json"]),
    )


def appointments(conn: sqlite3.Connection, cluster: str) -> list[AppointmentRow]:
    """The cluster's appointments, oldest first."""
    rows = conn.execute(
        "SELECT * FROM appointments WHERE cluster = ? ORDER BY issued_at, seq", (cluster,)
    ).fetchall()
    return [_appointment(r) for r in rows]


def newest(conn: sqlite3.Connection, cluster: str) -> AppointmentRow | None:
    row = conn.execute(
        "SELECT * FROM appointments WHERE cluster = ? ORDER BY issued_at DESC, seq DESC LIMIT 1",
        (cluster,),
    ).fetchone()
    return None if row is None else _appointment(row)


def records(conn: sqlite3.Connection, cluster: str, kind: sign.RecordKind) -> list[sign.Signed]:
    rows = conn.execute(
        "SELECT signed_json FROM cluster_records WHERE cluster = ? AND kind = ? ORDER BY seq",
        (cluster, kind.value),
    ).fetchall()
    return [sign.Signed.model_validate_json(r["signed_json"]) for r in rows]


def latest_plan(conn: sqlite3.Connection, cluster: str) -> Plan | None:
    plans = records(conn, cluster, sign.RecordKind.CLUSTER_PLAN)
    if not plans:
        return None
    last = plans[-1]
    return Plan(last.record.id, last, protocol.ClusterPlan.model_validate(last.record.body))


def accepted_nodes(conn: sqlite3.Connection, task: str) -> dict[int, work.JobRow]:
    """The plan nodes whose job was accepted on recheck, by index (kept by every version)."""
    found: dict[int, work.JobRow] = {}
    for job in work.jobs(conn, "accepted", cluster=task):
        body = job.cluster_job
        if body is not None:
            found.setdefault(body.node, job)
    return found


# ── Who is free ──────────────────────────────────────────────────────────────


def _busy(conn: sqlite3.Connection) -> set[str]:
    """The operator and members of every running cluster."""
    taken: set[str] = set()
    for row in cluster_tasks(conn, "running"):
        found = newest(conn, row.cluster)
        if found is not None:
            taken |= {found.operator, *found.members}
    return taken


def _accepted_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT r.node, count(*) AS n FROM jobs j JOIN results r ON r.id = j.result"
        " WHERE j.status = 'accepted' GROUP BY r.node"
    ).fetchall()
    return {r["node"]: r["n"] for r in rows}


def _available(found: store.Node, now: datetime, online: timedelta) -> bool:
    return (
        found.status == "active"
        and not found.revoked
        and found.takes_jobs
        and bool(found.capabilities)
        and db.parse_iso(found.last_seen) > now - online
    )


def free_nodes(conn: sqlite3.Connection, now: datetime, online: timedelta) -> list[store.Node]:
    """Free nodes, best first: most results accepted on recheck, then the oldest."""
    busy = _busy(conn)
    counts = _accepted_counts(conn)
    found = [
        n for n in store.nodes(conn) if _available(n, now, online) and n.fingerprint not in busy
    ]
    return sorted(found, key=lambda n: (-counts.get(n.fingerprint, 0), n.registered_at))


def _may_lead(believed: sign.Trust, found: store.Node, excluded: set[str]) -> bool:
    # The root and operators lead no cluster (a node would refuse the appointment).
    return (
        found.lead
        and found.fingerprint not in excluded
        and found.fingerprint != believed.root_fingerprint
        and believed.operator(found.fingerprint) is None
    )


def _appointer_stands(believed: sign.Trust, appointer: sign.NodeKey) -> bool:
    found = believed.operator(appointer.fingerprint)
    return found is not None and sign.Scope.APPOINT in found.scopes


def why_waiting(
    conn: sqlite3.Connection,
    appointer: sign.NodeKey | None,
    root: Ed25519PublicKey,
    now: datetime,
    online: timedelta,
    task: ClusterTaskRow,
    *,
    pool: list[store.Node] | None = None,
    believed: sign.Trust | None = None,
) -> tuple[Waiting, int]:
    """Why a waiting cluster task has no cluster yet, and how many nodes are free. `pool` and
    `believed`, when given, are `free_nodes` and `store.trust` already computed for `now`."""
    pool = free_nodes(conn, now, online) if pool is None else pool
    if appointer is None:
        return "no_appointer", len(pool)
    believed = store.trust(conn, root, now) if believed is None else believed
    if not _appointer_stands(believed, appointer):
        return "appointer_not_assigned", len(pool)
    if believed.fresh() is not None:
        return "trust_list", len(pool)
    excluded = work.cluster_operators(conn, task.id) if task.lost else set()
    if not any(_may_lead(believed, n, excluded) for n in pool):
        return "no_leader", len(pool)
    return "free", len(pool)


# ── The global operator's side ───────────────────────────────────────────────


def add(
    conn: sqlite3.Connection, signed: sign.Signed, root: Ed25519PublicKey, now: datetime
) -> ClusterTaskRow:
    """Keep a cluster task whose record a node would accept now (else `Refused`)."""
    record = signed.record
    if record.kind is not sign.RecordKind.CLUSTER_TASK:
        raise Refused(Reason.MALFORMED, "a cluster task is a cluster_task record", id=record.id)
    try:
        protocol.ClusterTask.model_validate(record.body)
    except ValidationError as e:
        raise Refused(Reason.MALFORMED, str(e), id=record.id) from e
    cluster = uuid.uuid4().hex
    with db.transaction(conn):
        store.trust(conn, root, now).receive(signed.model_dump(mode="json"))
        conn.execute(
            "INSERT INTO cluster_tasks (id, cluster, signer, signed_json, expires_at, status,"
            " created_at) VALUES (?, ?, ?, ?, ?, 'waiting', ?)",
            (
                record.id,
                cluster,
                record.signer,
                json.dumps(signed.model_dump(mode="json"), sort_keys=True),
                db.to_iso(record.expires_at),
                db.to_iso(now),
            ),
        )
        store.log(
            conn,
            "cluster_task_added",
            record.signer,
            {"task": record.id, "cluster": cluster, "expires_at": db.to_iso(record.expires_at)},
        )
        added = cluster_task(conn, record.id)
    assert added is not None
    return added


SETTLES_FROM: Final[dict[Decision, tuple[str, ...]]] = {
    "accept": ("operator",),
    "again": ("operator",),
    "drop": ("waiting", "running", "operator"),
}


def settle(
    conn: sqlite3.Connection, id: str, decision: Decision, now: datetime, *, note: str = ""
) -> ClusterTaskRow:
    """The global operator's word on a cluster task in its inbox: `accept` its result, send it
    back `again` (its result → a new plan version; a job that gave up → lent again from the
    start; operators lost → counted from zero), or `drop` it (its jobs with it)."""
    note = note.strip()[:2000]
    moment = db.to_iso(now)
    with db.transaction(conn):
        found = cluster_task(conn, id)
        if found is None:
            raise SettleError("missing")
        if found.status not in SETTLES_FROM[decision]:
            raise SettleError(found.status)
        if decision == "accept" and found.reason != "result":
            raise SettleError(found.reason or found.status)
        if decision == "again":
            if db.parse_iso(found.expires_at) <= now:
                raise SettleError("expired")
            plan = latest_plan(conn, found.cluster)
            version = 0 if plan is None else plan.body.version
            if found.reason == "result" and version >= protocol.MAX_PLAN_VERSIONS:
                raise SettleError("no_versions")
            replan = found.reason == "result"
            if found.reason == "job_given_up":
                for job in work.jobs(conn, "operator", cluster=found.id):
                    if db.parse_iso(job.expires_at) <= now:
                        _drop_job(conn, job, "expired", moment)
                        replan = True
                    elif not _untried_members(conn, found, job):
                        # Every member took it once (a node never takes a record twice): the
                        # cluster's operator plans it anew.
                        _drop_job(conn, job, "no_member_left", moment)
                        replan = True
                    else:
                        conn.execute(
                            "UPDATE jobs SET status = 'execute', result = NULL, reason = NULL,"
                            " round = round + 1 WHERE id = ?",
                            (job.id,),
                        )
            conn.execute(
                "UPDATE cluster_tasks SET status = 'waiting', reason = NULL, note = ?,"
                " replan = ?, lost = 0 WHERE id = ?",
                (note or None, int(replan), found.id),
            )
        elif decision == "accept":
            conn.execute(
                "UPDATE cluster_tasks SET status = 'accepted', note = ? WHERE id = ?",
                (note or None, found.id),
            )
        else:
            for job in work.jobs(conn, cluster=found.id):
                if job.status in ("execute", "recheck", "operator"):
                    _drop_job(conn, job, "cluster_dropped", moment)
            conn.execute(
                "UPDATE cluster_tasks SET status = 'dropped', note = ? WHERE id = ?",
                (note or None, found.id),
            )
        store.log(
            conn,
            "cluster_settled",
            None,
            {
                "task": found.id,
                "decision": decision,
                "was": found.status,
                "reason": found.reason,
                "note": bool(note),
            },
        )
        settled = cluster_task(conn, found.id)
    assert settled is not None
    return settled


def _untried_members(conn: sqlite3.Connection, found: ClusterTaskRow, job: work.JobRow) -> set[str]:
    """Members of the cluster's newest appointment never lent this job (in any round)."""
    current = newest(conn, found.cluster)
    if current is None:
        return set()
    tried = {
        r["node"]
        for r in conn.execute("SELECT node FROM leases WHERE job = ?", (job.id,)).fetchall()
    }
    return set(current.members) - tried


def _drop_job(conn: sqlite3.Connection, job: work.JobRow, why: str, moment: str) -> None:
    for row in conn.execute(
        "SELECT token FROM leases WHERE job = ? AND ended IS NULL", (job.id,)
    ).fetchall():
        work.end_lease(conn, row["token"], "withdrawn", moment)
    conn.execute("UPDATE jobs SET status = 'dropped', reason = ? WHERE id = ?", (why, job.id))
    store.log(conn, "cluster_job_dropped", None, {"job": job.id, "why": why})


def _to_inbox(
    conn: sqlite3.Connection, found: ClusterTaskRow, reason: str, details: dict[str, Any]
) -> None:
    conn.execute(
        "UPDATE cluster_tasks SET status = 'operator', reason = ? WHERE id = ?", (reason, found.id)
    )
    store.log(conn, "cluster_to_operator", None, {"task": found.id, "reason": reason, **details})


# ── What the registry does by itself ─────────────────────────────────────────


def tick(
    conn: sqlite3.Connection,
    appointer: sign.NodeKey | None,
    root: Ed25519PublicKey,
    now: datetime,
    online: timedelta,
) -> None:
    """Expire, replace lost operators, send what gave up to the inbox, appoint where enough
    nodes are free. Its own transaction; run before anything reads or changes a cluster."""
    work.expire(conn, now)
    moment = db.to_iso(now)
    with db.transaction(conn):
        for row in conn.execute(
            "SELECT id FROM cluster_tasks WHERE status IN ('waiting', 'running')"
            " AND expires_at <= ?",
            (moment,),
        ).fetchall():
            conn.execute("UPDATE cluster_tasks SET status = 'expired' WHERE id = ?", (row["id"],))
            store.log(conn, "cluster_expired", None, {"task": row["id"]})
        for found in cluster_tasks(conn, "running"):
            _supervise(conn, found, now, online)
        waiting = cluster_tasks(conn, "waiting")
        if appointer is None or not waiting:
            return
        pool = free_nodes(conn, now, online)
        if len(pool) < NEEDED:
            return  # nothing to appoint (and no signature to check)
        believed = store.trust(conn, root, now)
        if not _appointer_stands(believed, appointer) or believed.fresh() is not None:
            return  # every node would refuse the appointment
        for found in waiting:  # the oldest first
            if len(pool) < NEEDED:
                break
            excluded = work.cluster_operators(conn, found.id) if found.lost else set()
            leaders = [n for n in pool if _may_lead(believed, n, excluded)]
            if not leaders:
                continue  # its former operators may still lead a newer one
            leader = leaders[0]
            members = [n.fingerprint for n in pool if n is not leader][: sign.MAX_MEMBERS]
            _appoint(conn, appointer, believed, found, leader.fingerprint, members, now, len(pool))
            chosen = {leader.fingerprint, *members}
            pool = [n for n in pool if n.fingerprint not in chosen]


def _supervise(
    conn: sqlite3.Connection, found: ClusterTaskRow, now: datetime, online: timedelta
) -> None:
    for job in work.jobs(conn, "execute", cluster=found.id):
        held = conn.execute(
            "SELECT 1 FROM leases WHERE job = ? AND ended IS NULL", (job.id,)
        ).fetchone()
        if held is None and not _untried_members(conn, found, job):
            # Every member took it once and none may take it again: back to the operator.
            work.to_operator(conn, job, "no_member_left", {})
    given_up = work.jobs(conn, "operator", cluster=found.id)
    if given_up:
        plan = latest_plan(conn, found.cluster)
        last = plan is not None and plan.body.version >= protocol.MAX_PLAN_VERSIONS
        if last or any(j.job.runs_code for j in given_up):
            _to_inbox(
                conn,
                found,
                "job_given_up",
                {"jobs": [j.id for j in given_up], "runs_code": not last},
            )
            return
    current = newest(conn, found.cluster)
    if current is None:
        return
    leader = store.node(conn, current.operator)
    seen = found.operator_seen
    gone = (
        leader is None
        or leader.status != "active"
        or leader.revoked
        or not leader.lead
        or seen is None
        or now - db.parse_iso(seen) >= TERM
        or now >= current.expires_at  # polling, but not renewed (the appointer cannot sign)
    )
    if not gone:
        return
    lost = found.lost + 1
    conn.execute("UPDATE cluster_tasks SET lost = ? WHERE id = ?", (lost, found.id))
    store.log(conn, "operator_lost", current.operator, {"task": found.id, "cluster": found.cluster})
    if lost >= MAX_OPERATORS_LOST:
        _to_inbox(conn, found, "operators_lost", {"lost": lost})
    else:
        conn.execute("UPDATE cluster_tasks SET status = 'waiting' WHERE id = ?", (found.id,))


def _appoint(
    conn: sqlite3.Connection,
    appointer: sign.NodeKey,
    believed: sign.Trust,
    found: ClusterTaskRow,
    leader: str,
    members: list[str],
    now: datetime,
    free: int,
) -> AppointmentRow:
    """Sign, check as a node would, keep and journal an appointment (in the caller's
    transaction); the cluster task is then running."""
    before = newest(conn, found.cluster)
    # The newest appointment of a cluster is the one in force: each later than the last.
    issued = now if before is None else max(now, before.issued_at + timedelta(seconds=1))
    lifetime = min(TERM, db.parse_iso(found.expires_at) - issued)
    leader_key = store.node(conn, leader)
    assert leader_key is not None
    signed = sign.appoint(
        appointer,
        sign.decode_public(leader_key.public_key),
        cluster=found.cluster,
        task=found.id,
        members=members,
        issued_at=issued,
        lifetime=lifetime,
    )
    believed.receive(signed.model_dump(mode="json"))
    record = signed.record
    conn.execute(
        "INSERT INTO appointments (id, cluster, operator, members_json, issued_at, expires_at,"
        " signed_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            record.id,
            found.cluster,
            leader,
            json.dumps(members),
            db.to_iso(record.issued_at),
            db.to_iso(record.expires_at),
            json.dumps(signed.model_dump(mode="json"), sort_keys=True),
        ),
    )
    renewed = before is not None and before.operator == leader and found.status == "running"
    if renewed:  # the poll that asked for it set operator_seen already
        conn.execute("UPDATE cluster_tasks SET status = 'running' WHERE id = ?", (found.id,))
    else:  # a new operator: its term starts as if it had just polled
        conn.execute(
            "UPDATE cluster_tasks SET status = 'running', operator_seen = ? WHERE id = ?",
            (db.to_iso(now), found.id),
        )
    store.log(
        conn,
        "appointment_renewed" if renewed else "appointed",
        leader,
        {
            "task": found.id,
            "cluster": found.cluster,
            "appointment": record.id,
            "members": len(members),
            "free": free,
            "expires_at": db.to_iso(record.expires_at),
        },
    )
    kept = newest(conn, found.cluster)
    assert kept is not None
    return kept


# ── The cluster operator's side ──────────────────────────────────────────────


def poll(
    conn: sqlite3.Connection,
    caller: store.Node,
    appointer: sign.NodeKey | None,
    root: Ed25519PublicKey,
    now: datetime,
    online: timedelta,
) -> protocol.ClusterState:
    """The caller's cluster. For its operator, this is its heartbeat; past half its term it is
    appointed again (members gone replaced by free nodes)."""
    tick(conn, appointer, root, now, online)
    with db.transaction(conn):
        for found in cluster_tasks(conn, "running"):
            current = newest(conn, found.cluster)
            if current is None:
                continue
            if current.operator == caller.fingerprint:
                conn.execute(
                    "UPDATE cluster_tasks SET operator_seen = ? WHERE id = ?",
                    (db.to_iso(now), found.id),
                )
                if appointer is not None and now >= current.issued_at + TERM / 2:
                    _renew(conn, appointer, root, found, current, now, online)
                return _operator_state(conn, found)
            if caller.fingerprint in current.members:
                return protocol.ClusterState(
                    role="member",
                    cluster=found.cluster,
                    task=found.signed,
                    appointment=current.signed,
                )
    return protocol.ClusterState()


def _renew(
    conn: sqlite3.Connection,
    appointer: sign.NodeKey,
    root: Ed25519PublicKey,
    found: ClusterTaskRow,
    current: AppointmentRow,
    now: datetime,
    online: timedelta,
) -> None:
    believed = store.trust(conn, root, now)
    if not _appointer_stands(believed, appointer) or believed.fresh() is not None:
        return
    if db.parse_iso(found.expires_at) <= now:
        return
    stay = []
    for fingerprint in current.members:
        member = store.node(conn, fingerprint)
        if member is not None and _available(member, now, online):
            stay.append(fingerprint)
    pool = [n.fingerprint for n in free_nodes(conn, now, online)]
    wanted = max(len(current.members), 1)
    members = (stay + [f for f in pool if f != current.operator])[:wanted] or current.members
    try:
        _appoint(conn, appointer, believed, found, current.operator, members, now, len(pool))
    except Refused:
        return  # nodes would refuse it too: the operator is replaced when its term ends


def _operator_state(conn: sqlite3.Connection, found: ClusterTaskRow) -> protocol.ClusterState:
    every = appointments(conn, found.cluster)
    jobs = []
    for job in work.jobs(conn, cluster=found.id):
        handed = work.handed_in(conn, job.id)
        result: sign.Signed | None = None
        result_key: str | None = None
        if job.result is not None:
            row = conn.execute(
                "SELECT r.signed_json, n.public_key FROM results r"
                " JOIN nodes n ON n.fingerprint = r.node WHERE r.id = ?",
                (job.result,),
            ).fetchone()
            if row is not None:
                result = sign.Signed.model_validate_json(row["signed_json"])
                result_key = row["public_key"]
        jobs.append(
            protocol.ClusterJobState(
                job=job.signed,
                status=job.status,
                reason=job.reason,
                result=result,
                result_key=result_key,
                verdicts=[h.signed for h in handed if h.kind == "verdict"],
            )
        )
    results = records(conn, found.cluster, sign.RecordKind.CLUSTER_RESULT)
    return protocol.ClusterState(
        role="operator",
        cluster=found.cluster,
        task=found.signed,
        appointment=every[-1].signed,
        appointments=[a.signed for a in every],
        plans=records(conn, found.cluster, sign.RecordKind.CLUSTER_PLAN),
        jobs=jobs,
        result=results[-1] if results else None,
        note=found.note,
    )


def hand_in(
    conn: sqlite3.Connection,
    caller: store.Node,
    given: protocol.ClusterHandIn,
    root: Ed25519PublicKey,
    now: datetime,
) -> protocol.ClusterReceived:
    """Keep a plan, a job or the result from the cluster's operator in force (else `Refused`)."""
    signed = given.record
    record = signed.record

    def refuse(reason: Reason, detail: str) -> Refused:
        return Refused(reason, detail, id=record.id, signer=caller.fingerprint)

    if record.kind not in sign.CLUSTER_KINDS:
        raise refuse(Reason.MALFORMED, "not a record of a cluster")
    if record.signer != caller.fingerprint:
        raise refuse(Reason.UNKNOWN_SIGNER, "the record is not signed by the caller")
    try:
        ref = sign.ClusterRef.model_validate(record.body)
    except ValidationError as e:
        raise refuse(Reason.MALFORMED, f"{e.error_count()} problem(s) in the body") from e
    moment = db.to_iso(now)
    with db.transaction(conn):
        found = by_cluster(conn, ref.cluster)
        if found is None or found.id != ref.task:
            raise refuse(Reason.NOT_ALLOWED, "no such cluster and task")
        if found.status != "running":
            raise refuse(Reason.NOT_ALLOWED, f"the cluster task is {found.status}")
        every = appointments(conn, found.cluster)
        if not every or every[-1].operator != caller.fingerprint or now >= every[-1].expires_at:
            raise refuse(Reason.NOT_ALLOWED, "not the cluster's operator in force")
        began = every[-1].issued_at
        for earlier in reversed(every):
            if earlier.operator != caller.fingerprint:
                break
            began = earlier.issued_at
        if record.issued_at < began:
            raise refuse(Reason.NOT_ALLOWED, "dated before the caller's appointment began")
        if record.expires_at > db.parse_iso(found.expires_at):
            raise refuse(Reason.MALFORMED, "the record outlives its cluster task")
        if conn.execute("SELECT 1 FROM cluster_records WHERE id = ?", (record.id,)).fetchone():
            raise refuse(Reason.REPLAY, "this record was already handed in")
        store.trust(conn, root, now).receive(signed.model_dump(mode="json"))
        try:
            if record.kind is sign.RecordKind.CLUSTER_PLAN:
                version = _take_plan(conn, found, record, refuse, moment)
            elif record.kind is sign.RecordKind.CLUSTER_JOB:
                version = _take_job(conn, found, signed, refuse, moment)
            else:
                version = _take_result(conn, found, record, refuse)
        except ValidationError as e:
            raise refuse(Reason.MALFORMED, f"{e.error_count()} problem(s) in the body") from e
        conn.execute(
            "INSERT INTO cluster_records (id, cluster, kind, signer, version, signed_json,"
            " received_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                found.cluster,
                record.kind.value,
                caller.fingerprint,
                version,
                json.dumps(signed.model_dump(mode="json"), sort_keys=True),
                moment,
            ),
        )
        store.log(
            conn,
            "cluster_record",
            caller.fingerprint,
            {
                "task": found.id,
                "cluster": found.cluster,
                "kind": record.kind.value,
                "record": record.id,
                "version": version,
            },
        )
        now_found = cluster_task(conn, found.id)
    assert now_found is not None
    return protocol.ClusterReceived(
        cluster=found.cluster, record=record.id, kind=record.kind.value, status=now_found.status
    )


def _take_plan(
    conn: sqlite3.Connection,
    found: ClusterTaskRow,
    record: sign.Record,
    refuse: _Refuse,
    moment: str,
) -> int:
    plan = protocol.ClusterPlan.model_validate(record.body)
    before = latest_plan(conn, found.cluster)
    expected = 1 if before is None else before.body.version + 1
    if plan.version != expected:
        raise refuse(Reason.MALFORMED, f"the next plan is version {expected}")
    if before is not None:
        given_up = work.jobs(conn, "operator", cluster=found.id)
        if not given_up and not found.replan:
            raise refuse(Reason.NOT_ALLOWED, "nothing to plan again")
        for index in accepted_nodes(conn, found.id):
            if index >= len(plan.nodes) or plan.nodes[index] != before.body.nodes[index]:
                raise refuse(Reason.MALFORMED, f"node {index} was accepted: it stays as it is")
        for job in work.jobs(conn, cluster=found.id):
            body = job.cluster_job
            if body is None or job.status not in ("execute", "recheck", "operator"):
                continue
            kept = (
                body.node < len(plan.nodes)
                and plan.nodes[body.node] == before.body.nodes[body.node]
            )
            if job.status == "operator" or not kept:
                _drop_job(conn, job, "superseded", moment)
        conn.execute("UPDATE cluster_tasks SET replan = 0 WHERE id = ?", (found.id,))
    return plan.version


def _take_job(
    conn: sqlite3.Connection,
    found: ClusterTaskRow,
    signed: sign.Signed,
    refuse: _Refuse,
    moment: str,
) -> int:
    record = signed.record
    body = protocol.ClusterJob.model_validate(record.body)
    plan = latest_plan(conn, found.cluster)
    if plan is None or body.plan != plan.id or body.version != plan.body.version:
        raise refuse(Reason.NOT_ALLOWED, "not the cluster's current plan")
    if body.node >= len(plan.body.nodes):
        raise refuse(Reason.MALFORMED, "no such node in the plan")
    node = plan.body.nodes[body.node]
    job = body.job
    if (job.description, job.acceptance_criteria, job.kind, job.runs_code) != (
        node.description,
        node.acceptance_criteria,
        node.kind,
        node.runs_code,
    ):
        raise refuse(Reason.MALFORMED, "not the plan's node")
    accepted = accepted_nodes(conn, found.id)
    if any(d not in accepted for d in node.depends_on):
        raise refuse(Reason.NOT_ALLOWED, "a node it depends on is not accepted yet")
    for other in work.jobs(conn, cluster=found.id):
        other_body = other.cluster_job
        if (
            other_body is not None
            and other_body.node == body.node
            and other.status in ("execute", "recheck", "accepted", "operator")
        ):
            raise refuse(Reason.CONFLICT, "this node is lent already")
    conn.execute(
        "INSERT INTO jobs (id, signer, signed_json, expires_at, status, created_at, cluster)"
        " VALUES (?, ?, ?, ?, 'execute', ?, ?)",
        (
            record.id,
            record.signer,
            json.dumps(signed.model_dump(mode="json"), sort_keys=True),
            db.to_iso(record.expires_at),
            moment,
            found.id,
        ),
    )
    store.log(
        conn,
        "job_added",
        record.signer,
        {"job": record.id, "cluster": found.id, "node": body.node, "version": body.version},
    )
    return body.version


def _take_result(
    conn: sqlite3.Connection, found: ClusterTaskRow, record: sign.Record, refuse: _Refuse
) -> int:
    body = protocol.ClusterResult.model_validate(record.body)
    plan = latest_plan(conn, found.cluster)
    if plan is None or body.plan != plan.id or body.version != plan.body.version:
        raise refuse(Reason.NOT_ALLOWED, "not the cluster's current plan")
    last = accepted_nodes(conn, found.id).get(len(plan.body.nodes) - 1)
    if last is None or last.result != body.result:
        raise refuse(Reason.NOT_ALLOWED, "not the accepted result of the plan's last node")
    _to_inbox(conn, found, "result", {"result": body.result})
    return body.version
