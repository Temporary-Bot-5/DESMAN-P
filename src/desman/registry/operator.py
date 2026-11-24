"""Registry-side authority and actions for the operator API (ADR 0016)."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import TypeVar

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ValidationError

from desman.net import protocol as p
from desman.net import sign
from desman.registry import admin, clusters, moderation, store, work

TASKS = sign.Scope.TASKS
MODERATION = sign.Scope.MODERATION


class Refusal(Exception):
    def __init__(self, reason: str, status: int) -> None:
        self.reason = reason
        self.status = status
        super().__init__(reason)


def assignment(
    conn: sqlite3.Connection, fingerprint: str, root: Ed25519PublicKey, now: datetime
) -> sign.Operator:
    if not store.is_revoked(conn, fingerprint):
        rows = conn.execute(
            "SELECT signed_json FROM records WHERE subject = ? AND kind = 'assignment'"
            " ORDER BY issued_at DESC, seq DESC",
            (fingerprint,),
        )
        for row in rows:
            believed = sign.Trust(root, clock=lambda: now)
            try:
                record = believed.receive(row["signed_json"])
            except sign.Refused:
                continue
            found = believed.operator(fingerprint)
            if found is None or record.issued_at > now:
                continue
            if found.scopes & {TASKS, MODERATION}:
                return found
            break
    raise Refusal("not_operator", 403)


@dataclass(frozen=True)
class Actions:
    conn: sqlite3.Connection
    root: Ed25519PublicKey
    now: datetime
    appointer: Callable[[], sign.NodeKey | None]
    online: timedelta
    identity: p.OperatorIdentity

    def tick(self) -> None:
        clusters.tick(self.conn, self.appointer(), self.root, self.now, self.online)

    def check_record(self, signed: sign.Signed) -> None:
        if signed.record.signer != self.identity.fingerprint:
            raise Refusal("wrong_signer", 403)
        if signed.record.expires_at > self.identity.until:
            raise Refusal("expired", 403)


M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class Route:
    scope: sign.Scope | None
    perform: Callable[[Actions, bytes], tuple[BaseModel, str | None]]
    writes: bool


ROUTES: dict[str, Route] = {}


def route(
    path: str, body: type[M], scope: sign.Scope | None = None, *, writes: bool = False
) -> Callable[
    [Callable[[Actions, M], tuple[BaseModel, str | None]]],
    Callable[[Actions, M], tuple[BaseModel, str | None]],
]:
    def register(
        action: Callable[[Actions, M], tuple[BaseModel, str | None]],
    ) -> Callable[[Actions, M], tuple[BaseModel, str | None]]:
        def perform(context: Actions, data: bytes) -> tuple[BaseModel, str | None]:
            try:
                asked = body.model_validate_json(data)
            except ValidationError as e:
                raise Refusal("malformed", 400) from e
            return action(context, asked)

        ROUTES[path] = Route(scope, perform, writes=writes)
        return action

    return register


def _jobs(rows: list[work.JobRow]) -> list[p.OperatorJob]:
    return [p.OperatorJob.model_validate(row.as_json()) for row in rows]


def _cluster_fields(
    ctx: Actions,
    row: clusters.ClusterTaskRow,
    *,
    jobs: list[work.JobRow],
    current: clusters.AppointmentRow | None,
    plan: clusters.Plan | None,
) -> dict[str, object]:
    why, free = row.reason, None
    if row.status == "waiting":
        why, free = clusters.why_waiting(
            ctx.conn, ctx.appointer(), ctx.root, ctx.now, ctx.online, row
        )
    return row.as_json() | {
        "operator": None if current is None else current.operator,
        "members": 0 if current is None else len(current.members),
        "plan_version": 0 if plan is None else plan.body.version,
        "jobs": len(jobs),
        "accepted": sum(job.status == "accepted" for job in jobs),
        "why": why,
        "free": free,
    }


def _cluster(ctx: Actions, row: clusters.ClusterTaskRow) -> p.OperatorCluster:
    fields = _cluster_fields(
        ctx,
        row,
        jobs=work.jobs(ctx.conn, cluster=row.id),
        current=clusters.newest(ctx.conn, row.cluster),
        plan=clusters.latest_plan(ctx.conn, row.cluster),
    )
    return p.OperatorCluster.model_validate(fields)


def _settle_error(conn: sqlite3.Connection, kind: str, field: str, id: str) -> Refusal:
    latest = conn.execute(
        "SELECT action FROM registry_log WHERE action IN (?, ?)"
        " AND json_extract(details_json, ?) = ? ORDER BY seq DESC LIMIT 1",
        (f"{kind}_settled", f"{kind}_to_operator", field, id),
    ).fetchone()
    if latest is not None and latest["action"] == f"{kind}_settled":
        return Refusal("already_settled", 409)
    return Refusal("cannot_settle", 409)


@route(p.OPERATOR_WHOAMI, p.OperatorEmpty)
def whoami(ctx: Actions, asked: p.OperatorEmpty) -> tuple[p.OperatorIdentity, None]:
    return ctx.identity, None


@route(p.OPERATOR_INBOX, p.OperatorEmpty)
def inbox(ctx: Actions, asked: p.OperatorEmpty) -> tuple[p.OperatorInbox, None]:
    ctx.tick()
    return p.OperatorInbox(
        jobs=_jobs(work.jobs(ctx.conn, "operator", cluster=None)),
        clusters=[_cluster(ctx, row) for row in clusters.cluster_tasks(ctx.conn, "operator")],
    ), None


@route(p.OPERATOR_JOBS, p.OperatorJobQuery)
def jobs(ctx: Actions, asked: p.OperatorJobQuery) -> tuple[p.OperatorJobs, None]:
    work.expire(ctx.conn, ctx.now)
    return p.OperatorJobs(jobs=_jobs(work.jobs(ctx.conn, asked.status))), None


@route(p.OPERATOR_JOB, p.OperatorRef)
def job(ctx: Actions, asked: p.OperatorRef) -> tuple[p.OperatorJobDetail, str]:
    work.expire(ctx.conn, ctx.now)
    id = admin.find_job(ctx.conn, asked.id)
    row = work.job(ctx.conn, id)
    assert row is not None
    handed = [item.as_json() for item in work.handed_in(ctx.conn, id)]
    return p.OperatorJobDetail.model_validate(row.as_json() | {"handed_in": handed}), id


@route(p.OPERATOR_JOB_ADD, p.OperatorRecord, TASKS, writes=True)
def job_add(ctx: Actions, asked: p.OperatorRecord) -> tuple[p.OperatorJob, str]:
    ctx.check_record(asked.record)
    row = work.add(ctx.conn, asked.record, ctx.root, ctx.now)
    return p.OperatorJob.model_validate(row.as_json()), row.id


@route(p.OPERATOR_JOB_SETTLE, p.OperatorSettle, TASKS, writes=True)
def job_settle(ctx: Actions, asked: p.OperatorSettle) -> tuple[p.OperatorJob, str]:
    work.expire(ctx.conn, ctx.now)
    id = admin.find_job(ctx.conn, asked.id)
    admin.ordinary_job(ctx.conn, id)
    try:
        row = work.settle(ctx.conn, id, asked.decision, ctx.now)
    except work.SettleError as e:
        raise _settle_error(ctx.conn, "job", "$.job", id) from e
    return p.OperatorJob.model_validate(row.as_json()), id


@route(p.OPERATOR_CLUSTERS, p.OperatorClusterQuery)
def cluster_list(ctx: Actions, asked: p.OperatorClusterQuery) -> tuple[p.OperatorClusters, None]:
    ctx.tick()
    return p.OperatorClusters(
        clusters=[_cluster(ctx, row) for row in clusters.cluster_tasks(ctx.conn, asked.status)]
    ), None


@route(p.OPERATOR_CLUSTER, p.OperatorRef)
def cluster(ctx: Actions, asked: p.OperatorRef) -> tuple[p.OperatorClusterDetail, str]:
    ctx.tick()
    id = admin.find_cluster(ctx.conn, asked.id)
    row = clusters.cluster_task(ctx.conn, id)
    assert row is not None
    current = clusters.newest(ctx.conn, row.cluster)
    plan = clusters.latest_plan(ctx.conn, row.cluster)
    results = clusters.records(ctx.conn, row.cluster, sign.RecordKind.CLUSTER_RESULT)
    jobs = work.jobs(ctx.conn, cluster=id)
    fields = _cluster_fields(ctx, row, jobs=jobs, current=current, plan=plan)
    return p.OperatorClusterDetail.model_validate(
        fields
        | {
            "members_list": [] if current is None else current.members,
            "plan": None if plan is None else plan.body,
            "job_list": _jobs(jobs),
            "result": None
            if not results
            else p.ClusterResult.model_validate(results[-1].record.body),
        }
    ), id


@route(p.OPERATOR_CLUSTER_ADD, p.OperatorRecord, TASKS, writes=True)
def cluster_add(ctx: Actions, asked: p.OperatorRecord) -> tuple[p.OperatorCluster, str]:
    ctx.check_record(asked.record)
    row = clusters.add(ctx.conn, asked.record, ctx.root, ctx.now)
    return _cluster(ctx, row), row.id


@route(p.OPERATOR_CLUSTER_SETTLE, p.OperatorClusterSettle, TASKS, writes=True)
def cluster_settle(ctx: Actions, asked: p.OperatorClusterSettle) -> tuple[p.OperatorCluster, str]:
    ctx.tick()
    id = admin.find_cluster(ctx.conn, asked.id)
    try:
        row = clusters.settle(ctx.conn, id, asked.decision, ctx.now, note=asked.note or "")
    except clusters.SettleError as e:
        raise _settle_error(ctx.conn, "cluster", "$.task", id) from e
    return _cluster(ctx, row), id


@route(p.OPERATOR_COMPLAINTS, p.OperatorComplaintsQuery, MODERATION)
def complaints(ctx: Actions, asked: p.OperatorComplaintsQuery) -> tuple[p.OperatorComplaints, None]:
    return p.OperatorComplaints(complaints=moderation.complaints(ctx.conn, all=asked.all)), None


@route(p.OPERATOR_COMPLAINT, p.OperatorRef, MODERATION)
def complaint(ctx: Actions, asked: p.OperatorRef) -> tuple[p.ComplaintDetail, str]:
    result = moderation.complaint(ctx.conn, asked.id)
    return result, result.id


@route(p.OPERATOR_COMPLAINT_DISMISS, p.OperatorRef, MODERATION, writes=True)
def complaint_dismiss(ctx: Actions, asked: p.OperatorRef) -> tuple[p.OperatorRef, str]:
    id = moderation.dismiss(ctx.conn, asked.id, None, ctx.now)
    return p.OperatorRef(id=id), id


@route(p.OPERATOR_MODERATION, p.OperatorRecord, MODERATION, writes=True)
def moderate(ctx: Actions, asked: p.OperatorRecord) -> tuple[p.ModerationRow, str]:
    ctx.check_record(asked.record)
    id = asked.record.record.id
    row = moderation.add(ctx.conn, asked.record, ctx.root, ctx.now)
    return row, id


@route(p.OPERATOR_SUSPENSIONS, p.OperatorEmpty, MODERATION)
def suspensions(ctx: Actions, asked: p.OperatorEmpty) -> tuple[p.OperatorSuspensions, None]:
    rows = moderation.moderations(ctx.conn, now=ctx.now, root=ctx.root)
    return p.OperatorSuspensions(suspensions=rows), None


@route(p.OPERATOR_REPORTS, p.OperatorReportQuery)
def reports(ctx: Actions, asked: p.OperatorReportQuery) -> tuple[p.OperatorReports, None]:
    rows = ctx.conn.execute(
        "SELECT id, node, kind, created_at FROM reports WHERE (? IS NULL OR kind = ?)"
        " ORDER BY created_at DESC, id",
        (asked.kind, asked.kind),
    )
    return p.OperatorReports(
        reports=[p.OperatorReport.model_validate(dict(row)) for row in rows]
    ), None


@route(p.OPERATOR_REPORT, p.OperatorRef)
def report(ctx: Actions, asked: p.OperatorRef) -> tuple[p.OperatorReportDetail, str]:
    rows = ctx.conn.execute(
        "SELECT * FROM reports WHERE substr(id,1,?)=? LIMIT 2", (len(asked.id), asked.id)
    ).fetchall()
    if not rows:
        raise Refusal("not_found", 404)
    if len(rows) > 1:
        raise Refusal("ambiguous", 409)
    data = dict(rows[0])
    data["report"] = p.SignedReport.model_validate_json(data.pop("report_json"))
    return p.OperatorReportDetail.model_validate(data), rows[0]["id"]


@route(p.OPERATOR_NODES, p.OperatorEmpty)
def nodes(ctx: Actions, asked: p.OperatorEmpty) -> tuple[p.OperatorNodes, None]:
    return p.OperatorNodes(
        nodes=[p.OperatorNode.model_validate(node.as_json()) for node in store.nodes(ctx.conn)]
    ), None


@route(p.OPERATOR_OWNER, p.OperatorOwner, TASKS, writes=True)
def owner(ctx: Actions, asked: p.OperatorOwner) -> tuple[p.OperatorNode, str]:
    id = admin.set_owner(ctx.conn, asked.node, asked.owner)
    row = store.node(ctx.conn, id)
    assert row is not None
    return p.OperatorNode.model_validate(row.as_json()), id


@route(p.OPERATOR_LOG, p.OperatorLogQuery)
def log(ctx: Actions, asked: p.OperatorLogQuery) -> tuple[p.OperatorLog, None]:
    return p.OperatorLog(
        entries=[
            p.OperatorLogEntry.model_validate(asdict(row))
            for row in store.entries(ctx.conn, asked.limit)
        ]
    ), None
