"""Node-side cluster supervision: all continuation state comes from signed registry records."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any, Final

from pydantic import BaseModel

from desman import planner
from desman.core import calls, db, netlog
from desman.core.models import SubtaskSpec
from desman.llm.base import LLM
from desman.net import protocol, sign

CLUSTER_POLL: Final = timedelta(minutes=2)
MARKER: Final = "\n[truncated]"
Send = Callable[[sign.RecordKind, BaseModel], sign.Signed | None]


def role(conn: sqlite3.Connection, now: datetime) -> dict[str, Any] | None:
    """Newest role receipt, only until its appointment ends."""
    row = conn.execute(
        "SELECT action, details_json FROM network_log WHERE action IN "
        "('cluster_joined', 'cluster_renewed', 'cluster_left') ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    if row is None or row["action"] == "cluster_left":
        return None
    details: dict[str, Any] = json.loads(row["details_json"])
    if db.parse_iso(details["expires_at"]) <= now:
        return None
    return details


def left(conn: sqlite3.Connection, now: datetime, *, ended: bool = False) -> None:
    row = conn.execute(
        "SELECT action, details_json FROM network_log WHERE action IN "
        "('cluster_joined', 'cluster_renewed', 'cluster_left') ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    if row is not None and row["action"] != "cluster_left":
        current = json.loads(row["details_json"])
        netlog.record(conn, "cluster_left", {"cluster": current["cluster"], "ended": ended})


def receive(state: protocol.ClusterState, trust: sign.Trust) -> None:
    """Revalidate the whole snapshot on this node's Trust, including after restart.

    The caller uses Seen.inspect(): verifying a snapshot never marks a job as taken.
    Signature, authority, revocation, body and record linkage are still checked.
    """
    records: list[tuple[sign.Signed, sign.RecordKind, type[BaseModel]]] = [
        (a, sign.RecordKind.APPOINTMENT, sign.Appointment) for a in state.appointments
    ]
    if state.appointment is not None:
        records.append((state.appointment, sign.RecordKind.APPOINTMENT, sign.Appointment))
    if state.task is not None:
        records.append((state.task, sign.RecordKind.CLUSTER_TASK, protocol.ClusterTask))
    records += [(p, sign.RecordKind.CLUSTER_PLAN, protocol.ClusterPlan) for p in state.plans]
    records += [(j.job, sign.RecordKind.CLUSTER_JOB, protocol.ClusterJob) for j in state.jobs]
    if state.result is not None:
        records.append((state.result, sign.RecordKind.CLUSTER_RESULT, protocol.ClusterResult))
    for signed, expected, schema in records:
        record = signed.record
        if record.kind is not expected:
            raise sign.Refused(sign.Reason.MALFORMED, "unexpected cluster record kind")
        schema.model_validate(record.body)
        if record.kind in sign.CLUSTER_KINDS:
            ref = sign.ClusterRef.model_validate(record.body)
            if (
                ref.cluster != state.cluster
                or state.task is None
                or ref.task != state.task.record.id
            ):
                raise sign.Refused(sign.Reason.NOT_ALLOWED, "snapshot belongs to another cluster")
        trust.receive(signed.model_dump(mode="json"))
    for job in state.jobs:
        if job.result is not None:
            if job.result_key is None:
                raise sign.Refused(sign.Reason.MALFORMED, "result has no public key")
            sign.check_signature(job.result, sign.decode_public(job.result_key))
            result = protocol.Result.model_validate(job.result.record.body)
            if (
                job.result.record.kind is not sign.RecordKind.RESULT
                or result.job != job.job.record.id
            ):
                raise sign.Refused(sign.Reason.MALFORMED, "result names another job")
            if job.status == "accepted" and result.outcome != "done":
                raise sign.Refused(sign.Reason.MALFORMED, "accepted result is not done")


def joined(
    conn: sqlite3.Connection,
    appointment: sign.Appointed,
    who: str,
    now: datetime,
    *,
    from_trust: bool = False,
) -> None:
    if from_trust:
        previous = conn.execute(
            "SELECT action, details_json FROM network_log WHERE action IN "
            "('cluster_joined', 'cluster_renewed', 'cluster_left') AND "
            "json_extract(details_json, '$.cluster') = ? ORDER BY seq DESC LIMIT 1",
            (appointment.cluster,),
        ).fetchone()
        if (
            previous is not None
            and previous["action"] == "cluster_left"
            and json.loads(previous["details_json"]).get("ended")
        ):
            return
    details = dict(
        role=who,
        cluster=appointment.cluster,
        task=appointment.task,
        members=len(appointment.members),
        expires_at=db.to_iso(appointment.expires_at),
    )
    current = role(conn, now)
    if current != details:
        renewed = current is not None and all(
            current.get(k) == v for k, v in details.items() if k != "expires_at"
        )
        netlog.record(conn, "cluster_renewed" if renewed else "cluster_joined", details)


def inputs(values: dict[str, str]) -> dict[str, str]:
    """Fit escaped JSON bytes, with a visible cut marker and at most sixteen inputs."""
    out: dict[str, str] = {}
    remaining = protocol.MAX_INPUT_BYTES
    items = list(values.items())
    if len(items) > 16:
        name, value = items[15]
        items = items[:15] + [
            (
                name,
                "[additional inputs folded into this entry]\n"
                + value
                + "\n"
                + json.dumps(dict(items[16:]), ensure_ascii=False),
            )
        ]
    for i, (name, value) in enumerate(items):
        budget = remaining // (len(items) - i)
        if len(json.dumps(value, ensure_ascii=True)) > budget:
            lo, hi = 0, len(value)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(json.dumps(value[:mid] + MARKER, ensure_ascii=True)) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            value = value[:lo] + MARKER
        out[name] = value
        remaining -= len(json.dumps(value, ensure_ascii=True))
    return out


def plan_id(plan: protocol.ClusterPlan, note: str | None) -> str | None:
    """A note's receipt in the signed record ID, readable by any successor.

    The registry retains notes after taking a plan. Naming that plan from its task,
    version and note distinguishes a new request from a consumed one without local state
    or metadata inside the subtask descriptions. Other records keep their random IDs.
    """
    if not note:
        return None
    return hashlib.sha256(
        sign.canonical(
            {
                "cluster": plan.cluster,
                "task": plan.task,
                "version": plan.version,
                "note": note,
            }
        )
    ).hexdigest()[:32]


def nodes(llm: LLM, task: protocol.ClusterTask, note: str | None) -> list[protocol.PlanNode]:
    goal = json.dumps(
        dict(
            goal=task.goal,
            acceptance_criteria=task.acceptance_criteria,
            inputs=task.inputs,
            note=note,
        ),
        ensure_ascii=False,
    )
    planned = planner.plan(llm, goal, provider_fallback=False)
    if planned.source == "template":
        return [
            protocol.PlanNode(description=task.goal, acceptance_criteria=task.acceptance_criteria)
        ]
    return _ordered(planned.spec.subtasks)


def _ordered(parts: list[SubtaskSpec]) -> list[protocol.PlanNode]:
    ordered: list[protocol.PlanNode] = []
    pending = list(enumerate(parts))
    ids: dict[int, int] = {}
    while pending:
        index, sub = next((i, s) for i, s in pending if all(d in ids for d in s.depends_on))
        ordered.append(
            protocol.PlanNode(
                description=sub.description[:4000],
                acceptance_criteria=sub.acceptance_criteria[:2000],
                depends_on=[ids[d] for d in sub.depends_on],
                kind=sub.kind,
                runs_code=sub.runs_code,
            )
        )
        ids[index] = len(ordered) - 1
        pending.remove((index, sub))
    return ordered


def split(
    llm: LLM, node: protocol.PlanNode, why: str | None, inputs: dict[str, str]
) -> list[protocol.PlanNode] | None:
    parts = planner.split_task(
        llm, node.description, node.acceptance_criteria, node.kind, dict(inputs), why
    )
    if parts is None:
        return None
    # As in Plan.spec, every unused branch must feed the final result.
    ordered = _ordered(parts)
    used = {d for part in ordered for d in part.depends_on}
    last = ordered[-1]
    ordered[-1] = last.model_copy(
        update={
            "depends_on": sorted(
                set([*last.depends_on, *(i for i in range(len(ordered) - 1) if i not in used)])
            )
        }
    )
    return ordered


def supervise(
    state: protocol.ClusterState, llm: LLM, send: Send, stuck: Callable[[str], None]
) -> None:
    assert state.cluster is not None and state.task is not None
    task_id = state.task.record.id
    task = protocol.ClusterTask.model_validate(state.task.record.body)
    ref = dict(cluster=state.cluster, task=task_id)
    jobs = [(j, protocol.ClusterJob.model_validate(j.job.record.body)) for j in state.jobs]
    accepted = {b.node: j for j, b in jobs if j.status == "accepted"}
    latest = max(state.plans, key=lambda p: int(p.record.body["version"]), default=None)
    with calls.for_(calls.Purpose(calls.Role.NETWORK_PLANNER, task_id=task_id)):
        if latest is None:
            send(
                sign.RecordKind.CLUSTER_PLAN,
                protocol.ClusterPlan(**ref, version=1, nodes=nodes(llm, task, state.note)),
            )
            return
        plan = protocol.ClusterPlan.model_validate(latest.record.body)
        failed = next(((j, b) for j, b in jobs if j.status == "operator"), None)
        # Older registries omit replan; a returned result still authorizes a fresh plan.
        # Return after planning so dropped jobs are never lent under the old version.
        replan = state.replan or bool(
            state.result and state.result.record.body.get("plan") == latest.record.id
        )
        dropped = next(iter(state.dropped_nodes), None)
        if failed or replan:
            if plan.version >= protocol.MAX_PLAN_VERSIONS:
                stuck("versions")
                return
            if failed or dropped is not None:
                index = failed[1].node if failed else dropped
                assert index is not None
                old = plan.nodes[index]
                if old.runs_code:
                    stuck("runs_code")
                    return
                parts = split(
                    llm,
                    old,
                    "\n".join(s for s in (failed[0].reason if failed else None, state.note) if s),
                    failed[1].job.inputs if failed else {},
                )
                if not parts:
                    stuck("cannot_split")
                    return
                # Keep accepted indices stable and append the replacement's continuation.
                changed = _replace(plan.nodes, index, parts, set(accepted))
            else:
                fresh = nodes(llm, task, state.note)
                prefix = max(accepted, default=-1) + 1
                changed = plan.nodes[:prefix] + [
                    n.model_copy(
                        update={
                            "depends_on": sorted(
                                set([*range(prefix), *(prefix + d for d in n.depends_on)])
                            )
                        }
                    )
                    for n in fresh
                ]
            if changed is None or len(changed) > protocol.MAX_PLAN_NODES:
                stuck("nodes")
                return
            send(
                sign.RecordKind.CLUSTER_PLAN,
                protocol.ClusterPlan(**ref, version=plan.version + 1, nodes=changed),
            )
            return
    last = accepted.get(len(plan.nodes) - 1)
    if last is not None and last.result is not None:
        result = protocol.Result.model_validate(last.result.record.body)
        send(
            sign.RecordKind.CLUSTER_RESULT,
            protocol.ClusterResult(
                **ref,
                plan=latest.record.id,
                version=plan.version,
                result=last.result.record.id,
                summary=result.output[:2000],
            ),
        )
        return
    busy = {b.node for j, b in jobs if j.status in ("execute", "recheck", "accepted", "operator")}
    for index, node in enumerate(plan.nodes):
        if index in busy or any(d not in accepted for d in node.depends_on):
            continue
        values: dict[str, str] = {}
        for dep in node.depends_on:
            result_record = accepted[dep].result
            if result_record is None:
                return
            result = protocol.Result.model_validate(result_record.record.body)
            values[f"node-{dep}"] = result.output + "".join(
                f"\n{f.path}: {f.text}" for f in result.files
            )
        for name, value in task.inputs.items():
            values.setdefault(name, value)
        send(
            sign.RecordKind.CLUSTER_JOB,
            protocol.ClusterJob(
                **ref,
                plan=latest.record.id,
                version=plan.version,
                node=index,
                job=protocol.Job(**node.model_dump(exclude={"depends_on"}), inputs=inputs(values)),
            ),
        )


def _replace(
    old: list[protocol.PlanNode], index: int, parts: list[protocol.PlanNode], accepted: set[int]
) -> list[protocol.PlanNode] | None:
    """Rebuild the suffix, holding only accepted nodes at their original indices."""
    result = list(old[:index])
    # Tokens distinguish replacement parts from old nodes, whose indices may move.
    pending = [("part", i) for i in range(len(parts))] + [
        ("old", i) for i in range(index + 1, len(old))
    ]
    mapping = {("old", i): i for i in range(index)}

    def dependencies(token: tuple[str, int]) -> list[tuple[str, int]]:
        kind, i = token
        if kind == "part":
            return [("part", d) for d in parts[i].depends_on] or [
                ("old", d) for d in old[index].depends_on
            ]
        return [("part", len(parts) - 1) if d == index else ("old", d) for d in old[i].depends_on]

    while pending:
        slot = len(result)
        candidates = [("old", slot)] if slot in accepted else pending
        chosen = next(
            (
                token
                for token in candidates
                if token in pending
                and all(d in mapping for d in dependencies(token))
                and (token != ("old", len(old) - 1) or len(pending) == 1)
                and (token[0] != "old" or token[1] not in accepted or token[1] == slot)
            ),
            None,
        )
        if chosen is None:
            return None
        kind, i = chosen
        original = parts[i] if kind == "part" else old[i]
        updated = original.model_copy(
            update={"depends_on": [mapping[d] for d in dependencies(chosen)]}
        )
        if kind == "old" and i in accepted and updated != original:
            return None
        result.append(updated)
        mapping[chosen] = slot
        pending.remove(chosen)
    return result
