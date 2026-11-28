"""Read-only activity reports for the registry's root owner."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from desman.core import db
from desman.i18n import t
from desman.net import sign
from desman.registry import admin, operator, store


@dataclass(frozen=True)
class Authority:
    id: str
    kind: str
    issued_at: str
    expires_at: str
    scopes: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class Summary:
    fingerprint: str
    scopes: tuple[str, ...]
    issued_at: str
    expires_at: str
    state: str
    registry_key: bool
    appointer: bool
    last_request: store.LogEntry | None
    jobs: int
    cluster_tasks: int
    writes: int
    refusals: int
    complaint_reads: int


@dataclass(frozen=True)
class Worker:
    fingerprint: str
    owner: str | None
    leases: int


@dataclass(frozen=True)
class Usage:
    status: str
    count: int
    leases: int
    results: int
    verdicts: int
    cluster_results: int
    nodes: tuple[Worker, ...]


@dataclass(frozen=True)
class Detail:
    summary: Summary
    records: tuple[Authority, ...]
    jobs: tuple[Usage, ...]
    cluster_tasks: tuple[Usage, ...]
    journal: tuple[store.LogEntry, ...]


def _entry(row: sqlite3.Row) -> store.LogEntry:
    return store.LogEntry(
        row["seq"], row["ts"], row["action"], row["node"], json.loads(row["details_json"])
    )


def _summary(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    root: Ed25519PublicKey,
    now: datetime,
    own: str | None,
) -> Summary:
    fingerprint = row["subject"]
    signed = sign.parse(row["signed_json"])
    scopes = tuple(sorted(sign.Assignment.model_validate(signed.record.body).scopes))
    if store.is_revoked(conn, fingerprint):
        state = "revoked"
    elif signed.record.issued_at > now:
        state = "not_yet"
    elif signed.record.expires_at <= now:
        state = "expired"
    elif operator.verify_assignment(row["signed_json"], fingerprint, root, now) is None:
        state = "invalid"
    else:
        state = "active"
    counts = conn.execute(
        """SELECT
            (SELECT count(*) FROM jobs WHERE signer = :key) AS jobs,
            (SELECT count(*) FROM cluster_tasks WHERE signer = :key) AS cluster_tasks,
            count(CASE WHEN action = 'operator' THEN 1 END) AS writes,
            count(CASE WHEN action = 'refused' THEN 1 END) AS refusals,
            count(CASE WHEN action = 'operator_read_complaint' THEN 1 END) AS complaint_reads
        FROM registry_log WHERE node = :key""",
        {"key": fingerprint},
    ).fetchone()
    last = conn.execute(
        "SELECT * FROM registry_log WHERE node = ? ORDER BY seq DESC LIMIT 1", (fingerprint,)
    ).fetchone()
    return Summary(
        fingerprint,
        scopes,
        row["issued_at"],
        row["expires_at"],
        state,
        fingerprint == own,
        sign.Scope.APPOINT in scopes,
        None if last is None else _entry(last),
        counts["jobs"],
        counts["cluster_tasks"],
        counts["writes"],
        counts["refusals"],
        counts["complaint_reads"],
    )


_LATEST = """
    SELECT *, row_number() OVER (PARTITION BY subject ORDER BY issued_at DESC, seq DESC) AS rank
    FROM records WHERE kind = 'assignment'
"""


def summaries(
    conn: sqlite3.Connection, root: Ed25519PublicKey, now: datetime, own: str | None
) -> tuple[Summary, ...]:
    with db.read_snapshot(conn):
        rows = conn.execute(f"SELECT * FROM ({_LATEST}) WHERE rank = 1 ORDER BY subject").fetchall()
        return tuple(_summary(conn, row, root, now, own) for row in rows)


def _authority(row: sqlite3.Row) -> Authority:
    record = sign.parse(row["signed_json"]).record
    if record.kind == sign.RecordKind.ASSIGNMENT:
        scopes = tuple(sorted(sign.Assignment.model_validate(record.body).scopes))
        return Authority(record.id, row["kind"], row["issued_at"], row["expires_at"], scopes)
    reason = sign.Revocation.model_validate(record.body).reason
    return Authority(record.id, row["kind"], row["issued_at"], row["expires_at"], reason=reason)


def _usage(conn: sqlite3.Connection, key: str, *, cluster: bool) -> tuple[Usage, ...]:
    if cluster:
        records = "SELECT id, status FROM cluster_tasks WHERE signer = :key"
        jobs = "SELECT jobs.id, owned.status FROM owned JOIN jobs ON jobs.cluster = owned.id"
        cluster_results = """
            SELECT owned.status, count(*) AS total FROM owned
            JOIN cluster_tasks ON cluster_tasks.id = owned.id
            JOIN cluster_records ON cluster_records.cluster = cluster_tasks.cluster
            WHERE cluster_records.kind = 'cluster_result' GROUP BY owned.status
        """
    else:
        records = "SELECT id, status FROM jobs WHERE signer = :key"
        jobs = "SELECT id, status FROM owned"
        cluster_results = "SELECT NULL AS status, 0 AS total WHERE 0"
    cte = f"WITH owned AS ({records}), scoped_jobs AS ({jobs}) "
    groups = conn.execute(
        cte
        + f"""
        , counts AS (SELECT status, count(*) AS total FROM owned GROUP BY status)
        , returns AS (
            SELECT scoped_jobs.status,
                   count(CASE WHEN kind = 'result' THEN 1 END) AS results,
                   count(CASE WHEN kind = 'verdict' THEN 1 END) AS verdicts
            FROM scoped_jobs JOIN results ON results.job = scoped_jobs.id
            GROUP BY scoped_jobs.status
        )
        , cluster_returns AS ({cluster_results})
        SELECT counts.status, counts.total, coalesce(returns.results, 0) AS results,
               coalesce(returns.verdicts, 0) AS verdicts,
               coalesce(cluster_returns.total, 0) AS cluster_results
        FROM counts LEFT JOIN returns USING (status) LEFT JOIN cluster_returns USING (status)
        ORDER BY counts.status
        """,
        {"key": key},
    ).fetchall()
    workers: dict[str, list[Worker]] = {}
    for row in conn.execute(
        cte
        + """
        SELECT scoped_jobs.status, leases.node, nodes.owner, count(*) AS leases
        FROM scoped_jobs JOIN leases ON leases.job = scoped_jobs.id
        JOIN nodes ON nodes.fingerprint = leases.node
        GROUP BY scoped_jobs.status, leases.node ORDER BY leases.node
        """,
        {"key": key},
    ):
        workers.setdefault(row["status"], []).append(
            Worker(row["node"], row["owner"], row["leases"])
        )
    return tuple(
        Usage(
            row["status"],
            row["total"],
            sum(node.leases for node in workers.get(row["status"], ())),
            row["results"],
            row["verdicts"],
            row["cluster_results"],
            tuple(workers.get(row["status"], ())),
        )
        for row in groups
    )


def detail(
    conn: sqlite3.Connection,
    root: Ed25519PublicKey,
    now: datetime,
    own: str | None,
    prefix: str,
    limit: int,
) -> Detail:
    if re.fullmatch(r"[0-9a-fA-F]{8,64}", prefix) is None:
        raise admin.AdminError(t("Use 8 to 64 hexadecimal digits of the operator fingerprint."))
    with db.read_snapshot(conn):
        matches = conn.execute(
            f"SELECT * FROM ({_LATEST}) WHERE rank = 1 AND subject GLOB ? LIMIT 2",
            (prefix.lower() + "*",),
        ).fetchall()
        if not matches:
            raise admin.AdminError(t("No operator matches {key}.", key=prefix))
        if len(matches) > 1:
            raise admin.AdminError(t("More than one operator matches {key}.", key=prefix))
        row = matches[0]
        key = row["subject"]
        records = conn.execute(
            "SELECT * FROM records WHERE subject = ? ORDER BY issued_at DESC, seq DESC", (key,)
        )
        journal = conn.execute(
            """WITH owned_tasks AS (SELECT id, cluster FROM cluster_tasks WHERE signer = :key),
                owned_jobs AS (
                    SELECT id FROM jobs
                    WHERE signer = :key OR cluster IN (SELECT id FROM owned_tasks)
                )
            SELECT * FROM registry_log
            WHERE node = :key
                OR json_extract(details_json, '$.job') IN (SELECT id FROM owned_jobs)
                OR json_extract(details_json, '$.task') IN (SELECT id FROM owned_tasks)
                OR (action = 'operator' AND json_extract(details_json, '$.id') IN (
                    SELECT id FROM owned_jobs UNION ALL SELECT id FROM owned_tasks
                ))
                OR json_extract(details_json, '$.cluster') IN (SELECT id FROM owned_tasks)
                OR json_extract(details_json, '$.cluster') IN (SELECT cluster FROM owned_tasks)
            ORDER BY seq DESC LIMIT :limit""",
            {"key": key, "limit": limit},
        )
        return Detail(
            _summary(conn, row, root, now, own),
            tuple(_authority(r) for r in records),
            _usage(conn, key, cluster=False),
            _usage(conn, key, cluster=True),
            tuple(_entry(r) for r in journal),
        )
