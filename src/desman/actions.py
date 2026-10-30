"""The bodies of DESMAN's task commands, shared by the CLI (`desman.cli`) and the chat's catalog
of commands (`desman.commands`, ADR 0003 §3).

Each step that reads or changes something is a function returning data; what the person sees
is printed by a separate function. Problems are `click.ClickException`s with a message for the
person: the CLI prints them and exits, the chat shows them and tells the interlocutor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from desman.core import db
from desman.core.audit import AuditEntry, check_history, history
from desman.core.models import Subtask
from desman.core.report import InboxItem, TaskReport
from desman.core.states import Actor, SubtaskStatus, TaskStatus, allowed_targets
from desman.core.store import StaleStateError, Store
from desman.i18n import t, tn


def open_existing(db_path: Path | None) -> Store:
    """The store for commands that read tasks: they never create an empty database."""
    path = db_path or db.default_db_path()
    if not path.exists():
        raise click.ClickException(t("no database at {path} (nothing has run yet?)", path=path))
    try:
        return Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e


def find_task(store: Store, task: str) -> str:
    try:
        task_id: str = store.find_task(task)["id"]
    except LookupError as e:
        raise click.ClickException(str(e)) from e
    return task_id


def init_database(path: Path) -> int:
    """Create the database at `path` or bring it up to date; how many migrations ran."""
    conn = db.connect(path)
    try:
        applied = db.migrate(conn)
        db.verify_schema(conn)
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    finally:
        conn.close()
    return applied


def print_init(path: Path, applied: int) -> None:
    click.echo(
        tn(
            applied,
            "{path}: schema v{version} ({n} migration applied)",
            "{path}: schema v{version} ({n} migrations applied)",
            path=path,
            version=db.SCHEMA_VERSION,
        )
    )


# ── status, list ─────────────────────────────────────────────────────────────


def print_report(report: TaskReport) -> None:
    click.echo(
        t(
            "Task {id}  {status}  ({done}/{total} done)",
            id=report.id,
            status=report.status,
            done=report.done,
            total=report.total,
        )
    )
    click.echo(t("Goal: {goal}", goal=one_line(report.goal, 200)))
    for sub in report.subtasks:
        line = f"  [{sub.seq}] {sub.status:<11}  {one_line(sub.description, 60)}"
        notes = []
        if sub.part_of is not None:
            notes.append(t("part of {seq}", seq=sub.part_of))
        if sub.depends_on:
            notes.append(t("after {deps}", deps=", ".join(str(d) for d in sub.depends_on)))
        if sub.attempts.quality or sub.attempts.transient:
            notes.append(
                t(
                    "attempts q{quality}/t{transient}",
                    quality=sub.attempts.quality,
                    transient=sub.attempts.transient,
                )
            )
        if sub.lease_owner:
            notes.append(
                t("held by {owner} until {until}", owner=sub.lease_owner, until=sub.lease_until)
            )
        if sub.paused_until:
            notes.append(t("paused until {until}", until=sub.paused_until))
        if sub.reason and sub.status.value not in ("done", "running", "pending"):
            notes.append(one_line(sub.reason, 100))
        click.echo(line + (f"  ({'; '.join(notes)})" if notes else ""))


def print_summaries(summaries: list[TaskReport]) -> None:
    for summary in summaries:
        click.echo(
            f"{summary.id}  {summary.status:<11}  {summary.done:>3}/{summary.total:<3}  "
            f"{summary.created_at}  {one_line(summary.goal, 60)}"
        )


def one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ── audit ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskHistory:
    task_id: str
    goal: str
    status: TaskStatus
    entries: list[AuditEntry]
    problems: list[str]
    labels: dict[str, str]

    def as_json(self) -> dict[str, Any]:
        return {
            "task": {"id": self.task_id, "goal": self.goal, "status": self.status.value},
            "entries": [e.model_dump(mode="json") for e in self.entries],
            "problems": self.problems,
        }


def task_history(store: Store, task: str) -> TaskHistory:
    try:
        row = store.find_task(task)
    except LookupError as e:
        raise click.ClickException(str(e)) from e
    task_id = row["id"]
    with db.read_snapshot(store.conn):  # a worker may be committing meanwhile
        entries = [_redacted(e) for e in history(store.conn, task_id)]
        problems = check_history(store.conn, task_id)
        status = store.task_status(task_id)
        labels = {s.id: f"[{s.seq}] {s.description}" for s in store.list_subtasks(task_id)}
    return TaskHistory(task_id, row["goal"], status, entries, problems, labels)


def print_history(found: TaskHistory) -> None:
    click.echo(t("Task {id}  {status}", id=found.task_id, status=found.status.value))
    click.echo(t("Goal: {goal}", goal=found.goal))
    for entry in found.entries:
        click.echo(_format_entry(entry, found.labels))
    if found.problems:
        click.echo(t("History is incomplete:"), err=True)
        for problem in found.problems:
            click.echo(f"  {problem}", err=True)


def _redacted(entry: AuditEntry) -> AuditEntry:
    """Lease tokens let their holder act as the executor: show only enough to tell them apart."""
    lease = entry.details.get("lease")
    if not isinstance(lease, str):
        return entry
    return entry.model_copy(update={"details": {**entry.details, "lease": lease[:8] + "…"}})


def _format_entry(entry: AuditEntry, labels: dict[str, str]) -> str:
    label = labels.get(entry.subtask_id, entry.subtask_id) if entry.subtask_id else "task"
    label = " ".join(label.split())  # descriptions come from a planner and may span lines
    change = ""
    if entry.to_status is not None:
        change = f"{entry.from_status} -> {entry.to_status}"
    details = "  ".join(
        f"{key}={_short(value)}" for key, value in entry.details.items() if value not in ("", None)
    )
    return (
        f"{entry.seq:>5}  {entry.ts}  {label[:28]:<28}  {entry.actor:<9}  "
        f"{entry.action:<16}  {change:<22}  {details}".rstrip()
    )


def _short(value: Any, limit: int = 80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())  # one line per entry, whatever the content
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ── inbox, operator decisions ────────────────────────────────────────────────


def print_inbox(items: list[InboxItem]) -> None:
    if not items:
        click.echo(t("Nothing is waiting for you."))
        return
    for item in items:
        click.echo(
            f"{item.ref:<12} {item.status:<11}  {one_line(item.description, 50)}"
            f"  [{one_line(item.goal, 40)}]"
        )
        if item.reason:
            click.echo(f"{'':<12} {one_line(item.reason, 110)}")
    click.echo(t("Decide with: desman retry|accept|fail <ref> [--note TEXT]"))


@dataclass(frozen=True)
class Pending:
    """A decision on a subtask, checked but not made: the subtask as the operator saw it."""

    sub: Subtask
    version: int
    target: SubtaskStatus


@dataclass(frozen=True)
class Decision:
    sub: Subtask
    to: SubtaskStatus
    task_status: TaskStatus
    short: str

    def as_json(self) -> dict[str, Any]:
        return {
            "subtask_id": self.sub.id,
            "task_id": self.sub.task_id,
            "from": self.sub.status.value,
            "to": self.to.value,
            "task_status": self.task_status.value,
        }


def prepare_decision(
    store: Store, ref: str, target: SubtaskStatus, if_version: int | None = None
) -> Pending:
    try:
        with db.read_snapshot(store.conn):  # the status and its version, read together
            sub = store.find_subtask(ref)
            version = store.status_version(sub.id)
    except LookupError as e:
        raise click.ClickException(str(e)) from e
    if if_version is not None and if_version != version:
        raise click.ClickException(
            t(
                "[{seq}] changed since version {seen} (now {version}, {status}): "
                "look again with `desman inbox`",
                seq=sub.seq,
                seen=if_version,
                version=version,
                status=sub.status,
            )
        )
    if target not in allowed_targets(sub.status, Actor.HUMAN):
        options = sorted(s.value for s in allowed_targets(sub.status, Actor.HUMAN))
        raise click.ClickException(
            t(
                "[{seq}] is {status}: the operator can move it to {options}, not {target}",
                seq=sub.seq,
                status=sub.status,
                options=", ".join(options) if options else t("nothing"),
                target=target,
            )
        )
    return Pending(sub, version, target)


def apply_decision(store: Store, pending: Pending, note: str, action: str) -> Decision:
    sub = pending.sub
    try:
        # Not just "still in that status": unchanged since we looked (a retry that
        # re-escalated meanwhile is a newer situation the operator has not seen).
        moved = store.transition(
            sub.id,
            sub.status,
            pending.target,
            Actor.HUMAN,
            reason=note,
            action=action,
            expected_version=pending.version,
        )
    except StaleStateError as e:
        raise click.ClickException(t("it changed meanwhile, look again: {error}", error=e)) from e
    return Decision(sub, moved.status, store.task_status(sub.task_id), store.short_id(sub.task_id))


def print_decision(decision: Decision) -> None:
    sub = decision.sub
    click.echo(f"[{sub.seq}] {one_line(sub.description, 60)}: {sub.status} -> {decision.to}")
    if decision.task_status is TaskStatus.ACTIVE:
        click.echo(
            t("Task is active again. Continue with: desman resume {ref}", ref=decision.short)
        )
    else:
        click.echo(t("Task is {status}.", status=decision.task_status))


# ── cancel ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Cancelled:
    task_id: str
    status: TaskStatus
    cancelled: list[str]

    def as_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "status": self.status.value, "cancelled": self.cancelled}


def cancel_task(store: Store, task_id: str, note: str) -> Cancelled:
    try:
        cancelled = store.cancel_task(task_id, reason=note)
    except StaleStateError as e:  # a run changed it meanwhile: nothing was cancelled
        raise click.ClickException(t("it changed meanwhile, try again: {error}", error=e)) from e
    return Cancelled(task_id, store.task_status(task_id), [s.id for s in cancelled])


def print_cancelled(result: Cancelled) -> None:
    click.echo(
        tn(
            len(result.cancelled),
            "Task {id}: {n} subtask cancelled, task is {status}",
            "Task {id}: {n} subtasks cancelled, task is {status}",
            id=result.task_id,
            status=result.status,
        )
    )
