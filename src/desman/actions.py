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
from rich.text import Text

from desman import knowledge, providers, ui
from desman.config import ConfigError, open_config
from desman.core import db, operator
from desman.core.audit import AuditEntry, check_history, history
from desman.core.models import Subtask
from desman.core.report import (
    InboxItem,
    ModelSuggestion,
    TaskReport,
    inbox,
    short_description,
    task_report,
    task_summaries,
)
from desman.core.states import Actor, SubtaskStatus, TaskStatus, allowed_targets
from desman.core.stats import Stats
from desman.core.store import StaleStateError, Store
from desman.i18n import t, tn
from desman.llm import hub


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
        line = f"  [{sub.seq}] {sub.status:<11}  {short_description(sub.description)}"
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
        if sub.knowledge:
            click.echo("      " + t("knowledge: {paths}", paths=", ".join(sub.knowledge)))
    for text in suggestion_lines(report.suggestions):
        click.echo(text)


def suggestion_lines(suggestions: list[ModelSuggestion]) -> list[str]:
    """Models an executor suggested (ROADMAP 4.5b), with the command that adds one: nothing is
    downloaded until the person runs it (or says "yes" in the chat's window)."""
    if not suggestions:
        return []
    lines = [t("Suggested models (nothing is downloaded until you add one):")]
    for s in suggestions:
        lines.append(
            "  "
            + t(
                "[{seq}] {model} for {server}, {size}: {reason}",
                seq=s.seq,
                model=s.model,
                server=providers.find(s.server).name,
                size=hub.size_text(s.size),
                reason=one_line(s.reason, 200),
            )
        )
        if s.fits is False:
            lines.append(
                "    "
                + t("It does not fit in the free memory: it would run very slowly, or not at all.")
            )
        if s.gated:
            page = hub.Ref(s.model.split(":", 1)[0]).page
            lines.append(
                "    "
                + t(
                    "{model} is gated: accept its license at {url}, then add a Hugging Face "
                    "token with `desman keys add huggingface`.",
                    model=s.model,
                    url=page,
                )
            )
        lines.append(
            "    "
            + t("To use it: {command}", command=f"desman models add {s.model} --server {s.server}")
        )
    return lines


def list_tasks(
    store: Store,
    *,
    status: TaskStatus | None,
    limit: int,
    subtasks: bool = False,
    all: bool = False,
) -> list[TaskReport]:
    """A consistent task list, with full subtask data only when requested."""
    with db.read_snapshot(store.conn):
        summaries = task_summaries(store, status=status, limit=limit, all=all)
        return [task_report(store, summary.id) for summary in summaries] if subtasks else summaries


def print_summaries(summaries: list[TaskReport], subtasks: bool = False) -> None:
    for summary in summaries:
        click.echo(
            f"{summary.id}  {summary.status:<11}  {summary.done:>3}/{summary.total:<3}  "
            f"{summary.created_at}  {one_line(summary.goal, 60)}"
            + ("  " + t("Pinned") if summary.pinned else "")
            + ("  " + t("paused") if summary.paused_at else "")
        )
        if subtasks:
            for sub in summary.subtasks:
                click.echo(f"  [{sub.seq}] {sub.status}  {short_description(sub.description)}")


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


def print_inbox(items: list[InboxItem], hint: bool = True) -> None:
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
    if hint:
        click.echo(t("Decide with: desman retry|accept|fail <ref> [--note TEXT]"))


def choose_inbox(store: Store, session_id: str | None = None) -> dict[str, Any]:
    """Code-owned decision window, with optimistic versions from the displayed list."""
    waited = inbox(store)
    decisions: list[dict[str, Any]] = []
    try:
        while True:
            items = inbox(store)
            print_inbox(items, hint=False)
            if not items:
                break
            chosen = ui.select(
                t("Choose a subtask"),
                [
                    ui.Choice(
                        label=f"{i.ref}  {i.status}  {i.description}  [{i.goal}]",
                        value=i,
                        hint=i.reason or "",
                    )
                    for i in items
                ]
                + [ui.Choice(label=t("Leave it for now"), value=None)],
            )
            if chosen is None:
                break
            action = ui.select(
                t("Choose a decision"),
                [
                    ui.Choice(label=t("Retry"), value="retry"),
                    ui.Choice(label=t("Accept"), value="accept"),
                    ui.Choice(label=t("Fail"), value="fail"),
                    ui.Choice(label=t("Back"), value=None),
                ],
            )
            if action is None:
                continue
            note = ""
            if action == "retry":
                note = ui.text(t("Optional guidance for the next attempt"))
            elif action == "fail":
                note = ui.text(t("Why did it fail?"))
            try:
                config = open_config().load()
            except ConfigError:
                config = None
            note = providers.redact_keys(note, config)[0]
            target = {
                "retry": SubtaskStatus.READY,
                "accept": SubtaskStatus.DONE,
                "fail": SubtaskStatus.FAILED,
            }[action]
            try:
                pending = prepare_decision(store, chosen.ref, target, chosen.version)
                decision = apply_decision(
                    store,
                    pending,
                    note.strip()
                    or (
                        "failed by operator"
                        if action == "fail"
                        else f"{action} requested by operator"
                    ),
                    f"operator_{action}",
                )
            except click.ClickException as e:
                ui.error(e.format_message())
                continue
            print_decision(decision)
            data = decision.as_json()
            decisions.append(data)
            operator.record(
                store.conn,
                "operator_command",
                {"command": action, "args": {"subtask": chosen.ref}, "args_private": ["note"]},
                session_id=session_id,
            )
    except (ui.Cancelled, click.Abort):
        pass  # Decisions already applied remain in the returned result.
    return {"waiting": [i.model_dump(mode="json") for i in waited], "decisions": decisions}


def print_waiting(items: list[InboxItem], hint: str) -> None:
    """Subtasks that came to wait for the operator while they were not asking (ROADMAP 5.1):
    a worker's terminal and the chat show them as they arrive. `hint`: how to decide there."""
    if not items:
        return
    ui.warn(
        tn(
            len(items),
            "{n} subtask waits for your decision:",
            "{n} subtasks wait for your decision:",
        )
    )
    for item in items:
        ui.out(
            Text(f"  {item.ref}  ", style="key")
            + Text(f"{item.status}  {one_line(item.description, 50)}  [{one_line(item.goal, 40)}]")
        )
        if item.reason:
            ui.note(f"    {one_line(item.reason, 110)}")
    ui.note(hint)


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
    if action == "operator_fail":
        note = note.strip() or "failed by operator"
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
    click.echo(f"[{sub.seq}] {short_description(sub.description)}: {sub.status} -> {decision.to}")
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


# ── stats ────────────────────────────────────────────────────────────────────


def print_stats(stats: Stats, since: str) -> None:
    """`desman stats` for the person (ROADMAP 5.2); `since` as they gave it (e.g. 7d, all)."""
    everything = since == "all"
    ui.out(
        Text(
            t("Stats (all time)") if everything else t("Stats (last {since})", since=since),
            style="bold accent",
        )
    )
    if stats.empty:
        ui.note(t("Nothing has run yet.") if everything else t("Nothing ran in that time."))
        return
    if stats.tasks:
        ui.out(
            tn(
                sum(stats.tasks.values()),
                "{n} task: {statuses}.",
                "{n} tasks: {statuses}.",
                statuses=_counts(stats.tasks),
            )
        )
    sub = stats.subtasks
    if sub.total:
        ui.out(
            t(
                "Subtasks: {total}. Done: {done}, on the first try: {first} ({share}).",
                total=sub.total,
                done=sub.done,
                first=sub.first_try,
                share=_share(sub.first_try, sub.done),
            )
        )
        ui.out(
            t(
                "Retried: {retried}, escalated to you: {escalated}, split into parts: {split}.",
                retried=sub.retried,
                escalated=sub.escalated,
                split=sub.split,
            )
        )
    if stats.reviews:
        ui.out(t("Reviews: {verdicts}.", verdicts=_counts(stats.reviews)))
    if stats.failures:
        ui.out(t("Failures by kind: {kinds}.", kinds=_counts(stats.failures)))
    if stats.strategies:
        table = ui.table(t("strategy"), t("attempts"), t("accepted"), "%", title=t("Attempts"))
        for use in stats.strategies:
            table.add_row(
                use.strategy, str(use.started), str(use.accepted), _share(use.accepted, use.started)
            )
        ui.out("")
        ui.out(table)
    if stats.roles:
        table = ui.table(
            t("role"),
            t("calls"),
            t("failed"),
            t("tokens in"),
            t("tokens out"),
            title=t("Model calls by role"),
        )
        for role in stats.roles:
            table.add_row(
                role.role,
                str(role.calls),
                Text(str(role.failed), style="bad" if role.failed else "muted"),
                f"{role.prompt_tokens:,}",
                f"{role.completion_tokens:,}",
            )
        ui.out("")
        ui.out(table)
        table = ui.table(
            t("provider"),
            t("model"),
            t("calls"),
            t("failed"),
            t("limited"),
            t("tokens in"),
            t("tokens out"),
            t("average time"),
            title=t("Model calls by model"),
        )
        for m in stats.models:
            table.add_row(
                m.provider,
                m.model,
                str(m.calls),
                Text(str(m.failed), style="bad" if m.failed else "muted"),
                Text(str(m.limited), style="warn" if m.limited else "muted"),
                f"{m.prompt_tokens:,}",
                f"{m.completion_tokens:,}",
                f"{m.latency_ms / 1000:.1f} s",
            )
        ui.out("")
        ui.out(table)


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {n}" for name, n in counts.items())


def _share(part: int, whole: int) -> str:
    return f"{round(100 * part / whole)}%" if whole else "-"


def task_change(
    store: Store, operation: str, task: str | None = None, *, targets: list[str] | None = None
) -> dict[str, Any]:
    """Shared operator task controls; bulk operations recheck pins at application time."""
    try:
        if (
            operation not in {"pin", "unpin", "pause", "hide"}
            or task is None
            and operation in {"pin", "unpin"}
        ):
            raise ValueError("unknown task operation")
        if task is None:
            only = set(targets) if targets is not None else None
            count = (
                store.hide_finished(only=only)
                if operation == "hide"
                else store.pause_unfinished(only=only)
            )
            return {"operation": operation, "count": count}
        task_id = find_task(store, task)
        if operation == "pin":
            changed = store.pin_task(task_id)
        elif operation == "unpin":
            changed = store.pin_task(task_id, False)
        elif operation == "hide":
            changed = store.hide_task(task_id)
        elif operation == "pause":
            changed = store.pause_task(task_id)
        else:
            raise ValueError("unknown task operation")
    except ValueError as e:
        raise click.ClickException(t("Cannot change this task: {error}", error=str(e))) from e
    return {"task": task_id, "operation": operation, "changed": changed}


# ── Continuing a task (ADR 0010) ─────────────────────────────────────────────


@dataclass(frozen=True)
class Continuation:
    """What `task extend` / `task redo` will do, checked before the person is asked."""

    task_id: str
    ask: str
    redo: Subtask | None = None


def prepare_continuation(
    store: Store, task: str | None, ask: str, redo: str | None
) -> Continuation:
    """The task to continue and how; problems are the person's to read (ClickException)."""
    ask = " ".join(ask.split())
    if not ask:
        raise click.ClickException(t("Say what to add or why the step is redone."))
    sub = None
    if redo is not None:
        try:
            sub = store.find_subtask(redo)
        except LookupError as e:
            raise click.ClickException(str(e)) from e
        if sub.status is not SubtaskStatus.DONE or sub.parent_id is not None:
            raise click.ClickException(
                t("Only a done step is redone; {ref} is {status}.", ref=redo, status=sub.status)
            )
        task_id = sub.task_id
    else:
        task_id = find_task(store, task or "")
    row = store.find_task(task_id)
    if row["origin"] != "local":
        raise click.ClickException(t("A job lent by desman-global is not continued here."))
    if store.task_status(task_id) is TaskStatus.CANCELLED:
        raise click.ClickException(t("A cancelled task is not continued."))
    last = [s for s in store.list_subtasks(task_id) if s.parent_id is None][-1]
    if last.status is not SubtaskStatus.DONE:  # new work starts from the result
        raise click.ClickException(
            t(
                "Task {ref} is continued once its result is done; decide what waits for you "
                "first (retry, accept or fail).",
                ref=store.short_id(task_id),
            )
        )
    return Continuation(task_id, ask, sub)


def continue_task(
    store: Store, llm: Any, pending: Continuation, *, limit: int | None, by: str
) -> tuple[list[str], str | None]:
    """Plan the new subtasks with the model (the template if it fails) and add them to the
    task. `by`: who asked ("person" or "proposal"). Returns the new ids and why the model's
    plan was not used, if it was not."""
    from desman.planner import continuation, must_run

    task = store.find_task(pending.task_id)
    subtasks = [s for s in store.list_subtasks(pending.task_id) if s.parent_id is None]
    planned = continuation(
        llm,
        task["goal"],
        [(s.seq, s.description, s.status.value) for s in subtasks],
        pending.ask,
        redo=pending.redo.seq if pending.redo else None,
        priority=task["priority"],
        limit=limit,
        tree=knowledge.task_tree(),
    )
    last = subtasks[-1].id
    basis = [last] + ([pending.redo.id] if pending.redo and pending.redo.id != last else [])
    # A result that had to rest on a real run still has to (`planner.must_run`).
    spec = must_run(planned.spec) if subtasks[-1].runs_code else planned.spec
    try:
        ids = store.extend(
            pending.task_id,
            spec.subtasks,
            basis,
            details={
                **planned.details(),
                "by": by,
                ("redo" if pending.redo else "add"): pending.redo.id
                if pending.redo
                else pending.ask,
                **({"why": pending.ask} if pending.redo else {}),
            },
        )
    except (ValueError, LookupError) as e:  # cancelled or changed while it was planned
        raise click.ClickException(
            t("Task {ref} was not continued: {error}", ref=store.short_id(pending.task_id), error=e)
        ) from e
    return ids, planned.error
