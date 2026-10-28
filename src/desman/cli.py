"""CLI for DESMAN(P)."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from pydantic import ValidationError

from desman import __version__, i18n, providers, ui
from desman.agents import llm_executor, llm_reviewer
from desman.config import ConfigError, open_config
from desman.config_cli import config, keys, load_config, run_wizard, usage
from desman.core import calls, db
from desman.core.audit import AuditEntry, check_history, history
from desman.core.coordinator import Coordinator, RunResult
from desman.core.models import Priority, SubtaskSpec, TaskSpec
from desman.core.owner import current_owner, owner_is_gone
from desman.core.report import TaskReport, inbox, task_report, task_summaries
from desman.core.states import Actor, SubtaskStatus, TaskStatus, allowed_targets
from desman.core.store import DEFAULT_LEASE_SECONDS, StaleStateError, Store
from desman.i18n import t, tn
from desman.info import print_project_info
from desman.llm import router
from desman.llm.base import LLM
from desman.llm.fake import FakeLLM, ScriptExhausted
from desman.llm.openai_compat import OpenAICompatible

F = TypeVar("F", bound=Callable[..., Any])

# Exit codes of `run`, `resume` and `status`, so scripts can tell outcomes apart.
EXIT_DONE = 0
EXIT_NEEDS_HUMAN = 3
EXIT_PAUSED = 4
EXIT_STOPPED = 5  # failed or cancelled
EXIT_ACTIVE = 6  # still in progress (someone holds a lease, or nobody has run it)
TEMPLATE_CRITERIA = "The result fully achieves the stated goal."


def db_option(command: F) -> F:
    return click.option(
        "--db",
        "db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Database file (default: $DESMAN_DATA_DIR/desman.db or ~/.local/share/desman/desman.db).",
    )(command)


def open_existing(db_path: Path | None) -> Store:
    """The store for commands that read tasks: they never create an empty database."""
    path = db_path or db.default_db_path()
    if not path.exists():
        raise click.ClickException(t("no database at {path} (nothing has run yet?)", path=path))
    try:
        return Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e


class DesmanGroup(click.Group):
    """The root command: switches to the configured language before Click reads the
    arguments, so help and Click's own errors are translated too."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        i18n.set_language(_configured_language())
        i18n.translate_commands(self)
        return super().main(*args, **kwargs)


def _configured_language() -> str:
    """The language from the config; English if there is none or it does not load (the
    command itself then reports the broken file)."""
    try:
        return open_config().load().language
    except (ConfigError, OSError):
        return "en"


def _print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    # Not click.version_option: it fixes its message at import, before the language is known.
    if value and not ctx.resilient_parsing:
        click.echo(t("{prog}, version {version}", prog="desman", version=__version__))
        ctx.exit()


@click.group(cls=DesmanGroup, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_print_version,
    help="Show the version and exit.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)"""
    if ctx.invoked_subcommand is None:
        # Bare `desman` is the interactive entry point: configure on first run, then show info.
        config_manager = load_config()
        if not config_manager.is_configured():
            run_wizard(config_manager)
        print_project_info(config_manager)


@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    if ctx.parent:
        click.echo(ctx.parent.get_help())


cli.add_command(config)
cli.add_command(keys)
cli.add_command(usage)


@cli.command("init-db")
@db_option
def init_db(db_path: Path | None) -> None:
    """Create the database or apply pending migrations. Safe to re-run."""
    path = db_path or db.default_db_path()
    conn = db.connect(path)
    try:
        applied = db.migrate(conn)
        db.verify_schema(conn)
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    finally:
        conn.close()
    click.echo(
        tn(
            applied,
            "{path}: schema v{version} ({n} migration applied)",
            "{path}: schema v{version} ({n} migrations applied)",
            path=path,
            version=db.SCHEMA_VERSION,
        )
    )


@cli.command()
@click.argument("task")
@db_option
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def audit(task: str, db_path: Path | None, as_json: bool) -> None:
    """Show the full history of TASK (its id or a unique prefix of it).

    Every status change, reviewer verdict, checkpoint and discarded piece of work, in order.
    Exits with 1 if the history does not account for the current state.
    """
    store = open_existing(db_path)
    try:
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
    finally:
        store.conn.close()

    if as_json:
        report = {
            "task": {"id": task_id, "goal": row["goal"], "status": status.value},
            "entries": [e.model_dump(mode="json") for e in entries],
            "problems": problems,
        }
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        click.echo(t("Task {id}  {status}", id=task_id, status=status.value))
        click.echo(t("Goal: {goal}", goal=row["goal"]))
        for entry in entries:
            click.echo(_format_entry(entry, labels))
    if problems:
        click.echo(t("History is incomplete:"), err=True)
        for problem in problems:
            click.echo(f"  {problem}", err=True)
        raise SystemExit(1)


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


# ── Task commands ────────────────────────────────────────────────────────────


def json_option(command: F) -> F:
    return click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")(command)


def run_options(command: F) -> F:
    command = click.option(
        "--model",
        default=None,
        help="Model to use (default: the provider's free-tier default, see `desman keys check`).",
    )(command)
    command = click.option(
        "--provider",
        "provider_name",
        default=None,
        help="Provider to use, e.g. groq (default: the first configured one in the preference "
        "order). All its keys are used in turn.",
    )(command)
    command = click.option(
        "--fake-llm",
        "fake_script",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Drive the agents with a scripted fake provider (JSON script, see desman.llm.fake). "
        "Scripted latency and timeouts take real time.",
    )(command)
    command = click.option(
        "--wait",
        is_flag=True,
        help="Sleep through provider pauses instead of returning (exit code 4) when only "
        "paused work is left.",
    )(command)
    command = click.option(
        "--lease-seconds",
        type=click.IntRange(min=1),
        default=DEFAULT_LEASE_SECONDS,
        show_default=True,
        help="How long a claimed subtask stays ours without a heartbeat; after a crash, "
        "`resume` waits at most this long.",
    )(command)
    return command


@cli.command()
@click.argument("goal", required=False)
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A task graph as JSON (TaskSpec: goal, priority, subtasks with depends_on).",
)
@click.option(
    "--priority",
    type=click.Choice([p.name.lower() for p in Priority]),
    default=None,
    help="Overrides the priority (default: normal, or the spec's).",
)
@run_options
@db_option
@json_option
def run(
    goal: str | None,
    spec_path: Path | None,
    priority: str | None,
    fake_script: Path | None,
    provider_name: str | None,
    model: str | None,
    wait: bool,
    lease_seconds: int,
    db_path: Path | None,
    as_json: bool,
) -> None:
    """Create a task for GOAL (or from --spec) and work on it.

    Without --spec the plan is a single subtask: the goal itself (the LLM planner is ROADMAP
    2.7). Exit codes: 0 done, 3 waiting for the operator, 4 paused (provider limit),
    5 failed or cancelled, 6 still in progress elsewhere.
    """
    if (goal is None) == (spec_path is None):
        raise click.UsageError(t("give either GOAL or --spec, not both"))
    spec = _load_spec(goal, spec_path)
    if priority is not None:
        spec = spec.model_copy(update={"priority": int(Priority[priority.upper()])})
    llm = _provider(fake_script, provider_name, model, as_json)  # fail fast, before any task
    path = db_path or db.default_db_path()
    try:
        store = Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    _record_calls(llm, store)
    try:
        task_id = store.create_task(spec)
        if not as_json:
            click.echo(
                tn(
                    len(spec.subtasks),
                    "Task {id} created ({n} subtask)",
                    "Task {id} created ({n} subtasks)",
                    id=task_id,
                )
            )
            sys.stdout.flush()  # visible before any work starts, even through a pipe
        _drive(store, llm, task_id, wait, lease_seconds, as_json)
    finally:
        store.conn.close()


@cli.command()
@click.argument("task")
@run_options
@db_option
@json_option
def resume(
    task: str,
    fake_script: Path | None,
    provider_name: str | None,
    model: str | None,
    wait: bool,
    lease_seconds: int,
    db_path: Path | None,
    as_json: bool,
) -> None:
    """Continue TASK where it stopped: after a crash, a pause or the operator's decisions.

    Work a crashed run left behind is taken back at once if that run was on this machine,
    otherwise when its lease expires. Finished subtasks are never redone. Same exit codes as
    `run`.
    """
    llm = _provider(fake_script, provider_name, model, as_json)
    store = open_existing(db_path)
    _record_calls(llm, store)
    try:
        task_id = _find(store, task)
        _drive(store, llm, task_id, wait, lease_seconds, as_json)
    finally:
        store.conn.close()


@cli.command()
@click.argument("task")
@db_option
@json_option
def status(task: str, db_path: Path | None, as_json: bool) -> None:
    """Show TASK and its subtasks. Same exit codes as `run`."""
    store = open_existing(db_path)
    try:
        report = task_report(store, _find(store, task))
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    raise SystemExit(_exit_code(report.status, paused=_has_paused(report)))


@cli.command("list")
@click.option(
    "--status",
    "status_filter",
    type=click.Choice([s.value for s in TaskStatus]),
    default=None,
    help="Only tasks with this status.",
)
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True)
@db_option
@json_option
def list_tasks(status_filter: str | None, limit: int, db_path: Path | None, as_json: bool) -> None:
    """List tasks, newest first."""
    store = open_existing(db_path)
    try:
        wanted = TaskStatus(status_filter) if status_filter else None
        summaries = task_summaries(store, status=wanted, limit=limit)
    finally:
        store.conn.close()
    if as_json:
        rows = [s.model_dump(mode="json", exclude={"subtasks"}) for s in summaries]
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for summary in summaries:
        click.echo(
            f"{summary.id}  {summary.status:<11}  {summary.done:>3}/{summary.total:<3}  "
            f"{summary.created_at}  {_one_line(summary.goal, 60)}"
        )


def _load_spec(goal: str | None, spec_path: Path | None) -> TaskSpec:
    if spec_path is None:
        assert goal is not None
        if not goal.strip():
            raise click.UsageError(t("GOAL is empty"))
        return TaskSpec(
            goal=goal,
            subtasks=[SubtaskSpec(description=goal, acceptance_criteria=TEMPLATE_CRITERIA)],
        )
    try:
        return TaskSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as e:
        raise click.ClickException(
            t("{path}: not a valid task spec:\n{error}", path=spec_path, error=e)
        ) from e


def _provider(
    fake_script: Path | None, provider_name: str | None, model: str | None, as_json: bool
) -> LLM:
    if fake_script is not None:
        if provider_name is not None or model is not None:
            raise click.UsageError(t("--fake-llm replaces the provider: drop --provider/--model"))
        try:
            return FakeLLM.from_file(fake_script, sleep=time.sleep)  # latency is real
        except (ValidationError, ValueError) as e:
            raise click.ClickException(
                t("{path}: not a valid fake LLM script:\n{error}", path=fake_script, error=e)
            ) from e
    config = load_config().load()
    try:
        llm, choice = router.build(config, provider_name, model)
    except router.NoProvider as e:
        raise click.ClickException(str(e)) from e
    if not as_json:
        ui.note(
            tn(
                len(llm.keys),
                "Using {provider} · {model} ({n} key)",
                "Using {provider} · {model} ({n} keys)",
                provider=choice.provider.name,
                model=choice.model,
            )
        )
        if choice.provider.paid:
            ui.warn(
                t("{name} is a paid provider: every call is billed.", name=choice.provider.name)
            )
            ui.warn(providers.untested_note())
    return llm


def _record_calls(llm: LLM, store: Store) -> None:
    """Every real model call goes to model_calls (`desman usage`)."""
    if isinstance(llm, OpenAICompatible):
        llm.on_call = calls.recorder(store.conn)


def _find(store: Store, task: str) -> str:
    try:
        task_id: str = store.find_task(task)["id"]
    except LookupError as e:
        raise click.ClickException(str(e)) from e
    return task_id


def _drive(
    store: Store, llm: LLM, task_id: str, wait: bool, lease_seconds: int, as_json: bool
) -> None:
    coordinator = Coordinator(
        store,
        llm_executor(llm),
        llm_reviewer(llm),
        owner=current_owner(),
        lease_seconds=lease_seconds,
        abandoned=owner_is_gone,  # a run killed on this machine is taken over at once
    )
    try:
        result = coordinator.run(task_id, wait_for_pauses=wait)
    except ScriptExhausted as e:
        raise click.ClickException(t("fake LLM script ran out: {error}", error=e)) from e
    report = task_report(store, task_id)
    short = store.short_id(task_id)
    if as_json:
        out = {"run": _run_summary(result), "task": report.model_dump(mode="json")}
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
        if result.paused_until:
            click.echo(
                t(
                    "Paused by a provider limit until {until}. Continue with: desman resume {ref}",
                    until=result.paused_until,
                    ref=short,
                )
            )
    raise SystemExit(_exit_code(report.status, paused=result.paused_until is not None))


def _run_summary(result: RunResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "ticks": result.ticks,
        "paused_until": result.paused_until,
    }


def _exit_code(status: TaskStatus, *, paused: bool) -> int:
    if status is TaskStatus.DONE:
        return EXIT_DONE
    if status is TaskStatus.NEEDS_HUMAN:
        return EXIT_NEEDS_HUMAN
    if status is TaskStatus.ACTIVE:
        return EXIT_PAUSED if paused else EXIT_ACTIVE
    return EXIT_STOPPED


def _has_paused(report: TaskReport) -> bool:
    """Only paused work is left: nothing can move before a pause ends."""
    moving = {"ready", "running", "review"}
    statuses = {s.status.value for s in report.subtasks}
    return "paused" in statuses and not statuses & moving


def _print_report(report: TaskReport) -> None:
    click.echo(
        t(
            "Task {id}  {status}  ({done}/{total} done)",
            id=report.id,
            status=report.status,
            done=report.done,
            total=report.total,
        )
    )
    click.echo(t("Goal: {goal}", goal=_one_line(report.goal, 200)))
    for sub in report.subtasks:
        line = f"  [{sub.seq}] {sub.status:<11}  {_one_line(sub.description, 60)}"
        notes = []
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
            notes.append(_one_line(sub.reason, 100))
        click.echo(line + (f"  ({'; '.join(notes)})" if notes else ""))


def _one_line(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# ── Operator decisions (a minimal subset of ROADMAP 5.1) ─────────────────────


@cli.command("inbox")
@db_option
@json_option
def inbox_command(db_path: Path | None, as_json: bool) -> None:
    """List subtasks waiting for the operator (needs_human, failed), with the reason."""
    store = open_existing(db_path)
    try:
        items = inbox(store)
    finally:
        store.conn.close()
    if as_json:
        click.echo(
            json.dumps([i.model_dump(mode="json") for i in items], ensure_ascii=False, indent=2)
        )
        return
    if not items:
        click.echo(t("Nothing is waiting for you."))
        return
    for item in items:
        click.echo(
            f"{item.ref:<12} {item.status:<11}  {_one_line(item.description, 50)}"
            f"  [{_one_line(item.goal, 40)}]"
        )
        if item.reason:
            click.echo(f"{'':<12} {_one_line(item.reason, 110)}")
    click.echo(t("Decide with: desman retry|accept|fail <ref> [--note TEXT]"))


def if_version_option(command: F) -> F:
    return click.option(
        "--if-version",
        type=int,
        default=None,
        help="Act only if the subtask has not changed since `inbox --json` showed this version.",
    )(command)


@cli.command()
@click.argument("subtask")
@click.option(
    "--note", default=None, help="Guidance for the next attempt (passed to the executor)."
)
@if_version_option
@db_option
@json_option
def retry(
    subtask: str, note: str | None, if_version: int | None, db_path: Path | None, as_json: bool
) -> None:
    """Send SUBTASK back to work with a fresh attempt budget.

    Works on a subtask waiting for you or one marked failed (its dependents are unblocked).
    Continue the task with `desman resume`.

    SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.
    """
    reason = (note or "").strip() or "retry requested by operator"
    _decide(subtask, SubtaskStatus.READY, reason, "operator_retry", if_version, db_path, as_json)


@cli.command()
@click.argument("subtask")
@click.option("--note", default=None, help="Why it is acceptable.")
@if_version_option
@db_option
@json_option
def accept(
    subtask: str, note: str | None, if_version: int | None, db_path: Path | None, as_json: bool
) -> None:
    """Mark SUBTASK done as it is (its latest result is kept only if it was under review).

    SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.
    """
    reason = (note or "").strip() or "accepted by operator"
    _decide(subtask, SubtaskStatus.DONE, reason, "operator_accept", if_version, db_path, as_json)


@cli.command()
@click.argument("subtask")
@click.option("--note", required=True, help="Why it failed (recorded in the audit log).")
@if_version_option
@db_option
@json_option
def fail(
    subtask: str, note: str, if_version: int | None, db_path: Path | None, as_json: bool
) -> None:
    """Give up on SUBTASK: it becomes failed and everything depending on it is blocked.

    `desman retry` brings it back later.

    SUBTASK is "<task>/<number>" as `desman status` and `inbox` show it, or its id.
    """
    if not note.strip():
        raise click.BadParameter(t("say why it failed"), param_hint="--note")
    _decide(
        subtask, SubtaskStatus.FAILED, note.strip(), "operator_fail", if_version, db_path, as_json
    )


@cli.command()
@click.argument("task")
@click.option("--note", default="cancelled by operator", show_default=True)
@db_option
@json_option
def cancel(task: str, note: str, db_path: Path | None, as_json: bool) -> None:
    """Cancel TASK: every subtask not finished yet stops. A running one notices at its next
    heartbeat."""
    store = open_existing(db_path)
    try:
        task_id = _find(store, task)
        try:
            cancelled = store.cancel_task(task_id, reason=note)
        except StaleStateError as e:  # a run changed it meanwhile: nothing was cancelled
            raise click.ClickException(
                t("it changed meanwhile, try again: {error}", error=e)
            ) from e
        status = store.task_status(task_id)
    finally:
        store.conn.close()
    if as_json:
        out = {"task_id": task_id, "status": status.value, "cancelled": [s.id for s in cancelled]}
        click.echo(json.dumps(out, indent=2))
    else:
        click.echo(
            tn(
                len(cancelled),
                "Task {id}: {n} subtask cancelled, task is {status}",
                "Task {id}: {n} subtasks cancelled, task is {status}",
                id=task_id,
                status=status,
            )
        )


def _decide(
    ref: str,
    target: SubtaskStatus,
    note: str,
    action: str,
    if_version: int | None,
    db_path: Path | None,
    as_json: bool,
) -> None:
    store = open_existing(db_path)
    try:
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
        try:
            # Not just "still in that status": unchanged since we looked (a retry that
            # re-escalated meanwhile is a newer situation the operator has not seen).
            moved = store.transition(
                sub.id,
                sub.status,
                target,
                Actor.HUMAN,
                reason=note,
                action=action,
                expected_version=version,
            )
        except StaleStateError as e:
            raise click.ClickException(
                t("it changed meanwhile, look again: {error}", error=e)
            ) from e
        status = store.task_status(sub.task_id)
        short = store.short_id(sub.task_id)
    finally:
        store.conn.close()
    if as_json:
        out = {
            "subtask_id": sub.id,
            "task_id": sub.task_id,
            "from": sub.status.value,
            "to": moved.status.value,
            "task_status": status.value,
        }
        click.echo(json.dumps(out, indent=2))
        return
    click.echo(f"[{sub.seq}] {_one_line(sub.description, 60)}: {sub.status} -> {moved.status}")
    if status is TaskStatus.ACTIVE:
        click.echo(t("Task is active again. Continue with: desman resume {ref}", ref=short))
    else:
        click.echo(t("Task is {status}.", status=status))
