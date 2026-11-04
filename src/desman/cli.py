"""CLI for DESMAN(P).

G.7b reports independent proposal check hand-ins alongside job events.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import click
from pydantic import ValidationError

from desman import __version__, actions, i18n, proposals_cli, providers, runs, ui
from desman.actions import find_task, open_existing, print_report
from desman.chat import Chat
from desman.config import Config, ConfigError, Tuning, open_config
from desman.config_cli import (
    behaviour,
    config,
    keys,
    language,
    load_config,
    parse_window,
    permissions,
    personalize,
    run_wizard,
    usage,
)
from desman.core import calls, db
from desman.core.coordinator import DEFAULT_POLL_SECONDS, Coordinator, RunResult, WorkResult
from desman.core.models import Kind, Origin, Priority, TaskSpec
from desman.core.policy import Access
from desman.core.report import InboxWatch, TaskReport, inbox, task_report, task_summaries
from desman.core.sessions import Sessions
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.stats import collect
from desman.core.store import DEFAULT_LEASE_SECONDS, Store
from desman.global_cli import global_group
from desman.i18n import t, tn
from desman.info import print_project_info
from desman.llm import router
from desman.llm.base import LLM
from desman.llm.fake import FakeLLM, ScriptExhausted
from desman.models_cli import models
from desman.net import node
from desman.planner import Planned, must_run, plan, template_plan
from desman.proposals_cli import proposals_group
from desman.protected_cli import protected_group
from desman.tuning_cli import tuning

F = TypeVar("F", bound=Callable[..., Any])

# Exit codes of `run`, `resume` and `status`, so scripts can tell outcomes apart.
EXIT_DONE = 0
EXIT_NEEDS_HUMAN = 3
EXIT_PAUSED = 4
EXIT_STOPPED = 5  # failed or cancelled
EXIT_ACTIVE = 6  # still in progress (someone holds a lease, or nobody has run it)


def db_option(command: F) -> F:
    return click.option(
        "--db",
        "db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Database file (default: $DESMAN_DATA_DIR/desman.db or ~/.local/share/desman/desman.db).",
    )(command)


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
@click.option(
    "-c",
    "--continue",
    "resume",
    is_flag=True,
    help="Continue the last chat session instead of starting a new one.",
)
@click.option(
    "-t",
    "--temporary",
    is_flag=True,
    help="A temporary chat: nothing said is kept, it is erased when you leave.",
)
@click.pass_context
def cli(ctx: click.Context, resume: bool, temporary: bool) -> None:
    """DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)"""
    if ctx.invoked_subcommand is not None and (resume or temporary):
        raise click.UsageError(t("-c and -t are for the chat: give them without a command."))
    if resume and temporary:
        raise click.UsageError(t("A temporary chat starts afresh: -c and -t do not go together."))
    if ctx.invoked_subcommand is None:
        # Bare `desman` is the interactive entry point (ADR 0003): configure on first run,
        # then the banner and the chat.
        config_manager = load_config()
        if not config_manager.is_configured():
            run_wizard(config_manager)
        print_project_info(config_manager)
        try:
            conn = db.open_db()
        except (db.SchemaVersionError, db.SchemaMismatchError) as e:
            raise click.ClickException(str(e)) from e
        try:
            Chat(Sessions(conn), resume=resume, temporary=temporary).run()
        finally:
            conn.close()


@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    if ctx.parent:
        click.echo(ctx.parent.get_help())


cli.add_command(config)
cli.add_command(keys)
cli.add_command(usage)
cli.add_command(language)
cli.add_command(permissions)
cli.add_command(personalize)
cli.add_command(behaviour)
cli.add_command(tuning)
cli.add_command(models)
cli.add_command(protected_group, name="protected")
cli.add_command(proposals_group, name="proposals")
cli.add_command(global_group, name="global")


@cli.command("init-db")
@db_option
def init_db(db_path: Path | None) -> None:
    """Create the database or apply pending migrations. Safe to re-run."""
    path = db_path or db.default_db_path()
    actions.print_init(path, actions.init_database(path))


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
        found = actions.task_history(store, task)
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(found.as_json(), ensure_ascii=False, indent=2))
    else:
        actions.print_history(found)
    if found.problems:
        if as_json:
            click.echo(t("History is incomplete:"), err=True)
            for problem in found.problems:
                click.echo(f"  {problem}", err=True)
        raise SystemExit(1)


# ── Task commands ────────────────────────────────────────────────────────────


def json_option(command: F) -> F:
    return click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")(command)


def run_options(command: F) -> F:
    command = click.option(
        "--wait",
        is_flag=True,
        help="Sleep through provider pauses instead of returning (exit code 4) when only "
        "paused work is left.",
    )(command)
    return model_options(command)


def model_options(command: F) -> F:
    """How a run's agents are driven: shared by `run`, `resume` and `worker`."""
    command = click.option(
        "--no-cache",
        is_flag=True,
        help="Ask the model even when the same request already has an accepted reply "
        "(by default such a reply is reused for 7 days, which saves quota on a resume).",
    )(command)
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
@click.option(
    "--plan",
    "plan_mode",
    type=click.Choice(["llm", "template"]),
    default=None,
    help="How GOAL becomes subtasks: llm asks the model for a plan (the default; if it fails, "
    "the template is used), template makes the goal itself the only subtask (the default "
    "with --fake-llm).",
)
@click.option(
    "--runs-code",
    is_flag=True,
    help="The work must run code: it is done only once run_command really ran it, whatever "
    "the plan (without the shell_execution permission it waits for you).",
)
@run_options
@db_option
@json_option
def run(
    goal: str | None,
    spec_path: Path | None,
    priority: str | None,
    plan_mode: str | None,
    runs_code: bool,
    fake_script: Path | None,
    provider_name: str | None,
    model: str | None,
    wait: bool,
    lease_seconds: int,
    db_path: Path | None,
    as_json: bool,
    no_cache: bool,
) -> None:
    """Create a task for GOAL (or from --spec) and work on it.

    The model splits GOAL into subtasks; if it cannot, the goal itself is the only subtask
    (see --plan). Exit codes: 0 done, 3 waiting for the operator, 4 paused (provider limit),
    5 failed or cancelled, 6 still in progress elsewhere.
    """
    if (goal is None) == (spec_path is None):
        raise click.UsageError(t("give either GOAL or --spec, not both"))
    if spec_path is not None and plan_mode is not None:
        raise click.UsageError(t("--plan is for GOAL: a --spec is already a plan"))
    if goal is not None and not goal.strip():
        raise click.UsageError(t("GOAL is empty"))
    spec = _load_spec(spec_path) if spec_path is not None else None
    level = None if priority is None else int(Priority[priority.upper()])
    source = _provider(fake_script, provider_name, model, as_json)  # fail fast
    access = _executor_access(as_json)
    path = db_path or db.default_db_path()
    try:
        store = Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    try:
        runs.guard(store)  # before any model is asked
        models = _attach(source, store, cache=not no_cache, as_json=as_json)
        planned: Planned | None = None
        if spec is None:
            assert goal is not None
            by_model = plan_mode == "llm" or (plan_mode is None and fake_script is None)
            planned = _plan(
                models.for_(calls.Role.PLANNER), goal, level, by_model=by_model, as_json=as_json
            )
            spec = planned.spec
        elif level is not None:
            spec = spec.model_copy(update={"priority": level})
        if runs_code:  # held to it whatever the plan, the template's too
            spec = must_run(spec)
        details = planned.details() if planned else None
        task_id = store.create_task(spec, details=details)
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
        _drive(store, models, access, task_id, wait, lease_seconds, as_json, details)
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
    no_cache: bool,
) -> None:
    """Continue TASK where it stopped: after a crash, a pause or the operator's decisions.

    Work a crashed run left behind is taken back at once if that run was on this machine,
    otherwise when its lease expires. Finished subtasks are never redone. Same exit codes as
    `run`.
    """
    source = _provider(fake_script, provider_name, model, as_json)
    access = _executor_access(as_json)
    store = open_existing(db_path)
    try:
        runs.guard(store)  # before any model is asked
        task_id = find_task(store, task)
        if store.origin(task_id) is Origin.NETWORK:
            # Its models are the ones lent, within their quotas: only a worker has them.
            raise click.ClickException(
                t("A job from desman-global runs only in `desman worker`, with the models lent.")
            )
        models = _attach(source, store, cache=not no_cache, as_json=as_json)
        _drive(store, models, access, task_id, wait, lease_seconds, as_json)
    finally:
        store.conn.close()


@cli.command()
@click.option(
    "--until-idle",
    is_flag=True,
    help="Stop once no task has work left that a worker could do (by default the worker keeps "
    "waiting for new work, and for paused work to resume).",
)
@click.option(
    "--poll-seconds",
    type=click.FloatRange(min=0.1),
    default=DEFAULT_POLL_SECONDS,
    show_default=True,
    help="How often a worker with nothing to do looks for work again.",
)
@model_options
@db_option
@json_option
def worker(
    until_idle: bool,
    poll_seconds: float,
    fake_script: Path | None,
    provider_name: str | None,
    model: str | None,
    lease_seconds: int,
    db_path: Path | None,
    as_json: bool,
    no_cache: bool,
) -> None:
    """Work on the subtasks of every task until stopped (Ctrl+C).

    Several workers may run at once, in separate terminals, on one database: each takes the
    most urgent ready subtask of any task and reviews results waiting for a verdict, and never
    the same one as another worker. A worker that is stopped or killed leaves nothing behind:
    another worker, or `resume`, takes its subtask back.
    """
    source = _provider(fake_script, provider_name, model, as_json)
    access = _executor_access(as_json)
    store = open_existing(db_path)
    interrupted = False
    try:
        runs.guard(store)  # before any model is asked
        models = _attach(source, store, cache=not no_cache, as_json=as_json)
        manager = load_config()
        config = manager.load()
        crew = runs.network_crew(
            store,
            config,
            manager.load,
            given=models if source.fake is not None else None,
            cache=not no_cache,
        )
        coordinator = runs.coordinator(
            store, models, access, config, lease_seconds=lease_seconds, network=crew
        )
        coordinator.poll_seconds = poll_seconds
        watch = InboxWatch(store)
        network = node.Node(store.conn, manager.load, work=crew is not None)
        if not as_json:
            ui.note(t("Worker {owner} started. Ctrl+C stops it.", owner=coordinator.owner))
            _tell_waiting(watch)  # what already waits, then what comes to
            sys.stdout.flush()
        progress = None if as_json else _worker_progress(coordinator, watch)

        def ticked(result: WorkResult) -> None:
            if progress is not None:
                progress(result)
            _keep_in_touch(network, as_json)

        def called() -> None:
            _keep_in_touch(network, as_json)

        def idle() -> bool:
            if not as_json:
                _tell_waiting(watch)
            event = _keep_in_touch(network, as_json, idle=True)
            return event is not None and event.change is node.Change.WORK_TAKEN

        try:
            _keep_in_touch(network, as_json)
            # Between a step's model calls too: a job's lease at the registry is extended (and
            # its heartbeat sent) while one execution asks a slow model several times.
            with calls.watching(called):
                coordinator.work(until_idle=until_idle, on_tick=ticked, on_idle=idle)
        except KeyboardInterrupt:
            interrupted = True
        except ScriptExhausted as e:
            raise click.ClickException(t("fake LLM script ran out: {error}", error=e)) from e
        finally:
            network.close()
    finally:
        store.conn.close()
    if as_json:
        out = {
            "owner": coordinator.owner,
            "executed": coordinator.executed,
            "reviewed": coordinator.reviewed,
            "preempted": coordinator.preempted,
            "interrupted": interrupted,
        }
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    ui.note(
        t(
            "Worker stopped. Executions: {executed}, reviews: {reviewed}.",
            executed=coordinator.executed,
            reviewed=coordinator.reviewed,
        )
    )


def _worker_progress(coordinator: Coordinator, watch: InboxWatch) -> Callable[[object], None]:
    """A line whenever the worker's counts change (not on housekeeping alone), and whatever
    came to wait for the operator since the last look (any worker's, any task's)."""
    last = (0, 0)

    def show(_: object) -> None:
        nonlocal last
        now = (coordinator.executed, coordinator.reviewed)
        if now != last:
            last = now
            ui.note(
                t("Executions: {executed}, reviews: {reviewed}.", executed=now[0], reviewed=now[1])
            )
        _tell_waiting(watch)
        sys.stdout.flush()

    return show


def _keep_in_touch(network: node.Node, as_json: bool, *, idle: bool = False) -> node.Event | None:
    """desman-global between the worker's steps (G.5) and model calls: a contact when it is
    due, the jobs held tended, one asked for while the worker is `idle` (G.6b), and what
    changed told to the person (and returned). The network never stops the worker's own work."""
    event = network.poll(idle=idle)
    if event is None or as_json:
        return event
    if event.change is node.Change.CONNECTED:
        ui.ok(t("In touch with the desman-global registry {url}.", url=event.registry))
    elif event.change is node.Change.UNREACHABLE:
        ui.warn(
            t(
                "The desman-global registry cannot be reached ({error}); trying again in "
                "{seconds} s. Your own work goes on.",
                error=event.detail,
                seconds=round(event.retry_in or 0),
            )
        )
    elif event.change is node.Change.REFUSED:
        ui.warn(
            t(
                "The desman-global registry refused this node ({error}); trying again in "
                "{seconds} s. A clock that is off is the usual cause.",
                error=event.detail,
                seconds=round(event.retry_in or 0),
            )
        )
    elif event.change is node.Change.REVOKED:
        ui.warn(
            t(
                "This node's key is revoked: it no longer talks to desman-global. "
                "Your own work goes on."
            )
        )
    elif event.change is node.Change.NO_KEY:
        ui.warn(t("The node's key cannot be used: {error}", error=event.detail))
    elif event.change is node.Change.WORK_TAKEN:
        ui.note(
            t(
                "Took a job from desman-global ({stage}). Your own work still goes first.",
                stage=event.detail,
            )
        )
    elif event.change is node.Change.CHECK_HANDED_IN:
        ui.note(t("Handed a proposal check in to desman-global ({outcome}).", outcome=event.detail))
    elif event.change is node.Change.WORK_HANDED_IN:
        ui.note(t("Handed a job in to desman-global ({outcome}).", outcome=event.detail))
    elif event.change is node.Change.WORK_DROPPED:
        ui.warn(t("A job from desman-global was dropped: {reason}", reason=event.detail))
    sys.stdout.flush()
    return event


def _tell_waiting(watch: InboxWatch) -> None:
    actions.print_waiting(
        watch.new(), t("Decide with: desman retry|accept|fail <ref> [--note TEXT]")
    )
    sys.stdout.flush()


@cli.command()
@click.argument("task")
@db_option
@json_option
def status(task: str, db_path: Path | None, as_json: bool) -> None:
    """Show TASK and its subtasks. Same exit codes as `run`."""
    store = open_existing(db_path)
    try:
        report = task_report(store, find_task(store, task))
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print_report(report)
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
    actions.print_summaries(summaries)


def _plan(llm: LLM, goal: str, priority: int | None, *, by_model: bool, as_json: bool) -> Planned:
    level = Priority.NORMAL if priority is None else priority
    tuning = _tuning()
    if not by_model:
        return Planned(runs.tuned_spec(template_plan(goal, level), tuning), "template")
    if not as_json:
        ui.note(t("Planning…"))
        sys.stdout.flush()
    try:
        with calls.for_(calls.Purpose(calls.Role.PLANNER)):  # before the task exists
            planned = plan(llm, goal, level, limit=runs.tuned(tuning, "max_subtasks"))
    except ScriptExhausted as e:
        raise click.ClickException(t("fake LLM script ran out: {error}", error=e)) from e
    if planned.error is not None and not as_json:
        ui.warn(
            t(
                "The model did not make a plan, so the goal is the only subtask: {error}",
                error=planned.error,
            )
        )
    return Planned(runs.tuned_spec(planned.spec, tuning), planned.source, planned.error)


def _tuning() -> Tuning:
    """The person's tuning; DESMAN's defaults when there is no config (e.g. with --fake-llm)."""
    return _config_or_default().tuning


def _config_or_default() -> Config:
    """The config; an empty one when it cannot be read (a run with --fake-llm needs none)."""
    try:
        return open_config().load()
    except ConfigError:
        return Config()


def _load_spec(spec_path: Path) -> TaskSpec:
    try:
        return TaskSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as e:
        raise click.ClickException(
            t("{path}: not a valid task spec:\n{error}", path=spec_path, error=e)
        ) from e


@dataclass(frozen=True)
class _Source:
    """Where a run's models come from: a fake script, or the config (narrowed by --provider
    and --model)."""

    fake: LLM | None = None
    provider: str | None = None
    model: str | None = None


@proposals_group.command("analyze")
@click.option(
    "--since",
    default="all",
    show_default=True,
    help="Read the failures of this window: e.g. 24h, 7d, or 'all'.",
)
@run_options
@db_option
@json_option
def analyze(
    since: str,
    fake_script: Path | None,
    provider_name: str | None,
    model: str | None,
    wait: bool,
    lease_seconds: int,
    db_path: Path | None,
    as_json: bool,
    no_cache: bool,
) -> None:
    """Have the Improvement Analyst study the failures and propose a change to DESMAN's
    prompts or workflow, as a diff. Nothing is applied.

    The analysis is a task like any other (no tools): if it stops, `desman resume` continues
    it, and `desman proposals` shows the proposal once it has ended. Same exit codes as `run`.
    """
    start = proposals_cli.since_of(since)
    source = _provider(fake_script, provider_name, model, as_json)  # fail fast
    access = _executor_access(as_json)
    path = db_path or db.default_db_path()
    try:
        store = Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    try:
        runs.guard(store)  # before any model is asked
        analysis = proposals_cli.prepare(store, start, _config_or_default())
        if analysis is None:
            if as_json:
                click.echo(json.dumps({"proposal": None}, indent=2))
            else:
                proposals_cli.print_nothing_to_analyse()
            return
        models = _attach(source, store, cache=not no_cache, as_json=as_json)
        if not as_json:
            proposals_cli.print_start(store, analysis)
            sys.stdout.flush()
        proposal_id = analysis.proposal.id

        def then(as_json: bool) -> tuple[str, Any]:
            found = proposals_cli.finish(store, proposal_id)
            if not as_json:
                proposals_cli.print_proposal(store, found)
            return "proposal", proposals_cli.proposal_json(store, found)

        _drive(
            store,
            models,
            access,
            analysis.proposal.task_id,
            wait,
            lease_seconds,
            as_json,
            then=then,
        )
    finally:
        store.conn.close()


def _provider(
    fake_script: Path | None, provider_name: str | None, model: str | None, as_json: bool
) -> _Source:
    """Checks that the run can have models before anything else happens (fail fast)."""
    if fake_script is not None:
        if provider_name is not None or model is not None:
            raise click.UsageError(t("--fake-llm replaces the provider: drop --provider/--model"))
        try:
            return _Source(FakeLLM.from_file(fake_script, sleep=time.sleep))  # latency is real
        except (ValidationError, ValueError) as e:
            raise click.ClickException(
                t("{path}: not a valid fake LLM script:\n{error}", path=fake_script, error=e)
            ) from e
    try:
        router.lineup(load_config().load(), calls.Role.EXECUTOR, provider_name, model)
    except router.NoProvider as e:
        raise click.ClickException(str(e)) from e
    return _Source(provider=provider_name, model=model)


def _attach(source: _Source, store: Store, *, cache: bool, as_json: bool) -> runs.Models:
    """The run's models (`runs.models`: a lineup per role), said to the operator. The fake is
    not wrapped: its scripts test the retry policy and count every call."""
    if source.fake is not None:
        return runs.Models(source.fake, None)
    config = load_config().load()
    try:
        models = runs.models(
            config, store.conn, provider=source.provider, model=source.model, cache=cache
        )
    except router.NoProvider as e:
        raise click.ClickException(str(e)) from e
    choice = models.choice
    if as_json or choice is None:
        return models
    if choice.provider.key_env is None:
        url = choice.provider.endpoint(config)
        ui.note(
            t(
                "Using {provider} · {model} (on this machine)",
                provider=choice.provider.name,
                model=choice.model,
            )
            if providers.on_this_machine(url)
            else t(
                "Using {provider} · {model} (server at {url})",
                provider=choice.provider.name,
                model=choice.model,
                url=url,
            )
        )
    else:
        ui.note(
            tn(
                len(config.keys(choice.provider.key_env)),
                "Using {provider} · {model} ({n} key)",
                "Using {provider} · {model} ({n} keys)",
                provider=choice.provider.name,
                model=choice.model,
            )
        )
    reviewer = models.first(calls.Role.REVIEWER)
    if reviewer is not None and reviewer.label != choice.label:
        ui.note(
            t(
                "Plans and reviews by {provider} · {model}",
                provider=reviewer.provider.name,
                model=reviewer.model,
            )
        )
    code = _first_for(models, Kind.CODE)
    if code is not None and code.label != choice.label:
        ui.note(t("Code by {provider} · {model}", provider=code.provider.name, model=code.model))
    analysis = _first_for(models, Kind.ANALYSIS)
    if analysis is not None and analysis.label != choice.label:
        ui.note(
            t(
                "Analysis by {provider} · {model}",
                provider=analysis.provider.name,
                model=analysis.model,
            )
        )
    if choice.provider.paid:
        ui.warn(t("{name} is a paid provider: every call is billed.", name=choice.provider.name))
        ui.warn(providers.untested_note())
    if models.other_choice is not None:
        ui.note(
            t(
                "Retries may also use {provider} · {model}",
                provider=models.other_choice.provider.name,
                model=models.other_choice.model,
            )
        )
    return models


def _first_for(models: runs.Models, kind: Kind) -> router.Choice | None:
    """The model the executor asks first for a subtask of `kind` (None for a fake)."""
    choices: list[router.Choice] = getattr(models.executor(kind)[0], "choices", [])
    return choices[0] if choices else None


def _executor_access(as_json: bool) -> Access:
    return runs.executor_access(load_config().load(), stderr=as_json)


def _drive(
    store: Store,
    models: runs.Models,
    access: Access,
    task_id: str,
    wait: bool,
    lease_seconds: int,
    as_json: bool,
    plan_details: dict[str, Any] | None = None,
    then: Callable[[bool], tuple[str, Any]] | None = None,
) -> None:
    """Run the task and report it. `then(as_json)` comes after the run: what it returns joins
    the JSON output as (key, data); without --json it prints, after the task's report."""
    coordinator = runs.coordinator(
        store, models, access, load_config().load(), lease_seconds=lease_seconds
    )
    try:
        result = coordinator.run(task_id, wait_for_pauses=wait)
    except ScriptExhausted as e:
        raise click.ClickException(t("fake LLM script ran out: {error}", error=e)) from e
    report = task_report(store, task_id)
    short = store.short_id(task_id)
    if as_json:
        out: dict[str, Any] = {"run": _run_summary(result)}
        if plan_details is not None:
            out["plan"] = plan_details
        out["task"] = report.model_dump(mode="json")
        if then is not None:
            key, data = then(True)
            out[key] = data
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_report(report)
        if then is not None:
            then(False)
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


# ── Operator decisions (a minimal subset of ROADMAP 5.1) ─────────────────────


@cli.command("stats")
@click.option(
    "--since",
    default="all",
    show_default=True,
    help="Window: e.g. 30m, 24h, 7d, or 'all'.",
)
@db_option
@json_option
def stats_command(since: str, db_path: Path | None, as_json: bool) -> None:
    """How well tasks go: success, retries, reviews, failures, model calls and tokens by role
    and by provider and model. A window counts what happened in it (tasks created since)."""
    window = parse_window(since)
    store = open_existing(db_path)
    try:
        result = collect(store.conn, None if window is None else datetime.now(UTC) - window)
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(result.as_json(), ensure_ascii=False, indent=2))
        return
    actions.print_stats(result, since)


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
    actions.print_inbox(items)


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
        result = actions.cancel_task(store, find_task(store, task), note)
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(result.as_json(), indent=2))
    else:
        actions.print_cancelled(result)


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
        pending = actions.prepare_decision(store, ref, target, if_version)
        decision = actions.apply_decision(store, pending, note, action)
    finally:
        store.conn.close()
    if as_json:
        click.echo(json.dumps(decision.as_json(), indent=2))
    else:
        actions.print_decision(decision)
