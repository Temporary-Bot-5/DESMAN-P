"""`desman proposals`: what the Improvement Analyst proposed (ROADMAP 5.4), and how it did on
the benchmark (5.5).

`list` and `show` read; `analyze` (in `desman.cli`, next to `run`, whose way of driving a task
it shares) runs an analysis task; `evaluate` runs the benchmark on the current version and on
the proposed one (`desman.evaluation`). Every command first settles the analyses whose task has
ended since (`improve.settle`), so one finished by `resume` or a worker shows up too. Nothing
here applies a proposal. The chat calls the same functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.text import Text

from desman import actions, bench, evaluation, improve, ui
from desman.config import Config, ConfigError, open_config
from desman.config_cli import json_option, parse_window
from desman.core import calls, db, proposals
from desman.core.proposals import Evaluation, Proposal, Status
from desman.core.store import Store
from desman.i18n import t, tn
from desman.llm import router
from desman.protected_cli import db_option


@click.group("proposals", invoke_without_command=True)
@click.pass_context
def proposals_group(ctx: click.Context) -> None:
    """Changes to DESMAN's prompts and workflow that the Improvement Analyst proposed from
    the failures. Nothing is applied."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_command)


@proposals_group.command("list")
@click.option("--limit", type=click.IntRange(min=1), default=20, show_default=True)
@db_option
@json_option
def list_command(limit: int = 20, db_path: Path | None = None, as_json: bool = False) -> None:
    """The proposals, newest first."""
    store = actions.open_existing(db_path)
    try:
        improve.settle(store)
        found = proposals.listing(store.conn, limit)
        if as_json:
            click.echo(json.dumps(listing_json(found), ensure_ascii=False, indent=2))
            return
        print_listing(store, found)
    finally:
        store.conn.close()


@proposals_group.command("show")
@click.argument("proposal", type=click.IntRange(min=1))
@db_option
@json_option
def show_command(proposal: int, db_path: Path | None = None, as_json: bool = False) -> None:
    """PROPOSAL (its number): what it changes, why, and the diff."""
    store = actions.open_existing(db_path)
    try:
        improve.settle(store)
        found = find(store, proposal)
        if as_json:
            click.echo(json.dumps(proposal_json(store, found), ensure_ascii=False, indent=2))
            return
        print_proposal(store, found)
    finally:
        store.conn.close()


@proposals_group.command("evaluate")
@click.argument("proposal", type=click.IntRange(min=1))
@click.option(
    "--real",
    is_flag=True,
    help="Run the benchmark on your models (spends their quota) instead of the scripted fake "
    "model.",
)
@click.option("--provider", "provider_name", default=None, help="With --real: the provider.")
@click.option("--model", default=None, help="With --real: the model.")
@db_option
@json_option
def evaluate_command(
    proposal: int,
    real: bool = False,
    provider_name: str | None = None,
    model: str | None = None,
    db_path: Path | None = None,
    as_json: bool = False,
) -> None:
    """Run the benchmark on the current version of DESMAN and on it with PROPOSAL, each in a
    git worktree of its own, and compare them. Nothing is applied.

    By default the scripted fake model answers: it shows whether the proposed version still
    works, and what its prompts cost. With --real your models answer, which shows whether the
    work goes better and spends their quota.
    """
    if not real and (provider_name is not None or model is not None):
        raise click.UsageError(t("--provider and --model are for --real"))
    store = actions.open_existing(db_path)
    try:
        improve.settle(store)
        found = find(store, proposal)
        done = evaluate_now(
            store, found, real=real, provider=provider_name, model=model, quiet=as_json
        )
        if as_json:
            click.echo(json.dumps(evaluation_json(done), ensure_ascii=False, indent=2))
            return
        print_evaluation(done)
    finally:
        store.conn.close()


def evaluate_now(
    store: Store,
    proposal: Proposal,
    *,
    real: bool,
    provider: str | None = None,
    model: str | None = None,
    quiet: bool = False,
) -> Evaluation:
    """Evaluate `proposal` (`evaluation.evaluate`), telling the person what runs."""
    try:
        config = open_config().load()
    except ConfigError as e:
        if real:
            raise click.ClickException(str(e)) from e
        config = Config()
    if real:  # fail fast, before any worktree is made
        try:
            first = router.lineup(config, calls.Role.EXECUTOR, provider, model)[0]
        except router.NoProvider as e:
            raise click.ClickException(str(e)) from e
        if first.provider.paid:
            ui.warn(
                t("{name} is a paid provider: every call is billed.", name=first.provider.name),
                stderr=quiet,
            )
    if not quiet:
        ui.note(
            tn(
                len(bench.CASES),
                "The benchmark ({n} task) runs on the current version, then on the proposed "
                "one, each in a git worktree of its own.",
                "The benchmark ({n} tasks) runs on the current version, then on the proposed "
                "one, each in a git worktree of its own.",
            )
        )

    def on_run(side: evaluation.Side) -> None:
        if quiet:
            return
        ui.note(
            t("Running the current version…")
            if side == "base"
            else t("Running the proposed version…")
        )

    try:
        return evaluation.evaluate(
            store.conn,
            proposal,
            mode="real" if real else "fake",
            tuning=config.tuning,
            provider=provider,
            model=model,
            on_run=on_run,
        )
    except evaluation.EvaluationError as e:
        raise click.ClickException(
            t("Cannot evaluate proposal {id}: {reason}", id=proposal.id, reason=e)
        ) from e


def find(store: Store, proposal_id: int) -> Proposal:
    found = proposals.get(store.conn, proposal_id)
    if found is None:
        raise click.ClickException(t("No proposal {id}.", id=proposal_id))
    return found


# ── An analysis ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Analysis:
    """What `analyze` works on: a new analysis, or one already under way."""

    proposal: Proposal
    new: bool


def since_of(since: str) -> str | None:
    """`--since` (7d, all) as the window's start."""
    window = parse_window(since)
    return None if window is None else db.to_iso(datetime.now(UTC) - window)


def prepare(store: Store, since: str | None, config: Config | None = None) -> Analysis | None:
    """Settle what ended, then the analysis to run: the one under way (its task still has work
    left), else a new one over the settings of `config`. None when there is nothing to
    analyse."""
    improve.settle(store)
    under_way = proposals.analysing(store.conn)
    if under_way:
        return Analysis(under_way[0], new=False)
    try:
        started = improve.start(store, since, config)
    except proposals.UnderWay as e:  # another process started one just now
        found = proposals.get(store.conn, e.proposal_id)
        assert found is not None
        return Analysis(found, new=False)
    return None if started is None else Analysis(started, new=True)


def print_nothing_to_analyse() -> None:
    ui.ok(t("Nothing to analyse: no failed attempts or failed plans in that time."))


def print_start(store: Store, analysis: Analysis) -> None:
    ref = store.short_id(analysis.proposal.task_id)
    if analysis.new:
        ui.note(t("Analysing the failures as task {ref}.", ref=ref))
    else:
        ui.note(t("Continuing the analysis already under way (task {ref}).", ref=ref))


def finish(store: Store, proposal_id: int) -> Proposal:
    """After the analysis task ran: its proposal, settled if the task ended."""
    improve.settle(store)
    found = proposals.get(store.conn, proposal_id)
    assert found is not None
    return found


# ── Output ───────────────────────────────────────────────────────────────────


def status_label(status: Status) -> str:
    return {
        Status.ANALYSING: t("being analysed"),
        Status.PROPOSED: t("proposed"),
        Status.NOTHING: t("nothing to propose"),
        Status.UNUSABLE: t("unusable"),
    }[status]


def targets(proposal: Proposal) -> list[str]:
    """The prompts and settings it changes."""
    return improve.changed(proposal)


def print_listing(store: Store, found: list[Proposal]) -> None:
    if not found:
        ui.note(t("No proposals yet. Ask for one: desman proposals analyze"))
        return
    for p in found:
        when = db.parse_iso(p.created_at).astimezone().strftime("%Y-%m-%d %H:%M")
        what = ", ".join(targets(p)) or actions.one_line(p.summary or p.reason or "", 70)
        ui.out(Text(f"{p.id:>4}  {when}  {status_label(p.status)}  {what}".rstrip()))
    ui.note(t("See one: desman proposals show NUMBER"))


def print_proposal(store: Store, p: Proposal) -> None:
    ui.out(Text(t("Proposal {id}: {status}", id=p.id, status=status_label(p.status)), style="bold"))
    ref = store.short_id(p.task_id)
    if p.status is Status.ANALYSING:
        ui.note(
            t(
                "The analysis (task {ref}) has not ended. Continue it: desman resume {ref}",
                ref=ref,
            )
        )
        return
    if p.summary:
        ui.out(Text(p.summary))
    if p.reason:
        ui.warn(t("Not usable: {reason}", reason=p.reason))
    if p.status is not Status.PROPOSED:
        return
    changes = p.details.get("changes", [])
    settings = p.details.get("settings", [])
    if p.details.get("expected_effect"):
        ui.out(Text(t("Expected: {effect}", effect=p.details["expected_effect"])))
    if changes:
        ui.note(
            tn(
                len(changes),
                "{n} change to a prompt:",
                "{n} changes to prompts:",
            )
        )
        for change in changes:
            ui.out(Text(f"  {change['target']} ({change['file']})"))
        assert p.diff is not None
        ui.out(Text(p.diff))
        path = improve.file_for(store.conn, p)
        if path is not None and path.exists():
            ui.note(t("The diff is also in {path}", path=path))
    if settings:
        ui.note(tn(len(settings), "{n} change to a setting:", "{n} changes to settings:"))
        for item in settings:
            ui.out(Text(f"  {item['setting']}: {_value(item['old'])} → {_value(item['new'])}"))
        ui.note(t("Make it yourself if you agree: {command}", command=p.details["command"]))
    found = proposals.evaluations(store.conn, p.id)
    if found:
        print_evaluation(found[0])
        if len(found) > 1:
            ui.note(
                tn(
                    len(found) - 1,
                    "{n} earlier evaluation (see --json)",
                    "{n} earlier evaluations (see --json)",
                )
            )
    else:
        ui.note(t("Not evaluated yet: desman proposals evaluate {id}", id=p.id))
    ui.note(
        t(
            "Nothing was applied. Applying proposals comes later (ROADMAP 5.6); until then a "
            "person may apply the diff with git."
        )
    )


def verdict_label(verdict: str) -> str:
    return {
        "better": t("the proposed version did better"),
        "worse": t("the proposed version did worse"),
        "same": t("no clear difference"),
        "failed": t("could not be finished"),
    }.get(verdict, verdict)


def metric_label(metric: str) -> str:
    return {
        "tasks_done": t("tasks done"),
        "escalated": t("subtasks left to the person"),
        "plan_fallbacks": t("plans the model failed"),
        "rejected": t("results rejected by the reviewer"),
        "attempts": t("attempts"),
        "first_try": t("subtasks done at the first try"),
        "subtasks": t("subtasks"),
        "calls": t("model calls"),
        "failed_calls": t("failed model calls"),
        "tokens": t("tokens"),
    }.get(metric, metric)


def print_evaluation(e: Evaluation) -> None:
    mode = t("the scripted fake model") if e.mode == "fake" else t("your models")
    ui.out(
        Text(
            t(
                "Evaluation {n} ({mode}, commit {commit}): {verdict}",
                n=e.id,
                mode=mode,
                commit=e.commit[:12],
                verdict=verdict_label(e.verdict),
            ),
            style="bold",
        )
    )
    if e.reason:
        ui.warn(e.reason)
    base, candidate = evaluation.metrics_of(e)
    if base is not None:
        table = ui.table("", t("current version"), t("proposed version"))
        for metric in bench.METRICS:
            old = str(base[metric.id])
            new = "—" if candidate is None else str(candidate[metric.id])
            table.add_row(metric_label(metric.id), old, new)
        ui.out(table)
    if e.verdict == "failed":
        return
    if e.mode == "fake":
        ui.note(
            t(
                "The fake model answers the same whatever the prompt says: this shows that the "
                "proposed version still works, and what its prompts cost. Whether the work goes "
                "better shows on your models: --real."
            )
        )
    else:
        ui.note(
            t(
                "One run of real models is noisy: a small difference may be chance. Evaluating "
                "again adds another run."
            )
        )


def _value(value: Any) -> str:
    return t("default") if value is None else str(value)


def listing_json(found: list[Proposal]) -> list[dict[str, Any]]:
    return [
        p.model_dump(mode="json", exclude={"diff", "details"}) | {"targets": targets(p)}
        for p in found
    ]


def proposal_json(store: Store, p: Proposal) -> dict[str, Any]:
    path = improve.file_for(store.conn, p)
    return p.model_dump(mode="json") | {
        "file": None if path is None else str(path),
        "evaluations": [evaluation_json(e) for e in proposals.evaluations(store.conn, p.id)],
    }


def evaluation_json(e: Evaluation) -> dict[str, Any]:
    base, candidate = evaluation.metrics_of(e)
    return e.model_dump(mode="json") | {"metrics": {"base": base, "candidate": candidate}}
