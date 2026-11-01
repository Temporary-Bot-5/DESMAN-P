"""`desman proposals`: what the Improvement Analyst proposed (ROADMAP 5.4), and how it did on
the benchmark (5.5).

`list` and `show` read; `analyze` (in `desman.cli`, next to `run`, whose way of driving a task
it shares) runs an analysis task; `evaluate` runs the benchmark on the current version and on
the proposed one (`desman.evaluation`); `apply` and `revert` are the person's (5.6,
`desman.applying`): a git commit of the checkout, and the settings. Every command first settles
the analyses whose task has ended since (`improve.settle`), so one finished by `resume` or a
worker shows up too. The chat calls the same functions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.text import Text

from desman import actions, applying, bench, evaluation, improve, ui
from desman.config import Config, ConfigError, ConfigManager, open_config
from desman.config_cli import json_option, load_config, parse_window
from desman.core import calls, db, proposals
from desman.core.proposals import Application, Evaluation, Proposal, Status
from desman.core.store import Store
from desman.i18n import t, tn
from desman.llm import router
from desman.protected_cli import db_option


@click.group("proposals", invoke_without_command=True)
@click.pass_context
def proposals_group(ctx: click.Context) -> None:
    """Changes to DESMAN's prompts and workflow that the Improvement Analyst proposed from
    the failures. Only you apply one (apply), and you can take it back (revert)."""
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
            click.echo(json.dumps(listing_json(store, found), ensure_ascii=False, indent=2))
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


@proposals_group.command("apply")
@click.argument("proposal", type=click.IntRange(min=1))
@db_option
@json_option
def apply_command(proposal: int, db_path: Path | None = None, as_json: bool = False) -> None:
    """Apply PROPOSAL: its diff becomes a git commit of the checkout DESMAN runs from, on the
    branch it is on, and its settings are changed. Nothing else is touched; if you have
    uncommitted changes in those files, nothing is done. Take it back with `revert`."""
    _apply_or_revert(proposal, "applied", db_path, as_json)


@proposals_group.command("revert")
@click.argument("proposal", type=click.IntRange(min=1))
@db_option
@json_option
def revert_command(proposal: int, db_path: Path | None = None, as_json: bool = False) -> None:
    """Take back PROPOSAL, applied before: a commit reverting its commit, and its settings as
    they were (those you changed since are left as they are)."""
    _apply_or_revert(proposal, "reverted", db_path, as_json)


def _apply_or_revert(
    proposal: int, action: applying.Action, db_path: Path | None, as_json: bool
) -> None:
    manager = load_config()
    store = actions.open_existing(db_path)
    try:
        improve.settle(store)
        found = find(store, proposal)
        plan = plan_now(store, found, manager, action)
        if not as_json:
            for line in cautions(store, plan):
                ui.warn(line)
        done = carry_out_now(store, plan, manager)
        if as_json:
            click.echo(json.dumps(application_json(done), ensure_ascii=False, indent=2))
            return
        print_application(done)
    finally:
        store.conn.close()


def plan_now(
    store: Store, proposal: Proposal, manager: ConfigManager, action: applying.Action
) -> applying.Plan:
    """What applying or reverting `proposal` would do (`applying.plan_apply`/`plan_revert`)."""
    try:
        tuning = manager.load().tuning
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
    plan = applying.plan_apply if action == "applied" else applying.plan_revert
    try:
        return plan(store.conn, proposal, tuning)
    except applying.ApplyError as e:
        raise _refused(proposal.id, action, e) from e


def carry_out_now(
    store: Store, plan: applying.Plan, manager: ConfigManager, session_id: str | None = None
) -> Application:
    try:
        return applying.carry_out(store.conn, plan, manager, session_id=session_id)
    except applying.ApplyError as e:
        raise _refused(plan.proposal.id, plan.action, e) from e


def _refused(proposal_id: int, action: applying.Action, e: Exception) -> click.ClickException:
    if action == "applied":
        return click.ClickException(
            t("Cannot apply proposal {id}: {reason}", id=proposal_id, reason=e)
        )
    return click.ClickException(
        t("Cannot revert proposal {id}: {reason}", id=proposal_id, reason=e)
    )


def plan_lines(plan: applying.Plan) -> list[str]:
    """What the plan does, for the person (the chat's window)."""
    p = plan.proposal
    lines = [
        t("Apply proposal {id}:", id=p.id)
        if plan.action == "applied"
        else t("Revert proposal {id}:", id=p.id)
    ]
    if plan.repo is not None:
        files = ", ".join(plan.files)
        if plan.action == "applied" and plan.branch is not None:
            lines.append(
                t("a commit changing {files} on branch {branch}", files=files, branch=plan.branch)
            )
        elif plan.action == "applied":
            lines.append(t("a commit changing {files} on the detached HEAD", files=files))
        else:
            assert plan.last is not None and plan.last.commit is not None
            lines.append(
                t(
                    "a commit reverting {commit} ({files}) on {branch}",
                    commit=plan.last.commit[:12],
                    files=files,
                    branch=plan.branch or "HEAD",
                )
            )
        lines.append(t("in the checkout at {path}", path=plan.repo))
    for item in plan.settings:
        lines.append(f"{item['setting']}: {_value(item['old'])} → {_value(item['new'])}")
    if plan.left:
        lines.append(t("Left as you changed them since: {settings}", settings=", ".join(plan.left)))
    return lines


def cautions(store: Store, plan: applying.Plan) -> list[str]:
    """What the person should know before applying: how the proposal did on the benchmark, and
    that running processes keep the prompts they loaded."""
    found = []
    if plan.action == "applied":
        done = proposals.evaluations(store.conn, plan.proposal.id)
        if not done:
            found.append(
                t(
                    "Proposal {id} was not evaluated: desman proposals evaluate {id}",
                    id=plan.proposal.id,
                )
            )
        elif done[0].verdict in ("worse", "failed"):
            found.append(
                t("Its last evaluation: {verdict}.", verdict=verdict_label(done[0].verdict))
            )
    if plan.repo is not None:
        found.append(
            t("Workers and chats already running keep the old prompts until they start again.")
        )
    return found


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
    try:  # the person's tuning is the current version's, also with the fake
        config = open_config().load()
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
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
        state = status_label(p.status)
        last = proposals.last_application(store.conn, p.id)
        if last is not None:
            state += " · " + (t("applied") if last.action == "applied" else t("reverted"))
        ui.out(Text(f"{p.id:>4}  {when}  {state}  {what}".rstrip()))
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
    print_state(store, p)


def print_state(store: Store, p: Proposal) -> None:
    """Whether the proposal is applied, and how to change that."""
    last = proposals.last_application(store.conn, p.id)
    if last is not None and last.action == "applied":
        if last.commit is not None:
            ui.note(
                t(
                    "Applied in commit {commit}. Take it back: desman proposals revert {id}",
                    commit=last.commit[:12],
                    id=p.id,
                )
            )
        else:
            ui.note(t("Applied. Take it back: desman proposals revert {id}", id=p.id))
    elif last is not None:
        ui.note(t("Reverted. Apply it again: desman proposals apply {id}", id=p.id))
    else:
        ui.note(t("Not applied. If you agree, apply it: desman proposals apply {id}", id=p.id))


def print_application(a: Application) -> None:
    applied = a.action == "applied"
    if a.commit is not None:
        ui.ok(
            t("Proposal {id} applied in commit {commit}.", id=a.proposal_id, commit=a.commit[:12])
            if applied
            else t(
                "Proposal {id} reverted in commit {commit}.", id=a.proposal_id, commit=a.commit[:12]
            )
        )
    else:
        ui.ok(
            t("Proposal {id} applied.", id=a.proposal_id)
            if applied
            else t("Proposal {id} reverted.", id=a.proposal_id)
        )
    for item in a.details.get("settings", []):
        ui.out(Text(f"  {item['setting']}: {_value(item['old'])} → {_value(item['new'])}"))
    if a.details.get("left"):
        ui.warn(
            t(
                "Left as you changed them since: {settings}",
                settings=", ".join(a.details["left"]),
            )
        )
    if applied:
        ui.note(t("Take it back: desman proposals revert {id}", id=a.proposal_id))


def application_json(a: Application) -> dict[str, Any]:
    return a.model_dump(mode="json")


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


def listing_json(store: Store, found: list[Proposal]) -> list[dict[str, Any]]:
    return [
        p.model_dump(mode="json", exclude={"diff", "details"})
        | {"targets": targets(p), "applied": _applied(store, p)}
        for p in found
    ]


def _applied(store: Store, p: Proposal) -> bool:
    last = proposals.last_application(store.conn, p.id)
    return last is not None and last.action == "applied"


def proposal_json(store: Store, p: Proposal) -> dict[str, Any]:
    path = improve.file_for(store.conn, p)
    return p.model_dump(mode="json") | {
        "file": None if path is None else str(path),
        "evaluations": [evaluation_json(e) for e in proposals.evaluations(store.conn, p.id)],
        "applications": [application_json(a) for a in proposals.applications(store.conn, p.id)],
    }


def evaluation_json(e: Evaluation) -> dict[str, Any]:
    base, candidate = evaluation.metrics_of(e)
    return e.model_dump(mode="json") | {"metrics": {"base": base, "candidate": candidate}}
