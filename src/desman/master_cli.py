"""The person's measured admissions for master models (ROADMAP M.3)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any

import click
import httpx
from pydantic import BeforeValidator, TypeAdapter, ValidationError
from rich.text import Text

from desman import bench, providers, runs, ui
from desman.config import Config, ConfigManager, MasterChoices, ModelLabel, Tuning
from desman.config_cli import json_option, load_config
from desman.core import db, operator
from desman.core.retry import FailureKind
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.i18n import t
from desman.llm import catalog, check, router
from desman.protected_cli import connect, db_option
from desman.providers import CLOUD


def _lower_provider(value: object) -> object:
    if isinstance(value, str):
        provider, separator, model = value.partition("/")
        return provider.lower() + separator + model
    return value


MasterInputLabel = Annotated[ModelLabel, BeforeValidator(_lower_provider)]
_LABEL = TypeAdapter(MasterInputLabel)


def valid_label(model: str, config: Config) -> str:
    if providers.redact_keys(model, config)[1]:
        raise click.ClickException(
            t("an argument looked like an API key, so the command was dropped")
        )
    try:
        label = _LABEL.validate_python(model)
    except ValidationError as e:
        raise click.ClickException(t("Use a model label in the form PROVIDER/MODEL.")) from e

    source, _, model_id = label.partition("/")
    try:
        provider = providers.find(source)
    except providers.UnknownProvider as e:
        raise click.ClickException(str(e)) from e
    return f"{provider.id}/{model_id}"


def _summary(row: sqlite3.Row, result: bench.Result) -> dict[str, Any]:
    return {
        "id": row["id"],
        "model": row["model"],
        "checked_at": row["checked_at"],
        "version": row["version"],
        "passed": row["outcome"] == "passed",
        "outcome": row["outcome"],
        "done": sum(case.status == "done" for case in result.cases),
        "total": len(result.cases),
    }


def rows(conn: sqlite3.Connection, config: Config) -> list[dict[str, Any]]:
    marked = {f"{p.id}/{m.id}" for p in CLOUD for m in catalog.models_of(p) if m.master}
    latest = {
        row["model"]: _summary(row, bench.Result.model_validate_json(row["result_json"]))
        for row in conn.execute(
            "SELECT * FROM master_checks WHERE id IN (SELECT MAX(id) FROM master_checks GROUP BY model)"
        )
    }
    admitted, revoked = set(config.master.admitted), set(config.master.revoked)
    return [
        {
            "model": model,
            "catalog": model in marked,
            "admitted": model in admitted,
            "revoked": model in revoked,
            "last_check": latest.get(model),
        }
        for model in sorted(marked | admitted | revoked | latest.keys())
    ]


def _verdict(outcome: str) -> str:
    return {
        "passed": t("Passed"),
        "failed": t("Did not pass"),
        "unfinished": t("Unfinished"),
    }[outcome]


def print_list(models: list[dict[str, Any]]) -> None:
    table = ui.table(t("Model"), t("Master module"), t("Latest check"))
    for model in models:
        labels = []
        if model["catalog"]:
            labels.append(t("Catalog"))
        if model["admitted"]:
            labels.append(t("Admitted"))
        if model["revoked"]:
            labels.append(t("Revoked"))
        last = model["last_check"]
        checked = (
            t(
                "{verdict} · {done}/{total} done · {date} · v{version}",
                verdict=_verdict(last["outcome"]),
                done=last["done"],
                total=last["total"],
                date=last["checked_at"],
                version=last["version"],
            )
            if last is not None
            else t("Never checked")
        )
        table.add_row(Text(model["model"]), ", ".join(labels), checked)
    ui.out(table)


def check_notice(model: str) -> str:
    return t("Check {model} using the benchmark. This spends its provider’s quota.", model=model)


def _served(config: Config, choice: router.Choice) -> None:
    provider = choice.provider
    keys = [k.value for k in config.keys(provider.key_env)] if provider.key_env else [""]
    listed = False
    with httpx.Client() as client:
        for key in keys:
            try:
                response = check.models_response(config, provider, key, client)
                ids = check.listed_models(response)
            except (httpx.HTTPError, ValueError):
                continue
            listed = True
            if (
                check.served_by(provider, choice.model, ids)
                if provider.local
                else choice.model in ids
            ):
                return
    if listed:
        raise click.ClickException(t("The provider does not list {model}.", model=choice.label))
    raise click.ClickException(
        t("Could not read the provider’s model list for {model}.", model=choice.label)
    )


def _case_failures(conn: sqlite3.Connection, config: Config) -> tuple[dict[str, str], set[str]]:
    reasons: dict[str, str] = {}
    provider_stops: set[str] = set()
    for case_id, status, failure, reason in conn.execute(
        """SELECT json_extract(created.details_json, '$.benchmark'), a.to_status,
                  json_extract(a.details_json, '$.failure'),
                  COALESCE(json_extract(a.details_json, '$.cause'),
                           json_extract(a.details_json, '$.reason'))
           FROM audit_log created
           JOIN audit_log a ON a.seq = (
               SELECT seq FROM audit_log
               WHERE task_id = created.task_id AND subtask_id IS NOT NULL
                 AND to_status IN ('needs_human', 'paused', 'failed')
               ORDER BY seq DESC LIMIT 1
           )
           WHERE created.action = 'task_created'"""
    ):
        # The master benchmark runs one subtask without tools. A permanent failure here
        # means a provider rejection or that the master is unavailable (NO_MASTER).
        if (status == SubtaskStatus.PAUSED and failure == FailureKind.LIMIT) or (
            status == SubtaskStatus.NEEDS_HUMAN
            and failure in (FailureKind.LIMIT, FailureKind.PERMANENT)
        ):
            provider_stops.add(case_id)
        if reason:
            safe = " ".join(providers.redact_keys(reason, config)[0].split())
            reasons[case_id] = safe if len(safe) <= 200 else safe[:199] + "…"
    return reasons, provider_stops


def run_check(store: Store, config: Config, model: str, *, as_json: bool = False) -> dict[str, Any]:
    model = valid_label(model, config)
    runs.guard(store)
    try:
        choice = router.master_candidate(config, model)
        _served(config, choice)
        ui.warn(check_notice(choice.label), stderr=as_json)
        with (
            tempfile.TemporaryDirectory(prefix="desman-master-") as folder,
            closing(db.open_db(Path(folder) / "desman.db")) as conn,
        ):
            tuning = Tuning(models=config.tuning.models, keys=config.tuning.keys)
            result = bench.run(
                Store(conn),
                "real",
                tuning,
                config=config,
                master=choice.label,
                calls_conn=store.conn,
            )
            reasons, provider_stops = _case_failures(conn, config)
    except router.NoProvider as e:
        raise click.ClickException(str(e)) from e
    incomplete = {case.id for case in result.cases if case.status != TaskStatus.DONE}
    outcome = "failed"
    if bench.master_passed(result):
        outcome = "passed"
    elif incomplete and incomplete <= provider_stops:
        outcome = "unfinished"
    row = store.conn.execute(
        "INSERT INTO master_checks (model, version, outcome, result_json) VALUES (?, ?, ?, ?) "
        "RETURNING id, model, checked_at, version, outcome",
        (choice.label, result.version, outcome, result.model_dump_json()),
    ).fetchone()
    report = result.model_dump(mode="json")
    for case in report["cases"]:
        case["reason"] = None if case["status"] == TaskStatus.DONE else reasons.get(case["id"])
    return _summary(row, result) | {"result": report, "usage": bench.metrics(result)}


def print_check(data: dict[str, Any]) -> None:
    table = ui.table(t("Case"), t("Status"), t("Reason"))
    labels = {
        TaskStatus.ACTIVE: t("active"),
        TaskStatus.NEEDS_HUMAN: t("needs human"),
        TaskStatus.DONE: t("done"),
        TaskStatus.FAILED: t("failed"),
        TaskStatus.CANCELLED: t("cancelled"),
    }
    for case in data["result"]["cases"]:
        table.add_row(case["id"], labels[TaskStatus(case["status"])], Text(case["reason"] or ""))
    ui.out(table)
    usage = data["usage"]
    ui.note(t("{calls} calls · {tokens} tokens", calls=usage["calls"], tokens=usage["tokens"]))
    ui.note(
        t(
            "{verdict} · {done}/{total} done · {escalated} escalated",
            verdict=_verdict(data["outcome"]),
            done=data["done"],
            total=data["total"],
            escalated=usage["escalated"],
        )
    )

    if data["outcome"] == "unfinished":
        ui.note(
            t(
                "The provider refused or limited requests, or the model became unavailable. Try again later."
            )
        )


def require_pass(conn: sqlite3.Connection, model: str) -> None:
    latest = conn.execute(
        "SELECT outcome, version, result_json FROM master_checks WHERE model = ? ORDER BY id DESC LIMIT 1",
        (model,),
    ).fetchone()
    limits = json.loads(latest["result_json"]).get("limits") if latest is not None else None
    if (
        latest is None
        or latest["outcome"] != "passed"
        or latest["version"] != bench.VERSION
        or not isinstance(limits, dict)
        or {**runs.TUNING_DEFAULTS, **limits} != runs.TUNING_DEFAULTS
    ):
        raise click.ClickException(
            t(
                "{model} needs a passing check on the current benchmark. Run `desman master check {model}`.",
                model=model,
            )
        )


def change(
    conn: sqlite3.Connection,
    manager: ConfigManager,
    model: str,
    *,
    admit: bool,
    journal: bool = False,
) -> bool:
    # Serialize decisions with config edits and with checks being saved.
    with manager.locked(), db.transaction(conn):
        config = manager.load()
        model = valid_label(model, config)
        old = config.master
        add, remove = ("admitted", "revoked") if admit else ("revoked", "admitted")
        if admit:
            if model in old.admitted:
                return False
            require_pass(conn, model)
        values = old.model_dump()
        values[add].append(model)
        values[remove] = [label for label in values[remove] if label != model]
        config.master = MasterChoices.model_validate(values)
        if config.master == old:
            return False
        if journal:
            for action, details in changes(old, config.master):
                operator.write(conn, action, details)
        manager.save(config)
        return True


def changes(before: MasterChoices, after: MasterChoices) -> list[tuple[str, dict[str, Any]]]:
    return [
        (action, {"model": label})
        for field, action in (
            ("admitted", "operator_master_admit"),
            ("revoked", "operator_master_revoke"),
        )
        for label in sorted(set(getattr(after, field)) - set(getattr(before, field)))
    ]


def print_change(model: str, *, admit: bool, changed: bool) -> None:
    if changed:
        ui.ok(t("Admitted {model}.", model=model) if admit else t("Revoked {model}.", model=model))
    else:
        ui.note(
            t("{model} is already admitted.", model=model)
            if admit
            else t("{model} is already revoked.", model=model)
        )


def report_decision(model: str, *, admit: bool, changed: bool, as_json: bool) -> dict[str, Any]:
    data = {"model": model, "admitted": admit, "revoked": not admit, "changed": changed}
    if as_json:
        click.echo(json.dumps(data, ensure_ascii=False))
    else:
        print_change(model, admit=admit, changed=changed)
    return data


@click.group("master", invoke_without_command=True)
@db_option
@json_option
@click.pass_context
def master(ctx: click.Context, db_path: Path | None, as_json: bool) -> None:
    """List master models or manage benchmark admissions."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_command, db_path=db_path, as_json=as_json)


@master.command("list")
@db_option
@json_option
def list_command(db_path: Path | None, as_json: bool) -> None:
    """List master models and their latest checks."""
    config = load_config().load()
    with closing(connect(db_path)) as conn:
        data = rows(conn, config)
    if as_json:
        click.echo(json.dumps({"models": data}, ensure_ascii=False))
    else:
        print_list(data)


@master.command("check")
@click.argument("model", metavar="PROVIDER/MODEL")
@db_option
@json_option
def check_command(model: str, db_path: Path | None, as_json: bool) -> None:
    """Check one master model. A completed check exits 0 even if it did not pass."""
    config = load_config().load()
    with closing(connect(db_path)) as conn:
        data = run_check(Store(conn), config, model, as_json=as_json)
    if as_json:
        click.echo(json.dumps(data, ensure_ascii=False))
    else:
        print_check(data)


def _decision(db_path: Path | None, model: str, *, admit: bool, as_json: bool) -> None:
    manager = load_config()
    model = valid_label(model, manager.load())
    with closing(connect(db_path)) as conn:
        changed = change(conn, manager, model, admit=admit, journal=True)
    report_decision(model, admit=admit, changed=changed, as_json=as_json)


@master.command("admit")
@click.argument("model", metavar="PROVIDER/MODEL")
@db_option
@json_option
def admit_command(model: str, db_path: Path | None, as_json: bool) -> None:
    """Admit a master model with a current passing check."""
    _decision(db_path, model, admit=True, as_json=as_json)


@master.command("revoke")
@click.argument("model", metavar="PROVIDER/MODEL")
@db_option
@json_option
def revoke_command(model: str, db_path: Path | None, as_json: bool) -> None:
    """Exclude a model from master selection."""
    _decision(db_path, model, admit=False, as_json=as_json)
