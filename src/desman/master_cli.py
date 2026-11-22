"""The person's measured admissions for master models (ROADMAP M.3)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

import click
import httpx
from pydantic import TypeAdapter, ValidationError
from rich.text import Text

from desman import bench, providers, runs, ui
from desman.config import Config, ConfigManager, MasterChoices, ModelLabel
from desman.config_cli import json_option, load_config
from desman.core import db, operator
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.i18n import t
from desman.llm import catalog, check, router
from desman.protected_cli import connect, db_option
from desman.providers import CLOUD

_LABEL = TypeAdapter(ModelLabel)


def valid_label(model: str, config: Config) -> str:
    if providers.redact_keys(model, config)[1]:
        raise click.ClickException(
            t("an argument looked like an API key, so the command was dropped")
        )
    try:
        return _LABEL.validate_python(model)
    except ValidationError as e:
        raise click.ClickException(t("Use a model label in the form PROVIDER/MODEL.")) from e


def _summary(row: sqlite3.Row, result: bench.Result) -> dict[str, Any]:
    return {
        "id": row["id"],
        "model": row["model"],
        "checked_at": row["checked_at"],
        "version": row["version"],
        "passed": bool(row["passed"]),
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


def _verdict(passed: bool) -> str:
    return t("Passed") if passed else t("Did not pass")


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
                verdict=_verdict(last["passed"]),
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
            result = bench.run(
                Store(conn), "real", config.tuning, config=config, master=choice.label
            )
    except router.NoProvider as e:
        raise click.ClickException(str(e)) from e
    passed = bench.master_passed(result)
    row = store.conn.execute(
        "INSERT INTO master_checks (model, version, passed, result_json) VALUES (?, ?, ?, ?) "
        "RETURNING id, model, checked_at, version, passed",
        (choice.label, result.version, passed, result.model_dump_json()),
    ).fetchone()
    return _summary(row, result) | {
        "result": result.model_dump(mode="json"),
        "usage": bench.metrics(result),
    }


def print_check(data: dict[str, Any]) -> None:
    table = ui.table(t("Case"), t("Status"))
    labels = {
        TaskStatus.ACTIVE: t("active"),
        TaskStatus.NEEDS_HUMAN: t("needs human"),
        TaskStatus.DONE: t("done"),
        TaskStatus.FAILED: t("failed"),
        TaskStatus.CANCELLED: t("cancelled"),
    }
    for case in data["result"]["cases"]:
        table.add_row(case["id"], labels[TaskStatus(case["status"])])
    ui.out(table)
    usage = data["usage"]
    ui.note(t("{calls} calls · {tokens} tokens", calls=usage["calls"], tokens=usage["tokens"]))
    ui.note(
        t(
            "{verdict} · {done}/{total} done · {escalated} escalated",
            verdict=_verdict(data["passed"]),
            done=data["done"],
            total=data["total"],
            escalated=usage["escalated"],
        )
    )


def require_pass(conn: sqlite3.Connection, model: str) -> None:
    latest = conn.execute(
        "SELECT passed, version FROM master_checks WHERE model = ? ORDER BY id DESC LIMIT 1",
        (model,),
    ).fetchone()
    if latest is None or not latest["passed"] or latest["version"] != bench.VERSION:
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
        if admit:
            require_pass(conn, model)
        old = config.master
        add, remove = ("admitted", "revoked") if admit else ("revoked", "admitted")
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


def print_change(model: str, *, admit: bool) -> None:
    ui.ok(t("Admitted {model}.", model=model) if admit else t("Revoked {model}.", model=model))


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


def _decision(db_path: Path | None, model: str, *, admit: bool) -> None:
    manager = load_config()
    with closing(connect(db_path)) as conn:
        change(conn, manager, model, admit=admit, journal=True)
    print_change(model, admit=admit)


@master.command("admit")
@click.argument("model", metavar="PROVIDER/MODEL")
@db_option
def admit_command(model: str, db_path: Path | None) -> None:
    """Admit a master model with a current passing check."""
    _decision(db_path, model, admit=True)


@master.command("revoke")
@click.argument("model", metavar="PROVIDER/MODEL")
@db_option
def revoke_command(model: str, db_path: Path | None) -> None:
    """Exclude a model from master selection."""
    _decision(db_path, model, admit=False)
