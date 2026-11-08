"""Scoped module choices, with plain data functions shared by the CLI and page."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import click
from rich.text import Text

from desman import actions, live, providers, tuning_cli, ui
from desman.config import Config
from desman.config_cli import json_option, load_config
from desman.core import calls, choices, db, holds
from desman.core.models import Kind
from desman.core.report import task_report
from desman.core.store import Store
from desman.i18n import t
from desman.llm import router
from desman.llm.openai_compat import key_id


def scope_id(
    conn: sqlite3.Connection, task: str | None, session: str | None
) -> tuple[choices.Scope, str]:
    if (task is None) == (session is None):
        raise click.ClickException(t("Choose exactly one of --task and --session."))
    if task is not None:
        return "task", actions.find_task(Store(conn), task)
    rows = conn.execute(
        "SELECT id FROM sessions WHERE (id = ? OR substr(id, 1, length(?)) = ?) "
        "AND deleted_at IS NULL AND terminal = 0",
        (session, session, session),
    ).fetchall()
    if len(rows) != 1:
        raise click.ClickException(t("Unknown or ambiguous session ID."))
    return "session", str(rows[0]["id"])


def check_role(scope: choices.Scope, role: str | None) -> None:
    """Refuse a role that no call of `scope` reads (`choices.ROLES_BY_SCOPE`)."""
    if role is None:
        return
    if role not in choices.ROLES_BY_SCOPE[scope]:
        raise click.ClickException(
            t(
                "{role} cannot be chosen for one {scope}: only {roles}. Choose it for the role "
                "with `desman tuning`.",
                role=role,
                scope=scope,
                roles=", ".join(choices.ROLES_BY_SCOPE[scope]),
            )
        )


def preview(
    conn: sqlite3.Connection,
    config: Config,
    scope: choices.Scope,
    identity: str,
    role: str,
    model: str | None = None,
    key: str | None = None,
    *,
    fingerprints: dict[str, str | None] | None = None,
) -> tuple[choices.Choice, choices.Choice]:
    check_role(scope, role)
    try:
        choices.validate(conn, scope, identity, role)
        old = choices.get(conn, scope, identity, role)
        new = choices.Choice(
            model=old.model if model is None else None if model == "auto" else model,
            key=old.key if key is None else tuning_cli.resolve_key(config, key),
        )
    except ValueError:
        raise click.ClickException(t("Invalid module choice.")) from None
    if fingerprints is not None and role in fingerprints:
        new = choices.Choice(
            model=new.model,
            key=tuning_cli.resolved_key(config, role, new.model, fingerprints[role]),
        )
    return old, new


def change_lines(role: str, old: choices.Choice, new: choices.Choice, config: Config) -> list[str]:
    keys = live.key_labels(config)

    def shown(choice: choices.Choice) -> str:
        return f"{choice.model or 'auto'} · {keys.get(choice.key, t('missing')) if choice.key else 'auto'}"

    return [t("{what}: {old} → {new}", what=role, old=shown(old), new=shown(new))]


def set_module(
    conn: sqlite3.Connection,
    config: Config,
    scope: choices.Scope,
    identity: str,
    role: str,
    model: str | None = None,
    key: str | None = None,
    *,
    fingerprints: dict[str, str | None] | None = None,
) -> tuple[choices.Choice, choices.Choice]:
    _, new = preview(conn, config, scope, identity, role, model, key, fingerprints=fingerprints)
    return choices.set_choice(conn, scope, identity, role, new)


def modules_data(
    conn: sqlite3.Connection, config: Config, task_id: str | None = None
) -> list[dict[str, Any]]:
    """One safe record per task and role/kind that worked or is assigned work."""
    store = Store(conn)
    if task_id is not None:
        ids = [actions.find_task(store, task_id)]
    else:
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT t.id FROM tasks t WHERE t.hidden_at IS NULL AND EXISTS (SELECT 1 FROM subtasks s WHERE "
                "s.task_id = t.id AND s.status IN "
                "('pending', 'ready', 'running', 'review', 'needs_human', 'paused')) "
                "ORDER BY t.pinned DESC, t.created_at DESC"
            )
        ]
    result = []
    key_labels = live.key_labels(config)
    role_labels = {r["role"]: r["label"] for r in tuning_cli.role_rows(config)}
    for identity in ids:
        report = task_report(store, identity)
        network = store.find_task(identity)["origin"] == "network"
        modules = calls.modules(conn, identity)
        roles = {s.kind.value for s in report.subtasks}
        # Executors and review are assigned by the stored graph. Planning/decomposition
        # appear once they actually have task-attributed calls.
        roles.add("reviewer")
        history = conn.execute(
            "SELECT c.role, s.kind, COUNT(*) AS calls, "
            "SUM(c.prompt_tokens + c.completion_tokens) AS tokens "
            "FROM model_calls c LEFT JOIN subtasks s ON s.id = c.subtask_id "
            "WHERE c.task_id = ? GROUP BY c.role, "
            "CASE WHEN c.role = 'executor' THEN s.kind ELSE NULL END",
            (identity,),
        ).fetchall()
        used = {(r["kind"] or "text") if r["role"] == "executor" else r["role"]: r for r in history}
        roles.update(r for r in used if r in tuning_cli.ROLES)
        model_history = conn.execute(
            "SELECT c.role, s.kind, c.provider, c.model, COUNT(*) AS calls, "
            "COALESCE(SUM(c.prompt_tokens + c.completion_tokens), 0) AS tokens "
            "FROM model_calls c LEFT JOIN subtasks s ON s.id = c.subtask_id "
            "WHERE c.task_id = ? GROUP BY c.role, "
            "CASE WHEN c.role = 'executor' THEN s.kind ELSE NULL END, c.provider, c.model",
            (identity,),
        ).fetchall()
        model_usage = {
            (
                (r["kind"] or "text") if r["role"] == "executor" else r["role"],
                f"{r['provider']}/{r['model']}",
            ): r
            for r in model_history
        }
        by_id = {s.id: s for s in report.subtasks}
        for role in tuning_cli.ROLES:
            if role not in roles or role == "interlocutor":
                continue
            own = choices.Choice() if network else choices.get(conn, "task", identity, role)
            cfg = choices.effective(config, role, own)
            kind = Kind(role) if role in ("text", "code", "analysis") else None
            call_role = calls.Role.EXECUTOR if kind else calls.Role(role)
            try:
                lineup = router.lineup(cfg, call_role, kind=kind)
            except router.NoProvider:
                lineup = []
            if network:
                lineup = [c for c in lineup if c.provider.id in config.network.donated]
            removed = {
                (h.source, h.model)
                for h in holds.active(
                    conn, "network" if network else "task", "" if network else identity
                )
                if h.state == "removed"
            }
            lineup = [
                c
                for c in lineup
                if any((source, c.label) not in removed for source in router.sources(cfg, c))
            ]
            latest_call = conn.execute(
                "SELECT c.provider, c.model, c.key_id FROM model_calls c LEFT JOIN subtasks s ON s.id = c.subtask_id "
                "WHERE c.task_id = ? AND c.role = ? AND c.error_kind IS NULL "
                "AND (? IS NULL OR COALESCE(s.kind, 'text') = ?) ORDER BY c.seq DESC LIMIT 1",
                (
                    identity,
                    call_role.value,
                    kind.value if kind else None,
                    kind.value if kind else None,
                ),
            ).fetchone()
            first = lineup[0] if lineup else None
            if latest_call and (
                (
                    latest_call["key_id"] or latest_call["provider"],
                    f"{latest_call['provider']}/{latest_call['model']}",
                )
                in removed
            ):
                latest_call = None
            if latest_call:
                first = router.Choice(providers.find(latest_call["provider"]), latest_call["model"])
            elif (
                not removed
                and router.preferred(cfg, call_role, kind) is None
                and router.preferred_key(cfg, call_role, kind) is None
            ):
                first = None
            fingerprint = (
                latest_call["key_id"] if latest_call else router.preferred_key(cfg, call_role, kind)
            )
            if first:
                sources = [
                    source
                    for source in router.sources(cfg, first)
                    if (source, first.label) not in removed
                ]
                if fingerprint not in sources:
                    fingerprint = next(iter(sources), None)
            applicable = next(
                (
                    k
                    for k in tuning_cli.key_choices(config)
                    if k["fingerprint"] == fingerprint
                    and first is not None
                    and k["provider"] == first.provider.id
                ),
                None,
            )
            prior = used.get(role)
            shown_usage = model_usage.get((role, first.label)) if first else None
            relevant = [
                m
                for m in modules
                if m.role == call_role
                and (
                    kind is None
                    or (
                        by_id.get(m.subtask_id or "") is not None
                        and by_id[m.subtask_id or ""].kind == kind
                    )
                )
            ]
            latest = max(relevant, key=lambda m: m.last_seq) if relevant else None
            state = (
                live.Progress(store, identity, config)._state(latest, by_id, modules)[0]
                if latest
                else t("idle")
            )
            result.append(
                {
                    "task_id": identity,
                    "task_ref": store.short_id(identity),
                    "role": role,
                    "label": role_labels[role],
                    "model": first.label if first else None,
                    "provider": first.provider.id if first else None,
                    "key_fingerprint": fingerprint if applicable else None,
                    "key_label": key_labels.get(fingerprint or "", "auto")
                    if applicable
                    else "auto",
                    "source": "task"
                    if own != choices.Choice()
                    else "role"
                    if (getattr(config.tuning.models, role) or getattr(config.tuning.keys, role))
                    else "auto",
                    "model_calls": shown_usage["calls"] if shown_usage else 0,
                    "model_tokens": shown_usage["tokens"] if shown_usage else 0,
                    "calls": prior["calls"] if prior else 0,
                    "tokens": (prior["tokens"] or 0) if prior else 0,
                    "state": state,
                }
            )
    return result


def print_modules(rows: list[dict[str, Any]], as_json: bool = False) -> None:
    if as_json:
        click.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    table = ui.table(
        t("Task"), t("Role"), t("Model"), t("Key"), t("Source"), t("Calls"), t("Tokens"), t("State")
    )
    sources = {"task": t("Task"), "role": t("Role"), "auto": "auto"}
    for row in rows:
        shown = {**row, "source": sources[row["source"]]}
        table.add_row(
            *(
                Text(str(shown[k] if shown[k] is not None else "auto"))
                for k in (
                    "task_ref",
                    "label",
                    "model",
                    "key_label",
                    "source",
                    "calls",
                    "tokens",
                    "state",
                )
            )
        )
    ui.out(table)


@click.group("modules", invoke_without_command=True)
@click.option("--task", metavar="REF", help=t("One task (its reference)."))
@json_option
@click.pass_context
def modules(ctx: click.Context, task: str | None, as_json: bool) -> None:
    """Models and keys for one task or chat. Roles: planner, reviewer, decomposer, text, code, analysis, interlocutor."""
    if ctx.invoked_subcommand is None:
        show_modules(task, as_json)


def show_modules(task: str | None, as_json: bool) -> None:
    config = load_config().load()
    with closing(db.open_db()) as conn:
        print_modules(modules_data(conn, config, task), as_json)
        if not as_json:
            print_holds(conn)


def print_holds(conn: sqlite3.Connection) -> None:
    for hold in holds.listed(conn):
        ui.note(
            f"{hold.source} · {hold.model} · {hold.scope} {hold.scope_id} · {t('stopped') if hold.state == 'stopped' else t('removed')}"
        )


@modules.command("show")
@click.option("--task", metavar="REF", help=t("One task (its reference)."))
@json_option
def show(task: str | None, as_json: bool) -> None:
    """Show the modules of active tasks, or one task."""
    show_modules(task, as_json)


@modules.command("set")
@click.argument("role", type=click.Choice(tuning_cli.ROLES))
@click.option("--task", metavar="REF", help=t("One task (its reference)."))
@click.option(
    "--session", metavar="ID", help=t("One chat (its id as /sessions shows it): the interlocutor.")
)
@click.option(
    "--model", metavar="PROVIDER/MODEL|auto", help="The model asked first; auto: the role's."
)
@click.option(
    "--key",
    metavar="PROVIDER:KEY|auto",
    help="The key tried first (label or #N); auto: the role's.",
)
def set_command(
    role: str, task: str | None, session: str | None, model: str | None, key: str | None
) -> None:
    """Choose a model or key only here. Roles: planner, reviewer, decomposer, text, code, analysis, interlocutor."""
    config = load_config().load()
    if model is None and key is None:
        raise click.ClickException(t("Give --model or --key."))
    with closing(db.open_db()) as conn:
        scope, identity = scope_id(conn, task, session)
        old, new = set_module(conn, config, scope, identity, role, model, key)
        for line in change_lines(role, old, new, config):
            ui.ok(line)


@modules.command("clear")
@click.argument("role", type=click.Choice(tuning_cli.ROLES), required=False)
@click.option("--task", metavar="REF", help=t("One task (its reference)."))
@click.option(
    "--session", metavar="ID", help=t("One chat (its id as /sessions shows it): the interlocutor.")
)
def clear_command(role: str | None, task: str | None, session: str | None) -> None:
    """Clear choices only here. Roles: planner, reviewer, decomposer, text, code, analysis, interlocutor."""
    with closing(db.open_db()) as conn:
        scope, identity = scope_id(conn, task, session)
        check_role(scope, role)
        choices.clear(conn, scope, identity, role)
    ui.ok(t("Module choices cleared."))


def hold_scope(
    conn: sqlite3.Connection,
    task: str | None = None,
    chat: str | None = None,
    network: bool = False,
) -> tuple[holds.Scope, str]:
    if sum((task is not None, chat is not None, network)) > 1:
        raise ValueError("choose one hold scope")
    if network:
        return "network", ""
    if task is not None or chat is not None:
        return scope_id(conn, task, chat)
    return "all", ""


def assigned_module(
    conn: sqlite3.Connection, config: Config, row: dict[str, Any]
) -> dict[str, Any]:
    """Resolve Auto's provisional assignment for holds/cards, leaving its public label Auto."""
    if row["model"] is not None:
        return row
    identity, role = row["task_id"], row["role"]
    network = Store(conn).find_task(identity)["origin"] == "network"
    cfg = choices.effective(config, role, choices.get(conn, "task", identity, role))
    kind = Kind(role) if role in {k.value for k in Kind} else None
    try:
        lineup = router.lineup(cfg, calls.Role.EXECUTOR if kind else calls.Role(role), kind=kind)
    except router.NoProvider:
        return row
    removed = {
        (h.source, h.model)
        for h in holds.active(conn, "network" if network else "task", "" if network else identity)
        if h.state == "removed"
    }
    for choice in lineup:
        if network and choice.provider.id not in config.network.donated:
            continue
        sources = [
            source
            for source in router.sources(cfg, choice)
            if (source, choice.label) not in removed
        ]
        if sources:
            preferred = router.preferred_key(
                cfg, calls.Role.EXECUTOR if kind else calls.Role(role), kind
            )
            source = preferred if preferred in sources else sources[0]
            return {
                **row,
                "model": choice.label,
                "provider": choice.provider.id,
                "key_fingerprint": None if choice.provider.local else source,
            }
    return row


def hold_change(conn: sqlite3.Connection, config: Config, operation: str, hold: holds.Hold) -> None:
    """The sole page/CLI/catalog body; affected work is computed by code, not supplied by a model."""
    if operation == "stop":
        tasks = set()
        if hold.scope in {"all", "task"}:
            rows = modules_data(conn, config, hold.scope_id if hold.scope == "task" else None)
            for row in rows:
                row = assigned_module(conn, config, row)
                if row["model"] == hold.model and module_source(config, row) == hold.source:
                    tasks.add(row["task_id"])
            for identity in {row["task_id"] for row in rows}:
                modules = calls.modules(conn, identity)
                for module in modules:
                    if module.subtask_id is None:
                        continue
                    subtask = Store(conn).get_subtask(module.subtask_id)
                    if subtask.status.value not in {
                        "ready",
                        "running",
                        "review",
                        "paused",
                        "needs_human",
                    }:
                        continue
                    latest = max(m.last_seq for m in modules if m.subtask_id == module.subtask_id)
                    if (
                        module.last_seq == latest
                        and (module.key_id or module.provider) == hold.source
                        and f"{module.provider}/{module.model}" == hold.model
                    ):
                        tasks.add(identity)
        holds.stop(conn, hold, tasks)
    elif operation == "continue":
        holds.continue_(conn, hold)
    elif operation == "remove":
        holds.remove(conn, hold)
    elif operation == "restore":
        holds.restore(conn, hold)
    else:
        raise ValueError("unknown module hold operation")


def module_source(config: Config, row: dict[str, Any]) -> str:
    if row.get("provider") in {"ollama", "llamacpp"}:
        return str(row["provider"])
    if row.get("key_fingerprint"):
        return str(row["key_fingerprint"])
    provider = next((p for p in providers.PROVIDERS if p.id == row.get("provider")), None)
    keys = config.keys(provider.key_env or "") if provider else []
    return key_id(keys[0].value) if keys else ""


def hold_question(conn: sqlite3.Connection, config: Config, hold: holds.Hold) -> str:
    from desman.net.node import active_capabilities

    if hold.scope in {"all", "network"} and config.network.enabled:
        lent = active_capabilities(config, conn)
        if lent and all(
            (c.key or c.provider) == hold.source and f"{c.provider}/{c.model}" == hold.model
            for c in lent
        ):
            return t("Remove this module? If it is the last lent module, DESMAN-Global suspends.")
    return t("Remove this module?")


def _hold_command(
    operation: str, source: str, model: str, task: str | None, chat: str | None, network: bool
) -> None:
    config = load_config().load()
    with closing(db.open_db()) as conn:
        try:
            scope, identity = hold_scope(conn, task, chat, network)
            hold_change(
                conn,
                config,
                operation,
                holds.Hold(source=source, model=model, scope=scope, scope_id=identity),
            )
        except ValueError:
            raise click.ClickException(t("Invalid module hold.")) from None
    ui.ok(t("Module hold updated."))


def _register_hold_command(operation: str) -> None:
    @modules.command(operation, help=t("Stop, continue, remove or restore a module."))
    @click.argument("source")
    @click.argument("model")
    @click.option("--task", metavar="REF", help=t("One task (its reference)."))
    @click.option(
        "--chat", metavar="ID", help=t("One chat (its id as /sessions shows it): the interlocutor.")
    )
    @click.option("--network", is_flag=True, help=t("Modules lent to DESMAN-Global."))
    def command(source: str, model: str, task: str | None, chat: str | None, network: bool) -> None:
        _hold_command(operation, source, model, task, chat, network)


for _operation in ("stop", "continue", "remove", "restore"):
    _register_hold_command(_operation)
