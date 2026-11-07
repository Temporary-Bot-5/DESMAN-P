"""Scoped module choices, with plain data functions shared by the CLI and page."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Any

import click
from rich.text import Text

from desman import actions, live, tuning_cli, ui
from desman.config import Config
from desman.config_cli import json_option, load_config
from desman.core import calls, choices, db
from desman.core.models import Kind
from desman.core.report import task_report
from desman.core.store import Store
from desman.i18n import t
from desman.llm import router


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
) -> tuple[choices.Choice, choices.Choice]:
    _, new = preview(conn, config, scope, identity, role, model, key)
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
                "SELECT t.id FROM tasks t WHERE EXISTS (SELECT 1 FROM subtasks s WHERE "
                "s.task_id = t.id AND s.status IN "
                "('pending', 'ready', 'running', 'review', 'needs_human', 'paused')) "
                "ORDER BY t.created_at DESC"
            )
        ]
    result = []
    key_labels = live.key_labels(config)
    role_labels = {r["role"]: r["label"] for r in tuning_cli.role_rows(config)}
    for identity in ids:
        report = task_report(store, identity)
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
        by_id = {s.id: s for s in report.subtasks}
        for role in tuning_cli.ROLES:
            if role not in roles or role == "interlocutor":
                continue
            own = choices.get(conn, "task", identity, role)
            cfg = choices.effective(config, role, own)
            kind = Kind(role) if role in ("text", "code", "analysis") else None
            call_role = calls.Role.EXECUTOR if kind else calls.Role(role)
            try:
                lineup = router.lineup(cfg, call_role, kind=kind)
            except router.NoProvider:
                lineup = []
            first = lineup[0] if lineup else None
            fingerprint = router.preferred_key(cfg, call_role, kind)
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
@click.option("--task", metavar="REF", help="One task (its reference).")
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


@modules.command("show")
@click.option("--task", metavar="REF", help="One task (its reference).")
@json_option
def show(task: str | None, as_json: bool) -> None:
    """Show the modules of active tasks, or one task."""
    show_modules(task, as_json)


@modules.command("set")
@click.argument("role", type=click.Choice(tuning_cli.ROLES))
@click.option("--task", metavar="REF", help="One task (its reference).")
@click.option(
    "--session", metavar="ID", help="One chat (its id as /sessions shows it): the interlocutor."
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
@click.option("--task", metavar="REF", help="One task (its reference).")
@click.option(
    "--session", metavar="ID", help="One chat (its id as /sessions shows it): the interlocutor."
)
def clear_command(role: str | None, task: str | None, session: str | None) -> None:
    """Clear choices only here. Roles: planner, reviewer, decomposer, text, code, analysis, interlocutor."""
    with closing(db.open_db()) as conn:
        scope, identity = scope_id(conn, task, session)
        check_role(scope, role)
        choices.clear(conn, scope, identity, role)
    ui.ok(t("Module choices cleared."))
