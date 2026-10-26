"""CLI for DESMAN(P)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from desman import __version__
from desman.config import ConfigManager, default_config_path, reconfigure, run_initial_config
from desman.core import db
from desman.core.audit import AuditEntry, check_history, history
from desman.core.store import Store
from desman.info import print_project_info

F = TypeVar("F", bound=Callable[..., Any])


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
        raise click.ClickException(f"no database at {path} (nothing has run yet?)")
    try:
        return Store(db.open_db(path))
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="desman")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)"""
    if ctx.invoked_subcommand is None:
        # Bare `desman` is the interactive entry point: configure on first run, then show info.
        config_manager = ConfigManager(default_config_path())
        if not config_manager.is_configured():
            run_initial_config(config_manager)
        print_project_info(config_manager)


@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    if ctx.parent:
        click.echo(ctx.parent.get_help())


@cli.command()
def config() -> None:
    """Run/re-run DESMAN configuration."""
    reconfigure(ConfigManager(default_config_path()))


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
    click.echo(f"{path}: schema v{db.SCHEMA_VERSION} ({applied} migration(s) applied)")


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
        entries = history(store.conn, task_id)
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
        click.echo(f"Task {task_id}  {status.value}")
        click.echo(f"Goal: {row['goal']}")
        for entry in entries:
            click.echo(_format_entry(entry, labels))
    if problems:
        click.echo("History is incomplete:", err=True)
        for problem in problems:
            click.echo(f"  {problem}", err=True)
        raise SystemExit(1)


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
