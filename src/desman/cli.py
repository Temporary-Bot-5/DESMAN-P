"""CLI for DESMAN(P)."""

from __future__ import annotations

from pathlib import Path

import click

from desman import __version__
from desman.config import ConfigManager, default_config_path, reconfigure, run_initial_config
from desman.core import db
from desman.info import print_project_info


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
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Database file (default: $DESMAN_DATA_DIR/desman.db or ~/.local/share/desman/desman.db).",
)
def init_db(db_path: Path | None) -> None:
    """Create the database or apply pending migrations. Safe to re-run."""
    path = db_path or db.default_db_path()
    conn = db.connect(path)
    try:
        applied = db.migrate(conn)
    except db.SchemaVersionError as e:
        raise click.ClickException(str(e)) from e
    finally:
        conn.close()
    click.echo(f"{path}: schema v{db.SCHEMA_VERSION} ({applied} migration(s) applied)")
