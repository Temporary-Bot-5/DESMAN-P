"""Commands for the local search component."""

import json

import click

from desman import searxng, ui
from desman.config_cli import json_option, load_config
from desman.i18n import t


@click.group()
def search() -> None:
    """Manage the local SearXNG search engine."""


@search.command()
@click.option("--port", type=click.IntRange(1, 65535))
@click.option("--native", is_flag=True, help="Use the Python program.")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def setup(port: int | None, native: bool, yes: bool) -> None:
    """Set up local SearXNG."""
    ui.note(searxng.consent(native))
    if yes:
        searxng.setup(load_config(), port, force_native=native)
    else:
        flag = " --native" if native else ""
        port_flag = f" --port {port}" if port is not None else ""
        ui.note(
            t(
                "Nothing changed. Confirm with: {command}",
                command=f"desman search setup{flag}{port_flag} --yes",
            )
        )


@search.command()
@json_option
def status(as_json: bool) -> None:
    """Show the local SearXNG installation and check its search API."""
    result = searxng.status(load_config(), display=not as_json)
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))


@search.command()
def start() -> None:
    """Start local SearXNG."""
    searxng.operate(load_config(), "start")


@search.command()
def stop() -> None:
    """Stop local SearXNG."""
    searxng.operate(load_config(), "stop")


@search.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def remove(yes: bool) -> None:
    """Remove local SearXNG."""
    if yes:
        searxng.operate(load_config(), "remove")
    else:
        ui.note(t("Remove local SearXNG?"))
        ui.note(t("Nothing changed. Confirm with: {command}", command="desman search remove --yes"))
