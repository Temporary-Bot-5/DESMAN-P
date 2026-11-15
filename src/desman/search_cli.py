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
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def setup(port: int | None, yes: bool) -> None:
    """Set up SearXNG with Docker (about 100 MB)."""
    if yes or ui.confirm(searxng.consent()):
        searxng.setup(load_config(), port)
    else:
        ui.note(t("Nothing changed."))


@search.command()
@json_option
def status(as_json: bool) -> None:
    """Show SearXNG's container and check its search API."""
    result = searxng.status(load_config(), display=not as_json)
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))


@search.command()
def start() -> None:
    """Start the local SearXNG container."""
    searxng.operate(load_config(), "start")


@search.command()
def stop() -> None:
    """Stop the local SearXNG container."""
    searxng.operate(load_config(), "stop")


@search.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def remove(yes: bool) -> None:
    """Remove the SearXNG container; keep its settings."""
    if yes or ui.confirm(t("Remove the local SearXNG container?")):
        searxng.operate(load_config(), "remove")
    else:
        ui.note(t("Nothing changed."))
