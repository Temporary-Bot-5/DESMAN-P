"""`desman models`: a model from Hugging Face for a local server (ROADMAP 4.5b).

`show` reads what a model is; `add` has the server download it and makes it the server's model.
Both say the size and the machine's free memory first. Non-interactive: `add` refuses a model
that does not fit in the free memory unless `--force` says the person knows.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import click
import httpx

from desman import local_models, ui
from desman.config_cli import json_option, load_config
from desman.i18n import t
from desman.llm import hub


@click.group()
def models() -> None:
    """Models from Hugging Face for a local server (Ollama, llama.cpp)."""


@models.command("show")
@click.argument("model")
@json_option
def show(model: str, as_json: bool) -> None:
    """What MODEL (author/model[:quantization]) is: its size, and whether it fits in the
    memory that is free. Downloads nothing."""
    config = load_config().load()
    ref = local_models.parse(model)
    with httpx.Client() as client:
        found = local_models.look_up(config, ref, client)
    machine = hub.memory()
    if as_json:
        click.echo(json.dumps(model_json(found, machine), indent=2))
        return
    print_model(found, machine)


@models.command("add")
@click.argument("model")
@click.option(
    "--server",
    type=click.Choice(local_models.TARGETS),
    default="ollama",
    show_default=True,
    help="The local server that runs it.",
)
@click.option("--force", is_flag=True, help="Add it even if it does not fit in the free memory.")
def add(model: str, server: str, force: bool) -> None:
    """Have a local server download MODEL from Hugging Face (author/model[:quantization], GGUF
    only) and make it that server's model."""
    manager = load_config()
    config = manager.load()
    ref = local_models.parse(model)
    target = local_models.target(server)
    with httpx.Client() as client:
        found = local_models.look_up(config, ref, client)
        machine = hub.memory()
        print_model(found, machine, quiet_warnings=True)
        if hub.fits(found, machine) is False and not force:
            raise click.ClickException(
                t(
                    "{model} does not fit in the free memory: it would run very slowly, or not "
                    "at all. Add --force to add it anyway.",
                    model=found.ref.spec,
                )
            )
        lines, warnings = local_models.plan(config, found, target)
        for line in lines:
            ui.note(line)
        for warning in warnings:
            ui.warn(warning)
        local_models.fetch(config, found, target, client)
    with manager.edit() as saved:
        name = local_models.use(saved, found, target)
    ui.ok(t("{server} model set to {model}.", server=target.name, model=name))


def model_json(model: hub.Gguf, machine: hub.Memory) -> dict[str, object]:
    return {
        "model": model.ref.spec,
        "page": model.ref.page,
        "files": list(model.files),
        "size": model.size,
        "memory": model.memory,
        "gated": model.gated,
        "quantizations": list(model.quants),
        "free_memory": asdict(machine),
        "fits": hub.fits(model, machine),
    }


def print_model(model: hub.Gguf, machine: hub.Memory, *, quiet_warnings: bool = False) -> None:
    lines, warnings = local_models.describe(model, machine)
    for line in lines:
        ui.out(line)
    if len(model.quants) > 1:
        ui.note(t("Other quantizations: {quants}", quants=", ".join(model.quants)))
    if not quiet_warnings:
        for warning in warnings:
            ui.warn(warning)


@models.command("remove")
@click.argument("model")
def remove(model: str) -> None:
    """Remove one installed Ollama MODEL, keeping saved choices."""
    from contextlib import closing

    from desman.core import db, operator

    config = load_config().load()
    with closing(db.open_db()) as conn, httpx.Client() as client:
        found = local_models.removal_model(config, model, client)
        references = local_models.remove(config, found.name, client, conn)
        operator.record(
            conn,
            "operator_command",
            {
                "command": "models remove",
                "args": {"model": model},
                "missing_choices": references,
                "typed": True,
            },
        )
