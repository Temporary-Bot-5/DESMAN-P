"""Inspect and prepare the sandbox used to run generated code."""

import json
from typing import TypedDict

import click

from desman import ui
from desman.config_cli import json_option
from desman.core.sandbox import DOCKER_IMAGE, Sandbox, SandboxUnavailable
from desman.i18n import t


class Status(TypedDict):
    backend: str
    available: bool
    reason: str | None
    image: str | None
    image_present: bool | None


def sandbox_status(*, display: bool = True) -> Status:
    box = Sandbox()
    reason = box.unavailable()
    present = box.image_present
    result = Status(
        backend=box.backend,
        available=reason is None,
        reason=reason,
        image=DOCKER_IMAGE if box.backend == "docker" else None,
        image_present=present,
    )
    if display:
        ui.note(t("Sandbox backend: {backend}", backend=box.backend))
        if reason is None:
            ui.ok(t("The sandbox is available."))
        else:
            ui.warn(t("The sandbox is unavailable: {reason}", reason=reason))
        if box.backend == "docker":
            state = {
                True: t("Image present"),
                False: t("Image missing"),
                None: t("Image status unknown"),
            }[present]
            ui.note(t("Pinned sandbox image: {image} ({state})", image=DOCKER_IMAGE, state=state))
    return result


def _image_size(box: Sandbox) -> int:
    """Use compressed layer sizes; for an index, report the largest platform image."""
    try:
        result = box.docker_command(["manifest", "inspect", "--verbose", DOCKER_IMAGE])
        if result.returncode == 0:
            entries = json.loads(result.stdout)
            if isinstance(entries, dict):
                entries = [entries]
            sizes = [
                sum(int(layer["size"]) for layer in manifest["layers"])
                for entry in entries
                for manifest in [entry.get("OCIManifest") or entry.get("SchemaV2Manifest") or entry]
            ]
            if sizes and max(sizes) > 0:
                return round(max(sizes) / 1_000_000)
    except (SandboxUnavailable, AttributeError, ValueError, TypeError, KeyError):
        pass
    return 45


def prepare_setup(box: Sandbox) -> str | None:
    """Return the question used by the CLI, wizard and chat; None on bubblewrap."""
    if box.backend == "bwrap":
        return None
    problem = box.docker_problem()
    if problem is not None:
        raise click.ClickException(t("The sandbox is unavailable: {reason}", reason=problem))
    return t("Download the pinned sandbox image (about {size} MB)?", size=_image_size(box))


def setup_image(box: Sandbox) -> bool:
    """Apply an operator's confirmed setup and report the result."""
    if box.backend == "bwrap":
        ui.note(t("On Linux the sandbox uses bubblewrap. No image is needed."))
        return False
    try:
        result = box.docker_command(["pull", DOCKER_IMAGE], timeout=600)
    except SandboxUnavailable as exc:
        raise click.ClickException(t("Docker failed: {error}", error=str(exc))) from exc
    if result.returncode:
        raise click.ClickException(
            t("Docker failed: {error}", error=result.stderr.strip() or str(result.returncode))
        )
    problem = box.reprobe()
    if problem is not None:
        raise click.ClickException(t("The sandbox is unavailable: {reason}", reason=problem))
    ui.ok(t("The sandbox is available."))
    return True


def offer_setup(box: Sandbox, *, yes: bool = False) -> None:
    question = prepare_setup(box)
    if question is None:
        setup_image(box)
    elif yes or ui.confirm(question):
        if yes:
            ui.note(question)
        setup_image(box)


@click.group()
def sandbox() -> None:
    """Manage the sandbox for generated code."""


@sandbox.command()
@json_option
def status(as_json: bool) -> None:
    """Show the sandbox backend and availability."""
    result = sandbox_status(display=not as_json)
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False))


@sandbox.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def setup(yes: bool) -> None:
    """Install the pinned sandbox image if Docker is used."""
    offer_setup(Sandbox(), yes=yes)
