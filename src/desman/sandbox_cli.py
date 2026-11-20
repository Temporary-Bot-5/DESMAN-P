"""Inspect and prepare the sandbox used to run generated code."""

import json
from typing import NotRequired, TypedDict

import click

from desman import ui
from desman.config_cli import json_option
from desman.core.sandbox import (
    DOCKER_IMAGE,
    Sandbox,
    SandboxUnavailable,
    install_wasi,
    wasi_runtime,
)
from desman.i18n import t


class Status(TypedDict):
    backend: str
    available: bool
    reason: str | None
    image: str | None
    image_present: bool | None
    wasi_present: NotRequired[bool]


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
    if box.backend == "wasi":
        result["wasi_present"] = box.wasi_present
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
        if box.backend == "wasi":
            _show_wasi_files(result["wasi_present"])
    return result


def _show_wasi_files(present: bool) -> None:
    ui.note(
        t("WebAssembly sandbox files are present.")
        if present
        else t("WebAssembly sandbox files are missing.")
    )


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


def _needs_image(box: Sandbox) -> bool:
    if box.backend == "bwrap":
        ui.note(t("On Linux the sandbox uses bubblewrap. No image is needed."))
        return False
    problem = box.docker_problem()
    if problem is not None:
        raise click.ClickException(t("The sandbox is unavailable: {reason}", reason=problem))
    try:
        present = box.docker_image_present()
    except SandboxUnavailable as exc:
        raise click.ClickException(t("Docker failed: {error}", error=str(exc))) from exc
    if present:
        ui.note(
            t(
                "Pinned sandbox image: {image} ({state})",
                image=DOCKER_IMAGE,
                state=t("Image present"),
            )
        )
    return not present


def prepare_setup(box: Sandbox) -> str | None:
    """Describe the download when the selected backend needs setup."""
    if box.backend == "wasi":
        if not wasi_runtime():
            raise click.ClickException(t("The WebAssembly runtime (wasmtime) could not be loaded."))
        if box.wasi_present:
            _show_wasi_files(True)
            return None
        return t(
            "Download the pinned WebAssembly sandbox files (about 24 MB download, 52 MB on disk)?"
        )
    if not _needs_image(box):
        return None
    return t("Download the pinned sandbox image (about {size} MB)?", size=_image_size(box))


def apply_setup(box: Sandbox) -> bool:
    """Apply an operator's confirmed setup and report the result."""
    if box.backend == "wasi":
        if not wasi_runtime():
            raise click.ClickException(t("The WebAssembly runtime (wasmtime) could not be loaded."))
        try:
            installed = install_wasi(box.wasi_dir)
        except SandboxUnavailable as exc:
            raise click.ClickException(
                t("WebAssembly sandbox setup failed: {error}", error=str(exc))
            ) from exc
        box.reprobe()
        ui.ok(t("WebAssembly sandbox files are present."))
        return installed
    if not _needs_image(box):
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


def offer_setup(box: Sandbox) -> None:
    question = prepare_setup(box)
    if question is not None and ui.confirm(question):
        apply_setup(box)


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
    """Install the pinned files for the code sandbox."""
    box = Sandbox()
    question = prepare_setup(box)
    if question is None:
        return
    ui.note(question)
    if yes:
        apply_setup(box)
    else:
        ui.note(t("Nothing changed. Confirm with: {command}", command="desman sandbox setup --yes"))
