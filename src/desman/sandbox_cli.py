"""Inspect and prepare the sandbox used to run generated code."""

import json
from typing import Literal, NotRequired, TypedDict

import click

from desman import ui
from desman.config_cli import json_option
from desman.core.sandbox import (
    DOCKER_IMAGE,
    WSL_PIN,
    Sandbox,
    SandboxUnavailable,
    enable_wsl,
    install_wasi,
    install_wsl,
    remove_wsl,
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
    wsl_present: NotRequired[bool]
    wsl_pin: NotRequired[str]


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
    if box.platform == "win32":
        try:
            result["wsl_present"] = (
                box.wsl_distribution_present() if box.wsl_present is None else box.wsl_present
            )
        except SandboxUnavailable:
            result["wsl_present"] = False
        result["wsl_pin"] = WSL_PIN
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
        if "wsl_present" in result:
            _show_wsl_files(result["wsl_present"])
    return result


def _show_wsl_files(present: bool) -> None:
    ui.note(
        t("WSL sandbox distribution {pin} is present.", pin=WSL_PIN)
        if present
        else t("WSL sandbox distribution {pin} is missing.", pin=WSL_PIN)
    )


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
    if box.platform.startswith("linux"):
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


SetupBackend = Literal["bwrap", "docker", "wasi", "wsl", "enable-wsl"]


def setup_backend(box: Sandbox, *, force_wsl: bool = False) -> SetupBackend:
    if force_wsl and box.platform != "win32":
        raise click.ClickException(t("WSL sandbox setup requires Windows."))
    if box.platform.startswith("linux"):
        return "bwrap"
    if not force_wsl and box.docker_problem() is None:
        return "docker"
    if box.wsl_usable():
        return "wsl"
    if force_wsl:
        return "enable-wsl"
    return "wasi" if box.platform in ("win32", "darwin") else "docker"


def prepare_setup(box: Sandbox, backend: SetupBackend | None = None) -> str | None:
    """Describe the download when the selected backend needs setup."""
    backend = backend or setup_backend(box)
    if backend == "enable-wsl":
        return t("Enable WSL? Windows may ask for administrator rights and needs a restart.")
    if backend == "wsl":
        try:
            if box.wsl_distribution_present():
                problem = box.wsl_problem()
                if problem is not None:
                    raise SandboxUnavailable(problem)
                _show_wsl_files(True)
                return None
        except SandboxUnavailable as exc:
            raise click.ClickException(
                t("WSL sandbox setup failed: {error}", error=str(exc))
            ) from exc
        return t(
            "Set up the WSL sandbox (about 4 MB download, 60 MB on disk)? "
            "Installing its packages needs internet access once."
        )
    if backend == "wasi":
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


def apply_setup(box: Sandbox, backend: SetupBackend | None = None) -> bool:
    """Apply an operator's confirmed setup and report the result."""
    backend = backend or setup_backend(box)
    if backend in ("wsl", "enable-wsl"):
        try:
            if backend == "enable-wsl":
                enable_wsl(box)
                ui.ok(t("Restart Windows, then run `desman sandbox setup --wsl --yes`."))
                return False
            installed = install_wsl(box)
        except SandboxUnavailable as exc:
            raise click.ClickException(
                t("WSL sandbox setup failed: {error}", error=str(exc))
            ) from exc
        _show_wsl_files(True)
        return installed
    if backend == "wasi":
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
    backend = setup_backend(box)
    question = prepare_setup(box, backend)
    if question is not None and ui.confirm(question):
        apply_setup(box, backend)


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
@click.option("--wsl", is_flag=True, help="Use the Windows Subsystem for Linux.")
def setup(yes: bool, wsl: bool) -> None:
    """Install the pinned files for the code sandbox."""
    box = Sandbox()
    backend = setup_backend(box, force_wsl=wsl)
    question = prepare_setup(box, backend)
    if question is None:
        return
    ui.note(question)
    if yes:
        apply_setup(box, backend)
    else:
        command = "desman sandbox setup" + (" --wsl" if wsl else "") + " --yes"
        ui.note(t("Nothing changed. Confirm with: {command}", command=command))


def remove_wsl_sandbox(box: Sandbox) -> bool:
    if box.platform != "win32":
        raise click.ClickException(t("WSL sandbox removal requires Windows."))
    try:
        removed = remove_wsl(box)
    except SandboxUnavailable as exc:
        raise click.ClickException(
            t("WSL sandbox removal failed: {error}", error=str(exc))
        ) from exc
    ui.ok(t("The WSL sandbox has been removed."))
    return removed


def wsl_removal_question() -> str:
    return t("Remove the WSL sandbox distribution and its files?")


@sandbox.command()
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.option("--wsl", is_flag=True, required=True, help="Use the Windows Subsystem for Linux.")
def remove(yes: bool, wsl: bool) -> None:
    """Remove the WSL sandbox distribution."""
    if not wsl:
        raise click.UsageError(t("Use --wsl to select the WSL sandbox."))
    ui.note(wsl_removal_question())
    if yes:
        remove_wsl_sandbox(Sandbox())
    else:
        ui.note(
            t(
                "Nothing changed. Confirm with: {command}",
                command="desman sandbox remove --wsl --yes",
            )
        )
