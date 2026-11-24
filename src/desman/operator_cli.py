"""Operator status and key management shared by the terminal and command catalog."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
from pydantic import TypeAdapter, ValidationError
from rich.text import Text

from desman import net, ui
from desman.config import ConfigError, ConfigManager, RegistryUrl
from desman.config_cli import load_config
from desman.core import db, netlog
from desman.global_cli import journal
from desman.i18n import t
from desman.net import operator, sign

_KEY_HELP = "desman operator key import FILE | desman operator key create"


def current_key() -> sign.NodeKey | None:
    try:
        return operator.load_key()
    except (OSError, sign.KeyFileError) as e:
        raise click.ClickException(t("The operator key cannot be used.")) from e


def fingerprint(key: sign.NodeKey | None) -> str | None:
    return None if key is None else key.fingerprint


def show_status(
    manager: ConfigManager, conn: sqlite3.Connection | None = None
) -> dict[str, object]:
    key = current_key()
    if key is None:
        ui.note(t("No operator key. Use {command}.", command=_KEY_HELP))
        return {"fingerprint": None}
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=key.fingerprint)))
    try:
        address = net.operator_registry(manager.load())
    except ConfigError as e:
        raise click.ClickException(str(e)) from e
    if address is None:
        ui.note(t("No registry address. Set it with --registry URL."))
        return {"fingerprint": key.fingerprint, "registry": None}
    ui.out(Text(t("Registry: {address}", address=address)))
    try:
        identity = operator.OperatorClient(address, key).whoami()
    except operator.OperatorError as e:
        raise click.ClickException(
            t("Operator status unavailable ({reason}).", reason=e.reason)
        ) from e
    ui.out(
        Text(
            t(
                "Operator {fingerprint} assigned ({scopes}) until {date}.",
                fingerprint=identity.fingerprint,
                scopes=", ".join(identity.scopes),
                date=identity.until.isoformat(),
            )
        )
    )
    with journal(conn) as opened:
        netlog.record(opened, "operator_status", {"fingerprint": identity.fingerprint})
    return {**identity.model_dump(mode="json"), "registry": address}


@dataclass(frozen=True)
class KeyChange:
    action: Literal["import", "create", "remove"]
    before: str | None
    key: sign.NodeKey | None
    registry: str | None = None

    @property
    def lines(self) -> list[str]:
        if self.action == "remove":
            if self.before is None:
                return [t("No operator key. Use {command}.", command=_KEY_HELP)]
            return [t("Remove operator key {fingerprint}.", fingerprint=self.before)]
        assert self.key is not None
        lines = [t("Install operator key {fingerprint}.", fingerprint=self.key.fingerprint)]
        if self.before is not None and self.before != self.key.fingerprint:
            lines.append(t("Remove operator key {fingerprint}.", fingerprint=self.before))
        if self.registry is not None:
            lines.append(t("Registry: {address}", address=self.registry))
        return lines

    def apply(self, manager: ConfigManager, conn: sqlite3.Connection | None = None) -> str | None:
        if self.action == "remove" and self.before is None:
            return None
        try:
            with journal(conn) as opened, manager.locked(), db.transaction(opened):
                current = current_key()
                if fingerprint(current) != self.before:
                    raise click.ClickException(
                        t("The operator key changed. Run the command again.")
                    )
                config = manager.load() if self.registry is not None else None
                logged = self.before if self.key is None else self.key.fingerprint
                netlog.write(opened, f"operator_key_{self.action}", {"fingerprint": logged})
                if self.key is None:
                    operator.remove_key()
                else:
                    operator.install_key(self.key, replace=self.before not in (None, logged))
                if config is not None:
                    try:
                        config.network.registry = self.registry
                        manager.save(config)
                    except (ConfigError, OSError):
                        if current is None:
                            operator.remove_key()
                        else:
                            operator.install_key(current, replace=True)
                        raise
        except (OSError, sign.KeyFileError) as e:
            raise click.ClickException(t("The operator key cannot be used.")) from e
        except ConfigError as e:
            raise click.ClickException(str(e)) from e
        return logged


def prepare_import(source: Path, registry: str | None, *, replace: bool) -> KeyChange:
    try:
        key = operator.read_key(source)
    except (OSError, sign.KeyFileError) as e:
        raise click.ClickException(t("The operator key cannot be used.")) from e
    before = fingerprint(current_key())
    if before not in (None, key.fingerprint) and not replace:
        raise click.ClickException(
            t("An operator key already exists. Use --replace to replace it.")
        )
    return KeyChange("import", before, key, registry)


def prepare_create(registry: str | None) -> KeyChange:
    if current_key() is not None:
        raise click.ClickException(t("An operator key already exists."))
    return KeyChange("create", None, sign.NodeKey.generate(), registry)


def prepare_remove() -> KeyChange:
    return KeyChange("remove", fingerprint(current_key()), None)


def _registry(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return TypeAdapter(RegistryUrl).validate_python(value)
    except ValidationError as e:
        raise click.ClickException(t("Invalid registry URL.")) from e


@click.group("operator")
def operator_group() -> None:
    """Manage access to the registry as an operator."""


@operator_group.command("status")
def status_command() -> None:
    """Show operator access and its term."""
    show_status(load_config())


@operator_group.group("key")
def key_group() -> None:
    """Manage this machine's operator key."""


@key_group.command("import")
@click.argument("file", type=click.Path(path_type=Path))
@click.option("--registry", metavar="URL")
@click.option("--replace", is_flag=True, help="Replace the existing operator key.")
def import_command(file: Path, registry: str | None, replace: bool) -> None:
    """Import an operator key from a PEM file."""
    change = prepare_import(file, _registry(registry), replace=replace)
    value = change.apply(load_config())
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=value)))


@key_group.command("create")
@click.option("--registry", metavar="URL")
def create_command(registry: str | None) -> None:
    """Create an operator key and show its public key."""
    change = prepare_create(_registry(registry))
    value = change.apply(load_config())
    assert change.key is not None
    public = sign.encode_public(change.key.public)
    ui.out(Text(t("Operator key: {fingerprint}", fingerprint=value)))
    ui.out(Text(public))
    ui.out(Text(f"desman-registry assign {public}"))


@key_group.command("remove")
@click.option("--yes", is_flag=True, help="Confirm removal.")
def remove_command(yes: bool) -> None:
    """Remove this machine's operator key."""
    change = prepare_remove()
    for line in change.lines:
        ui.out(Text(line))
    if yes:
        change.apply(load_config())
