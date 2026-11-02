"""`desman global`: whether this machine takes part in desman-global, and what it lends (G.3).

Off unless the person joins (ADR 0004, decision 4). Joining creates the node's key (`node.key`,
shown by its fingerprint only) and switches desman-global on; leaving takes back everything
lent and switches it off, while the key stays until the person deletes it. A module lent
(`donate`) is a model provider of this machine with a daily quota of calls and tokens: its key
never leaves the machine, a paid provider is never lent, and a free cloud provider only once
its terms were checked (`providers.can_donate`).

Only leaving talks to the registry here: once the config is saved off, it says goodbye to the
registry it was on with (`farewell`, `node.tell_left`); the rest of the contact is the worker's (`net.node`, G.5). Both get the
address from `net.registry()`, which has none while desman-global is off. Every change is written to `network_log`. The commands
are non-interactive; the chat calls the same functions after the person's "yes".
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import click
import httpx
from rich.text import Text

from desman import net, providers, ui
from desman.config import Config, ConfigManager, Donation, Network
from desman.config_cli import json_option, load_config
from desman.core import db, netlog
from desman.i18n import t
from desman.net import node, sign
from desman.providers import Provider

DEFAULT_DONATION = Donation(calls_per_day=100, tokens_per_day=100_000)
"""A module's quota when the person names none."""

NO_KEY: Final = "node_key_unusable"
"""`Left.unheard` when the goodbye could not be signed: nothing was sent."""


@dataclass(frozen=True)
class Status:
    enabled: bool
    registry: str | None
    """Where the node would go (None: no address known yet)."""
    node: str | None
    """The node key's fingerprint, if there is one."""
    donated: dict[str, Donation]

    def as_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "registry": self.registry,
            "node": self.node,
            "donated": {name: d.model_dump() for name, d in self.donated.items()},
        }


Entry = tuple[str, dict[str, Any]]
"""An entry of `network_log`: action, details."""


@contextmanager
def journal(conn: sqlite3.Connection | None) -> Iterator[sqlite3.Connection]:
    """The database whose `network_log` records the change: the caller's, or the default one."""
    if conn is not None:
        yield conn
        return
    try:
        opened = db.open_db()
    except (db.SchemaVersionError, db.SchemaMismatchError) as e:
        raise click.ClickException(str(e)) from e
    try:
        yield opened
    finally:
        opened.close()


def commit(conn: sqlite3.Connection, entries: list[Entry], save: Callable[[], None]) -> None:
    """Save the config and journal its change together: the entries are written in the
    transaction the config is saved in, so a save that fails journals nothing."""
    with db.transaction(conn):
        for action, details in entries:
            netlog.write(conn, action, details)
        save()


@contextmanager
def _change(
    manager: ConfigManager, conn: sqlite3.Connection | None
) -> Iterator[tuple[Config, list[Entry]]]:
    """The config to change and the entries that say how, saved and journaled together under
    the config's lock: two commands at once journal in the order they saved. Nothing is saved
    when the block raises or journals nothing."""
    with journal(conn) as db_conn, manager.locked():
        config = manager.load()
        entries: list[Entry] = []
        yield config, entries
        if entries:
            commit(db_conn, entries, lambda: manager.save(config))


def node_fingerprint(key_path: Path | None = None) -> str | None:
    """The node's fingerprint, without creating a key; None if there is none (or it is unusable)."""
    path = sign.default_node_key_path() if key_path is None else key_path
    if not path.exists():
        return None
    try:
        return sign.load_node_key(path).fingerprint
    except sign.KeyFileError:
        return None


def usable_node_key(key_path: Path | None = None, *, create: bool = True) -> sign.NodeKey | None:
    """The node's key (created if `create` and there is none), or why it cannot be used."""
    path = sign.default_node_key_path() if key_path is None else key_path
    try:
        if create:
            return sign.node_key(path)
        return sign.load_node_key(path) if path.exists() or path.is_symlink() else None
    except (sign.KeyFileError, OSError) as e:
        raise click.ClickException(t("The node's key cannot be used: {error}", error=e)) from e


def status(config: Config, key_path: Path | None = None) -> Status:
    network = config.network
    return Status(
        enabled=network.enabled,
        registry=network.registry or net.DEFAULT_REGISTRY,
        node=node_fingerprint(key_path),
        donated=dict(network.donated),
    )


# ── Changes ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Joined:
    changed: bool
    registry: str
    node: str


def join_registry(config: Config, registry: str | None) -> str:
    """The registry joining would use: the one given, the one saved, or the shipped one."""
    found = registry or config.network.registry or net.DEFAULT_REGISTRY
    if found is None:
        raise click.ClickException(
            t(
                "No registry address is known yet: give one with `desman global join --registry URL`."
            )
        )
    try:
        Network(registry=found)
    except ValueError:
        raise click.ClickException(
            t("{url} is not a registry address: https://host[:port][/path].", url=found)
        ) from None
    return found


def joining(config: Config, registry: str | None, key: sign.NodeKey) -> tuple[Joined, Entry | None]:
    """`config` switched on (changed in place), and its journal entry if anything changed."""
    address = join_registry(config, registry)
    before = config.network.model_copy(deep=True)
    config.network.enabled = True
    if registry is not None or config.network.registry is not None:
        config.network.registry = address
    if config.network == before:
        return Joined(False, address, key.fingerprint), None
    return Joined(True, address, key.fingerprint), (
        "joined",
        {"node": key.fingerprint, "registry": address},
    )


@dataclass(frozen=True)
class Left:
    withdrawn: list[str]
    """What was lent and is taken back."""
    registry: str | None = None
    """The registry to say goodbye to: the one desman-global was on with (None: none)."""
    told: bool = False
    """The registry heard the goodbye (or did not list the node anyway)."""
    unheard: str | None = None
    """Why it was not told: `NO_KEY` (nothing sent), no answer, or its refusal."""


def leaving(config: Config) -> tuple[Left | None, list[Entry]]:
    """`config` switched off with nothing lent (changed in place): what was taken back and
    from which registry, None if it was off already; and the journal entries. Nothing is sent:
    the goodbye (`farewell`) waits until this config is saved, so a save that fails leaves
    the registry as it was."""
    if not config.network.enabled and not config.network.donated:
        return None, []
    registry = net.registry(config)
    withdrawn = sorted(config.network.donated)
    config.network.enabled = False
    config.network.donated = {}
    return Left(withdrawn, registry), [("left", {"withdrawn": withdrawn})]


def farewell(
    left: Left | None,
    *,
    conn: sqlite3.Connection | None = None,
    transport: httpx.BaseTransport | None = None,
    key_path: Path | None = None,
) -> Left | None:
    """Once `leaving`'s config is saved: tell the registry this node leaves (G.5), sent once,
    and journal how it went. A goodbye that is not heard is not sent again: the registry
    stops listing a node that falls silent."""
    if left is None or left.registry is None:
        return left
    path = sign.default_node_key_path() if key_path is None else key_path
    try:
        key = sign.load_node_key(path)
    except (sign.KeyFileError, OSError):
        with journal(conn) as db_conn:
            netlog.record(db_conn, "leave_not_sent", {"registry": left.registry, "why": NO_KEY})
        return replace(left, unheard=NO_KEY)
    unheard = node.tell_left(left.registry, key, transport=transport)
    details: dict[str, Any] = {
        "registry": left.registry,
        "node": key.fingerprint,
        "heard": unheard is None,
    }
    if unheard is not None:
        details["error"] = unheard
    with journal(conn) as db_conn:
        netlog.record(db_conn, "leave_sent", details)
    return replace(left, told=unheard is None, unheard=unheard)


def join(
    manager: ConfigManager,
    registry: str | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    key_path: Path | None = None,
) -> Joined:
    """Switch desman-global on, with the node's key (created the first time)."""
    with _change(manager, conn) as (config, entries):
        join_registry(config, registry)  # a bad address creates no key
        key = usable_node_key(key_path)
        assert key is not None
        joined, entry = joining(config, registry, key)
        entries += [entry] if entry else []
    return joined


def leave(
    manager: ConfigManager,
    *,
    conn: sqlite3.Connection | None = None,
    transport: httpx.BaseTransport | None = None,
    key_path: Path | None = None,
) -> Left | None:
    """Switch desman-global off and take back every module lent, then tell the registry: what
    was taken back, None if it was off already. The node's key stays."""
    with _change(manager, conn) as (config, entries):
        left, found = leaving(config)
        entries += found
    return farewell(left, conn=conn, transport=transport, key_path=key_path)


def donatable(config: Config, name: str) -> Provider:
    """The provider `name` names, if the person may lend it now; otherwise why not."""
    if not config.network.enabled:
        raise click.ClickException(t("desman-global is off: join it first (`desman global join`)."))
    try:
        provider = providers.find(name)
    except providers.UnknownProvider as e:
        raise click.ClickException(str(e)) from e
    if provider.paid:
        raise click.ClickException(
            t("{name} is a paid provider: it is never lent.", name=provider.name)
        )
    if not providers.can_donate(provider):
        raise click.ClickException(
            t(
                "{name}'s free tier has not been checked for serving others, so it cannot be lent.",
                name=provider.name,
            )
        )
    if not provider.is_ready(config):
        raise click.ClickException(
            t("{name} is not set up yet (`desman config`).", name=provider.name)
        )
    return provider


def donation(base: Donation, calls: int | None, tokens: int | None) -> Donation:
    data = base.model_dump()
    if calls is not None:
        data["calls_per_day"] = calls
    if tokens is not None:
        data["tokens_per_day"] = tokens
    try:
        return Donation.model_validate(data)
    except ValueError:
        raise click.ClickException(
            t(
                "A quota is 1–100000 calls and 1000–100000000 tokens a day.",
            )
        ) from None


@dataclass(frozen=True)
class Lent:
    provider: str
    old: Donation | None
    new: Donation | None

    @property
    def changed(self) -> bool:
        return self.old != self.new


def donate(
    manager: ConfigManager,
    name: str,
    calls: int | None = None,
    tokens: int | None = None,
    *,
    conn: sqlite3.Connection | None = None,
) -> Lent:
    """Lend a provider (or change the quota of one lent already)."""
    with _change(manager, conn) as (config, entries):
        provider = donatable(config, name)
        old = config.network.donated.get(provider.id)
        new = donation(old or DEFAULT_DONATION, calls, tokens)
        config.network.donated[provider.id] = new
        if old != new:
            action = "donated" if old is None else "quota"
            entries.append((action, {"provider": provider.id, **new.model_dump()}))
    return Lent(provider.id, old, new)


def lent_provider(config: Config, name: str) -> str:
    """The id of a provider that is lent now, named any way `providers.find` takes."""
    try:
        provider = providers.find(name)
    except providers.UnknownProvider as e:
        raise click.ClickException(str(e)) from e
    if provider.id not in config.network.donated:
        raise click.ClickException(t("{name} is not lent.", name=provider.name))
    return provider.id


def set_quota(
    manager: ConfigManager,
    name: str,
    calls: int | None,
    tokens: int | None,
    *,
    conn: sqlite3.Connection | None = None,
) -> Lent:
    """Change the quota of a module lent already."""
    with _change(manager, conn) as (config, entries):
        lent = lent_provider(config, name)
        old = config.network.donated[lent]
        new = donation(old, calls, tokens)
        config.network.donated[lent] = new
        if old != new:
            entries.append(("quota", {"provider": lent, **new.model_dump()}))
    return Lent(lent, old, new)


def undonate(manager: ConfigManager, name: str, *, conn: sqlite3.Connection | None = None) -> Lent:
    with _change(manager, conn) as (config, entries):
        lent = lent_provider(config, name)
        old = config.network.donated.pop(lent)
        entries.append(("undonated", {"provider": lent}))
    return Lent(lent, old, None)


def left_json(left: Left | None) -> dict[str, Any]:
    return {
        "changed": left is not None,
        "withdrawn": [] if left is None else left.withdrawn,
        "registry_told": left is not None and left.told,
    }


# ── Output ───────────────────────────────────────────────────────────────────


def quota_text(donation: Donation) -> str:
    return t(
        "{calls} calls and {tokens} tokens a day",
        calls=donation.calls_per_day,
        tokens=donation.tokens_per_day,
    )


def provider_name(provider_id: str) -> str:
    try:
        return providers.find(provider_id).name
    except providers.UnknownProvider:
        return provider_id


def print_status(found: Status) -> None:
    table = ui.table("", "", title=t("desman-global"))
    table.show_header = False
    table.add_row(Text(t("taking part"), style="muted"), t("yes") if found.enabled else t("no"))
    table.add_row(Text(t("registry"), style="muted"), found.registry or t("not known yet"))
    table.add_row(Text(t("node key"), style="muted"), found.node or t("none yet"))
    for provider_id, donation in sorted(found.donated.items()):
        table.add_row(
            Text(t("lent: {name}", name=provider_name(provider_id)), style="muted"),
            quota_text(donation),
        )
    ui.out(table)
    if found.enabled:
        ui.note(
            t(
                "While `desman worker` runs, this node registers with the registry, says it "
                "is alive and fetches the revocations; nothing else is sent yet. "
                "`desman global leave` takes everything back."
            )
        )
    else:
        ui.note(
            t(
                "Off: nothing leaves this machine for desman-global. `desman global join` "
                "switches it on; lending a module is `desman global donate PROVIDER`."
            )
        )


def print_joined(joined: Joined) -> None:
    if not joined.changed:
        ui.note(t("Already taking part in desman-global."))
    else:
        ui.ok(t("Taking part in desman-global, registry {url}.", url=joined.registry))
    ui.note(t("This node's key: {fingerprint}", fingerprint=joined.node))


def print_left(left: Left | None) -> None:
    if left is None:
        ui.note(t("desman-global is off already."))
        return
    ui.ok(t("desman-global is off; nothing lent any more."))
    for provider_id in left.withdrawn:
        ui.ok(t("Taken back: {name}", name=provider_name(provider_id)))
    if left.unheard == NO_KEY:
        ui.warn(
            t(
                "The registry was not told: the node's key cannot be used, so nothing was "
                "sent. It stops listing this node once it falls silent."
            )
        )
    elif left.unheard:
        ui.warn(
            t(
                "The registry could not be told ({error}); it may list this node a while "
                "longer. Nothing more is sent to it.",
                error=left.unheard,
            )
        )
    ui.note(t("The node's key stays; delete it yourself if you want it gone."))


def print_lent(lent: Lent) -> None:
    name = provider_name(lent.provider)
    if not lent.changed:
        ui.note(t("Nothing changed."))
    elif lent.new is None:
        ui.ok(t("{name} is not lent any more.", name=name))
    elif lent.old is None:
        ui.ok(t("{name} lent: {quota}.", name=name, quota=quota_text(lent.new)))
    else:
        ui.ok(
            t(
                "{name}: {old} → {new}",
                name=name,
                old=quota_text(lent.old),
                new=quota_text(lent.new),
            )
        )


# ── The commands ─────────────────────────────────────────────────────────────


@click.group("global", invoke_without_command=True)
@click.pass_context
def global_group(ctx: click.Context) -> None:
    """desman-global: whether this machine takes part, and what it lends."""
    if ctx.invoked_subcommand is None:
        print_status(status(load_config().load()))


@global_group.command("status")
@json_option
def status_command(as_json: bool) -> None:
    """Whether this machine takes part in desman-global, its key, what it lends."""
    found = status(load_config().load())
    if as_json:
        click.echo(json.dumps(found.as_json(), ensure_ascii=False, indent=2))
    else:
        print_status(found)


@global_group.command("join")
@click.option("--registry", metavar="URL", help="The registry's address (https://...).")
@json_option
def join_command(registry: str | None, as_json: bool) -> None:
    """Take part in desman-global: creates this node's key the first time."""
    joined = join(load_config(), registry)
    if as_json:
        click.echo(
            json.dumps(
                {"changed": joined.changed, "registry": joined.registry, "node": joined.node}
            )
        )
    else:
        print_joined(joined)


@global_group.command("leave")
@json_option
def leave_command(as_json: bool) -> None:
    """Stop taking part in desman-global and take back everything lent. The key stays."""
    left = leave(load_config())
    if as_json:
        click.echo(json.dumps(left_json(left)))
    else:
        print_left(left)


calls_option = click.option("--calls", type=int, metavar="N", help="Calls a day (1–100000).")
tokens_option = click.option(
    "--tokens", type=int, metavar="N", help="Tokens a day (1000–100000000)."
)


def _lent_json(lent: Lent) -> str:
    return json.dumps(
        {
            "provider": lent.provider,
            "changed": lent.changed,
            "quota": None if lent.new is None else lent.new.model_dump(),
        }
    )


@global_group.command("donate")
@click.argument("provider")
@calls_option
@tokens_option
@json_option
def donate_command(provider: str, calls: int | None, tokens: int | None, as_json: bool) -> None:
    """Lend PROVIDER to desman-global with a daily quota (its key stays on this machine)."""
    lent = donate(load_config(), provider, calls, tokens)
    if as_json:
        click.echo(_lent_json(lent))
    else:
        print_lent(lent)


@global_group.command("undonate")
@click.argument("provider")
@json_option
def undonate_command(provider: str, as_json: bool) -> None:
    """Take PROVIDER back from desman-global."""
    lent = undonate(load_config(), provider)
    if as_json:
        click.echo(_lent_json(lent))
    else:
        print_lent(lent)


@global_group.command("quota")
@click.argument("provider")
@calls_option
@tokens_option
@json_option
def quota_command(provider: str, calls: int | None, tokens: int | None, as_json: bool) -> None:
    """Show or change the daily quota of PROVIDER, lent to desman-global."""
    manager = load_config()
    if calls is None and tokens is None:
        config = manager.load()
        lent = lent_provider(config, provider)
        donation = config.network.donated[lent]
        if as_json:
            click.echo(json.dumps({"provider": lent, "quota": donation.model_dump()}))
        else:
            ui.out(f"{provider_name(lent)}: {quota_text(donation)}")
        return
    changed = set_quota(manager, provider, calls, tokens)
    if as_json:
        click.echo(_lent_json(changed))
    else:
        print_lent(changed)
