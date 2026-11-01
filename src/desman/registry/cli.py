"""`desman-registry`: the registry on the owner's server (ADR 0004; ROADMAP G.4).

`serve` answers nodes; `assign` and `revoke` sign records with the root key; `backup` copies
the database; `nodes` and `log` show what the registry knows and did. Non-interactive, like
every subcommand. It reads no `config.json`: its directory is `--dir` (or
$DESMAN_REGISTRY_DIR), its language $DESMAN_REGISTRY_LANGUAGE.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import click

from desman import i18n, ui
from desman.i18n import t
from desman.net import sign
from desman.registry import admin, server, store

LANGUAGE_ENV: Final = "DESMAN_REGISTRY_LANGUAGE"
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class RegistryGroup(click.Group):
    """Switches to $DESMAN_REGISTRY_LANGUAGE before Click reads the arguments."""

    def main(self, *args: Any, **kwargs: Any) -> Any:
        code = os.environ.get(LANGUAGE_ENV, "en")
        i18n.set_language(code if code in i18n.CATALOGS else "en")
        i18n.translate_commands(self)
        return super().main(*args, **kwargs)


def _now() -> datetime:
    return datetime.now(UTC)


def _dir(ctx: click.Context) -> Path:
    found: Path = ctx.obj
    return found


def _fail(e: admin.AdminError) -> click.ClickException:
    return click.ClickException(str(e))


@click.group(cls=RegistryGroup)
@click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="The registry's directory: database and keys (default: $DESMAN_REGISTRY_DIR or ~/.local/share/desman-registry).",
)
@click.pass_context
def cli(ctx: click.Context, directory: Path | None) -> None:
    """desman-registry: the registry of desman-global, run on the owner's server."""
    ctx.obj = store.default_dir() if directory is None else directory


@cli.command()
@click.option(
    "--host",
    default=server.DEFAULT_HOST,
    show_default=True,
    help="Address to listen on; keep 127.0.0.1 behind the reverse proxy that gives TLS.",
)
@click.option(
    "--port",
    default=server.DEFAULT_PORT,
    show_default=True,
    type=click.IntRange(0, 65535),
    help="Port to listen on.",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Answer nodes until stopped (plain HTTP: TLS is the reverse proxy's)."""
    path = _dir(ctx) / store.DB_NAME
    store.open_registry(path).close()
    httpd = server.make_server(server.Registry(path), host, port)
    address = f"http://{host}:{httpd.server_port}"
    ui.ok(t("desman-registry listens on {address} (Ctrl+C stops it).", address=address))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


@cli.command()
@click.argument("dest", type=click.Path(dir_okay=False, path_type=Path))
@click.pass_context
def backup(ctx: click.Context, dest: Path) -> None:
    """Copy the database to DEST while the server runs."""
    try:
        admin.backup(_dir(ctx) / store.DB_NAME, dest)
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(t("Backup written: {path}", path=str(dest)))


def _root_option(command: Any) -> Any:
    return click.option(
        "--root-key",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="The root's private key (default: root.key in the registry's directory).",
    )(command)


def _open(ctx: click.Context, root_key: Path | None) -> tuple[Any, sign.NodeKey]:
    directory = _dir(ctx)
    root = admin.load_root_key(directory / store.ROOT_KEY if root_key is None else root_key)
    return store.open_registry(directory / store.DB_NAME), root


@cli.command()
@click.argument(
    "public_key", required=False, type=click.Path(dir_okay=False, exists=True, path_type=Path)
)
@click.option(
    "--scope",
    "scopes",
    multiple=True,
    type=click.Choice([s.value for s in sign.Scope]),
    help="What the operator may sign; repeat for several (default: all).",
)
@click.option(
    "--days",
    default=admin.ASSIGNMENT_DAYS,
    show_default=True,
    type=click.IntRange(min=1),
    help="How long the assignment lasts.",
)
@_root_option
@click.pass_context
def assign(
    ctx: click.Context,
    public_key: Path | None,
    scopes: tuple[str, ...],
    days: int,
    root_key: Path | None,
) -> None:
    """Assign an operator, signed by the root: the key in PUBLIC_KEY (a PEM file), or the
    registry's own operator key (created on first use)."""
    try:
        conn, root = _open(ctx, root_key)
        try:
            operator = (
                admin.operator_key(_dir(ctx) / store.OPERATOR_KEY)
                if public_key is None
                else admin.public_key(public_key)
            )
            wanted = {sign.Scope(s) for s in scopes} or set(sign.Scope)
            done = admin.assign(conn, root, operator, wanted, days=days, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t(
            "Operator {fingerprint} assigned ({scopes}) until {date}.",
            fingerprint=done.operator,
            scopes=", ".join(done.scopes),
            date=done.expires_at.date().isoformat(),
        )
    )


@cli.command()
@click.argument("fingerprint")
@click.option("--reason", default="", help="Why, kept in the record (at most 500 characters).")
@_root_option
@click.pass_context
def revoke(ctx: click.Context, fingerprint: str, reason: str, root_key: Path | None) -> None:
    """Revoke a key (a node's or an operator's), signed by the root."""
    if not _FINGERPRINT.match(fingerprint):
        raise click.BadParameter(
            t("Not a key's fingerprint (64 hex digits): {key}", key=fingerprint),
            param_hint="FINGERPRINT",
        )
    if len(reason) > 500:
        raise click.BadParameter(
            t("The reason is longer than 500 characters."), param_hint="--reason"
        )
    try:
        conn, root = _open(ctx, root_key)
        try:
            admin.revoke(conn, root, fingerprint, reason, now=_now())
        finally:
            conn.close()
    except admin.AdminError as e:
        raise _fail(e) from e
    ui.ok(
        t("Revoked {fingerprint}: nodes refuse it from their next poll.", fingerprint=fingerprint)
    )


def _json_option(command: Any) -> Any:
    return click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")(command)


def _status(found: store.Node) -> str:
    if found.revoked:
        return t("revoked")
    return t("taking part") if found.status == "active" else t("has left")


@cli.command()
@_json_option
@click.pass_context
def nodes(ctx: click.Context, as_json: bool) -> None:
    """The nodes that registered: status, what they lend, when seen last."""
    conn = store.open_registry(_dir(ctx) / store.DB_NAME)
    try:
        found = store.nodes(conn)
    finally:
        conn.close()
    if as_json:
        click.echo(json.dumps([n.as_json() for n in found], ensure_ascii=False, indent=2))
        return
    if not found:
        ui.note(t("No node has registered yet."))
        return
    table = ui.table(t("Node"), t("Status"), t("Lends"), t("Last seen"))
    for n in found:
        lends = ", ".join(
            c.provider if c.model is None else f"{c.provider}/{c.model}" for c in n.capabilities
        )
        table.add_row(n.fingerprint[:16], _status(n), lends or "—", n.last_seen)
    ui.out(table)


@cli.command("log")
@click.option(
    "--limit",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help="How many of the latest entries.",
)
@_json_option
@click.pass_context
def log_command(ctx: click.Context, limit: int, as_json: bool) -> None:
    """The registry's journal: registrations, departures, assignments, revocations, refusals."""
    conn = store.open_registry(_dir(ctx) / store.DB_NAME)
    try:
        found = store.entries(conn, limit)
    finally:
        conn.close()
    if as_json:
        click.echo(
            json.dumps(
                [
                    {"seq": e.seq, "ts": e.ts, "action": e.action, "node": e.node, **e.details}
                    for e in found
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not found:
        ui.note(t("The journal is empty."))
        return
    table = ui.table(t("Time"), t("Action"), t("Node"), t("Details"))
    for e in found:
        details = json.dumps(e.details, ensure_ascii=False, sort_keys=True)
        table.add_row(e.ts, e.action, (e.node or "—")[:16], details)
    ui.out(table)
