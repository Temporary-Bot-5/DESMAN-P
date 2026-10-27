"""`desman config ...` and `desman keys ...`: the configuration and the API keys in it.

Only `desman config` (the wizard), `config reset` (asks before deleting) and `keys add`
(asks for the key, hidden) prompt; everything else is non-interactive. A key is never taken
from the command line, where shell history and `ps` would keep it, and never printed whole.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import click
import httpx
from rich.text import Text

from desman import providers, ui, wizard
from desman.config import ConfigError, ConfigManager, open_config
from desman.core import calls, db
from desman.llm import check
from desman.llm.openai_compat import key_id

F = TypeVar("F", bound=Callable[..., Any])


def load_config() -> ConfigManager:
    """The config for commands that use it; a broken file is an error, never "unconfigured"."""
    manager = open_config()
    try:
        manager.load()
    except ConfigError as e:
        raise click.ClickException(
            f"config {e}\nFix the file, or run `desman config reset` to start over."
        ) from e
    return manager


def run_wizard(manager: ConfigManager) -> None:
    try:
        wizard.run(manager)
    except ConfigError as e:
        raise click.ClickException(f"config {e}; run `desman config` again") from e
    except (ui.Cancelled, click.Abort):
        ui.out("")
        ui.warn("Configuration cancelled; nothing was saved.")
        raise SystemExit(1) from None


def db_option(command: F) -> F:
    return click.option(
        "--db",
        "db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="Database file with the call log (default: the usual one).",
    )(command)


def json_option(command: F) -> F:
    return click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")(command)


# ── desman config ───────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configure DESMAN: providers, API keys, permissions (`show`, `reset`)."""
    if ctx.invoked_subcommand is None:
        run_wizard(load_config())


@config.command("show")
@json_option
def config_show(as_json: bool) -> None:
    """Show the configuration with API keys masked."""
    manager = load_config()
    out = providers.describe(manager.load())
    if as_json:
        click.echo(json.dumps({"path": str(manager.config_path), **out}, indent=2))
        return

    ui.out(Text("Config  ", style="muted") + Text(str(manager.config_path)))
    if not out["configured"]:
        ui.warn("Not configured yet: run `desman config`.")
    ui.out(Text("Language  ", style="muted") + Text(out["language"]))
    granted = [name for name, on in out["permissions"].items() if on]
    ui.out(Text("Permissions  ", style="muted") + Text(", ".join(granted) or "none"))
    if out["permissions"]["admin_rights"]:
        ui.warn("WARNING: admin_rights is on: agents may act with elevated privileges.")

    rows = ui.table("provider", "status", "keys", "settings", title="Providers")
    for row in out["providers"]:
        if row["local"]:
            status = Text("local", style="muted")
        elif row["ready"]:
            status = Text("ready", style="good")
        else:
            status = Text("not set", style="muted")
        keys = "  ".join(
            k["value"] + (f" ({k['label']})" if k["label"] else "") for k in row.get("keys", [])
        )
        if "keys" in row and not keys:
            keys = "-"
        settings = "  ".join(f"{env}={value or '-'}" for env, value in row["settings"].items())
        rows.add_row(row["name"], status, keys, settings)
    ui.out("")
    ui.out(rows)
    for env, values in out["unknown_keys"].items():
        ui.warn(f"unknown key {env}={', '.join(values)} (no provider uses this name)")
    for env, value in out["unknown_settings"].items():
        ui.warn(f"unknown setting {env}={value}")


@config.command("reset")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def config_reset(yes: bool) -> None:
    """Delete the configuration file, API keys included."""
    manager = open_config()  # a broken file can be reset too: that is the way out
    path = manager.config_path
    if not (path.exists() or path.is_symlink()):
        ui.note(f"No configuration at {path}; nothing to delete.")
        return
    if not yes:
        try:
            count = sum(len(v) for v in manager.load().api_keys.values())
            what = f"{path} and the {count} API key(s) in it"
        except ConfigError:
            what = f"{path} (it does not load)"
        if not ui.confirm(f"Delete {what}?", default=False):
            ui.note("Kept.")
            raise SystemExit(1)
    target = path.resolve() if path.is_symlink() else None
    manager.delete()
    ui.ok(f"Deleted {path}. Run `desman config` to set DESMAN up again.")
    if target is not None and target.exists():
        ui.warn(f"It was a link: only the link is gone, {target} is left as it was.")


# ── desman keys ─────────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def keys(ctx: click.Context) -> None:
    """API keys: list, add, remove, import (always shown masked)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(keys_list)


@keys.command("list")
@json_option
@db_option
def keys_list(as_json: bool = False, db_path: Path | None = None) -> None:
    """How many keys each provider has, and what each spent in the last 24 hours."""
    config = load_config().load()
    spent = {u.key_id: u for u in _usage(db_path, timedelta(hours=24))}
    rows: list[dict[str, Any]] = []
    for p in providers.CLOUD:
        assert p.key_env is not None
        keys = []
        for i, k in enumerate(config.keys(p.key_env), 1):
            u = spent.get(key_id(k.value))
            keys.append(
                {
                    "n": i,
                    "value": providers.mask(k.value),
                    "label": providers.label(k.label),
                    "key_id": key_id(k.value),
                    "calls_24h": u.calls if u else 0,
                    "tokens_24h": (u.prompt_tokens + u.completion_tokens) if u else 0,
                }
            )
        rows.append(
            {
                "provider": p.id,
                "name": p.name,
                "env": p.key_env,
                "ready": p.is_ready(config),
                "keys": keys,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    total = sum(len(r["keys"]) for r in rows)
    with_keys = [r for r in rows if r["keys"]]
    table = ui.table("provider", "#", "key", "label", "calls 24h", "tokens 24h", title="API keys")
    for r in rows:
        if not r["keys"]:
            table.add_row(Text(r["name"], style="muted"), "", Text("-", style="muted"), "", "", "")
        for k in r["keys"]:
            name = r["name"] if k["n"] == 1 else ""
            table.add_row(
                name,
                str(k["n"]),
                k["value"],
                k["label"],
                str(k["calls_24h"]),
                f"{k['tokens_24h']:,}",
            )
    ui.out(table)
    ui.out("")
    ui.note(
        f"{total} key(s) from {len(with_keys)} of {len(rows)} providers. "
        "`desman keys check` tests them (no tokens spent), `desman usage` shows totals."
    )


@keys.command("check")
@click.argument("provider", required=False)
@json_option
def keys_check(provider: str | None, as_json: bool) -> None:
    """Ask each provider whether its keys work and what limits it reports. Spends no tokens."""
    config = load_config().load()
    targets = (
        [_provider(provider)] if provider else [p for p in providers.CLOUD if p.is_ready(config)]
    )
    if not targets:
        raise click.ClickException("no keys to check: add one with `desman keys add PROVIDER`")
    results = []
    with httpx.Client() as client:
        for p in targets:
            if not p.is_ready(config):
                raise click.ClickException(f"{p.name} is not configured")
            results += check.check(config, p, client)
    if as_json:
        click.echo(json.dumps([asdict(r) for r in results], indent=2, default=str))
    else:
        table = ui.table("provider", "#", "status", "models", "default model")
        for r in results:
            p = providers.find(r.provider)
            status = (
                Text("ok", style="good")
                if r.ok
                else Text(f"failed ({r.http_status or 'no reply'})", style="bad")
            )
            default = (
                ""
                if r.default_model_listed is None
                else Text(
                    p.default_model + ("" if r.default_model_listed else " (not listed)"),
                    style="good" if r.default_model_listed else "warn",
                )
            )
            table.add_row(p.name, str(r.number), status, str(r.models or ""), default)
        ui.out(table)
        ui.out("")
        for r in results:
            name = f"{providers.find(r.provider).name} #{r.number}"
            if not r.ok:
                ui.warn(f"{name}: {r.detail}")
            if r.limits:
                ui.note(f"{name}: " + ", ".join(f"{k}: {v}" for k, v in r.limits.items()))
        for p in targets:
            ui.note(f"{p.name} limits and quotas: {p.limits_url}")
    if not all(r.ok for r in results):
        raise SystemExit(1)


# ── desman usage ────────────────────────────────────────────────────────────


@click.command()
@click.option(
    "--since",
    default="24h",
    show_default=True,
    help="Window: e.g. 30m, 24h, 7d, or 'all'.",
)
@json_option
@db_option
def usage(since: str, as_json: bool, db_path: Path | None) -> None:
    """Model calls and tokens spent, per provider and key (from the call log)."""
    window = _window(since)
    config = load_config().load()
    names = {}
    for p in providers.CLOUD:
        for i, k in enumerate(config.keys(p.key_env or ""), 1):
            names[key_id(k.value)] = f"#{i} {providers.mask(k.value)}"
    rows = _usage(db_path, window)
    if as_json:
        out = [{**asdict(u), "key": names.get(u.key_id or "", "removed key")} for u in rows]
        click.echo(json.dumps(out, indent=2))
        return
    if not rows:
        ui.note(f"No model calls {'yet' if window is None else 'in the last ' + since}.")
        return
    table = ui.table(
        "provider",
        "key",
        "calls",
        "failed",
        "limited",
        "tokens in",
        "tokens out",
        "last call",
        title=f"Usage ({'all time' if window is None else 'last ' + since})",
    )
    for u in rows:
        table.add_row(
            u.provider,
            names.get(u.key_id or "", "removed key"),
            str(u.calls),
            Text(str(u.failed), style="bad" if u.failed else "muted"),
            Text(str(u.limited), style="warn" if u.limited else "muted"),
            f"{u.prompt_tokens:,}",
            f"{u.completion_tokens:,}",
            (u.last_at or "")[:16].replace("T", " "),
        )
    ui.out(table)
    for u in rows:
        if u.limits:
            shown = ", ".join(f"{k.removeprefix('x-ratelimit-')}: {v}" for k, v in u.limits.items())
            ui.note(f"{u.provider} {names.get(u.key_id or '', '')}: {shown}")


def _window(since: str) -> timedelta | None:
    if since == "all":
        return None
    units = {"m": 60, "h": 3600, "d": 86400}
    number, unit = since[:-1], since[-1:]
    if unit not in units or not number.isdigit() or int(number) == 0:
        raise click.BadParameter("use e.g. 30m, 24h, 7d or all", param_hint="--since")
    return timedelta(seconds=int(number) * units[unit])


def _usage(db_path: Path | None, window: timedelta | None) -> list[calls.KeyUsage]:
    path = db_path or db.default_db_path()
    if not path.exists():
        return []
    conn = db.open_db(path)
    try:
        return calls.usage(conn, datetime.now(UTC) - window if window else None)
    finally:
        conn.close()


def _provider(name: str) -> providers.Provider:
    try:
        p = providers.find(name)
    except providers.UnknownProvider as e:
        raise click.BadParameter(str(e), param_hint="PROVIDER") from e
    if p.key_env is None:
        raise click.BadParameter(f"{p.name} is local and has no API key", param_hint="PROVIDER")
    return p


@keys.command("add")
@click.argument("provider")
@click.option("--label", default="", help="A name to tell this key apart, e.g. 'work'.")
@click.option("--stdin", "from_stdin", is_flag=True, help="Read the key from standard input.")
def keys_add(provider: str, label: str, from_stdin: bool) -> None:
    """Add an API key to PROVIDER (id, name or variable, e.g. groq). Several per provider."""
    p = _provider(provider)
    assert p.key_env is not None
    manager = load_config()
    if from_stdin:
        value = sys.stdin.readline().strip()
    else:  # asked before the lock is taken: nobody waits while the user types
        ui.note(f"Get a key: {p.url}")
        value = ui.secret(f"API key for {p.name} ({p.key_env}), hidden")
    if not value:
        raise click.UsageError("no key given")
    with manager.edit() as config:
        try:
            added = config.add_key(p.key_env, value, label)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--label") from e
        n = len(config.keys(p.key_env))
    if not added:
        ui.warn(f"{p.name} already has this key; nothing changed.")
        return
    ui.ok(f"Added {providers.mask(value)} to {p.name} (key #{n}, {n} in total).")
    missing = [s for s in p.settings if not s.default and not config.settings.get(s.env)]
    for s in missing:
        ui.warn(f"{p.name} also needs {s.label} ({s.env}): set it in `desman config`.")


@keys.command("remove")
@click.argument("provider")
@click.argument("number", type=int, required=False)
@click.option("--all", "remove_all", is_flag=True, help="Remove every key of PROVIDER.")
def keys_remove(provider: str, number: int | None, remove_all: bool) -> None:
    """Remove key NUMBER (as `desman keys` lists it) of PROVIDER, or --all of them."""
    p = _provider(provider)
    assert p.key_env is not None
    if remove_all == (number is not None):
        raise click.UsageError("give a key NUMBER or --all")
    with load_config().edit() as config:
        count = len(config.keys(p.key_env))
        if count == 0:
            raise click.ClickException(f"{p.name} has no keys")
        if remove_all:
            config.api_keys.pop(p.key_env, None)
            removed = count
        else:
            assert number is not None
            try:
                config.remove_key(p.key_env, number - 1)
            except IndexError:
                raise click.BadParameter(
                    f"{p.name} has {count} key(s): pick 1 to {count}", param_hint="NUMBER"
                ) from None
            removed = 1
    ui.ok(f"Removed {removed} key(s) from {p.name}; {count - removed} left.")


@keys.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def keys_import(file: Path) -> None:
    """Add the keys in FILE (JSON or NAME=value lines) to the config. Keys already saved and
    names no provider uses are skipped; values are never printed."""
    try:
        pairs = providers.parse_key_file(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise click.ClickException(f"cannot read {file}: {e}") from e
    if not pairs:
        raise click.ClickException(f"no keys found in {file}")
    manager = load_config()
    added: dict[str, int] = {}
    already: dict[str, int] = {}
    unknown: list[str] = []
    with manager.edit() as config:
        for env, value in pairs:
            p = providers.by_key_env(env)
            if p is None:
                unknown.append(env)
                continue
            bucket = added if config.add_key(env, value) else already
            bucket[p.name] = bucket.get(p.name, 0) + 1
    table = ui.table("provider", "added", "already saved", title=f"Imported from {file.name}")
    for name in sorted(set(added) | set(already)):
        table.add_row(name, str(added.get(name, 0)), str(already.get(name, 0)))
    ui.out(table)
    ui.out("")
    if unknown:
        ui.warn(f"skipped names no provider uses: {', '.join(dict.fromkeys(unknown))}")
    ui.ok(f"{sum(added.values())} key(s) added, {sum(already.values())} already saved.")
    if added:
        ui.note(f"The keys are now in {manager.config_path}; {file} can be deleted.")
