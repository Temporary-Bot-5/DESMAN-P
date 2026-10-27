"""`desman config ...` and `desman keys ...`: the configuration and the API keys in it.

Only `desman config` (the wizard), `config reset` (asks before deleting) and `keys add`
(asks for the key, hidden) prompt; everything else is non-interactive. A key is never taken
from the command line, where shell history and `ps` would keep it, and never printed whole.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from rich.text import Text

from desman import providers, ui, wizard
from desman.config import ConfigError, ConfigManager, open_config

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
    except (ui.Cancelled, click.Abort):
        ui.out("")
        ui.warn("Configuration cancelled; nothing was saved.")
        raise SystemExit(1) from None


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
    manager.delete()
    ui.ok(f"Deleted {path}. Run `desman config` to set DESMAN up again.")


# ── desman keys ─────────────────────────────────────────────────────────────


@click.group(invoke_without_command=True)
@click.pass_context
def keys(ctx: click.Context) -> None:
    """API keys: list, add, remove, import (always shown masked)."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(keys_list)


@keys.command("list")
@json_option
def keys_list(as_json: bool = False) -> None:
    """How many keys each provider has."""
    config = load_config().load()
    rows: list[dict[str, Any]] = []
    for p in providers.CLOUD:
        assert p.key_env is not None
        rows.append(
            {
                "provider": p.id,
                "name": p.name,
                "env": p.key_env,
                "ready": p.is_ready(config),
                "keys": [
                    {"n": i, "value": providers.mask(k.value), "label": k.label}
                    for i, k in enumerate(config.keys(p.key_env), 1)
                ],
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    total = sum(len(r["keys"]) for r in rows)
    with_keys = [r for r in rows if r["keys"]]
    table = ui.table("provider", "#", "key", "label", title="API keys")
    for r in rows:
        if not r["keys"]:
            table.add_row(Text(r["name"], style="muted"), "", Text("-", style="muted"), "")
        for k in r["keys"]:
            name = r["name"] if k["n"] == 1 else ""
            table.add_row(name, str(k["n"]), k["value"], k["label"])
    ui.out(table)
    ui.out("")
    ui.note(f"{total} key(s) from {len(with_keys)} of {len(rows)} providers.")


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
    config = manager.load()
    if from_stdin:
        value = sys.stdin.readline().strip()
    else:
        ui.note(f"Get a key: {p.url}")
        value = ui.secret(f"API key for {p.name} ({p.key_env}), hidden")
    if not value:
        raise click.UsageError("no key given")
    if not config.add_key(p.key_env, value, label):
        ui.warn(f"{p.name} already has this key; nothing changed.")
        return
    manager.save(config)
    n = len(config.keys(p.key_env))
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
    manager = load_config()
    config = manager.load()
    count = len(config.keys(p.key_env))
    if remove_all == (number is not None):
        raise click.UsageError("give a key NUMBER or --all")
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
    manager.save(config)
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
    manager = load_config()
    config = manager.load()
    added: dict[str, int] = {}
    already: dict[str, int] = {}
    unknown: list[str] = []
    for env, value in pairs:
        p = providers.by_key_env(env)
        if p is None:
            unknown.append(env)
            continue
        bucket = added if config.add_key(env, value) else already
        bucket[p.name] = bucket.get(p.name, 0) + 1
    if added:
        manager.save(config)
    table = ui.table("provider", "added", "already saved", title=f"Imported from {file.name}")
    for name in sorted(set(added) | set(already)):
        table.add_row(name, str(added.get(name, 0)), str(already.get(name, 0)))
    ui.out(table)
    ui.out("")
    if unknown:
        ui.warn(f"skipped names no provider uses: {', '.join(dict.fromkeys(unknown))}")
    if not pairs:
        raise click.ClickException(f"no keys found in {file}")
    ui.ok(f"{sum(added.values())} key(s) added, {sum(already.values())} already saved.")
    if added:
        ui.note(f"The keys are now in {manager.config_path}; {file} can be deleted.")
