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
from desman.config import Config, ConfigError, ConfigManager, open_config
from desman.core import calls, db
from desman.i18n import t, tn
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
            t(
                "config {error}\nFix the file, or run `desman config reset` to start over.",
                error=e,
            )
        ) from e
    return manager


def run_wizard(manager: ConfigManager) -> None:
    try:
        wizard.run(manager)
    except ConfigError as e:
        raise click.ClickException(t("config {error}; run `desman config` again", error=e)) from e
    except (ui.Cancelled, click.Abort):
        ui.out("")
        ui.warn(t("Configuration cancelled; nothing was saved."))
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

    ui.out(Text(t("Config") + "  ", style="muted") + Text(str(manager.config_path)))
    if not out["configured"]:
        ui.warn(t("Not configured yet: run `desman config`."))
    ui.out(Text(t("Language") + "  ", style="muted") + Text(out["language"]))
    granted = [name for name, on in out["permissions"].items() if on]
    ui.out(Text(t("Permissions") + "  ", style="muted") + Text(", ".join(granted) or t("none")))
    if out["permissions"]["admin_rights"]:
        ui.warn(t("WARNING: admin_rights is on: agents may act with elevated privileges."))

    rows = ui.table(t("provider"), t("status"), t("keys"), t("settings"), title=t("Providers"))
    for row in out["providers"]:
        if row["local"]:
            status = Text(t("local"), style="muted")
        elif row["ready"]:
            status = Text(t("ready"), style="good")
        else:
            status = Text(t("not set"), style="muted")
        keys = "  ".join(
            k["value"] + (f" ({k['label']})" if k["label"] else "") for k in row.get("keys", [])
        )
        if "keys" in row and not keys:
            keys = "-"
        settings = "  ".join(f"{env}={value or '-'}" for env, value in row["settings"].items())
        name = Text(row["name"])
        if row["paid"]:
            name.append(f" · {t('paid')}", style="warn")
        rows.add_row(name, status, keys, settings)
    ui.out("")
    ui.out(rows)
    for env, values in out["unknown_keys"].items():
        ui.warn(
            t(
                "unknown key {env}={values} (no provider uses this name)",
                env=env,
                values=", ".join(values),
            )
        )
    for env, value in out["unknown_settings"].items():
        ui.warn(t("unknown setting {env}={value}", env=env, value=value))


@config.command("reset")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def config_reset(yes: bool) -> None:
    """Delete the configuration file, API keys included."""
    manager = open_config()  # a broken file can be reset too: that is the way out
    path = manager.config_path
    if not (path.exists() or path.is_symlink()):
        ui.note(t("No configuration at {path}; nothing to delete.", path=path))
        return
    if not yes:
        try:
            count = sum(len(v) for v in manager.load().api_keys.values())
            question = tn(
                count,
                "Delete {path} and the {n} API key in it?",
                "Delete {path} and the {n} API keys in it?",
                path=path,
            )
        except ConfigError:
            question = t("Delete {path} (it does not load)?", path=path)
        if not ui.confirm(question, default=False):
            ui.note(t("Kept."))
            raise SystemExit(1)
    target = path.resolve() if path.is_symlink() else None
    manager.delete()
    ui.ok(t("Deleted {path}. Run `desman config` to set DESMAN up again.", path=path))
    if target is not None and target.exists():
        ui.warn(
            t("It was a link: only the link is gone, {target} is left as it was.", target=target)
        )


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
                "paid": p.paid,
                "ready": p.is_ready(config),
                "keys": keys,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    total = sum(len(r["keys"]) for r in rows)
    with_keys = [r for r in rows if r["keys"]]
    table = ui.table(
        t("provider"),
        "#",
        t("key"),
        t("label"),
        t("calls 24h"),
        t("tokens 24h"),
        title=t("API keys"),
    )
    for r in rows:
        shown = r["name"] + (f" · {t('paid')}" if r["paid"] else "")
        if not r["keys"]:
            table.add_row(Text(shown, style="muted"), "", Text("-", style="muted"), "", "", "")
        for k in r["keys"]:
            name = shown if k["n"] == 1 else ""
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
        tn(
            total,
            "{n} key from {with_keys} of {providers} providers. `desman keys check` tests them "
            "(no tokens spent), `desman usage` shows totals.",
            "{n} keys from {with_keys} of {providers} providers. `desman keys check` tests them "
            "(no tokens spent), `desman usage` shows totals.",
            with_keys=len(with_keys),
            providers=len(rows),
        )
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
        raise click.ClickException(t("no keys to check: add one with `desman keys add PROVIDER`"))
    results = []
    with httpx.Client() as client:
        for p in targets:
            if not p.is_ready(config):
                raise click.ClickException(t("{name} is not configured", name=p.name))
            results += check.check(config, p, client)
    if as_json:
        click.echo(json.dumps([asdict(r) for r in results], indent=2, default=str))
    else:
        table = ui.table(t("provider"), "#", t("status"), t("models"), t("default model"))
        for r in results:
            p = providers.find(r.provider)
            status = (
                Text(t("ok"), style="good")
                if r.ok
                else Text(
                    t("failed ({status})", status=r.http_status or t("no reply")), style="bad"
                )
            )
            default = (
                ""
                if r.default_model_listed is None
                else Text(
                    p.default_model + ("" if r.default_model_listed else " " + t("(not listed)")),
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
            ui.note(t("{name} limits and quotas: {url}", name=p.name, url=p.limits_url))
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
        ui.note(
            t("No model calls yet.")
            if window is None
            else t("No model calls in the last {since}.", since=since)
        )
        return
    lines = _usage_lines(config, rows)
    table = ui.table(
        t("provider"),
        t("key"),
        t("calls"),
        t("failed"),
        t("limited"),
        t("tokens in"),
        t("tokens out"),
        t("last call"),
        title=t("Usage (all time)") if window is None else t("Usage (last {since})", since=since),
    )
    for provider, label, u in lines:
        if u is None:  # a configured key with no calls in the window
            table.add_row(provider, label, Text("0", style="muted"), *[Text("-", "muted")] * 5)
            continue
        table.add_row(
            provider,
            label,
            str(u.calls),
            Text(str(u.failed), style="bad" if u.failed else "muted"),
            Text(str(u.limited), style="warn" if u.limited else "muted"),
            f"{u.prompt_tokens:,}",
            f"{u.completion_tokens:,}",
            (u.last_at or "")[:16].replace("T", " "),
        )
    ui.out(table)
    if any(u is not None and u.key_id not in names for _, _, u in lines):
        ui.note(
            t(
                '"removed": calls in this period went through a key that is no longer in the config '
                "(deleted or replaced). The characters in brackets start its fingerprint."
            )
        )
    _print_limits(lines)


Line = tuple[str, str, calls.KeyUsage | None]


def _usage_lines(config: Config, rows: list[calls.KeyUsage]) -> list[Line]:
    """Every key of each provider that made calls, in `desman keys` order (keys without calls
    included), then the keys that were removed since."""
    lines: list[Line] = []
    known = [p.id for p in providers.CLOUD]
    for pid in [*known, *sorted({u.provider for u in rows} - set(known))]:
        logged = {u.key_id: u for u in rows if u.provider == pid}
        if not logged:
            continue
        configured = []
        if pid in known:
            p = providers.find(pid)
            for i, k in enumerate(config.keys(p.key_env or ""), 1):
                configured.append(key_id(k.value))
                lines.append((pid, f"#{i} {providers.mask(k.value)}", logged.get(key_id(k.value))))
        for kid, u in logged.items():
            if kid not in configured:
                lines.append((pid, t("removed ({id}…)", id=(kid or "?")[:4]), u))
    return lines


def _print_limits(lines: list[Line]) -> None:
    table = ui.table(
        t("provider"),
        t("key"),
        t("limit"),
        t("left"),
        t("resets in"),
        title=t("Limits reported with the latest call"),
    )
    blocked = []
    older = False
    for provider, label, u in lines:
        if u is None or not u.limits:
            continue
        found, other = calls.limits(u.limits)
        # Limits from an earlier call (later ones sent none) are history, not the key's state.
        row_style = "" if u.limits_current else "muted"
        older = older or not u.limits_current
        first = True
        for limit in found:
            left = f"{limit.remaining or '?'} / {limit.limit or '?'}"
            bad = limit.exhausted_for_good and u.limits_current
            table.add_row(
                provider if first else "",
                label if first else "",
                limit.name,
                Text(left, style="bad" if bad else row_style),
                limit.reset or "",
                style=row_style,
            )
            first = False
            if bad:
                blocked.append(f"{provider} {label}")
        for header, value in other.items():
            table.add_row(
                provider if first else "",
                label if first else "",
                header,
                value,
                "",
                style=row_style,
            )
            first = False
    if not table.rows:
        return
    ui.out("")
    ui.out(table)
    if older:
        ui.note(t("Grey rows come from an earlier call: later calls with that key sent no limits."))
    for who in dict.fromkeys(blocked):
        ui.warn(
            t(
                "{key}: the limit is 0, so this account has no access to the API yet (a free plan "
                "may need activating on the provider's site). DESMAN sets such a key aside and "
                "tries the provider's other keys.",
                key=who,
            )
        )


def _window(since: str) -> timedelta | None:
    if since == "all":
        return None
    units = {"m": 60, "h": 3600, "d": 86400}
    number, unit = since[:-1], since[-1:]
    if unit not in units or not number.isdigit() or int(number) == 0:
        raise click.BadParameter(t("use e.g. 30m, 24h, 7d or all"), param_hint="--since")
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
        raise click.BadParameter(
            t("{name} is local and has no API key", name=p.name), param_hint="PROVIDER"
        )
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
        ui.note(t("Get a key: {url}", url=p.url))
        value = ui.secret(t("API key for {name} ({env}), hidden", name=p.name, env=p.key_env))
    if not value:
        raise click.UsageError(t("no key given"))
    with manager.edit() as config:
        try:
            added = config.add_key(p.key_env, value, label)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--label") from e
        n = len(config.keys(p.key_env))
    if not added:
        ui.warn(t("{name} already has this key; nothing changed.", name=p.name))
        return
    ui.ok(
        t(
            "Added {key} to {name} (key #{number}, {number} in total).",
            key=providers.mask(value),
            name=p.name,
            number=n,
        )
    )
    if p.paid:
        ui.warn(
            t(
                "{name} is billed per use; DESMAN uses it only with `desman run --provider {id}`.",
                name=p.name,
                id=p.id,
            )
        )
    missing = [s for s in p.settings if not s.default and not config.settings.get(s.env)]
    for s in missing:
        ui.warn(
            t(
                "{name} also needs {label} ({env}): set it in `desman config`.",
                name=p.name,
                label=s.label,
                env=s.env,
            )
        )


@keys.command("remove")
@click.argument("provider")
@click.argument("number", type=int, required=False)
@click.option("--all", "remove_all", is_flag=True, help="Remove every key of PROVIDER.")
def keys_remove(provider: str, number: int | None, remove_all: bool) -> None:
    """Remove key NUMBER (as `desman keys` lists it) of PROVIDER, or --all of them."""
    p = _provider(provider)
    assert p.key_env is not None
    if remove_all == (number is not None):
        raise click.UsageError(t("give a key NUMBER or --all"))
    with load_config().edit() as config:
        count = len(config.keys(p.key_env))
        if count == 0:
            raise click.ClickException(t("{name} has no keys", name=p.name))
        if remove_all:
            config.api_keys.pop(p.key_env, None)
            removed = count
        else:
            assert number is not None
            try:
                config.remove_key(p.key_env, number - 1)
            except IndexError:
                raise click.BadParameter(
                    tn(
                        count,
                        "{name} has {n} key: pick 1",
                        "{name} has {n} keys: pick 1 to {n}",
                        name=p.name,
                    ),
                    param_hint="NUMBER",
                ) from None
            removed = 1
    ui.ok(
        tn(
            removed,
            "Removed {n} key from {name}; {left} left.",
            "Removed {n} keys from {name}; {left} left.",
            name=p.name,
            left=count - removed,
        )
    )


@keys.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def keys_import(file: Path) -> None:
    """Add the keys in FILE (JSON or NAME=value lines) to the config. Keys already saved and
    names no provider uses are skipped; values are never printed."""
    try:
        pairs = providers.parse_key_file(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise click.ClickException(t("cannot read {file}: {error}", file=file, error=e)) from e
    if not pairs:
        raise click.ClickException(t("no keys found in {file}", file=file))
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
    table = ui.table(
        t("provider"),
        t("added"),
        t("already saved"),
        title=t("Imported from {file}", file=file.name),
    )
    for name in sorted(set(added) | set(already)):
        table.add_row(name, str(added.get(name, 0)), str(already.get(name, 0)))
    ui.out(table)
    ui.out("")
    if unknown:
        ui.warn(
            t(
                "skipped names no provider uses: {names}",
                names=", ".join(dict.fromkeys(unknown)),
            )
        )
    ui.ok(
        tn(
            sum(added.values()),
            "{n} key added, {already} already saved.",
            "{n} keys added, {already} already saved.",
            already=sum(already.values()),
        )
    )
    if added:
        ui.note(
            t(
                "The keys are now in {path}; {file} can be deleted.",
                path=manager.config_path,
                file=file,
            )
        )
