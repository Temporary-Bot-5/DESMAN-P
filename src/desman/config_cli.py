"""`desman config ...` and `desman keys ...`: the configuration and the API keys in it.

Only `desman config` (the wizard), `config reset` (asks before deleting) and `keys add`
(asks for the key, hidden) prompt; everything else is non-interactive. A key is never taken
from the command line, where shell history and `ps` would keep it, and never printed whole.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import click
import httpx
from rich.text import Text

from desman import i18n, providers, ui, wizard
from desman.config import Config, ConfigError, ConfigManager, Permissions, open_config
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
    print_config(manager.config_path, out)


def print_config(path: Path, out: dict[str, Any]) -> None:
    """`providers.describe()` of the config at `path`, for the person."""
    ui.out(Text(t("Config") + "  ", style="muted") + Text(str(path)))
    if not out["configured"]:
        ui.warn(t("Not configured yet: run `desman config`."))
    ui.out(Text(t("Language") + "  ", style="muted") + Text(out["language"]))
    granted = [name for name, on in out["permissions"].items() if on]
    ui.out(Text(t("Permissions") + "  ", style="muted") + Text(", ".join(granted) or t("none")))
    if out["permissions"]["admin_rights"]:
        ui.warn(t("WARNING: admin_rights is on: agents may act with elevated privileges."))
    profile = t("on (/profile in the chat)") if out["profile"] else t("off")
    ui.out(Text(t("Profile") + "  ", style="muted") + Text(profile))

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
        if row["search"]:
            name.append(f" · {t('web search')}", style="muted")
        rows.add_row(name, status, keys, settings)
    ui.out("")
    ui.out(rows)
    if any(row["paid"] and row["ready"] for row in out["providers"]):
        ui.warn(providers.untested_note())
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
    if not yes and not ui.confirm(reset_question(manager), default=False):
        ui.note(t("Kept."))
        raise SystemExit(1)
    reset_config(manager)


def reset_question(manager: ConfigManager) -> str:
    """What deleting the config takes with it, as a question to the person."""
    path = manager.config_path
    try:
        count = sum(len(v) for v in manager.load().api_keys.values())
    except ConfigError:
        return t("Delete {path} (it does not load)?", path=path)
    return tn(
        count,
        "Delete {path} and the {n} API key in it?",
        "Delete {path} and the {n} API keys in it?",
        path=path,
    )


def reset_config(manager: ConfigManager) -> None:
    path = manager.config_path
    target = path.resolve() if path.is_symlink() else None
    manager.delete()
    ui.ok(t("Deleted {path}. Run `desman config` to set DESMAN up again.", path=path))
    if target is not None and target.exists():
        ui.warn(
            t("It was a link: only the link is gone, {target} is left as it was.", target=target)
        )


# ── desman language ─────────────────────────────────────────────────────────


@click.command()
@click.argument("name", metavar="LANGUAGE", required=False)
def language(name: str | None) -> None:
    """Show the interface languages, or switch to LANGUAGE (a code such as ru, or its name)."""
    manager = load_config()
    if name is None:
        print_languages(manager.load().language)
        return
    set_language(manager, language_code(name))


def print_languages(current: str) -> None:
    table = ui.table(t("code"), t("language"), "", title=t("Languages"))
    for code, catalog in i18n.CATALOGS.items():
        mark = Text("✓", style="good") if code == current else ""
        table.add_row(code, catalog.name, mark)
    ui.out(table)
    ui.out("")
    ui.note(t("Switch with `desman language CODE`."))


def set_language(manager: ConfigManager, code: str) -> None:
    with manager.edit() as config:
        config.language = code  # type: ignore[assignment]  # a key of CATALOGS: a valid code
    i18n.set_language(code)  # the confirmation already in the new language
    ui.ok(t("Language set to {name}.", name=i18n.CATALOGS[code].name))


def language_code(name: str) -> str:
    wanted = name.strip().casefold()
    for code, catalog in i18n.CATALOGS.items():
        if wanted in (code, catalog.name.casefold()):
            return code
    raise click.BadParameter(
        t(
            "unknown language {name!r} (known: {codes})",
            name=name,
            codes=", ".join(i18n.CATALOGS),
        ),
        param_hint="LANGUAGE",
    )


# ── desman permissions ──────────────────────────────────────────────────────

PERMISSIONS = tuple(Permissions.model_fields)


@click.command()
@click.option(
    "--grant",
    multiple=True,
    type=click.Choice(PERMISSIONS),
    help="Allow agents this (repeat for several).",
)
@click.option(
    "--revoke",
    multiple=True,
    type=click.Choice(PERMISSIONS),
    help="Take a permission back (repeat for several).",
)
def permissions(grant: tuple[str, ...], revoke: tuple[str, ...]) -> None:
    """Show what agents may do, or --grant / --revoke a permission. Only you set them."""
    manager = load_config()
    if not grant and not revoke:
        print_permissions(manager.load().permissions)
        return
    if not permission_changes(manager.load().permissions, grant, revoke):
        ui.note(t("Nothing changed."))
        return
    print_permission_changes(set_permissions(manager, grant, revoke))


PermissionChange = tuple[str, bool, bool]
"""(permission, was on, will be on)"""


def permission_changes(
    current: Permissions, grant: Sequence[str], revoke: Sequence[str]
) -> list[PermissionChange]:
    """What granting and revoking would change; a permission in both is an error."""
    if both := sorted(set(grant) & set(revoke)):
        raise click.UsageError(t("both granted and revoked: {names}", names=", ".join(both)))
    if unknown := sorted((set(grant) | set(revoke)) - set(PERMISSIONS)):
        raise click.UsageError(
            t(
                "unknown permission {names} (known: {known})",
                names=", ".join(unknown),
                known=", ".join(PERMISSIONS),
            )
        )
    changes = []
    for name in PERMISSIONS:
        was = bool(getattr(current, name))
        now = True if name in grant else False if name in revoke else was
        if now != was:
            changes.append((name, was, now))
    return changes


def set_permissions(
    manager: ConfigManager, grant: Sequence[str], revoke: Sequence[str]
) -> list[PermissionChange]:
    with manager.edit() as config:
        changes = permission_changes(config.permissions, grant, revoke)
        for name, _, now in changes:
            setattr(config.permissions, name, now)
    return changes


def print_permissions(current: Permissions) -> None:
    labels = wizard.permission_labels()
    table = ui.table(t("permission"), "", t("what agents may do"), title=t("Permissions"))
    for name in PERMISSIONS:
        on = bool(getattr(current, name))
        mark = Text("✓", style="good") if on else Text("-", style="muted")
        table.add_row(name, mark, labels[name])
    ui.out(table)
    ui.out("")
    ui.note(t("Change with `desman permissions --grant NAME` or `--revoke NAME`."))
    if current.admin_rights:
        ui.warn(t("WARNING: admin_rights is on: agents may act with elevated privileges."))


def print_permission_changes(changes: list[PermissionChange]) -> None:
    for name, _, now in changes:
        ui.ok(t("{name} granted.", name=name) if now else t("{name} revoked.", name=name))
    if any(name == "admin_rights" and now for name, _, now in changes):
        ui.warn(t("admin_rights is on: agents may act with elevated privileges."))


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
    rows = key_rows(load_config().load(), db_path)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    print_key_rows(rows)


def key_rows(config: Config, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Each provider with keys, its keys masked and what each spent in the last 24 hours."""
    spent = {u.key_id: u for u in _usage(db_path, timedelta(hours=24))}
    rows: list[dict[str, Any]] = []
    for p in providers.CLOUD + tuple(s for s in providers.SEARCH if s.key_env is not None):
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
                "search": p.search,
                "ready": p.is_ready(config),
                "keys": keys,
            }
        )
    return rows


def print_key_rows(rows: list[dict[str, Any]]) -> None:
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
        if r["search"]:
            shown += f" · {t('web search')}"
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
    targets, results = check_keys(load_config().load(), provider)
    if as_json:
        click.echo(json.dumps([asdict(r) for r in results], indent=2, default=str))
    else:
        print_checks(targets, results)
    if not all(r.ok for r in results):
        raise SystemExit(1)


def check_keys(
    config: Config, provider: str | None
) -> tuple[list[providers.Provider], list[check.KeyCheck]]:
    """The providers asked (`provider`, or every ready one) and what they said of each key."""
    targets = (
        [_provider(provider, search=False)]
        if provider
        else [p for p in providers.CLOUD if p.is_ready(config)]
    )
    if not targets:
        raise click.ClickException(t("no keys to check: add one with `desman keys add PROVIDER`"))
    results = []
    with httpx.Client() as client:
        for p in targets:
            if not p.is_ready(config):
                raise click.ClickException(t("{name} is not configured", name=p.name))
            results += check.check(config, p, client)
    return targets, results


def print_checks(targets: list[providers.Provider], results: list[check.KeyCheck]) -> None:
    table = ui.table(t("provider"), "#", t("status"), t("models"), t("default model"))
    for r in results:
        p = providers.find(r.provider)
        status = (
            Text(t("ok"), style="good")
            if r.ok
            else Text(t("failed ({status})", status=r.http_status or t("no reply")), style="bad")
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
    window = parse_window(since)
    config = load_config().load()
    rows = _usage(db_path, window)
    if as_json:
        click.echo(json.dumps(usage_json(config, rows), indent=2))
        return
    print_usage(config, rows, window, since)


def usage_since(since: str, db_path: Path | None = None) -> list[calls.KeyUsage]:
    """Calls per provider and key in the window `since` (e.g. 24h, 7d, all)."""
    return _usage(db_path, parse_window(since))


def usage_json(config: Config, rows: list[calls.KeyUsage]) -> list[dict[str, Any]]:
    names = _key_names(config)
    return [{**asdict(u), "key": names.get(u.key_id or "", "removed key")} for u in rows]


def _key_names(config: Config) -> dict[str, str]:
    names = {}
    for p in providers.CLOUD:
        for i, k in enumerate(config.keys(p.key_env or ""), 1):
            names[key_id(k.value)] = f"#{i} {providers.mask(k.value)}"
    return names


def print_usage(
    config: Config, rows: list[calls.KeyUsage], window: timedelta | None, since: str
) -> None:
    names = _key_names(config)
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


def parse_window(since: str) -> timedelta | None:
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


def _provider(name: str, *, search: bool = True) -> providers.Provider:
    """A provider with keys; with `search`, a web search engine too."""
    try:
        p = providers.find_keyed(name) if search else providers.find(name)
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
    p = keyed_provider(provider)
    manager = load_config()
    # Asked before the lock is taken: nobody waits while the user types.
    value = sys.stdin.readline().strip() if from_stdin else ask_key(p)
    if not value:
        raise click.UsageError(t("no key given"))
    print_added(add_key(manager, p, value, label))


def keyed_provider(name: str) -> providers.Provider:
    """A provider or search engine that takes API keys, for `keys add` and `keys remove`."""
    p = _provider(name)
    assert p.key_env is not None
    return p


def ask_key(p: providers.Provider) -> str:
    """The key for `p`, typed in a hidden field; empty when the person gave none."""
    ui.note(t("Get a key: {url}", url=p.url))
    return ui.secret(t("API key for {name} ({env}), hidden", name=p.name, env=p.key_env))


@dataclass(frozen=True)
class Added:
    provider: providers.Provider
    added: bool
    masked: str
    key_id: str
    number: int
    config: Config


def add_key(manager: ConfigManager, p: providers.Provider, value: str, label: str = "") -> Added:
    assert p.key_env is not None
    with manager.edit() as config:
        try:
            added = config.add_key(p.key_env, value, label)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--label") from e
        n = len(config.keys(p.key_env))
    return Added(p, added, providers.mask(value), key_id(value), n, config)


def print_added(result: Added) -> None:
    p, config = result.provider, result.config
    if not result.added:
        ui.warn(t("{name} already has this key; nothing changed.", name=p.name))
        return
    ui.ok(
        t(
            "Added {key} to {name} (key #{number}, {number} in total).",
            key=result.masked,
            name=p.name,
            number=result.number,
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
        ui.warn(providers.untested_note())
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
    p = keyed_provider(provider)
    if remove_all == (number is not None):
        raise click.UsageError(t("give a key NUMBER or --all"))
    print_removed(remove_keys(load_config(), p, number))


@dataclass(frozen=True)
class Removal:
    """Keys of a provider that are to go (or went), as `desman keys` numbers them."""

    provider: providers.Provider
    count: int
    """How many keys the provider had."""
    removed: list[tuple[int, str, str]]
    """(number, masked key, label) of each key removed."""


def removal(config: Config, p: providers.Provider, number: int | None) -> Removal:
    """Which keys removing key `number` (None: all of them) of `p` takes; checked, not done."""
    assert p.key_env is not None
    keys = config.keys(p.key_env)
    if not keys:
        raise click.ClickException(t("{name} has no keys", name=p.name))
    if number is not None and not 1 <= number <= len(keys):
        raise click.BadParameter(
            tn(
                len(keys),
                "{name} has {n} key: pick 1",
                "{name} has {n} keys: pick 1 to {n}",
                name=p.name,
            ),
            param_hint="NUMBER",
        )
    numbers = range(1, len(keys) + 1) if number is None else [number]
    removed = [(i, providers.mask(keys[i - 1].value), keys[i - 1].label) for i in numbers]
    return Removal(p, len(keys), removed)


def remove_keys(manager: ConfigManager, p: providers.Provider, number: int | None) -> Removal:
    """Remove key `number` of `p`, or all of them; checked again under the config's lock."""
    assert p.key_env is not None
    with manager.edit() as config:
        done = removal(config, p, number)
        if number is None:
            config.api_keys.pop(p.key_env, None)
        else:
            config.remove_key(p.key_env, number - 1)
    return done


def print_removed(done: Removal) -> None:
    removed = len(done.removed)
    ui.ok(
        tn(
            removed,
            "Removed {n} key from {name}; {left} left.",
            "Removed {n} keys from {name}; {left} left.",
            name=done.provider.name,
            left=done.count - removed,
        )
    )


@keys.command("import")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def keys_import(file: Path) -> None:
    """Add the keys in FILE (JSON or NAME=value lines) to the config. Keys already saved and
    names no provider uses are skipped; values are never printed."""
    pairs = read_key_file(file)
    manager = load_config()
    print_imported(manager, file, import_keys(manager, pairs))


def read_key_file(file: Path) -> list[tuple[str, str]]:
    """The (variable, key) pairs in `file`; none is an error."""
    try:
        pairs = providers.parse_key_file(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise click.ClickException(t("cannot read {file}: {error}", file=file, error=e)) from e
    if not pairs:
        raise click.ClickException(t("no keys found in {file}", file=file))
    return pairs


@dataclass(frozen=True)
class Imported:
    added: dict[str, int]
    """Keys added, per provider name."""
    already: dict[str, int]
    unknown: list[str]
    """Variable names no provider uses."""


def import_keys(
    manager: ConfigManager, pairs: list[tuple[str, str]], *, dry_run: bool = False
) -> Imported:
    """Add `pairs` to the config; `dry_run` only counts what would be added."""
    if dry_run:
        return _add_pairs(manager.load().model_copy(deep=True), pairs)
    with manager.edit() as config:
        return _add_pairs(config, pairs)


def _add_pairs(config: Config, pairs: list[tuple[str, str]]) -> Imported:
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
    return Imported(added, already, list(dict.fromkeys(unknown)))


def print_imported(manager: ConfigManager, file: Path, result: Imported) -> None:
    added, already, unknown = result.added, result.already, result.unknown
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
                names=", ".join(unknown),
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
