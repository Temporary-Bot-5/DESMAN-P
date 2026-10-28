"""The interactive `desman config` wizard (bilingual en/ru).

Language, then a provider menu that stays open until "Done": pick a provider, add or remove
its keys (several per provider are fine) or change its settings, pick the next one. Then the
permissions. Nothing is saved until the end; leaving with Ctrl+C or Esc saves nothing.
"""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.text import Text

from desman import config as config_module
from desman import i18n, providers, ui
from desman.config import Config, ConfigManager, Permissions
from desman.i18n import t, tn
from desman.providers import Provider


def _permission_labels() -> dict[str, str]:
    return {
        "admin_rights": t("act with administrator / root privileges"),
        "fs_outside_access": t("touch files outside the DESMAN workspace"),
        "network_access": t("use the network beyond model providers"),
        "shell_execution": t("run shell commands"),
    }


class _Session:
    def __init__(self, config: Config) -> None:
        self.config = config
        # Accepted before if a cloud key was already saved; otherwise asked once, on first add.
        self.privacy_ok = any(config.keys(p.key_env or "") for p in providers.CLOUD)


def run(manager: ConfigManager) -> None:
    """The whole wizard. Raises ui.Cancelled when the user leaves; nothing is saved then."""
    seen = manager.snapshot()
    saved = manager.load()  # a broken file stops here: it is never written over
    ui.header(
        t("DESMAN(P) — Configuration"), t("Decentralized Self-Evolving Modular Agents Network")
    )
    lang = ui.select(
        "Language / Язык",
        [ui.Choice(catalog.name, code) for code, catalog in i18n.CATALOGS.items()],
        default=saved.language,
    )
    i18n.set_language(lang)  # the rest of the wizard, and of this command, in that language
    config = saved.model_copy(deep=True, update={"language": lang, "configured": True})
    session = _Session(config)

    _providers_menu(session)
    config.permissions = _permissions(config.permissions)
    manager.save_if_unchanged(config, seen)  # a `keys add` meanwhile is not lost

    ui.out("")
    ui.ok(t("Configuration saved to {path}", path=manager.config_path))
    if not providers.ready(config):
        ui.warn(t("No cloud key yet: cloud models will not work until you add one."))
    if config.permissions.admin_rights:
        ui.warn(t("admin_rights is on: agents may act with elevated privileges."))
    ui.note(t('Next: `desman config show` to review, `desman run "..."` to start.'))


def _status(config: Config, p: Provider) -> str:
    if p.local:
        values = [config.settings.get(s.env) or s.default for s in p.settings]
        return " ".join(v for v in values if v) or t("not set")
    n = len(config.keys(p.key_env or ""))
    if n == 0:
        return t("no key")
    count = tn(n, "{n} key", "{n} keys")
    missing = [s.label for s in p.settings if not s.default and not config.settings.get(s.env)]
    if missing:
        return f"{count} · {t('needs {what}', what=', '.join(missing))}"
    return f"{count} ✓"


def _providers_menu(s: _Session) -> None:
    ui.section(t("Model providers"))
    ui.note(t("Several providers and several keys per provider are fine: pick one, then the next."))
    last: Provider | None = None
    while True:
        # Same order as providers.PROVIDERS (free, paid, local), so the numbers stay stable.
        choices: list[ui.Choice | ui.Heading] = []
        for title, group in (
            (t("Free tier"), providers.FREE),
            (t("Paid: billed per use"), providers.PAID),
            (t("Local"), providers.LOCAL),
        ):
            choices.append(ui.Heading(title))
            choices += [ui.Choice(p.name, p, _status(s.config, p)) for p in group]
        choices.append(ui.Choice(t("Done — continue"), None))
        picked = ui.select(
            t("Pick a provider to add keys or change settings"), choices, default=last
        )
        if picked is None:
            return
        last = picked
        if picked.local or s.config.keys(picked.key_env or ""):
            _provider_menu(s, picked)
        else:  # nothing to choose yet: straight to the key
            _add_key(s, picked)


def _provider_menu(s: _Session, p: Provider) -> None:
    while True:
        _show_keys(s.config, p)
        actions: list[ui.Choice] = []
        if p.key_env is not None:
            actions.append(ui.Choice(t("Add a key"), "add"))
            if s.config.keys(p.key_env):
                actions.append(ui.Choice(t("Remove a key"), "remove"))
        for setting in p.settings:
            current = s.config.settings.get(setting.env) or setting.default
            actions.append(
                ui.Choice(
                    t("Set {label} ({env})", label=setting.label, env=setting.env),
                    setting,
                    current or t("not set"),
                )
            )
        actions.append(ui.Choice(t("Back"), None))
        action = ui.select(p.name, actions)
        if action is None:
            return
        if action == "add":
            _add_key(s, p)
        elif action == "remove":
            _remove_key(s, p)
        else:
            _set(s, p, action)


def _show_keys(config: Config, p: Provider) -> None:
    if p.key_env is None:
        return
    keys = config.keys(p.key_env)
    rows = ui.table("#", p.key_env, "label")
    for i, k in enumerate(keys, 1):
        rows.add_row(str(i), providers.mask(k.value), providers.label(k.label))
    if keys:
        ui.out(rows)


def _add_key(s: _Session, p: Provider) -> None:
    assert p.key_env is not None
    if not s.privacy_ok:
        ui.out(
            Panel(
                Text(
                    t(
                        "Cloud models receive your prompts, context and data: they leave your machine and are processed by the provider."
                    )
                ),
                title=t("Data privacy"),
                title_align="left",
                border_style="warn",
            )
        )
        if not ui.confirm(
            t("I understand my data will be processed by third parties"), default=False
        ):
            return
        s.privacy_ok = True
    if p.paid and not s.config.keys(p.key_env):
        ui.out(
            Panel(
                Text(
                    t(
                        "Every call to {name} is billed to your account. DESMAN never picks a paid "
                        "provider by itself: it is used only with `desman run --provider {id}`. "
                        "Set a spending limit in the provider's console.",
                        name=p.name,
                        id=p.id,
                    )
                ),
                title=t("Paid provider"),
                title_align="left",
                border_style="warn",
            )
        )
        ui.warn(providers.untested_note())
        if not ui.confirm(t("I understand that calls to {name} cost money", name=p.name)):
            return
    ui.note(t("Get a key: {url}", url=p.url))
    value = ui.secret(
        t("API key for {name} ({env}), hidden; Enter to cancel", name=p.name, env=p.key_env)
    )
    if not value:
        return
    label = (
        ui.text(t("Label for this key (optional, e.g. 'second account')"))
        if s.config.keys(p.key_env)
        else ""
    )
    if config_module.looks_like_secret(label) or (label and label in value):
        ui.warn(t("That label looks like a key, so it was left out: a label is a name."))
        label = ""
    if s.config.add_key(p.key_env, value, label):
        ui.ok(t("Key added to {name}.", name=p.name))
    else:
        ui.warn(t("{name} already has this key.", name=p.name))
    for setting in p.settings:
        if not setting.default and not s.config.settings.get(setting.env):
            _set(s, p, setting)


def _remove_key(s: _Session, p: Provider) -> None:
    assert p.key_env is not None
    keys = s.config.keys(p.key_env)
    choices = [
        ui.Choice(f"#{i} {providers.mask(k.value)}", i - 1, providers.label(k.label))
        for i, k in enumerate(keys, 1)
    ]
    choices.append(ui.Choice(t("Back"), None))
    index = ui.select(t("Which key?"), choices)
    if index is None:
        return
    s.config.remove_key(p.key_env, index)
    ui.ok(t("Key removed from {name}.", name=p.name))


def _set(s: _Session, p: Provider, setting: Any) -> None:
    current = s.config.settings.get(setting.env) or setting.default
    value = ui.text(
        t(
            "{label} for {name} ({env}); '-' clears it",
            label=setting.label,
            name=p.name,
            env=setting.env,
        ),
        current,
    )
    if value == "-":
        s.config.settings.pop(setting.env, None)
    elif value:
        s.config.settings[setting.env] = value


def _permissions(current: Permissions) -> Permissions:
    ui.section(t("Agent permissions"))
    ui.note(
        t(
            "This is prototype software and agents can act unexpectedly. Grant only what you accept the risks of. Everything is off unless you turn it on."
        )
    )
    names = list(Permissions.model_fields)
    labels = _permission_labels()
    granted = ui.checkbox(
        t("Allow agents to"),
        [ui.Choice(labels[name], name) for name in names],
        checked={name for name in names if getattr(current, name)},
    )
    return Permissions(**{name: name in granted for name in names})
