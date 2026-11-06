"""The interactive `desman config` wizard, in the configured language (`desman.i18n`).

Language, then a provider menu that stays open until "Done": pick a provider, add or remove
its keys (several per provider are fine) or change its settings, pick the next one. Its role
settings menu chooses the model and key each role asks first. Then the
permissions, then personalisation: whether the interlocutor may study the conversation and keep
notes about the person, and how it talks (`Config.behaviour`), which holds either way. Last,
whether this machine takes part in desman-global (off unless the person says yes). Nothing is
saved until the end; leaving with Ctrl+C or Esc saves nothing.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.panel import Panel
from rich.text import Text

from desman import config as config_module
from desman import i18n, net, providers, ui
from desman.config import BEHAVIOUR_CHARS, Behaviour, Config, ConfigManager, Network, Permissions
from desman.core import search
from desman.i18n import t, tn
from desman.providers import Provider

SEARXNG_DOC = "https://github.com/Temporary-Bot-5/DESMAN-P/blob/master/docs/searxng.md"


def permission_labels() -> dict[str, str]:
    return {
        "admin_rights": t("act with administrator / root privileges"),
        "fs_outside_access": t("touch files outside the DESMAN workspace"),
        "network_access": t("use the network beyond model providers"),
        "shell_execution": t("run shell commands"),
    }


def behaviour_labels() -> dict[str, str]:
    """What each part of `Behaviour` is, for the person."""
    return {
        "length": t("length of answers"),
        "tone": t("tone"),
        "style": t("style"),
        "custom": t("in your words"),
    }


def behaviour_values() -> dict[str, dict[str | None, str]]:
    """The choices of each part of `Behaviour`; None is the interlocutor's own judgement."""
    auto = t("as it sees fit")
    return {
        "length": {None: auto, "short": t("short"), "detailed": t("detailed")},
        "tone": {None: auto, "formal": t("formal"), "informal": t("informal")},
        "style": {None: auto, "simple": t("simple"), "technical": t("technical")},
    }


def behaviour_shown(behaviour: Behaviour, part: str) -> str:
    """One part of `behaviour` as the person reads it."""
    if part == "custom":
        return behaviour.custom or "-"
    value: str | None = getattr(behaviour, part)
    return behaviour_values()[part][value]


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
        t("Language"),
        [ui.Choice(catalog.name, code) for code, catalog in i18n.CATALOGS.items()],
        default=saved.language,
    )
    i18n.set_language(lang)  # the rest of the wizard, and of this command, in that language
    config = saved.model_copy(deep=True, update={"language": lang, "configured": True})
    session = _Session(config)

    _providers_menu(session)
    config.permissions = _permissions(config.permissions)
    config.personalize = ask_personalize(config.personalize)
    if ui.confirm(t("Set up how the interlocutor talks to you?"), default=False):
        config.behaviour = ask_behaviour(config.behaviour, config)
    entries, settle, said = network_change(config, ask_network(config))
    save(manager, config, seen, entries, settle)  # a `keys add` meanwhile is not lost
    said()

    ui.out("")
    ui.ok(t("Configuration saved to {path}", path=manager.config_path))
    if not providers.ready(config):
        ui.warn(t("No cloud key yet: cloud models will not work until you add one."))
    if config.permissions.admin_rights:
        ui.warn(t("admin_rights is on: agents may act with elevated privileges."))
    ui.note(t('Next: `desman config show` to review, `desman run "..."` to start.'))


SECTIONS = ("providers", "permissions", "personalize", "behaviour", "network", "roles")
"""Parts of the wizard that can be opened on their own (from the chat, ADR 0003 §3)."""


def run_section(manager: ConfigManager, section: str) -> None:
    """One section of the wizard; saved when it ends, nothing saved when the user leaves."""
    seen = manager.snapshot()
    config = manager.load().model_copy(deep=True)
    if section == "providers":
        _providers_menu(_Session(config))
    elif section == "permissions":
        config.permissions = _permissions(config.permissions)
    elif section == "personalize":
        config.personalize = ask_personalize(config.personalize)
    elif section == "behaviour":
        config.behaviour = ask_behaviour(config.behaviour, config)
    elif section == "roles":
        ask_roles(config)
    elif section == "network":
        entries, settle, said = network_change(config, ask_network(config))
        save(manager, config, seen, entries, settle)
        said()
        return
    else:
        raise ValueError(f"no wizard section {section!r}")
    manager.save_if_unchanged(config, seen)
    ui.ok(t("Configuration saved to {path}", path=manager.config_path))


def ask_roles(config: Config) -> None:
    """Edit role preferences on the wizard's unsaved copy via the shared change function."""
    from desman import tuning_cli

    ui.section(t("Models and keys for roles"))
    while True:
        rows = tuning_cli.role_rows(config)
        role = ui.select(
            t("Models and keys for roles"),
            [
                ui.Choice(
                    row["label"],
                    row["role"],
                    f"{row['model'] or tuning_cli.AUTO} · {row['key_label']}",
                )
                for row in rows
            ]
            + [ui.Choice(t("Done — continue"), None)],
        )
        if role is None:
            return
        row = next(row for row in rows if row["role"] == role)
        model_choices = [ui.Choice(t("DESMAN's choice"), tuning_cli.AUTO)] + [
            ui.Choice(label, label) for label in row["candidate_models"]
        ]
        if row["model"] and row["model"] not in row["candidate_models"]:
            model_choices.append(
                ui.Choice(f"{row['model']} ({t('not available now')})", row["model"])
            )
        model = ui.select(
            t("model: {role}", role=role),
            model_choices,
            default=row["model"] or tuning_cli.AUTO,
        )
        provider = None if model == tuning_cli.AUTO else model.split("/", 1)[0]
        keys = [k for k in row["candidate_keys"] if provider is None or k["provider"] == provider]
        key_choices = [ui.Choice(t("DESMAN's choice"), tuning_cli.AUTO)] + [
            ui.Choice(f"{k['provider']}: {k['label']}", k["selector"]) for k in keys
        ]
        # None means keep the saved fingerprint, even if its key is missing or filtered out.
        if row["key_fingerprint"] and not any(
            k["fingerprint"] == row["key_fingerprint"] for k in keys
        ):
            key_choices.append(ui.Choice(f"{row['key_label']} ({t('not available now')})", None))
        key = ui.select(
            t("key: {role}", role=role),
            key_choices,
            default=next(
                (k["selector"] for k in keys if k["fingerprint"] == row["key_fingerprint"]),
                None if row["key_fingerprint"] else tuning_cli.AUTO,
            ),
        )
        # Persist with the rest of the wizard, preserving cancellation and conflict checks.
        old, config.tuning, _ = tuning_cli.changed_tuning(
            config,
            tuning_cli.TuningChange(models={role: model}, keys={} if key is None else {role: key}),
        )
        tuning_cli.print_change(old, config.tuning, config)


def _status(config: Config, p: Provider) -> str:
    if p.key_env is None:
        missing = [s.label for s in p.settings if not s.default and not config.settings.get(s.env)]
        if p.local and missing:
            return t("needs {what}", what=", ".join(missing))
        if p.local:
            return f"{p.model(config)} ✓"
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
        # Same order as providers.PROVIDERS (free, paid, local), then the search engines, so
        # the numbers stay stable.
        choices: list[ui.Choice | ui.Heading] = []
        for title, group in (
            (t("Free tier"), providers.FREE),
            (t("Paid: billed per use"), providers.PAID),
            (t("Local"), providers.LOCAL),
            (t("Web search (for the web_search tool)"), providers.SEARCH),
            (t("Models for local servers"), providers.HUB),
        ):
            choices.append(ui.Heading(title))
            choices += [ui.Choice(p.name, p, _status(s.config, p)) for p in group]
        choices.append(ui.Choice(t("Done — continue"), None))
        choices.append(ui.Choice(t("Models and keys for roles"), "roles"))
        picked = ui.select(
            t("Pick a provider to add keys or change settings"), choices, default=last
        )
        if picked is None:
            return
        if picked == "roles":
            ask_roles(s.config)
            continue
        last = picked
        if picked.key_env is None or s.config.keys(picked.key_env):
            _provider_menu(s, picked)
        else:  # nothing to choose yet: straight to the key
            _add_key(s, picked)


def _provider_menu(s: _Session, p: Provider) -> None:
    if p.id == "searxng":
        ui.note(
            t(
                "SearXNG is a search engine you run yourself: it asks Google, Bing, DuckDuckGo "
                "and others for you, without keys or accounts. How to start one: {doc}",
                doc=SEARXNG_DOC,
            )
        )
    if p.local:
        ui.note(
            t(
                "A model on your own machine: free, private while its server runs here, but "
                "slow. Start {name}, set the model it serves, then check it with `desman keys "
                "check {id}`. DESMAN asks it when the cloud models cannot take a call. How to "
                "start it: {url}",
                name=p.name,
                id=p.id,
                url=p.url,
            )
        )
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
        if p.local:
            actions.append(ui.Choice(t("Download a model from Hugging Face"), "hub"))
        actions.append(ui.Choice(t("Back"), None))
        action = ui.select(p.name, actions)
        if action is None:
            return
        if action == "hub":
            _download(s, p)
        elif action == "add":
            _add_key(s, p)
        elif action == "remove":
            _remove_key(s, p)
        else:
            _set(s, p, action)


def _download(s: _Session, p: Provider) -> None:
    """A model from Hugging Face for local server `p`: its size and the free memory first, then
    the person's "yes", then the download."""
    import click
    import httpx

    from desman import local_models
    from desman.llm import hub

    text = ui.text(
        t("Model on Hugging Face (author/model or author/model:Q4_K_M); Enter to cancel")
    )
    if not text:
        return
    try:
        ref = local_models.parse(text)
        with httpx.Client() as client:
            found = local_models.look_up(s.config, ref, client)
            lines, warnings = local_models.describe(found, hub.memory())
            planned, cautions = local_models.plan(s.config, found, p)
            for line in lines + planned:
                ui.note(line)
            for warning in warnings + cautions:
                ui.warn(warning)
            if p.id == "ollama":  # the pull is done at once; the settings when the wizard ends
                ui.note(t("Ollama keeps the downloaded model even if you leave without saving."))
            question = t("Download it anyway?") if warnings else t("Download it and use it?")
            if not ui.confirm(question, default=not warnings):
                return
            local_models.fetch(s.config, found, p, client)
    except click.ClickException as e:
        ui.warn(e.format_message())
        return
    name = local_models.use(s.config, found, p)
    ui.ok(t("{server} model set to {model}.", server=p.name, model=name))


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
    if not s.privacy_ok and not p.hub:  # a Hugging Face token reads listings: no task data
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
        if p.id == "searxng":
            _check_searxng(value)
    if p.local and setting.env not in (p.model_env, p.context_env):
        address = p.setting(s.config, setting.env)
        if not providers.on_this_machine(address):
            ui.warn(
                t(
                    "{url} is not this machine: tasks' prompts, context and data will be sent "
                    "there.",
                    url=address,
                )
            )


def _check_searxng(url: str) -> None:
    problem = search.check_searxng(url, search.client())
    if problem is None:
        ui.ok(t("SearXNG answers: web search works."))
    else:
        ui.warn(t("SearXNG at {url} cannot be used yet: {problem}", url=url, problem=problem))


def _permissions(current: Permissions) -> Permissions:
    ui.section(t("Agent permissions"))
    ui.note(
        t(
            "This is prototype software and agents can act unexpectedly. Grant only what you accept the risks of. Everything is off unless you turn it on."
        )
    )
    names = list(Permissions.model_fields)
    labels = permission_labels()
    granted = ui.checkbox(
        t("Allow agents to"),
        [ui.Choice(labels[name], name) for name in names],
        checked={name for name in names if getattr(current, name)},
    )
    return Permissions(**{name: name in granted for name in names})


def ask_personalize(current: bool) -> bool:
    ui.section(t("Personalisation"))
    ui.note(
        t(
            "If you agree, the interlocutor studies how you talk with it, anything you write "
            "included, and keeps short notes about you (/profile in the chat shows and erases "
            "them). The notes go to the model providers with every message."
        )
    )
    return ui.confirm(
        t("Do you want the agent to study your conversations and become more personal to you?"),
        default=current,
    )


def ask_behaviour(current: Behaviour, config: Config) -> Behaviour:
    """How the interlocutor talks: each part chosen, Enter keeps it; a text of your own."""
    ui.section(t("How the interlocutor talks"))
    ui.note(t("This holds whether or not it keeps notes about you."))
    labels, values = behaviour_labels(), behaviour_values()
    chosen: dict[str, object] = {}
    for part, options in values.items():
        chosen[part] = ui.select(
            labels[part].capitalize(),
            [ui.Choice(label, value) for value, label in options.items()],
            default=getattr(current, part),
        )
    custom = ui.text(
        t("Anything else about how it should talk ('-' for nothing)"), default=current.custom
    )
    custom, cut = providers.redact_keys(" ".join(custom.split()), config)
    if cut:
        ui.warn(t("That looked like an API key, so it was cut."))
    if len(custom) > BEHAVIOUR_CHARS:
        ui.warn(t("Cut to {n} characters.", n=BEHAVIOUR_CHARS))
        custom = custom[:BEHAVIOUR_CHARS]
    chosen["custom"] = "" if custom == "-" else custom
    return Behaviour.model_validate(chosen)


@dataclass(frozen=True)
class NetworkChoice:
    join: bool
    registry: str | None = None
    """An address the person typed (asked only when none is known)."""


def ask_network(config: Config) -> NetworkChoice:
    """Whether to take part in desman-global; `network_change` applies it to the config as
    `desman global join|leave` would, and it is saved with the rest."""
    ui.section(t("desman-global"))
    ui.note(
        t(
            "desman-global is an opt-in network of DESMANs on people's machines, coordinated by "
            "the project's registry. It is off unless you say yes, and while it is off nothing "
            "leaves this machine for it. Taking part lends nothing by itself: you choose each "
            "model provider to lend, and its daily quota, with `desman global donate PROVIDER`."
        )
    )
    if not ui.confirm(t("Take part in desman-global?"), default=config.network.enabled):
        return NetworkChoice(join=False)
    if config.network.registry is not None or net.DEFAULT_REGISTRY is not None:
        return NetworkChoice(join=True)
    while True:
        address = ui.text(t("The registry's address, https://... ('-' for not now)")).strip()
        if address in ("", "-"):
            ui.note(t("Not now: desman-global stays off."))
            return NetworkChoice(join=False)
        try:
            Network(registry=address)
        except ValueError:
            ui.error(t("{url} is not a registry address: https://host[:port][/path].", url=address))
            continue
        return NetworkChoice(join=True, registry=address)


Settle = Callable[[sqlite3.Connection], None]
"""What runs once the wizard's config is saved, still under the config's lock."""


def network_change(
    config: Config, choice: NetworkChoice
) -> tuple[list[tuple[str, dict[str, Any]]], Settle | None, Callable[[], None]]:
    """`config` changed as the person chose (in place), the journal entries saying how, what
    to do once it is saved (leaving: the registry's goodbye), and what to tell them then.
    Raises before anything is saved if the node's key cannot be used."""
    from desman import global_cli  # it reaches the wizard through config_cli

    was_on = config.network.enabled
    if choice.join and (not was_on or choice.registry is not None):
        key = global_cli.usable_node_key()
        assert key is not None
        joined, entry = global_cli.joining(config, choice.registry, key)
        return ([entry] if entry else []), None, lambda: global_cli.print_joined(joined)
    if not choice.join and was_on:
        left, entries = global_cli.leaving(config)
        done = [left]

        def settle(conn: sqlite3.Connection) -> None:
            # Only once the config is saved off: a save refused sends nothing.
            done[0] = global_cli.farewell(left, conn=conn)

        return entries, settle, lambda: global_cli.print_left(done[0])
    return [], None, lambda: None


def save(
    manager: ConfigManager,
    config: Config,
    seen: str | None,
    entries: list[tuple[str, dict[str, Any]]],
    settle: Settle | None = None,
) -> None:
    """The wizard's config, and the network's journal entries with it: both or neither.
    `settle` runs after the save under the same lock, so no `join` slips in between."""
    if not entries and settle is None:
        manager.save_if_unchanged(config, seen)
        return
    from desman import global_cli

    with global_cli.journal(None) as conn, manager.locked():
        manager.check_unchanged(seen)
        global_cli.commit(conn, entries, lambda: manager.save(config))
        if settle is not None:
            settle(conn)
