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
from desman import providers, ui
from desman.config import Config, ConfigManager, Permissions
from desman.providers import Provider

TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "title": "DESMAN(P) — Configuration",
        "subtitle": "Decentralized Self-Evolving Modular Agents Network",
        "language": "Language",
        "providers": "Model providers",
        "providers_q": "Pick a provider to add keys or change settings",
        "providers_note": (
            "Several providers and several keys per provider are fine: pick one, then the next."
        ),
        "done": "Done — continue",
        "keys_n": "{n} keys",
        "key_1": "1 key",
        "no_key": "no key",
        "needs": "needs {what}",
        "local": "local",
        "unset": "not set",
        "privacy_title": "Data privacy",
        "privacy_body": (
            "Cloud models receive your prompts, context and data: they leave your machine and "
            "are processed by the provider."
        ),
        "privacy_q": "I understand my data will be processed by third parties",
        "key_q": "API key for {name} ({env}), hidden; Enter to cancel",
        "label_q": "Label for this key (optional, e.g. 'second account')",
        "added": "Key added to {name}.",
        "duplicate": "{name} already has this key.",
        "bad_label": "That label looks like a key, so it was left out: a label is a name.",
        "action_q": "{name}",
        "add_key": "Add a key",
        "remove_key": "Remove a key",
        "set": "Set {label} ({env})",
        "back": "Back",
        "which_key": "Which key?",
        "removed": "Key removed from {name}.",
        "setting_q": "{label} for {name} ({env}); '-' clears it",
        "get_key": "Get a key: {url}",
        "permissions": "Agent permissions",
        "permissions_body": (
            "This is prototype software and agents can act unexpectedly. Grant only what you "
            "accept the risks of. Everything is off unless you turn it on."
        ),
        "permissions_q": "Allow agents to",
        "perm_admin_rights": "act with administrator / root privileges",
        "perm_fs_outside_access": "touch files outside the DESMAN workspace",
        "perm_network_access": "use the network beyond model providers",
        "perm_shell_execution": "run shell commands",
        "admin_warning": "admin_rights is on: agents may act with elevated privileges.",
        "saved": "Configuration saved to {path}",
        "next": 'Next: `desman config show` to review, `desman run "..."` to start.',
        "no_cloud": "No cloud key yet: cloud models will not work until you add one.",
    },
    "ru": {
        "title": "DESMAN(P) — Настройка",
        "subtitle": "Децентрализованная самоэволюционирующая модульная сеть агентов",
        "language": "Язык",
        "providers": "Провайдеры моделей",
        "providers_q": "Выберите провайдера, чтобы добавить ключи или изменить настройки",
        "providers_note": (
            "Можно настроить несколько провайдеров и несколько ключей у каждого: выберите "
            "одного, потом следующего."
        ),
        "done": "Готово — дальше",
        "keys_n": "ключей: {n}",
        "key_1": "1 ключ",
        "no_key": "нет ключа",
        "needs": "нужен {what}",
        "local": "локальный",
        "unset": "не задано",
        "privacy_title": "Приватность данных",
        "privacy_body": (
            "Облачные модели получают ваши промпты, контекст и данные: они покидают ваш "
            "компьютер и обрабатываются провайдером."
        ),
        "privacy_q": "Я понимаю, что мои данные обработают третьи стороны",
        "key_q": "API-ключ для {name} ({env}), ввод скрыт; Enter — отмена",
        "label_q": "Метка ключа (необязательно, например «второй аккаунт»)",
        "added": "Ключ добавлен к {name}.",
        "duplicate": "У {name} уже есть этот ключ.",
        "bad_label": "Метка похожа на ключ, поэтому не сохранена: метка — это название.",
        "action_q": "{name}",
        "add_key": "Добавить ключ",
        "remove_key": "Удалить ключ",
        "set": "Задать {label} ({env})",
        "back": "Назад",
        "which_key": "Какой ключ?",
        "removed": "Ключ удалён у {name}.",
        "setting_q": "{label} для {name} ({env}); '-' — очистить",
        "get_key": "Получить ключ: {url}",
        "permissions": "Разрешения агентов",
        "permissions_body": (
            "Это прототип, агенты могут действовать неожиданно. Давайте только то, риски чего "
            "принимаете. Всё выключено, пока вы не включите."
        ),
        "permissions_q": "Разрешить агентам",
        "perm_admin_rights": "действовать с правами администратора / root",
        "perm_fs_outside_access": "работать с файлами вне рабочей папки DESMAN",
        "perm_network_access": "выходить в сеть помимо провайдеров моделей",
        "perm_shell_execution": "выполнять shell-команды",
        "admin_warning": "admin_rights включён: агенты могут действовать с повышенными правами.",
        "saved": "Настройки сохранены в {path}",
        "next": 'Дальше: `desman config show` — проверить, `desman run "..."` — запустить.',
        "no_cloud": "Облачных ключей пока нет: облачные модели не заработают, пока не добавите.",
    },
}


class _Session:
    def __init__(self, config: Config, t: dict[str, str]) -> None:
        self.config = config
        self.t = t
        # Accepted before if a cloud key was already saved; otherwise asked once, on first add.
        self.privacy_ok = any(config.keys(p.key_env or "") for p in providers.CLOUD)


def run(manager: ConfigManager) -> None:
    """The whole wizard. Raises ui.Cancelled when the user leaves; nothing is saved then."""
    seen = manager.snapshot()
    saved = manager.load()  # a broken file stops here: it is never written over
    ui.header(TEXTS["en"]["title"], TEXTS["en"]["subtitle"])
    lang = ui.select(
        "Language / Язык",
        [ui.Choice("English", "en"), ui.Choice("Русский", "ru")],
        default=saved.language,
    )
    t = TEXTS[lang]
    config = saved.model_copy(deep=True, update={"language": lang, "configured": True})
    session = _Session(config, t)

    _providers_menu(session)
    config.permissions = _permissions(config.permissions, t)
    manager.save_if_unchanged(config, seen)  # a `keys add` meanwhile is not lost

    ui.out("")
    ui.ok(t["saved"].format(path=manager.config_path))
    if not providers.ready(config):
        ui.warn(t["no_cloud"])
    if config.permissions.admin_rights:
        ui.warn(t["admin_warning"])
    ui.note(t["next"])


def _status(config: Config, p: Provider, t: dict[str, str]) -> str:
    if p.local:
        values = [config.settings.get(s.env) or s.default for s in p.settings]
        return " ".join(v for v in values if v) or t["unset"]
    n = len(config.keys(p.key_env or ""))
    if n == 0:
        return t["no_key"]
    count = t["key_1"] if n == 1 else t["keys_n"].format(n=n)
    missing = [s.label for s in p.settings if not s.default and not config.settings.get(s.env)]
    return f"{count} · {t['needs'].format(what=', '.join(missing))}" if missing else f"{count} ✓"


def _providers_menu(s: _Session) -> None:
    ui.section(s.t["providers"])
    ui.note(s.t["providers_note"])
    last: Provider | None = None
    while True:
        choices = [ui.Choice(p.name, p, _status(s.config, p, s.t)) for p in providers.PROVIDERS]
        choices.append(ui.Choice(s.t["done"], None))
        picked = ui.select(s.t["providers_q"], choices, default=last)
        if picked is None:
            return
        last = picked
        if picked.local or s.config.keys(picked.key_env or ""):
            _provider_menu(s, picked)
        else:  # nothing to choose yet: straight to the key
            _add_key(s, picked)


def _provider_menu(s: _Session, p: Provider) -> None:
    t = s.t
    while True:
        _show_keys(s.config, p)
        actions: list[ui.Choice] = []
        if p.key_env is not None:
            actions.append(ui.Choice(t["add_key"], "add"))
            if s.config.keys(p.key_env):
                actions.append(ui.Choice(t["remove_key"], "remove"))
        for setting in p.settings:
            current = s.config.settings.get(setting.env) or setting.default
            actions.append(
                ui.Choice(
                    t["set"].format(label=setting.label, env=setting.env),
                    setting,
                    current or t["unset"],
                )
            )
        actions.append(ui.Choice(t["back"], None))
        action = ui.select(t["action_q"].format(name=p.name), actions)
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
    t = s.t
    assert p.key_env is not None
    if not s.privacy_ok:
        ui.out(
            Panel(
                Text(t["privacy_body"]),
                title=t["privacy_title"],
                title_align="left",
                border_style="warn",
            )
        )
        if not ui.confirm(t["privacy_q"], default=False):
            return
        s.privacy_ok = True
    ui.note(t["get_key"].format(url=p.url))
    value = ui.secret(t["key_q"].format(name=p.name, env=p.key_env))
    if not value:
        return
    label = ui.text(t["label_q"]) if s.config.keys(p.key_env) else ""
    if config_module.looks_like_secret(label) or (label and label in value):
        ui.warn(t["bad_label"])
        label = ""
    if s.config.add_key(p.key_env, value, label):
        ui.ok(t["added"].format(name=p.name))
    else:
        ui.warn(t["duplicate"].format(name=p.name))
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
    choices.append(ui.Choice(s.t["back"], None))
    index = ui.select(s.t["which_key"], choices)
    if index is None:
        return
    s.config.remove_key(p.key_env, index)
    ui.ok(s.t["removed"].format(name=p.name))


def _set(s: _Session, p: Provider, setting: Any) -> None:
    current = s.config.settings.get(setting.env) or setting.default
    value = ui.text(
        s.t["setting_q"].format(label=setting.label, name=p.name, env=setting.env), current
    )
    if value == "-":
        s.config.settings.pop(setting.env, None)
    elif value:
        s.config.settings[setting.env] = value


def _permissions(current: Permissions, t: dict[str, str]) -> Permissions:
    ui.section(t["permissions"])
    ui.note(t["permissions_body"])
    names = list(Permissions.model_fields)
    granted = ui.checkbox(
        t["permissions_q"],
        [ui.Choice(t[f"perm_{name}"], name) for name in names],
        checked={name for name in names if getattr(current, name)},
    )
    return Permissions(**{name: name in granted for name in names})
