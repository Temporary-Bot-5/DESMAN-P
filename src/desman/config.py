"""Configuration management for DESMAN."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from desman import providers

CONFIG_DIR_ENV = "DESMAN_CONFIG_DIR"


def project_root() -> Path | None:
    """The source checkout DESMAN runs from (src/desman/config.py -> repo), or None when it is
    installed as a package and there is no project folder to keep the config in."""
    root = Path(__file__).resolve().parents[2]
    return (
        root if (root / "pyproject.toml").is_file() and (root / "src" / "desman").is_dir() else None
    )


def default_config_path() -> Path:
    """Config file location, first that applies:

    1. $DESMAN_CONFIG_DIR/config.json;
    2. config.json in the project folder (a source checkout); `.gitignore` keeps it out of git;
    3. ~/.config/desman/config.json for an installed package.
    """
    config_dir = os.environ.get(CONFIG_DIR_ENV)
    if config_dir:
        return Path(config_dir) / "config.json"
    root = project_root()
    base = root if root is not None else home_config_dir()
    return base / "config.json"


def home_config_dir() -> Path:
    return Path.home() / ".config" / "desman"


def open_config() -> ConfigManager:
    """The ConfigManager the CLI uses. A checkout that used to keep its config in the home
    directory moves it into the project folder once, so saved keys are not lost."""
    path = default_config_path()
    old = home_config_dir() / "config.json"
    if path != old and not path.exists() and old.is_file() and CONFIG_DIR_ENV not in os.environ:
        adopt_config(old, path)
    return ConfigManager(path)


def adopt_config(old: Path, new: Path) -> None:
    """Move `old` to `new` (0600). A config already at `new` wins and `old` is left alone."""
    data = old.read_bytes()
    try:
        fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # created meanwhile, e.g. by another `desman`
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        new.unlink(missing_ok=True)
        raise
    old.unlink()
    click.echo(f"Moved the configuration from {old} to {new}.", err=True)


SCHEMA_VERSION = 1

# Renamed to the provider's own variable name (schema 0 -> 1).
_RENAMED_KEYS = {"VERCEL_AI_GATEWAY_KEY": "AI_GATEWAY_API_KEY"}


class ConfigError(Exception):
    """config.json exists but cannot be used. Never treated as "not configured": the wizard
    would then write over the keys in it."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path


class Permissions(BaseModel):
    """Set by the human in `desman config` only; the upper bound of what agents may do."""

    model_config = ConfigDict(extra="forbid")

    admin_rights: StrictBool = False
    fs_outside_access: StrictBool = False
    network_access: StrictBool = False
    shell_execution: StrictBool = False


class Config(BaseModel):
    """The whole of config.json."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    configured: StrictBool = False
    language: Literal["en", "ru"] = "en"
    permissions: Permissions = Field(default_factory=Permissions)
    api_keys: dict[str, StrictStr] = Field(default_factory=dict)  # secrets
    settings: dict[str, StrictStr] = Field(default_factory=dict)  # not secret: hosts, account ids


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a parsed config.json up to SCHEMA_VERSION. Schema 0 is the file written before
    there was a schema_version (it had `version: "0.1.0"` and `local_settings`)."""
    version = data.get("schema_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise ValueError(f"schema_version must be a whole number, not {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {version} is newer than this DESMAN understands ({SCHEMA_VERSION})"
        )
    data = dict(data)
    if version == 0:
        data.pop("version", None)
        data["settings"] = data.pop("local_settings", {})
        keys = data.get("api_keys")
        if isinstance(keys, dict):
            data["api_keys"] = {_RENAMED_KEYS.get(k, k): v for k, v in keys.items()}
        data["schema_version"] = 1
    return data


class ConfigManager:
    """Reads and writes config.json."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        # Only a directory we create gets 0700; an existing one (maybe chosen by the user via
        # DESMAN_CONFIG_DIR) is left alone. The file itself is always 0600 (save).
        self.config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def load(self) -> Config:
        """The saved config; an unconfigured one when there is no file. Raises ConfigError
        when the file cannot be read, is not JSON or does not fit the schema."""
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return Config()
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(self.config_path, f"cannot read it ({e})") from e
        try:
            data = json.loads(text)
        except ValueError as e:
            raise ConfigError(self.config_path, f"not valid JSON ({e})") from e
        if not isinstance(data, dict):
            raise ConfigError(self.config_path, "not a JSON object")
        try:
            return Config.model_validate(upgrade(data))
        except ValueError as e:  # pydantic's ValidationError is one
            raise ConfigError(self.config_path, str(e)) from e

    def is_configured(self) -> bool:
        return self.load().configured

    def save(self, config: Config) -> None:
        """Save configuration atomically, readable by the owner only (it holds API keys).

        Written to a private temporary file and renamed over the old one: a crash never
        leaves a half-written or world-readable config behind. A symlinked config (e.g. into
        a dotfiles directory) is updated at its target, so the link stays. The old file is
        made private before it is replaced, so another hard link to it does not keep the
        previous keys readable by others.
        """
        text = config.model_copy(update={"configured": True}).model_dump_json(indent=2)
        target = self.config_path.resolve() if self.config_path.is_symlink() else self.config_path
        tmp = target.with_name(f".{target.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            if target.exists():
                os.chmod(target, 0o600)  # the old inode, shared by any other hard link
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


# ─── Color constants (RGB tuples for true color) ───
PRIMARY = (0, 238, 255)  # #00EEFF - Cyan - main color
WARNING = (255, 0, 0)  # #ff0000 - Red - warnings
INTERACTIVE = (0, 255, 153)  # #00ff99 - Green - interactive elements
DIM = (136, 136, 136)  # #888888 - Dim text


def style_primary(text: str) -> str:
    return click.style(text, fg=PRIMARY)


def style_warning(text: str) -> str:
    return click.style(text, fg=WARNING, bold=True)


def style_interactive(text: str) -> str:
    return click.style(text, fg=INTERACTIVE, bold=True)


def style_dim(text: str) -> str:
    return click.style(text, fg=DIM)


def style_success(text: str) -> str:
    return click.style(text, fg=INTERACTIVE, bold=True)


def style_error(text: str) -> str:
    return click.style(text, fg=WARNING, bold=True)


# ─── Language strings ───
TEXTS: dict[str, dict[str, Any]] = {
    "en": {
        "welcome_title": "DESMAN(P) — Configuration",
        "welcome_subtitle": "Multimodal Weak Agent Framework",
        "language_prompt": "Select language / Выберите язык:",
        # Step 1 (now after mode selection for API/both)
        "step1_title": "STEP 1 — Data Privacy",
        "step1_body": (
            "DESMAN sends your prompts, context, and data to external API providers "
            "when you use cloud models. This means your conversations leave your machine."
        ),
        "step1_confirm": "I understand my data will be processed by third parties.",
        # Step 2
        "step2_title": "STEP 2 — Agent Permissions",
        "step2_body": (
            "This is prototype software. Agents can act unexpectedly. "
            "You control their capabilities below — enabling permissions means you accept the associated risks."
        ),
        "step2_perm_admin": "Administrator / root privileges",
        "step2_perm_fs": "File system access outside DESMAN working directory",
        "step2_perm_net": "Network access (beyond API providers)",
        "step2_perm_shell": "Shell command execution",
        # Step 3
        "step3_title": "STEP 3 — Model Providers",
        "step3_mode_prompt": "How do you want to run models?",
        "step3_mode_options": [
            "Cloud APIs only (requires API keys)",
            "Local models only (Ollama, llama.cpp)",
            "Both — I'll configure both",
        ],
        "step3_api_warning": (
            "Without at least one API key, cloud models will not work. "
            "You can add keys later via `desman config`."
        ),
        "step3_local_info": (
            "Local models run on your hardware. Ensure Ollama (ollama.com) "
            "or llama.cpp is installed and a model is pulled."
        ),
        "step3_select_prompt": "Enter provider numbers to configure (comma-separated), or press Enter to skip:",
        "step3_key_prompt": "API key for {name} ({env}):",
        "step3_key_keep": "[saved: Enter keeps it, '-' deletes it]",
        "step3_local_prompt": "Setting for {name} ({env}) [current: {default}; '-' clears it]:",
        "complete_title": "Configuration Complete",
        "complete_msg": "DESMAN is ready. Run `desman` to start.",
        "config_path": "Config saved to",
        "already_configured": "DESMAN is already configured. Use `desman config` to reconfigure.",
        "aborted": "Configuration aborted.",
    },
    "ru": {
        "welcome_title": "DESMAN(P) — Настройка",
        "welcome_subtitle": "Мультимодальная структура слабых агентов",
        "language_prompt": "Select language / Выберите язык:",
        # Step 1
        "step1_title": "ШАГ 1 — Приватность данных",
        "step1_body": (
            "DESMAN отправляет ваши промпты, контекст и данные внешним API-провайдерам "
            "при использовании облачных моделей. Это означает, что ваши диалоги покидают ваш компьютер."
        ),
        "step1_confirm": "Я понимаю, что мои данные будут обработаны третьими сторонами.",
        # Step 2
        "step2_title": "ШАГ 2 — Разрешения агентов",
        "step2_body": (
            "Это прототип. Агенты могут действовать неожиданно. "
            "Вы сами управляете их возможностями ниже — включение разрешения означает принятие связанных рисков."
        ),
        "step2_perm_admin": "Права администратора / root",
        "step2_perm_fs": "Доступ к файлам за пределами рабочей директории DESMAN",
        "step2_perm_net": "Сетевой доступ (кроме API-провайдеров)",
        "step2_perm_shell": "Выполнение shell-команд",
        # Step 3
        "step3_title": "ШАГ 3 — Провайдеры моделей",
        "step3_mode_prompt": "Как вы хотите запускать модели?",
        "step3_mode_options": [
            "Только облачные API (нужны API-ключи)",
            "Только локальные модели (Ollama, llama.cpp)",
            "И то, и другое — настрою оба варианта",
        ],
        "step3_api_warning": (
            "Без хотя бы одного API-ключа облачные модели работать не будут. "
            "Ключи можно добавить позже командой `desman config`."
        ),
        "step3_local_info": (
            "Локальные модели запускаются на вашем железе. Убедитесь, что установлен "
            "Ollama (ollama.com) или llama.cpp и загружена хотя бы одна модель."
        ),
        "step3_select_prompt": "Введите номера провайдеров для настройки (через запятую), или Enter для пропуска:",
        "step3_key_prompt": "API-ключ для {name} ({env}):",
        "step3_key_keep": "[сохранён: Enter — оставить, '-' — удалить]",
        "step3_local_prompt": "Настройка {name} ({env}) [текущее: {default}; '-' — очистить]:",
        "complete_title": "Настройка завершена",
        "complete_msg": "DESMAN готов. Запустите `desman` для старта.",
        "config_path": "Конфиг сохранён в",
        "already_configured": "DESMAN уже настроен. Используйте `desman config` для перенастройки.",
        "aborted": "Настройка прервана.",
    },
}


def get_text(lang: str, key: str) -> str:
    text: str = TEXTS.get(lang, TEXTS["en"]).get(key, key)
    return text


def select_language() -> str:
    """Prompt user to select language."""
    click.echo()
    click.echo(style_primary("═" * 60))
    click.echo(style_primary("  DESMAN(P) — Configuration"))
    click.echo(style_primary("  Multimodal Weak Agent Framework"))
    click.echo(style_primary("═" * 60))
    click.echo()
    click.echo(style_interactive("Select language / Выберите язык:"))
    click.echo(f"  {style_interactive('[1]')} English")
    click.echo(f"  {style_interactive('[2]')} Русский")
    click.echo()

    while True:
        choice = click.prompt(style_interactive("Choice [1/2]"), default="1", show_default=False)
        if choice in ("1", "2"):
            return "en" if choice == "1" else "ru"
        click.echo(style_error("Please enter 1 or 2 / Введите 1 или 2"))


def confirm_step(lang: str, message: str) -> bool:
    """Ask for confirmation with styled prompt. Enter = No."""
    click.echo()
    click.echo(style_warning(message))
    click.echo()
    while True:
        choice = click.prompt(style_interactive("Continue? [Y/N]"), default="N", show_default=False)
        choice = choice.strip().lower()
        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no", ""):
            click.echo(style_error("Aborted."))
            raise SystemExit(1)
        click.echo(style_error("Please enter Y or N"))


def _run_provider_selection(config: Config, lang: str, t: dict[str, Any]) -> Config:
    """Ask for providers. Starts from what is saved: keys and settings that are not re-entered
    stay. Enter keeps a saved value, '-' deletes it."""
    click.echo(style_interactive(t["step3_mode_prompt"]))
    for i, opt in enumerate(t["step3_mode_options"], 1):
        click.echo(f"  {style_interactive(f'[{i}]')} {opt}")
    click.echo()

    while True:
        mode_choice = click.prompt(
            style_interactive("Choice [1/2/3]"), default="1", show_default=False
        )
        if mode_choice in ("1", "2", "3"):
            break
        click.echo(style_error("Please enter 1, 2, or 3"))

    mode = int(mode_choice)  # 1=api, 2=local, 3=both
    api_keys = dict(config.api_keys)
    settings = dict(config.settings)

    if mode in (1, 3):
        # Step 1: Data Privacy (only for API/both modes)
        click.echo()
        click.echo(style_primary(f"━━━ {t['step1_title']} ━━━"))
        click.echo()
        click.echo(t["step1_body"])
        confirm_step(lang, t["step1_confirm"])

        click.echo()
        click.echo(style_warning(t["step3_api_warning"]))
        click.echo()
        click.echo(style_interactive("Available cloud providers:"))
        click.echo()
        for i, provider in enumerate(providers.CLOUD, 1):
            click.echo(f"  {style_interactive(f'[{i}]')} {style_primary(provider.name)}")
            click.echo(f"      {style_dim(provider.url)}")
        click.echo()

        click.echo(style_interactive(t["step3_select_prompt"]))
        selection = click.prompt("", default="", show_default=False)
        for part in selection.split(","):
            idx = int(part) - 1 if part.strip().isdigit() else -1
            if not 0 <= idx < len(providers.CLOUD):
                continue
            provider = providers.CLOUD[idx]
            assert provider.key_env is not None
            prompt = t["step3_key_prompt"].format(name=provider.name, env=provider.key_env)
            if provider.key_env in api_keys:
                prompt += " " + t["step3_key_keep"]
            key = click.prompt(
                style_interactive(prompt), default="", show_default=False, hide_input=True
            ).strip()
            if key == "-":
                api_keys.pop(provider.key_env, None)
            elif key:
                api_keys[provider.key_env] = key
            for setting in provider.settings:
                _prompt_setting(settings, provider.name, setting.env, setting.default, t)

    if mode in (2, 3):
        click.echo()
        click.echo(style_primary(t["step3_local_info"]))
        click.echo()
        click.echo(style_interactive("Local provider settings:"))
        click.echo()
        for i, provider in enumerate(providers.LOCAL, 1):
            click.echo(f"  {style_interactive(f'[{i}]')} {style_primary(provider.name)}")
            click.echo()
            for setting in provider.settings:
                _prompt_setting(settings, provider.name, setting.env, setting.default, t)

    return config.model_copy(update={"api_keys": api_keys, "settings": settings})


def _prompt_setting(
    settings: dict[str, str], name: str, env: str, default: str, t: dict[str, Any]
) -> None:
    current = settings.get(env, "") or default
    val = click.prompt(
        style_interactive(
            t["step3_local_prompt"].format(name=name, env=env, default=current or "none")
        ),
        default=current,
        show_default=True,
    ).strip()
    if val == "-":
        settings.pop(env, None)
    elif val:
        settings[env] = val


def _prompt_yes_no(prompt_text: str) -> bool:
    """Prompt for yes/no. Enter = No."""
    while True:
        choice = click.prompt(style_interactive(prompt_text), default="N", show_default=False)
        choice = choice.strip().lower()
        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no", ""):
            return False
        click.echo(style_error("Please enter Y or N"))


def _run_permissions_step(lang: str, t: dict[str, Any]) -> Permissions:
    """Ask for each permission; Enter means no."""
    click.echo()
    click.echo(style_primary(f"━━━ {t['step2_title']} ━━━"))
    click.echo()
    click.echo(t["step2_body"])
    click.echo()
    return Permissions(
        admin_rights=_prompt_yes_no(f"  [?] {t['step2_perm_admin']} [Y/N]"),
        fs_outside_access=_prompt_yes_no(f"  [?] {t['step2_perm_fs']} [Y/N]"),
        network_access=_prompt_yes_no(f"  [?] {t['step2_perm_net']} [Y/N]"),
        shell_execution=_prompt_yes_no(f"  [?] {t['step2_perm_shell']} [Y/N]"),
    )


def run_initial_config(config_manager: ConfigManager, force: bool = False) -> None:
    """Run the initial configuration wizard."""
    if not force and config_manager.is_configured():
        click.echo(get_text("en", "already_configured"))
        return
    _wizard(config_manager, title=None)


def reconfigure(config_manager: ConfigManager) -> None:
    """Run the wizard again, starting from the saved configuration."""
    _wizard(config_manager, title="DESMAN(P) — Reconfiguration")


def _wizard(config_manager: ConfigManager, title: str | None) -> None:
    saved = config_manager.load()  # a broken file stops here: it is never written over
    lang = select_language()  # every time
    t = TEXTS[lang]

    click.echo()
    click.echo(style_primary("═" * 60))
    click.echo(style_primary(f"  {title or t['welcome_title']}"))
    if title is None:
        click.echo(style_primary(f"  {t['welcome_subtitle']}"))
    click.echo(style_primary("═" * 60))
    click.echo()

    # Step 3: Model Providers (mode selection first)
    click.echo(style_primary(f"━━━ {t['step3_title']} ━━━"))
    click.echo()
    config = _run_provider_selection(saved, lang, t)
    config = config.model_copy(
        update={"language": lang, "permissions": _run_permissions_step(lang, t)}
    )
    config_manager.save(config)

    click.echo()
    click.echo(style_success(f"✅ {t['complete_title']}!"))
    click.echo(f"{style_primary(t['config_path'])}: {config_manager.config_path}")
    click.echo()
    click.echo(t["complete_msg"])
