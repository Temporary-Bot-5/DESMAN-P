"""Configuration management for DESMAN."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any

import click

CONFIG_DIR_ENV = "DESMAN_CONFIG_DIR"


def default_config_path() -> Path:
    """Config file location: $DESMAN_CONFIG_DIR/config.json or ~/.config/desman/config.json."""
    config_dir = os.environ.get(CONFIG_DIR_ENV)
    base = Path(config_dir) if config_dir else Path.home() / ".config" / "desman"
    return base / "config.json"


class ConfigManager:
    """Manages DESMAN configuration."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        # Only a directory we create gets 0700; an existing one (maybe chosen by the user via
        # DESMAN_CONFIG_DIR) is left alone. The file itself is always 0600 (save_config).
        self.config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def is_configured(self) -> bool:
        """Check if DESMAN is configured."""
        return bool(self._load_config().get("configured", False))

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if not self.config_path.exists():
            return {"configured": False}
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
                return data
        except (json.JSONDecodeError, OSError):
            return {"configured": False}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save configuration atomically, readable by the owner only (it holds API keys).

        Written to a private temporary file and renamed over the old one: a crash or a
        serialisation error never leaves a half-written or world-readable config behind.
        """
        config["configured"] = True
        text = json.dumps(config, indent=2, ensure_ascii=False)
        tmp = self.config_path.with_name(f".{self.config_path.name}.{secrets.token_hex(6)}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.config_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def get_api_keys(self) -> dict[str, str]:
        """Get stored API keys."""
        api_keys: dict[str, str] = self._load_config().get("api_keys", {})
        return api_keys

    def get_permissions(self) -> dict[str, bool]:
        """Get agent permissions."""
        permissions: dict[str, bool] = self._load_config().get("permissions", {})
        return permissions

    def get_language(self) -> str:
        """Get configured language."""
        language: str = self._load_config().get("language", "en")
        return language

    def get_local_settings(self) -> dict[str, str]:
        """Get local provider settings."""
        local_settings: dict[str, str] = self._load_config().get("local_settings", {})
        return local_settings


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
        "step3_api_providers": [
            {
                "name": "NVIDIA NIM",
                "env": "NVIDIA_API_KEY",
                "url": "https://build.nvidia.com/explore/discover",
            },
            {
                "name": "Google AI Studio",
                "env": "GOOGLE_API_KEY",
                "url": "https://aistudio.google.com/apikey",
            },
            {"name": "GROQ", "env": "GROQ_API_KEY", "url": "https://console.groq.com/keys"},
            {
                "name": "Mistral AI",
                "env": "MISTRAL_API_KEY",
                "url": "https://console.mistral.ai/api-keys/",
            },
            {
                "name": "Cloudflare Workers AI",
                "env": "CLOUDFLARE_API_TOKEN",
                "url": "https://dash.cloudflare.com/profile/api-tokens",
            },
            {
                "name": "Cohere",
                "env": "COHERE_API_KEY",
                "url": "https://dashboard.cohere.com/api-keys",
            },
            {"name": "Aion Labs", "env": "AION_API_KEY", "url": "https://aionlabs.ai/"},
            {
                "name": "OpenRouter",
                "env": "OPENROUTER_API_KEY",
                "url": "https://openrouter.ai/keys",
            },
            {
                "name": "Vercel AI Gateway",
                "env": "VERCEL_AI_GATEWAY_KEY",
                "url": "https://vercel.com/dashboard/stores/ai-gateway",
            },
        ],
        "step3_local_providers": [
            {
                "name": "Ollama",
                "env": "OLLAMA_HOST",
                "url": "https://ollama.com",
                "default": "http://localhost:11434",
            },
            {
                "name": "llama.cpp",
                "env": "LLAMA_CPP_PATH",
                "url": "https://github.com/ggerganov/llama.cpp",
                "default": "",
            },
        ],
        "step3_select_prompt": "Enter provider numbers to configure (comma-separated), or press Enter to skip:",
        "step3_key_prompt": "API key for {name} ({env}):",
        "step3_key_keep": "[saved: Enter keeps it, '-' deletes it]",
        "step3_local_prompt": "Setting for {name} ({env}) [current: {default}]:",
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
        "step3_api_providers": [
            {
                "name": "NVIDIA NIM",
                "env": "NVIDIA_API_KEY",
                "url": "https://build.nvidia.com/explore/discover",
            },
            {
                "name": "Google AI Studio",
                "env": "GOOGLE_API_KEY",
                "url": "https://aistudio.google.com/apikey",
            },
            {"name": "GROQ", "env": "GROQ_API_KEY", "url": "https://console.groq.com/keys"},
            {
                "name": "Mistral AI",
                "env": "MISTRAL_API_KEY",
                "url": "https://console.mistral.ai/api-keys/",
            },
            {
                "name": "Cloudflare Workers AI",
                "env": "CLOUDFLARE_API_TOKEN",
                "url": "https://dash.cloudflare.com/profile/api-tokens",
            },
            {
                "name": "Cohere",
                "env": "COHERE_API_KEY",
                "url": "https://dashboard.cohere.com/api-keys",
            },
            {"name": "Aion Labs", "env": "AION_API_KEY", "url": "https://aionlabs.ai/"},
            {
                "name": "OpenRouter",
                "env": "OPENROUTER_API_KEY",
                "url": "https://openrouter.ai/keys",
            },
            {
                "name": "Vercel AI Gateway",
                "env": "VERCEL_AI_GATEWAY_KEY",
                "url": "https://vercel.com/dashboard/stores/ai-gateway",
            },
        ],
        "step3_local_providers": [
            {
                "name": "Ollama",
                "env": "OLLAMA_HOST",
                "url": "https://ollama.com",
                "default": "http://localhost:11434",
            },
            {
                "name": "llama.cpp",
                "env": "LLAMA_CPP_PATH",
                "url": "https://github.com/ggerganov/llama.cpp",
                "default": "",
            },
        ],
        "step3_select_prompt": "Введите номера провайдеров для настройки (через запятую), или Enter для пропуска:",
        "step3_key_prompt": "API-ключ для {name} ({env}):",
        "step3_key_keep": "[сохранён: Enter — оставить, '-' — удалить]",
        "step3_local_prompt": "Настройка {name} ({env}) [текущее: {default}]:",
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


def _run_provider_selection(
    config_manager: ConfigManager, lang: str, t: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Run provider selection and return (api_keys, local_settings)."""
    # Mode selection
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
    # Start from what is saved: reconfiguring must not lose keys that are not re-entered.
    api_keys: dict[str, str] = dict(config_manager.get_api_keys())
    local_settings: dict[str, str] = dict(config_manager.get_local_settings())

    if mode in (1, 3):
        # Step 1: Data Privacy (only for API/both modes)
        click.echo()
        click.echo(style_primary(f"━━━ {t['step1_title']} ━━━"))
        click.echo()
        click.echo(t["step1_body"])
        confirm_step(lang, t["step1_confirm"])

        # Cloud API providers
        click.echo()
        click.echo(style_warning(t["step3_api_warning"]))
        click.echo()
        click.echo(style_interactive("Available cloud providers:"))
        click.echo()

        api_providers = t["step3_api_providers"]
        for i, p in enumerate(api_providers, 1):
            click.echo(f"  {style_interactive(f'[{i}]')} {style_primary(p['name'])}")
        click.echo()

        click.echo(style_interactive(t["step3_select_prompt"]))
        selection = click.prompt("", default="", show_default=False)
        if selection.strip():
            indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(api_providers):
                    p = api_providers[idx]
                    prompt = t["step3_key_prompt"].format(name=p["name"], env=p["env"])
                    if p["env"] in api_keys:
                        prompt += " " + t["step3_key_keep"]
                    key = click.prompt(
                        style_interactive(prompt),
                        default="",
                        show_default=False,
                        hide_input=True,
                    ).strip()
                    if key == "-":
                        api_keys.pop(p["env"], None)
                    elif key:
                        api_keys[p["env"]] = key

    if mode in (2, 3):
        # Local providers
        click.echo()
        click.echo(style_primary(t["step3_local_info"]))
        click.echo()
        click.echo(style_interactive("Local provider settings:"))
        click.echo()

        local_providers = t["step3_local_providers"]
        for i, p in enumerate(local_providers, 1):
            click.echo(f"  {style_interactive(f'[{i}]')} {style_primary(p['name'])}")
            click.echo()

            default_val = p.get("default", "")
            current = config_manager.get_local_settings().get(p["env"], "")
            default_val = current or default_val
            val = click.prompt(
                style_interactive(
                    t["step3_local_prompt"].format(
                        name=p["name"], env=p["env"], default=default_val or "none"
                    )
                ),
                default=default_val,
                show_default=True,
            )
            if val.strip():
                local_settings[p["env"]] = val.strip()

    return api_keys, local_settings


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


def _run_permissions_step(lang: str, t: dict[str, Any]) -> dict[str, bool]:
    """Run permissions step and return permissions dict."""
    click.echo()
    click.echo(style_primary(f"━━━ {t['step2_title']} ━━━"))
    click.echo()
    click.echo(t["step2_body"])
    click.echo()

    permissions: dict[str, bool] = {}
    permissions["admin_rights"] = _prompt_yes_no(f"  [?] {t['step2_perm_admin']} [Y/N]")
    permissions["fs_outside_access"] = _prompt_yes_no(f"  [?] {t['step2_perm_fs']} [Y/N]")
    permissions["network_access"] = _prompt_yes_no(f"  [?] {t['step2_perm_net']} [Y/N]")
    permissions["shell_execution"] = _prompt_yes_no(f"  [?] {t['step2_perm_shell']} [Y/N]")
    return permissions


def run_initial_config(config_manager: ConfigManager, force: bool = False) -> None:
    """Run the initial configuration wizard."""
    if not force and config_manager.is_configured():
        click.echo(get_text("en", "already_configured"))
        return

    # Language selection (every time)
    lang = select_language()
    t = TEXTS[lang]

    # Welcome
    click.echo()
    click.echo(style_primary("═" * 60))
    click.echo(style_primary(f"  {t['welcome_title']}"))
    click.echo(style_primary(f"  {t['welcome_subtitle']}"))
    click.echo(style_primary("═" * 60))
    click.echo()

    # Step 3: Model Providers (mode selection first)
    click.echo(style_primary(f"━━━ {t['step3_title']} ━━━"))
    click.echo()

    api_keys, local_settings = _run_provider_selection(config_manager, lang, t)

    # Step 2: Permissions
    permissions = _run_permissions_step(lang, t)

    # Save
    config = {
        "configured": True,
        "language": lang,
        "permissions": permissions,
        "api_keys": api_keys,
        "local_settings": local_settings,
        "version": "0.1.0",
    }
    config_manager.save_config(config)

    click.echo()
    click.echo(style_success(f"✅ {t['complete_title']}!"))
    click.echo(f"{style_primary(t['config_path'])}: {config_manager.config_path}")
    click.echo()
    click.echo(t["complete_msg"])


def reconfigure(config_manager: ConfigManager) -> None:
    """Run reconfiguration - language selection every time."""
    # Language selection (every time)
    lang = select_language()
    t = TEXTS[lang]

    click.echo()
    click.echo(style_primary("═" * 60))
    click.echo(style_primary("  DESMAN(P) — Reconfiguration"))
    click.echo(style_primary("═" * 60))
    click.echo()

    # Step 3: Model Providers (mode selection first)
    click.echo(style_primary(f"━━━ {t['step3_title']} ━━━"))
    click.echo()

    api_keys, local_settings = _run_provider_selection(config_manager, lang, t)

    # Step 2: Permissions
    permissions = _run_permissions_step(lang, t)

    # Save
    config = {
        "configured": True,
        "language": lang,
        "permissions": permissions,
        "api_keys": api_keys,
        "local_settings": local_settings,
        "version": "0.1.0",
    }
    config_manager.save_config(config)

    click.echo()
    click.echo(style_success(f"✅ {t['complete_title']}!"))
    click.echo(f"{style_primary(t['config_path'])}: {config_manager.config_path}")
    click.echo()
    click.echo(t["complete_msg"])
