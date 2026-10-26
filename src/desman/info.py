"""Project info display."""
import json
from pathlib import Path

import click

from desman import __version__, __author__

CONFIG_FILE = Path.home() / ".config" / "desman" / "config.json"


def get_configured_language() -> str:
    """Get language from config file."""
    if not CONFIG_FILE.exists():
        return "en"
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("language", "en")
    except (json.JSONDecodeError, OSError):
        return "en"


# ASCII Art Banner
BANNER = """████░░█████░█████░█░░░█░█████░█░░░█
█░░░█░█░░░░░█░░░░░██░██░█░░░█░██░░█
█░░░█░█████░█████░█░█░█░█████░█░█░█
█░░░█░█░░░░░░░░░█░█░░░█░█░░░█░█░░██
████░░█████░█████░█░░░█░█░░░█░█░░░█"""

TAGLINE = "Decentralized Self-Evolving Modular Agents Network"


TEXTS = {
    "en": {
        "author": "Author",
        "version": "Version",
        "description": "Description",
        "desc_body": "DESMAN is a multimodal weak agent framework requiring multiple API keys for different providers or local models.",
        "commands": "Commands",
        "cmd_config": "desman config     - Configure / reconfigure DESMAN",
        "cmd_help": "desman help       - Show this help",
        "config_path": "Config",
        "start_prompt": "Start? [Enter/Control+C]",
    },
    "ru": {
        "author": "Автор",
        "version": "Версия",
        "description": "Описание",
        "desc_body": "DESMAN — мультимодальная структура слабых агентов, требующая множества API ключей для разных провайдеров или локальных моделей.",
        "commands": "Команды",
        "cmd_config": "desman config     - Настроить/переконфигурировать DESMAN",
        "cmd_help": "desman help       - Показать эту справку",
        "config_path": "Конфиг",
        "start_prompt": "Начать? [Enter/Control+C]",
    },
}


# Color constants matching config.py (RGB tuples for true color)
PRIMARY = (0, 238, 255)      # #00EEFF
INTERACTIVE = (0, 255, 153)  # #00ff99
DIM = (136, 136, 136)        # #888888


def style_primary(text: str) -> str:
    return click.style(text, fg=PRIMARY)


def style_interactive(text: str) -> str:
    return click.style(text, fg=INTERACTIVE, bold=True)


def style_dim(text: str) -> str:
    return click.style(text, fg=DIM)


def print_project_info() -> None:
    """Print project information with ASCII banner."""
    lang = get_configured_language()
    t = TEXTS[lang]

    # ASCII Banner
    click.echo(style_primary(BANNER))
    click.echo()
    click.echo(style_primary(TAGLINE))
    click.echo()

    # Info
    click.echo(f"{style_interactive(t['author'] + ':')} Temporary Bot 5; AI")
    click.echo(f"{style_interactive(t['version'] + ':')} {__version__}")
    click.echo()
    click.echo(style_interactive(t['description'] + ":"))
    click.echo(f"  {t['desc_body']}")
    click.echo()
    click.echo(style_interactive(t['commands'] + ":"))
    click.echo(f"  {style_dim(t['cmd_config'])}")
    click.echo(f"  {style_dim(t['cmd_help'])}")
    click.echo()
    click.echo(f"{style_interactive(t['config_path'] + ':')} {style_dim(str(CONFIG_FILE))}")
    click.echo()

    # Start prompt - wait for Enter
    click.echo(style_interactive(t["start_prompt"]))
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        click.echo()
        click.echo(style_dim("Aborted."))
        raise SystemExit(0)
