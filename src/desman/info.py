"""Project info display."""

from __future__ import annotations

import click

from desman import __author__, __version__
from desman.config import ConfigManager, style_dim, style_interactive, style_primary

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


def print_project_info(config_manager: ConfigManager) -> None:
    """Print project information with ASCII banner."""
    t = TEXTS.get(config_manager.get_language(), TEXTS["en"])

    # ASCII Banner
    click.echo(style_primary(BANNER))
    click.echo()
    click.echo(style_primary(TAGLINE))
    click.echo()

    # Info
    click.echo(f"{style_interactive(t['author'] + ':')} {__author__}; AI")
    click.echo(f"{style_interactive(t['version'] + ':')} {__version__}")
    click.echo()
    click.echo(style_interactive(t["description"] + ":"))
    click.echo(f"  {t['desc_body']}")
    click.echo()
    click.echo(style_interactive(t["commands"] + ":"))
    click.echo(f"  {style_dim(t['cmd_config'])}")
    click.echo(f"  {style_dim(t['cmd_help'])}")
    click.echo()
    click.echo(
        f"{style_interactive(t['config_path'] + ':')} {style_dim(str(config_manager.config_path))}"
    )
    click.echo()

    # Start prompt - wait for Enter
    click.echo(style_interactive(t["start_prompt"]))
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        click.echo()
        click.echo(style_dim("Aborted."))
        raise SystemExit(0) from None
