"""What bare `desman` shows: the banner, the state of the setup and the commands."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from desman import __author__, __version__, providers, ui
from desman.config import ConfigManager

BANNER = """████░░█████░█████░█░░░█░█████░█░░░█
█░░░█░█░░░░░█░░░░░██░██░█░░░█░██░░█
█░░░█░█████░█████░█░█░█░█████░█░█░█
█░░░█░█░░░░░░░░░█░█░░░█░█░░░█░█░░██
████░░█████░█████░█░░░█░█░░░█░█░░░█"""

TAGLINE = "Decentralized Self-Evolving Modular Agents Network"

COMMANDS = {
    "en": [
        ('desman run "GOAL"', "plan and run a task"),
        ("desman list / status", "tasks and their progress"),
        ("desman inbox", "what waits for your decision"),
        ("desman keys", "API keys per provider"),
        ("desman config", "providers, keys, permissions"),
        ("desman help", "all commands"),
    ],
    "ru": [
        ('desman run "ЦЕЛЬ"', "спланировать и выполнить задачу"),
        ("desman list / status", "задачи и их ход"),
        ("desman inbox", "что ждёт вашего решения"),
        ("desman keys", "API-ключи по провайдерам"),
        ("desman config", "провайдеры, ключи, разрешения"),
        ("desman help", "все команды"),
    ],
}

TEXTS = {
    "en": {"ready": "{n} provider(s) ready", "none": "no provider ready: run `desman config`"},
    "ru": {"ready": "готово провайдеров: {n}", "none": "нет готовых провайдеров: `desman config`"},
}


def print_project_info(config_manager: ConfigManager) -> None:
    config = config_manager.load()
    lang = config.language
    t = TEXTS[lang]

    ui.out(Text(BANNER, style="accent"))
    ui.out(Text(TAGLINE, style="bold"))
    ui.out(Text(f"v{__version__} · {__author__}; AI", style="muted"))
    ui.out("")

    ready = providers.ready(config)
    status = (
        Text("● ", style="good") + Text(t["ready"].format(n=len(ready)))
        if ready
        else Text("● ", style="warn") + Text(t["none"], style="warn")
    )
    if ready:
        status.append("  " + ", ".join(p.name for p in ready), style="muted")

    commands = Table.grid(padding=(0, 3))
    commands.add_column(style="key", no_wrap=True)
    commands.add_column(style="muted")
    for cmd, what in COMMANDS[lang]:
        commands.add_row(cmd, what)

    ui.out(
        Panel(
            Group(status, Text(""), commands),
            border_style="accent",
            padding=(1, 2),
            expand=False,
        )
    )
    ui.note(f"config: {config_manager.config_path}")
