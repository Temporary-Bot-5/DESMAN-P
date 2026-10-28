"""What bare `desman` shows: the banner, the state of the setup and the commands."""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from desman import __author__, __version__, providers, ui
from desman.config import ConfigManager
from desman.i18n import t, tn

BANNER = """████░░█████░█████░█░░░█░█████░█░░░█
█░░░█░█░░░░░█░░░░░██░██░█░░░█░██░░█
█░░░█░█████░█████░█░█░█░█████░█░█░█
█░░░█░█░░░░░░░░░█░█░░░█░█░░░█░█░░██
████░░█████░█████░█░░░█░█░░░█░█░░░█"""


def _commands() -> list[tuple[str, str]]:
    return [
        (t('desman run "GOAL"'), t("plan and run a task")),
        ("desman list / status", t("tasks and their progress")),
        ("desman inbox", t("what waits for your decision")),
        ("desman keys", t("API keys per provider")),
        ("desman config", t("providers, keys, permissions")),
        ("desman help", t("all commands")),
    ]


def print_project_info(config_manager: ConfigManager) -> None:
    config = config_manager.load()

    ui.out(Text(BANNER, style="accent"))
    ui.out(Text(t("Decentralized Self-Evolving Modular Agents Network"), style="bold"))
    ui.out(Text(f"v{__version__} · {__author__}; AI", style="muted"))
    ui.out("")

    ready = providers.ready(config)
    status = (
        Text("● ", style="good") + Text(tn(len(ready), "{n} provider ready", "{n} providers ready"))
        if ready
        else Text("● ", style="warn")
        + Text(t("no provider ready: run `desman config`"), style="warn")
    )
    if ready:
        status.append("  " + ", ".join(p.name for p in ready), style="muted")

    commands = Table.grid(padding=(0, 3))
    commands.add_column(style="key", no_wrap=True)
    commands.add_column(style="muted")
    for cmd, what in _commands():
        commands.add_row(cmd, what)

    ui.out(
        Panel(
            Group(status, Text(""), commands),
            border_style="accent",
            padding=(1, 2),
            expand=False,
        )
    )
    ui.note(t("config: {path}", path=config_manager.config_path))
