"""Terminal look and prompts, shared by every command.

Output goes through rich. Prompts use arrow-key menus (questionary) in a real terminal and
fall back to numbered prompts otherwise, so scripts, pipes and tests can still drive them by
typing numbers. A cancelled prompt (Ctrl+C, Esc) raises `Cancelled`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import click
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from desman.i18n import t

T = TypeVar("T")

ACCENT = "#00EEFF"
GOOD = "#00FF99"
BAD = "#FF4D4D"
WARN = "#FFB000"
MUTED = "#888888"

THEME = Theme(
    {
        "accent": ACCENT,
        "good": GOOD,
        "bad": BAD,
        "warn": WARN,
        "muted": MUTED,
        "key": "bold " + ACCENT,
    }
)


class Cancelled(Exception):
    """The user left a prompt (Ctrl+C or Esc)."""


def console(stderr: bool = False) -> Console:
    """A console on the current stdout/stderr (CliRunner swaps them per test)."""
    stream = sys.stderr if stderr else sys.stdout
    terminal = stream.isatty()
    width = None if terminal else int(os.environ.get("COLUMNS", "100"))
    return Console(
        file=stream,
        theme=THEME,
        highlight=False,
        width=width,
        force_terminal=None if terminal else False,
        soft_wrap=not terminal,  # piped output: one message, one line (for grep and scripts)
    )


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def out(*items: RenderableType | str) -> None:
    console().print(*items)


def header(title: str, subtitle: str = "") -> None:
    body = Text(title, style="bold accent")
    if subtitle:
        body.append("\n" + subtitle, style="muted")
    out(Panel(body, border_style="accent", padding=(0, 2), expand=False))


def section(title: str) -> None:
    out("")
    out(Text(f"── {title} ", style="bold accent") + Text("─" * max(0, 50 - len(title)), "muted"))


def note(text: str) -> None:
    out(Text(text, style="muted"))


def ok(text: str) -> None:
    out(Text("✓ ", style="good") + Text(text))


def warn(text: str) -> None:
    out(Text("! ", style="bold warn") + Text(text, style="warn"))


def error(text: str) -> None:
    console(stderr=True).print(Text("✗ ", style="bold bad") + Text(text, style="bad"))


def table(*columns: str, title: str | None = None) -> Table:
    t = Table(
        *columns,
        title=title,
        title_justify="left",
        title_style="bold accent",
        header_style="muted",
        border_style="muted",
        box=None,
        pad_edge=False,
        padding=(0, 2, 0, 0),
    )
    return t


# ── Prompts ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Choice:
    label: str
    value: Any
    hint: str = ""


def select(question: str, choices: Sequence[Choice], default: Any = None) -> Any:
    """One of `choices` by value."""
    if not choices:
        raise ValueError("nothing to choose from")
    width = max(len(c.label) for c in choices)  # hints line up in a column
    if interactive():
        import questionary

        q_choices = [
            questionary.Choice(
                title=[("", c.label.ljust(width))]
                + ([("class:hint", f"  {c.hint}")] if c.hint else []),
                value=i,
            )
            for i, c in enumerate(choices)
        ]
        start = next((i for i, c in enumerate(choices) if c.value == default), 0)
        answer = questionary.select(
            question,
            choices=q_choices,
            default=q_choices[start],
            qmark="›",
            instruction=t("(↑/↓, Enter)"),
            style=_style(),
        ).ask()
        if answer is None:
            raise Cancelled
        return choices[answer].value

    out(Text(question, style="bold"))
    for i, c in enumerate(choices, 1):
        line = Text(f"  {i:>2}  ", style="key") + Text(c.label.ljust(width))
        if c.hint:
            line += Text(f"  {c.hint}", style="muted")
        out(line)
    start = next((i for i, c in enumerate(choices, 1) if c.value == default), 1)
    while True:
        raw = click.prompt("  ›", default=str(start), show_default=False).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1].value
        error(t("enter a number from 1 to {n}", n=len(choices)))


def checkbox(question: str, choices: Sequence[Choice], checked: set[Any]) -> set[Any]:
    """The values the user ticks. Without a terminal: a yes/no question per choice."""
    if interactive():
        import questionary

        answer = questionary.checkbox(
            question,
            choices=[
                questionary.Choice(c.label, value=i, checked=c.value in checked)
                for i, c in enumerate(choices)
            ],
            qmark="›",
            instruction=t("(Space toggles, Enter confirms)"),
            style=_style(),
        ).ask()
        if answer is None:
            raise Cancelled
        return {choices[i].value for i in answer}

    out(Text(question, style="bold"))
    return {c.value for c in choices if confirm(f"  {c.label}", default=c.value in checked)}


def confirm(question: str, default: bool = False) -> bool:
    if interactive():
        import questionary

        answer = questionary.confirm(question, default=default, qmark="›", style=_style()).ask()
        if answer is None:
            raise Cancelled
        return bool(answer)
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        raw = click.prompt(f"{question} {hint}", default="", show_default=False).strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "д", "да"):
            return True
        if raw in ("n", "no", "н", "нет"):
            return False
        error(t("enter y or n"))


def text(question: str, default: str = "") -> str:
    if interactive():
        import questionary

        answer = questionary.text(question, default=default, qmark="›", style=_style()).ask()
        if answer is None:
            raise Cancelled
        return str(answer).strip()
    return str(click.prompt(question, default=default, show_default=bool(default))).strip()


def secret(question: str) -> str:
    """Hidden input; never echoed."""
    if interactive():
        import questionary

        answer = questionary.password(question, qmark="›", style=_style()).ask()
        if answer is None:
            raise Cancelled
        return str(answer).strip()
    return str(click.prompt(question, default="", show_default=False, hide_input=True)).strip()


def _style() -> Any:
    import questionary  # imports prompt_toolkit: only when a terminal prompt is shown

    return questionary.Style(
        [
            ("qmark", f"fg:{ACCENT} bold"),
            ("question", "bold"),
            ("pointer", f"fg:{ACCENT} bold"),
            ("highlighted", f"fg:{ACCENT} bold"),
            ("selected", f"fg:{GOOD}"),
            ("answer", f"fg:{GOOD} bold"),
            ("instruction", f"fg:{MUTED}"),
            ("hint", f"fg:{MUTED}"),
        ]
    )
