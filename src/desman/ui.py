"""Terminal look and prompts, shared by every command.

Output goes through rich. Prompts use arrow-key menus (questionary) in a real terminal and
fall back to numbered prompts otherwise, so scripts, pipes and tests can still drive them by
typing numbers. A cancelled prompt (Ctrl+C, Esc) raises `Cancelled`.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import click
from rich.cells import cell_len
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


def warn(text: str, *, stderr: bool = False) -> None:
    """`stderr` keeps stdout clean for machine output (`--json`)."""
    console(stderr).print(Text("! ", style="bold warn") + Text(text, style="warn"))


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


@dataclass(frozen=True)
class Heading:
    """A section title between choices; it cannot be picked and takes no number."""

    title: str


def select(question: str, items: Sequence[Choice | Heading], default: Any = None) -> Any:
    """One of the choices among `items` by value."""
    choices = [c for c in items if isinstance(c, Choice)]
    if not choices:
        raise ValueError("nothing to choose from")
    width = max(cell_len(c.label) for c in choices)  # hints line up in a column
    if interactive():
        import questionary

        q_items: list[Any] = []
        for item in items:
            if isinstance(item, Heading):
                q_items.append(questionary.Separator(f"── {item.title}"))
                continue
            q_items.append(
                questionary.Choice(
                    title=[("", _pad(item.label, width))]
                    + ([("class:hint", f"  {item.hint}")] if item.hint else []),
                    value=sum(
                        isinstance(q, questionary.Choice)
                        and not isinstance(q, questionary.Separator)
                        for q in q_items
                    ),
                )
            )
        start = next((i for i, c in enumerate(choices) if c.value == default), 0)
        answer = questionary.select(
            question,
            choices=q_items,
            default=next(q for q in q_items if getattr(q, "value", None) == start),
            qmark="›",
            instruction=t("(↑/↓, Enter)"),
            style=_style(),
        ).ask()
        if answer is None:
            raise Cancelled
        return choices[answer].value

    out(Text(question, style="bold"))
    number = 0
    for item in items:
        if isinstance(item, Heading):
            out(Text(f"  ── {item.title}", style="muted"))
            continue
        number += 1
        line = Text(f"  {number:>2}  ", style="key") + Text(_pad(item.label, width))
        if item.hint:
            line += Text(f"  {item.hint}", style="muted")
        out(line)
    start = next((i for i, c in enumerate(choices, 1) if c.value == default), 1)
    while True:
        raw = click.prompt("  ›", default=str(start), show_default=False).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1].value
        error(t("enter a number from 1 to {n}", n=len(choices)))


def _pad(text: str, width: int) -> str:
    """`text` padded to `width` terminal cells (CJK characters take two)."""
    return text + " " * max(0, width - cell_len(text))


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

        answer = questionary.confirm(
            question, default=default, qmark="›", instruction=t("(Y/N)") + " ", style=_style()
        ).ask()
        if answer is None:
            raise Cancelled
        return bool(answer)
    while True:  # Enter keeps the default
        raw = click.prompt(f"{question} {t('(Y/N)')}", default="", show_default=False)
        raw = _fold(raw.strip())
        if not raw:
            return default
        if raw in _answers("y", "yes", t("yes")):
            return True
        if raw in _answers("n", "no", t("no")):
            return False
        error(t("enter y or n"))


def _answers(*words: str) -> set[str]:
    """Each word and its first letter, folded: y, yes, and the language's own (ja, j)."""
    return {form for word in words for form in (_fold(word), _fold(word)[:1]) if form}


def _fold(text: str) -> str:
    """Case-insensitive, and blind to the dot on Turkish i: HAYIR lowers to "hayir", while
    the catalog writes "hayır"."""
    return text.casefold().replace("i\u0307", "i").replace("ı", "i")


def text(question: str, default: str = "") -> str:
    if interactive():
        import questionary

        answer = questionary.text(question, default=default, qmark="›", style=_style()).ask()
        if answer is None:
            raise Cancelled
        return str(answer).strip()
    return str(click.prompt(question, default=default, show_default=bool(default))).strip()


def line(commands: Mapping[str, str] | None = None) -> str | None:
    """A line typed in the chat; None when the person leaves (Ctrl+D, Ctrl+C, end of input).

    In a terminal the line is edited with prompt_toolkit (questionary's own base): ↑ for earlier
    lines, and `commands` (name → what it does) as suggestions: typing `/` lists them under
    the line, Tab completes the name. Otherwise a line is read from stdin and no prompt is
    printed: the chat then runs line by line, in pipes and in tests.
    """
    if interactive():
        try:
            return str(_chat_prompt().prompt(completer=_completer(commands or {})))
        except (EOFError, KeyboardInterrupt):
            out("")
            return None
    raw = sys.stdin.readline()
    return raw.rstrip("\r\n") if raw else None


_prompt: Any = None
"""The chat's prompt, made once so that ↑ reaches the lines typed earlier in this run."""


def _chat_prompt() -> Any:
    global _prompt
    if _prompt is None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.styles import Style

        _prompt = PromptSession(
            [("class:mark", "› ")],
            complete_while_typing=True,
            reserve_space_for_menu=6,
            style=Style.from_dict(
                {
                    "mark": f"fg:{ACCENT} bold",
                    "completion-menu": "bg:default",
                    "completion-menu.completion": f"fg:{ACCENT} bg:default",
                    "completion-menu.completion.current": f"fg:#000000 bg:{ACCENT} bold",
                    "completion-menu.meta.completion": f"fg:{MUTED} bg:default",
                    "completion-menu.meta.completion.current": f"fg:#000000 bg:{ACCENT}",
                }
            ),
        )
    return _prompt


def slash_matches(typed: str, commands: Mapping[str, str]) -> list[tuple[str, str]]:
    """The commands (name, what it does) a line typed so far may become: while it is `/` and
    the start of a name; nothing once it holds a space or is a message."""
    if not typed.startswith("/") or any(c.isspace() for c in typed):
        return []
    return [(name, what) for name, what in commands.items() if name.startswith(typed.lower())]


def _completer(commands: Mapping[str, str]) -> Any:
    from prompt_toolkit.completion import Completer, Completion

    class Slash(Completer):
        def get_completions(self, document: Any, event: Any) -> Iterator[Completion]:
            typed = document.text_before_cursor
            for name, what in slash_matches(typed, commands):
                yield Completion(name, -len(typed), display_meta=what)

    return Slash()


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
