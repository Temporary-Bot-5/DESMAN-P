"""Output and prompts shared by every command, with a swappable Front.

Output goes through rich. Prompts use arrow-key menus (questionary) in a real terminal and
fall back to numbered prompts otherwise, so scripts, pipes and tests can still drive them by
typing numbers. A cancelled prompt (Ctrl+C, Esc) raises `Cancelled`.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, TypeVar

import click
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
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
        # Rich cannot parse a theme name inside a style string ("bold accent" would be lost,
        # silently), so the bold forms used across the code are names of their own.
        "bold accent": "bold " + ACCENT,
        "bold warn": "bold " + WARN,
        "bold bad": "bold " + BAD,
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


def _terminal_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _terminal_out(*items: RenderableType | str) -> None:
    console().print(*items)


def _terminal_header(title: str, subtitle: str = "") -> None:
    body = Text(title, style="bold accent")
    if subtitle:
        body.append("\n" + subtitle, style="muted")
    out(Panel(body, border_style="accent", padding=(0, 2), expand=False))


def _terminal_section(title: str) -> None:
    out("")
    out(Text(f"── {title} ", style="bold accent") + Text("─" * max(0, 50 - len(title)), "muted"))


def _terminal_note(text: str) -> None:
    out(Text(text, style="muted"))


def _terminal_ok(text: str) -> None:
    out(Text("✓ ", style="good") + Text(text))


def _terminal_warn(text: str, *, stderr: bool = False) -> None:
    """`stderr` keeps stdout clean for machine output (`--json`)."""
    console(stderr).print(Text("! ", style="bold warn") + Text(text, style="warn"))


def _terminal_error(text: str) -> None:
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


def _terminal_select(question: str, items: Sequence[Choice | Heading], default: Any = None) -> Any:
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


def _terminal_checkbox(question: str, choices: Sequence[Choice], checked: set[Any]) -> set[Any]:
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


def _terminal_confirm(question: str, default: bool = False) -> bool:
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


def _terminal_text(question: str, default: str = "") -> str:
    if interactive():
        import questionary

        answer = questionary.text(question, default=default, qmark="›", style=_style()).ask()
        if answer is None:
            raise Cancelled
        return str(answer).strip()
    return str(click.prompt(question, default=default, show_default=bool(default))).strip()


def _terminal_line(commands: Mapping[str, str] | None = None) -> str | None:
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
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.styles import Style

        class RedactedHistory(InMemoryHistory):
            def append_string(self, string: str) -> None:
                from desman.config import ConfigError, open_config
                from desman.providers import redact_keys

                try:
                    config = open_config().load()
                except ConfigError:
                    config = None
                super().append_string(redact_keys(string, config)[0])

        _prompt = PromptSession(
            [("class:mark", "› ")],
            history=RedactedHistory(),
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
    """Match slash names and explicit CLI paths; only ! paths may cross spaces."""
    if not typed.startswith(("/", "!")) or (
        typed.startswith("/") and any(c.isspace() for c in typed)
    ):
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


def _terminal_secret(question: str) -> str:
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


def clean(text: str) -> str:
    """Treat even a compromised registry's output as plain terminal text."""
    text = re.sub(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-_])", "", text)
    return "".join(
        "�"
        if (unicodedata.category(c) == "Cc" and c not in "\n\t")
        or "\u202a" <= c <= "\u202e"
        or "\u2066" <= c <= "\u2069"
        or c in "\u200e\u200f\u061c"
        else c
        for c in text
    )


class Action(BaseModel):
    """A panel action answered by the person, consumed only by the chat thread."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class SessionAction(Action):
    type: Literal["open_session", "pin", "unpin", "delete_session"]
    id: str = Field(min_length=1, max_length=128)


class NewSessionAction(Action):
    type: Literal["new_session"]
    temporary: bool


class SimpleAction(Action):
    type: Literal["delete_all", "settings", "messenger", "terminal"]


class DecideAction(Action):
    type: Literal["decide"]
    ref: str = Field(min_length=1, max_length=128)
    version: int = Field(ge=0)
    decision: Literal["retry", "accept", "fail"]
    note: str = Field(max_length=20000)

    @model_validator(mode="after")
    def required_note(self) -> DecideAction:
        if self.decision == "fail" and not self.note.strip():
            raise ValueError("fail needs a note")
        return self


ACTIONS = TypeAdapter(
    Annotated[
        SessionAction | NewSessionAction | SimpleAction | DecideAction, Field(discriminator="type")
    ]
)


class Front(Protocol):
    """Define shared UI operations so chat and commands can use either front."""

    def banner(self, value: str) -> str: ...
    def reset(self) -> None: ...
    def greeting(self) -> AbstractContextManager[None]: ...
    def session(self, id: str | None, temporary: bool) -> None: ...
    def terminal_seen(self, seq: int) -> None: ...
    def transcript_limit(self) -> int: ...
    def interactive(self) -> bool: ...
    def out(self, *items: RenderableType | str) -> None: ...
    def header(self, title: str, subtitle: str = "") -> None: ...
    def section(self, title: str) -> None: ...
    def note(self, text: str) -> None: ...
    def ok(self, text: str) -> None: ...
    def warn(self, text: str, *, stderr: bool = False) -> None: ...
    def error(self, text: str) -> None: ...
    def select(
        self, question: str, items: Sequence[Choice | Heading], default: Any = None
    ) -> Any: ...
    def checkbox(self, question: str, choices: Sequence[Choice], checked: set[Any]) -> set[Any]: ...
    def confirm(self, question: str, default: bool = False) -> bool: ...
    def text(self, question: str, default: str = "") -> str: ...
    def line(self, commands: Mapping[str, str] | None = None) -> str | Action | None: ...
    def secret(self, question: str) -> str: ...
    def live_start(self, terminal: bool | None = None) -> bool: ...
    def live_update(self, renderable: RenderableType) -> None: ...
    def live_stop(self) -> None: ...


class TerminalFront:
    """Keep terminal rendering and prompts behind the shared front contract."""

    def __init__(self) -> None:
        self._live: Live | None = None

    def banner(self, value: str) -> str:
        return value

    def reset(self) -> None:
        pass

    @contextmanager
    def greeting(self) -> Iterator[None]:
        yield

    def session(self, id: str | None, temporary: bool) -> None:
        pass

    def terminal_seen(self, seq: int) -> None:
        pass

    def transcript_limit(self) -> int:
        return 6

    def interactive(self) -> bool:
        return _terminal_interactive()

    def out(self, *items: RenderableType | str) -> None:
        return _terminal_out(*items)

    def header(self, title: str, subtitle: str = "") -> None:
        return _terminal_header(title, subtitle)

    def section(self, title: str) -> None:
        return _terminal_section(title)

    def note(self, text: str) -> None:
        return _terminal_note(text)

    def ok(self, text: str) -> None:
        return _terminal_ok(text)

    def warn(self, text: str, *, stderr: bool = False) -> None:
        return _terminal_warn(text, stderr=stderr)

    def error(self, text: str) -> None:
        return _terminal_error(text)

    def select(self, question: str, items: Sequence[Choice | Heading], default: Any = None) -> Any:
        return _terminal_select(question, items, default)

    def checkbox(self, question: str, choices: Sequence[Choice], checked: set[Any]) -> set[Any]:
        return _terminal_checkbox(question, choices, checked)

    def confirm(self, question: str, default: bool = False) -> bool:
        return _terminal_confirm(question, default)

    def text(self, question: str, default: str = "") -> str:
        return _terminal_text(question, default)

    def line(self, commands: Mapping[str, str] | None = None) -> str | None:
        return _terminal_line(commands)

    def secret(self, question: str) -> str:
        return _terminal_secret(question)

    def live_start(self, terminal: bool | None = None) -> bool:
        enabled = console().is_terminal if terminal is None else terminal
        if enabled:
            self._live = Live(console=console(), auto_refresh=False, transient=True)
            self._live.start()
        return enabled

    def live_update(self, renderable: RenderableType) -> None:
        if self._live is not None:
            self._live.update(renderable, refresh=True)

    def live_stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


_front: Front = TerminalFront()


@contextmanager
def use(front: Front) -> Iterator[None]:
    """Select a front for synchronous UI calls and restore it even after cancellation."""
    global _front
    previous, _front = _front, front
    try:
        yield
    finally:
        _front = previous


@contextmanager
def _ui_writes() -> Iterator[None]:
    global _ui_depth
    stdout = sys.stdout.target if isinstance(sys.stdout, _Tee) else sys.stdout
    stderr = sys.stderr.target if isinstance(sys.stderr, _Tee) else sys.stderr
    _ui_depth += 1
    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield
    finally:
        _ui_depth -= 1


_ui_depth = 0


@dataclass
class Recording:
    buffer: io.StringIO
    redact: Callable[[str], str]

    @property
    def text(self) -> str:
        return Text.from_ansi(self.buffer.getvalue()).plain


_recordings: list[Recording] = []


def _record(*items: RenderableType | str) -> None:
    if _ui_depth:
        return
    for recording in _recordings:
        _RecordingConsole(recording).print(*items)


class _RecordingConsole(Console):
    """Redact Rich text before layout can wrap a key across lines or table borders."""

    def __init__(self, recording: Recording) -> None:
        super().__init__(file=recording.buffer, width=100, color_system=None, theme=THEME)
        self.recording = recording

    def render(
        self, renderable: RenderableType, options: ConsoleOptions | None = None
    ) -> Iterable[Segment]:
        if isinstance(renderable, str):
            renderable = Text.from_markup(renderable)
        if isinstance(renderable, Text):
            redacted = self.recording.redact(renderable.plain)
            if redacted != renderable.plain:
                renderable = Text(redacted, style=renderable.style)
        yield from super().render(renderable, options)


class _Tee(io.TextIOBase):
    def __init__(self, target: Any, recording: Recording) -> None:
        self.target = target
        self.recording = recording

    encoding = "utf-8"

    def isatty(self) -> bool:
        return bool(self.target.isatty())

    def fileno(self) -> int:
        return int(self.target.fileno())

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self.recording.buffer.write(value)
        return int(self.target.write(value))

    def flush(self) -> None:
        self.target.flush()


@contextmanager
def recording(
    *, show: bool = True, redact: Callable[[str], str] | None = None
) -> Iterator[Recording]:
    """Tee output into colourless text, optionally keeping notices off the active front."""
    rec = Recording(io.StringIO(), redact or (lambda value: value))
    _recordings.append(rec)
    previous = _front
    try:
        if not show:
            # A quiet terminal front still uses the same UI rendering paths.
            with (
                redirect_stdout(_Tee(io.StringIO(), rec)),
                redirect_stderr(_Tee(io.StringIO(), rec)),
                use(TerminalFront()),
            ):
                yield rec
        else:
            with redirect_stdout(_Tee(sys.stdout, rec)), redirect_stderr(_Tee(sys.stderr, rec)):
                yield rec
    finally:
        _recordings.remove(rec)
        assert _front is previous


def live_start(terminal: bool | None = None) -> bool:
    return _front.live_start(terminal)


def live_update(renderable: RenderableType) -> None:
    _front.live_update(renderable)


def live_stop() -> None:
    _front.live_stop()


def interactive() -> bool:
    return _front.interactive()


def out(*items: RenderableType | str) -> None:
    _record(*items)
    with _ui_writes():
        _front.out(*items)


def header(title: str, subtitle: str = "") -> None:
    _record(
        Panel(Text(title + ("\n" + subtitle if subtitle else "")), padding=(0, 2), expand=False)
    )
    with _ui_writes():
        _front.header(title, subtitle)


def section(title: str) -> None:
    _record(Text(f"── {title} " + "─" * max(0, 50 - len(title))))
    with _ui_writes():
        _front.section(title)


def note(text: str) -> None:
    _record(Text(text))
    with _ui_writes():
        _front.note(text)


def ok(text: str) -> None:
    _record(Text("✓ " + text))
    with _ui_writes():
        _front.ok(text)


def warn(text: str, *, stderr: bool = False) -> None:
    _record(Text("! " + text))
    with _ui_writes():
        _front.warn(text, stderr=stderr)


def error(text: str) -> None:
    _record(Text("✗ " + text))
    with _ui_writes():
        _front.error(text)


def select(question: str, items: Sequence[Choice | Heading], default: Any = None) -> Any:
    return _front.select(question, items, default)


def checkbox(question: str, choices: Sequence[Choice], checked: set[Any]) -> set[Any]:
    return _front.checkbox(question, choices, checked)


def confirm(question: str, default: bool = False) -> bool:
    return _front.confirm(question, default)


def text(question: str, default: str = "") -> str:
    return _front.text(question, default)


def line(commands: Mapping[str, str] | None = None) -> str | Action | None:
    return _front.line(commands)


def secret(question: str) -> str:
    return _front.secret(question)


def banner(value: str) -> str:
    return _front.banner(value)


def reset() -> None:
    _front.reset()


@contextmanager
def greeting() -> Iterator[None]:
    with _front.greeting():
        yield


def session(id: str | None, temporary: bool) -> None:
    _front.session(id, temporary)


def transcript_limit() -> int:
    return _front.transcript_limit()


def terminal_seen(seq: int) -> None:
    _front.terminal_seen(seq)
