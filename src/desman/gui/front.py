"""Thread-safe browser events and synchronous input for the front in ADR 0005."""

from __future__ import annotations

import _thread
import io
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from threading import Condition
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text

from desman import ui


class Answer(BaseModel):
    """Untrusted browser input; prompt-specific checks are made by the waiting front."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    id: int
    value: Any = None
    cancel: bool = False


class WebFront(ui.Front):
    """Adapt synchronous UI calls to browser state without moving authority to the page."""

    def __init__(self, limit: int = 5000) -> None:
        self.condition = Condition()
        self.events: deque[dict[str, Any]] = deque(maxlen=limit)
        self.seq = 0
        self.state = 0
        self.live: str | None = None
        self.prompt: dict[str, Any] | None = None
        self._answer: dict[str, Any] | None = None
        self._prompt_id = 0
        self.busy = False
        self.ended = False

    def emit(self, event_type: str, **data: Any) -> None:
        """Append retained display output and wake readers waiting for changes."""
        with self.condition:
            self.seq += 1
            self.state += 1
            self.events.append({"seq": self.seq, "type": event_type, **data})
            self.condition.notify_all()

    def poll(self, after: int, timeout: float = 25, state: int | None = None) -> dict[str, Any]:
        """Return retained events and current state so reloads preserve prompts and progress."""
        with self.condition:
            previous = self.state if state is None else state
            self.condition.wait_for(
                lambda: self.seq > after or self.state != previous or self.ended, timeout=timeout
            )
            return {
                "events": [dict(e) for e in self.events if e["seq"] > after],
                "prompt": self.prompt,
                "live": self.live,
                "state": self.state,
                "busy": self.busy,
                "ended": self.ended,
            }

    @staticmethod
    def render(*items: RenderableType | str) -> str:
        """Export escaped Rich HTML, leaving prose wrapping to the browser."""
        console = Console(
            record=True,
            file=io.StringIO(),
            theme=ui.THEME,
            width=100,
            force_terminal=True,
            color_system="truecolor",
        )
        console.print(*items, soft_wrap=all(isinstance(item, (str, Text)) for item in items))
        return console.export_html(inline_styles=True, code_format="{code}")

    def out(self, *items: RenderableType | str) -> None:
        """Publish Rich output so command bodies share the browser transcript."""
        self.emit("out", level="out", html=self.render(*items))

    def header(self, title: str, subtitle: str = "") -> None:
        """Render a heading panel with the shared terminal theme."""
        body = Text(title, style="bold accent")
        if subtitle:
            body.append("\n" + subtitle, style="muted")
        self.emit(
            "out",
            level="header",
            html=self.render(Panel(body, border_style="accent", padding=(0, 2), expand=False)),
        )

    def section(self, title: str) -> None:
        """Render a section divider to keep command output recognisable."""
        self.emit(
            "out",
            level="section",
            html=self.render(
                Text(f"── {title} ", style="bold accent")
                + Text("─" * max(0, 50 - len(title)), "muted")
            ),
        )

    def ok(self, text: str) -> None:
        """Publish a success message using the shared theme."""
        self.emit("out", level="ok", html=self.render(Text("✓ ", style="good") + Text(text)))

    def note(self, text: str) -> None:
        """Publish a muted note using the shared theme."""
        self.emit("out", level="note", html=self.render(Text(text, style="muted")))

    def warn(self, text: str, *, stderr: bool = False) -> None:
        """Publish a warning in the browser regardless of terminal stream."""
        self.emit("out", level="warn", html=self.render(Text("! " + text, style="warn")))

    def error(self, text: str) -> None:
        """Publish an error using the shared theme."""
        self.emit("out", level="error", html=self.render(Text("✗ " + text, style="bad")))

    def interactive(self) -> bool:
        """Enable human input because the browser provides prompt windows."""
        return True

    def _ask(self, kind: str, question: str = "", **data: Any) -> Any:
        with self.condition:
            self._prompt_id += 1
            self._answer = None
            self.prompt = {
                "type": "prompt",
                "id": self._prompt_id,
                "kind": kind,
                "question": question,
                **data,
            }
            self.emit("prompt", **{k: v for k, v in self.prompt.items() if k != "type"})
            try:
                # Timed waits let Ctrl+C reach the main thread on Windows too.
                while self._answer is None:
                    self.condition.wait(timeout=0.2)
                answer = self._answer
                assert answer is not None
                if answer.get("cancel") is True:
                    if kind == "line":
                        return None
                    raise ui.Cancelled
                return answer["value"]
            finally:
                self.prompt = None
                self._answer = None

    def answer(self, data: dict[str, Any]) -> bool:
        """Validate input against the open prompt before waking its caller."""
        try:
            answer = Answer.model_validate(data)
        except ValidationError:
            return False
        with self.condition:
            prompt = self.prompt
            if prompt is None or answer.id != prompt["id"] or self._answer is not None:
                return False
            value: Any = answer.value
            kind = prompt["kind"]
            count = sum("heading" not in item for item in prompt.get("items", []))

            def index(v: Any) -> bool:
                return type(v) is int and 0 <= v < count

            cancelled = answer.cancel
            valid = cancelled
            if kind == "select":
                valid |= index(value)
            elif kind == "checkbox":
                valid |= (
                    isinstance(value, list)
                    and all(index(v) for v in value)
                    and len(set(value)) == len(value)
                )
            elif kind == "confirm":
                valid |= type(value) is bool
            else:
                valid |= isinstance(value, str) and len(value) <= 20000
            if not valid:
                return False
            echo: Any = None if cancelled else value
            if kind == "secret":
                echo = "••••"
            elif not cancelled and kind in {"select", "checkbox"}:
                choices = [i["label"] for i in prompt["items"] if "heading" not in i]
                echo = choices[value] if kind == "select" else [choices[i] for i in value]
            self._answer = {"cancel": True} if cancelled else {"value": value}
            self.prompt = None
            self.emit("prompt_closed", id=prompt["id"], kind=kind, echo=echo)
            self.condition.notify_all()
            return True

    def select(
        self, question: str, items: Sequence[ui.Choice | ui.Heading], default: Any = None
    ) -> Any:
        """Wait for one choice and return its application value."""
        choices = [c for c in items if isinstance(c, ui.Choice)]
        if not choices:
            raise ValueError("nothing to choose from")
        value = self._ask(
            "select",
            question,
            items=[
                {"heading": c.title}
                if isinstance(c, ui.Heading)
                else {"label": c.label, "hint": c.hint}
                for c in items
            ],
            default=next((i for i, c in enumerate(choices) if c.value == default), 0),
        )
        return choices[value].value

    def checkbox(self, question: str, choices: Sequence[ui.Choice], checked: set[Any]) -> set[Any]:
        """Wait for checked choices and return their application values."""
        indexes = self._ask(
            "checkbox",
            question,
            items=[{"label": c.label, "hint": c.hint} for c in choices],
            default=[i for i, c in enumerate(choices) if c.value in checked],
        )
        return {choices[i].value for i in indexes}

    def confirm(self, question: str, default: bool = False) -> bool:
        """Wait for an explicit boolean decision from the person."""
        return bool(self._ask("confirm", question, default=default))

    def text(self, question: str, default: str = "") -> str:
        """Read trimmed text through a browser prompt window."""
        return str(self._ask("text", question, default=default)).strip()

    def secret(self, question: str) -> str:
        """Read a secret without retaining its value in browser events."""
        return str(self._ask("secret", question)).strip()

    def line(self, commands: Mapping[str, str] | None = None) -> str | None:
        """Wait for a chat line with the same command hints as the terminal."""
        value = self._ask("line", commands=dict(commands or {}))
        return None if value is None else str(value)

    def live_start(self, terminal: bool | None = None) -> bool:
        """Mark work busy so the page exposes its Stop control."""
        with self.condition:
            self.busy = True
            self.emit("busy", busy=True)
        return True

    def live_update(self, renderable: RenderableType) -> None:
        """Replace transient progress without evicting transcript events."""
        html = self.render(renderable)
        with self.condition:
            self.live = html
            self.state += 1
            self.condition.notify_all()

    def live_stop(self) -> None:
        """Clear transient progress and hide the Stop control."""
        with self.condition:
            self.busy = False
            self.live = None
            self.state += 1
            self.condition.notify_all()

    def stop(self) -> None:
        """Interrupt busy work on the main thread, where the chat handles cancellation."""
        with self.condition:
            if self.busy:
                _thread.interrupt_main()

    def end(self) -> None:
        """Mark the session finished and wake any waiting browser."""
        with self.condition:
            self.ended = True
            self.emit("end")

    @contextmanager
    def capture(self) -> Iterator[None]:
        """Route command stdout and stderr to the page, restoring both on exit."""
        stream = Output(self)
        with redirect_stdout(stream), redirect_stderr(stream):
            yield


class Output(io.TextIOBase):
    """Plain writes by command bodies never pass through Rich markup parsing."""

    def __init__(self, front: WebFront) -> None:
        self.front = front

    encoding = "utf-8"

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text stream requires str")
        if text:
            self.front.out(Text(text))
        return len(text)

    def flush(self) -> None:
        pass
