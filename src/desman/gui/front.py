"""Thread-safe browser events and synchronous input for the front in ADR 0005."""

from __future__ import annotations

import _thread
import io
import json
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from html import escape
from pathlib import Path
from threading import Condition
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from desman.gui.runners import Runners

from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.text import Text

from desman import providers, ui
from desman.config import ConfigError, open_config
from desman.thread_output import redirect_stderr, redirect_stdout


class Answer(BaseModel):
    """Untrusted browser input; prompt-specific checks are made by the waiting front."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    id: int
    value: Any = None
    cancel: bool = False


class WebFront(ui.Front):
    """Adapt synchronous UI calls to browser state without moving authority to the page."""

    def __init__(
        self,
        limit: int = 5000,
        redact: Callable[[str], str] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runners: Runners | None = None
        self.database: Path | None = None
        self._changed: set[str] = set()
        self._transcript_cache: tuple[tuple[Any, ...], list[str]] | None = None
        self.clock = clock
        self.worker_count = 3
        self.idle: Callable[[], None] | None = None
        self.condition = Condition()
        self.events: deque[dict[str, Any]] = deque(maxlen=limit)
        self.seq = 0
        self.state = 0
        self.live: str | None = None
        self.prompt: dict[str, Any] | None = None
        self._answer: dict[str, Any] | None = None
        self._input_recordings: tuple[ui.Recording, ...] = ()
        self._prompt_id = 0
        self.busy = False
        self.redact: Callable[[str], str] = redact or redact_keys
        """What typed text is shown as: the event log is replayed to every reload, so a key
        pasted into a line never stays in it (the chat itself cuts it before keeping it)."""
        self._redact_html = redact or (lambda text: redact_keys(text, escaped=True))
        self.ended = False
        self.current: str | None = None
        self.temporary = False
        self.terminal_seq = 0
        self._greeting = False
        self._panel_quiet = False
        self.pending_choice: dict[str, str | None] | None = None
        self.settings_outcome: dict[str, Any] | None = None

    def session_changed(self, id: str) -> None:
        """Invalidate only a session; its transcript is read from SQLite by the page poll."""
        with self.condition:
            self._changed.add(id)
            self.state += 1
            self.condition.notify_all()

    def next_prompt_id(self) -> int:
        with self.condition:
            self._prompt_id += 1
            return self._prompt_id

    def pending_module(self, choice: dict[str, str | None] | None) -> None:
        with self.condition:
            self.pending_choice = choice

    @contextmanager
    def panel_output(self, quiet: bool) -> Iterator[None]:
        previous, self._panel_quiet = self._panel_quiet, quiet
        try:
            yield
        finally:
            self._panel_quiet = previous

    def settings_result(self, id: str, ok: bool, message: str) -> None:
        with self.condition:
            self.settings_outcome = {"id": id, "ok": ok, "message": self.redact(message)}

    def emit(self, event_type: str, **data: Any) -> None:
        """Append retained display output and wake readers waiting for changes."""
        if event_type == "out" and self._panel_quiet:
            return
        if event_type == "out" and isinstance(data.get("html"), str):
            data["html"] = self._redact_html(data["html"])
        with self.condition:
            self.seq += 1
            self.state += 1
            self.events.append(
                {
                    "seq": self.seq,
                    "type": event_type,
                    **({"greeting": True} if self._greeting and event_type == "out" else {}),
                    **data,
                }
            )
            self.condition.notify_all()

    def reset(self) -> None:
        with self.condition:
            self.events.clear()
            self.emit("reset")

    def clear_screen(self) -> None:
        self.reset()

    @contextmanager
    def greeting(self) -> Iterator[None]:
        previous, self._greeting = self._greeting, True
        try:
            yield
        finally:
            self._greeting = previous

    def session(self, id: str | None, temporary: bool) -> None:
        with self.condition:
            self.current, self.temporary = id, temporary
            if id is not None and self.runners is not None:
                self.session_changed(id)

    def terminal_seen(self, seq: int) -> None:
        with self.condition:
            self.terminal_seq = seq

    def banner(self, value: str) -> str:
        return value.replace("░", " ")

    def transcript_limit(self) -> int:
        return 200

    def action(self, action: ui.Action) -> bool:
        """Use the open line once; never run application code in a server thread."""
        with self.condition:
            if self.prompt is None or self.prompt["kind"] != "line" or self._answer is not None:
                return False
            prompt = self.prompt
            self._answer = {"value": action}
            self.prompt = None
            self.emit("prompt_closed", id=prompt["id"], kind="line", echo=None)
            self.condition.notify_all()
            return True

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
                "busy": self.busy
                if self.runners is None
                else bool(self.runners.status(self.current or "")["state"]),
                **self._session_poll(),
                "ended": self.ended,
            }

    def _session_poll(self) -> dict[str, Any]:
        if self.runners is None:
            return {}
        import sqlite3

        from desman.chat import _render
        from desman.core.sessions import Sessions

        mailbox = self.runners.mailbox(self.current)
        result: dict[str, Any] = {
            "pending": mailbox.prompt if mailbox else None,
            "turn": self.runners.status(self.current or ""),
        }
        if mailbox is not None:
            result["session_live"] = mailbox.live
        if self.current in self._changed and self.database is not None:
            conn = sqlite3.connect(self.database.resolve().as_uri() + "?mode=ro", uri=True)
            try:
                conn.row_factory = sqlite3.Row
                messages = Sessions(conn).screen(self.current, last=200)
                boundary = conn.execute(
                    "SELECT cleared_seq, terminal FROM sessions WHERE id = ?", (self.current,)
                ).fetchone()
                hidden = bool(boundary and boundary["cleared_seq"] and not boundary["terminal"])
                cancelled = tuple(m.seq for m in messages if m.cancelled_at is not None)
                from desman.i18n import language, t

                try:
                    config_stamp = open_config().config_path.stat().st_mtime_ns
                except OSError:
                    config_stamp = 0
                cache_key = (
                    self.current,
                    hidden,
                    cancelled,
                    messages[0].seq if messages else 0,
                    messages[-1].seq if messages else 0,
                    config_stamp,
                    language(),
                )
                if self._transcript_cache is None or self._transcript_cache[0] != cache_key:
                    self._transcript_cache = (
                        cache_key,
                        (
                            [
                                self.render(
                                    Text(
                                        t(
                                            "Earlier messages hidden by /clear (the model still sees them)."
                                        ),
                                        style="muted",
                                    )
                                )
                            ]
                            if hidden
                            else []
                        )
                        + [
                            self._redact_html(
                                self.render(
                                    styled
                                    if m.styled
                                    and (styled := Text.from_ansi(m.styled)).plain == m.text
                                    and self.redact(styled.plain) == styled.plain
                                    else _render(m)
                                )
                            )
                            for m in messages
                        ],
                    )
                result["transcript"] = self._transcript_cache[1]
                result["session_id"] = self.current
                result["transcript_version"] = (
                    f"{hidden}:{cancelled}:{self.current}:{messages[0].seq if messages else 0}:{messages[-1].seq if messages else 0}:{config_stamp}:{language()}"
                )
            finally:
                conn.close()
            # Every poll returns a current DB snapshot after a change, so reconnects/reloads
            # do not depend on consuming a transient notification exactly once.
        return result

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
        safe: list[RenderableType | str] = []
        for item in items:
            if isinstance(item, str):
                item = Text.from_markup(item)
            if isinstance(item, Text):
                redacted = self.redact(item.plain)
                if redacted != item.plain:
                    item = Text(redacted, style=item.style)
            safe.append(item)
        if self._panel_quiet:
            ui._record(*safe)
        self.emit(
            "out",
            level="out",
            html=self.render(*safe),
            context=any(isinstance(item, ui.ContextText) for item in items),
        )

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
        self.emit("out", level="ok", html=self.render(ui.notice_text("ok", text)))

    def note(self, text: str) -> None:
        """Publish a muted note using the shared theme."""
        self.emit("out", level="note", html=self.render(ui.notice_text("note", text)))

    def warn(self, text: str, *, stderr: bool = False) -> None:
        """Publish a warning in the browser regardless of terminal stream."""
        self.emit("out", level="warn", html=self.render(ui.notice_text("warn", text)))

    def error(self, text: str) -> None:
        """Publish an error using the shared theme."""
        self.emit("out", level="error", html=self.render(ui.notice_text("error", text)))

    def interactive(self) -> bool:
        """Enable human input because the browser provides prompt windows."""
        return True

    def _ask(self, kind: str, question: str = "", **data: Any) -> Any:
        with self.condition:
            prompt_id = self.next_prompt_id()
            self._answer = None
            self._input_recordings = ui._recordings.get()
            prompt = {
                "type": "prompt",
                "id": prompt_id,
                "kind": kind,
                "question": question,
                "terminal": self._panel_quiet,
                **data,
            }

            def public(value: Any) -> Any:
                if isinstance(value, str):
                    return self._redact_html(self.redact(value))
                if isinstance(value, dict):
                    return {k: public(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [public(v) for v in value]
                return value

            # Even code-built previews/defaults can contain a key copied into a setting.
            self.prompt = public(prompt)
            if self._panel_quiet and question:
                ui._record(Text(self.redact(question)))
            self.emit("prompt", **{k: v for k, v in self.prompt.items() if k != "type"})
            try:
                # Timed waits let Ctrl+C reach the main thread on Windows too.
                due = self.clock() + 2
                while self._answer is None:
                    if self.ended:
                        raise ui.Cancelled
                    self.condition.wait(
                        timeout=min(0.2, max(0, due - self.clock())) if kind == "line" else 0.2
                    )
                    if kind == "line" and self.idle is not None and self.clock() >= due:
                        self.condition.release()
                        try:
                            self.idle()
                        finally:
                            self.condition.acquire()
                            due = self.clock() + 2
                    elif self.clock() >= due:
                        due = self.clock() + 2
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
                self._input_recordings = ()

    def answer(self, data: dict[str, Any]) -> bool:
        """Validate input against the open prompt before waking its caller."""
        try:
            answer = Answer.model_validate(data)
        except ValidationError:
            return False
        if self.runners is not None:
            mailbox = self.runners.mailbox(self.current)
            if (
                mailbox is not None
                and mailbox.prompt is not None
                and mailbox.prompt["id"] == answer.id
            ):
                return mailbox.answer(data)
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
            elif isinstance(echo, str):  # a key pasted into the chat: cut as the chat cuts it
                echo = self.redact(echo)
            elif not cancelled and kind in {"select", "checkbox"}:
                choices = [i["label"] for i in prompt["items"] if "heading" not in i]
                echo = choices[value] if kind == "select" else [choices[i] for i in value]
            self._answer = {"cancel": True} if cancelled else {"value": value}
            self.prompt = None
            if prompt.get("terminal") and echo is not None:
                for recording in self._input_recordings:
                    ui._RecordingConsole(recording).print(Text(str(echo)))
                echo = None
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

    def line(self, commands: Mapping[str, str] | None = None) -> str | ui.Action | None:
        """Wait for a chat line with the same command hints as the terminal."""
        value = self._ask("line", commands=dict(commands or {}))
        return value if isinstance(value, ui.Action) else (None if value is None else str(value))

    def live_start(self, terminal: bool | None = None) -> bool:
        """Mark work busy so the page exposes its Stop control."""
        with self.condition:
            self.busy = True
            self.emit("busy", busy=True)
        return True

    def live_update(self, renderable: RenderableType) -> None:
        """Replace transient progress without evicting transcript events."""
        html = self._redact_html(self.render(renderable))
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
        if self.runners is not None:
            self.runners.stop(self.current)
            return
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


def redact_keys(text: str, *, escaped: bool = False) -> str:
    """`text` with the configured keys and anything shaped like a key cut out."""
    try:
        config = open_config().load()
    except (ConfigError, OSError):
        config = None  # the shapes of keys are still cut
    if config is not None:
        # Settings and previews may contain JSON-escaped copies; Rich uses html.escape.
        keys = {
            form
            for keys in config.api_keys.values()
            for k in keys
            for form in (
                [json.dumps(k.value)[1:-1], escape(k.value)]
                if escaped
                else [json.dumps(k.value)[1:-1]]
            )
        }
        for key in sorted(keys, key=len, reverse=True):
            if key:
                text = text.replace(key, providers.REDACTED)
    return providers.redact_keys(text, config)[0]


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
