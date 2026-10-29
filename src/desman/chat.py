"""The chat that bare `desman` opens after the banner (ROADMAP phase C, ADR 0003).

It continues the session written to last. A line starting with `/` is one of the chat's own
commands (`/help` lists them); anything else is a message, saved in the session. The
interlocutor that answers messages comes in C.3; until then the chat says so.

Everything shown here goes through `ui`, so without a terminal the chat runs line by line and
its windows (`/sessions`) take numbers.
"""

from __future__ import annotations

from collections.abc import Callable

import click
from rich.text import Text

from desman import ui
from desman.core import db
from desman.core.sessions import Kind, Message, Role, Sessions
from desman.i18n import t, tn

SHOWN_ON_RETURN = 6
"""How many of the last messages are shown when a session is continued or switched to."""

LISTED_SESSIONS = 20


def _commands() -> dict[str, str]:
    return {
        "/help": t("the chat's commands"),
        "/new": t("start a new session"),
        "/sessions": t("earlier sessions; pick one to continue it"),
        "/exit": t("leave the chat (Ctrl+D does too)"),
    }


class Chat:
    def __init__(self, sessions: Sessions, read: Callable[[], str | None] = ui.line) -> None:
        self.sessions = sessions
        self.read = read
        latest = sessions.latest()
        self.session_id: str | None = latest.id if latest else None

    def run(self) -> None:
        self._greet()
        while (line := self.read()) is not None:
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not self._command(line):
                    return
                continue
            self._message(line)

    # ── Messages ─────────────────────────────────────────────────────────────

    def _message(self, text: str) -> None:
        self.session_id = self.sessions.add(self.session_id, Role.USER, text).session_id
        ui.note(
            t(
                "Saved in the session. Nobody answers here yet: the interlocutor comes in a "
                'later version. Meanwhile `desman run "GOAL"` runs a task.'
            )
        )

    def _greet(self) -> None:
        if self.session_id is None:
            ui.note(t("New session. /help lists the chat's commands."))
            return
        latest = self.sessions.latest()
        assert latest is not None
        ui.note(
            tn(
                latest.messages,
                "Continuing the session of {when} ({n} message). /new starts another, "
                "/help lists the chat's commands.",
                "Continuing the session of {when} ({n} messages). /new starts another, "
                "/help lists the chat's commands.",
                when=_when(latest.last_at),
            )
        )
        self._show_last()

    def _show_last(self) -> None:
        assert self.session_id is not None
        for message in self.sessions.messages(self.session_id, last=SHOWN_ON_RETURN):
            ui.out(_render(message))

    # ── The chat's own commands ──────────────────────────────────────────────

    def _command(self, line: str) -> bool:
        """Run a chat command; False when it ends the chat."""
        name = line.split()[0].lower()
        if name == "/exit":
            return False
        if name == "/help":
            self._help()
        elif name == "/new":
            self._new()
        elif name == "/sessions":
            self._choose_session()
        else:
            ui.error(t("unknown command {name}: /help lists the chat's commands", name=name))
        return True

    def _help(self) -> None:
        commands = ui.table("", "")
        commands.show_header = False
        for name, what in _commands().items():
            commands.add_row(Text(name, style="key"), Text(what, style="muted"))
        ui.out(commands)
        ui.note(t("Anything else you write is a message."))

    def _new(self) -> None:
        if self.session_id is None:
            ui.note(t("This session is new already."))
            return
        self.session_id = None
        ui.ok(t("New session: it starts with your next message."))

    def _choose_session(self) -> None:
        recent = self.sessions.recent(limit=LISTED_SESSIONS)
        if not recent:
            ui.note(t("No sessions yet: the first starts with your first message."))
            return
        choices = [
            ui.Choice(
                label=f"{_when(s.last_at)}  {_shorten(s.opening) or '…'}",
                value=s.id,
                hint=tn(s.messages, "{n} message", "{n} messages")
                + (" · " + t("current") if s.id == self.session_id else ""),
            )
            for s in recent
        ]
        if self.session_id is None:
            choices.insert(0, ui.Choice(label=t("the new session"), value=None, hint=t("current")))
        try:
            chosen = ui.select(t("Which session?"), choices, default=self.session_id)
        except (ui.Cancelled, click.Abort):
            return
        if chosen == self.session_id:
            return
        self.session_id = chosen
        if chosen is None:
            ui.ok(t("New session: it starts with your next message."))
            return
        last_at = next(s.last_at for s in recent if s.id == chosen)
        ui.ok(t("Switched to the session of {when}.", when=_when(last_at)))
        self._show_last()


def _render(message: Message) -> Text:
    if message.role is Role.USER:
        return Text("› ", style="key") + Text(message.text)
    if message.kind is not Kind.MESSAGE or message.role is Role.DESMAN:
        return Text(message.text, style="muted")
    return Text(message.text)


def _when(moment: str) -> str:
    """A database timestamp in the person's local time, to the minute."""
    return db.parse_iso(moment).astimezone().strftime("%Y-%m-%d %H:%M")


def _shorten(text: str, width: int = 50) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"
