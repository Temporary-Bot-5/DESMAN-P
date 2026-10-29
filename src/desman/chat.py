"""The chat that bare `desman` opens after the banner (ROADMAP phase C, ADR 0003).

It continues the session written to last. A line starting with `/` is one of the chat's own
commands (`/help` lists them); anything else is a message, saved in the session and answered by
the interlocutor (`desman.interlocutor`, C.3) with one turn: a reply, a question back, a task
or a command. A task is an ordinary task of the session: planned, run by the coordinator while
the chat waits (Ctrl+C stops it, `desman resume` continues), then retold. What waits for the
operator is shown with the same decisions as `desman inbox`. Commands come with their catalog
(C.5); until then the chat says so.

Without a ready provider the chat still opens; the program, not a model, says what is missing.

Everything shown here goes through `ui`, so without a terminal the chat runs line by line and
its windows (`/sessions`) take numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import click
import httpx
from rich.text import Text

from desman import providers, runs, ui
from desman.config import Config, ConfigError, open_config
from desman.core import db
from desman.core.report import TaskReport, inbox, task_report
from desman.core.search import client
from desman.core.sessions import Kind, Message, Role, Sessions
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.i18n import t, tn
from desman.interlocutor import Clarify, Interlocutor, NewTask, Reply, excerpt
from desman.llm import router
from desman.llm.base import ProviderError
from desman.llm.structured import StructuredOutputError
from desman.planner import plan

SHOWN_ON_RETURN = 6
"""How many of the last messages are shown when a session is continued or switched to."""

HISTORY_MESSAGES = 200
"""The most messages read for the interlocutor; it keeps the newest that fit its budget."""

LISTED_SESSIONS = 20
"""Sessions per page of `/sessions`; the last choice of a full page leads to earlier ones."""

_EARLIER = "earlier"  # not a session id: those are uuid hex


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
        self.store = Store(sessions.conn, clock=sessions.clock)
        self.read = read
        self.models: runs.Models | None = None
        """Built at the first message that needs them, again when the config changes."""
        self._models_for: Config | None = None
        self._http: httpx.Client | None = None
        """The web tools' client, shared by the session's tasks and closed with the chat."""
        latest = sessions.latest()
        self.session_id: str | None = latest.id if latest else None

    def run(self) -> None:
        self._greet()
        try:
            while (line := self.read()) is not None:
                command = line.strip()
                if not command:
                    continue
                if command.startswith("/"):
                    if not self._command(command):
                        return
                    continue
                self._message(line)  # kept as typed: indentation may matter (code, a list)
        finally:
            if self._http is not None:
                self._http.close()

    # ── Messages ─────────────────────────────────────────────────────────────

    def _message(self, text: str) -> None:
        config = self._config()
        text, cut = providers.redact_keys(text, config)
        if cut:  # before anything keeps it: the history, a model, a task's goal
            ui.warn(
                t(
                    "That looked like an API key, so it was cut from your message: keys are "
                    "never kept in the chat or sent to a model. Add a key with `desman keys "
                    "add PROVIDER`."
                )
            )
        models = None if config is None else self._models(config)
        if config is None or models is None:
            self.session_id = self.sessions.add(self.session_id, Role.USER, text).session_id
            return
        interlocutor = Interlocutor(models.llm, config.language)
        self._catch_up(interlocutor)  # so the model answers knowing how those tasks ended
        self.session_id = self.sessions.add(self.session_id, Role.USER, text).session_id
        conversation = self.sessions.messages(self.session_id, last=HISTORY_MESSAGES)
        try:
            turn = interlocutor.turn(conversation)
        except ProviderError as e:
            ui.error(t("The model did not answer ({error}). Your message is kept.", error=e))
            return
        except StructuredOutputError:
            ui.error(
                t(
                    "The model's answer made no sense, even when asked again. Your message is "
                    "kept: write again or put it differently."
                )
            )
            return
        if isinstance(turn, Reply):
            self._say(turn.text)
        elif isinstance(turn, Clarify):
            self._say(turn.question)
        elif isinstance(turn, NewTask):
            self._task(turn, config, models, interlocutor)
        else:  # the model's command is never run as it is: the catalog of commands decides (C.5)
            self._event(
                t(
                    "DESMAN commands cannot be run from the chat yet. In the terminal, "
                    "`desman --help` lists them."
                )
            )

    def _config(self) -> Config | None:
        try:
            return open_config().load()
        except ConfigError as e:
            ui.error(str(e))
            return None

    def _models(self, config: Config) -> runs.Models | None:
        """The models for `config`: built anew when it changed (a key replaced meanwhile)."""
        if self.models is None or config != self._models_for:
            self.models, self._models_for = None, None
            try:
                self.models = runs.models(config, self.sessions.conn)
                self._models_for = config
            except router.NoProvider:
                ui.warn(
                    t(
                        "Your message is kept, but no model is ready to answer it. Add a key "
                        "with `desman keys add PROVIDER` or set one up in `desman config`, "
                        "then write again."
                    )
                )
        return self.models

    def _say(self, text: str, task_id: str | None = None) -> None:
        assert self.session_id is not None
        message = self.sessions.add(self.session_id, Role.INTERLOCUTOR, text, task_id=task_id)
        ui.out(_render(message))

    def _event(self, text: str, task_id: str | None = None) -> None:
        assert self.session_id is not None
        message = self.sessions.add(
            self.session_id, Role.DESMAN, text, kind=Kind.EVENT, task_id=task_id
        )
        ui.out(_render(message))

    # ── Tasks ────────────────────────────────────────────────────────────────

    def _task(
        self, turn: NewTask, config: Config, models: runs.Models, interlocutor: Interlocutor
    ) -> None:
        try:
            access = runs.executor_access(config)
        except click.ClickException as e:
            ui.error(e.message)
            return
        ui.note(t("Planning…"))
        planned = plan(models.llm, turn.goal)
        if planned.error is not None:
            ui.warn(
                t(
                    "The model did not make a plan, so the goal is the only subtask: {error}",
                    error=planned.error,
                )
            )
        task_id = self.store.create_task(
            planned.spec,
            details={**planned.details(), "title": turn.title},
            session_id=self.session_id,
        )
        ref = self.store.short_id(task_id)
        self._event(
            tn(
                len(planned.spec.subtasks),
                "Task {ref} started: {title} ({n} subtask).",
                "Task {ref} started: {title} ({n} subtasks).",
                ref=ref,
                title=turn.title,
            ),
            task_id,
        )
        try:
            if self._http is None:
                self._http = client()
            runs.coordinator(self.store, models, config=config, access=access, http=self._http).run(
                task_id
            )
        except KeyboardInterrupt:
            self._event(
                t("Task {ref} stopped. `desman resume {ref}` continues it.", ref=ref), task_id
            )
            return
        self._outcome(task_id, ref, interlocutor)

    def _catch_up(self, interlocutor: Interlocutor) -> None:
        """Retell the session's tasks that changed while the chat was not watching them (a
        `desman resume` or a decision in the terminal), once they have settled."""
        if self.session_id is None:
            return
        for task_id in self.sessions.changed_tasks(self.session_id):
            report = task_report(self.store, task_id)
            if report.status is TaskStatus.ACTIVE and _paused_until(report) is None:
                continue  # still moving (another terminal), or stopped: told when it settles
            self._outcome(task_id, self.store.short_id(task_id), interlocutor)

    def _outcome(self, task_id: str, ref: str, interlocutor: Interlocutor) -> None:
        report = task_report(self.store, task_id)
        paused_until = _paused_until(report)
        waiting = [item for item in inbox(self.store) if item.task_id == task_id]
        if report.status is TaskStatus.DONE:
            self._event(t("Task {ref} is done.", ref=ref), task_id)
        elif report.status is TaskStatus.NEEDS_HUMAN:
            self._event(t("Task {ref} waits for you.", ref=ref), task_id)
        elif paused_until is not None:
            self._event(
                t(
                    "Task {ref} is paused by a provider limit until {until}. "
                    "`desman resume {ref}` continues it.",
                    ref=ref,
                    until=_when(paused_until),
                ),
                task_id,
            )
        else:
            self._event(t("Task {ref} is {status}.", ref=ref, status=report.status), task_id)
        for item in waiting:  # the same decisions as `desman inbox`
            ui.out(
                Text(f"{item.ref}  ", style="key")
                + Text(f"{item.status}  {_shorten(item.description, 60)}")
            )
            if item.reason:
                ui.out(Text(f"    {_shorten(item.reason, 110)}", style="muted"))
        if waiting:
            ui.note(
                t(
                    "Decide in the terminal: desman retry|accept|fail <ref> [--note TEXT], "
                    "then desman resume {ref}.",
                    ref=ref,
                )
            )
        result = self.store.final_result(task_id)
        outcome = {
            "goal": report.goal,
            "status": report.status.value,
            "result": None if result is None else excerpt(result),
            "waiting_for_the_person": [
                {"subtask": i.description, "status": i.status.value, "reason": i.reason}
                for i in waiting
            ],
        }
        try:
            self._say(interlocutor.retell(outcome), task_id)
        except (ProviderError, StructuredOutputError):
            if result is not None:  # the work is not lost with the retelling
                self._event(
                    t("The model could not retell the result; here it is as it is:")
                    + "\n"
                    + json.dumps(result, ensure_ascii=False, indent=2),
                    task_id,
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
        offset = 0
        while True:
            page = self.sessions.recent(limit=LISTED_SESSIONS + 1, offset=offset)
            if not page and offset == 0:
                ui.note(t("No sessions yet: the first starts with your first message."))
                return
            more, page = len(page) > LISTED_SESSIONS, page[:LISTED_SESSIONS]
            choices = [
                ui.Choice(
                    label=f"{_when(s.last_at)}  {_shorten(s.opening) or '…'}",
                    value=s.id,
                    hint=tn(s.messages, "{n} message", "{n} messages")
                    + (" · " + t("current") if s.id == self.session_id else ""),
                )
                for s in page
            ]
            if self.session_id is None and offset == 0:
                choices.insert(
                    0, ui.Choice(label=t("the new session"), value=None, hint=t("current"))
                )
            if more:  # every session stays reachable, a page at a time
                choices.append(ui.Choice(label=t("earlier sessions…"), value=_EARLIER))
            try:
                chosen = ui.select(t("Which session?"), choices, default=self.session_id)
            except (ui.Cancelled, click.Abort):
                return
            if chosen != _EARLIER:
                break
            offset += LISTED_SESSIONS
        if chosen == self.session_id:
            return
        self.session_id = chosen
        if chosen is None:
            ui.ok(t("New session: it starts with your next message."))
            return
        last_at = next(s.last_at for s in page if s.id == chosen)
        ui.ok(t("Switched to the session of {when}.", when=_when(last_at)))
        self._show_last()


def _render(message: Message) -> Text:
    if message.role is Role.USER:
        return Text("› ", style="key") + Text(message.text)
    if message.kind is not Kind.MESSAGE or message.role is Role.DESMAN:
        return Text(message.text, style="muted")
    return Text(message.text)


def _paused_until(report: TaskReport) -> str | None:
    """When the task can go on, if only paused work is left in it; otherwise None."""
    if report.status is not TaskStatus.ACTIVE:
        return None
    moving = {SubtaskStatus.READY, SubtaskStatus.RUNNING, SubtaskStatus.REVIEW}
    if any(s.status in moving for s in report.subtasks):
        return None
    pauses = [
        s.paused_until
        for s in report.subtasks
        if s.status is SubtaskStatus.PAUSED and s.paused_until is not None
    ]
    return min(pauses) if pauses else None


def _when(moment: str) -> str:
    """A database timestamp in the person's local time, to the minute."""
    return db.parse_iso(moment).astimezone().strftime("%Y-%m-%d %H:%M")


def _shorten(text: str, width: int = 50) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"
