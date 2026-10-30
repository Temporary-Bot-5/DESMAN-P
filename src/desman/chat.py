"""The chat that bare `desman` opens after the banner (ROADMAP phase C, ADR 0003).

It continues the session written to last. A line starting with `/` is one of the chat's own
commands (`/help` lists them); anything else is a message, saved in the session and answered by
the interlocutor (`desman.interlocutor`, C.3) with one turn: a reply, a question back, a task
or a command. A task is an ordinary task of the session: planned, run by the coordinator while
the chat waits (Ctrl+C stops it, `desman resume` continues), then retold. What waits for the
operator is shown with the same decisions as `desman inbox`.

A command is looked up in the catalog (`desman.commands`, C.5) and its arguments validated
there. A read runs at once; a change is shown in a window drawn from the command itself and
applied only on the person's "yes"; input (a key, a section of the wizard) is typed by the
person into a window. What it did goes back to the interlocutor as an event of the session,
and changes, declined ones too, into the operator's journal (`core.operator`).

Without a ready provider the chat still opens; the program, not a model, says what is missing.

Everything shown here goes through `ui`, so without a terminal the chat runs line by line and
its windows (`/sessions`) take numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import click
import httpx
from rich.text import Text

from desman import commands, providers, runs, ui
from desman.config import Config, ConfigError, open_config
from desman.core import calls, db, operator
from desman.core.context import CHARS_PER_TOKEN
from desman.core.policy import Access
from desman.core.report import TaskReport, inbox, task_report
from desman.core.search import client
from desman.core.sessions import Kind, Message, Role, SessionInfo, Sessions
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.i18n import t, tn
from desman.interlocutor import Clarify, Command, Interlocutor, NewTask, Reply, excerpt
from desman.live import Progress
from desman.llm import router
from desman.llm.base import ProviderError
from desman.llm.structured import StructuredOutputError
from desman.planner import plan

SHOWN_ON_RETURN = 6
"""How many of the last messages are shown when a session is continued or switched to."""

COMMAND_RESULT_TOKENS = 800
"""How much of a command's result the interlocutor gets: it has to fit its history budget."""

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
        "/rename": t("name this session: /rename NAME"),
        "/delete": t("delete this session from the list (after your yes)"),
        "/exit": t("leave the chat (Ctrl+D does too)"),
    }


class Chat:
    def __init__(self, sessions: Sessions, read: Callable[[], str | None] | None = None) -> None:
        self.sessions = sessions
        self.store = Store(sessions.conn, clock=sessions.clock)
        self.read = read or (lambda: ui.line(_commands()))
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
                    if not self._slash(command):
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
            with self._for(calls.Role.INTERLOCUTOR):
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
        else:
            self._command(turn, config, models, interlocutor)

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

    def _for(self, role: calls.Role) -> AbstractContextManager[None]:
        """The model calls inside are the session's own (not any task's)."""
        return calls.for_(calls.Purpose(role, session_id=self.session_id))

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

    # ── Commands ─────────────────────────────────────────────────────────────

    def _command(
        self, turn: Command, config: Config, models: runs.Models, interlocutor: Interlocutor
    ) -> None:
        """The command the interlocutor asked for, through the catalog; never as it was
        written. Only the person's answer in a window built here applies a change."""
        entry = commands.find(turn.name)
        if entry is None:
            name = providers.redact_keys(turn.name, config)[0]
            ui.error(t("DESMAN has no command {name}.", name=name))
            self._result(
                t("Asked for {name}: there is no such command.", name=name),
                {"error": "unknown command", "commands": [e.name for e in commands.CATALOG]},
                config,
            )
            return
        try:
            args = commands.parse(entry, turn.args, config)
        except commands.BadArgs as e:
            ui.error(f"{entry.name}: {e}")
            self._result(
                t("Asked for {name} with arguments that do not fit.", name=entry.name),
                {"error": str(e)},
                config,
            )
            return
        command = commands.shown(entry, args)
        kind = entry.kind_of(args)
        before = self._saved_config()
        context = commands.Context(
            self.store,
            open_config(),
            start_task=lambda title, goal: self._task(
                NewTask(type="task", title=title, goal=goal), config, models, interlocutor
            ),
            resume_task=lambda task_id: self._resume(task_id, config, models, interlocutor),
        )
        try:
            made = entry.handler(context, args)
            if isinstance(made, commands.Proposal):
                if not self._confirmed(made):
                    made = commands.Outcome({"declined": True}, done=False)
                else:
                    made = made.apply()
        except click.ClickException as e:
            ui.error(e.format_message())
            self._result(
                t("{command} did not run: {error}", command=command, error=e.format_message()),
                {"error": e.format_message()},
                config,
            )
            return
        except (ui.Cancelled, click.Abort):
            ui.out("")
            made = commands.Outcome({"declined": True}, done=False)
        if kind is not commands.Kind.READ:
            self._journal(entry, args, made.done, before)
        if not made.done:
            ui.note(t("Nothing changed."))
            self._result(t("{command}: the person declined.", command=command), None, config)
        elif made.data is not None:
            self._result(t("{command} ran.", command=command), made.data, config)

    def _confirmed(self, proposal: commands.Proposal) -> bool:
        """The window of a change: what it does, as the command worded it, and the person's
        answer. Nothing the model said is shown here or counts as an answer."""
        ui.section(t("Confirm"))
        for line in proposal.lines:
            ui.out(Text("  " + line))
        for warning in proposal.warnings:
            ui.warn(warning)
        return ui.confirm(proposal.question or t("Apply this change?"), default=False)

    def _journal(
        self, entry: commands.Entry, args: commands.Args, done: bool, before: Config | None
    ) -> None:
        """The operator's journal: the command, confirmed or declined, and what it changed in
        the config (a key by its fingerprint, a permission by its name)."""
        conn = self.store.conn
        details = {"command": entry.name, "args": args.model_dump(mode="json")}
        action = "operator_command" if done else "operator_declined"
        operator.record(conn, action, details, session_id=self.session_id)
        for change, about in commands.config_changes(before, self._saved_config()):
            operator.record(conn, change, about, session_id=self.session_id)

    def _saved_config(self) -> Config | None:
        """The config as saved (None: no file, or one that does not load)."""
        manager = open_config()
        if not manager.config_path.exists():
            return None
        try:
            return manager.load()
        except ConfigError:
            return None

    def _result(self, said: str, data: Any, config: Config | None) -> None:
        """What a command did, for the interlocutor's next turns: an event of the session,
        not printed (the person saw it already). Never a key: see `commands.for_interlocutor`."""
        assert self.session_id is not None
        text = said
        if data is not None:
            configs = [config, self._saved_config()]
            budget = COMMAND_RESULT_TOKENS * CHARS_PER_TOKEN
            text += "\n" + commands.for_interlocutor(data, configs, budget)
        self.sessions.add(self.session_id, Role.DESMAN, text, kind=Kind.EVENT)

    # ── Tasks ────────────────────────────────────────────────────────────────

    def _task(
        self, turn: NewTask, config: Config, models: runs.Models, interlocutor: Interlocutor
    ) -> None:
        access = self._access(config)
        if access is None:
            return
        ui.note(t("Planning…"))
        with self._for(calls.Role.PLANNER):  # the task does not exist yet: the session asked
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
        self._work(task_id, ref, config, models, access, interlocutor)

    def _resume(
        self, task_id: str, config: Config, models: runs.Models, interlocutor: Interlocutor
    ) -> None:
        access = self._access(config)
        if access is None:
            return
        ref = self.store.short_id(task_id)
        self._event(t("Task {ref} continues.", ref=ref), task_id)
        self._work(task_id, ref, config, models, access, interlocutor)

    def _access(self, config: Config) -> Access | None:
        try:
            return runs.executor_access(config)
        except click.ClickException as e:
            ui.error(e.message)
            return None

    def _work(
        self,
        task_id: str,
        ref: str,
        config: Config,
        models: runs.Models,
        access: Access,
        interlocutor: Interlocutor,
    ) -> None:
        """Run the task while the chat waits, then retell how it went."""
        try:
            if self._http is None:
                self._http = client()
            with (
                Progress(self.store, task_id, config) as progress,
                calls.watching(progress.refresh),
            ):
                runs.coordinator(
                    self.store,
                    models,
                    config=config,
                    access=access,
                    http=self._http,
                    watch=progress.refresh,
                ).run(task_id)
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
            with self._for(calls.Role.INTERLOCUTOR):
                retold = interlocutor.retell(outcome)
            self._say(retold, task_id)
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
            if message.kind is Kind.EVENT:  # a command's data or a result shown once is enough
                first = message.text.split("\n", 1)[0]
                message = message.model_copy(update={"text": first})
            ui.out(_render(message))

    # ── The chat's own commands ──────────────────────────────────────────────

    def _slash(self, line: str) -> bool:
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
        elif name == "/rename":
            self._rename(line.strip()[len(name) :].strip())
        elif name == "/delete":
            self._delete()
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

    def _current(self) -> SessionInfo | None:
        """The session written to, or None (with a note) while it has not started."""
        current = None if self.session_id is None else self.sessions.get(self.session_id)
        if current is None:
            ui.note(t("This session starts with your next message: there is nothing to it yet."))
        return current

    def _rename(self, title: str) -> None:
        current = self._current()
        if current is None:
            return
        if not title:
            try:
                title = ui.text(t("Name of this session ('-' gives back the first message)"))
            except (ui.Cancelled, click.Abort):
                return
            if not title:
                return
        title, cut = providers.redact_keys(title, self._config())
        if cut:
            ui.warn(t("That looked like an API key, so it was cut from the name."))
        new = None if title.strip() == "-" else title
        self.sessions.rename(current.id, new)
        renamed = self.sessions.get(current.id)
        assert renamed is not None
        operator.record(
            self.store.conn,
            "operator_session_renamed",
            {"title": renamed.title},
            session_id=current.id,
        )
        ui.ok(t("The session is called {name} now.", name=_shorten(renamed.name) or "…"))

    def _delete(self) -> None:
        current = self._current()
        if current is None:
            return
        ui.section(t("Confirm"))
        ui.out(
            Text(
                "  "
                + t(
                    "Delete the session of {when}: {name}",
                    when=_when(current.last_at),
                    name=_shorten(current.name) or "…",
                )
            )
        )
        ui.note(
            t(
                "It leaves the list and the chat will not continue it. Its messages stay in "
                "the database, with the history of its tasks."
            )
        )
        try:
            yes = ui.confirm(t("Delete it?"), default=False)
        except (ui.Cancelled, click.Abort):
            yes = False
        if not yes:
            ui.note(t("Nothing changed."))
            return
        self.sessions.delete(current.id)
        operator.record(self.store.conn, "operator_session_deleted", session_id=current.id)
        self.session_id = None
        ui.ok(t("Session deleted. A new one starts with your next message."))

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
                    label=f"{_when(s.last_at)}  {_shorten(s.name) or '…'}",
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
