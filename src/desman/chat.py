"""The chat that bare `desman` opens after the banner (ROADMAP phase C, ADR 0003).

It starts a new session (`desman -c` continues the last one, `/sessions` any earlier one;
`desman -t` opens a temporary chat, erased when it ends). A line starting with `/` is one of the chat's own
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

When the session's history no longer fits the interlocutor's budget, the older messages are
summarised first (`desman.compaction`, C.6) by a task of their own, and `/compact` does it at
once. A summary that fails leaves the older messages out of the model's view, with a note in
the chat; the database keeps them either way.

How the person set the interlocutor to talk (`Config.behaviour`) goes with every turn; if they
agreed to personalisation, so do the notes of the user's profile (C.7, `core.profile`), which
the interlocutor may change with any turn, each change shown as it is made. `/profile` shows
both: there the person changes how it talks, adds, changes or erases notes, and turns
personalisation on or off.

Without a ready provider the chat still opens; the program, not a model, says what is missing.

Everything shown here goes through `ui`, so without a terminal the chat runs line by line and
its windows (`/sessions`) take numbers.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
from typing import Any

import click
import httpx
from pydantic import ValidationError
from rich.text import Text

from desman import (
    actions,
    commands,
    compaction,
    config_cli,
    messenger_cli,
    modules_cli,
    providers,
    runs,
    task_files,
    tuning_cli,
    ui,
    wizard,
)
from desman.actions import suggestion_lines
from desman.config import Config, ConfigError, open_config
from desman.core import calls, choices, db, operator
from desman.core.context import CHARS_PER_TOKEN, estimate_tokens
from desman.core.policy import Access
from desman.core.profile import NOTE_CHARS, Author, Note, Profile, ProfileFull
from desman.core.report import InboxWatch, TaskReport, inbox, task_report
from desman.core.search import client
from desman.core.sessions import History, Kind, Message, Role, SessionInfo, Sessions
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import AmbiguousIdError, NotFoundError, Store
from desman.core.workspace import PathEscapeError, workspace_root_for
from desman.i18n import t, tn
from desman.interlocutor import (
    AddNote,
    ChangeNote,
    Clarify,
    Command,
    Interlocutor,
    NewTask,
    ProfileEdit,
    Reply,
    excerpt,
    unbacked_code,
)
from desman.live import Progress
from desman.llm import hub, router
from desman.llm.base import Completion, ProviderError, Request
from desman.llm.structured import StructuredOutputError
from desman.net import protocol
from desman.planner import must_run, plan

SHOWN_ON_RETURN = 6
"""How many of the last messages are shown when a session is continued or switched to."""

COMMAND_RESULT_TOKENS = 800
"""How much of a command's result the interlocutor gets: it has to fit its history budget."""

HISTORY_MESSAGES = 200
"""The most messages read for the interlocutor; it keeps the newest that fit its budget."""

LISTED_SESSIONS = 20
"""Sessions per page of `/sessions`; the last choice of a full page leads to earlier ones."""

_EARLIER = "earlier"  # not a session id: those are uuid hex


TERMINAL_COMMANDS = {
    "/help",
    "/clear",
    "/context",
    "/new",
    "/temporary",
    "/sessions",
    "/messages",
    "/back",
    "/exit",
    "/terminal",
}

KEPT_ONLY = ("/sessions", "/rename", "/compact", "/delete", "/pin")
"""The chat's commands about kept sessions: a temporary chat has none of them."""


def _commands(temporary: bool = False, terminal: bool = False) -> dict[str, str]:
    every = {
        "/terminal": t("open General Terminal"),
        "/back": t("return from General Terminal to the previous chat"),
        "/help": t("the chat's commands"),
        "/clear": t("clear the screen (history is kept)"),
        "/context": t("show the context of the next model call: /context [full]"),
        "/new": t("start a new session"),
        "/sessions": t("earlier sessions; pick one to continue it"),
        "/rename": t("name this session: /rename NAME"),
        "/compact": t("summarise this session's older messages now"),
        "/messages": t("read messages privately: /messages [NAME]"),
        "/profile": t("how the interlocutor talks to you and what it notes about you"),
        "/delete": t("delete this session from the list (after your yes)"),
        "/pin": t("pin or unpin this session"),
        "/delete-all": t("delete all unpinned sessions after confirmation"),
        "/temporary": t("start a temporary chat"),
        "/exit": t("leave the chat (Ctrl+D does too)"),
    }
    if terminal:
        return {name: what for name, what in every.items() if name in TERMINAL_COMMANDS}
    if temporary:
        every["/new"] = t("start a normal session (this temporary chat is erased)")
    return {name: what for name, what in every.items() if not (temporary and name in KEPT_ONLY)}


class Chat:
    def __init__(
        self,
        sessions: Sessions,
        read: Callable[[], str | ui.Action | None] | None = None,
        *,
        resume: bool = False,
        temporary: bool = False,
    ) -> None:
        if resume and temporary:
            raise ValueError("a temporary chat never continues a session")
        self.sessions = sessions
        self.temporary = temporary
        """Nothing said is kept (C.11): the session is erased when the chat ends; no profile
        notes, no summaries, no cached replies."""
        self._terminal_id = sessions.terminal()
        self._previous: tuple[str | None, bool] | None = None
        self._terminal_seen = 0
        self._pending_choice: choices.Choice | None = None
        self.store = Store(sessions.conn, clock=sessions.clock)
        self.profile = Profile(sessions.conn, clock=sessions.clock)
        self.read = read or (
            lambda: ui.line(
                {**_commands(self.temporary, self.in_terminal), **commands.completions()}
            )
        )
        self.models: runs.Models | None = None
        """Built at the first message that needs them, again when the config changes."""
        self._models_for: Config | None = None
        self._http: httpx.Client | None = None
        """The web tools' client, shared by the session's tasks and closed with the chat."""
        latest = sessions.latest() if resume else None
        self.session_id: str | None = latest.id if latest else None
        """A new session (it starts with the first message) unless asked to continue the last
        one (`desman -c`); /sessions continues any."""
        self.waiting = InboxWatch(self.store)
        """What waits for the person's decision: told at the start, then as it comes (5.1)."""
        self._gave_up: dict[str, int] = {}
        """Per session, the last message when a summary failed: until as much again is said,
        the older messages just fall out of the model's view."""
        self._told_blocked: str | None = None
        """The block of protected files last told (5.3): told again only when it changes."""

    def run(self) -> None:
        self.sessions.sweep()  # what a temporary chat that crashed left behind
        with ui.greeting():
            self._greet(show_messages=False)
        if self.session_id is not None:
            self._show_last()
        try:
            while True:
                self._notices()
                ui.session(self.session_id, self.temporary)
                if (line := self.read()) is None:
                    break
                if isinstance(line, ui.Action):
                    self._action(line)
                    continue
                command = line.strip()
                if not command:
                    continue
                if command.startswith("/"):
                    if self.in_terminal:
                        name = command.split()[0].lower()
                        if name in {
                            "/new",
                            "/temporary",
                            "/sessions",
                            "/back",
                            "/terminal",
                            "/exit",
                        }:
                            self.sessions.add(
                                self._terminal_id,
                                Role.USER,
                                providers.redact_keys(line, self._saved_config())[0],
                            )
                            keep_going = self._slash(command)
                        else:
                            keep_going = self._terminal_output(partial(self._slash, command), line)
                    else:
                        keep_going = self._slash(command)
                    if not keep_going:
                        return
                    continue
                if command.startswith("!") or self.in_terminal:
                    self._terminal_command(line)
                    continue
                self._message(line)  # kept as typed: indentation may matter (code, a list)
        finally:
            if self._http is not None:
                self._http.close()
            self._erase()
            if self._previous is not None and self._previous[1] and self._previous[0] is not None:
                self.sessions.erase(self._previous[0])

    @property
    def in_terminal(self) -> bool:
        return self.session_id == self._terminal_id

    def _terminal_open(self) -> None:
        if self.in_terminal:
            return
        self._previous = (self.session_id, self.temporary)
        self.session_id, self.temporary = self._terminal_id, False
        self.models, self._models_for = None, None
        ui.reset()
        ui.session(self.session_id, False)
        ui.note(t("→ General Terminal"))
        self._show_last()
        self._seen_terminal()

    def _seen_terminal(self) -> None:
        row = self.sessions.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ? AND kind = 'event'",
            (self._terminal_id,),
        ).fetchone()
        self._terminal_seen = int(row[0])
        ui.terminal_seen(self._terminal_seen)

    def _back(self) -> None:
        if not self.in_terminal:
            return
        previous, self._previous = self._previous, None
        if previous is None:
            self._new()
            return
        session_id, temporary = previous
        if session_id is not None and (
            (temporary and not self.sessions.messages(session_id))
            or (not temporary and self.sessions.get(session_id) is None)
        ):
            session_id, temporary = None, False
        self.session_id, self.temporary = session_id, temporary
        self.models, self._models_for = None, None
        ui.reset()
        ui.session(session_id, temporary)
        if session_id is not None:
            self._show_last()
        else:
            self._greet()

    def _terminal_output(
        self, operation: Callable[[], Any], line: str | None = None, *, keep_inputs: bool = False
    ) -> Any:
        config = self._saved_config()
        if line is not None:
            self.sessions.add(self._terminal_id, Role.USER, providers.redact_keys(line, config)[0])
        with (
            ui.panel_output(not self.in_terminal),
            ui.recording(
                show=self.in_terminal,
                redact=lambda text: providers.redact_keys(text, config)[0],
                keep_inputs=keep_inputs,
            ) as rec,
        ):
            try:
                return operation()
            finally:
                text = providers.redact_keys(rec.text, config)[0]
                text = providers.redact_keys(text, self._saved_config())[0]
                if text.strip():
                    self.sessions.add(self._terminal_id, Role.DESMAN, text, kind=Kind.EVENT)
                if self.in_terminal:
                    self._seen_terminal()

    def _terminal_command(self, line: str) -> None:
        if self.in_terminal and not line.strip().startswith(("/", "!")):
            self._terminal_output(lambda: ui.note(t("Commands start with /, e.g. /status")), line)
            return
        self._terminal_open()
        command = line.strip()
        self._terminal_output(
            lambda: self._typed(command[1:] if command.startswith(("!", "/")) else command), line
        )

    def _notices(self) -> None:
        with ui.recording(
            show=self.in_terminal,
            redact=lambda text: providers.redact_keys(text, self._saved_config())[0],
        ) as rec:
            self._tell_waiting()
            self._tell_blocked()
        if rec.text.strip():
            text = providers.redact_keys(rec.text, self._saved_config())[0]
            self.sessions.add(self._terminal_id, Role.DESMAN, text, kind=Kind.EVENT)
            if self.in_terminal:
                self._seen_terminal()
            else:
                count = self.sessions.conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ? AND kind = 'event' "
                    "AND seq > ? AND seq > (SELECT cleared_seq FROM sessions WHERE id = ?)",
                    (self._terminal_id, self._terminal_seen, self._terminal_id),
                ).fetchone()[0]
                ui.note(t("{n} new in General Terminal (/terminal)", n=count))

    def _erase(self) -> None:
        """A temporary chat's session, erased: when it ends or a new one starts."""
        if self.temporary and self.session_id is not None:
            self.sessions.erase(self.session_id)

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
        if config is not None and self.session_id is None and self._pending_choice is not None:
            config = choices.effective(config, "interlocutor", self._pending_choice)
        models = None if config is None else self._models(config)
        if config is None or models is None:
            self.session_id = self._add_user(text)
            return
        personal = config.personalize and not self.temporary  # a temporary chat notes nothing
        notes = self.profile.notes() if personal else None
        interlocutor = Interlocutor(
            models.for_(calls.Role.INTERLOCUTOR), config.language, notes, config.behaviour
        )
        self._catch_up(interlocutor)  # so the model answers knowing how those tasks ended
        if self.session_id is not None and not self.temporary:  # nothing kept, nothing summarised
            before = self.sessions.history(self.session_id, last=HISTORY_MESSAGES)
            after = self._gave_up.get(self.session_id, 0)
            incoming = estimate_tokens(text)
            if compaction.needed(before, after=after, extra=incoming):
                self._compact(before, models, incoming=incoming)
        self.session_id = self._add_user(text)
        history = self.sessions.history(self.session_id, last=HISTORY_MESSAGES)
        try:
            with self._for(calls.Role.INTERLOCUTOR):
                turn = interlocutor.turn(history)
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
        if personal:  # off: the model was told nothing of it, and nothing is kept
            self._note(turn.profile, config)
            interlocutor.profile = self.profile.notes()  # a task's retelling follows them
        if isinstance(turn, Reply):
            self._say(turn.text)
            self._fallback_note(interlocutor)
        elif isinstance(turn, Clarify):
            self._say(turn.question)
            self._fallback_note(interlocutor)
        elif isinstance(turn, NewTask):
            self._task(turn, config, models, interlocutor)
        else:
            self._command(turn, config, models, interlocutor)

    def _add_user(self, text: str) -> str:
        """The person's message, kept in the session (a new one with the first message)."""
        pending = self._pending_choice if self.session_id is None else None
        added = self.sessions.add(
            self.session_id, Role.USER, text, temporary=self.temporary, initial_choice=pending
        )
        if pending is not None:
            self._pending_choice = None
            ui.pending_module(None)
        return added.session_id

    def _config(self) -> Config | None:
        try:
            return open_config().load()
        except ConfigError as e:
            ui.error(str(e))
            return None

    def _models(self, config: Config, *, command: bool = False) -> runs.Models | None:
        """The models for `config`: built anew when it changed (a key replaced meanwhile)."""
        if self.models is None or config != self._models_for:
            self.models, self._models_for = None, None
            try:
                # A temporary chat's replies are not kept in the cache either.
                self.models = runs.models(config, self.sessions.conn, cache=not self.temporary)
                self._models_for = config
            except router.NoProvider:
                if command:
                    return None
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
        if self.session_id is None:
            ui.out(Text(text))
            return
        message = self.sessions.add(self.session_id, Role.INTERLOCUTOR, text, task_id=task_id)
        ui.out(_render(message))

    def _event(self, text: str, task_id: str | None = None) -> None:
        if self.session_id is None or self.in_terminal:
            ui.note(text)
            return
        message = self.sessions.add(
            self.session_id, Role.DESMAN, text, kind=Kind.EVENT, task_id=task_id
        )
        ui.out(_render(message))

    def _note(self, edits: list[ProfileEdit], config: Config) -> None:
        """The interlocutor's edits of the profile, each shown as it is made. One that does
        not apply (no such note, no room) is left out: the model sees the notes next time."""
        by = Author.INTERLOCUTOR
        for edit in edits:
            try:
                if isinstance(edit, AddNote):
                    text = providers.redact_keys(edit.add, config)[0]
                    if any(n.text.casefold() == text.casefold() for n in self.profile.notes()):
                        continue
                    note = self.profile.add(text, by, session_id=self.session_id)
                    ui.note(t("Noted about you: {note}", note=note.text))
                elif isinstance(edit, ChangeNote):
                    text = providers.redact_keys(edit.to, config)[0]
                    was = self.profile.get(edit.change)
                    changed = self.profile.change(edit.change, text, by, session_id=self.session_id)
                    if was is not None and changed is not None and changed.text != was.text:
                        ui.note(t("Changed a note about you: {note}", note=changed.text))
                else:
                    was = self.profile.get(edit.forget)
                    if was is not None and self.profile.forget(
                        edit.forget, by, session_id=self.session_id
                    ):
                        ui.note(t("Forgot a note about you: {note}", note=was.text))
            except ValueError:  # ProfileFull, or a text the key cut made empty
                continue

    # ── Commands ─────────────────────────────────────────────────────────────

    def _command(
        self,
        turn: Command,
        config: Config,
        models: runs.Models | None,
        interlocutor: Interlocutor | None,
        *,
        typed: bool = False,
        typed_entry: commands.Entry | None = None,
    ) -> None:
        """The command the interlocutor asked for, through the catalog; never as it was
        written. Model-requested changes need the person's answer in a code-built window; typed commands keep confirmations and previews required by the CLI."""
        entry = typed_entry if typed else commands.find(turn.name)
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
        if entry.name in ("run", "resume", "worker", "proposals analyze") and models is None:
            ui.error(t("DESMAN needs configuring before this command can use models."))
            return

        def start_task(title: str, goal: str, runs_code: bool) -> None:
            assert models is not None
            self._task(
                NewTask(type="task", title=title, goal=goal, runs_code=runs_code),
                config,
                models,
                interlocutor,
            )

        def resume_task(task_id: str) -> None:
            assert models is not None
            self._resume(task_id, config, models, interlocutor)

        context = commands.Context(
            self.store,
            open_config(),
            start_task=start_task,
            resume_task=resume_task,
            session_id=lambda: self.session_id,
            in_terminal=self.in_terminal,
        )
        try:
            made = entry.handler(context, args)
            if isinstance(made, commands.Proposal):
                if (
                    not typed
                    or entry.typed_confirm
                    or isinstance(args, commands.TaskPauseArgs)
                    and args.all
                    or isinstance(args, commands.TaskHideArgs)
                    and args.finished
                ) and not self._confirmed(made):
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
        if typed or kind is not commands.Kind.READ:
            self._journal(entry, args, made.done, before, typed=typed)
        if not made.done:
            ui.note(t("Nothing changed."))
            self._result(
                t("{command}: the person declined.", command=command),
                {
                    "language_switch_declined": True,
                    "instruction": "Do not propose another language switch in this session.",
                }
                if entry.name == "language"
                else None,
                config,
            )
        elif made.data is not None or typed:
            self._result(
                t("The person ran {command}.", command=command)
                if typed
                else t("{command} ran.", command=command),
                made.data,
                config,
            )

    def _typed(self, line: str) -> None:
        """A person's explicit command line never becomes a model turn or a user message."""
        if line.strip() == "clear":
            self.sessions.clear_terminal()
            if self.in_terminal:
                ui.clear_screen()
                self._seen_terminal()
            return
        if line.strip() == "help" and self.in_terminal:
            # The listing needs no saved settings: shown even when config.json is broken.
            entry = commands.find("help")
            assert entry is not None
            entry.handler(
                commands.Context(
                    self.store,
                    open_config(),
                    start_task=lambda *_: None,
                    resume_task=lambda _: None,
                    session_id=lambda: self.session_id,
                    in_terminal=True,
                ),
                commands.NoArgs(),
            )
            return
        config = self._config()
        if config is None:
            return
        word = line.split(maxsplit=1)[0] if line.split() else ""
        known = {entry.name.split()[0] for entry in commands.CATALOG} | {"desman"}
        if self.in_terminal and word and word not in known:
            ui.error(
                t(
                    "“{word}” is not a command. General Terminal takes only DESMAN's commands "
                    "(`help` lists them); to talk, go back to a chat with /back.",
                    word=providers.redact_keys(word, config)[0],
                ).replace("`help`", "`/help`")
            )
            return
        try:
            parsed = commands.parse_line(line, config)
        except (commands.BadArgs, click.ClickException, ValueError) as e:
            message = e.format_message() if isinstance(e, click.ClickException) else str(e)
            ui.error(providers.redact_keys(message, config)[0])
            return
        if parsed is None:
            return
        entry, args = parsed
        models = None
        interlocutor = None
        if entry.name in ("run", "resume", "worker", "proposals analyze"):
            models = self._models(config, command=True)
            if models is not None and not self.in_terminal:
                interlocutor = Interlocutor(
                    models.for_(calls.Role.INTERLOCUTOR),
                    config.language,
                    self.profile.notes() if config.personalize and not self.temporary else None,
                    config.behaviour,
                )
        self._command(
            Command(type="command", name=entry.name, args=args.model_dump(mode="json")),
            config,
            models,
            interlocutor,
            typed=True,
            typed_entry=entry,
        )

    def _confirmed(self, proposal: commands.Proposal, *, in_prompt: bool = False) -> bool:
        """The window of a change: what it does, as the command worded it, and the person's
        answer. Nothing the model said is shown here or counts as an answer."""
        if in_prompt:
            return ui.confirm(
                "\n".join(
                    [
                        *proposal.lines,
                        *proposal.warnings,
                        proposal.question or t("Apply this change?"),
                    ]
                ),
                default=False,
            )
        ui.section(t("Confirm"))
        for line in proposal.lines:
            ui.out(Text("  " + line))
        for warning in proposal.warnings:
            ui.warn(warning)
        return ui.confirm(proposal.question or t("Apply this change?"), default=False)

    def _journal(
        self,
        entry: commands.Entry,
        args: commands.Args,
        done: bool,
        before: Config | None,
        *,
        typed: bool = False,
    ) -> None:
        """The operator's journal: the command, confirmed or declined, and what it changed in
        the config (a key by its fingerprint, a permission by its name). Private fields
        are recorded only by name. A temporary chat's
        command is kept without its arguments' values: a goal or a note is the chat's text."""
        conn = self.store.conn
        given = args.model_dump(mode="json", exclude_defaults=True)
        details: dict[str, Any] = {
            "command": entry.name,
            "args": args.model_dump(mode="json", exclude=set(entry.private)),
        }
        if entry.private:
            details["args_private"] = list(entry.private)
        if self.temporary:  # the decision is the person's, kept; what the chat said is not
            details = {"command": entry.name, "args_given": sorted(given), "temporary": True}
        if typed:
            details["typed"] = True
        action = "operator_command" if done else "operator_declined"
        operator.record(conn, action, details, session_id=self.session_id)
        for change, about in commands.config_changes(before, self._saved_config()):
            operator.record(
                conn,
                change,
                {**about, **({"typed": True} if typed else {})},
                session_id=self.session_id,
            )

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
        if self.session_id is None or self.in_terminal:
            return
        text = providers.redact_keys(said, config)[0]
        if data is not None:
            configs = [config, self._saved_config()]
            budget = COMMAND_RESULT_TOKENS * CHARS_PER_TOKEN
            text += "\n" + commands.for_interlocutor(data, configs, budget)
        self.sessions.add(self.session_id, Role.DESMAN, text, kind=Kind.EVENT)

    # ── Compaction ───────────────────────────────────────────────────────────

    def _compact(
        self,
        history: History,
        models: runs.Models,
        keep_tokens: int = compaction.KEEP_TOKENS,
        incoming: int = 0,
    ) -> str | None:
        """Summarise the older messages of `history` (to make room for a new message of
        `incoming` tokens) by a task of their own; the summary, or None if there was nothing
        to summarise or no summary came of it."""
        assert self.session_id is not None
        part = compaction.split(history, keep_tokens, incoming)
        if part is None:
            return None
        try:
            runs.guard(self.store)
        except click.ClickException:  # told on its own (`_tell_blocked`); tried again later
            return None
        task_id = self.store.create_task(
            compaction.spec(history, part),
            details={
                "title": compaction.GOAL,
                "summarises": self.session_id,
                "covers": part.covers,
            },
        )  # not a task of the session: the person did not ask for it, it is never retold
        ui.note(t("Summarising the older messages of this session…"))
        summary = None
        blocked = False
        try:
            runs.summarizer(self.store, models).run(task_id)
            if self.store.task_status(task_id) is TaskStatus.DONE:
                summary = compaction.summary_of(self.store.final_result(task_id))
        except KeyboardInterrupt:
            pass
        except click.ClickException:  # protected files changed meanwhile: told on its own
            blocked = True
        if self.store.task_status(task_id) not in (TaskStatus.DONE, TaskStatus.CANCELLED):
            # Waiting for a decision or a quota: nobody needs it any more, it leaves the inbox.
            self.store.cancel_task(task_id, "the chat went on without this summary")
        if summary is None:
            if not blocked:  # once approved, a summary is tried again
                self._gave_up[self.session_id] = history.messages[-1].seq
            self._event(
                t(
                    "The older messages could not be summarised, so the model sees only the "
                    "latest ones now. All of them stay in the database."
                ),
                task_id,
            )
            return None
        self.sessions.add(
            self.session_id,
            Role.DESMAN,
            summary,
            kind=Kind.SUMMARY,
            task_id=task_id,
            covers=part.covers,
        )
        return summary

    def _compact_now(self) -> None:
        """`/compact`: every message after the last summary goes into a new one. It reads the
        newest of them within `compaction.INPUT_TOKENS` (and the history's limit); the older
        ones are covered unread, and the task is told how many."""
        current = self._current()
        config = None if current is None else self._config()
        models = None if config is None else self._models(config)
        if current is None or models is None:
            return
        history = self.sessions.history(current.id, last=HISTORY_MESSAGES)
        if not history.messages:
            ui.note(t("Nothing new to summarise: the last summary covers every message."))
            return
        summary = self._compact(history, models, keep_tokens=0)
        if summary is not None:
            ui.ok(t("Summarised. The model sees this instead of the older messages:"))
            ui.out(Text(summary, style="muted"))

    # ── Tasks ────────────────────────────────────────────────────────────────

    def _task(
        self, turn: NewTask, config: Config, models: runs.Models, interlocutor: Interlocutor | None
    ) -> None:
        access = self._access(config)
        if access is None:
            return
        root = workspace_root_for(self.store.conn)
        given = [task_files.handover(root, source, ref) for source, ref in self._sources(turn)]
        for handover in given:
            if handover.left:
                ui.warn(
                    t(
                        "Some files of task {source} are too many or too large to copy, so the "
                        "new task does not get them: {files}",
                        source=handover.ref,
                        files=", ".join(f.path for f in handover.left),
                    )
                )
        goal = turn.goal
        paths = [path for handover in given for path in handover.paths()]
        if paths:  # what the workers see is the goal: the files are named in it, by code
            goal += (
                "\n\nFiles of earlier work are in this task's workspace (read them with the "
                "file tools): " + ", ".join(paths) + "."
            )
        ui.note(t("Planning…"))
        with self._for(calls.Role.PLANNER):  # the task does not exist yet: the session asked
            planned = plan(
                models.for_(calls.Role.PLANNER),
                goal,
                limit=runs.tuned(config.tuning, "max_subtasks"),
            )
        if planned.error is not None:
            ui.warn(
                t(
                    "The model did not make a plan, so the goal is the only subtask: {error}",
                    error=planned.error,
                )
            )
        spec = must_run(planned.spec) if turn.runs_code else planned.spec
        copied: dict[str, list[str]] = {}

        def copy(_conn: sqlite3.Connection, new_task: str) -> None:
            # Inside the transaction that creates the task: no worker can claim its work
            # before the files its goal names are there, and a failed copy leaves no task.
            for handover in given:
                copied[handover.ref] = task_files.hand_over(root, new_task, handover)

        try:
            task_id = self.store.create_task(
                runs.tuned_spec(spec, config.tuning),
                details={**planned.details(), "title": turn.title},
                session_id=self.session_id,
                also=copy if given else None,
            )
        except (OSError, PathEscapeError) as e:
            ui.error(
                t(
                    "The files of an earlier task could not be copied, so the task was not "
                    "started: {error}",
                    error=e,
                )
            )
            return
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
        for source_ref, made in copied.items():
            if made:
                self._event(
                    t(
                        "Task {ref} got copies of the files of task {source}: {files}",
                        ref=ref,
                        source=source_ref,
                        files=", ".join(made),
                    ),
                    task_id,
                )
        self._work(task_id, ref, config, models, access, interlocutor)

    def _sources(self, turn: NewTask) -> list[tuple[str, str]]:
        """The tasks of `turn.files_from` (id and short ref) that are this session's own; the
        others are said to be left out (the interlocutor named a task not had here)."""
        found: list[tuple[str, str]] = []
        for ref in turn.files_from:
            try:
                row = self.store.find_task(ref)
            except (NotFoundError, AmbiguousIdError):
                row = None
            if row is None or row["session_id"] != self.session_id:
                ui.warn(
                    t(
                        "Task {ref} is not a task of this chat, so its files are not given.",
                        ref=ref,
                    )
                )
                continue
            if row["id"] not in (task_id for task_id, _ in found):
                found.append((row["id"], self.store.short_id(row["id"])))
        return found

    def _resume(
        self, task_id: str, config: Config, models: runs.Models, interlocutor: Interlocutor | None
    ) -> None:
        access = self._access(config)
        if access is None:
            return
        ref = self.store.short_id(task_id)
        self._event(t("Task {ref} continues.", ref=ref), task_id)
        self._work(task_id, ref, config, models, access, interlocutor)

    def _access(self, config: Config) -> Access | None:
        try:
            runs.guard(self.store)
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
        interlocutor: Interlocutor | None,
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
        except click.ClickException as e:  # protected files changed while it ran (5.3)
            ui.error(e.message)
            self._event(
                t("Task {ref} stopped. `desman resume {ref}` continues it.", ref=ref), task_id
            )
            return
        except KeyboardInterrupt:
            self._event(
                t("Task {ref} stopped. `desman resume {ref}` continues it.", ref=ref), task_id
            )
            return
        if self.store.find_task(task_id)["paused_at"] is not None:
            self._event(t("Task {ref} paused. `/resume {ref}` continues it.", ref=ref), task_id)
            return
        if interlocutor is None:
            actions.print_report(task_report(self.store, task_id))
            result = self.store.final_result(task_id)
            if result is not None:
                # Cut keys from each string before json.dumps escapes quotes and backslashes.
                shown = json.dumps(_redacted(result, config), ensure_ascii=False, indent=2)
                ui.out(Text(t("The result as it is:") + "\n" + shown))
            self._files(task_id, ref, workspace_root_for(self.store.conn))
        else:
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
        self.waiting.told(waiting)  # shown below: not again as a notice
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
        for line in suggestion_lines(report.suggestions):
            ui.note(line)
        result = self.store.final_result(task_id)
        root = workspace_root_for(self.store.conn)
        files = task_files.for_retelling(root, task_id)
        outcome = {
            "goal": report.goal,
            "status": report.status.value,
            "result": None if result is None else excerpt(result),
            "waiting_for_the_person": [
                {"subtask": i.description, "status": i.status.value, "reason": i.reason}
                for i in waiting
            ],
        }
        if files:
            outcome["files"] = files
        if report.suggestions:  # the person may ask for one: `models add`, with the window
            outcome["suggested_models"] = [
                {
                    "model": s.model,
                    "server": s.server,
                    "reason": s.reason,
                    "size": hub.size_text(s.size),
                    "fits_in_free_memory": s.fits,
                }
                for s in report.suggestions
            ]
        try:
            with self._for(calls.Role.INTERLOCUTOR):
                retold = interlocutor.retell(outcome)
        except (ProviderError, StructuredOutputError):
            if result is not None:  # the work is not lost with the retelling
                self._event(
                    t("The model could not retell the result; here it is as it is:")
                    + "\n"
                    + json.dumps(result, ensure_ascii=False, indent=2),
                    task_id,
                )
        else:
            # Code the task never produced must not reach the person as its work.
            if unbacked_code(retold, [result, [f["content"] for f in files]]):
                self._event(
                    t(
                        "The retelling showed code that task {ref} did not produce, so it is "
                        "not shown.",
                        ref=ref,
                    )
                    + (
                        ""
                        if result is None
                        else "\n"
                        + t("The result as it is:")
                        + "\n"
                        + json.dumps(result, ensure_ascii=False, indent=2)
                    ),
                    task_id,
                )
            else:
                self._say(retold, task_id)
                self._fallback_note(interlocutor, task_id)
        self._files(task_id, ref, root)

    def _files(self, task_id: str, ref: str, root: Path) -> None:
        """Where the task's files are, told by code (the interlocutor sees it as an event)."""
        found = task_files.listing(root, task_id)
        if not found:
            return
        lines = [f"  {f.path} ({_size(f.size)})" for f in found[: task_files.MAX_LISTED]]
        more = len(found) - task_files.MAX_LISTED
        if more > 0:
            lines.append("  " + tn(more, "…and {n} more file", "…and {n} more files"))
        self._event(
            t(
                "Files of task {ref} are in {folder}:",
                ref=ref,
                folder=task_files.folder(root, task_id),
            )
            + "\n"
            + "\n".join(lines),
            task_id,
        )

    def _greet(self, *, show_messages: bool = True) -> None:
        if self.temporary:
            ui.note(
                t(
                    "Temporary chat: what is said here is not kept and is erased when you "
                    "leave; tasks you ask for stay, like any task. /help lists the chat's "
                    "commands."
                )
            )
            return
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
        if show_messages:
            self._show_last()

    def _tell_waiting(self) -> None:
        """Subtasks that came to wait for the person since the last look, from any task: a
        worker's, another session's, a `desman run`'s, and this session's own unless the chat
        has just retold them (`_outcome` marks those told)."""
        actions.print_waiting(
            self.waiting.new(),
            t("Decide with `inbox`, or `retry`, `accept`, `fail` REF."),
        )

    def _tell_blocked(self) -> None:
        """Protected files changed without the person's approval (5.3): tasks cannot run, so
        the person hears it before asking for one, and how to look and approve, here too."""
        try:
            runs.guard(self.store)
            said = None
        except click.ClickException as e:
            said = e.message
        if said is not None and said != self._told_blocked:
            ui.warn(said)
            ui.note(
                t("Type `protected show` to see the change and `protected approve` to approve it.")
            )
        self._told_blocked = said

    def _show_last(self) -> None:
        assert self.session_id is not None
        for message in self.sessions.screen(
            self.session_id, last=max(SHOWN_ON_RETURN, ui.transcript_limit())
        ):
            if (
                message.kind is Kind.EVENT and not self.in_terminal
            ):  # a command's data or a result shown once is enough
                first = message.text.split("\n", 1)[0]
                message = message.model_copy(update={"text": first})
            elif message.kind is Kind.SUMMARY:  # its start says what it is; /compact shows it
                message = message.model_copy(update={"text": _shorten(message.text, 100)})
            ui.out(_render(message))

    # ── The chat's own commands ──────────────────────────────────────────────

    def _slash(self, line: str) -> bool:
        """Run a chat command; False when it ends the chat."""
        name = line.split()[0].lower()
        if name not in _commands():
            if self.in_terminal:
                self._typed(line[1:])
            else:
                self._terminal_command(line)
            return True
        if name == "/clear":
            if self.in_terminal:
                self.sessions.clear_terminal()
            ui.clear_screen()
            return True
        if name == "/context":
            self._context(line.split()[1:])
            return True
        if self.in_terminal and name not in TERMINAL_COMMANDS:
            ui.error(t("{name} is not available in General Terminal", name=name))
            return True
        if self.temporary and name in KEPT_ONLY:
            ui.error(t("{name} is not available in a temporary chat", name=name))
            return True
        if name == "/exit":
            return False
        if name == "/terminal":
            self._terminal_open()
        elif name == "/back":
            self._back()
        elif name == "/help":
            self._help()
        elif name == "/new":
            self._new()
        elif name == "/temporary":
            self._new(temporary=True)
        elif name == "/pin":
            self._pin()
        elif name == "/delete-all":
            self._delete_all()
        elif name == "/sessions":
            self._choose_session()
        elif name == "/rename":
            self._rename(line.strip()[len(name) :].strip())
        elif name == "/delete":
            self._delete()
        elif name == "/compact":
            self._compact_now()
        elif name == "/messages":
            try:
                body = protocol.InboxQuery(with_name=line.strip()[len(name) :].strip() or None)
                result = messenger_cli.perform(open_config(), self.store.conn, body)
                messenger_cli.print_result(result, view="messages")
            except ConfigError as e:
                ui.error(str(e))
            except ValidationError:
                ui.error(t("Invalid messenger request; nothing was sent."))
            except click.ClickException as e:
                ui.error(e.format_message())
        elif name == "/profile":
            self._profile()
        else:
            ui.error(t("unknown command {name}: /help lists the chat's commands", name=name))
        return True

    def _action(self, action: ui.Action) -> None:
        if isinstance(action, ui.TaskControlAction):
            command = (
                f"resume {action.id}"
                if action.operation == "resume"
                else f"task {action.operation} {action.id}"
            )
            self._terminal_output(lambda: self._typed(command), keep_inputs=True)
            return
        if isinstance(action, ui.SimpleAction) and action.type in {"hide_finished", "pause_tasks"}:
            operation = "hide" if action.type == "hide_finished" else "pause"
            self._terminal_output(lambda: self._bulk_tasks(operation), keep_inputs=True)
            return
        if isinstance(action, ui.SimpleAction) and action.type == "report_problem":
            self._terminal_output(self._report_problem, keep_inputs=True)
            return
        if isinstance(action, ui.SettingsAction):
            self._terminal_output(lambda: self._settings_action(action), keep_inputs=True)
            return
        if isinstance(action, ui.ModuleAction):
            # The chat validates its originating scope inside the terminal recording.
            self._terminal_output(lambda: self._module_action(action), keep_inputs=True)
            return
        if isinstance(action, ui.SimpleAction) and action.type == "terminal":
            self._terminal_open()
            return
        if isinstance(action, ui.DecideAction) or (
            isinstance(action, ui.SimpleAction) and action.type == "messenger"
        ):
            if not isinstance(action, ui.DecideAction):
                self._terminal_open()
            self._terminal_output(lambda: self._dispatch_action(action))
        else:
            self._dispatch_action(action)

    def _bulk_tasks(self, operation: str) -> None:
        entry = commands.find("task " + operation)
        assert entry is not None
        args = (
            commands.TaskHideArgs(finished=True)
            if operation == "hide"
            else commands.TaskPauseArgs(all=True)
        )
        context = commands.Context(self.store, open_config(), lambda *_: None, lambda _: None)
        proposal = entry.handler(context, args)
        if isinstance(proposal, commands.Proposal):
            confirmed = self._confirmed(proposal, in_prompt=True)
            if confirmed:
                proposal.apply()
            self._journal(entry, args, confirmed, self._saved_config(), typed=True)

    def _report_problem(self) -> None:
        try:
            description = ui.text(t("Describe the problem:")).strip()
        except (ui.Cancelled, click.Abort):  # Cancel on the window: back to the chat
            ui.note(t("Nothing changed."))
            return
        if not description:
            return
        config = self._config()
        if config is None:
            return
        entry = commands.find("global report")
        assert entry is not None
        before = self._saved_config()
        try:
            args = commands.parse(entry, {"text": description}, config)
            context = commands.Context(
                self.store,
                open_config(),
                start_task=lambda *_: None,
                resume_task=lambda _: None,
                session_id=lambda: self.session_id,
            )
            proposal = entry.handler(context, args)
            assert isinstance(proposal, commands.Proposal)
            if self._confirmed(proposal, in_prompt=True):
                made = proposal.apply()
            else:
                made = commands.Outcome(None, done=False)
                ui.note(t("Nothing changed."))
            self._journal(entry, args, made.done, before, typed=True)
        except (ui.Cancelled, click.Abort):
            ui.note(t("Nothing changed."))
        except (commands.BadArgs, click.ClickException) as e:
            message = e.format_message() if isinstance(e, click.ClickException) else str(e)
            ui.error(providers.redact_keys(message, config)[0])

    def _settings_action(self, action: ui.SettingsAction) -> None:
        with ui.recording(
            redact=lambda text: providers.redact_keys(text, self._saved_config())[0]
        ) as notices:
            self._apply_settings_action(action, notices)

    def _apply_settings_action(self, action: ui.SettingsAction, notices: ui.Recording) -> None:
        """Human fields use catalog bodies and the same redacted operator journal."""
        from desman.gui.settings import command, role_change
        from desman.settings_actions import Forget, Models, ProviderSetting, Roles

        before = self._saved_config()
        message, ok = "", True
        try:
            config = open_config().load()
            change = action.change
            if isinstance(change, Forget):
                if not config.personalize:
                    raise ValueError("personalisation off")
                self.profile.forget(change.note, Author.PERSON, session_id=self.session_id)
            elif isinstance(change, Models) and change.operation == "use":
                from desman import local_models

                if change.server != "ollama":
                    raise ValueError("page uses Ollama only")
                with open_config().edit() as edited:
                    local_models.select_installed(edited, change.model)
                entry = commands.find("models add")
                assert entry is not None
                self._journal(
                    entry,
                    commands.ModelAddArgs(model=change.model, server="ollama"),
                    True,
                    before,
                    typed=True,
                )
            elif isinstance(change, Roles):
                old, new, edited = tuning_cli.set_tuning(open_config(), role_change(change, config))
                tuning_cli.print_change(old, new, edited)
                entry = commands.find("tuning")
                assert entry is not None
                self._journal(
                    entry,
                    commands.TuningArgs(models={change.role: change.model or "auto"}, keys=None),
                    True,
                    before,
                    typed=True,
                )
            elif isinstance(change, ProviderSetting):
                from desman import wizard

                if providers.redact_keys(change.value, config)[1]:
                    raise ValueError("a key belongs only in the hidden field")
                p = providers.find(change.provider)
                setting = next(s for s in p.settings if s.env == change.setting)
                with open_config().edit() as edited:
                    wizard.set_provider_setting(edited, p, setting, change.value)
                entry = commands.find("config")
                assert entry is not None
                self._journal(
                    entry, commands.ConfigArgs(section="providers"), True, before, typed=True
                )
            else:
                entry, args = command(change, config)
                context = commands.Context(
                    self.store,
                    open_config(),
                    start_task=lambda *_: None,
                    resume_task=lambda _: None,
                    session_id=lambda: self.session_id,
                )
                with ui.recording() as shown:
                    made = entry.handler(context, args)
                    if isinstance(made, commands.Proposal):
                        # Keep confirmations belonging to downloads/join/reset. Ordinary fields
                        # are already the person's explicit act, as with !COMMAND.
                        confirm = entry.name in (
                            "models add",
                            "models remove",
                            "global join",
                            "config reset",
                        )
                        made = (
                            made.apply()
                            if not confirm or self._confirmed(made, in_prompt=True)
                            else commands.Outcome(None, done=False)
                        )
                if isinstance(args, commands.KeyAddArgs):
                    # The newly saved key may also have been pasted into its label.
                    args = args.model_copy(
                        update={"label": providers.redact_keys(args.label, self._saved_config())[0]}
                    )
                self._journal(entry, args, made.done, before, typed=True)
                ok = made.done
                if entry.name in ("keys check", "models show", "protected show"):
                    message = shown.text.strip()
                if not ok:
                    message = t("Nothing changed.")
        except (ui.Cancelled, click.Abort):
            ok, message = False, t("Nothing changed.")
        except (
            ValueError,
            StopIteration,
            ConfigError,
            click.ClickException,
            commands.BadArgs,
        ) as e:
            ok = False
            message = (
                e.format_message()
                if isinstance(e, click.ClickException)
                else t("Please check your answer and try again.")
            )
            ui.error(message)
        if ok and not message:
            message = t("Saved")
            ui.ok(message)
        if notices.warnings:
            message = "\n".join([message, *dict.fromkeys(notices.warnings)])
        ui.settings_result(action.id, ok, message)

    def _module_action(self, action: ui.ModuleAction) -> None:
        config = self._config()
        if config is None:
            return
        from desman.gui.settings import listed_choice

        try:
            if action.scope == "session" and (
                action.scope_id != self.session_id or self.in_terminal
            ):
                raise ValueError("not the current chat")
            row = next(r for r in tuning_cli.role_rows(config) if r["role"] == action.role)
            if action.scope == "role":
                saved = choices.Choice(model=row["model"], key=row["key_fingerprint"])
            elif action.scope_id is None:
                saved = self._pending_choice or choices.Choice()
            else:
                saved = choices.get(self.store.conn, action.scope, action.scope_id, action.role)
            if (
                action.model is not None
                and action.model not in row["candidate_models"]
                and action.model != saved.model
                and not listed_choice(config, action.model, action.key)
            ):
                raise ValueError("unavailable model")
            tuning_cli.resolved_key(config, action.role, action.model, action.key)
            if action.scope == "role":
                old, new, edited = tuning_cli.set_tuning(
                    open_config(),
                    tuning_cli.TuningChange(
                        models={action.role: action.model or "auto"},
                        fingerprints={action.role: action.key},
                    ),
                )
                before = edited.model_copy(update={"tuning": old})
                for name, about in commands.config_changes(before, edited):
                    operator.record(self.store.conn, name, about, session_id=self.session_id)
                lines = tuning_cli.lines(old, new, edited)
            elif action.scope == "session" and action.scope_id is None:
                self._pending_choice = choices.Choice(model=action.model, key=action.key)
                ui.pending_module(self._pending_choice.model_dump())
                lines = []
            else:
                assert action.scope_id is not None
                old_choice, new_choice = modules_cli.set_module(
                    self.store.conn,
                    config,
                    action.scope,
                    action.scope_id,
                    action.role,
                    action.model or "auto",
                    fingerprints={action.role: action.key},
                )
                lines = modules_cli.change_lines(action.role, old_choice, new_choice, config)
            for line in lines:
                ui.ok(line)
        except (ValueError, click.ClickException):
            ui.error(t("Invalid module choice."))

    def _dispatch_action(self, action: ui.Action) -> None:
        """Dispatch validated browser choices through the chat's existing command bodies."""
        if isinstance(action, ui.SessionAction):
            selected = self.sessions.get(action.id)
            if selected is None:
                ui.error(t("This session is no longer available."))
                return
            if action.type == "open_session":
                self._open_session(selected)
            elif action.type == "delete_session":
                self._delete(selected.id)
            else:
                self._pin(selected.id, action.type == "pin")
        elif isinstance(action, ui.NewSessionAction):
            self._new(action.temporary)
        elif isinstance(action, ui.SimpleAction):
            if action.type == "delete_all":
                self._delete_all()
            elif action.type == "messenger":
                self._slash("/messages")
            else:
                self._typed("config")
        elif isinstance(action, ui.DecideAction):
            config = self._config()
            if config is None:
                return
            entry = commands.find(action.decision)
            assert entry is not None
            self._command(
                Command(
                    type="command",
                    name=action.decision,
                    args={
                        "subtask": action.ref,
                        "if_version": action.version,
                        "note": action.note or None,
                    },
                ),
                config,
                None,
                None,
                typed=True,
                typed_entry=entry,
            )
        ui.session(self.session_id, self.temporary)

    def _fallback_note(self, interlocutor: Interlocutor, task_id: str | None = None) -> None:
        llm = interlocutor.llm
        if not isinstance(llm, router.Fallback):
            return
        call = llm.last_call
        if not call.answered or not any(item.model == llm.label for item in call.skipped):
            return
        reasons = {
            "memory": t("model may not fit in memory"),
            "busy": t("busy"),
            "limit": t("limit reached"),
            "refused": t("no such model or no access"),
            "key": t("key refused"),
            "unavailable": t("unavailable"),
            "context": t("context too small"),
            "unfit": t("model skipped"),
        }
        skipped = "; ".join(
            f"{item.model}: "
            + (
                t(
                    "context too small for this chat ({needed} tokens needed, {context} set)",
                    needed=item.needed,
                    context=item.context,
                )
                if item.reason == "context"
                else reasons[
                    "key" if item.reason == "refused" and item.status in (401, 403) else item.reason
                ]
            )
            for item in call.skipped
        )
        self._event(
            providers.redact_keys(
                t("Answered by {model}. Skipped: {skipped}", model=call.answered, skipped=skipped),
                self._saved_config(),
            )[0],
            task_id,
        )

    def _context(self, args: list[str]) -> None:
        if self.in_terminal:
            ui.note(t("General Terminal talks to no model."))
            return
        if args not in ([], ["full"]):
            ui.note(t("Usage: /context [full]"))
            return
        config = self._config()
        if config is None:
            return
        own = (
            choices.get(self.store.conn, "session", self.session_id, "interlocutor")
            if self.session_id
            else self._pending_choice or choices.Choice()
        )
        config = choices.effective(config, "interlocutor", own)
        try:
            lineup = router.lineup(config, calls.Role.INTERLOCUTOR)
        except router.NoProvider:
            lineup = []
        labels = [c.label for c in lineup]
        model_line = t("Model: {model}", model=labels[0] if labels else t("missing"))
        if len(labels) > 1:
            rest = ", ".join(labels[1:4])
            if len(labels) > 4:
                rest += " " + t("+{count} more", count=len(labels) - 4)
            model_line += " · " + t("then: {models}", models=rest)
        history = (
            self.sessions.history(self.session_id, last=HISTORY_MESSAGES)
            if self.session_id
            else History(summary=None, messages=[])
        )
        notes = self.profile.notes() if config.personalize and not self.temporary else None
        interlocutor = Interlocutor(_InspectionLLM(), config.language, notes, config.behaviour)
        placeholder = Message(
            seq=0,
            session_id=self.session_id or "",
            at="",
            role=Role.USER,
            kind=Kind.MESSAGE,
            text="[your next message]",
        )
        history = history.model_copy(update={"messages": [*history.messages, placeholder]})
        request = interlocutor.build_request(history)
        system = sum(estimate_tokens(m.content) for m in request.messages if m.role == "system")
        messages = sum(estimate_tokens(m.content) for m in request.messages if m.role != "system")
        lines = [
            model_line,
            t(
                "≈ {total} tokens in total (system {system}, messages {messages})",
                total=system + messages,
                system=system,
                messages=messages,
            ),
        ]
        for message in request.messages:
            lines.extend(["", str(message.role), message.content])
        ui.out(ui.ContextText(providers.redact_keys("\n".join(lines), config)[0]))

    def _help(self) -> None:
        commands.print_help(terminal=self.in_terminal, temporary=self.temporary)

    def _leave_terminal(self) -> None:
        if self.in_terminal and self._previous is not None:
            previous, self._previous = self._previous, None
            if previous[1] and previous[0] is not None:
                self.sessions.erase(previous[0])

    def _drop_pending(self) -> None:
        self._pending_choice = None
        ui.pending_module(None)

    def _new(self, temporary: bool = False) -> None:
        self._drop_pending()
        self._leave_terminal()
        if self.session_id is None and self.temporary == temporary:
            ui.reset()
            ui.note(t("This session is new already."))
            return
        self._erase()
        self.temporary = temporary
        self.models, self._models_for = None, None
        self.session_id = None
        ui.reset()
        ui.session(None, temporary)
        if temporary:
            self._greet()
        else:
            ui.ok(t("New session: it starts with your next message."))

    def _pin(self, id: str | None = None, pinned: bool | None = None) -> None:
        current = self.sessions.get(id) if id is not None else self._current()
        if current is None:
            return
        pinned = not current.pinned if pinned is None else pinned
        self.sessions.pin(current.id, pinned)
        operator.record(
            self.store.conn,
            "operator_session_pinned" if pinned else "operator_session_unpinned",
            session_id=current.id,
        )
        ui.ok(t("Session pinned.") if pinned else t("Session unpinned."))

    def _delete_all(self) -> None:
        count = sum(not session.pinned for session in self.sessions.recent(limit=-1))
        if not count:
            ui.note(t("No unpinned sessions to delete."))
            return
        ui.section(t("Confirm"))
        ui.note(
            tn(
                count,
                "Delete {count} unpinned session? Pinned sessions stay; messages remain in the database.",
                "Delete {count} unpinned sessions? Pinned sessions stay; messages remain in the database.",
                count=count,
            )
        )
        try:
            if not ui.confirm(t("Delete them?"), default=False):
                return
        except (ui.Cancelled, click.Abort):
            return
        deleted = self.sessions.delete_all()
        operator.record(self.store.conn, "operator_sessions_deleted", {"count": deleted})
        if (
            not self.temporary
            and not self.in_terminal
            and self.session_id is not None
            and self.sessions.get(self.session_id) is None
        ):
            self.session_id = None
            ui.reset()
        ui.ok(tn(deleted, "Deleted {count} session.", "Deleted {count} sessions.", count=deleted))

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

    def _delete(self, id: str | None = None) -> None:
        current = self.sessions.get(id) if id is not None else self._current()
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
        if current.id == self.session_id:
            self.session_id = None
            ui.reset()
        if self.session_id is None:
            ui.ok(t("Session deleted. A new one starts with your next message."))
        else:
            ui.ok(tn(1, "Deleted {count} session.", "Deleted {count} sessions.", count=1))

    def _profile(self) -> None:
        """`/profile`: the two things that shape how the interlocutor talks (how the person set
        it, and the notes it keeps if they let it), and what the person may do with them."""
        while True:
            config = self._config()
            if config is None:
                return
            notes = self.profile.notes()
            ui.section(t("How the interlocutor talks"))
            for part, label in wizard.behaviour_labels().items():
                shown = wizard.behaviour_shown(config.behaviour, part)
                ui.out(Text(f"  {label}: ", style="muted") + Text(shown))
            ui.section(t("Notes about you"))
            if config.personalize:
                ui.note(
                    t(
                        "The interlocutor keeps these notes about you to talk the way you like. "
                        "They go to the model providers with every message, and free "
                        "providers may keep what they get."
                    )
                )
            else:
                ui.note(t("Personalisation is off: the interlocutor neither sees nor keeps notes."))
            for note in notes:
                by = t("by you") if note.author is Author.PERSON else t("by the interlocutor")
                ui.out(Text("  • " + note.text) + Text(f"  ({by})", style="muted"))
            if not notes and config.personalize:
                ui.out(Text("  " + t("No notes yet."), style="muted"))
            choices = [ui.Choice(label=t("change how it talks"), value="behaviour")]
            if config.personalize:
                choices.append(ui.Choice(label=t("add a note"), value="add"))
            if notes:
                choices += [
                    ui.Choice(label=t("change a note"), value="change"),
                    ui.Choice(label=t("forget a note"), value="forget"),
                    ui.Choice(label=t("forget all notes"), value="clear"),
                ]
            choices += [
                ui.Choice(label=t("turn personalisation off"), value="personalize")
                if config.personalize
                else ui.Choice(label=t("turn personalisation on"), value="personalize"),
                ui.Choice(label=t("done"), value="done"),
            ]
            try:
                chosen = ui.select(t("What now?"), choices, default="done")
                if chosen == "done":
                    return
                self._profile_do(chosen, notes, config)
            except (ui.Cancelled, click.Abort):
                return

    def _profile_do(self, chosen: str, notes: list[Note], config: Config) -> None:
        by, session = Author.PERSON, self.session_id
        if chosen == "personalize":
            on = wizard.ask_personalize(config.personalize)
            self._edit_config(lambda c: setattr(c, "personalize", on))
            if on != config.personalize:
                ui.ok(
                    t("Personalisation is on.")
                    if on
                    else t(
                        "Personalisation is off. Notes already kept stay, unused, until you "
                        "erase them."
                    )
                )
        elif chosen == "behaviour":
            behaviour = wizard.ask_behaviour(config.behaviour, config)
            self._edit_config(lambda c: setattr(c, "behaviour", behaviour))
            for line in config_cli.behaviour_lines(config.behaviour, behaviour):
                ui.ok(line)
        elif chosen == "clear":
            if ui.confirm(t("Forget every note?"), default=False):
                self.profile.clear(session_id=session)
                ui.ok(t("Every note is forgotten."))
        elif chosen == "add":
            text = self._note_text(t("The note"), "", config)
            if text:
                try:
                    self.profile.add(text, by, session_id=session)
                except ProfileFull:
                    ui.error(t("The profile is full: change or forget a note first."))
        else:
            note = ui.select(
                t("Which note?"),
                [ui.Choice(label=_shorten(n.text, 70), value=n) for n in notes],
            )
            if chosen == "forget":
                self.profile.forget(note.id, by, session_id=session)
                return
            text = self._note_text(t("The note"), note.text, config)
            if text:
                self.profile.change(note.id, text, by, session_id=session)

    def _edit_config(self, change: Callable[[Config], None]) -> None:
        """Change the config under its lock (another command's key survives) and journal it."""
        with open_config().edit() as edited:
            before = edited.model_copy(deep=True)
            change(edited)
        for action, about in commands.config_changes(before, edited):
            operator.record(self.store.conn, action, about, session_id=self.session_id)

    def _note_text(self, question: str, default: str, config: Config) -> str:
        """A note the person types: keys cut, and too long a note refused."""
        text, cut = providers.redact_keys(ui.text(question, default=default), config)
        if cut:
            ui.warn(t("That looked like an API key, so it was cut from the note."))
        text = " ".join(text.split())
        if len(text) > NOTE_CHARS:
            ui.error(t("A note is at most {n} characters.", n=NOTE_CHARS))
            return ""
        return text

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
                    label=f"{s.id[:8]}  {_when(s.last_at)}  {_shorten(s.name) or '…'}",
                    value=s.id,
                    hint=(t("pinned") + " · " if s.pinned else "")
                    + tn(s.messages, "{n} message", "{n} messages")
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
        if chosen is None:
            self._new()
        else:
            selected = self.sessions.get(chosen)
            if selected is None:
                ui.error(t("This session is no longer available."))
                return
            self._open_session(selected)

    def _open_session(self, selected: SessionInfo) -> None:
        self._leave_terminal()
        self._erase()
        self.temporary = False
        self._drop_pending()
        self.session_id = selected.id
        self.models, self._models_for = None, None
        ui.reset()
        ui.session(selected.id, False)
        ui.ok(t("Switched to the session of {when}.", when=_when(selected.last_at)))
        self._show_last()


def _redacted(value: Any, config: Config | None) -> Any:
    """`value` (JSON data) with configured keys and key-shaped strings cut from every string."""
    if isinstance(value, str):
        return providers.redact_keys(value, config)[0]
    if isinstance(value, list):
        return [_redacted(v, config) for v in value]
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            name, n = _redacted(k, config), 1
            while name in out:  # two names that each held a key: keep both values
                n += 1
                name = f"{_redacted(k, config)} ({n})"
            out[name] = _redacted(v, config)
        return out
    return value


def _render(message: Message) -> Text:
    if message.kind is Kind.SUMMARY:
        return Text(t("Summary of the older messages:") + "\n" + message.text, style="muted")
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


def _size(size: int) -> str:
    """A file's size for people (decimal units)."""
    if size < 1_000:
        return f"{size} B"
    if size < 1_000_000:
        return f"{size / 1_000:.1f} KB"
    return f"{size / 1_000_000:.1f} MB"


def _shorten(text: str, width: int = 50) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


class _InspectionLLM:
    """Context inspection builds requests without constructing a network client."""

    name = "inspection"

    def complete(self, request: Request) -> Completion:
        raise AssertionError("context inspection must not call a model")
