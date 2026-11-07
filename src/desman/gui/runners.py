"""Session-owned synchronous turns, dispatched independently of the browser input loop."""

from __future__ import annotations

import io
import sqlite3
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from typing import Any

import click
from rich.console import RenderableType
from rich.text import Text

from desman import commands, providers, runs, ui
from desman.chat import Chat, run_turn
from desman.config import ConfigError, open_config
from desman.core import db
from desman.core.sessions import History, Kind, Message, Role, Sessions
from desman.core.store import Clock
from desman.gui.front import WebFront
from desman.i18n import t
from desman.interlocutor import Interlocutor
from desman.llm.structured import completion_guard

_turn_locks: dict[tuple[Path, int], tuple[Lock, int]] = {}
_turn_locks_guard = Lock()


@contextmanager
def turn_lock(key: tuple[Path, int]) -> Iterator[None]:
    with _turn_locks_guard:
        lock, users = _turn_locks.get(key, (Lock(), 0))
        _turn_locks[key] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _turn_locks_guard:
            _, users = _turn_locks[key]
            if users == 1:
                del _turn_locks[key]
            else:
                _turn_locks[key] = (lock, users - 1)


class Stopped(BaseException):
    """Unwind a stopped turn without treating it as a declined command or model error."""

    recorded = False


class TurnSessions(Sessions):
    """Persist input immediately while each turn sees only the inputs it can answer.

    A queued user message can precede the previous turn's reply in database order. Put
    the current input last in model context and keep later queued inputs out of it.
    """

    user_through: int | None = None
    current_user: int | None = None

    def history(self, session_id: str, last: int | None = None) -> History:
        history = super().history(session_id)
        messages = [
            m
            for m in history.messages
            if m.seq != self.current_user
            and (m.role is not Role.USER or self.user_through is None or m.seq <= self.user_through)
        ]
        if self.current_user is not None:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND seq = ?",
                (session_id, self.current_user),
            ).fetchone()
            if row is not None:
                messages.append(Message.model_validate(dict(row)))
        earlier = max(0, len(messages) - last) if last is not None else 0
        return History(
            summary=history.summary,
            messages=messages[-last:] if last else messages,
            earlier=earlier,
        )


class RunnerChat(Chat):
    def _catch_up(self, interlocutor: Interlocutor) -> None:
        # A previous task's retelling does not answer the new user input.
        reply_to, self.sessions.reply_to = self.sessions.reply_to, None
        try:
            super()._catch_up(interlocutor)
        finally:
            self.sessions.reply_to = reply_to

    def _outcome(self, task_id: str, ref: str, interlocutor: Interlocutor) -> None:
        watermark = self.sessions.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM audit_log WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        try:
            super()._outcome(task_id, ref, interlocutor)
        except Stopped as stopped:
            assert self.session_id is not None
            # A stopped retelling is settled too: idle polling must not immediately
            # restart it. New task changes beyond this watermark remain discoverable.
            self.sessions.add(
                self.session_id,
                Role.DESMAN,
                t("Stopped; the reply that came later was dropped"),
                kind=Kind.EVENT,
                task_id=task_id,
                outcome_audit=watermark,
            )
            stopped.recorded = True
            raise


def runner_chat(sessions: Sessions) -> Chat:
    return RunnerChat(sessions, background_tasks=True)


@dataclass(eq=False)
class Work:
    text: str | None
    added: bool = False
    user_seq: int | None = None
    stopped: Event = field(default_factory=Event)
    slot: bool = False


@dataclass
class Runner:
    session_id: str
    temporary: bool
    queue: deque[Work] = field(default_factory=deque)
    current: Work | None = None
    thread: Thread | None = None
    front: SessionFront | None = None
    erase: bool = False


class SessionFront(WebFront):
    """A private prompt mailbox; all output is durable, never sent to the open screen."""

    def __init__(self, owner: Runners, runner: Runner, sessions: Sessions) -> None:
        super().__init__(redact=owner.front.redact)
        self.owner, self.runner, self.sessions = owner, runner, sessions

    def next_prompt_id(self) -> int:
        return self.owner.front.next_prompt_id()

    def emit(self, event_type: str, **data: Any) -> None:
        # Prompt metadata lives only in this mailbox (especially hidden input).
        super().emit(event_type, **data)
        self.owner.front.session_changed(self.runner.session_id)

    def out(self, *items: RenderableType | str) -> None:
        safe = [Text.from_markup(i) if isinstance(i, str) else i for i in items]
        if self._panel_quiet:
            return
        console = ui._RecordingConsole(ui.Recording(io.StringIO(), self.redact))
        console.print(*safe)
        text = self.redact(console.recording.text).strip()
        if text:
            self.sessions.add(self.runner.session_id, Role.DESMAN, text, kind=Kind.EVENT)
            self.changed(self.runner.session_id)

    def changed(self, session_id: str) -> None:
        with self.owner.front.condition:
            if self.owner.front.current == session_id:
                self.sessions.mark_read(session_id)
            self.owner.front.session_changed(session_id)

    def note(self, text: str) -> None:
        if text == t("Planning…"):
            self.live_update(Text(text))
        else:
            self.out(Text(text))

    def warn(self, text: str, *, stderr: bool = False) -> None:
        self.out(Text(text))

    def error(self, text: str) -> None:
        self.out(Text(text))

    def ok(self, text: str) -> None:
        self.out(Text(text))

    def header(self, title: str, subtitle: str = "") -> None:
        self.out(Text(title + ("\n" + subtitle if subtitle else "")))

    def section(self, title: str) -> None:
        self.out(Text(title))

    def live_update(self, renderable: RenderableType) -> None:
        super().live_update(renderable)
        self.owner.front.session_changed(self.runner.session_id)

    def _ask(self, kind: str, question: str = "", **data: Any) -> Any:
        # Stop/page shutdown cancels even a prompt created after cancellation began.
        work = self.runner.current
        if work is None or work.stopped.is_set() or self.owner.closed:
            raise ui.Cancelled
        with self.owner.condition:
            self.owner.active -= 1
            work.slot = False
            self.owner.condition.notify_all()
        try:
            return super()._ask(kind, question, **data)
        finally:
            with self.owner.condition:
                if not work.stopped.is_set():
                    self.owner.tickets.append(work)
                    self.owner.condition.notify_all()
                    self.owner.condition.wait_for(
                        lambda: (
                            work.stopped.is_set()
                            or self.owner._eligible(work)
                            and self.owner.active < self.owner.cap
                        )
                    )
                    if not work.stopped.is_set():
                        self.owner.tickets.remove(work)
                        self.owner.active += 1
                        work.slot = True
                        self.owner.condition.notify_all()
                if work.stopped.is_set():
                    raise Stopped


class Runners:
    def __init__(
        self,
        path: Path,
        front: WebFront,
        *,
        cap: int = 4,
        factory: Callable[[Sessions], Chat] = runner_chat,
        clock: Callable[[], float] = time.monotonic,
        session_clock: Clock | None = None,
    ) -> None:
        self.path, self.front, self.cap = path, front, cap
        self.factory, self.clock = factory, clock
        self.session_clock = session_clock
        self.condition = Condition()
        self.runners: dict[str, Runner] = {}
        self.tickets: deque[Work] = deque()
        self.active = 0
        self.closed = False
        self.summary_failures: dict[str, dict[str, int]] = {}
        front.runners = self
        front.database = path

    def recover(self, sessions: Sessions) -> None:
        """Called only at page startup; ordinary polling never replays inputs."""
        for message in sessions.pending_turns():
            sessions.register_turn(message.seq)
            self.submit(message.session_id, message.text, added=True, user_seq=message.seq)

    def set_cap(self, cap: int) -> None:
        with self.condition:
            self.cap = cap
            self.condition.notify_all()

    def submit(
        self,
        session_id: str,
        text: str | None,
        *,
        temporary: bool = False,
        added: bool = False,
        user_seq: int | None = None,
    ) -> None:
        with self.condition:
            if self.closed:
                return
            runner = self.runners.get(session_id)
            if runner is None:
                runner = Runner(session_id, temporary)
                self.runners[session_id] = runner
            # Polling a settled task queues at most one retelling behind a live turn.
            if text is None and (runner.current is not None or runner.queue):
                return
            work = Work(text, added, user_seq)
            runner.queue.append(work)
            self.tickets.append(work)
            if runner.thread is None:
                runner.thread = Thread(target=self._run, args=(runner,), daemon=True)
                runner.thread.start()
            self.condition.notify_all()
        self.front.session_changed(session_id)

    def status(self, session_id: str) -> dict[str, Any]:
        with self.condition:
            runner = self.runners.get(session_id)
            if runner is None:
                return {"state": None, "queued": 0, "ahead": 0}
            pending = runner.front is not None and runner.front.prompt is not None
            first = runner.current or (runner.queue[0] if runner.queue else None)
            waiting = first in self.tickets
            ahead = (
                self.active + list(self.tickets).index(first)
                if waiting and first is not None
                else 0
            )
            return {
                "state": "pending" if pending else "queued" if waiting else "answering",
                "queued": len(runner.queue) + int(waiting and runner.current is not None),
                "ahead": ahead,
            }

    def mailbox(self, session_id: str | None) -> SessionFront | None:
        with self.condition:
            runner = self.runners.get(session_id or "")
            return runner.front if runner else None

    def _run(self, runner: Runner) -> None:
        conn: sqlite3.Connection | None = None
        sink: SessionFront | None = None
        try:
            conn = db.open_db(self.path)
            sessions = (
                TurnSessions(conn, clock=self.session_clock)
                if self.session_clock
                else TurnSessions(conn)
            )
            sink = SessionFront(self, runner, sessions)
            with self.condition:
                runner.front = sink
            # One context per session lifetime, never shared with the dispatcher.
            chat = self.factory(sessions)
            chat.session_id, chat.temporary = runner.session_id, runner.temporary
            chat.turn_sink = sink.changed
            with self.condition:
                chat._gave_up = self.summary_failures.setdefault(runner.session_id, {})
            with ui.use(sink), sink.capture():
                while True:
                    with self.condition:
                        if not runner.queue:
                            break
                        work = runner.queue.popleft()
                        runner.current = work

                        def ready(work: Work = work) -> bool:
                            return (
                                work.stopped.is_set()
                                or self._eligible(work)
                                and self.active < self.cap
                            )

                        self.condition.wait_for(ready)
                        if work.stopped.is_set():
                            runner.current = None
                            continue
                        self.tickets.remove(work)
                        self.active += 1
                        work.slot = True
                        self.condition.notify_all()
                    self.front.session_changed(runner.session_id)
                    try:
                        with self.condition:
                            queued_users = [
                                w.user_seq for w in runner.queue if w.user_seq is not None
                            ]
                        sessions.reply_to = work.user_seq
                        sessions.current_user = work.user_seq
                        sessions.user_through = (
                            work.user_seq
                            if work.user_seq is not None
                            else (
                                min(queued_users) - 1
                                if queued_users
                                else sessions.conn.execute(
                                    "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ? AND role = 'user'",
                                    (runner.session_id,),
                                ).fetchone()[0]
                            )
                        )
                        sink.ended = False

                        def check(work: Work = work) -> None:
                            if work.stopped.is_set():
                                raise Stopped

                        with completion_guard(check):
                            if work.user_seq is None:
                                self._execute(chat, runner, work)
                            else:
                                with turn_lock((self.path.resolve(), work.user_seq)):
                                    if not sessions.turn_settled(work.user_seq):
                                        self._execute(chat, runner, work)
                    except Stopped as stopped:
                        if not stopped.recorded:
                            sessions.add(
                                runner.session_id,
                                Role.DESMAN,
                                t("Stopped; the reply that came later was dropped"),
                                kind=Kind.EVENT,
                                settle=True,
                            )
                    except ui.Cancelled:
                        pass
                    except Exception as exc:
                        # No raw exception/model content in diagnostics or the event log.
                        sink.error(t("The chat turn failed ({error}).", error=type(exc).__name__))
                    finally:
                        with self.condition:
                            runner.current = None
                            if work.slot:
                                self.active -= 1
                                work.slot = False
                            sink.live = None
                            self.condition.notify_all()
                        sink.changed(runner.session_id)
                if chat._http is not None:
                    chat._http.close()
        except Exception as exc:
            with self.condition:
                failed = set(runner.queue)
                if runner.current is not None:
                    failed.add(runner.current)
                    if runner.current.slot:
                        self.active -= 1
                self.tickets = deque(w for w in self.tickets if w not in failed)
                runner.current = None
                runner.queue.clear()
                self.condition.notify_all()
            if sink is not None:
                sink.error(t("The chat turn failed ({error}).", error=type(exc).__name__))
        finally:
            if conn is not None:
                if runner.erase or self.closed and runner.temporary:
                    Sessions(conn).erase(runner.session_id)
                conn.close()
            with self.condition:
                # Submit and retirement share this lock, so work cannot be stranded.
                runner.thread = None
                runner.front = None
                if runner.queue:
                    runner.thread = Thread(target=self._run, args=(runner,), daemon=True)
                    runner.thread.start()
                else:
                    self.runners.pop(runner.session_id, None)
                self.condition.notify_all()
            self.front.session_changed(runner.session_id)

    def _eligible(self, work: Work) -> bool:
        def ready(candidate: Work) -> bool:
            return any(
                r.current is candidate or r.current is None and r.queue and r.queue[0] is candidate
                for r in self.runners.values()
            )

        return next((w for w in self.tickets if ready(w)), None) is work

    def _execute(self, chat: Chat, runner: Runner, work: Work) -> None:
        if work.text is None:
            config = chat._config()
            models = chat._models(config, command=True) if config else None
            if config is not None and models is not None:
                from desman.core import calls
                from desman.interlocutor import Interlocutor

                chat._catch_up(
                    Interlocutor(
                        models.for_(calls.Role.INTERLOCUTOR),
                        config.language,
                        chat.profile.notes()
                        if config.personalize and not runner.temporary
                        else None,
                        config.behaviour,
                    )
                )
        elif work.text.strip() == "/compact":
            chat._compact_now()
        elif work.text.startswith(("/", "!")):
            chat._typed(work.text[1:])
        else:
            run_turn(runner.session_id, work.text, context=chat, added=work.added)

    def stop(self, session_id: str | None) -> None:
        with self.condition:
            runner = self.runners.get(session_id or "")
            if runner is None:
                return
            work = runner.current
            conn = db.open_db(self.path)
            try:
                sessions = Sessions(conn)
                for item in [work, *runner.queue]:
                    if item is not None and item.user_seq is not None:
                        sessions.settle_turn(item.user_seq, cancelled=True)
            finally:
                conn.close()
            if work is not None:
                work.stopped.set()
            for queued in runner.queue:
                queued.stopped.set()
            self.tickets = deque(w for w in self.tickets if not w.stopped.is_set())
            runner.queue.clear()
            sink = runner.front
            self.condition.notify_all()
        if sink is not None:
            with sink.condition:
                sink.ended = True
                if sink.prompt is not None:
                    sink._answer = {"cancel": True}
                    sink.condition.notify_all()
        self.front.session_changed(session_id or "")

    def defer_erase(self, session_id: str) -> bool:
        with self.condition:
            runner = self.runners.get(session_id)
            if runner is None:
                return False
            runner.erase = True
            return True

    def close(self, timeout: float = 10) -> None:
        with self.condition:
            self.closed = True
            # Queued inputs stay pending in SQLite for the next page. A running
            # model may still finish durably during or after the grace period.
            for runner in self.runners.values():
                for work in runner.queue:
                    work.stopped.set()
                runner.queue.clear()
                if runner.current is not None and not runner.current.slot:
                    runner.current.stopped.set()
            self.tickets = deque(w for w in self.tickets if not w.stopped.is_set())
            self.condition.notify_all()
            sinks = [r.front for r in self.runners.values() if r.front is not None]
        # Shutdown cancels human input; an in-flight reply can still finish durably.
        for sink in sinks:
            assert sink is not None
            with sink.condition:
                sink.ended = True
                if sink.prompt is not None:
                    sink._answer = {"cancel": True}
                    sink.condition.notify_all()
        deadline = self.clock() + timeout
        joined: set[Thread] = set()
        while True:
            with self.condition:
                threads = [
                    r.thread
                    for r in self.runners.values()
                    if r.thread is not None and r.thread not in joined
                ]
            if not threads:
                return
            for thread in threads:
                thread.join(max(0, deadline - self.clock()))
                if thread.is_alive():
                    return  # the total grace period elapsed; replies remain durable
                joined.add(thread)


class PageChat(Chat):
    """The main thread owns navigation and dispatch; a runner owns each model turn."""

    def __init__(self, sessions: Sessions, runners: Runners, **options: Any) -> None:
        super().__init__(sessions, background_tasks=True, **options)
        self.runners = runners
        self.dispatch_turn = self.dispatch

    def _erase_session(self, session_id: str) -> None:
        if not self.runners.defer_erase(session_id):
            super()._erase_session(session_id)

    def _notices(self) -> None:
        config = self._saved_config()
        if config is not None:
            self.runners.set_cap(runs.tuned(config.tuning, "chat_turns"))
        super()._notices()

    def background_outcomes(self, enqueue: Callable[[str, bool], None] | None = None) -> None:
        super().background_outcomes(
            enqueue or (lambda id, temporary: self.runners.submit(id, None, temporary=temporary))
        )

    def dispatch(self, line: str) -> bool:
        text = line.strip()
        manager = open_config()
        if text == "/compact" and self.session_id is None:
            return False
        try:
            config = manager.load()
        except ConfigError:
            return False
        handed_off = False
        model_command = text == "/compact"
        if text.startswith(("/", "!")) or self.in_terminal:
            try:
                parsed = commands.parse_line(text.lstrip("/!"), config)
            except (commands.BadArgs, click.ClickException, ValueError):
                parsed = None
            if parsed is not None and parsed[0].name in {
                "run",
                "resume",
                "worker",
                "proposals analyze",
                "proposals evaluate",
            }:
                model_command = True
                self._terminal_open()
                handed_off = True
                line = "/" + text.lstrip("/!")
        else:
            model_command = True
        if not model_command:
            return False
        added = not line.strip().startswith(("/", "!"))
        user_seq = None
        safe, cut = providers.redact_keys(line, config)
        if added:
            self.session_id = self._add_user(safe)
            user_seq = self.sessions.conn.execute(
                "SELECT MAX(seq) FROM messages WHERE session_id = ?", (self.session_id,)
            ).fetchone()[0]
            self.sessions.register_turn(user_seq)
            if cut:
                self._event(
                    t(
                        "That looked like an API key, so it was cut from your message: keys are "
                        "never kept in the chat or sent to a model. Add a key with `desman keys "
                        "add PROVIDER`."
                    )
                )
        # Queues retain only the redacted input. A key removed from configuration while
        # the turn waits must not reappear when that turn eventually calls a model.
        line = safe
        if handed_off:
            self.sessions.add(self.session_id, Role.USER, safe)
        assert self.session_id is not None
        ui.session(self.session_id, self.temporary)
        self.runners.set_cap(runs.tuned(config.tuning, "chat_turns"))
        self.runners.submit(
            self.session_id, line, temporary=self.temporary, added=added, user_seq=user_seq
        )
        return True
