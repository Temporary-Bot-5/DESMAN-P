"""T.6g: single-key limits wait after routing, without holding task leases."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import FakeClock
from desman import provider_wait, runs
from desman.chat import Chat
from desman.config import ConfigManager, default_config_path
from desman.core.coordinator import Coordinator, Review, Success, Verdict, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.sessions import Role, Sessions
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.interlocutor import CHAT
from desman.llm.base import RateLimited
from desman.llm.fake import FakeLLM, RateLimit
from desman.planner import plan
from helpers import make_config

PLAN = json.dumps({"subtasks": [{"description": "Write a poem", "acceptance_criteria": "A poem"}]})


def test_planner_waits_ten_minutes_then_uses_model(
    clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    llm = FakeLLM(RateLimit(retry_after=600), PLAN)
    with provider_wait.waiting(sleep=clock.sleep):
        result = plan(llm, "Write a poem")
    assert result.source == "llm"
    assert clock.slept == 600
    assert "Waiting 600 s" in capsys.readouterr().out


def test_planner_long_limit_falls_back_and_shows_reset(
    capsys: pytest.CaptureFixture[str], clock: FakeClock
) -> None:
    with provider_wait.waiting(sleep=clock.sleep):
        result = plan(FakeLLM(RateLimit(retry_after=901)), "Write a poem")
    assert result.source == "template"
    assert clock.slept == 0
    assert "Provider limit until" in capsys.readouterr().out


def test_planner_interrupt_does_not_create_template() -> None:
    def stop(seconds: float) -> None:
        raise KeyboardInterrupt

    with provider_wait.waiting(sleep=stop), pytest.raises(KeyboardInterrupt):
        plan(FakeLLM(RateLimit(retry_after=600)), "Write a poem")


def test_waits_have_a_total_bound(clock: FakeClock) -> None:
    with provider_wait.waiting(sleep=clock.sleep):
        result = plan(FakeLLM(RateLimit(retry_after=600), RateLimit(retry_after=600)), "poem")
    assert result.source == "template"
    assert clock.slept == 600


@pytest.mark.parametrize("seconds", [600, 900, 901])
def test_task_wait_bound_releases_lease(store: Store, clock: FakeClock, seconds: int) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="poem", subtasks=[SubtaskSpec(description="poem", acceptance_criteria="done")]
        )
    )
    attempts = 0

    def execute(work: Work) -> Success:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited("only-key", seconds)
        return Success(content={"text": "A poem"})

    def review(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        return Review(verdict=Verdict.ACCEPT)

    def sleep(delay: float) -> None:
        [subtask] = store.list_subtasks(task_id)
        assert subtask.status is SubtaskStatus.PAUSED
        assert subtask.lease_owner is None
        clock.sleep(delay)

    coordinator = Coordinator(store, execute, review, sleep=sleep)
    result = coordinator.run(task_id, short_pause_seconds=provider_wait.PROVIDER_WAIT)
    if seconds <= 900:
        assert result.status is TaskStatus.DONE
        assert clock.slept == seconds
    else:
        assert result.paused_until is not None
        assert clock.slept == 0


@pytest.mark.parametrize("interrupt", [False, True])
def test_chat_waits_and_keeps_message_on_interrupt(
    store: Store, clock: FakeClock, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    llm = FakeLLM().on(CHAT, RateLimit(retry_after=600), '{"type":"reply","text":"Hello again"}')
    monkeypatch.setattr(runs, "models", lambda *args, **kwargs: runs.Models(llm, None))

    def sleep(seconds: float) -> None:
        if interrupt:
            raise KeyboardInterrupt
        clock.sleep(seconds)

    sessions = Sessions(store.conn, clock=clock)
    chat = Chat(sessions, sleep=sleep)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            chat._message("hello")
    else:
        chat._message("hello")
    assert chat.session_id is not None
    messages = sessions.messages(chat.session_id)
    assert messages[0].role is Role.USER and messages[0].text == "hello"
    assert any(m.text == "Hello again" for m in messages) is not interrupt
    assert clock.slept == (0 if interrupt else 600)


def test_untried_model_answers_before_wait(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.llm.router import Fallback

    limited = FakeLLM(name="limited")

    def slow_limit(request: Any) -> Any:
        error = RateLimited("limited", 600)
        error.slow = True
        raise error

    monkeypatch.setattr(limited, "complete", slow_limit)
    good = FakeLLM(PLAN, name="good")
    chain = Fallback([(limited, None), (good, None)])
    with provider_wait.waiting(sleep=clock.sleep):
        result = plan(chain, "Write a poem")
    assert result.source == "llm"
    assert clock.slept == 0
    assert len(good.calls) == 1


def test_known_quota_reset_waits(clock: FakeClock) -> None:
    from desman.llm.fake import Quota

    llm = FakeLLM(Quota(resets_in=600), PLAN, clock=clock)
    with provider_wait.waiting(sleep=clock.sleep, clock=clock):
        result = plan(llm, "Write a poem")
    assert result.source == "llm"
    assert clock.slept == 600


def test_wait_scope_does_not_leak(clock: FakeClock) -> None:
    with provider_wait.waiting(sleep=clock.sleep):
        pass
    result = plan(FakeLLM(RateLimit(retry_after=600)), "Write a poem")
    assert result.source == "template"
    assert clock.slept == 0


def test_task_interrupt_leaves_paused_work_resumable(store: Store) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="poem", subtasks=[SubtaskSpec(description="poem", acceptance_criteria="done")]
        )
    )

    def execute(work: Work) -> Success:
        raise RateLimited("only-key", 600)

    def stop(seconds: float) -> None:
        raise KeyboardInterrupt

    coordinator = Coordinator(store, execute, lambda *_: Review(verdict=Verdict.ACCEPT), sleep=stop)
    with pytest.raises(KeyboardInterrupt):
        coordinator.run(task_id, short_pause_seconds=provider_wait.PROVIDER_WAIT)
    [subtask] = store.list_subtasks(task_id)
    assert subtask.status is SubtaskStatus.PAUSED
    assert subtask.lease_owner is None


def test_cancelled_turn_does_not_call_model_again_after_wait(clock: FakeClock) -> None:
    from desman.llm.structured import completion_guard

    stopped = False

    def check() -> None:
        if stopped:
            raise RuntimeError("turn stopped")

    def sleep(seconds: float) -> None:
        nonlocal stopped
        clock.sleep(seconds)
        stopped = True

    llm = FakeLLM(RateLimit(retry_after=600), PLAN)
    with (
        completion_guard(check),
        provider_wait.waiting(sleep=sleep),
        pytest.raises(RuntimeError, match="turn stopped"),
    ):
        plan(llm, "Write a poem")
    assert len(llm.calls) == 1


def test_cli_planning_enables_wait(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> None:
    from functools import partial

    from desman.cli import _plan

    monkeypatch.setattr(
        provider_wait, "waiting", partial(provider_wait.waiting, sleep=clock.sleep, clock=clock)
    )
    result = _plan(
        FakeLLM(RateLimit(retry_after=600), PLAN), "Write a poem", None, by_model=True, as_json=True
    )
    assert result.source == "llm"
    assert clock.slept == 600


def test_retelling_waits_without_losing_result(store: Store, clock: FakeClock) -> None:
    from desman.interlocutor import RETELL, Interlocutor

    task_id = store.create_task(
        TaskSpec(
            goal="poem", subtasks=[SubtaskSpec(description="poem", acceptance_criteria="done")]
        )
    )
    coordinator = Coordinator(
        store,
        lambda _: Success(content={"text": "A poem"}),
        lambda *_: Review(verdict=Verdict.ACCEPT),
    )
    coordinator.run(task_id)
    sessions = Sessions(store.conn, clock=clock)
    chat = Chat(sessions, sleep=clock.sleep)
    chat.session_id = sessions.add(None, Role.USER, "Write a poem").session_id
    llm = FakeLLM().on(RETELL, RateLimit(retry_after=600), '{"text":"A poem"}')
    chat._retell_outcome(task_id, store.short_id(task_id), Interlocutor(llm, "en"))
    assert clock.slept == 600
    assert any(m.text == "A poem" for m in sessions.messages(chat.session_id))


def test_compaction_waits_between_attempts(
    store: Store, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions = Sessions(store.conn, clock=clock)
    chat = Chat(sessions, sleep=clock.sleep)
    chat.session_id = sessions.add(None, Role.USER, "Please remember my preferences.").session_id
    llm = (
        FakeLLM()
        .on(
            "Execute subtask",
            RateLimit(retry_after=600),
            '{"summary":"The person wants preferences remembered."}',
        )
        .on("Review subtask", '{"verdict":"accept"}')
    )
    original = runs.summarizer

    def summarizer(*args: Any, **kwargs: Any) -> Coordinator:
        result = original(*args, **kwargs)
        result.sleep = clock.sleep
        return result

    monkeypatch.setattr(runs, "summarizer", summarizer)
    summary = chat._compact(
        sessions.history(chat.session_id), runs.Models(llm, None), keep_tokens=0
    )
    assert summary == "The person wants preferences remembered."
    assert clock.slept == 600


@pytest.mark.parametrize("seconds", [600, 901])
def test_terminal_chat_task_waits_or_reports_long_pause(
    store: Store,
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seconds: int,
) -> None:
    from desman.interlocutor import RETELL

    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    llm = (
        FakeLLM()
        .on(CHAT, '{"type":"task","title":"Poem","goal":"Write a poem"}')
        .on("Plan task", PLAN)
        .on(
            "Execute subtask",
            RateLimit(retry_after=seconds),
            '{"text":"A short poem about birds."}',
        )
        .on("Review subtask", '{"verdict":"accept"}')
        .on(RETELL, '{"text":"The task status is shown above."}')
    )
    monkeypatch.setattr(runs, "models", lambda *args, **kwargs: runs.Models(llm, None))
    original = runs.coordinator

    def coordinator(*args: Any, **kwargs: Any) -> Coordinator:
        loop = original(*args, **kwargs)
        loop.sleep = clock.sleep
        return loop

    monkeypatch.setattr(runs, "coordinator", coordinator)
    chat = Chat(Sessions(store.conn, clock=clock), sleep=clock.sleep)
    chat._message("Write a poem")
    row = store.conn.execute("SELECT id FROM tasks").fetchone()
    assert row is not None
    output = capsys.readouterr().out
    if seconds == 600:
        assert store.task_status(row[0]) is TaskStatus.DONE
        assert clock.slept == 600
        assert "Waiting 600 s" in output
    else:
        assert store.list_subtasks(row[0])[0].status is SubtaskStatus.PAUSED
        assert clock.slept == 0
        assert "paused by a provider limit until" in output
        assert "`desman run --wait` or `desman worker`" in output


def test_all_slow_models_limited_wait_once(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman.llm.router import Fallback

    limited = FakeLLM(name="limited")

    def slow_limit(request: Any) -> Any:
        error = RateLimited("limited", 600)
        error.slow = True
        raise error

    monkeypatch.setattr(limited, "complete", slow_limit)
    recovers = FakeLLM(RateLimit(retry_after=600), PLAN, name="recovers")
    chain = Fallback([(limited, None), (recovers, None)])
    with provider_wait.waiting(sleep=clock.sleep):
        result = plan(chain, "Write a poem")
    assert result.source == "llm"
    assert clock.slept == 600
