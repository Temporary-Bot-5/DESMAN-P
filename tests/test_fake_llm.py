"""Tests for the scripted fake provider, and the coordinator driven by LLM-backed agents."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import FakeClock
from desman.core import db
from desman.core.coordinator import Coordinator, Failure, Review, Success, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind, RetryPolicy
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.llm.base import (
    Message,
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
)
from desman.llm.fake import (
    GARBAGE,
    FakeLLM,
    Quota,
    RateLimit,
    Rejected,
    Reply,
    Script,
    ScriptExhausted,
    ServerError,
    Timeout,
    garbage,
)

S = SubtaskStatus


def ask(text: str, **kwargs: Any) -> Request:
    return Request(
        messages=[Message(role="system", content="sys"), Message(role="user", content=text)],
        **kwargs,
    )


# ── The fake on its own ──────────────────────────────────────────────────────


def test_steps_are_served_in_order() -> None:
    llm = FakeLLM("one", Reply(text="two", finish_reason="length"))
    first = llm.complete(ask("a"))
    second = llm.complete(ask("b"))
    assert (first.text, first.finish_reason) == ("one", "stop")
    assert (second.text, second.finish_reason) == ("two", "length")
    assert [r.prompt for r in llm.calls] == ["a", "b"]
    assert llm.remaining == 0
    with pytest.raises(ScriptExhausted, match="script"):
        llm.complete(ask("c"))


def test_completion_reports_provider_model_and_usage() -> None:
    llm = FakeLLM("abcdefgh", "x", name="groq")
    done = llm.complete(ask("12345678"))
    assert (done.provider, done.model) == ("groq", "groq-model")
    assert (done.usage.prompt_tokens, done.usage.completion_tokens) == (3, 2)  # "sys" + 8 chars
    assert llm.complete(ask("q", model="llama-free")).model == "llama-free"


def test_max_tokens_cuts_the_reply() -> None:
    llm = FakeLLM("x" * 100)
    done = llm.complete(ask("q", max_tokens=5))
    assert (done.text, done.finish_reason, done.usage.completion_tokens) == ("x" * 20, "length", 5)


def test_repeat_last_serves_the_final_step_forever() -> None:
    llm = FakeLLM("first", "again", repeat_last=True)
    assert [llm.complete(ask("q")).text for _ in range(4)] == ["first", "again", "again", "again"]


def test_routes_by_prompt_and_by_predicate() -> None:
    llm = (
        FakeLLM("default")
        .on("Review", "accept", repeat_last=True)
        .on(lambda r: r.max_tokens == 7, "short")
        .on("Plan", "plan-1")
    )
    assert llm.complete(ask("Plan the task")).text == "plan-1"
    assert llm.complete(ask("Review it")).text == "accept"
    assert llm.complete(ask("Review it again", max_tokens=7)).text == "accept"  # earlier route wins
    assert llm.complete(ask("other", max_tokens=7)).text == "short"
    assert llm.complete(ask("anything")).text == "default"
    with pytest.raises(ScriptExhausted, match="'Plan'"):
        llm.complete(ask("Plan again"))  # an exhausted route never falls through to the script


def test_responder_steps_see_the_request() -> None:
    llm = FakeLLM(lambda r: f"echo {r.prompt}", lambda r: RateLimit(retry_after=1))
    assert llm.complete(ask("hi")).text == "echo hi"
    with pytest.raises(RateLimited):
        llm.complete(ask("hi"))
    assert [type(step).__name__ for step in llm.served] == ["Reply", "RateLimit"]


def test_responder_returning_a_non_step_is_a_script_bug() -> None:
    llm = FakeLLM(lambda r: {"type": "reply", "text": "x"})  # type: ignore[arg-type, return-value]
    with pytest.raises(TypeError, match="dict"):
        llm.complete(ask("q"))


def test_failure_steps_raise_typed_provider_errors(clock: FakeClock) -> None:
    llm = FakeLLM(
        Timeout(after=30),
        RateLimit(retry_after=12),
        RateLimit(retry_after=None),
        Quota(resets_in=3600),
        Quota(resets_in=None),
        ServerError(status=502),
        Rejected(status=403, message="key revoked"),
        clock=clock,
        sleep=clock.sleep,
        name="p",
    )
    now = clock()
    with pytest.raises(ProviderTimeout) as timeout:
        llm.complete(ask("q"))
    assert timeout.value.kind is FailureKind.TRANSIENT and clock.slept == 30  # the wait is real-ish
    with pytest.raises(RateLimited) as limited:
        llm.complete(ask("q"))
    assert limited.value.kind is FailureKind.LIMIT
    assert limited.value.paused_until(now) == now + timedelta(seconds=12)
    with pytest.raises(RateLimited) as unknown:
        llm.complete(ask("q"))
    assert unknown.value.paused_until(now) is None
    with pytest.raises(QuotaExhausted) as quota:
        llm.complete(ask("q"))
    assert quota.value.kind is FailureKind.LIMIT
    assert quota.value.paused_until(now) == clock() + timedelta(hours=1)  # from the call's time
    with pytest.raises(QuotaExhausted) as no_reset:
        llm.complete(ask("q"))
    assert no_reset.value.resets_at is None
    with pytest.raises(ProviderUnavailable) as down:
        llm.complete(ask("q"))
    assert (down.value.kind, down.value.status) == (FailureKind.TRANSIENT, 502)
    with pytest.raises(ProviderRejected, match="key revoked") as rejected:
        llm.complete(ask("q"))
    assert (rejected.value.kind, rejected.value.status) == (FailureKind.PERMANENT, 403)
    assert str(rejected.value).startswith("p: ")


def test_latency_uses_the_injected_sleep(clock: FakeClock) -> None:
    llm = FakeLLM(Reply(text="slow", latency=2.5), sleep=clock.sleep)
    llm.complete(ask("q"))
    assert clock.slept == 2.5


def test_garbage_is_never_a_json_object() -> None:
    # What 2.3 has to cope with: none of these parses straight into the dict we asked for.
    for name, text in GARBAGE.items():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        assert not isinstance(parsed, dict), name
    assert garbage("truncated").finish_reason == "length"
    assert garbage("fenced").finish_reason == "stop"


def test_script_exhaustion_is_not_an_ordinary_exception() -> None:
    assert not issubclass(ScriptExhausted, Exception)


def test_scripts_load_from_json(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            {
                "steps": ["plain", {"type": "rate_limit", "retry_after": 5}],
                "routes": [{"match": "Review", "steps": [{"type": "reply", "text": "accept"}]}],
            }
        )
    )
    llm = FakeLLM.from_file(path, clock=clock)
    assert llm.complete(ask("Review")).text == "accept"
    assert llm.complete(ask("x")).text == "plain"
    with pytest.raises(RateLimited):
        llm.complete(ask("x"))
    assert FakeLLM.from_script(Script(steps=["a"], repeat_last=True)).complete(ask("x")).text == "a"


@pytest.mark.parametrize(
    "bad",
    [
        '{"steps": [{"type": "explode"}]}',
        '{"steps": [{"type": "rejected", "status": 429}]}',
        '{"steps": [{"type": "server_error", "status": 404}]}',
        '{"steps": [{"type": "timeout", "after": -1}]}',
        '{"steps": [], "extra": 1}',
        '{"steps": [{"type": "reply"}]}',
    ],
)
def test_bad_scripts_are_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        FakeLLM.from_script(bad)


def test_chaos_is_deterministic_per_seed() -> None:
    def run(seed: int) -> list[str]:
        llm = FakeLLM.chaos(seed, "ok")
        outcomes = []
        for _ in range(200):
            try:
                outcomes.append(llm.complete(ask("q")).text)
            except ProviderError as e:
                outcomes.append(type(e).__name__)
        return outcomes

    assert run(1) == run(1)
    assert run(1) != run(2)
    kinds = set(run(3))
    assert {"ok", "ProviderTimeout", "RateLimited", "ProviderUnavailable"} <= kinds
    assert kinds & set(GARBAGE.values())


def test_chaos_rates_are_checked() -> None:
    with pytest.raises(ValueError):
        FakeLLM.chaos(0, "ok", garbage_rate=0.6, timeout_rate=0.6)
    with pytest.raises(ValueError):
        FakeLLM.chaos(0, "ok", timeout_rate=-0.1)
    quiet = FakeLLM.chaos(
        0, "ok", garbage_rate=0, timeout_rate=0, rate_limit_rate=0, server_error_rate=0
    )
    assert {quiet.complete(ask("q")).text for _ in range(50)} == {"ok"}


def test_request_contract() -> None:
    with pytest.raises(ValidationError):
        Request(messages=[])
    assert Request(messages=[Message(role="system", content="only")]).prompt == ""
    with pytest.raises(ValueError, match="aware"):
        QuotaExhausted("p", datetime(2026, 1, 1))  # noqa: DTZ001


# ── Coordinator with LLM-backed agents ───────────────────────────────────────
#
# A deliberately naive pair of agents: the real ones (Phase 2) parse better, but the point here
# is that every provider behaviour ends in the right state through the real loop.


def agents(llm: FakeLLM) -> tuple[Any, Any]:
    def executor(work: Work) -> Success | Failure:
        feedback = work.context.feedback.text if work.context.feedback else ""
        reply = llm.complete(ask(f"Execute: {work.subtask.description}\n{feedback}"))
        try:
            content = json.loads(reply.text)
        except json.JSONDecodeError:
            content = None
        if not isinstance(content, dict):
            return Failure(
                kind=FailureKind.QUALITY, reason=f"not a JSON object: {reply.text[:40]!r}"
            )
        return Success(content=content)

    def reviewer(subtask: Subtask, artifact: dict[str, Any] | None) -> Review:
        reply = llm.complete(ask(f"Review: {subtask.description}: {artifact}"))
        return Review.model_validate({"verdict": reply.text.strip(), "feedback": "llm said so"})

    return executor, reviewer


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="ok")])


def coordinator(store: Store, clock: FakeClock, llm: FakeLLM, **kwargs: Any) -> Coordinator:
    executor, reviewer = agents(llm)
    return Coordinator(store, executor, reviewer, sleep=clock.sleep, **kwargs)


def make(*steps: Any, clock: FakeClock, review: tuple[Any, ...] = ("accept",)) -> FakeLLM:
    return (
        FakeLLM(clock=clock, sleep=clock.sleep)
        .on("Execute", *steps)
        .on("Review", *review, repeat_last=True)
    )


def test_garbage_reply_is_retried_until_valid(store: Store, clock: FakeClock) -> None:
    llm = make(garbage("fenced"), garbage("truncated"), '{"answer": 42}', clock=clock)
    task_id = store.create_task(one_step())
    result = coordinator(store, clock, llm).run(task_id)
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert store.latest_artifact(sub.id) == {"answer": 42}
    assert sub.attempts.quality == 2
    assert llm.remaining == 0


def test_rate_limit_pauses_without_spending_an_attempt(store: Store, clock: FakeClock) -> None:
    llm = make(RateLimit(retry_after=30), '{"ok": 1}', clock=clock)
    task_id = store.create_task(one_step())
    start = clock()
    paused = coordinator(store, clock, llm).run(task_id)
    assert paused.status is TaskStatus.ACTIVE
    assert paused.paused_until == db.to_iso(start + timedelta(seconds=30))
    [sub] = store.list_subtasks(task_id)
    assert sub.status is S.PAUSED
    assert (sub.attempts.quality, sub.attempts.transient) == (0, 0)
    done = coordinator(store, clock, llm).run(task_id, wait_for_pauses=True)
    assert done.status is TaskStatus.DONE
    assert clock() >= start + timedelta(seconds=30)


def test_exhausted_quota_pauses_until_it_resets(store: Store, clock: FakeClock) -> None:
    llm = make(Quota(resets_in=3600), '{"ok": 1}', clock=clock)
    task_id = store.create_task(one_step())
    start = clock()
    paused = coordinator(store, clock, llm).run(task_id)
    assert paused.paused_until == db.to_iso(start + timedelta(hours=1))


def test_timeouts_and_server_errors_are_transient(store: Store, clock: FakeClock) -> None:
    llm = make(
        Timeout(after=30), ServerError(status=502), Timeout(after=30), '{"ok": 1}', clock=clock
    )
    task_id = store.create_task(one_step())
    result = coordinator(store, clock, llm).run(task_id)
    assert result.status is TaskStatus.DONE
    assert clock.slept >= 60  # the fake's timeouts took (fake) time
    assert llm.remaining == 0


def test_persistent_outage_ends_with_the_human(store: Store, clock: FakeClock) -> None:
    llm = FakeLLM(ServerError(), repeat_last=True, clock=clock, sleep=clock.sleep)
    task_id = store.create_task(one_step())
    policy = RetryPolicy(max_transient=3)
    result = coordinator(store, clock, llm, policy=policy).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    [sub] = store.list_subtasks(task_id)
    assert sub.status is S.NEEDS_HUMAN
    assert len(llm.calls) == 3


def test_rejected_request_goes_straight_to_the_human(store: Store, clock: FakeClock) -> None:
    llm = make(Rejected(status=401, message="bad key"), clock=clock)
    task_id = store.create_task(one_step())
    result = coordinator(store, clock, llm).run(task_id)
    assert result.status is TaskStatus.NEEDS_HUMAN
    assert len(llm.calls) == 1  # no retry can fix a bad key


def test_rate_limited_reviewer_reviews_later_without_rerunning(
    store: Store, clock: FakeClock
) -> None:
    llm = make('{"ok": 1}', clock=clock, review=(RateLimit(retry_after=10), "accept"))
    task_id = store.create_task(one_step())
    start = clock()
    paused = coordinator(store, clock, llm).run(task_id)
    assert paused.paused_until == db.to_iso(start + timedelta(seconds=10))
    [sub] = store.list_subtasks(task_id)
    assert sub.status is S.PAUSED and sub.attempts.transient == 0
    result = coordinator(store, clock, llm).run(task_id, wait_for_pauses=True)
    assert result.status is TaskStatus.DONE
    executions = [r for r in llm.calls if r.prompt.startswith("Execute")]
    assert len(executions) == 1


def test_garbage_verdict_is_not_taken_as_a_verdict(store: Store, clock: FakeClock) -> None:
    llm = make('{"ok": 1}', clock=clock, review=(garbage("prose"), "accept"))
    task_id = store.create_task(one_step())
    result = coordinator(store, clock, llm).run(task_id)
    assert result.status is TaskStatus.DONE
    [sub] = store.list_subtasks(task_id)
    assert sub.attempts.quality == 0  # the work was fine; only the review was retried


def test_script_bug_fails_the_test_instead_of_looking_like_an_outage(
    store: Store, clock: FakeClock
) -> None:
    llm = make(clock=clock)  # no Execute steps at all
    task_id = store.create_task(one_step())
    with pytest.raises(ScriptExhausted):
        coordinator(store, clock, llm).run(task_id)


def test_json_script_drives_the_loop(store: Store, clock: FakeClock, tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "routes": [
                    {"match": "Execute", "steps": [{"type": "timeout"}, '{"x": 1}']},
                    {"match": "Review", "steps": ["accept"], "repeat_last": True},
                ]
            }
        )
    )
    llm = FakeLLM.from_file(path, clock=clock, sleep=clock.sleep)
    task_id = store.create_task(one_step())
    assert coordinator(store, clock, llm).run(task_id).status is TaskStatus.DONE


@pytest.mark.parametrize("seed", range(60))
def test_chaos_always_terminates_and_never_fails_on_its_own(
    store: Store, clock: FakeClock, seed: int
) -> None:
    def reply(request: Request) -> str:
        return "accept" if request.prompt.startswith("Review") else '{"ok": true}'

    llm = FakeLLM.chaos(seed, reply, garbage_rate=0.2, clock=clock, sleep=clock.sleep)
    spec = TaskSpec(
        goal="g",
        subtasks=[
            SubtaskSpec(description="a", acceptance_criteria="ok"),
            SubtaskSpec(description="b", acceptance_criteria="ok", depends_on=[0]),
            SubtaskSpec(description="c", acceptance_criteria="ok", depends_on=[0]),
        ],
    )
    task_id = store.create_task(spec)
    result = coordinator(store, clock, llm).run(task_id, wait_for_pauses=True, max_ticks=2000)
    assert result.status in (TaskStatus.DONE, TaskStatus.NEEDS_HUMAN)
    statuses = {s.status for s in store.list_subtasks(task_id)}
    assert S.FAILED not in statuses and S.RUNNING not in statuses


def test_fake_clock_is_aware() -> None:
    assert FakeLLM().clock().tzinfo is UTC
