"""Tests for varied retries: the attempt number picks how the attempt differs (ROADMAP 2.9)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.agents import (
    EXECUTOR_SYSTEM,
    EXECUTOR_SYSTEM_REPHRASED,
    REPHRASE_TEMPERATURE,
    llm_executor,
    llm_reviewer,
)
from desman.cli import EXIT_DONE, cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.coordinator import Coordinator
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.retry import RetryPolicy, Strategy, strategy_for
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.llm import router
from desman.llm.fake import FakeLLM
from helpers import make_config

S = Strategy
REJECT = '{"verdict": "retry", "feedback": "1. wrong"}'
ACCEPT = '{"verdict": "accept"}'


def test_each_attempt_number_has_its_strategy() -> None:
    assert [strategy_for(n) for n in range(1, 7)] == [
        S.FIRST, S.FEEDBACK, S.REPHRASE, S.OTHER_MODEL, S.OTHER_MODEL, S.OTHER_MODEL
    ]  # fmt: skip
    # Nothing to switch to: rephrase again rather than claim another model was used.
    assert [strategy_for(n, other_model=False) for n in range(1, 6)] == [
        S.FIRST, S.FEEDBACK, S.REPHRASE, S.REPHRASE, S.REPHRASE
    ]  # fmt: skip


def one_step() -> TaskSpec:
    return TaskSpec(goal="g", subtasks=[SubtaskSpec(description="a", acceptance_criteria="c")])


def run(store: Store, clock: FakeClock, main: FakeLLM, other: FakeLLM | None, **kw: Any) -> str:
    task_id = store.create_task(one_step())
    coordinator = Coordinator(
        store,
        llm_executor(main, other),
        llm_reviewer(main),
        sleep=clock.sleep,
        strategies=lambda n: strategy_for(n, other_model=other is not None),
        **kw,
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    return task_id


def executions(llm: FakeLLM) -> list[Any]:
    return [r for r in llm.calls if r.prompt.startswith("Execute")]


def test_every_attempt_differs_from_the_one_before(store: Store, clock: FakeClock) -> None:
    main = FakeLLM().on("Execute", '{"v": 1}', '{"v": 2}', '{"v": 3}')
    main.on("Review", REJECT, REJECT, REJECT, ACCEPT)
    other = FakeLLM().on("Execute", '{"v": 4}')
    task_id = run(store, clock, main, other)

    first, feedback, rephrased = executions(main)
    assert first.messages[0].content == feedback.messages[0].content == EXECUTOR_SYSTEM
    assert first.temperature == feedback.temperature == 0.0
    assert json.loads(feedback.prompt.partition(":\n")[2])["feedback_on_previous_attempt"]
    assert rephrased.messages[0].content == EXECUTOR_SYSTEM_REPHRASED
    assert rephrased.temperature == REPHRASE_TEMPERATURE
    [switched] = executions(other)  # the fourth attempt went to the other model
    assert "1. wrong" in switched.prompt
    assert len(executions(main)) == 3

    started = [e.details for e in history(store.conn, task_id) if e.action == "attempt_started"]
    assert started == [
        {"attempt": 1, "strategy": "first"},
        {"attempt": 2, "strategy": "feedback"},
        {"attempt": 3, "strategy": "rephrase"},
        {"attempt": 4, "strategy": "other_model"},
    ]
    assert check_history(store.conn, task_id) == []


def test_without_another_model_the_audit_says_rephrase(store: Store, clock: FakeClock) -> None:
    main = FakeLLM().on("Execute", '{"v": 1}', repeat_last=True)
    main.on("Review", REJECT, REJECT, REJECT, ACCEPT)
    task_id = run(store, clock, main, None)
    assert len(executions(main)) == 4
    assert executions(main)[3].messages[0].content == EXECUTOR_SYSTEM_REPHRASED
    started = [
        e.details["strategy"] for e in history(store.conn, task_id) if e.action == "attempt_started"
    ]
    assert started == ["first", "feedback", "rephrase", "rephrase"]


def test_a_transient_retry_keeps_the_strategy(store: Store, clock: FakeClock) -> None:
    # An interrupted attempt is not a new attempt: same number, same strategy.
    from desman.llm.fake import ServerError

    main = FakeLLM().on("Execute", ServerError(status=503), '{"v": 1}')
    main.on("Review", ACCEPT)
    task_id = run(store, clock, main, None, policy=RetryPolicy())
    started = [e.details for e in history(store.conn, task_id) if e.action == "attempt_started"]
    assert started == [{"attempt": 1, "strategy": "first"}] * 2


# ── Another provider ─────────────────────────────────────────────────────────

GROQ = "gsk_" + "a" * 48
OPENROUTER = "sk-or-" + "b" * 48


def test_the_alternative_is_the_next_ready_free_provider() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "OPENROUTER_API_KEY": OPENROUTER})
    _, groq = router.build(config)
    found = router.alternative(config, groq)
    assert found is not None and found[1].provider.id == "openrouter"
    _, openrouter = router.build(config, "openrouter")
    found = router.alternative(config, openrouter)
    assert found is not None and found[1].provider.id == "groq"
    assert router.alternative(make_config(api_keys={"GROQ_API_KEY": GROQ}), groq) is None


def test_a_paid_provider_is_never_the_alternative() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "DEEPSEEK_API_KEY": "sk-" + "c" * 40})
    _, groq = router.build(config)
    assert router.alternative(config, groq) is None


@pytest.mark.parametrize("pinned", [[], ["--provider", "groq"]])
def test_run_switches_provider_for_the_fourth_attempt(
    monkeypatch: pytest.MonkeyPatch, pinned: list[str]
) -> None:
    ConfigManager(default_config_path()).save(
        make_config(api_keys={"GROQ_API_KEY": GROQ, "OPENROUTER_API_KEY": OPENROUTER})
    )
    executed_on: list[str] = []
    reviews = iter([REJECT, REJECT, REJECT, ACCEPT])

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        if prompt.startswith("Execute"):
            executed_on.append(request.url.host)
        # Each result differs: the same result reviewed again would get the cached verdict.
        text = next(reviews) if prompt.startswith("Review") else f'{{"v": {len(executed_on)}}}'
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    result = CliRunner().invoke(cli, ["run", "g", "--plan", "template", *pinned])
    assert result.exit_code == EXIT_DONE, result.output
    if pinned:  # the operator chose groq: every attempt stays there
        assert "Retries may also use" not in result.output
        assert executed_on == ["api.groq.com"] * 4
    else:
        assert "Retries may also use OpenRouter" in result.output
        assert executed_on == ["api.groq.com"] * 3 + ["openrouter.ai"]
    conn = db.open_db(db.default_db_path())
    try:
        strategies = [
            json.loads(r[0])["strategy"]
            for r in conn.execute(
                "SELECT details_json FROM audit_log WHERE action = 'attempt_started' ORDER BY seq"
            )
        ]
    finally:
        conn.close()
    assert strategies[-1] == ("rephrase" if pinned else "other_model")


def test_a_coordinator_without_another_model_never_claims_a_switch(
    store: Store, clock: FakeClock
) -> None:
    main = FakeLLM().on("Execute", '{"v": 1}', '{"v": 2}', '{"v": 3}', '{"v": 4}')
    main.on("Review", REJECT, REJECT, REJECT, ACCEPT)
    task_id = store.create_task(one_step())
    # The defaults: no strategies given, no other LLM.
    coordinator = Coordinator(store, llm_executor(main), llm_reviewer(main), sleep=clock.sleep)
    assert coordinator.run(task_id).status is TaskStatus.DONE
    started = [
        e.details["strategy"] for e in history(store.conn, task_id) if e.action == "attempt_started"
    ]
    assert started == ["first", "feedback", "rephrase", "rephrase"]
    assert executions(main)[3].messages[0].content == EXECUTOR_SYSTEM_REPHRASED


def test_an_unusable_reply_is_feedback_for_the_next_attempt(store: Store, clock: FakeClock) -> None:
    from desman.llm.fake import ServerError, garbage

    prose = garbage("prose")
    main = FakeLLM().on("Execute", prose, prose, ServerError(status=503), '{"v": 1}')
    main.on("Review", ACCEPT)
    task_id = store.create_task(one_step())
    coordinator = Coordinator(store, llm_executor(main), llm_reviewer(main), sleep=clock.sleep)
    assert coordinator.run(task_id).status is TaskStatus.DONE
    second_try = json.loads(executions(main)[-1].prompt.partition(":\n")[2])
    # The interruption in between is not feedback; the unusable reply before it is.
    assert second_try["feedback_on_previous_attempt"] == (
        "unusable reply: The reply is not a JSON object."
    )
    assert second_try["attempt"] == 2
