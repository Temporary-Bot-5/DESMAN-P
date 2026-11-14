"""Models per role, the fallback lineup and the rate limiter in the database (ROADMAP 4.5)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import RootModel

from conftest import FakeClock
from desman import providers, runs
from desman.agents import llm_executor, llm_reviewer
from desman.core import db
from desman.core.audit import check_history, history
from desman.core.calls import Role
from desman.core.coordinator import Coordinator
from desman.core.models import Kind, SubtaskSpec, TaskSpec
from desman.core.report import task_report
from desman.core.retry import strategy_for
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.llm import catalog, router
from desman.llm.base import (
    LLM,
    GenerationFailed,
    Message,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
    among,
    noticing_unfit,
)
from desman.llm.cache import Cached
from desman.llm.fake import (
    FakeLLM,
    Quota,
    RateLimit,
    Rejected,
    ServerError,
    Timeout,
    ToolUseFailed,
)
from desman.llm.limiter import Buckets, Limited
from desman.llm.structured import complete_json
from helpers import make_config

GROQ = "gsk_" + "a" * 48
GOOGLE = "AIza" + "b" * 35
OPENROUTER = "sk-or-" + "c" * 48
DEEPSEEK = "sk-" + "d" * 40
MISTRAL = "m" * 32
REQUEST = Request(messages=[Message(role="user", content="hi")])


def labels(choices: list[router.Choice]) -> list[str]:
    return [c.label for c in choices]


# ── The catalog ──────────────────────────────────────────────────────────────


def test_every_cloud_provider_is_in_the_catalog_with_its_default_model_first() -> None:
    for p in providers.CLOUD:
        models = catalog.CATALOG[p.id]
        assert models[0].id == p.default_model
        assert len({m.id for m in models}) == len(models)
        assert all(m.rpm > 0 and m.context > 0 and m.strengths for m in models)
    assert set(catalog.CATALOG) == {p.id for p in providers.CLOUD}


def test_every_role_says_what_it_needs() -> None:
    assert set(catalog.NEEDS) == set(Role)


def test_a_model_the_catalog_does_not_know_gets_cautious_defaults() -> None:
    groq = providers.find("groq")
    unknown = catalog.model(groq, "some-new-model")
    assert unknown.id == "some-new-model" and not unknown.strict_json
    assert unknown.rpm == catalog.models_of(groq)[0].rpm


# ── Lineups ──────────────────────────────────────────────────────────────────


def test_each_role_asks_the_model_best_for_it_first() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    executor = labels(router.lineup(config, Role.EXECUTOR))
    reviewer = labels(router.lineup(config, Role.REVIEWER))
    assert executor[0] == "groq/openai/gpt-oss-20b"  # the preferred provider's default
    # Judging needs analysis: a model strong at it, even of a provider later in preference,
    # comes before one that is not.
    assert reviewer[:3] == [
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.8-27b",
        "google/gemini-3.8-flash",
    ]
    assert labels(router.lineup(config, Role.PLANNER)) == reviewer
    assert sorted(executor) == sorted(reviewer)  # the same models, in another order
    assert executor.index("groq/qwen/qwen3.8-27b") < executor.index(
        "google/gemini-3.5-flash-lite"
    )  # with the need met by both, the provider's place decides


def test_a_model_that_keeps_to_json_comes_before_one_that_does_not() -> None:
    config = make_config(
        api_keys={"CLOUDFLARE_API_TOKEN": "t" * 40}, settings={"CLOUDFLARE_ACCOUNT_ID": "acc"}
    )
    assert labels(router.lineup(config, Role.EXECUTOR)) == [
        "cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "cloudflare/@cf/meta/llama-3.1-8b-instruct",
    ]


def test_provider_and_model_narrow_the_lineup() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    google = labels(router.lineup(config, Role.REVIEWER, "google"))
    assert google == ["google/gemini-3.8-flash", "google/gemini-3.5-flash-lite"]
    pinned = router.lineup(config, Role.REVIEWER, "google", "gemini-x")
    assert labels(pinned) == ["google/gemini-x"]
    assert labels(router.lineup(config, Role.PLANNER, model="m")) == ["groq/m"]


def test_a_paid_provider_is_in_a_lineup_only_by_name() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "DEEPSEEK_API_KEY": DEEPSEEK})
    for role in Role:
        assert not any(c.provider.paid for c in router.lineup(config, role))
    assert labels(router.lineup(config, Role.EXECUTOR, "deepseek")) == ["deepseek/deepseek-flash"]
    with pytest.raises(router.NoProvider, match="never picks a paid one"):
        router.lineup(make_config(api_keys={"DEEPSEEK_API_KEY": DEEPSEEK}), Role.EXECUTOR)


# ── Fallback ─────────────────────────────────────────────────────────────────


def members(*llms: LLM) -> list[tuple[LLM, router.Choice | None]]:
    return [(llm, None) for llm in llms]


def test_a_limited_model_hands_the_call_to_the_next_one() -> None:
    first = FakeLLM(RateLimit(retry_after=600), name="a")
    second = FakeLLM("ok", name="b")
    lineup = router.Fallback(members(first, second))
    assert lineup.complete(REQUEST).text == "ok"
    assert lineup.name == lineup.label == "a"


@pytest.mark.parametrize(
    "trouble", [ServerError(), Rejected(status=401), Quota()], ids=["down", "key refused", "quota"]
)
def test_any_provider_trouble_moves_the_call_on(trouble: ServerError | Rejected | Quota) -> None:
    lineup = router.Fallback(members(FakeLLM(trouble, name="a"), FakeLLM("ok", name="b")))
    assert lineup.complete(REQUEST).provider == "b"


def test_a_model_unable_to_answer_hands_the_call_on_and_is_not_asked_again() -> None:
    """A native tool call where text was asked (gpt-oss on Groq) is the model's quirk, not the
    work's fault: the next model answers within the same attempt, and the lineup keeps away
    from the unfit one; whoever listens hears of it, to keep it in the task's history."""
    first = FakeLLM(ToolUseFailed(), "never asked again", name="a")
    second = FakeLLM("ok", "ok again", name="b")
    lineup = router.Fallback(members(first, second))
    heard: list[str] = []
    with noticing_unfit(heard.append):
        assert lineup.complete(REQUEST).provider == "b"
        assert lineup.complete(REQUEST).provider == "b"
    assert len(first.calls) == 1 and heard == ["a"]


def test_the_last_model_s_own_failure_is_left_to_the_retry_policy() -> None:
    lineup = router.Fallback(members(FakeLLM(ToolUseFailed(), name="a")))
    with pytest.raises(GenerationFailed):
        lineup.complete(REQUEST)
    both = router.Fallback(
        members(FakeLLM(ToolUseFailed(), name="a"), FakeLLM(ToolUseFailed(), name="b"))
    )
    with pytest.raises(GenerationFailed):  # the one left is unfit too: nothing is hidden
        both.complete(REQUEST)


def test_when_every_model_fails_a_limit_wins_and_the_soonest_one() -> None:
    lineup = router.Fallback(
        members(
            FakeLLM(ServerError(), name="a"),
            FakeLLM(RateLimit(retry_after=3600), name="b"),
            FakeLLM(RateLimit(retry_after=60), name="c"),
        )
    )
    with pytest.raises(RateLimited) as raised:
        lineup.complete(REQUEST)
    assert raised.value.provider == "c"
    down = router.Fallback(
        members(FakeLLM(Rejected(status=401), name="a"), FakeLLM(ServerError(), name="b"))
    )
    with pytest.raises(ProviderUnavailable):  # passing: retried, not handed to a person
        down.complete(REQUEST)


def backed(llm: FakeLLM, clock: FakeClock) -> router.Backoff:
    return router.Backoff(
        llm, sleep=clock.sleep, jitter=lambda: 1.0, monotonic=lambda: clock().timestamp()
    )


def test_a_slow_failure_is_not_handed_on(clock: FakeClock) -> None:
    """A timeout took most of what the lease allows: the next model could outlast it."""
    second = FakeLLM("ok", name="b")
    slow = backed(FakeLLM(Timeout(after=60), name="a", sleep=clock.sleep), clock)
    with pytest.raises(ProviderTimeout):
        router.Fallback(members(slow, second)).complete(REQUEST)
    assert second.calls == []


def test_quick_failures_are_handed_on_however_long_they_took_together(clock: FakeClock) -> None:
    """Three quick failures and the waits between them add up to more than a slow one."""
    flaky = FakeLLM(*[Timeout(after=6)] * 3, name="a", sleep=clock.sleep)
    lineup = router.Fallback(members(backed(flaky, clock), FakeLLM("ok", name="b")))
    started = clock()
    assert lineup.complete(REQUEST).text == "ok"
    assert len(flaky.calls) == 3 and (clock() - started).total_seconds() > 10


def test_a_model_found_unfit_for_the_task_is_not_asked(store: Store) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    first, second, third = router.lineup(config, Role.EXECUTOR)[:3]
    a, b, c = (FakeLLM(x, name=x, repeat_last=True) for x in "abc")
    lineup = router.Fallback([(a, first), (b, second), (c, third)])
    avoid = REQUEST.model_copy(update={"avoid": frozenset({second.label})})
    assert lineup.complete(avoid).text == "a"
    limited = router.Fallback(
        [(FakeLLM(RateLimit(retry_after=600), name="a"), first), (b, second), (c, third)]
    )
    assert limited.complete(avoid).text == "c"  # not the unfit second one
    assert b.calls == []
    # An entry from before models were told apart names the provider: all its models are out.
    google = router.lineup(config, Role.EXECUTOR, "google")[0]
    g = FakeLLM("google", name="g")
    old = REQUEST.model_copy(update={"avoid": frozenset({"groq"})})
    assert router.Fallback([(a, first), (b, second), (g, google)]).complete(old).text == "google"
    # When every model is unfit, they are asked all the same (the retry policy goes on by number).
    assert router.Fallback([(a, first), (b, second)]).complete(old).text == "a"


def test_old_unfit_entries_name_a_provider_and_still_count() -> None:
    assert among("groq/openai/gpt-oss-20b", {"groq"})
    assert among("groq/openai/gpt-oss-20b", {"groq/openai/gpt-oss-20b"})
    assert not among("groq/openai/gpt-oss-120b", {"groq/openai/gpt-oss-20b"})
    assert among("fake", {"fake"}) and not among("google/m", {"groq"})


def test_a_model_too_small_for_the_request_is_skipped() -> None:
    config = make_config(
        api_keys={"GROQ_API_KEY": GROQ, "CLOUDFLARE_API_TOKEN": "t" * 40},
        settings={"CLOUDFLARE_ACCOUNT_ID": "acc"},
    )
    small = router.Choice(providers.find("cloudflare"), "@cf/meta/llama-3.1-8b-instruct")
    large = router.lineup(config, Role.EXECUTOR)[0]
    tiny, big = FakeLLM("tiny", name="cloudflare"), FakeLLM("big", name="groq")
    lineup = router.Fallback([(tiny, small), (big, large)])
    long = Request(messages=[Message(role="user", content="x" * 30_000)])
    assert lineup.complete(long).text == "big"
    assert tiny.calls == []
    # When no model is large enough, each is tried all the same.
    only = router.Fallback([(FakeLLM("tried", name="cloudflare"), small)])
    assert only.complete(long).text == "tried"


def test_an_accepted_reply_is_kept_by_the_model_that_wrote_it(store: Store) -> None:
    first = Cached(FakeLLM(RateLimit(retry_after=600), name="a"), store.conn, "m")
    second = Cached(FakeLLM('{"v": 1}', name="b"), store.conn, "m")
    lineup = router.Fallback([(first, None), (second, None)])
    complete_json(lineup, REQUEST, RootModel[dict[str, int]], heading="Ask")
    assert second.complete(REQUEST).text == '{"v": 1}' and second.hits == 1


# ── The rate limiter ─────────────────────────────────────────────────────────


def test_a_bucket_gives_a_minute_of_requests_then_refills(store: Store, clock: FakeClock) -> None:
    buckets = Buckets(store.conn, clock)
    assert [buckets.take("groq/m", 3) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert buckets.take("groq/m", 3) == pytest.approx(20.0)  # one token every 20 s
    clock.advance(20)
    assert buckets.take("groq/m", 3) == 0.0
    assert buckets.take("other/m", 3) == 0.0  # each model has its own


def test_the_pace_outlives_a_restart_and_is_shared_by_workers(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "desman.db"
    first = db.open_db(path)
    assert [Buckets(first, clock).take("groq/m", 2) for _ in range(2)] == [0.0, 0.0]
    Buckets(first, clock).block("google/m", clock() + timedelta(hours=1))
    first.close()
    worker, other = db.open_db(path), db.open_db(path)
    assert Buckets(worker, clock).take("groq/m", 2) == pytest.approx(30.0)
    assert Buckets(other, clock).take("google/m", 2) == pytest.approx(3600.0)
    worker.close()
    other.close()


def test_every_request_sent_takes_a_token(store: Store, clock: FakeClock) -> None:
    """A refused key and the next one tried, a blip and its repeat: three requests, three
    tokens, for one call."""
    replies = iter(
        [
            httpx.Response(401, json={"error": {"message": "bad key"}}),
            httpx.Response(503, json={}),
            ok("fine"),
        ]
    )
    statuses: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = next(replies)
        statuses.append(response.status_code)
        return response

    from desman.llm.openai_compat import OpenAICompatible

    api = OpenAICompatible(
        "groq",
        "https://api.example/v1",
        [GROQ, "gsk_" + "z" * 48],
        "m",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    buckets = Buckets(store.conn, clock)
    llm = backed_api(Limited(api, buckets, "groq/m", 60, sleep=clock.sleep), clock)
    assert llm.complete(REQUEST).text == "fine"
    assert statuses == [401, 503, 200]
    # 60 a minute to begin with, three taken (the bucket refills only as the clock moves).
    row = store.conn.execute("SELECT tokens FROM rate_buckets WHERE scope = 'groq/m'").fetchone()
    assert row["tokens"] == pytest.approx(60 - 3 + clock.slept)


def backed_api(llm: Limited, clock: FakeClock) -> router.Backoff:
    return router.Backoff(llm, sleep=clock.sleep, jitter=lambda: 1.0, monotonic=lambda: 0.0)


def test_a_short_wait_is_waited_out_a_long_one_is_not_called(
    store: Store, clock: FakeClock
) -> None:
    buckets = Buckets(store.conn, clock)
    inner = FakeLLM("1", "2", "3", name="groq")
    limited = Limited(inner, buckets, "groq/m", 12, sleep=clock.sleep)  # a token every 5 s
    for _ in range(12):
        buckets.take("groq/m", 12)
    assert limited.complete(REQUEST).text == "1"
    assert clock.slept == pytest.approx(5.0)
    slow = Limited(FakeLLM("x", name="slow"), buckets, "slow/m", 1, sleep=clock.sleep)
    buckets.take("slow/m", 1)  # the next token in 60 s: longer than MAX_WAIT
    with pytest.raises(RateLimited) as raised:
        slow.complete(REQUEST)
    assert raised.value.retry_after == pytest.approx(60.0)
    assert isinstance(slow.inner, FakeLLM) and slow.inner.calls == []


def test_a_limit_the_provider_reported_keeps_the_model_aside(
    store: Store, clock: FakeClock
) -> None:
    buckets = Buckets(store.conn, clock)
    inner = FakeLLM(RateLimit(retry_after=120), "later", name="groq")
    limited = Limited(inner, buckets, "groq/m", 30, sleep=clock.sleep)
    with pytest.raises(RateLimited):
        limited.complete(REQUEST)
    assert buckets.blocked_until("groq/m") == clock() + timedelta(seconds=120)
    with pytest.raises(RateLimited):  # not called again before then
        limited.complete(REQUEST)
    assert len(inner.calls) == 1
    clock.advance(121)
    assert limited.complete(REQUEST).text == "later"

    resets = clock() + timedelta(hours=5)
    quota = Limited(
        FakeLLM(Quota(resets_in=5 * 3600), name="google", clock=clock), buckets, "google/m", 10
    )
    with pytest.raises(QuotaExhausted):
        quota.complete(REQUEST)
    assert buckets.blocked_until("google/m") == resets


def test_a_block_is_never_shortened(store: Store, clock: FakeClock) -> None:
    buckets = Buckets(store.conn, clock)
    buckets.block("g/m", clock() + timedelta(hours=2))
    buckets.block("g/m", clock() + timedelta(minutes=1))
    assert buckets.blocked_until("g/m") == clock() + timedelta(hours=2)


def test_a_v18_database_upgrades_to_rate_buckets(tmp_path: Path) -> None:
    from test_db import _db_at_version

    path = tmp_path / "v18.db"
    _db_at_version(path, 18).close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert Buckets(conn).take("groq/m", 1) == 0.0
    conn.close()


# ── A run with real clients (HTTP mocked) ────────────────────────────────────


Handler = Callable[[httpx.Request], httpx.Response]


def ok(text: str, model: str = "m") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def run_models(
    conn: sqlite3.Connection, clock: FakeClock, handler: Handler, **kw: Any
) -> runs.Models:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "OPENROUTER_API_KEY": OPENROUTER})
    config.tuning.models.text = "groq/openai/gpt-oss-20b"
    return runs.models(
        config,
        conn,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        **kw,
    )


def sent(request: httpx.Request) -> tuple[str, str, str]:
    body = json.loads(request.content)
    return request.url.host, body["model"], body["messages"][-1]["content"].split(":")[0]


def test_roles_ask_their_own_models(store: Store, clock: FakeClock) -> None:
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        return ok('{"verdict": "accept"}' if "Review" in seen[-1][2] else '{"v": 1}')

    models = run_models(store.conn, clock, handler)
    assert models.choice is not None and models.choice.label == "groq/openai/gpt-oss-20b"
    reviewer = models.first(Role.REVIEWER)
    assert reviewer is not None and reviewer.label == "groq/openai/gpt-oss-120b"
    assert models.other_choice is not None and models.other_choice.provider.id == "openrouter"
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    assert [(model, what) for _, model, what in seen] == [
        ("openai/gpt-oss-20b", "Execute subtask"),
        ("openai/gpt-oss-120b", "Review subtask"),
    ]


def test_a_limited_model_costs_no_attempt_the_next_one_answers(
    store: Store, clock: FakeClock
) -> None:
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        if seen[-1][1] == "openai/gpt-oss-20b":
            return httpx.Response(429, headers={"retry-after": "600"}, json={})
        return ok('{"verdict": "accept"}' if "Review" in seen[-1][2] else '{"v": 1}')

    models = run_models(store.conn, clock, handler)
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    executed = [model for _, model, what in seen if what == "Execute subtask"]
    assert executed == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    [subtask] = store.list_subtasks(task_id)
    assert subtask.attempts.quality == 0 and subtask.attempts.transient == 0
    # Kept aside in the database: the next run does not knock on it again.
    buckets = Buckets(store.conn, clock)
    until = buckets.blocked_until("groq/openai/gpt-oss-20b")
    assert until is not None and abs(until - clock() - timedelta(seconds=600)) < timedelta(
        seconds=1
    )
    before = len(seen)
    again = run_models(store.conn, clock, handler)
    again.llm.complete(REQUEST)
    assert [model for _, model, _ in seen[before:]] == ["openai/gpt-oss-120b"]


def test_every_model_limited_pauses_until_the_soonest(store: Store, clock: FakeClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        wait = "120" if request.url.host == "openrouter.ai" else "900"
        return httpx.Response(429, headers={"retry-after": wait}, json={})

    models = run_models(store.conn, clock, handler)
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    result = runs.summarizer(store, models).run(task_id, wait_for_pauses=False)
    assert result.paused_until is not None
    paused = db.parse_iso(result.paused_until) - clock()  # the soonest: OpenRouter's
    assert abs(paused - timedelta(seconds=120)) < timedelta(seconds=1)
    [subtask] = store.list_subtasks(task_id)
    assert subtask.status is SubtaskStatus.PAUSED


def test_an_unfit_model_is_told_apart_from_the_other_models_of_its_provider(
    store: Store, clock: FakeClock
) -> None:
    """gpt-oss calling a tool natively on Groq (C.8): that model is avoided, by its label."""
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        if seen[-1][1] == "openai/gpt-oss-20b":
            error = {"message": "model called a tool", "code": "tool_use_failed"}
            return httpx.Response(400, json={"error": error})
        return ok('{"verdict": "accept"}' if "Review" in seen[-1][2] else '{"v": 1}')

    models = run_models(store.conn, clock, handler)
    assert models.other is not None
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    coordinator = Coordinator(
        store,
        llm_executor(models.llm, models.other),
        llm_reviewer(models.for_(Role.REVIEWER)),
        sleep=clock.sleep,
        strategies=strategy_for,
        providers=(runs.label(models.llm), (runs.label(models.other),)),
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert store.unfit_providers(task_id) == {"groq/openai/gpt-oss-20b"}
    executed = [(host, model) for host, model, what in seen if what == "Execute subtask"]
    # The lineup's next model answers in the same attempt: the quirk costs no attempt.
    assert executed == [
        ("api.groq.com", "openai/gpt-oss-20b"),
        ("api.groq.com", "openai/gpt-oss-120b"),
    ]
    assert check_history(store.conn, task_id) == []
    strategies = [
        e.details["strategy"] for e in history(store.conn, task_id) if e.action == "attempt_started"
    ]
    assert strategies == ["first"]


def test_a_pinned_model_leaves_no_other(store: Store, clock: FakeClock) -> None:
    models = run_models(store.conn, clock, lambda r: ok("{}"), model="openai/gpt-oss-20b")
    assert models.other is None and models.other_choice is None
    for role in Role:
        first = models.first(role)
        assert first is not None and first.label == "groq/openai/gpt-oss-20b"


def test_a_fake_has_no_lineup() -> None:
    fake = FakeLLM()
    models = runs.Models(fake, None)
    assert models.for_(Role.REVIEWER) is fake and models.first(Role.REVIEWER) is None
    assert runs.label(fake) == fake.name


def test_an_unfit_model_the_lineup_fell_back_to_is_not_asked_again(
    store: Store, clock: FakeClock
) -> None:
    """The first model is limited, the second calls a tool natively: the next attempt skips
    both (the first is still aside, the second unfit) instead of spending attempts on it."""
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        model, what = seen[-1][1], seen[-1][2]
        if model == "openai/gpt-oss-20b":
            return httpx.Response(429, headers={"retry-after": "3600"}, json={})
        if model == "openai/gpt-oss-120b" and what.startswith("Execute"):
            error = {"message": "model called a tool", "code": "tool_use_failed"}
            return httpx.Response(400, json={"error": error})
        return ok('{"verdict": "accept"}' if what.startswith("Review") else '{"v": 1}')

    models = run_models(store.conn, clock, handler)
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert models.other is not None
    coordinator = Coordinator(
        store,
        llm_executor(models.llm, models.other),
        llm_reviewer(models.for_(Role.REVIEWER)),
        sleep=clock.sleep,
        strategies=strategy_for,
        providers=(runs.label(models.llm), (runs.label(models.other),)),
    )
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert store.unfit_providers(task_id) == {"groq/openai/gpt-oss-120b"}
    executed = [model for _, model, what in seen if what == "Execute subtask"]
    assert executed.count("openai/gpt-oss-120b") == 1
    assert executed[-1] == "qwen/qwen3.8-27b"


# ── The kind of a subtask picks its executor's model (4.5, part 2) ───────────


def test_every_kind_says_what_it_needs() -> None:
    assert set(catalog.KIND_NEEDS) == set(Kind)
    assert catalog.need(Role.EXECUTOR) is catalog.NEEDS[Role.EXECUTOR]
    assert catalog.need(Role.EXECUTOR, Kind.CODE) is catalog.Strength.CODE
    # Other roles do the same work whatever the subtask: a reviewer judges code too.
    assert catalog.need(Role.REVIEWER, Kind.CODE) is catalog.Strength.ANALYSIS


def test_the_executors_lineup_follows_the_kind_of_its_subtask() -> None:
    config = make_config(api_keys={"MISTRAL_API_KEY": MISTRAL})

    def first(kind: Kind | None) -> str:
        return router.lineup(config, Role.EXECUTOR, kind=kind)[0].label

    assert first(None) == first(Kind.TEXT) == "mistral/mistral-small-latest"
    assert first(Kind.CODE) == "mistral/codestral-latest"
    assert first(Kind.ANALYSIS) == "mistral/mistral-medium-latest"
    # Pinned, the model is the one asked whatever the kind.
    pinned = router.lineup(config, Role.EXECUTOR, model="mistral-small-latest", kind=Kind.CODE)
    assert labels(pinned) == ["mistral/mistral-small-latest"]


def test_a_subtask_keeps_its_kind_and_so_do_its_parts(store: Store) -> None:
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="script", acceptance_criteria="c", kind=Kind.CODE),
                SubtaskSpec(description="note", acceptance_criteria="c", depends_on=[0]),
            ],
        )
    )
    script, note = store.list_subtasks(task_id)
    assert (script.kind, note.kind) == (Kind.CODE, Kind.TEXT)
    claimed = store.claim("w")
    assert claimed is not None and claimed.lease_token is not None
    parts = [
        SubtaskSpec(description="a", acceptance_criteria="c", kind=Kind.CODE),
        SubtaskSpec(description="b", acceptance_criteria="c", kind=Kind.ANALYSIS),
    ]
    made = store.decompose(script.id, claimed.lease_token, parts)
    assert [p.kind for p in made] == [Kind.CODE, Kind.ANALYSIS]
    reported = {s.description: s.kind for s in task_report(store, task_id).subtasks}
    assert reported == {"script": Kind.CODE, "note": Kind.TEXT, "a": Kind.CODE, "b": Kind.ANALYSIS}


def test_the_kind_check_matches_the_kinds(store: Store) -> None:
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    [subtask] = store.list_subtasks(task_id)
    for kind in Kind:
        store.conn.execute("UPDATE subtasks SET kind = ? WHERE id = ?", (kind.value, subtask.id))
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE subtasks SET kind = 'poetry' WHERE id = ?", (subtask.id,))


def test_a_v19_database_upgrades_its_subtasks_to_text(tmp_path: Path) -> None:
    from test_db import _db_at_version, add_subtask, add_task

    path = tmp_path / "v19.db"
    old = _db_at_version(path, 19)
    subtask_id = add_subtask(old, add_task(old), 0, status="ready")
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert Store(conn).get_subtask(subtask_id).kind is Kind.TEXT
    conn.close()


def mistral_models(conn: sqlite3.Connection, clock: FakeClock, handler: Handler) -> runs.Models:
    return runs.models(
        make_config(api_keys={"MISTRAL_API_KEY": MISTRAL, "GROQ_API_KEY": GROQ}),
        conn,
        provider="mistral",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock,
        sleep=clock.sleep,
        cache=False,
    )


def test_each_subtask_is_executed_by_the_model_for_its_kind(store: Store, clock: FakeClock) -> None:
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        return ok('{"verdict": "accept"}' if "Review" in seen[-1][2] else f'{{"v": {len(seen)}}}')

    models = mistral_models(store.conn, clock, handler)
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="code it", acceptance_criteria="c", kind=Kind.CODE),
                SubtaskSpec(
                    description="weigh it",
                    acceptance_criteria="c",
                    kind=Kind.ANALYSIS,
                    depends_on=[0],
                ),
                SubtaskSpec(description="say it", acceptance_criteria="c", depends_on=[1]),
            ],
        )
    )
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    executed = [model for _, model, what in seen if what == "Execute subtask"]
    assert executed == ["codestral-latest", "mistral-medium-latest", "mistral-small-latest"]
    reviewed = {model for _, model, what in seen if what == "Review subtask"}
    assert reviewed == {"mistral-medium-latest"}  # the reviewer's model, whatever the kind


def test_an_unfit_model_is_avoided_by_the_kind_that_asks_it(store: Store, clock: FakeClock) -> None:
    """Codestral failing as unfit: a code subtask's next attempt asks the other model at once
    (`avoid_main`), since the main one for its kind is the unfit one."""
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(sent(request))
        if seen[-1][1] == "codestral-latest":
            error = {"message": "model called a tool", "code": "tool_use_failed"}
            return httpx.Response(400, json={"error": error})
        return ok('{"verdict": "accept"}' if "Review" in seen[-1][2] else f'{{"v": {len(seen)}}}')

    models = mistral_models(store.conn, clock, handler)
    task_id = store.create_task(
        TaskSpec(
            goal="g",
            subtasks=[
                SubtaskSpec(description="code it", acceptance_criteria="c", kind=Kind.CODE),
                SubtaskSpec(description="say it", acceptance_criteria="c", depends_on=[0]),
            ],
        )
    )
    code, text = store.list_subtasks(task_id)
    assert models.labels(code) == (
        "mistral/codestral-latest",
        ("mistral/mistral-small-latest", "mistral/mistral-medium-latest"),
    )
    assert models.labels(text)[0] == "mistral/mistral-small-latest"
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    assert store.unfit_providers(task_id) == {"mistral/codestral-latest"}
    executed = [model for _, model, what in seen if what == "Execute subtask"]
    assert executed == ["codestral-latest", "mistral-small-latest", "mistral-small-latest"]
    assert check_history(store.conn, task_id) == []
    strategies = [
        (e.subtask_id, e.details["strategy"])
        for e in history(store.conn, task_id)
        if e.action == "attempt_started"
    ]
    assert strategies == [(code.id, "first"), (text.id, "first")]  # no attempt spent on it


def test_run_names_the_models_for_code_and_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from desman.cli import EXIT_DONE, cli
    from desman.config import ConfigManager, default_config_path

    ConfigManager(default_config_path()).save(make_config(api_keys={"MISTRAL_API_KEY": MISTRAL}))

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return ok('{"verdict": "accept"}' if prompt.startswith("Review") else '{"v": 1}')

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    result = CliRunner().invoke(cli, ["run", "g", "--plan", "template"])
    assert result.exit_code == EXIT_DONE, result.output
    assert "Using Mistral AI · mistral-small-latest" in result.output
    assert "Code by Mistral AI · codestral-latest" in result.output
    assert "Analysis by Mistral AI · mistral-medium-latest" in result.output


def test_ollama_auto_and_chosen_without_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    from desman import local_models

    config = make_config(api_keys={"GROQ_API_KEY": GROQ})

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("routing must not discover installed models")

    monkeypatch.setattr(local_models, "installed", forbidden)
    for role in (Role.PLANNER, Role.EXECUTOR, Role.REVIEWER, Role.INTERLOCUTOR):
        assert all(c.provider.id != "ollama" for c in router.lineup(config, role))
    config.tuning.models.interlocutor = "ollama/removed"
    choices = router.lineup(config, Role.INTERLOCUTOR)
    assert choices[0].label == "ollama/removed"
    assert sum(c.provider.id == "ollama" for c in choices) == 1

    # A removed tag returns Ollama's ordinary 404; the existing fallback asks the next model.
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'removed' not found"})

    with httpx.Client(transport=httpx.MockTransport(missing)) as client:
        local = router.connect(config, choices[0], client=client)
        fallback = router.Fallback([(local, choices[0]), (FakeLLM("cloud ok"), choices[1])])
        assert fallback.complete(REQUEST).text == "cloud ok"
    config.api_keys = {}
    assert labels(router.lineup(config, Role.INTERLOCUTOR)) == ["ollama/removed"]


def test_default_ollama_discovery_never_opens_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    from desman import local_models
    from desman.config import Config

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("default Ollama discovery opened a socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    assert local_models.installed(Config(), refresh=True) == ([], False)
