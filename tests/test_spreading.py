"""Capacity-aware Auto routing with task/chat affinity and explicit-choice precedence."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import httpx

from conftest import FakeClock
from desman import runs
from desman.core import calls, db
from desman.core.sessions import Role as SessionRole
from desman.core.sessions import Sessions
from desman.core.store import Store
from desman.llm import router
from desman.llm.limiter import Buckets
from helpers import make_config
from test_db import _db_at_version
from test_routing import GOOGLE, GROQ, MISTRAL, REQUEST, ok


def test_spreading_tasks_sticky_and_other(store: Store, clock: FakeClock) -> None:
    config = make_config(
        api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE, "MISTRAL_API_KEY": MISTRAL}
    )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        return ok("answer", model)

    models = runs.models(
        config,
        store.conn,
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    starts = []
    for task in ("one", "two", "three"):
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task)):
            first = models.llm.complete(REQUEST)
            starts.append(first.provider)
            assert runs.label(models.llm) == f"{first.provider}/{first.model}"
            # Large changes in relative load do not disturb this task's choice.
            assert models.llm.complete(REQUEST).provider == first.provider
            assert models.other is not None
            assert runs.label(models.other) != runs.label(models.llm)
            alternate = models.other.complete(REQUEST)
            assert (alternate.provider, alternate.model) != (first.provider, first.model)
    assert len(set(starts)) == 3


def test_blocked_and_failover_rerank(store: Store, clock: FakeClock) -> None:
    config = make_config(
        api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE, "MISTRAL_API_KEY": MISTRAL}
    )
    choices = router.lineup(config, calls.Role.EXECUTOR)
    tier = router.spread_tier(choices, calls.Role.EXECUTOR)
    order = router.capacity_order(store.conn, config, tier, clock)
    Buckets(store.conn, clock).block(tier[0].label, clock() + timedelta(hours=1))
    ranked = order(choices)
    assert ranked[len(tier) - 1] == tier[0]
    assert ranked[len(tier) :] == [c for c in choices if c not in tier]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append(model)
        # Non-retried refused key forces a fallback.
        if len(seen) == 1:
            return httpx.Response(401, json={"error": {"message": "refused"}})
        return ok("answer", model)

    models = runs.models(
        config,
        store.conn,
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    session = Sessions(store.conn).add(None, SessionRole.USER, "hello").session_id
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, session_id=session)):
        models.llm.complete(REQUEST)
        # The next call still avoids the provider whose shared bucket is blocked.
        reply = models.llm.complete(REQUEST)
        assert reply.provider != tier[0].provider.id


def test_pins_disable_spreading(store: Store, clock: FakeClock) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    config.tuning.models.text = "groq/openai/gpt-oss-20b"
    models = runs.models(
        config,
        store.conn,
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: ok("ok", json.loads(r.content)["model"]))
        ),
    )
    for task in ("one", "two", "three"):
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task)):
            assert models.llm.complete(REQUEST).provider == "groq"


def test_previous_version_upgrade_adds_capacity_index(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    old = _db_at_version(path, len(db.MIGRATIONS) - 1)
    old.execute(
        "INSERT INTO model_calls (provider, model, prompt_hash, latency_ms) VALUES ('p', 'm', 'h', 0)"
    )
    old.close()
    conn = db.open_db(path)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 1
    assert [
        r["name"] for r in conn.execute("PRAGMA index_info(ix_model_calls_provider_model_at)")
    ] == ["provider", "model", "at"]
    conn.close()


def test_capacity_uses_key_count_and_ten_minute_window(store: Store, clock: FakeClock) -> None:
    from desman.config import ApiKey

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "MISTRAL_API_KEY": MISTRAL})
    config.api_keys["GROQ_API_KEY"].append(ApiKey(value="second-key"))
    choices = router.lineup(config, calls.Role.EXECUTOR)
    tier = router.spread_tier(choices, calls.Role.EXECUTOR)
    assert len(tier) == 2
    for choice in tier:
        store.conn.execute(
            "INSERT INTO model_calls (provider, model, at, prompt_hash, latency_ms) VALUES (?, ?, ?, 'h', 0)",
            (choice.provider.id, choice.model, db.to_iso(clock())),
        )
    order = router.capacity_order(store.conn, config, tier, clock)
    assert order(choices)[0].provider.id == "groq"  # one call over two keys
    for _ in range(3):
        store.conn.execute(
            "INSERT INTO model_calls (provider, model, at, prompt_hash, latency_ms) VALUES (?, ?, ?, 'h', 0)",
            (tier[0].provider.id, tier[0].model, db.to_iso(clock())),
        )
    assert order(choices)[0].provider.id == "mistral"
    clock.sleep(601)
    assert order(choices)[0].provider.id == "groq"  # stale calls no longer count
    plan = store.conn.execute(
        "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM model_calls WHERE provider = ? AND model = ? AND at >= ?",
        ("groq", "m", db.to_iso(clock())),
    ).fetchall()
    assert any("ix_model_calls_provider_model_at" in row["detail"] for row in plan)


def test_auto_with_chosen_key_and_explicit_provider_stays_first(
    store: Store, clock: FakeClock
) -> None:
    from desman.llm.openai_compat import key_id

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    config.tuning.keys.text = key_id(GOOGLE)
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: ok("ok", json.loads(r.content)["model"]))
    )
    models = runs.models(
        config, store.conn, clock=clock, sleep=clock.sleep, cache=False, client=client
    )
    for task in ("one", "two", "three"):
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=task)):
            assert models.llm.complete(REQUEST).provider == "google"
    explicit = runs.models(
        config,
        store.conn,
        provider="groq",
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=client,
    )
    assert explicit.llm.complete(REQUEST).provider == "groq"


def test_chat_scope_is_sticky_and_unattributed_calls_spread(store: Store, clock: FakeClock) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    models = runs.models(
        config,
        store.conn,
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: ok("ok", json.loads(r.content)["model"]))
        ),
    )
    session = Sessions(store.conn).add(None, SessionRole.USER, "hello").session_id
    with calls.for_(calls.Purpose(calls.Role.INTERLOCUTOR, session_id=session)):
        llm = models.for_(calls.Role.INTERLOCUTOR)
        first = llm.complete(REQUEST)
        assert llm.complete(REQUEST).provider == first.provider
    assert models.llm.complete(REQUEST).provider == "google"


def test_failover_invalidates_affinity_before_next_call() -> None:
    from desman.llm.fake import FakeLLM, RateLimit

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    tier = router.spread_tier(router.lineup(config, calls.Role.EXECUTOR), calls.Role.EXECUTOR)
    first, second = tier
    attempts = 0

    def order(choices: list[router.Choice]) -> list[router.Choice]:
        nonlocal attempts
        attempts += 1
        return choices if attempts == 1 else list(reversed(choices))

    llm = router.Fallback(
        [
            (FakeLLM(RateLimit(retry_after=600), name="limited"), first),
            (FakeLLM("ok", repeat_last=True, name="answer"), second),
        ],
        order=order,
    )
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id="one")):
        assert llm.complete(REQUEST).text == "ok"
        assert attempts == 1
        assert llm.complete(REQUEST).text == "ok"
        assert attempts == 2
        assert llm.complete(REQUEST).text == "ok"
        assert attempts == 2  # affinity established after reranking


def test_recent_failure_recovers_after_success_or_one_hour(store: Store, clock: FakeClock) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    choices = router.lineup(config, calls.Role.EXECUTOR)
    tier = router.spread_tier(choices, calls.Role.EXECUTOR)
    first, healthy = tier
    order = router.capacity_order(store.conn, config, tier, clock)

    def record(choice: router.Choice, error: str | None, age: int = 0) -> None:
        store.conn.execute(
            "INSERT INTO model_calls (provider, model, at, prompt_hash, latency_ms, error_kind) "
            "VALUES (?, ?, ?, 'h', 0, ?)",
            (choice.provider.id, choice.model, db.to_iso(clock() - timedelta(seconds=age)), error),
        )

    for _ in range(4):
        record(healthy, None)
    record(first, "permanent", 601)  # outside usage window, still a recent failure
    assert order(choices)[0] == healthy
    record(first, None)
    assert order(choices)[0] == first
    record(first, "permanent")  # same timestamp: latest seq decides
    assert order(choices)[0] == healthy
    record(first, "quality")
    assert order(choices)[0] == first
    record(first, "permanent")
    clock.sleep(3601)
    assert order(choices)[0] == first


def test_scope_caches_evict_oldest_and_refresh_active_scopes(
    store: Store, clock: FakeClock
) -> None:
    from desman.llm.fake import FakeLLM

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    tier = router.spread_tier(router.lineup(config, calls.Role.EXECUTOR), calls.Role.EXECUTOR)
    llm = router.Fallback(
        [(FakeLLM("ok", repeat_last=True), c) for c in tier], order=lambda choices: choices
    )
    models = runs.models(
        config,
        store.conn,
        clock=clock,
        sleep=clock.sleep,
        cache=False,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: ok("ok", json.loads(r.content)["model"]))
        ),
    )
    for scope in [*map(str, range(256)), "0", "256"]:
        with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id=scope)):
            llm.complete(REQUEST)
            models.llm.complete(REQUEST)
    assert len(llm._scopes) == 256
    assert ("task", "0") in llm._scopes
    assert ("task", "1") not in llm._scopes
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id="0")):
        assert runs.label(models.llm) != "Auto"
        retained = runs.label(models.llm)
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id="1")):
        # Evicted scopes revert to the default label; recent scopes retain actual usage.
        assert runs.label(models.llm) == tier[0].label
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id="256")):
        assert runs.label(models.llm) != retained
