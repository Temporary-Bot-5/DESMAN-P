"""Transient model health and readable result rendering (T.6l)."""

from datetime import timedelta

import pytest

from conftest import FakeClock
from desman.actions import plain_math, result_text
from desman.core import calls, db
from desman.core.store import Store
from desman.llm import router
from desman.llm.fake import FakeLLM
from helpers import make_config
from test_routing import GOOGLE, GROQ, REQUEST


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"\( 700 \text{ км} \times 7 \text{ л} / 100 \)", "700 км × 7 л / 100"),
        ("$5 and $10", "$5 and $10"),
        (r"$\pi r^2 \approx 28,3$ м², $t \to \infty$", "π r² ≈ 28,3 м², t → ∞"),
        (r"`\(x^2\)`", r"`\(x^2\)`"),
        ("```latex\n\\(x^2\\)\n```", "```latex\n\\(x^2\\)\n```"),
        (r"$x^2$ and $$\frac{a+b}{c}$$", "x² and (a+b)/c"),
        (r"\[\sqrt{x} \leq x^{3} \ne \unknown{x}\]", r"√x ≤ x³ ≠ \unknown{x}"),
        (r"\(\mathrm{x}\cdot 2\div 3\approx 1\ge 0\pm 1\,\;\ \%\)", "x· 2÷ 3≈ 1≥ 0± 1 %"),
    ],
)
def test_plain_math(source: str, expected: str) -> None:
    assert plain_math(source) == expected
    original = {"text": source}
    assert result_text(original) == expected
    assert original == {"text": source}


@pytest.mark.parametrize("ignored", ["limit", "quality", "permanent"])
def test_model_health(store: Store, clock: FakeClock, ignored: str) -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    choices = router.lineup(config, calls.Role.REVIEWER)[:2]
    sick, healthy = choices

    def record(kind: str | None, minutes: int = 0) -> None:
        store.conn.execute(
            "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, error_kind, at) "
            "VALUES (?, ?, '', 0, ?, ?)",
            (sick.provider.id, sick.model, kind, db.to_iso(clock() - timedelta(minutes=minutes))),
        )

    order = router.health_order(store.conn, clock)
    record("transient", 16)
    record(ignored)
    record(ignored)
    assert order(choices) == choices
    record("transient")
    assert order(choices) == choices
    record("transient")
    assert order(choices) == [healthy, sick]
    pinned = router.health_order(store.conn, clock, keep=sick.label)
    assert pinned(choices) == choices
    first, second = FakeLLM("sick"), FakeLLM("healthy")
    fallback = router.Fallback([(first, sick), (second, healthy)], health=order)
    assert fallback.complete(REQUEST).text == "healthy"
    clock.sleep(1)
    record(None)
    assert order(choices) == choices
    assert fallback.complete(REQUEST).text == "sick"
    record("transient")
    record("transient")
    store.conn.execute(
        "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, error_kind, at) "
        "VALUES (?, ?, '', 0, 'transient', ?), (?, ?, '', 0, 'transient', ?)",
        (
            healthy.provider.id,
            healthy.model,
            db.to_iso(clock()),
            healthy.provider.id,
            healthy.model,
            db.to_iso(clock()),
        ),
    )
    assert order(choices) == choices
    all_sick = router.Fallback([(FakeLLM("still asked"), sick), (FakeLLM(), healthy)], health=order)
    assert all_sick.complete(REQUEST).text == "still asked"
    clock.sleep(901)
    assert order(choices) == choices


def test_runs_refreshes_health_for_whole_lineup(store: Store, clock: FakeClock) -> None:
    from desman import runs
    from desman.llm.base import LLM

    config = make_config(api_keys={"GROQ_API_KEY": GROQ, "GOOGLE_API_KEY": GOOGLE})
    seen: list[str] = []

    def wrap(llm: LLM, choice: router.Choice) -> LLM:
        def answer(request: object) -> str:
            seen.append(choice.label)
            return "answer"

        return FakeLLM(answer, repeat_last=True)

    models = runs.models(config, store.conn, clock=clock, sleep=clock.sleep, cache=False, wrap=wrap)
    with calls.for_(calls.Purpose(calls.Role.EXECUTOR, task_id="held")):
        models.llm.complete(REQUEST)
        original = seen[-1]
        provider, model = original.split("/", 1)
        for _ in range(2):
            store.conn.execute(
                "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, error_kind, at) "
                "VALUES (?, ?, '', 0, 'transient', ?)",
                (provider, model, db.to_iso(clock())),
            )
        models.llm.complete(REQUEST)
        assert seen[-1] != original
        clock.sleep(1)
        store.conn.execute(
            "INSERT INTO model_calls(provider, model, prompt_hash, latency_ms, at) "
            "VALUES (?, ?, '', 0, ?)",
            (provider, model, db.to_iso(clock())),
        )
        models.llm.complete(REQUEST)
        assert seen[-1] == original
