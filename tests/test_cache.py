"""Tests for the response cache (ROADMAP 2.6): accepted replies only, by exact request."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import httpx
import pytest
from click.testing import CliRunner, Result
from pydantic import BaseModel

from conftest import FakeClock
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.llm import cache, router
from desman.llm.base import Completion, Message, Request, Usage
from desman.llm.fake import FakeLLM, ScriptItem, garbage
from desman.llm.structured import complete_json
from helpers import make_config


class Answer(BaseModel):
    answer: int


def ask(
    text: str = "Ask:\n{}",
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> Request:
    return Request(
        messages=[Message(role="user", content=text)],
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


@pytest.fixture
def conn() -> sqlite3.Connection:
    return db.open_db(":memory:")


def cached(
    conn: sqlite3.Connection, clock: FakeClock, *steps: ScriptItem
) -> tuple[cache.Cached, FakeLLM]:
    fake = FakeLLM(*steps)
    return cache.Cached(fake, conn, "m", clock=clock), fake


def test_an_accepted_reply_answers_the_same_request_again(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    llm, fake = cached(conn, clock, '{"answer": 1}')
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=1)
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=1)  # no step left
    assert len(fake.calls) == 1 and llm.hits == 1


@pytest.mark.parametrize(
    "other",
    [
        ask('Ask:\n{"attempt": 2}'),  # a retry asks something else
        ask(temperature=0.7),
        ask(max_tokens=50),
        ask(model="other-model"),
    ],
)
def test_a_different_request_is_not_served(
    conn: sqlite3.Connection, clock: FakeClock, other: Request
) -> None:
    llm, fake = cached(conn, clock, '{"answer": 1}', '{"answer": 2}')
    complete_json(llm, ask(), Answer, heading="Ask")
    assert complete_json(llm, other, Answer, heading="Ask") == Answer(answer=2)
    assert len(fake.calls) == 2


def test_the_same_request_to_another_provider_is_not_served(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    first, _ = cached(conn, clock, '{"answer": 1}')
    complete_json(first, ask(), Answer, heading="Ask")
    other = cache.Cached(FakeLLM('{"answer": 2}', name="other"), conn, "m", clock=clock)
    assert complete_json(other, ask(), Answer, heading="Ask") == Answer(answer=2)


def test_an_unusable_reply_is_never_kept(conn: sqlite3.Connection, clock: FakeClock) -> None:
    # Garbage, then a good repair: only the repair is kept. Asking again calls the model,
    # which may well answer properly this time, instead of replaying the garbage.
    llm, fake = cached(conn, clock, garbage("prose"), '{"answer": 1}', '{"answer": 3}')
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=1)
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=3)
    assert len(fake.calls) == 3
    assert conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0] == 2


def test_a_plain_call_keeps_nothing(conn: sqlite3.Connection, clock: FakeClock) -> None:
    llm, fake = cached(conn, clock, "anything", "anything")
    llm.complete(ask())
    llm.complete(ask())
    assert len(fake.calls) == 2  # nobody accepted it


def test_old_replies_expire_and_are_pruned(conn: sqlite3.Connection, clock: FakeClock) -> None:
    llm, fake = cached(conn, clock, '{"answer": 1}', '{"answer": 2}', '{"answer": 3}')
    complete_json(llm, ask(), Answer, heading="Ask")
    clock.advance((cache.TTL + timedelta(seconds=1)).total_seconds())
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=2)
    complete_json(llm, ask("Ask:\n[]"), Answer, heading="Ask")  # an accept prunes the expired
    keys = {row[0] for row in conn.execute("SELECT key FROM response_cache")}
    assert len(keys) == 2 and cache.request_key("fake", "m", ask()) in keys


def test_a_cached_reply_keeps_what_the_model_said(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    reply = Completion(
        text='{"answer": 5}',
        provider="fake",
        model="served-model",
        usage=Usage(prompt_tokens=7, completion_tokens=3),
        finish_reason="length",
    )
    llm, _ = cached(conn, clock)
    llm.accept(ask(), reply)
    assert llm.complete(ask()) == reply


# ── Through the CLI ──────────────────────────────────────────────────────────

KEY = "gsk_" + "k" * 48


def test_a_second_run_reuses_accepted_replies_unless_told_not_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"GROQ_API_KEY": KEY}))
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        sent.append(prompt.split(":")[0])
        text = '{"verdict": "accept"}' if prompt.startswith("Review") else '{"answer": 42}'
        if prompt.startswith("Plan"):
            text = '{"subtasks": [{"description": "d", "acceptance_criteria": "c"}]}'
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
    monkeypatch.setattr(router, "_sleep", lambda seconds: None)

    def run(*extra: str) -> Result:
        return CliRunner().invoke(cli, ["run", "same goal", *extra])

    assert run().exit_code == 0
    assert sent == ["Plan task", "Execute subtask", "Review subtask"]
    assert run().exit_code == 0
    assert len(sent) == 3  # the same questions, the plan too: answered from the cache
    assert run("--no-cache").exit_code == 0
    assert len(sent) == 6


def test_reuse_does_not_extend_a_replys_life(conn: sqlite3.Connection, clock: FakeClock) -> None:
    llm, fake = cached(conn, clock, '{"answer": 1}', '{"answer": 2}')
    complete_json(llm, ask(), Answer, heading="Ask")  # day 0
    clock.advance(timedelta(days=6).total_seconds())
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=1)  # a hit
    clock.advance(timedelta(days=2).total_seconds())  # day 8: the day-0 reply has expired
    assert complete_json(llm, ask(), Answer, heading="Ask") == Answer(answer=2)
    assert len(fake.calls) == 2
