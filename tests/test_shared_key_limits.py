"""Shared account access and reported token windows (T.6h F54/F55)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from conftest import FakeClock
from desman.config_cli import print_model_blocks
from desman.core import db
from desman.llm.base import Message, ProviderRejected, RateLimited, Request
from desman.llm.limiter import Buckets, Limited
from desman.llm.openai_compat import OpenAICompatible, key_id

REQUEST = Request(messages=[Message(role="user", content="hello")], max_tokens=100)


def api(
    conn: sqlite3.Connection,
    clock: FakeClock,
    transport: httpx.MockTransport,
    keys: list[str] | None = None,
    model: str = "m",
) -> Limited:
    inner = OpenAICompatible(
        "groq",
        "https://example.test/v1",
        keys or ["secret"],
        model,
        clock=clock,
        client=httpx.Client(transport=transport),
    )
    return Limited(inner, Buckets(conn, clock), "groq/" + model, 100, sleep=clock.sleep)


def reply(headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, headers=headers, json={"choices": [{"message": {"content": "ok"}}]})


@pytest.mark.parametrize(
    "status,headers",
    [(429, {"x-ratelimit-limit-requests": "0"}), (403, {}), (400, {}), (404, {}), (402, {})],
)
def test_account_refusal_shared_keyed_and_expires(
    tmp_path: Path,
    clock: FakeClock,
    status: int,
    headers: dict[str, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock.now = datetime(2099, 1, 1, tzinfo=UTC)
    path = tmp_path / "limits.db"
    first, second = db.open_db(path), db.open_db(path)
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers["authorization"])
        return httpx.Response(status, headers=headers, text="requires a valid credit card")

    transport = httpx.MockTransport(handle)
    a = api(first, clock, transport)
    # 402 is a permanent refusal only for explicitly chosen paid providers.
    assert isinstance(a.inner, OpenAICompatible)
    a.inner.paid = True
    with pytest.raises(ProviderRejected):
        a.complete(REQUEST)
    b = api(second, clock, transport)
    with pytest.raises(ProviderRejected, match="unavailable to this key"):
        b.complete(REQUEST)
    assert len(sent) == 1
    print_model_blocks(path)
    output = capsys.readouterr().out
    assert "unavailable to this key" in output and "secret" not in output
    # Other models/keys still work; key material is never persisted.
    good = httpx.MockTransport(lambda r: reply())
    assert api(second, clock, good, ["other"]).complete(REQUEST).text == "ok"
    assert api(second, clock, good, model="other").complete(REQUEST).text == "ok"
    assert "secret" not in "\n".join(first.iterdump())
    clock.advance(86401)
    assert api(second, clock, good).complete(REQUEST).text == "ok"
    # The original client also retries after expiration.
    assert isinstance(a.inner, OpenAICompatible)
    a.inner.client = httpx.Client(transport=good)
    assert a.complete(REQUEST).text == "ok"
    first.close()
    second.close()


def test_refused_first_key_rotates_and_other_process_skips_it(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "limits.db"
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers["authorization"])
        return httpx.Response(403, text="requires a card") if sent[-1] == "Bearer bad" else reply()

    transport = httpx.MockTransport(handle)
    for _ in range(2):
        conn = db.open_db(path)
        assert api(conn, clock, transport, ["bad", "good"]).complete(REQUEST).text == "ok"
        conn.close()
    assert sent == ["Bearer bad", "Bearer good", "Bearer good"]


@pytest.mark.parametrize("wait", [5, 60])
def test_reported_tokens_reserve_across_connections(
    tmp_path: Path, clock: FakeClock, wait: int
) -> None:
    clock.now = datetime(2099, 1, 1, tzinfo=UTC)
    path = tmp_path / "limits.db"
    first, second = db.open_db(path), db.open_db(path)
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return reply(
            {"x-ratelimit-remaining-tokens": "250", "x-ratelimit-reset-tokens": f"{wait}s"}
        )

    transport = httpx.MockTransport(handle)
    assert api(first, clock, transport).complete(REQUEST).text == "ok"
    assert api(second, clock, transport).complete(REQUEST).text == "ok"  # 101 reserved
    assert api(first, clock, transport).complete(REQUEST).text == "ok"  # another 101
    start = clock()
    if wait > 10:
        with pytest.raises(RateLimited):
            api(second, clock, transport).complete(REQUEST)
        assert len(sent) == 3
    else:
        assert api(second, clock, transport).complete(REQUEST).text == "ok"
        assert (clock() - start).total_seconds() == wait
    first.close()
    second.close()


def test_low_token_key_rotates_without_blocking_other_keys(
    tmp_path: Path, clock: FakeClock
) -> None:
    conn = db.open_db(tmp_path / "limits.db")
    buckets = Buckets(conn, clock)
    buckets.report_tokens("tokens:groq/m:" + key_id("low"), 0, clock() + timedelta(seconds=60))
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers["authorization"])
        return reply()

    assert (
        api(conn, clock, httpx.MockTransport(handle), ["low", "good"]).complete(REQUEST).text
        == "ok"
    )
    assert sent == ["Bearer good"]
    assert buckets.blocked_until("groq/m") is None
    conn.close()


@pytest.mark.parametrize("status", [400, 401])
def test_bad_request_and_bad_auth_do_not_disable_model(
    tmp_path: Path, clock: FakeClock, status: int
) -> None:
    conn = db.open_db(tmp_path / "limits.db")
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, text="invalid request" if status == 400 else "invalid key")

    limited = api(conn, clock, httpx.MockTransport(handle))
    with pytest.raises(ProviderRejected):
        limited.complete(REQUEST)
    assert (
        conn.execute("SELECT scope FROM rate_buckets WHERE scope LIKE 'unavailable:%'").fetchall()
        == []
    )
    if status == 401:
        with pytest.raises(ProviderRejected):
            limited.complete(REQUEST)
        assert len(sent) == 1  # whole-key refusal remains process-local, as before
    conn.close()


@pytest.mark.parametrize(
    "remaining,reset",
    [
        ("bad", "1m"),
        ("nan", "1m"),
        ("-1", "1m"),
        ("50", "bad"),
        ("50", "inf"),
        ("50", "99999999999999999999"),
    ],
)
def test_unusable_token_headers_do_not_prevent_calls(
    tmp_path: Path, clock: FakeClock, remaining: str, reset: str
) -> None:
    conn = db.open_db(tmp_path / "limits.db")
    limited = api(
        conn,
        clock,
        httpx.MockTransport(
            lambda r: reply(
                {"x-ratelimit-remaining-tokens": remaining, "x-ratelimit-reset-tokens": reset}
            )
        ),
    )
    for _ in range(3):
        assert limited.complete(REQUEST).text == "ok"
    assert (
        conn.execute("SELECT scope FROM rate_buckets WHERE scope LIKE 'tokens:%'").fetchall() == []
    )
    conn.close()


def test_token_wait_falls_back_without_sending(tmp_path: Path, clock: FakeClock) -> None:
    from desman.llm.fake import FakeLLM
    from desman.llm.router import Fallback

    conn = db.open_db(tmp_path / "limits.db")
    buckets = Buckets(conn, clock)
    until = clock() + timedelta(minutes=1)
    buckets.report_tokens("tokens:groq/m:" + key_id("secret"), 10, until)
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return reply()

    first = api(conn, clock, httpx.MockTransport(handle))
    lineup = Fallback([(first, None), (FakeLLM("fallback"), None)])
    assert lineup.complete(REQUEST).text == "fallback"
    assert sent == []
    assert buckets.blocked_until("groq/m") is None  # no provider-wide block from a preflight skip
    clock.advance(61)
    assert first.complete(REQUEST).text == "ok"
    conn.close()
