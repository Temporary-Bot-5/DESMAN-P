"""Shared account access and reported token windows (T.6h F54/F55)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
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


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = db.open_db(tmp_path / "test.db")
    yield connection
    connection.close()


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
    [(429, {"x-ratelimit-limit-requests": "0"}), (403, {}), (400, {})],
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
    with pytest.raises(ProviderRejected, match="credit card") as remembered:
        b.complete(REQUEST)
    assert remembered.value.status == status
    assert len(sent) == 1
    print_model_blocks(path)
    output = capsys.readouterr().out
    assert "unavailable to key #" in output and "credit card" in output and "secret" not in output
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
def test_reported_tokens_wait_across_connections(
    tmp_path: Path, clock: FakeClock, wait: int
) -> None:
    path = tmp_path / "limits.db"
    first, second = db.open_db(path), db.open_db(path)
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return reply({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": f"{wait}s"})

    transport = httpx.MockTransport(handle)
    assert api(first, clock, transport).complete(REQUEST).text == "ok"
    start = clock()
    if wait > 10:
        with pytest.raises(RateLimited):
            api(second, clock, transport).complete(REQUEST)
        assert len(sent) == 1
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


def test_latest_token_report_wins_without_deductions(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    buckets = Buckets(conn, clock)
    scope = "tokens:groq/m:fingerprint"
    buckets.report_tokens(scope, 200, clock() + timedelta(seconds=60))
    for _ in range(5):
        assert buckets.take_tokens(scope, 150) == 0
    buckets.report_tokens(scope, 300, clock() + timedelta(seconds=30))
    assert buckets.take_tokens(scope, 250) == 0
    assert buckets.take_tokens(scope, 301) == 30
    buckets.report_tokens(scope, 10, clock() + timedelta(seconds=5))
    assert buckets.take_tokens(scope, 11) == 5


@pytest.mark.parametrize("status", [402, 403, 404])
def test_other_refusals_are_not_shared_as_account_blocks(
    conn: sqlite3.Connection, clock: FakeClock, status: int
) -> None:
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, text="balance or generic refusal")

    transport = httpx.MockTransport(handle)
    for _ in range(2):
        limited = api(conn, clock, transport)
        assert isinstance(limited.inner, OpenAICompatible)
        limited.inner.paid = True
        with pytest.raises(ProviderRejected):
            limited.complete(REQUEST)
    assert not conn.execute("SELECT * FROM model_refusals").fetchall()
    assert len(sent) == (1 if status == 404 else 2)
    if status == 404:
        assert Buckets(conn, clock).blocked_until("gone:groq/m") is not None


def test_after_key_db_failure_keeps_completion_and_journal(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    limited = api(conn, clock, httpx.MockTransport(lambda r: reply()))
    assert isinstance(limited.inner, OpenAICompatible)
    events = []
    limited.inner.on_call = lambda r: events.append("journal")

    def fail(*args: object) -> None:
        events.append("limits")
        raise sqlite3.OperationalError("locked")

    limited.inner.after_key = fail
    assert limited.complete(REQUEST).text == "ok"
    assert events == ["journal", "limits"]


def test_keys_check_clears_only_checked_fingerprints(
    conn: sqlite3.Connection, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from desman import providers
    from desman.config import Config
    from desman.config_cli import check_keys
    from desman.llm import check

    path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
    for scope in ("groq/m:checked", "groq/m:other", "mistral/m:checked"):
        conn.execute(
            "INSERT INTO model_refusals VALUES (?, 403, 'requires a card', ?)",
            (scope, db.to_iso(clock() + timedelta(hours=24))),
        )
    monkeypatch.setattr(db, "default_db_path", lambda: path)
    monkeypatch.setattr(providers.Provider, "is_ready", lambda *args: True)
    monkeypatch.setattr(
        check, "check", lambda *args: [check.KeyCheck("groq", 1, "checked", True, 200, "ok")]
    )
    check_keys(Config(), "groq")
    assert [r[0] for r in conn.execute("SELECT scope FROM model_refusals ORDER BY scope")] == [
        "groq/m:other",
        "mistral/m:checked",
    ]


@pytest.mark.parametrize("failure", ["pace", "timeout", "transport"])
def test_failed_send_does_not_spend_reported_tokens(
    conn: sqlite3.Connection, clock: FakeClock, failure: str
) -> None:
    from desman.llm.base import ProviderTimeout, ProviderUnavailable
    from desman.llm.limiter import Paced
    from desman.llm.router import estimated_tokens

    scope = "tokens:groq/m:" + key_id("secret")
    buckets = Buckets(conn, clock)
    buckets.report_tokens(scope, 200, clock() + timedelta(seconds=60))

    def handle(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        raise httpx.ConnectError("unreachable")

    limited = api(conn, clock, httpx.MockTransport(handle))
    assert isinstance(limited.inner, OpenAICompatible)
    estimates = []
    original = limited.inner.before_key
    assert original is not None

    def before(fingerprint: str, model: str, estimate: int) -> None:
        estimates.append(estimate)
        original(fingerprint, model, estimate)

    limited.inner.before_key = before
    if failure == "pace":
        buckets.block("groq/m", clock() + timedelta(seconds=60))
    with pytest.raises((Paced, ProviderTimeout, ProviderUnavailable)):
        limited.complete(REQUEST)
    assert estimates == [estimated_tokens(REQUEST)]
    assert (
        conn.execute("SELECT tokens FROM rate_buckets WHERE scope = ?", (scope,)).fetchone()[0]
        == 200
    )


@pytest.mark.parametrize("status", [400, 403])
@pytest.mark.parametrize(
    "words",
    [
        "requires a plan",
        "payment required",
        "model is not available to your account",
        "requires a valid credit card",
    ],
)
def test_only_clear_account_access_words_are_remembered(status: int, words: str) -> None:
    from desman.llm.openai_compat import account_unavailable

    assert account_unavailable(ProviderRejected("groq", status, words))
    assert not account_unavailable(ProviderRejected("groq", status, "content policy refusal"))


def test_expired_report_replaces_old_token_window(
    conn: sqlite3.Connection, clock: FakeClock
) -> None:
    limited = api(conn, clock, httpx.MockTransport(lambda r: reply()))
    scope = "tokens:groq/m:" + key_id("secret")
    limited.buckets.report_tokens(scope, 0, clock() + timedelta(seconds=60))
    limited.after_key(
        key_id("secret"),
        "m",
        {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "0"},
        None,
    )
    assert limited.complete(REQUEST).text == "ok"
