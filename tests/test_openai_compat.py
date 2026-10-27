"""The OpenAI-compatible client, the router, the call log and `keys check`, against a mock
HTTP transport. No test here touches the network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from desman import providers
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import calls, db
from desman.core.retry import FailureKind
from desman.llm import router
from desman.llm.base import (
    Message,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
)
from desman.llm.openai_compat import CallRecord, OpenAICompatible, key_id
from helpers import make_config

KEY_A = "gsk_" + "a" * 48
KEY_B = "gsk_" + "b" * 48
REQUEST = Request(messages=[Message(role="user", content="Execute subtask:\n{}")])
NOW = datetime(2026, 1, 1, tzinfo=UTC)

Handler = Callable[[httpx.Request], httpx.Response]


def reply(text: str = '{"ok": true}', **extra: Any) -> httpx.Response:
    body = {
        "model": "m-served",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
        **extra,
    }
    return httpx.Response(200, json=body, headers={"x-ratelimit-remaining-requests": "99"})


def client(
    handler: Handler, keys: list[str] | None = None
) -> tuple[OpenAICompatible, list[CallRecord]]:
    records: list[CallRecord] = []
    llm = OpenAICompatible(
        "groq",
        "https://api.example/v1/",
        keys or [KEY_A],
        "m-default",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        on_call=records.append,
        clock=lambda: NOW,
    )
    return llm, records


def test_success_and_what_is_sent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply()

    llm, records = client(handler)
    done = llm.complete(REQUEST.model_copy(update={"max_tokens": 50}))
    assert (done.text, done.provider, done.model) == ('{"ok": true}', "groq", "m-served")
    assert (done.usage.prompt_tokens, done.usage.completion_tokens) == (12, 5)
    [sent] = seen
    assert str(sent.url) == "https://api.example/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {KEY_A}"
    body = json.loads(sent.content)
    assert body == {
        "model": "m-default",
        "messages": [{"role": "user", "content": "Execute subtask:\n{}"}],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    [record] = records
    assert record.key_id == key_id(KEY_A) and KEY_A not in repr(record)
    assert (record.prompt_tokens, record.completion_tokens, record.error_kind) == (12, 5, None)
    assert record.limits == {"x-ratelimit-remaining-requests": "99"}
    assert record.response_hash is not None and record.http_status == 200


@pytest.mark.parametrize(
    ("response", "error", "kind"),
    [
        (httpx.Response(401, json={"error": {"message": "Invalid API Key"}}), ProviderRejected, FailureKind.PERMANENT),
        (httpx.Response(400, json={"error": "model not found"}), ProviderRejected, FailureKind.PERMANENT),
        (httpx.Response(404, text="no such model"), ProviderRejected, FailureKind.PERMANENT),
        (httpx.Response(429, headers={"retry-after": "7"}), RateLimited, FailureKind.LIMIT),
        (httpx.Response(429, json={"error": {"message": "Please try again in 1.5s"}}), RateLimited, FailureKind.LIMIT),
        (httpx.Response(429, json={"error": {"message": "You exceeded your current quota"}}), QuotaExhausted, FailureKind.LIMIT),
        (httpx.Response(429, json={"error": {"message": "Rate limit exceeded: free-models-per-day"}}), QuotaExhausted, FailureKind.LIMIT),
        (httpx.Response(402, json={"error": {"message": "Insufficient credits"}}), QuotaExhausted, FailureKind.LIMIT),
        (httpx.Response(429, headers={"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}), ProviderRejected, FailureKind.PERMANENT),
        (httpx.Response(503, text="overloaded"), ProviderUnavailable, FailureKind.TRANSIENT),
        (httpx.Response(408), ProviderUnavailable, FailureKind.TRANSIENT),
        (httpx.Response(200, json={"choices": []}), ProviderUnavailable, FailureKind.TRANSIENT),
        (httpx.Response(200, text="<html>"), ProviderUnavailable, FailureKind.TRANSIENT),
        (httpx.Response(200, json={"choices": [{"message": {"content": None}}]}), ProviderUnavailable, FailureKind.TRANSIENT),
    ],
)  # fmt: skip
def test_http_outcomes_become_typed_errors(
    response: httpx.Response, error: type[Exception], kind: FailureKind
) -> None:
    llm, records = client(lambda r: response)
    with pytest.raises(error) as raised:
        llm.complete(REQUEST)
    assert raised.value.kind is kind  # type: ignore[attr-defined]
    assert records[-1].error_kind == kind.value and KEY_A not in str(raised.value)


def test_retry_after_reaches_the_pause() -> None:
    llm, _ = client(lambda r: httpx.Response(429, headers={"retry-after": "7"}))
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.paused_until(NOW) == NOW + timedelta(seconds=7)
    llm, _ = client(
        lambda r: httpx.Response(429, json={"error": {"message": "try again in 250ms"}})
    )
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.retry_after == 0.25


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (httpx.ReadTimeout("slow"), ProviderTimeout),
        (httpx.ConnectError("down"), ProviderUnavailable),
    ],
)
def test_network_failures(exc: Exception, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    llm, records = client(handler)
    with pytest.raises(error):
        llm.complete(REQUEST)
    assert records[-1].http_status is None and records[-1].error_kind == "transient"


def by_key(responses: dict[str, Iterator[httpx.Response]]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"].removeprefix("Bearer ")
        return next(responses[key])

    return handler


def test_a_refused_key_is_skipped_for_good() -> None:
    llm, records = client(
        by_key({KEY_A: iter([httpx.Response(401)]), KEY_B: iter([reply(), reply()])}),
        [KEY_A, KEY_B],
    )
    assert llm.complete(REQUEST).text == '{"ok": true}'
    assert llm.complete(REQUEST).text == '{"ok": true}'  # A is not tried again
    assert [r.key_id for r in records] == [key_id(KEY_A), key_id(KEY_B), key_id(KEY_B)]


def test_a_limited_key_waits_and_the_next_one_works() -> None:
    llm, records = client(
        by_key(
            {
                KEY_A: iter([httpx.Response(429, headers={"retry-after": "30"})]),
                KEY_B: iter([reply(), httpx.Response(429, headers={"retry-after": "5"})]),
            }
        ),
        [KEY_A, KEY_B],
    )
    assert llm.complete(REQUEST).text == '{"ok": true}'
    with pytest.raises(RateLimited):  # B limited too: the last key's error
        llm.complete(REQUEST)
    with pytest.raises(RateLimited) as raised:  # both aside: no HTTP call, the soonest wins
        llm.complete(REQUEST)
    assert raised.value.retry_after == 5
    assert len(records) == 3


def test_a_bad_request_does_not_burn_the_other_keys() -> None:
    llm, records = client(
        lambda r: httpx.Response(400, json={"error": "bad model"}), [KEY_A, KEY_B]
    )
    with pytest.raises(ProviderRejected):
        llm.complete(REQUEST)
    assert len(records) == 1


def test_every_key_refused() -> None:
    llm, _ = client(lambda r: httpx.Response(403), [KEY_A, KEY_B])
    with pytest.raises(ProviderRejected):
        llm.complete(REQUEST)
    with pytest.raises(ProviderRejected, match="every key was refused"):
        llm.complete(REQUEST)


# ── Router ────────────────────────────────────────────────────────────────────


def test_router_choice() -> None:
    config = make_config(api_keys={"MISTRAL_API_KEY": "m", "GROQ_API_KEY": "g"})
    assert router.choose(config).id == "groq"  # preference order, not file order
    assert router.choose(config, "mistral").id == "mistral"
    with pytest.raises(router.NoProvider, match="desman keys add google"):
        router.choose(config, "google")
    with pytest.raises(router.NoProvider, match="local"):
        router.choose(config, "ollama")
    with pytest.raises(router.NoProvider, match="no provider is configured"):
        router.choose(make_config())
    cloudflare = make_config(
        api_keys={"CLOUDFLARE_API_TOKEN": "t"}, settings={"CLOUDFLARE_ACCOUNT_ID": "acc1"}
    )
    llm, choice = router.build(cloudflare, model="@cf/x")
    assert llm.base_url == "https://api.cloudflare.com/client/v4/accounts/acc1/ai/v1"
    assert choice.model == "@cf/x" and llm.model == "@cf/x"


def test_every_preferred_provider_exists_and_is_complete() -> None:
    ids = {p.id for p in providers.CLOUD}
    assert set(providers.PREFERENCE) == ids
    for p in providers.CLOUD:
        assert p.base_url.startswith("https://") and p.default_model and p.limits_url


# ── Through the CLI ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    real = httpx.Client

    def install(handler: Handler) -> None:
        monkeypatch.setattr(
            httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
        )

    return install


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def test_run_on_a_real_provider_logs_its_calls(mock_http: Callable[[Handler], None]) -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"GROQ_API_KEY": KEY_A}))

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return reply('{"verdict": "accept"}' if prompt.startswith("Review") else '{"answer": 42}')

    mock_http(handler)
    result = desman("run", "answer the question")
    assert result.exit_code == 0, result.output
    assert "Using GROQ · openai/gpt-oss-20b (1 key)" in result.output

    conn = db.open_db(db.default_db_path())
    [row] = calls.usage(conn)
    conn.close()
    assert (row.provider, row.key_id, row.calls, row.failed) == ("groq", key_id(KEY_A), 2, 0)
    assert (row.prompt_tokens, row.completion_tokens) == (24, 10)

    shown = desman("usage")
    assert shown.exit_code == 0 and "#1 gsk_…aaaa" in shown.output and "24" in shown.output
    listed = desman("keys")
    assert listed.exit_code == 0
    line = next(x for x in listed.output.splitlines() if x.startswith("GROQ"))
    assert line.split()[-2:] == ["2", "34"]  # calls and tokens in the last 24 hours
    assert json.loads(desman("usage", "--json").output)[0]["key"] == "#1 gsk_…aaaa"
    assert desman("usage", "--since", "1x").exit_code == 2


def test_run_refuses_fake_and_real_together(tmp_path: Any) -> None:
    script = tmp_path / "s.json"
    script.write_text('{"steps": []}')
    result = desman("run", "g", "--fake-llm", str(script), "--provider", "groq")
    assert result.exit_code == 2 and "drop --provider" in result.output


def test_keys_check(mock_http: Callable[[Handler], None]) -> None:
    manager = ConfigManager(default_config_path())
    manager.save(make_config(api_keys={"GROQ_API_KEY": KEY_A, "OPENROUTER_API_KEY": KEY_B}))
    with manager.edit() as config:
        config.add_key("GROQ_API_KEY", "gsk_" + "c" * 48)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"  # never a completion: no tokens spent
        key = request.headers["authorization"].removeprefix("Bearer ")
        if request.url.host == "openrouter.ai":
            assert request.url.path == "/api/v1/key"
            data = {"is_free_tier": True, "free_model_daily_requests": {"used": 3, "limit": 50}}
            return httpx.Response(200, json={"data": data})
        if key == KEY_A:
            return httpx.Response(
                200,
                json={"data": [{"id": "openai/gpt-oss-20b"}, {"id": "other"}]},
                headers={"x-ratelimit-limit-requests": "1000"},
            )
        return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

    mock_http(handler)
    result = desman("keys", "check")
    assert result.exit_code == 1, result.output  # one key is refused
    assert "free requests today: 3 of 50" in result.output
    assert "x-ratelimit-limit-requests: 1000" in result.output
    out = {
        (r["provider"], r["number"]): r
        for r in json.loads(desman("keys", "check", "--json").output)
    }
    assert out[("groq", 1)]["ok"] and out[("groq", 1)]["default_model_listed"]
    assert out[("groq", 1)]["models"] == 2
    assert not out[("groq", 2)]["ok"] and out[("groq", 2)]["http_status"] == 401
    assert out[("openrouter", 1)]["ok"]
    for output in (result.output, desman("keys", "check", "--json").output):
        assert KEY_A not in output and KEY_B not in output
    assert desman("keys", "check", "groq").exit_code == 1
    assert desman("keys", "check", "mistral").exit_code == 1  # not configured


def test_recorder_writes_append_only_rows() -> None:
    conn = db.open_db(":memory:")
    calls.recorder(conn)(
        CallRecord("groq", "m", "abc", "p", None, 0, 0, 5, 429, "limit", "rate limited", {"x": "1"})
    )
    [u] = calls.usage(conn)
    assert (u.calls, u.failed, u.limited, u.last_error, u.limits) == (
        1,
        1,
        1,
        "rate limited",
        {"x": "1"},
    )
    with pytest.raises(Exception, match="append-only"):
        conn.execute("UPDATE model_calls SET model = 'x'")
    assert calls.usage(conn, NOW + timedelta(days=36500)) == []  # the window excludes it
    conn.close()


def test_provider_messages_are_kept() -> None:
    google = httpx.Response(404, json=[{"error": {"code": 404, "message": "model retired"}}])
    llm, _ = client(lambda r: google)
    with pytest.raises(ProviderRejected, match="404: model retired"):
        llm.complete(REQUEST)
    upstream = {
        "error": {
            "message": "Provider returned error",
            "metadata": {"raw": "x is rate-limited upstream"},
        }
    }
    llm, _ = client(lambda r: httpx.Response(503, json=upstream))
    with pytest.raises(
        ProviderUnavailable, match="Provider returned error: x is rate-limited upstream"
    ):
        llm.complete(REQUEST)
