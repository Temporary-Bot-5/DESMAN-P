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
from desman.llm import check, router
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
    with pytest.raises(ProviderRejected, match="rejected with 403"):  # the provider's own answer
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


def test_every_free_provider_is_preferred_and_no_paid_one_is() -> None:
    assert set(providers.PREFERENCE) == {p.id for p in providers.FREE}
    assert all(p.paid for p in providers.PAID) and not any(p.paid for p in providers.FREE)
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


# ── Review regressions (PR #18) ──────────────────────────────────────────────


def test_a_limited_key_wins_over_a_refused_one() -> None:
    llm, _ = client(
        by_key(
            {
                KEY_A: iter([httpx.Response(429, headers={"retry-after": "5"})]),
                KEY_B: iter([httpx.Response(401)]),
            }
        ),
        [KEY_A, KEY_B],
    )
    with pytest.raises(RateLimited) as raised:  # a pause, not the operator's inbox
        llm.complete(REQUEST)
    assert raised.value.retry_after == 5


def test_the_soonest_key_decides_the_pause() -> None:
    llm, _ = client(
        by_key(
            {
                KEY_A: iter([httpx.Response(429, headers={"retry-after": "5"})]),
                KEY_B: iter([httpx.Response(429, headers={"retry-after": "3600"})]),
            }
        ),
        [KEY_A, KEY_B],
    )
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.retry_after == 5


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, json={"error": {"message": "daily quota exceeded"}}),
        httpx.Response(402, json={"error": {"message": "no credits"}}),
    ],
)
def test_a_quota_without_a_hint_pauses_long(response: httpx.Response) -> None:
    llm, _ = client(lambda r: response)
    with pytest.raises(QuotaExhausted) as raised:
        llm.complete(REQUEST)
    assert raised.value.paused_until(NOW) == NOW + timedelta(hours=1)
    with pytest.raises(QuotaExhausted) as again:  # the key stays aside as long: no new request
        llm.complete(REQUEST)
    assert again.value.paused_until(NOW) == NOW + timedelta(hours=1)


def test_tokens_of_an_unusable_reply_are_counted() -> None:
    body = {"choices": [], "usage": {"prompt_tokens": 200, "completion_tokens": 4}}
    llm, records = client(lambda r: httpx.Response(200, json=body))
    with pytest.raises(ProviderUnavailable):
        llm.complete(REQUEST)
    assert (records[-1].prompt_tokens, records[-1].completion_tokens) == (200, 4)


def test_retry_after_as_an_http_date() -> None:
    later = datetime.now(UTC) + timedelta(hours=1)
    header = later.strftime("%a, %d %b %Y %H:%M:%S GMT")
    llm, _ = client(lambda r: httpx.Response(429, headers={"retry-after": header}))
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.retry_after is not None and 3500 < raised.value.retry_after <= 3600
    past = "Tue, 01 Jan 2002 00:00:00 GMT"
    llm, _ = client(lambda r: httpx.Response(429, headers={"retry-after": past}))
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.retry_after == 0


def test_reset_headers_of_a_used_up_limit() -> None:
    headers = {
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "2m52.8s",
        "x-ratelimit-remaining-tokens": "500",
        "x-ratelimit-reset-tokens": "1h0m0s",  # not used up: not the reason
    }
    llm, _ = client(lambda r: httpx.Response(429, headers=headers))
    with pytest.raises(RateLimited) as raised:
        llm.complete(REQUEST)
    assert raised.value.retry_after == pytest.approx(172.8)


def test_an_echoed_key_is_redacted() -> None:
    echo = httpx.Response(401, json={"error": {"message": f"Invalid API key: {KEY_A}"}})
    llm, records = client(lambda r: echo)
    with pytest.raises(ProviderRejected) as raised:
        llm.complete(REQUEST)
    assert KEY_A not in str(raised.value) and "<key>" in str(raised.value)
    assert all(KEY_A not in (r.error or "") for r in records)


@pytest.mark.parametrize(
    ("host", "response"),
    [
        ("api.groq.com", httpx.Response(200, text="<html>proxy login</html>")),
        ("api.groq.com", httpx.Response(200, json={"object": "list"})),
        ("openrouter.ai", httpx.Response(200, json={"data": "nope"})),
    ],
)
def test_keys_check_needs_a_real_answer(
    mock_http: Callable[[Handler], None], host: str, response: httpx.Response
) -> None:
    env = "OPENROUTER_API_KEY" if host == "openrouter.ai" else "GROQ_API_KEY"
    ConfigManager(default_config_path()).save(make_config(api_keys={env: KEY_A}))
    mock_http(lambda r: response)
    result = desman("keys", "check", "--json")
    assert result.exit_code == 1
    [check] = json.loads(result.output)
    assert not check["ok"] and "could not be verified" in check["detail"]


# ── Paid providers ───────────────────────────────────────────────────────────

ANTHROPIC_KEY = "sk-ant-" + "c" * 40


def test_paid_providers_are_never_chosen_by_themselves() -> None:
    config = make_config(api_keys={"OPENAI_API_KEY": "sk-paid"})
    with pytest.raises(router.NoProvider, match="only paid providers.*--provider openai"):
        router.choose(config)
    assert router.choose(config, "openai").id == "openai"  # by name: yes
    both = make_config(api_keys={"OPENAI_API_KEY": "sk-paid", "COHERE_API_KEY": "co"})
    assert router.choose(both).id == "cohere"  # the free one, though OpenAI comes first in lists


def test_anthropic_requests_follow_its_rules(mock_http: Callable[[Handler], None]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply()

    mock_http(handler)
    config = make_config(api_keys={"ANTHROPIC_API_KEY": ANTHROPIC_KEY})
    llm, choice = router.build(config, "anthropic")
    assert choice.model == "claude-haiku-5-5"
    llm.complete(REQUEST)
    [sent] = seen
    assert str(sent.url) == "https://api.anthropic.com/v1/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {ANTHROPIC_KEY}"
    assert sent.headers["x-api-key"] == ANTHROPIC_KEY
    assert sent.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(sent.content)
    assert "temperature" not in body  # Claude 4.7+ rejects anything but its default
    assert body["max_tokens"] == 4096  # a request without one gets the provider's default
    llm.complete(REQUEST.model_copy(update={"max_tokens": 50}))
    assert json.loads(seen[-1].content)["max_tokens"] == 50


def test_a_paid_account_out_of_money_goes_to_a_person() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "Insufficient Balance"}})

    llm, _ = client(handler)
    llm.paid = True
    with pytest.raises(ProviderRejected, match="balance is used up") as caught:
        llm.complete(REQUEST)
    assert caught.value.kind is FailureKind.PERMANENT


def test_running_on_a_paid_provider_says_so(mock_http: Callable[[Handler], None]) -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"DEEPSEEK_API_KEY": "sk-ds"}))

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return reply('{"verdict": "accept"}' if prompt.startswith("Review") else '{"answer": 42}')

    mock_http(handler)
    refused = desman("run", "g")
    assert refused.exit_code == 1 and "--provider deepseek" in refused.output
    result = desman("run", "g", "--provider", "deepseek")
    assert result.exit_code == 0, result.output
    assert "DeepSeek is a paid provider: every call is billed." in result.output
    shown = desman("config", "show").output
    assert "DeepSeek · paid" in shown and "GROQ · paid" not in shown


def test_key_check_sends_the_providers_own_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5"}]})

    config = make_config(api_keys={"ANTHROPIC_API_KEY": ANTHROPIC_KEY})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        [result] = check.check(config, providers.find("anthropic"), http)
    assert result.ok and result.default_model_listed
    assert str(seen[0].url) == "https://api.anthropic.com/v1/models"
    assert seen[0].headers["x-api-key"] == ANTHROPIC_KEY
    assert seen[0].headers["anthropic-version"] == "2023-06-01"


# ── `desman usage` readability (owner's report) ─────────────────────────────


def test_usage_lists_every_key_names_removed_ones_and_reads_limits() -> None:
    ConfigManager(default_config_path()).save(
        make_config(
            api_keys={
                "GROQ_API_KEY": [{"value": KEY_A}, {"value": KEY_B}],
                "MISTRAL_API_KEY": "mstr_key",
            }
        )
    )
    conn = db.open_db()
    record = calls.recorder(conn)
    groq_limits = {
        "x-ratelimit-limit-requests": "1000",
        "x-ratelimit-remaining-requests": "998",
        "x-ratelimit-reset-requests": "2m52.8s",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-remaining-tokens": "7319",
        "retry-after": "3",
    }
    record(CallRecord("groq", "m", key_id(KEY_A), "p", "r", 10, 5, 5, 200, None, None, groq_limits))
    record(CallRecord("groq", "m", "0123456789ab", "p", None, 0, 0, 5, 401, "permanent", "x"))
    mistral_limits = {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}
    record(
        CallRecord(
            "mistral",
            "m",
            key_id("mstr_key"),
            "p",
            None,
            0,
            0,
            5,
            429,
            "permanent",
            "rate limited",
            mistral_limits,
        )
    )
    conn.close()

    shown = desman("usage").output
    lines = shown.splitlines()
    usage_table = lines[: next(i for i, line in enumerate(lines) if line.startswith("Limits"))]
    groq = [line for line in usage_table if line.startswith("groq")]
    assert [line.split()[1] for line in groq] == ["#1", "#2", "removed"]
    assert groq[1].split()[3] == "0"  # the second key: listed, no calls
    assert "removed (0123…)" in groq[2] and "no longer in the config" in shown
    assert any("requests" in line and "998 / 1000" in line and "2m52.8s" in line for line in lines)
    assert any("tokens" in line and "7319 / 8000" in line for line in lines)
    assert any("retry-after" in line and "3" in line for line in lines)  # not a limit: as is
    assert "the limit is 0, so this account has no access" in shown
    assert KEY_A not in shown and KEY_B not in shown


def test_limits_group_headers_by_limit() -> None:
    found, other = calls.limits(
        {"X-RateLimit-Limit-Tokens": "10", "x-ratelimit-remaining-tokens": "4", "via": "edge"}
    )
    assert found == [calls.Limit("tokens", "4", "10", None)]
    assert other == {"via": "edge"}
    assert not found[0].exhausted_for_good
    assert calls.Limit("req-minute", "0", "0", None).exhausted_for_good
    assert not calls.Limit("x", "0", "n/a", None).exhausted_for_good


def test_a_key_with_a_zero_limit_is_set_aside_for_the_next() -> None:
    used: list[str] = []
    zero = {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}

    def handler(request: httpx.Request) -> httpx.Response:
        used.append(request.headers["authorization"])
        if request.headers["authorization"] == f"Bearer {KEY_A}":
            return httpx.Response(429, headers=zero)
        return reply()

    llm, _ = client(handler, [KEY_A, KEY_B])
    llm.complete(REQUEST)
    llm.complete(REQUEST)
    assert used == [f"Bearer {KEY_A}", f"Bearer {KEY_B}", f"Bearer {KEY_B}"]  # A not again


def test_a_zero_limit_seen_before_a_success_is_history() -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"MISTRAL_API_KEY": "mk"}))
    conn = db.open_db()
    record = calls.recorder(conn)
    zero = {"x-ratelimit-limit-req-minute": "0", "x-ratelimit-remaining-req-minute": "0"}
    record(
        CallRecord("mistral", "m", key_id("mk"), "p", None, 0, 0, 5, 429, "permanent", "x", zero)
    )
    record(CallRecord("mistral", "m", key_id("mk"), "p", "r", 3, 2, 5, 200, None, None))
    [u] = calls.usage(conn)
    conn.close()
    assert u.limits == zero and not u.limits_current
    shown = desman("usage").output
    assert "the limit is 0" not in shown
    assert "Grey rows come from an earlier call" in shown
