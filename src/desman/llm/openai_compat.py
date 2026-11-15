"""A client for cloud OpenAI-compatible APIs, llama.cpp and native Ollama chat.
Local servers take no key: their one "key" is empty.

One `complete()` is one chat completion. HTTP outcomes become the typed errors of
`llm/base.py`, so the retry policy can tell a bad key from a rate limit from an outage:

- 400/404/422 and 401/403: `ProviderRejected` (permanent), except a 400 refusing the model's
  own reply (`tool_use_failed`): `GenerationFailed` (quality, tried again differently);
- 402, or a 429 that speaks of a quota or a daily limit: `QuotaExhausted` (pause);
- other 429: `RateLimited`, with `Retry-After` when the provider sends it (pause);
- 408, 5xx, timeouts, connection errors, a 200 without a usable reply: transient.

A provider may have several keys. A key that is refused (401/403) or limited (429/402) is set
aside and the next one is tried within the same call; the error of the last key is raised
when none works. Limited keys stay aside for this process until their pause ends.

Every HTTP attempt is reported to `on_call` (a `CallRecord`), which the CLI stores in
`model_calls`. The record holds hashes of prompt and reply and a fingerprint of the key,
never the key or the text.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import ValidationError

from desman.llm.base import (
    Completion,
    GenerationFailed,
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
    RequestTooLarge,
    Usage,
)
from desman.llm.tool_calls import ToolCall, normalize_tool_calls

DEFAULT_TIMEOUT = 60.0
DEFAULT_LIMIT_PAUSE = 60.0  # a rate limit without a hint
# A used-up quota without a reset hint: daily quotas reset hours later, so probing every minute
# would only burn requests. An hour is a cheap, conservative probe.
DEFAULT_QUOTA_PAUSE = 3600.0
_QUOTA_WORDS = re.compile(r"quota|per[ -]day|daily|exhausted|insufficient|credits", re.I)
_RETRY_IN = re.compile(r"(?:try again|retry) in (?:about )?(\d+(?:\.\d+)?)\s*(ms|s)", re.I)


def key_id(key: str) -> str:
    """A stable fingerprint of a key: tells keys apart in usage, reveals nothing."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CallRecord:
    provider: str
    model: str
    key_id: str | None
    prompt_hash: str
    response_hash: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    http_status: int | None
    error_kind: str | None  # None: success
    error: str | None
    limits: dict[str, str] = field(default_factory=dict)
    error_type: str | None = None
    """The error's class: `RateLimited` and `QuotaExhausted` are both of kind "limit"."""


class OpenAICompatible:
    """One provider's chat API, with key rotation and shared HTTP journaling."""

    def __init__(
        self,
        provider: str,
        base_url: str,
        keys: Sequence[str],
        model: str,
        *,
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        on_call: Callable[[CallRecord], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        auth: Sequence[tuple[str, str]] = (("Authorization", "Bearer {key}"),),
        send_temperature: bool = True,
        max_tokens: int | None = None,
        paid: bool = False,
        context: int | None = None,
    ) -> None:
        if not keys:
            raise ValueError(f"{provider}: no API key")
        self.name = provider
        self.base_url = base_url.rstrip("/")
        self.keys = list(keys)
        self.model = model
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)
        self.on_call = on_call
        self.clock = clock
        self.monotonic = monotonic
        self._aside_until: dict[int, datetime] = {}  # key index -> limited until
        self._preflight: set[int] = set()
        self._quota: set[int] = set()  # aside for a used-up quota, not a rate limit
        self._refused: set[int] = set()  # key indexes the provider refused
        self._refusal: ProviderRejected | None = None  # the latest, with the provider's words
        self.auth = tuple(auth)
        self.send_temperature = send_temperature
        self.max_tokens = max_tokens
        self.paid = paid
        self.prepare_context: Callable[[str], tuple[int, bool]] | None = None
        self.memory_risk = False
        self.context = context
        self.before_key: Callable[[str, str, int], None] | None = None
        self.after_key: Callable[[str, str, dict[str, str], ProviderError | None], None] | None = (
            None
        )
        self.pace: Callable[[], None] | None = None
        """Called before every request sent (one per key tried): the rate limiter
        (`llm.limiter.Limited`) waits there, or raises `RateLimited` and nothing is sent."""

    def complete(self, request: Request) -> Completion:
        model = request.model or self.model
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if self.send_temperature:
            body["temperature"] = request.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.max_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.name == "ollama":
            if self.prepare_context is not None:
                self.context, self.memory_risk = self.prepare_context(model)
            options = {"num_ctx": self.context or 16384}
            if self.send_temperature:
                options["temperature"] = body.pop("temperature")
            if "max_tokens" in body:
                options["num_predict"] = body.pop("max_tokens")
            body.update(stream=False, options=options)
            if body.pop("response_format", None) is not None:
                body["format"] = "json"
        prompt_hash = text_hash(json.dumps(body["messages"], ensure_ascii=False))

        now = self.clock()
        usable = [
            i
            for i in range(len(self.keys))
            if i not in self._refused and self._aside_until.get(i, now) <= now
        ]
        if not usable:
            raise self._all_aside(now)
        for i in usable:
            self._aside_until.pop(i, None)
            self._quota.discard(i)
            self._preflight.discard(i)
            try:
                if self.before_key is not None:
                    from desman.llm.router import estimated_tokens

                    estimate = estimated_tokens(
                        request.model_copy(update={"max_tokens": max_tokens})
                    )
                    self.before_key(key_id(self.keys[i]), model, estimate)
            except ProviderRejected as e:
                self._refusal = e
                continue
            except RateLimited as e:
                self._aside_until[i] = self.clock() + timedelta(
                    seconds=DEFAULT_LIMIT_PAUSE if e.retry_after is None else e.retry_after
                )
                self._preflight.add(i)
                continue
            if self.pace is not None:
                self.pace()  # request bucket limits apply to every key, outside key rotation
            try:
                return self._attempt(i, model, body, prompt_hash)
            except ProviderRejected as e:
                if e.status not in (401, 402, 403) and not account_unavailable(e):
                    raise
                if (
                    self.after_key is None
                    or e.status in (401, 402)
                    or (e.status == 403 and not account_unavailable(e))
                ):
                    self._refused.add(i)
                self._refusal = e
            except (RateLimited, QuotaExhausted) as e:
                if isinstance(e, QuotaExhausted):
                    self._quota.add(i)
                else:
                    self._quota.discard(i)
                self._aside_until[i] = e.paused_until(self.clock()) or self.clock() + timedelta(
                    seconds=DEFAULT_LIMIT_PAUSE
                )
        # No key worked. Decide from all keys, not from the last one tried: while any key
        # will come back, this is a pause until the soonest; only when every key was refused
        # is it permanent.
        raise self._all_aside(self.clock())

    def _all_aside(self, now: datetime) -> ProviderError:
        waiting = {i: until for i, until in self._aside_until.items() if i not in self._refused}
        if not waiting:
            return self._refusal or ProviderRejected(self.name, 401, "every key was refused")
        soonest = min(waiting, key=lambda i: waiting[i])
        if soonest in self._preflight:
            from desman.llm.limiter import Paced

            return Paced(self.name, retry_after=max(0.0, (waiting[soonest] - now).total_seconds()))
        if soonest in self._quota:
            return QuotaExhausted(self.name, waiting[soonest])
        return RateLimited(
            self.name, retry_after=max(0.0, (waiting[soonest] - now).total_seconds())
        )

    def _attempt(
        self, index: int, model: str, body: dict[str, Any], prompt_hash: str
    ) -> Completion:
        key = self.keys[index]
        started = self.monotonic()
        response: httpx.Response | None = None

        def record(
            error: ProviderError | None,
            completion: Completion | None = None,
        ) -> None:
            spent = (
                _usage(response, native=self.name == "ollama")
                if response is not None
                else Usage(prompt_tokens=0, completion_tokens=0)
            )
            if self.on_call is not None:
                self.on_call(
                    CallRecord(
                        provider=self.name,
                        model=completion.model if completion else model,
                        key_id=key_id(key) if key else None,
                        prompt_hash=prompt_hash,
                        response_hash=text_hash(completion.text) if completion else None,
                        # From the response, not the completion: a 200 without usable text may
                        # still have cost tokens.
                        prompt_tokens=spent.prompt_tokens,
                        completion_tokens=spent.completion_tokens,
                        latency_ms=max(0, round((self.monotonic() - started) * 1000)),
                        http_status=response.status_code if response is not None else None,
                        error_kind=error.kind.value if error else None,
                        error_type=type(error).__name__ if error else None,
                        error=str(error)[:300] if error else None,
                        limits=_limit_headers(response) if response is not None else {},
                    )
                )
            if self.after_key is not None:
                # Limit reporting must not discard the completion or its journal.
                with suppress(sqlite3.Error):
                    self.after_key(
                        key_id(key),
                        model,
                        _limit_headers(response) if response is not None else {},
                        error,
                    )

        try:
            with self.client.stream(
                "POST",
                (self.base_url.removesuffix("/v1") + "/api/chat")
                if self.name == "ollama"
                else f"{self.base_url}/chat/completions",
                json=body,
                headers={name: value.format(key=key) for name, value in self.auth},
                timeout=self.timeout,
            ) as response:
                chunks = []
                for chunk in response.iter_bytes():
                    if self.monotonic() - started >= self.timeout:
                        raise httpx.ReadTimeout("whole-call deadline exceeded")
                    chunks.append(chunk)
                response = httpx.Response(
                    response.status_code,
                    headers={k: v for k, v in response.headers.items() if k != "content-encoding"},
                    content=b"".join(chunks),
                )
        except httpx.TimeoutException:
            error: ProviderError = ProviderTimeout(self.name, self.timeout)
            record(error)
            raise error from None
        except httpx.TransportError as e:
            error = ProviderUnavailable(self.name, None, type(e).__name__)
            record(error)
            raise error from None

        if response.status_code == 200:
            try:
                completion = self._parse(response, model)
            except ProviderError as e:
                record(e)
                raise
            record(None, completion)
            return completion
        if response.status_code == 400 and _code(response) == "tool_use_failed":
            try:
                failed = response.json()["error"]["failed_generation"]
            except (ValueError, KeyError, TypeError):
                failed = None
            recovered = _tool_text(failed)
            if recovered is not None:
                completion = Completion(
                    text=recovered, provider=self.name, model=model, usage=_usage(response)
                )
                record(None, completion)
                return completion
        error = self._error(response)
        error.model = model
        record(error)
        raise error

    def _parse(self, response: httpx.Response, model: str) -> Completion:
        try:
            data = response.json()
            choice = data if self.name == "ollama" else data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError("message is not an object")
            text = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderUnavailable(self.name, 200, "reply without a message") from None
        if not isinstance(text, str) or not text.strip():
            recovered = _tool_text(message.get("tool_calls"))
            if recovered is not None:
                text = recovered
            elif "MALFORMED_FUNCTION_CALL" in str(choice.get("finish_reason", "")).upper():
                raise GenerationFailed(self.name, "reply without text: MALFORMED_FUNCTION_CALL")
        if not isinstance(text, str):  # reasoning or tools without text
            reason = (
                "a native tool call instead of text"
                if message.get("tool_calls")
                else "reply without text"
            )
            details = f"finish_reason={choice.get('finish_reason')}, message keys: {', '.join(sorted(message))}"
            raise ProviderUnavailable(self.name, 200, f"{reason}; {details}")
        return Completion(
            text=text,
            provider=self.name,
            model=str(data.get("model") or model),
            usage=_usage(response, native=self.name == "ollama"),
            finish_reason=str(
                choice.get("done_reason" if self.name == "ollama" else "finish_reason") or "stop"
            ),
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        message = self._redact(_message(response))
        if status == 413 and "request too large" in message.lower():
            return RequestTooLarge(self.name, status, message)
        if status == 429 and _zero_limit(response):
            # A limit of 0 never resets: the account has no access to this API or model yet
            # (e.g. a plan that still has to be activated). Waiting would wait forever.
            return ProviderRejected(self.name, 429, f"the account's limit is 0. {message}".strip())
        if status == 402 and self.paid:
            # A paid account out of money: no reset to wait for, a person has to top it up.
            return ProviderRejected(self.name, 402, f"the balance is used up. {message}".strip())
        if status == 402 or (status == 429 and _QUOTA_WORDS.search(message)):
            now = self.clock()
            wait = _retry_after(response, message, now)
            # Without a hint the pause is long: the same key would only be refused again.
            return QuotaExhausted(
                self.name, now + timedelta(seconds=DEFAULT_QUOTA_PAUSE if wait is None else wait)
            )
        if status == 429:
            return RateLimited(self.name, _retry_after(response, message, self.clock()))
        if status == 400 and _code(response) in _GENERATION_FAILED:
            return GenerationFailed(self.name, message)
        if status in (408, 409) or status >= 500:
            return ProviderUnavailable(self.name, status, message)
        return ProviderRejected(self.name, status, message)

    def _redact(self, text: str) -> str:
        """A provider may echo the key it was sent ("invalid key sk-..."): never keep one."""
        for key in self.keys:
            if key and key in text:
                text = text.replace(key, "<key>")
        return text


def account_unavailable(error: ProviderRejected) -> bool:
    """Account/model access refusals; a malformed request must never disable a model."""
    if error.status == 429:
        return True  # Only a zero-limit 429 is classified as ProviderRejected.
    return error.status in (400, 403) and bool(
        re.search(
            r"requires? (?:a |an )?(?:valid )?(?:credit card|card|payment|paid plan|plan|subscription)|"
            r"(?:credit card|payment|paid plan|plan|subscription).{0,30}required|"
            r"model.{0,80}(?:not available|unavailable|not enabled).{0,40}(?:account|plan|key)",
            error.message,
            re.I,
        )
    )


def _tool_text(value: Any) -> str | None:
    """Recover only validated native calls; unrelated provider error data is discarded."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        value = value["tool_calls"] if set(value) == {"tool_calls"} else [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 10:
        return None
    normalized = normalize_tool_calls({"tool_calls": value})
    calls = []
    for item in normalized["tool_calls"]:
        try:
            calls.append(ToolCall.model_validate(item).model_dump())
        except ValidationError:
            return None
    return json.dumps({"tool_calls": calls}, ensure_ascii=False)


def _usage(response: httpx.Response, *, native: bool = False) -> Usage:
    try:
        data = response.json()
    except (ValueError, httpx.ResponseNotRead):
        data = None
    if native and isinstance(data, dict):
        return Usage(
            prompt_tokens=_count(data.get("prompt_eval_count")),
            completion_tokens=_count(data.get("eval_count")),
        )
    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return Usage(
        prompt_tokens=_count(usage.get("prompt_tokens")),
        completion_tokens=_count(usage.get("completion_tokens")),
    )


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _message(response: httpx.Response) -> str:
    """The provider's own error message, short. Keys are never echoed by providers, but the
    text is cut anyway: it ends up in the audit log."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200].strip()
    if isinstance(data, list) and data:  # Google wraps the error in a list
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        # OpenRouter puts the upstream provider's own words in metadata.raw.
        raw = (
            (error.get("metadata") or {}).get("raw")
            if isinstance(error.get("metadata"), dict)
            else None
        )
        if isinstance(raw, str) and raw:
            return f"{error.get('message') or ''}: {raw}"[:200].strip(": ")
    if isinstance(error, dict):
        text = error.get("message") or error.get("code") or ""
    elif isinstance(error, str):
        text = error
    elif isinstance(data, dict):
        text = data.get("message") or data.get("detail") or ""
    else:
        text = ""
    return str(text)[:200].strip()


_GENERATION_FAILED = frozenset({"tool_use_failed"})
"""Error codes for a reply the provider refused, not a request: Groq's when a model (e.g.
gpt-oss) calls a tool natively although the request offered none."""


def _code(response: httpx.Response) -> str | None:
    """The provider's error code (`{"error": {"code": ...}}`), if it sends one."""
    try:
        data = response.json()
    except ValueError:
        return None
    error = data.get("error") if isinstance(data, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


def _retry_after(response: httpx.Response, message: str, now: datetime) -> float | None:
    """Seconds to wait, from (in this order) `Retry-After` (seconds or an HTTP date), Google's
    `RetryInfo` in the body, the message ("try again in 1.5s"), or the reset time of a rate
    limit that has run out (`x-ratelimit-reset[-name]` next to a `remaining` of 0)."""
    header = response.headers.get("retry-after", "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
        when = _moment(header)
        if when is not None:
            return max(0.0, (when - now).total_seconds())
    delay = _retry_info(response)
    if delay is not None:
        return delay
    found = _RETRY_IN.search(message)
    if found:
        value = float(found[1])
        return value / 1000 if found[2].lower() == "ms" else value
    headers = {k.lower(): v.strip() for k, v in response.headers.items()}
    waits = []
    for name, remaining in headers.items():
        # x-ratelimit-remaining-requests (OpenAI, Groq), x-ratelimit-remaining (OpenRouter)...
        if not name.startswith("x-ratelimit-remaining") or remaining != "0":
            continue
        reset = headers.get(name.replace("remaining", "reset", 1), "")
        wait = reset_in(reset, now)
        if wait is not None:
            waits.append(wait)
    return max(waits) if waits else None


def _retry_info(response: httpx.Response) -> float | None:
    """Google: {"error": {"details": [{"@type": "...RetryInfo", "retryDelay": "52s"}]}}."""
    try:
        data = response.json()
    except ValueError:
        return None
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            delay = detail.get("retryDelay")
            if isinstance(delay, str):
                return _duration(delay)
    return None


def reset_in(value: str, now: datetime) -> float | None:
    """A reset header as seconds from `now`: a duration ("2m52.8s", "30"), a Unix time in
    seconds or milliseconds (OpenRouter), or a date."""
    try:
        number = float(value)
    except ValueError:
        if (duration := _duration(value)) is not None:
            return duration
        when = _moment(value)
        return None if when is None else max(0.0, (when - now).total_seconds())
    if number > 1e12:  # milliseconds since the epoch
        number /= 1000
    if number > 1e9:  # seconds since the epoch (2001 onwards): a moment, not a duration
        return max(0.0, number - now.timestamp())
    return _duration(value)


def _moment(text: str) -> datetime | None:
    """An HTTP date or an ISO 8601 time, timezone-aware (UTC if it names none)."""
    when: datetime | None
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        try:
            when = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if when is not None and when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _duration(text: str) -> float | None:
    """ "2m52.8s", "5.1s", "250ms", "1h0m0s" -> seconds."""
    text = text.strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    scale = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    return sum(float(n) * scale[u] for n, u in parts)


def _zero_limit(response: httpx.Response) -> bool:
    """A rate-limit header that says the limit itself (not what remains) is 0."""
    for name, value in response.headers.items():
        name = name.lower()
        if "ratelimit-limit" in name and value.strip() == "0":
            return True
    return False


def _limit_headers(response: httpx.Response) -> dict[str, str]:
    """Rate-limit headers as the provider sent them (names differ per provider)."""
    names = {k.lower(): v for k, v in response.headers.items()}
    return {
        k: v for k, v in names.items() if "ratelimit" in k or (k.startswith("x-") and "limit" in k)
    }
