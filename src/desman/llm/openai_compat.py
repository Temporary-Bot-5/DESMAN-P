"""A client for OpenAI-compatible chat APIs (every cloud provider in `providers.CLOUD`).

One `complete()` is one chat completion. HTTP outcomes become the typed errors of
`llm/base.py`, so the retry policy can tell a bad key from a rate limit from an outage:

- 400/404/422 and 401/403: `ProviderRejected` (permanent);
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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from desman.llm.base import (
    Completion,
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
    Usage,
)

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


class OpenAICompatible:
    """An `LLM` over one provider's OpenAI-compatible API, with one or more keys."""

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
        self._quota: set[int] = set()  # aside for a used-up quota, not a rate limit
        self._refused: set[int] = set()  # key indexes the provider refused
        self._refusal: ProviderRejected | None = None  # the latest, with the provider's words

    def complete(self, request: Request) -> Completion:
        model = request.model or self.model
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
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
            try:
                return self._attempt(i, model, body, prompt_hash)
            except ProviderRejected as e:
                # 401/403: the key is refused; a 429 rejection is a limit of 0 (an account
                # without access). Both belong to this key: set it aside, try the next one.
                if e.status not in (401, 403, 429):
                    raise  # the request is wrong, not the key: another key will not help
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
            if self.on_call is None:
                return
            spent = (
                _usage(response)
                if response is not None
                else Usage(prompt_tokens=0, completion_tokens=0)
            )
            self.on_call(
                CallRecord(
                    provider=self.name,
                    model=completion.model if completion else model,
                    key_id=key_id(key),
                    prompt_hash=prompt_hash,
                    response_hash=text_hash(completion.text) if completion else None,
                    # From the response, not the completion: a 200 without usable text may
                    # still have cost tokens.
                    prompt_tokens=spent.prompt_tokens,
                    completion_tokens=spent.completion_tokens,
                    latency_ms=max(0, round((self.monotonic() - started) * 1000)),
                    http_status=response.status_code if response is not None else None,
                    error_kind=error.kind.value if error else None,
                    error=str(error)[:300] if error else None,
                    limits=_limit_headers(response) if response is not None else {},
                )
            )

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
                timeout=self.timeout,
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
        error = self._error(response)
        record(error)
        raise error

    def _parse(self, response: httpx.Response, model: str) -> Completion:
        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderUnavailable(self.name, 200, "reply without a message") from None
        if not isinstance(text, str):  # e.g. null content with only reasoning or tool calls
            raise ProviderUnavailable(self.name, 200, "reply without text")
        return Completion(
            text=text,
            provider=self.name,
            model=str(data.get("model") or model),
            usage=_usage(response),
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        message = self._redact(_message(response))
        if status == 429 and _zero_limit(response):
            # A limit of 0 never resets: the account has no access to this API or model yet
            # (e.g. a plan that still has to be activated). Waiting would wait forever.
            return ProviderRejected(self.name, 429, f"the account's limit is 0. {message}".strip())
        if status == 402 or (status == 429 and _QUOTA_WORDS.search(message)):
            now = self.clock()
            wait = _retry_after(response, message)
            # Without a hint the pause is long: the same key would only be refused again.
            return QuotaExhausted(
                self.name, now + timedelta(seconds=DEFAULT_QUOTA_PAUSE if wait is None else wait)
            )
        if status == 429:
            return RateLimited(self.name, _retry_after(response, message))
        if status in (408, 409) or status >= 500:
            return ProviderUnavailable(self.name, status, message)
        return ProviderRejected(self.name, status, message)

    def _redact(self, text: str) -> str:
        """A provider may echo the key it was sent ("invalid key sk-..."): never keep one."""
        for key in self.keys:
            if key and key in text:
                text = text.replace(key, "<key>")
        return text


def _usage(response: httpx.Response) -> Usage:
    try:
        data = response.json()
    except ValueError:
        data = None
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


def _retry_after(response: httpx.Response, message: str) -> float | None:
    """Seconds to wait, from (in this order) `Retry-After` (seconds or an HTTP date), the
    message ("try again in 1.5s"), or the reset time of a rate limit that has run out."""
    header = response.headers.get("retry-after", "").strip()
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(header)
        except (TypeError, ValueError):
            when = None
        if when is not None:
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())
    found = _RETRY_IN.search(message)
    if found:
        value = float(found[1])
        return value / 1000 if found[2].lower() == "ms" else value
    waits = []
    for what in ("requests", "tokens"):  # OpenAI and Groq style: x-ratelimit-reset-requests
        if response.headers.get(f"x-ratelimit-remaining-{what}", "").strip() == "0":
            wait = _duration(response.headers.get(f"x-ratelimit-reset-{what}", ""))
            if wait is not None:
                waits.append(wait)
    return max(waits) if waits else None


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
