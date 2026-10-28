"""Which provider and model a run uses, built from the config (ROADMAP 2.2).

One provider per run: the one asked for, or the first ready one in `providers.PREFERENCE`
(free tiers only: a paid provider is used only when asked for). All its keys are used (the
client moves to the next key when one is refused or limited). A retry with another model
(ROADMAP 2.9) uses `alternative()`: the next ready free provider. Fallback across providers on
failures and per-role models are ROADMAP 4.5.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from desman import providers
from desman.config import Config
from desman.i18n import t
from desman.llm.base import (
    LLM,
    Completion,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    Request,
)
from desman.llm.openai_compat import CallRecord, OpenAICompatible
from desman.providers import Provider


class NoProvider(LookupError):
    pass


def choose(config: Config, wanted: str | None = None) -> Provider:
    """The provider to use: `wanted` (id, name or key variable) or the preferred ready one."""
    if wanted is not None:
        try:
            provider = providers.find(wanted)
        except providers.UnknownProvider as e:
            raise NoProvider(str(e)) from e
        if provider.local:
            raise NoProvider(
                t("{name} is local: local models are not supported yet", name=provider.name)
            )
        if not provider.is_ready(config):
            if provider.settings:
                raise NoProvider(
                    t(
                        "{name} is not configured: add a key with `desman keys add {id}` and set "
                        "its settings in `desman config`",
                        name=provider.name,
                        id=provider.id,
                    )
                )
            raise NoProvider(
                t(
                    "{name} is not configured: add a key with `desman keys add {id}`",
                    name=provider.name,
                    id=provider.id,
                )
            )
        return provider
    ready = {p.id: p for p in providers.ready(config)}
    for pid in providers.PREFERENCE:
        if pid in ready:
            return ready[pid]
    paid = [p for p in providers.ready(config) if p.paid]
    if paid:
        raise NoProvider(
            t(
                "only paid providers are configured ({names}), and DESMAN never picks a paid one "
                "by itself: choose it with --provider, e.g. --provider {id}",
                names=", ".join(p.name for p in paid),
                id=paid[0].id,
            )
        )
    raise NoProvider(t("no provider is configured: add a key with `desman keys add PROVIDER`"))


@dataclass(frozen=True)
class Choice:
    provider: Provider
    model: str


def build(
    config: Config,
    provider: str | None = None,
    model: str | None = None,
    *,
    on_call: Callable[[CallRecord], None] | None = None,
    client: httpx.Client | None = None,
) -> tuple[OpenAICompatible, Choice]:
    return _build(config, choose(config, provider), model, on_call=on_call, client=client)


def alternative(
    config: Config,
    current: Choice,
    *,
    on_call: Callable[[CallRecord], None] | None = None,
    client: httpx.Client | None = None,
) -> tuple[OpenAICompatible, Choice] | None:
    """Another provider for a retry that should not repeat the same model: the first ready
    free one after `current` in the preference order (never a paid one). None if there is none."""
    ready = {p.id: p for p in providers.ready(config) if not p.paid}
    for pid in providers.PREFERENCE:
        if pid in ready and pid != current.provider.id:
            return _build(config, ready[pid], None, on_call=on_call, client=client)
    return None


def _build(
    config: Config,
    chosen: Provider,
    model: str | None,
    *,
    on_call: Callable[[CallRecord], None] | None,
    client: httpx.Client | None,
) -> tuple[OpenAICompatible, Choice]:
    assert chosen.key_env is not None
    use = model or chosen.default_model
    llm = OpenAICompatible(
        chosen.id,
        chosen.endpoint(config),
        [k.value for k in config.keys(chosen.key_env)],
        use,
        client=client,
        on_call=on_call,
        auth=chosen.auth,
        send_temperature=chosen.send_temperature,
        max_tokens=chosen.max_tokens,
        paid=chosen.paid,
    )
    return llm, Choice(chosen, use)


# ── Backoff ──────────────────────────────────────────────────────────────────

ATTEMPTS = 3
"""Calls per request: the first one and up to two repeats after a blip."""
BASE_DELAY = 1.0
MAX_DELAY = 8.0
SHORT_LIMIT = 10.0
"""A rate limit that asks to wait this long or less is waited out here; a longer one becomes a
pause of the subtask (the coordinator's business)."""
QUICK_FAILURE = 10.0
"""Only a failure that came back within this many seconds is retried. A slow one (a timeout,
a 504 after a long wait) is not: repeating it could outlast the subtask's lease, which nobody
renews while the executor waits on the model."""


def _sleep(seconds: float) -> None:
    time.sleep(seconds)  # looked up at call time, so tests replace it


class Backoff:
    """Repeats a call that failed for a passing reason, before it costs the subtask anything.

    A 5xx or a dropped connection that came back quickly is retried after 1s, then 2s (each
    delay jittered to between half and all of it, capped at MAX_DELAY), up to ATTEMPTS calls
    in all; a rate limit of SHORT_LIMIT seconds or less is waited out. Anything else (refused
    key, quota, a long limit, a slow failure such as a timeout) and the last failure propagate
    unchanged: the retry policy decides then. So a call takes at most about ATTEMPTS ×
    QUICK_FAILURE + the waits longer than it did before: far less than a lease.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        attempts: int = ATTEMPTS,
        sleep: Callable[[float], None] | None = None,
        jitter: Callable[[], float] = random.random,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.inner = llm
        self.name = llm.name
        self.attempts = attempts
        self.sleep = sleep
        self.jitter = jitter
        self.monotonic = monotonic

    def complete(self, request: Request) -> Completion:
        for attempt in range(1, self.attempts + 1):
            started = self.monotonic()
            try:
                return self.inner.complete(request)
            except (ProviderTimeout, ProviderUnavailable):
                if attempt == self.attempts or self.monotonic() - started > QUICK_FAILURE:
                    raise
                delay = min(MAX_DELAY, BASE_DELAY * 2 ** (attempt - 1))
                delay *= 0.5 + self.jitter() / 2
            except RateLimited as e:
                if attempt == self.attempts or e.retry_after is None:
                    raise
                if e.retry_after > SHORT_LIMIT:
                    raise
                delay = e.retry_after
            (self.sleep or _sleep)(delay)
        raise AssertionError("unreachable")  # pragma: no cover
