"""A deterministic, scripted LLM provider (ROADMAP 1.8).

Every test that involves a model runs on this, never on a real API. A script is a list of
steps, served one per call, in order:

    llm = FakeLLM('{"answer": 42}', garbage("fenced"), RateLimit(retry_after=30), Timeout())

A step is a reply (a plain string is shorthand for one), or a failure that raises the same
`ProviderError` a real client would: a timeout, a 429, an exhausted quota, a 5xx, a 4xx. A
step can also be a function of the request, for replies that depend on the prompt.

Calls from different roles interleave, so a fake can route by prompt instead of by order:

    llm = FakeLLM().on("Review", "accept", repeat_last=True).on("Plan", plan_json)

Running out of script raises `ScriptExhausted`, a `BaseException`, so the coordinator does not
swallow it as a transient error and the test fails loudly. Time comes from the injected `clock`
and `sleep` (pass `FakeClock`), so timeouts and latency cost no real time.

Scripts are also loadable from JSON (`from_file`): the crash test (1.11) drives `desman run`
in a subprocess with one.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from desman.core.context import CHARS_PER_TOKEN, estimate_tokens
from desman.llm.base import (
    Completion,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    QuotaExhausted,
    RateLimited,
    Request,
    Usage,
)


class _Step(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Reply(_Step):
    type: Literal["reply"] = "reply"
    text: str
    finish_reason: str = "stop"
    latency: float = Field(default=0.0, ge=0)


class Timeout(_Step):
    type: Literal["timeout"] = "timeout"
    after: float = Field(default=30.0, ge=0)
    """How long the client waited before giving up (the fake sleeps that long)."""


class RateLimit(_Step):
    type: Literal["rate_limit"] = "rate_limit"
    retry_after: float | None = Field(default=60.0, ge=0)


class Quota(_Step):
    type: Literal["quota"] = "quota"
    resets_in: float | None = Field(default=3600.0, ge=0)


class ServerError(_Step):
    type: Literal["server_error"] = "server_error"
    status: int = Field(default=503, ge=500, le=599)


class Rejected(_Step):
    type: Literal["rejected"] = "rejected"
    status: int = Field(default=401, ge=400, le=499)
    message: str = ""

    @field_validator("status")
    @classmethod
    def _not_429(cls, status: int) -> int:
        if status == 429:
            raise ValueError("429 is a rate limit: use RateLimit")
        return status


AnyStep = Reply | Timeout | RateLimit | Quota | ServerError | Rejected
Step = Annotated[AnyStep, Field(discriminator="type")]
StepLike = AnyStep | str
Responder = Callable[[Request], StepLike]
ScriptItem = StepLike | Responder
Matcher = str | Callable[[Request], bool]

GARBAGE: Final[dict[str, str]] = {
    "empty": "",
    "prose": "Sure! I have completed the task as requested.",
    "fenced": '```json\n{"answer": 42}\n```',
    "wrapped": 'Here is the result: {"answer": 42} Let me know if you need anything else!',
    "truncated": '{"answer": "the result is',
    "trailing_comma": '{"answer": 42,}',
    "single_quotes": "{'answer': 42}",
    "not_object": "[1, 2, 3]",
}
"""What free models actually send back instead of the JSON they were asked for."""


def garbage(name: str) -> Reply:
    """A typical malformed reply; "truncated" also reports finish_reason="length"."""
    return Reply(text=GARBAGE[name], finish_reason="length" if name == "truncated" else "stop")


class ScriptExhausted(BaseException):
    """The fake has no step left for a call: a bug in the test's script, never a provider
    failure. BaseException on purpose, so `except Exception` in the loop does not hide it."""


class Route(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    match: str
    """Substring of the last user message."""
    steps: list[Step | str]
    repeat_last: bool = False


class Script(BaseModel):
    """The JSON form of a fake: `{"steps": [...], "routes": [...], "repeat_last": false}`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: list[Step | str] = Field(default_factory=list)
    routes: list[Route] = Field(default_factory=list)
    repeat_last: bool = False


class _Queue:
    def __init__(self, label: str, items: tuple[ScriptItem, ...], repeat_last: bool) -> None:
        self.label = label
        self.items = items
        self.repeat_last = repeat_last
        self.position = 0

    def take(self) -> ScriptItem:
        if self.position < len(self.items):
            self.position += 1
            return self.items[self.position - 1]
        if self.repeat_last and self.items:
            return self.items[-1]
        raise ScriptExhausted(f"{self.label}: no step left (served {self.position})")

    @property
    def remaining(self) -> int:
        return len(self.items) - self.position


class FakeLLM:
    def __init__(
        self,
        *steps: ScriptItem,
        name: str = "fake",
        repeat_last: bool = False,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.name = name
        self.clock = clock or (lambda: datetime.now(UTC))
        self.sleep = sleep or (lambda _seconds: None)
        self._default = _Queue("script", steps, repeat_last)
        self._routes: list[tuple[Callable[[Request], bool], _Queue]] = []
        self.calls: list[Request] = []
        self.served: list[AnyStep] = []
        """The step each call resolved to, in call order."""

    def on(self, match: Matcher, *steps: ScriptItem, repeat_last: bool = False) -> FakeLLM:
        """Serve `steps` to calls whose prompt contains `match` (or for which it returns True).
        Routes are tried in the order they were added, before the main script."""
        matches: Callable[[Request], bool]
        if isinstance(match, str):
            text = match
            matches = lambda request: text in request.prompt  # noqa: E731
        else:
            matches = match
        self._routes.append((matches, _Queue(f"route {match!r}", steps, repeat_last)))
        return self

    @property
    def remaining(self) -> int:
        """Steps not yet served (repeated last steps not counted)."""
        return self._default.remaining + sum(queue.remaining for _, queue in self._routes)

    def complete(self, request: Request) -> Completion:
        self.calls.append(request)
        queue = next((q for matches, q in self._routes if matches(request)), self._default)
        item = queue.take()
        step = _resolve(item(request) if callable(item) else item)
        self.served.append(step)
        return self._serve(step, request)

    def _serve(self, step: AnyStep, request: Request) -> Completion:
        if isinstance(step, Timeout):
            self.sleep(step.after)
            raise ProviderTimeout(self.name, step.after)
        if isinstance(step, RateLimit):
            raise RateLimited(self.name, step.retry_after)
        if isinstance(step, Quota):
            resets = (
                None if step.resets_in is None else self.clock() + timedelta(seconds=step.resets_in)
            )
            raise QuotaExhausted(self.name, resets)
        if isinstance(step, ServerError):
            raise ProviderUnavailable(self.name, step.status)
        if isinstance(step, Rejected):
            raise ProviderRejected(self.name, step.status, step.message)
        self.sleep(step.latency)
        text, finish = step.text, step.finish_reason
        if request.max_tokens is not None and estimate_tokens(text) > request.max_tokens:
            text, finish = text[: request.max_tokens * CHARS_PER_TOKEN], "length"
        prompt_tokens = sum(estimate_tokens(m.content) for m in request.messages)
        return Completion(
            text=text,
            provider=self.name,
            model=request.model or f"{self.name}-model",
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=estimate_tokens(text)),
            finish_reason=finish,
        )

    @classmethod
    def from_script(
        cls,
        script: Script | str,
        *,
        name: str = "fake",
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> FakeLLM:
        """Build from a `Script` or its JSON text."""
        parsed = script if isinstance(script, Script) else Script.model_validate_json(script)
        fake = cls(
            *parsed.steps, name=name, repeat_last=parsed.repeat_last, clock=clock, sleep=sleep
        )
        for route in parsed.routes:
            fake.on(route.match, *route.steps, repeat_last=route.repeat_last)
        return fake

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        name: str = "fake",
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> FakeLLM:
        return cls.from_script(
            path.read_text(encoding="utf-8"), name=name, clock=clock, sleep=sleep
        )

    @classmethod
    def chaos(
        cls,
        seed: int,
        reply: ScriptItem,
        *,
        garbage_rate: float = 0.1,
        timeout_rate: float = 0.1,
        rate_limit_rate: float = 0.1,
        server_error_rate: float = 0.05,
        name: str = "fake",
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> FakeLLM:
        """An endless script: each call fails at the given rates, otherwise serves `reply`.
        The same seed gives the same sequence."""
        rates = (garbage_rate, timeout_rate, rate_limit_rate, server_error_rate)
        if any(rate < 0 for rate in rates) or sum(rates) > 1:
            raise ValueError("rates must be non-negative and sum to at most 1")
        rng = random.Random(seed)
        junk = sorted(GARBAGE)

        def responder(request: Request) -> StepLike:
            roll = rng.random()
            for rate, make in zip(rates, _CHAOS_STEPS, strict=True):
                if roll < rate:
                    return make(rng, junk)
                roll -= rate
            return _resolve(reply(request) if callable(reply) else reply)

        return cls(responder, name=name, repeat_last=True, clock=clock, sleep=sleep)


_CHAOS_STEPS: Final[tuple[Callable[[random.Random, list[str]], StepLike], ...]] = (
    lambda rng, junk: garbage(rng.choice(junk)),
    lambda rng, junk: Timeout(after=rng.choice([10.0, 30.0])),
    lambda rng, junk: RateLimit(retry_after=rng.choice([None, 5.0, 60.0])),
    lambda rng, junk: ServerError(status=rng.choice([500, 502, 503])),
)


def _resolve(item: StepLike) -> AnyStep:
    if isinstance(item, str):
        return Reply(text=item)
    if not isinstance(item, _Step):  # a responder returned something odd: a bug in the script
        raise TypeError(f"script step must be a str or a step, not {type(item).__name__}")
    return item
