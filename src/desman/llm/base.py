"""What every LLM provider looks like to the rest of DESMAN: one call, typed errors.

Providers are unreliable by assumption (free tiers), so failure is part of the interface. A call
either returns a `Completion` or raises a `ProviderError` whose `kind` says what the retry
policy should do with it (ROADMAP 1.1):

- `ProviderRejected` (400/401/403…): PERMANENT, the request will never work as is;
- `RateLimited`, `QuotaExhausted` (429, daily quota): LIMIT, pause without spending an attempt;
- `ProviderTimeout`, `ProviderUnavailable` (5xx, connection reset): TRANSIENT.

A reply that arrives but is unusable (not JSON, wrong schema) is not a provider error: it is
the model's answer, and judging it is the caller's job (a QUALITY failure).

The coordinator maps a `ProviderError` escaping an executor or reviewer to its kind, so agents
can simply let these propagate. Real HTTP clients (2.2) raise the same classes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from desman.core.retry import FailureKind


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class Request(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    """None: the provider's default model."""
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    @property
    def prompt(self) -> str:
        """The last user message: what most routing and assertions look at."""
        return next((m.content for m in reversed(self.messages) if m.role == "user"), "")


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class Completion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    provider: str
    model: str
    usage: Usage
    finish_reason: str = "stop"
    """"length" when the reply was cut off by max_tokens."""


class LLM(Protocol):
    name: str

    def complete(self, request: Request) -> Completion: ...


class ProviderError(Exception):
    kind: ClassVar[FailureKind]
    unfit_model: ClassVar[bool] = False
    """The model cannot produce what was asked (not this request's fault): the task's next
    attempts go to another model, if there is one."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider

    def paused_until(self, now: datetime) -> datetime | None:
        """For LIMIT errors: when to try again; None lets the store use its default pause."""
        return None


class ProviderRejected(ProviderError):
    kind = FailureKind.PERMANENT

    def __init__(self, provider: str, status: int, message: str = "") -> None:
        super().__init__(provider, f"rejected with {status}" + (f": {message}" if message else ""))
        self.status = status


class GenerationFailed(ProviderError):
    """The provider refused the model's own reply (Groq's `tool_use_failed`: the model called
    a tool natively instead of writing text). The request was fine: another attempt, worded
    differently or on another model, may well succeed, so it is a quality failure."""

    kind = FailureKind.QUALITY
    unfit_model = True  # gpt-oss on Groq does it every time tools are described

    def __init__(self, provider: str, message: str = "") -> None:
        super().__init__(
            provider,
            "the model's reply was refused: it called a tool natively instead of answering "
            "in text (write the JSON object as the reply)" + (f": {message}" if message else ""),
        )


class RateLimited(ProviderError):
    kind = FailureKind.LIMIT

    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        hint = f", retry after {retry_after:g}s" if retry_after is not None else ""
        super().__init__(provider, f"rate limited{hint}")
        self.retry_after = retry_after

    def paused_until(self, now: datetime) -> datetime | None:
        return None if self.retry_after is None else now + timedelta(seconds=self.retry_after)


class QuotaExhausted(ProviderError):
    kind = FailureKind.LIMIT

    def __init__(self, provider: str, resets_at: datetime | None = None) -> None:
        if resets_at is not None and resets_at.tzinfo is None:
            raise ValueError("resets_at must be timezone-aware")
        hint = f", resets at {resets_at.isoformat()}" if resets_at is not None else ""
        super().__init__(provider, f"quota exhausted{hint}")
        self.resets_at = resets_at

    def paused_until(self, now: datetime) -> datetime | None:
        return self.resets_at


class ProviderTimeout(ProviderError):
    kind = FailureKind.TRANSIENT

    def __init__(self, provider: str, after: float) -> None:
        super().__init__(provider, f"no reply within {after:g}s")
        self.after = after


class ProviderUnavailable(ProviderError):
    kind = FailureKind.TRANSIENT

    def __init__(self, provider: str, status: int | None = None, message: str = "") -> None:
        what = f"server error {status}" if status is not None else "unreachable"
        super().__init__(provider, what + (f": {message}" if message else ""))
        self.status = status
