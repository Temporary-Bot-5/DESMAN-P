"""Which models a run asks, built from the config (ROADMAP 2.2, 4.5).

Every role (planner, executor, reviewer, decomposer, interlocutor) gets its own *lineup*: the
models of the ready free providers, best for that role first (`catalog.need`: what the role
asks of a model, for the executor what its subtask's kind asks, then a model that keeps to JSON, then the provider's place in
`providers.PREFERENCE`, then the model's place in the catalog). A local model (ROADMAP 4.5a:
Ollama, llama.cpp), once its model is set, comes after every cloud one: it costs nothing and
has no quota, but it is slow, so it takes the calls the cloud ones cannot. A paid provider is used only
when asked for (`--provider`); then the lineup is that provider's models. `--model` pins one
model: the lineup is that model alone. A model the person chose for a role (`config.tuning`,
`desman tuning`) goes first in that role's lineup when it is among them; the rest stay behind
it in their order.

A role's calls go through `Fallback`: a model that cannot take the call now (a limit, a server
down, a key refused, a request larger than it takes) hands it to the next one in the lineup,
within the same attempt. A retry with another model (ROADMAP 2.9) asks `other_lineup()`: the
lineup without the main model, starting with the best one on another provider (another family of
models, usually; an unfit-model failure has so far been one provider's quirk), or the next one
of the same provider when there is no other. Every key of a provider is used (the client moves
to the next key when one is refused or limited).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from desman import providers
from desman.config import Config
from desman.core.calls import Role
from desman.core.models import Kind
from desman.core.retry import FailureKind
from desman.i18n import t
from desman.llm import catalog
from desman.llm.base import (
    LLM,
    Completion,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    Request,
    among,
    unfit_seen,
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
        if not provider.is_ready(config):
            if provider.local:
                raise NoProvider(
                    t(
                        "{name} has no model set: choose one in `desman config` (the server's "
                        "address and context size are there too)",
                        name=provider.name,
                    )
                )
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
    free = _free(config)
    if free:
        return free[0]
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


def _free(config: Config) -> list[Provider]:
    """The ready free providers in the preference order, the local ones last."""
    ready = {p.id: p for p in providers.ready(config) if not p.paid}
    cloud = [ready[pid] for pid in providers.PREFERENCE if pid in ready]
    return cloud + [p for p in ready.values() if p.local]


@dataclass(frozen=True)
class Choice:
    provider: Provider
    model: str
    known: catalog.Model | None = field(default=None, compare=False)
    """What is known of a local model (its context, from the settings); None: the catalog's."""

    @property
    def spec(self) -> catalog.Model:
        return self.known or catalog.model(self.provider, self.model)

    @property
    def label(self) -> str:
        """`provider/model`: how a model is told apart (its rate bucket, an unfit verdict)."""
        return f"{self.provider.id}/{self.model}"


def lineup(
    config: Config,
    role: Role,
    provider: str | None = None,
    model: str | None = None,
    *,
    kind: Kind | None = None,
) -> list[Choice]:
    """The models `role` asks, best first; the executor's for a subtask of `kind`. Raises
    `NoProvider` (with what to do about it)."""
    first = choose(config, provider)
    if model is not None:
        return [_choice(config, first, model)]
    pool = [first] if provider is not None else _free(config)
    need = catalog.need(role, kind)
    ranked = [
        (p.local, need not in c.spec.strengths, not c.spec.strict_json, place, order, c)
        for place, p in enumerate(pool)
        for order, c in enumerate(_choices(config, p))
    ]
    choices = [row[-1] for row in sorted(ranked, key=lambda row: row[:5])]
    chosen = preferred(config, role, kind)
    if chosen is not None:  # the person's choice goes first, if it is among them
        choices.sort(key=lambda c: c.label != chosen)
    return choices


def tuning_field(role: Role, kind: Kind | None = None) -> str | None:
    """The field of `ModelChoices` for `role` (the executor's: its subtask's kind); None for a
    role the person does not choose a model for (the interlocutor)."""
    if role is Role.EXECUTOR:
        return (kind or Kind.TEXT).value
    if role in (Role.PLANNER, Role.REVIEWER, Role.DECOMPOSER):
        return role.value
    return None


def candidates(config: Config, name: str) -> list[str]:
    """The models the role of `ModelChoices` field `name` may ask now (ready free providers),
    as `provider/model`, best first; none when no provider is ready."""
    kind = Kind(name) if name in {k.value for k in Kind} else None
    role = Role.EXECUTOR if kind is not None else Role(name)
    try:
        return [c.label for c in lineup(config, role, kind=kind)]
    except NoProvider:
        return []


def preferred(config: Config, role: Role, kind: Kind | None = None) -> str | None:
    """The model the person chose for `role` (`config.tuning.models`), as `provider/model`."""
    name = tuning_field(role, kind)
    return None if name is None else getattr(config.tuning.models, name)


def _choices(config: Config, p: Provider) -> list[Choice]:
    if p.local:
        return [_choice(config, p, p.model(config))]
    return [Choice(p, m.id) for m in catalog.models_of(p)]


def _choice(config: Config, p: Provider, model: str) -> Choice:
    if p.local:
        return Choice(p, model, catalog.local_model(model, p.context(config)))
    return Choice(p, model)


def other_lineup(choices: Sequence[Choice]) -> list[Choice]:
    """For an attempt with another model: the lineup without its first model, the best one on
    another provider first (none left: empty)."""
    if not choices:
        return []
    main, rest = choices[0], [c for c in choices[1:] if c.label != choices[0].label]
    elsewhere = next((c for c in rest if c.provider.id != main.provider.id), None)
    if elsewhere is None:
        return rest
    return [elsewhere, *(c for c in rest if c is not elsewhere)]


def connect(
    config: Config,
    choice: Choice,
    *,
    on_call: Callable[[CallRecord], None] | None = None,
    client: httpx.Client | None = None,
) -> OpenAICompatible:
    """The client of one model: every key of its provider (a local server takes none)."""
    chosen = choice.provider
    keys = [""] if chosen.key_env is None else [k.value for k in config.keys(chosen.key_env)]
    return OpenAICompatible(
        chosen.id,
        chosen.endpoint(config),
        keys,
        choice.model,
        client=client,
        timeout=chosen.timeout,
        on_call=on_call,
        auth=chosen.auth,
        send_temperature=chosen.send_temperature,
        max_tokens=chosen.max_tokens,
        paid=chosen.paid,
    )


# ── Fallback ─────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 3
"""A cautious estimate of a prompt's size (fewer characters a token than English averages)."""
REPLY_TOKENS = 1_024
"""Room left for the reply when a request sets no max_tokens."""


def estimated_tokens(request: Request) -> int:
    chars = sum(len(m.content) for m in request.messages)
    return chars // CHARS_PER_TOKEN + (request.max_tokens or REPLY_TOKENS)


class Fallback:
    """One role's models: a call goes to the first one that can take it.

    A model the task found unfit (`Request.avoid`) is skipped, and so is one whose context is
    too small for the request (unless that leaves none: then the rest are tried all the same).
    A provider error moves the call to the next model, except a quality one: that is the
    model's own answer, judged by the retry policy (which may switch models for the next
    attempt). A model unable to answer such a request at all (`ProviderError.unfit_model`:
    `GenerationFailed`, a native tool call where text was asked) is the exception: the call
    moves on, and this lineup does not ask that model again, so an attempt (the only one that
    combines a split subtask's parts, say) is not lost to a quirk. When it was the last model
    left, its error is raised as it is. Nor does a slow failure (`ProviderError.slow`, set by
    `Backoff`: one call that took over `QUICK_FAILURE` seconds), for the reason `Backoff` does
    not repeat one: nobody renews the subtask's lease while it waits. When the models tried failed, the error raised is a
    limit if any model was limited (the soonest one: the subtask pauses until then), else a
    passing one, else the first model's.
    """

    def __init__(self, members: Sequence[tuple[LLM, Choice | None]]) -> None:
        if not members:
            raise ValueError("a lineup needs at least one model")
        self.members = list(members)
        first, choice = self.members[0]
        self.name = first.name
        self.label = first.name if choice is None else choice.label
        self._last: tuple[Completion, LLM] | None = None
        self._unfit: set[str] = set()
        """Models that proved unable to answer (`unfit_model`): asked only when none else is."""

    @property
    def choices(self) -> list[Choice]:
        return [c for _, c in self.members if c is not None]

    def complete(self, request: Request) -> Completion:
        need = estimated_tokens(request)
        fit = [
            (llm, choice)
            for llm, choice in self.members
            if (choice is None or not among(choice.label, request.avoid))
            and (choice.label if choice is not None else llm.name) not in self._unfit
        ]
        fit = fit or self.members
        large = [m for m in fit if m[1] is None or m[1].spec.context >= need]
        members = large or fit
        errors: list[ProviderError] = []
        for i, (llm, choice) in enumerate(members):
            try:
                reply = llm.complete(request)
            except ProviderError as e:
                if e.kind is FailureKind.QUALITY:
                    if not e.unfit_model or i == len(members) - 1:
                        raise
                    label = llm.name if choice is None else choice.label
                    self._unfit.add(label)
                    unfit_seen(label)  # kept in the task's history: later runs avoid it too
                    continue
                errors.append(e)
                if e.slow:
                    break
                continue
            self._last = (reply, llm)
            return reply
        raise _telling(errors)

    def accept(self, request: Request, reply: Completion) -> None:
        """Passed on to the model that wrote `reply` (it keeps accepted replies, `llm.cache`)."""
        if self._last is None or self._last[0] is not reply:
            return
        accept = getattr(self._last[1], "accept", None)
        if accept is not None:
            accept(request, reply)


def _telling(errors: list[ProviderError]) -> ProviderError:
    now = datetime.now(UTC)
    limits = [e for e in errors if e.kind is FailureKind.LIMIT]
    if limits:
        return min(limits, key=lambda e: e.paused_until(now) or datetime.max.replace(tzinfo=UTC))
    passing = [e for e in errors if e.kind is FailureKind.TRANSIENT]
    return passing[0] if passing else errors[0]


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
            except ProviderError as e:
                # Said of this call alone, not of the repeats before it (`Fallback` asks).
                e.slow = self.monotonic() - started > QUICK_FAILURE
                if not isinstance(e, (ProviderTimeout, ProviderUnavailable, RateLimited)):
                    raise
                if attempt == self.attempts or e.slow:
                    raise
                if isinstance(e, RateLimited):
                    if e.retry_after is None or e.retry_after > SHORT_LIMIT:
                        raise
                    delay = e.retry_after
                else:
                    delay = min(MAX_DELAY, BASE_DELAY * 2 ** (attempt - 1))
                    delay *= 0.5 + self.jitter() / 2
            (self.sleep or _sleep)(delay)
        raise AssertionError("unreachable")  # pragma: no cover
