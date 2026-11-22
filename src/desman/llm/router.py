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
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import httpx

from desman import providers
from desman.config import Config
from desman.core import calls, db, holds
from desman.core.calls import Role
from desman.core.models import Kind
from desman.core.retry import FailureKind
from desman.i18n import t
from desman.llm import catalog
from desman.llm import waiting as provider_wait
from desman.llm.base import (
    LLM,
    BudgetExhausted,
    Completion,
    ProviderError,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    Request,
    RequestTooLarge,
    among,
    time_for_call,
    unfit_seen,
)
from desman.llm.limiter import Busy
from desman.llm.openai_compat import CallRecord, OpenAICompatible, key_id
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
        if not _ready(config, provider):
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


def _ready(config: Config, provider: Provider) -> bool:
    return provider.is_ready(config)


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

    memory_risk: bool = field(default=False, compare=False)
    advisory_context: bool = field(default=False, compare=False)

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
    kind: catalog.Lane | None = None,
    conn: sqlite3.Connection | None = None,
    scope: holds.Scope = "all",
    scope_id: str = "",
    local_choices: dict[tuple[str, str], Choice] | None = None,
) -> list[Choice]:
    """The models `role` asks, best first; the executor's for a subtask of `kind`. Raises
    `NoProvider` (with what to do about it)."""

    def choice(p: Provider, name: str) -> Choice:
        if not p.local or local_choices is None:
            return _choice(config, p, name)
        identity = (p.id, name)
        if identity not in local_choices:
            # Models owns the measured metadata snapshot. A newly scoped model uses its
            # configured context until the next Models instance, without contacting a server.
            local_choices[identity] = Choice(
                p,
                name,
                catalog.local_model(
                    name, p.context(config) or (16384 if p.id == "ollama" else None)
                ),
                advisory_context=p.context_env not in config.settings or p.context(config) is None,
            )
        return local_choices[identity]

    chosen = preferred(config, role, kind)
    selected: Provider | None = None
    if chosen is not None and provider is None and model is None:
        try:
            candidate = providers.find(chosen.split("/", 1)[0])
            if candidate.id == "ollama" or _ready(config, candidate):
                selected = candidate
        except providers.UnknownProvider:
            pass
    if kind == catalog.MASTER:
        return _master_lineup(
            config, provider, model, conn, scope, scope_id, local_choices=local_choices
        )
    first = selected if selected is not None else choose(config, provider)
    if model is not None:
        return filter_holds(config, [choice(first, model)], conn, scope, scope_id)
    pool = [first] if provider is not None else _free(config)
    need = catalog.need(role, kind)
    ranked = [
        (p.local, need not in c.spec.strengths, not c.spec.strict_json, place, order, c)
        for place, p in enumerate(pool)
        for order, c in enumerate([choice(p, p.model(config))] if p.local else _choices(config, p))
    ]
    choices = [row[-1] for row in sorted(ranked, key=lambda row: row[:5])]
    if chosen is not None and provider is None and all(c.label != chosen for c in choices):
        provider_id, _, model_id = chosen.partition("/")
        try:
            selected = providers.find(provider_id)
        except providers.UnknownProvider:
            selected = None
        if selected is not None and (selected.id == "ollama" or _ready(config, selected)):
            choices.append(choice(selected, model_id))
    if chosen is not None:  # the person's choice goes first, if it is among them
        choices.sort(key=lambda c: c.label != chosen)
    elif provider is None and (source := _key_provider(config, preferred_key(config, role, kind))):
        # Auto model with a chosen key: that key's provider is asked first (its best models
        # first, as ranked), the others stay behind it for a fallback.
        choices.sort(key=lambda c: c.provider.id != source)
    return filter_holds(config, choices, conn, scope, scope_id)


def qualified(config: Config, choice: Choice, *, named: bool = False) -> bool:
    """Whether the choice meets the person's admissions and the master context floor."""
    return (
        choice.label not in config.master.revoked
        and master_limits(choice)
        and (
            choice.spec.master
            or choice.label in config.master.admitted
            or (choice.provider.local and (named or choice.label == config.tuning.models.master))
        )
    )


def master_limits(choice: Choice) -> bool:
    return (
        choice.spec.context >= catalog.MASTER_LOCAL_CONTEXT and choice.spec.request_tokens is None
    )


def master_candidate(
    config: Config, label: str, *, local_choices: dict[tuple[str, str], Choice] | None = None
) -> Choice:
    """Resolve exactly the model named for a benchmark, regardless of admissions."""
    provider_id, sep, model_id = label.partition("/")
    if not sep or not provider_id or not model_id:
        raise NoProvider(t("Use a model label in the form PROVIDER/MODEL."))
    provider = choose(config, provider_id)
    if not provider.local and all(m.id != model_id for m in catalog.models_of(provider)):
        raise NoProvider(t("{model} is unknown to the DESMAN model catalog.", model=label))
    known = (local_choices or {}).get((provider.id, model_id))
    choice = known if known is not None else _choice(config, provider, model_id)
    if not master_limits(choice):
        raise NoProvider(
            t(
                "{model} needs a context of at least {context} tokens and no per-request token cap.",
                model=label,
                context=catalog.MASTER_LOCAL_CONTEXT,
            )
        )
    return choice


def _master_lineup(
    config: Config,
    provider: str | None,
    model: str | None,
    conn: sqlite3.Connection | None,
    scope: holds.Scope,
    scope_id: str,
    *,
    local_choices: dict[tuple[str, str], Choice] | None,
) -> list[Choice]:
    """The executor's of a master subtask: the lineup of analysis, its qualified models only
    (the person's chosen one first, if qualified); none qualified is `NoProvider`."""
    # Ordered by the master's own choices (model and key), never by the analysis executor's:
    # those are the Improvement Analyst's to propose, the master's only the person's.
    tuned = config.tuning.model_copy(deep=True)
    tuned.models.analysis = config.tuning.models.master
    tuned.keys.analysis = config.tuning.keys.master
    base = config.model_copy(update={"tuning": tuned})
    choices = lineup(
        base,
        Role.EXECUTOR,
        provider,
        model,
        kind=Kind.ANALYSIS,
        conn=conn,
        scope=scope,
        scope_id=scope_id,
        local_choices=local_choices,
    )
    if model is None:
        pool = _free(config) if provider is None else [choose(config, provider)]
        for p in pool:
            if not p.local:
                continue
            for label in config.master.admitted:
                source, _, model_id = label.partition("/")
                if source != p.id or any(c.label == label for c in choices):
                    continue
                known = (local_choices or {}).get((source, model_id))
                if known is None and local_choices is not None:
                    known = Choice(p, model_id, catalog.local_model(model_id, p.context(config)))
                    local_choices[source, model_id] = known
                choices.append(known if known is not None else _choice(config, p, model_id))
        choices = filter_holds(config, choices, conn, scope, scope_id)
    named = provider is not None or model is not None
    masters = [
        c
        for c in choices
        # A paid model only when the person names its provider for this run, as everywhere.
        if qualified(config, c, named=named) and (provider is not None or not c.provider.paid)
    ]
    if not masters:
        local_note = (
            " "
            + t(
                "For Ollama, also set its context (OLLAMA_CONTEXT) to at least {context}.",
                context=catalog.MASTER_LOCAL_CONTEXT,
            )
            if providers.find("ollama").model(config)
            else ""
        )
        raise NoProvider(
            t(
                "No model ready for a master module: configure a catalog or admitted model "
                "({models}), or a local model with a context of at least {context} tokens that "
                "you name with `desman tuning --model master=PROVIDER/MODEL`.",
                models=", ".join(master_models(config)),
                context=catalog.MASTER_LOCAL_CONTEXT,
            )
            + local_note
        )
    return masters


def may_be_master(label: str, config: Config | None = None) -> bool:
    """Check eligibility when choosing a preference, including a named local model."""
    provider_id, sep, model_id = label.partition("/")
    if not sep or not model_id:
        return False
    try:
        p = providers.find(provider_id)
    except providers.UnknownProvider:
        return False
    if config is None and p.local:
        return True  # a local context is only known with the person's config
    config = config or Config()
    return qualified(config, _choice(config, p, model_id), named=True)


def master_models(config: Config | None = None, *, include_paid: bool = False) -> list[str]:
    """Catalog and admitted labels eligible for master selection."""
    config = config or Config()
    marked = [f"{p.id}/{m.id}" for p in providers.CLOUD for m in catalog.models_of(p) if m.master]
    return [
        label
        for label in dict.fromkeys([*marked, *config.master.admitted])
        if may_be_master(label, config)
        and (include_paid or not providers.find(label.partition("/")[0]).paid)
    ]


def sources(config: Config, choice: Choice) -> list[str]:
    if choice.provider.local:
        return [choice.provider.id]
    return [key_id(k.value) for k in config.keys(choice.provider.key_env or "")]


def filter_holds(
    config: Config,
    choices: list[Choice],
    conn: sqlite3.Connection | None,
    scope: holds.Scope = "all",
    scope_id: str = "",
) -> list[Choice]:
    if conn is None:
        return choices
    held = {(h.source, h.model) for h in holds.active(conn, scope, scope_id)}
    return [
        c for c in choices if any((source, c.label) not in held for source in sources(config, c))
    ]


def spread_tier(
    choices: Sequence[Choice], role: Role, kind: catalog.Lane | None = None
) -> list[Choice]:
    """First model per provider in the leading equally suitable run."""
    need = catalog.need(role, kind)

    def rank(c: Choice) -> tuple[bool, bool, bool]:
        return c.provider.local, need not in c.spec.strengths, not c.spec.strict_json

    tier: list[Choice] = []
    for choice in choices:
        if rank(choice) != rank(choices[0]):
            break
        if not choice.provider.paid and all(c.provider.id != choice.provider.id for c in tier):
            tier.append(choice)
    return tier


def capacity_order(
    conn: sqlite3.Connection,
    config: Config,
    tier: Sequence[Choice],
    clock: Callable[[], datetime],
) -> Callable[[list[Choice]], list[Choice]]:
    """One indexed statement reads recent usage and limiter blocks for the tier."""

    def order(choices: list[Choice]) -> list[Choice]:
        if len(tier) < 2:
            return choices
        now = clock()
        values = ",".join("(?, ?, ?)" for _ in tier)
        args: list[object] = [v for c in tier for v in (c.provider.id, c.model, c.label)]
        rows = conn.execute(
            f"WITH candidates(provider, model, scope) AS (VALUES {values}) "
            "SELECT c.scope, b.blocked_until, "
            "(SELECT COUNT(*) FROM model_calls m WHERE m.provider = c.provider "
            "AND m.model = c.model AND m.at >= ?) AS used, "
            "(SELECT m.error_kind IS NOT NULL AND m.error_kind != 'quality' "
            "FROM model_calls m WHERE m.provider = c.provider AND m.model = c.model "
            "AND m.at >= ? ORDER BY m.at DESC, m.seq DESC LIMIT 1) AS failed "
            "FROM candidates c LEFT JOIN rate_buckets b ON b.scope = c.scope",
            (*args, db.to_iso(now - timedelta(minutes=10)), db.to_iso(now - timedelta(hours=1))),
        ).fetchall()
        usage = {r["scope"]: r for r in rows}

        def score(c: Choice) -> tuple[bool, bool, float]:
            row = usage[c.label]
            blocked = row["blocked_until"]
            keys = max(1, len(config.keys(c.provider.key_env or "")))
            return (
                bool(blocked and db.parse_iso(blocked) > now),
                bool(row["failed"]),
                row["used"] / (c.spec.rpm * keys),
            )

        head = sorted(tier, key=score)
        return [*head, *(c for c in choices if c not in tier)]

    return order


def health_order(
    conn: sqlite3.Connection,
    clock: Callable[[], datetime],
    *,
    keep: str | None = None,
) -> Callable[[list[Choice]], list[Choice]]:
    """Stably defer repeated transient failures using the shared, indexed call journal."""

    def order(choices: list[Choice]) -> list[Choice]:
        if len(choices) < 2:
            return choices
        values = ",".join("(?, ?)" for _ in choices)
        args = [v for c in choices for v in (c.provider.id, c.model)]
        rows = conn.execute(
            f"WITH candidates(provider, model) AS (VALUES {values}) "
            "SELECT c.provider, c.model, "
            "SUM(CASE WHEN m.error_kind = 'transient' THEN 1 ELSE 0 END) AS failures, "
            "MAX(CASE WHEN m.error_kind = 'transient' THEN m.seq END) AS failed, "
            "MAX(CASE WHEN m.error_kind IS NULL THEN m.seq END) AS succeeded "
            "FROM candidates c JOIN model_calls m "
            "ON m.provider = c.provider AND m.model = c.model WHERE m.at >= ? "
            "GROUP BY c.provider, c.model",
            (*args, db.to_iso(clock() - timedelta(minutes=15))),
        ).fetchall()
        unhealthy = {
            (row["provider"], row["model"])
            for row in rows
            if row["failures"] >= 2 and row["failed"] > (row["succeeded"] or 0)
        }
        # The person's choice keeps its incoming position, including after slow failures.
        movable = iter(
            sorted(
                (c for c in choices if c.label != keep),
                key=lambda c: (c.provider.id, c.model) in unhealthy,
            )
        )
        return [c if c.label == keep else next(movable) for c in choices]

    return order


def call_scope() -> tuple[str, str] | None:
    purpose = calls.purpose()
    if purpose.task_id is not None:
        return "task", purpose.task_id
    if purpose.session_id is not None:
        return "session", purpose.session_id
    return None


def _key_provider(config: Config, fingerprint: str | None) -> str | None:
    """The provider whose configured key has `fingerprint` (None: none, or no fingerprint)."""
    if fingerprint is None:
        return None
    for p in providers.ready(config):
        if p.key_env and any(key_id(k.value) == fingerprint for k in config.keys(p.key_env)):
            return p.id
    return None


def tuning_field(role: Role, kind: catalog.Lane | None = None) -> str | None:
    """The field of `ModelChoices` for `role` (the executor's: its subtask's kind); None for a
    role without a tuning preference."""
    if role is Role.EXECUTOR:
        if isinstance(kind, Kind):
            return kind.value
        return catalog.MASTER if kind == catalog.MASTER else Kind.TEXT.value
    if role in (Role.PLANNER, Role.REVIEWER, Role.DECOMPOSER, Role.INTERLOCUTOR):
        return role.value
    return None


def candidates(config: Config, name: str) -> list[str]:
    """The models the role of `ModelChoices` field `name` may ask now (ready free providers),
    as `provider/model`, best first; none when no provider is ready."""
    kind: catalog.Lane | None = (
        catalog.MASTER
        if name == catalog.MASTER
        else Kind(name)
        if name in {k.value for k in Kind}
        else None
    )
    role = Role.EXECUTOR if kind is not None else Role(name)
    try:
        return [c.label for c in lineup(config, role, kind=kind)]
    except NoProvider:
        return []


def preferred(config: Config, role: Role, kind: catalog.Lane | None = None) -> str | None:
    """The model the person chose for `role` (`config.tuning.models`), as `provider/model`."""
    name = tuning_field(role, kind)
    return None if name is None else getattr(config.tuning.models, name)


def preferred_key(config: Config, role: Role, kind: catalog.Lane | None = None) -> str | None:
    """The fingerprint of the key the person chose to ask first for this role."""
    name = tuning_field(role, kind)
    return None if name is None else getattr(config.tuning.keys, name)


def _choices(config: Config, p: Provider) -> list[Choice]:
    if p.local:
        return [_choice(config, p, p.model(config))]
    return [Choice(p, m.id) for m in catalog.models_of(p)]


def _local_context(config: Config, p: Provider, model: str) -> int | None:
    if p.id == "ollama":
        from desman.local_models import ollama_context

        return ollama_context(config, model)[0]
    return p.context(config)


def _choice(config: Config, p: Provider, model: str) -> Choice:
    if p.id == "ollama":
        from desman.local_models import ollama_context

        context, risk = ollama_context(config, model)
        advisory = p.context_env not in config.settings or p.context(config) is None
        return Choice(p, model, catalog.local_model(model, context), risk, advisory)
    if p.local:
        configured = p.context(config)
        advisory = p.context_env not in config.settings or configured is None
        return Choice(p, model, catalog.local_model(model, configured), advisory_context=advisory)
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
    first_key: str | None = None,
    timeout_cap: float | None = None,
    on_call: Callable[[CallRecord], None] | None = None,
    client: httpx.Client | None = None,
) -> OpenAICompatible:
    """Every provider key, the selected fingerprint first if present; local takes no key."""
    chosen = choice.provider
    keys = [""] if chosen.key_env is None else [k.value for k in config.keys(chosen.key_env)]
    if first_key is not None:
        keys.sort(key=lambda key: key_id(key) != first_key)
    connected = OpenAICompatible(
        chosen.id,
        chosen.endpoint(config),
        keys,
        choice.model,
        client=client,
        timeout=chosen.timeout,
        timeout_cap=timeout_cap,
        on_call=on_call,
        auth=chosen.auth,
        send_temperature=chosen.send_temperature,
        max_tokens=chosen.max_tokens,
        paid=chosen.paid,
        local=chosen.local,
        context=choice.known.context
        if choice.known
        else _local_context(config, chosen, choice.model),
    )

    if chosen.id == "ollama":
        from desman.local_models import ollama_context, prepare_ollama_context

        connected.prepare_context = lambda model: prepare_ollama_context(
            config, model, connected.client
        )
        connected.memory_risk = (
            choice.memory_risk if choice.known else ollama_context(config, choice.model)[1]
        )
    return connected


# ── Fallback ─────────────────────────────────────────────────────────────────

CHARS_PER_TOKEN = 3
"""A cautious estimate of a prompt's size (fewer characters a token than English averages)."""
REPLY_TOKENS = 1_024
"""Room left for the reply when a request sets no max_tokens."""


ESTIMATE_SLACK = 2.0
"""How far over a model's known capacity an estimate may be and the model still be tried: the
estimate counts characters, not tokens."""


def estimated_tokens(request: Request) -> int:
    chars = sum(len(m.content) for m in request.messages)
    return chars // CHARS_PER_TOKEN + (request.max_tokens or REPLY_TOKENS)


def request_ceiling(choice: Choice, conn: sqlite3.Connection | None = None) -> int | None:
    """Latest reported Groq token ceiling, otherwise the catalog's free-tier limit."""
    ceiling = choice.spec.request_tokens
    if choice.provider.id != "groq" or conn is None:
        return ceiling
    row = conn.execute(
        "SELECT json_extract(limits_json, '$.\"x-ratelimit-limit-tokens\"') "
        "FROM model_calls WHERE provider = ? AND model = ? "
        "AND CAST(json_extract(limits_json, '$.\"x-ratelimit-limit-tokens\"') AS INTEGER) > 0 "
        "ORDER BY seq DESC LIMIT 1",
        (choice.provider.id, choice.model),
    ).fetchone()
    if row is not None:
        try:
            return int(row[0])
        except (ValueError, TypeError):
            pass
    return ceiling


@dataclass(frozen=True)
class SkippedModel:
    model: str
    reason: str
    status: int | None = None
    needed: int | None = None
    context: int | None = None


@dataclass(frozen=True)
class LastCall:
    answered: str | None = None
    skipped: tuple[SkippedModel, ...] = ()


def _request_limit(choice: Choice, conn: sqlite3.Connection | None) -> int | None:
    ceiling = request_ceiling(choice, conn)
    # Local defaults and uncatalogued cloud contexts are estimates.
    if choice.advisory_context or (choice.provider.local and choice.known is None):
        return ceiling
    if (
        not choice.provider.local
        and choice.known is None
        and not any(known.id == choice.model for known in catalog.models_of(choice.provider))
    ):
        return ceiling
    return min(ceiling, choice.spec.context) if ceiling is not None else choice.spec.context


class Fallback:
    """One role's models: a call goes to the first one that can take it.

    A model with a known context or request ceiling too small for the request is skipped.
    Among fitting models, those the task found unfit (`Request.avoid`) go unused when possible.
    A provider size refusal skips that model for requests at least as large.
    A provider error moves the call to the next model, except a quality one: that is the
    model's own answer, judged by the retry policy (which may switch models for the next
    attempt). A model unable to answer such a request at all (`ProviderError.unfit_model`:
    `GenerationFailed`, a native tool call where text was asked) is the exception: the call
    moves on, and this lineup does not ask that model again, so an attempt (the only one that
    combines a split subtask's parts, say) is not lost to a quirk. When it was the last model
    left, its error is raised as it is. Nor does a slow failure (`ProviderError.slow`, set by
    `Backoff`: one call that took over `QUICK_FAILURE` seconds), for the reason `Backoff` does
    not repeat one: nobody renews the subtask's lease while it waits. If models remain
    untried, the slow error propagates and the next call in that scope starts with them.
    When all models were tried, the error raised is a
    limit if any model was limited (the soonest one: the subtask pauses until then), else a
    passing one, a size refusal, or the first model's error.
    """

    def __init__(
        self,
        members: Sequence[tuple[LLM, Choice | None]],
        *,
        order: Callable[[list[Choice]], list[Choice]] | None = None,
        sticky: bool = True,
        health: Callable[[list[Choice]], list[Choice]] | None = None,
        on_used: Callable[[str], None] | None = None,
        label_for: Callable[[], str | None] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        if not members:
            raise ValueError("a lineup needs at least one model")
        self.members = list(members)
        self.conn = conn
        self.order = order
        self.sticky = sticky
        self.health = health
        self.on_used = on_used
        self.label_for = label_for
        self._scopes: dict[tuple[str, str], list[Choice]] = {}
        first, choice = self.members[0]
        self.name = first.name
        self.label = first.name if choice is None else choice.label
        self._last: tuple[Completion, LLM] | None = None
        self._call = LastCall()
        self._deferred: dict[tuple[str, str] | None, list[str]] = {}
        self._untried: dict[tuple[str, str] | None, set[str]] = {}
        self._refused_sizes: dict[str, int] = {}
        self._unfit: set[str] = set()
        """Models that proved unable to answer (`unfit_model`): asked only when none else is."""

    @property
    def label(self) -> str:
        return (self.label_for() if self.label_for is not None else None) or self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def last_call(self) -> LastCall:
        return self._call

    @property
    def has_untried(self) -> bool:
        """Whether a slow failure left models untried in the current call scope."""
        return bool(self._untried.get(call_scope()))

    @property
    def choices(self) -> list[Choice]:
        return [c for _, c in self.members if c is not None]

    def complete(self, request: Request) -> Completion:
        self._call = LastCall()
        skipped: list[SkippedModel] = []
        need = estimated_tokens(request)
        scope = call_scope()
        ordered_members = self.members
        if self.order is not None:
            held = self._scopes.get(scope) if scope is not None and self.sticky else None
            choices = self.choices
            choices = self.order(choices) if held is None else held
            ordered_members = [m for c in choices for m in self.members if m[1] == c]
        deferred = self._deferred.get(scope, [])
        if deferred:
            ordered_members = sorted(
                ordered_members,
                key=lambda m: (
                    deferred.index(m[1].label if m[1] else m[0].name)
                    if (m[1].label if m[1] else m[0].name) in deferred
                    else -1
                ),
            )
        health_changed = False
        if self.health is not None:
            before_health = [c for _, c in ordered_members if c is not None]
            healthy_choices = self.health(before_health)
            health_changed = healthy_choices != before_health
            health_members = iter(m for c in healthy_choices for m in ordered_members if m[1] == c)
            ordered_members = [m if m[1] is None else next(health_members) for m in ordered_members]
        sized = [
            (llm, choice)
            for llm, choice in ordered_members
            if need < self._refused_sizes.get(choice.label if choice else llm.name, need + 1)
        ]
        limits = {
            choice.label: _request_limit(choice, self.conn)
            for _, choice in self.members
            if choice is not None
        }
        large = [
            (llm, choice)
            for llm, choice in sized
            if choice is None or (limit := limits[choice.label]) is None or limit >= need
        ]
        # Preserve lineup order among models that fit, then try advisory underestimates.
        if large:
            large.sort(key=lambda m: max(0, need - m[1].spec.context) if m[1] is not None else 0)
        elif sized:
            # The token estimate is rough: a request just over the largest capacity gets a try
            # there; one far over it would only be refused (or cut) by the model.
            largest = max(limits[choice.label] or 0 for _, choice in sized if choice is not None)
            if need <= largest * ESTIMATE_SLACK:
                large = [
                    (llm, choice)
                    for llm, choice in sized
                    if choice is not None and limits[choice.label] == largest
                ]
        fit = [
            (llm, choice)
            for llm, choice in large
            if (choice is None or not among(choice.label, request.avoid))
            and (choice.label if choice is not None else llm.name) not in self._unfit
        ]
        members = fit or large
        for llm, choice in self.members:
            if (llm, choice) not in members:
                limit = limits[choice.label] if choice is not None else None
                skipped.append(
                    SkippedModel(
                        llm.name if choice is None else choice.label,
                        "context"
                        if (llm, choice) not in sized or (limit is not None and limit < need)
                        else "unfit",
                        needed=need,
                        context=limit,
                    )
                )
        untried = self._untried.get(scope, {c.label if c else m.name for m, c in members}).copy()
        errors: list[ProviderError] = []
        if not members:
            self._call = LastCall(skipped=tuple(skipped))
            self._untried.pop(scope, None)
            if not sized:
                raise RequestTooLarge(self.name, 413, f"no model can take {need} tokens")
            largest = max(
                (limits[choice.label] or 0 for _, choice in sized if choice is not None), default=0
            )
            raise RequestTooLarge(
                self.name,
                413,
                f"request needs {need} tokens; largest available request ceiling is {largest}. "
                "Choose a model with a larger request ceiling or reduce the task context.",
            )
        failed_over = False
        for i, (llm, choice) in enumerate(members):
            self._call = LastCall(skipped=tuple(skipped))
            time_for_call(self.name)
            label = llm.name if choice is None else choice.label
            untried.discard(label)
            if self.on_used is not None:
                self.on_used(label)
            if (
                self.order is not None
                and scope is not None
                and self.sticky
                and not failed_over
                and not health_changed
            ):
                self._scopes.pop(scope, None)
                self._scopes[scope] = sorted(choices, key=lambda c: c.label != label)
                if len(self._scopes) > 256:
                    self._scopes.pop(next(iter(self._scopes)))
            try:
                reply = llm.complete(request)
            except BudgetExhausted:
                raise
            except ProviderError as e:
                self._untried.pop(scope, None)
                reason = (
                    "context"
                    if isinstance(e, RequestTooLarge)
                    else "busy"
                    if isinstance(e, Busy)
                    else "limit"
                    if e.kind is FailureKind.LIMIT
                    or (isinstance(e, ProviderRejected) and e.status == 429)
                    else "refused"
                    if isinstance(e, ProviderRejected)
                    else "unavailable"
                )
                inner = llm
                nested = getattr(inner, "inner", None)
                while nested is not None:
                    inner = nested
                    nested = getattr(inner, "inner", None)
                if getattr(inner, "memory_risk", False):
                    reason = "memory"
                skipped.append(
                    SkippedModel(
                        label, reason, e.status if isinstance(e, ProviderRejected) else None
                    )
                )
                self._call = LastCall(skipped=tuple(skipped))
                if isinstance(e, RequestTooLarge):
                    self._refused_sizes[label] = need
                elif e.kind is FailureKind.QUALITY:
                    if not e.unfit_model or i == len(members) - 1:
                        raise
                    self._unfit.add(label)
                    unfit_seen(label)  # kept in the task's history: later runs avoid it too
                    continue
                failed_over = True
                if scope is not None:
                    self._scopes.pop(scope, None)
                errors.append(e)
                if (
                    e.slow
                    and not isinstance(e, RequestTooLarge)
                    and not (e.kind is FailureKind.LIMIT and provider_wait.enabled())
                ):
                    if i < len(members) - 1:
                        self._untried[scope] = untried.intersection(
                            c.label if c else m.name for m, c in members[i + 1 :]
                        )
                        rotated = members[i + 1 :] + members[: i + 1]
                        self._deferred[scope] = [c.label if c else m.name for m, c in rotated]
                        if len(self._deferred) > 256:
                            expired = next(iter(self._deferred))
                            self._deferred.pop(expired)
                            self._untried.pop(expired, None)
                        raise
                    break
                continue
            self._untried.pop(scope, None)
            self._deferred.pop(scope, None)
            self._last = (reply, llm)
            ahead = [
                c.label if c else m.name
                for m, c in ordered_members[: ordered_members.index((llm, choice))]
            ]
            ahead.extend(
                label
                for model, candidate in members[:i]
                if (label := candidate.label if candidate else model.name) not in ahead
            )
            ordered = tuple(item for label in ahead for item in skipped if item.model == label)
            self._call = LastCall(f"{reply.provider}/{reply.model}", ordered)
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
    if passing:
        return passing[0]
    return next((e for e in errors if isinstance(e, RequestTooLarge)), errors[0])


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
            except BudgetExhausted:
                raise
            except ProviderError as e:
                # Said of this call alone, not of the repeats before it (`Fallback` asks).
                e.slow = self.monotonic() - started > QUICK_FAILURE
                if isinstance(e, RequestTooLarge):
                    raise
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
            if delay >= time_for_call(self.name):
                raise BudgetExhausted(self.name)
            (self.sleep or _sleep)(delay)
        raise AssertionError("unreachable")  # pragma: no cover
