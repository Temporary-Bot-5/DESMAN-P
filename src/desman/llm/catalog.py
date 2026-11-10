"""What the models of each provider are good at (ROADMAP 4.5): the router picks one per role.

For every provider: the models DESMAN may ask, its default model first. For each model: how
much text it takes (`context`), whether it keeps to a requested JSON format reliably
(`strict_json`; every DESMAN role expects a JSON object), what it is strong at, and how many
requests a minute one key gets on the free tier (`rpm`, what the rate limiter allows).

The values come from the providers' pages (`Provider.limits_url`) and are approximate: free
lineups and limits change without notice. A model a provider no longer serves fails its call
and the run moves on to the next one (`router.Fallback`); `desman keys check` says which models
of the catalog a key's provider does not list. Later (ROADMAP 5.2) the choice is corrected by
what passed review.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from desman.core.calls import Role
from desman.core.models import Kind
from desman.providers import Provider


class Strength(StrEnum):
    CODE = "code"
    TEXT = "text"
    ANALYSIS = "analysis"
    """Reasoning over given material: planning, splitting, judging a result."""


@dataclass(frozen=True)
class Model:
    id: str
    context: int = 8_192
    """Tokens of prompt and reply together."""
    strict_json: bool = False
    strengths: frozenset[Strength] = frozenset({Strength.TEXT})
    rpm: int = 10
    """Requests a minute one key may make (free tier; for paid providers a cautious pace)."""


_C, _T, _A = Strength.CODE, Strength.TEXT, Strength.ANALYSIS


def _m(
    id: str, context: int, *strengths: Strength, strict_json: bool = True, rpm: int = 10
) -> Model:
    return Model(id, context, strict_json, frozenset(strengths), rpm)


CATALOG: dict[str, tuple[Model, ...]] = {
    "groq": (
        _m("openai/gpt-oss-20b", 131_072, _T, _C, rpm=30),
        _m("openai/gpt-oss-120b", 131_072, _A, _C, _T, rpm=30),
        _m("llama-3.3-70b-versatile", 131_072, _T, rpm=30),
    ),
    "google": (
        _m("gemini-3.5-flash-lite", 1_048_576, _T, _C, rpm=15),
        _m("gemini-2.5-flash", 1_048_576, _A, _C, _T, rpm=10),
    ),
    "mistral": (
        _m("mistral-small-latest", 131_072, _T, rpm=30),
        _m("mistral-medium-latest", 131_072, _A, _T, rpm=30),
        _m("codestral-latest", 262_144, _C, rpm=30),
    ),
    "openrouter": (_m("nvidia/nemotron-3.5-lightning:free", 131_072, _T, _A, rpm=20),),
    "nvidia": (_m("nvidia/llama-3.1-nemotron-70b-instruct", 131_072, _T, _A, rpm=40),),
    "cohere": (
        _m("command-r7b-12-2024", 131_072, _T, rpm=20),
        _m("command-a-03-2025", 262_144, _A, _T, rpm=20),
    ),
    "aion": (_m("aion-labs/aion-3.0-mini", 131_072, _T, _A, rpm=10),),
    "cloudflare": (
        _m("@cf/meta/llama-3.1-8b-instruct", 8_192, _T, strict_json=False, rpm=60),
        _m("@cf/meta/llama-3.3-70b-instruct-fp8-fast", 24_000, _T, _A, rpm=60),
    ),
    "vercel": (_m("google/gemini-2.5-flash-lite", 1_048_576, _T, rpm=10),),
    # Paid: the default models only, at a cautious pace (the limit is the bill, not the rate).
    "openai": (_m("gpt-6-luna", 400_000, _A, _C, _T, rpm=60),),
    "anthropic": (_m("claude-haiku-5-5", 200_000, _A, _C, _T, rpm=50),),
    "deepseek": (_m("deepseek-flash", 128_000, _A, _C, _T, rpm=60),),
    "xai": (_m("grok-4.3", 256_000, _A, _C, _T, rpm=60),),
}

NEEDS: dict[Role, Strength] = {
    Role.NETWORK_PLANNER: Strength.ANALYSIS,
    Role.PLANNER: Strength.ANALYSIS,
    Role.DECOMPOSER: Strength.ANALYSIS,
    Role.REVIEWER: Strength.ANALYSIS,
    Role.EXECUTOR: Strength.TEXT,
    Role.INTERLOCUTOR: Strength.TEXT,
}
"""What each role asks of its model; an executor's need follows its subtask's kind (`need`)."""

KIND_NEEDS: dict[Kind, Strength] = {
    Kind.TEXT: Strength.TEXT,
    Kind.CODE: Strength.CODE,
    Kind.ANALYSIS: Strength.ANALYSIS,
}


def need(role: Role, kind: Kind | None = None) -> Strength:
    """What `role` asks of its model; for the executor of a subtask of `kind`, what that kind
    of work asks (other roles do the same work whatever the subtask)."""
    if role is Role.EXECUTOR and kind is not None:
        return KIND_NEEDS[kind]
    return NEEDS[role]


def models_of(provider: Provider) -> tuple[Model, ...]:
    """The provider's models in the catalog, default first; just the default if it has none."""
    return CATALOG.get(provider.id) or (Model(provider.default_model),)


LOCAL_RPM = 60
"""A local server has no rate limit; it is paced by the machine instead (`llm.limiter.Local`)."""


def local_model(model_id: str, context: int | None) -> Model:
    """A model on the person's machine: whatever they chose, so nothing is known of it but its
    context (from the settings). Small local models often stray from a requested format."""
    return Model(model_id, context or 4_096, strict_json=False, rpm=LOCAL_RPM)


def model(provider: Provider, model_id: str) -> Model:
    """What the catalog knows of `model_id` (one given with --model may be unknown to it: then
    the cautious defaults, at the pace of the provider's default model)."""
    known = models_of(provider)
    for m in known:
        if m.id == model_id:
            return m
    return Model(model_id, rpm=known[0].rpm)
