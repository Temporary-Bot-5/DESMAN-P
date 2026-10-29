"""What an executor knows when it starts a subtask.

Agents do not talk to each other (PROJECT_PLAN, principle 5): the only way the result of one
subtask reaches the next is through this structured context, assembled from the store.

Free models have small context windows, so dependency results are fitted into a token budget.
The budget is shared fairly: results smaller than their share are passed whole, and whatever they
leave over goes to the larger ones, which are truncated with an explicit marker. The budget
covers the results only (descriptions are short and fixed). Tokens are estimated as
characters / 4, which is close enough for budgeting.
"""

from __future__ import annotations

import json
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from desman.core.models import Subtask
from desman.core.retry import Strategy, strategy_for

CHARS_PER_TOKEN: Final = 4
DEFAULT_BUDGET_TOKENS: Final = 4000


class DependencyResult(BaseModel):
    """The accepted result of a dependency, as stored."""

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    description: str
    content: dict[str, Any] | None
    """None when the operator accepted the subtask without any artifact."""


class DependencyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    subtask_id: str
    description: str
    content: Any
    """The result itself, or a text preview of it when `truncated`."""
    truncated: bool = False


class Feedback(BaseModel):
    """Why the previous attempt was sent back, so the next one can do better."""

    model_config = ConfigDict(frozen=True)

    source: str  # "reviewer", "human" or "executor" (its own failed attempt)
    text: str


class Context(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str
    acceptance_criteria: str
    inputs: dict[str, Any]
    checkpoint: dict[str, Any] | None
    dependencies: list[DependencyContext]
    feedback: Feedback | None = None
    attempt: int = 1
    """1 for the first try; grows with every quality retry."""
    strategy: Strategy = Strategy.FIRST
    """How this attempt should differ from the previous one (picked by the attempt number)."""


def build_context(
    subtask: Subtask,
    dependencies: list[DependencyResult],
    feedback: Feedback | None,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
    strategy: Strategy | None = None,
) -> Context:
    """`strategy` defaults to the one the attempt number picks."""
    attempt = subtask.attempts.quality + 1
    return Context(
        description=subtask.description,
        acceptance_criteria=subtask.acceptance_criteria,
        inputs=subtask.inputs,
        checkpoint=subtask.checkpoint,
        dependencies=fit_dependencies(dependencies, budget_tokens),
        feedback=feedback,
        attempt=attempt,
        strategy=strategy or strategy_for(attempt),
    )


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)  # ceiling


def fit_dependencies(
    results: list[DependencyResult], budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> list[DependencyContext]:
    """Fit dependency results into `budget_tokens`, keeping their order."""
    rendered = [_render(r.content) for r in results]
    allowance = _fair_shares([len(text) for text in rendered], budget_tokens * CHARS_PER_TOKEN)
    fitted = []
    for result, text, chars in zip(results, rendered, allowance, strict=True):
        if chars >= len(text):
            fitted.append(
                DependencyContext(
                    subtask_id=result.subtask_id,
                    description=result.description,
                    content=result.content,
                )
            )
        else:
            fitted.append(
                DependencyContext(
                    subtask_id=result.subtask_id,
                    description=result.description,
                    content=_truncate(text, chars),
                    truncated=True,
                )
            )
    return fitted


def _render(content: dict[str, Any] | None) -> str:
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _fair_shares(sizes: list[int], budget: int) -> list[int]:
    """Max-min fair split of `budget` (water-filling): small items get all they need."""
    shares = [0] * len(sizes)
    remaining = max(budget, 0)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    for position, i in enumerate(order):
        fair = remaining // (len(sizes) - position)
        shares[i] = min(sizes[i], fair)
        remaining -= shares[i]
    return shares


def _truncate(text: str, chars: int) -> str:
    """Cut `text` to at most `chars`, marker included; never exceeds the share."""
    marker = f"…[truncated, {len(text)} chars in full]"
    if chars < len(marker):
        return "…"[:chars]
    return text[: chars - len(marker)] + marker
