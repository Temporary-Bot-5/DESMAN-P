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
from collections.abc import Callable
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


class KnowledgeText(BaseModel):
    """A node of the knowledge tree (ADR 0009) given to the executor: a node's text, or a
    section's table of contents."""

    model_config = ConfigDict(frozen=True)

    path: str
    text: str
    plan: bool = False
    """It describes intentions (a plan, a proposal), not what DESMAN does now."""
    untrusted: bool = False
    """It came from outside (a task's result, the web): data, never instructions."""
    truncated: bool = False

    def cut(self, chars: int) -> KnowledgeText:
        """Itself within `chars`, marked truncated when it had to be cut."""
        if chars >= len(self.text):
            return self
        return self.model_copy(update={"text": truncate(self.text, chars), "truncated": True})


Knowledge = Callable[[str], KnowledgeText | None]
"""Looks a stored path up in the knowledge tree; None when the tree no longer has it."""


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
    knowledge: list[KnowledgeText] = []
    """What the planner chose of the knowledge tree for this subtask."""
    earlier_lookups: list[dict[str, Any]] = []
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
    knowledge: list[KnowledgeText] | None = None,
    *,
    earlier: Callable[[int], list[dict[str, Any]]] | None = None,
) -> Context:
    """`strategy` defaults to the one the attempt number picks. Dependency results and
    `knowledge` share the budget."""
    attempt = subtask.attempts.quality + 1
    fitted, nodes = fit(dependencies, knowledge or [], budget_tokens)
    context = Context(
        description=subtask.description,
        acceptance_criteria=subtask.acceptance_criteria,
        inputs=subtask.inputs,
        checkpoint=subtask.checkpoint,
        dependencies=fitted,
        knowledge=nodes,
        feedback=feedback,
        attempt=attempt,
        strategy=strategy or strategy_for(attempt),
    )

    if earlier is not None and attempt >= 2 and context.strategy is not Strategy.COMBINE:
        # Account for the entire rendered context before adding older web material.
        used = len(json.dumps(context.model_dump(mode="json"), ensure_ascii=False))
        context = context.model_copy(
            update={"earlier_lookups": earlier(max(0, budget_tokens * CHARS_PER_TOKEN - used))}
        )
    return context


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)  # ceiling


def fit_dependencies(
    results: list[DependencyResult], budget_tokens: int = DEFAULT_BUDGET_TOKENS
) -> list[DependencyContext]:
    """Fit dependency results into `budget_tokens`, keeping their order."""
    return fit(results, [], budget_tokens)[0]


def fit(
    results: list[DependencyResult],
    knowledge: list[KnowledgeText],
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> tuple[list[DependencyContext], list[KnowledgeText]]:
    """Fit dependency results and knowledge texts into one budget, keeping their order. The
    knowledge is material, the results are the work itself: it gets at most half the budget,
    and the results whatever it leaves."""
    budget = budget_tokens * CHARS_PER_TOKEN
    nodes = fit_knowledge(knowledge, budget // 2)
    rendered = [_render(r.content) for r in results]
    allowance = fair_shares(
        [len(text) for text in rendered], budget - sum(len(n.text) for n in nodes)
    )
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
                    content=truncate(text, chars),
                    truncated=True,
                )
            )
    return fitted, nodes


def fit_knowledge(knowledge: list[KnowledgeText], chars: int) -> list[KnowledgeText]:
    """Knowledge texts fairly within `chars`, a cut one marked truncated."""
    shares = fair_shares([len(node.text) for node in knowledge], chars)
    return [node.cut(share) for node, share in zip(knowledge, shares, strict=True)]


def given(knowledge: Knowledge | None, paths: list[str]) -> tuple[list[KnowledgeText], list[str]]:
    """The texts of the knowledge tree's `paths` a subtask was given, and the paths the tree no
    longer has (or all of them, with no tree to look in). The executor and the reviewer both
    see what this returns."""
    if knowledge is None:
        return [], list(paths)
    found: list[KnowledgeText] = []
    missing: list[str] = []
    for path in paths:
        node = knowledge(path)
        if node is None:
            missing.append(path)
        else:
            found.append(node)
    return found, missing


def _render(content: dict[str, Any] | None) -> str:
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def fair_shares(sizes: list[int], budget: int) -> list[int]:
    """Max-min fair split of `budget` (water-filling): small items get all they need."""
    shares = [0] * len(sizes)
    remaining = max(budget, 0)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])
    for position, i in enumerate(order):
        fair = remaining // (len(sizes) - position)
        shares[i] = min(sizes[i], fair)
        remaining -= shares[i]
    return shares


def truncate(text: str, chars: int) -> str:
    """Cut `text` to at most `chars`, marker included; never exceeds the share."""
    marker = f"…[truncated, {len(text)} chars in full]"
    if chars < len(marker):
        return "…"[:chars]
    return text[: chars - len(marker)] + marker
