"""Reviewer/Validator - validates subtask outputs against acceptance criteria using LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass

from desman.core.models import Artifact, Subtask
from desman.router import ModelRouter


@dataclass
class ReviewResult:
    """Result of reviewing a subtask's output."""

    accepted: bool
    feedback: str
    score: float  # 0.0 - 1.0
    required_changes: list[str] | None = None


class Reviewer:
    """Validates subtask outputs against acceptance criteria using LLM."""

    def __init__(self, model_router: ModelRouter | None = None) -> None:
        self.model_router = model_router

    def _build_review_prompt(self, subtask: Subtask, artifacts: list[Artifact]) -> str:
        """Build prompt for LLM review."""
        artifacts_json = json.dumps(
            [
                {
                    "type": art.type,
                    "content": art.content_json,
                    "path": art.path,
                }
                for art in artifacts
            ],
            ensure_ascii=False,
            indent=2,
        )

        return f"""You are a strict reviewer evaluating an AI agent's work against acceptance criteria.

**Subtask Description:** {subtask.description}
**Acceptance Criteria:** {subtask.acceptance_criteria}

**Artifacts Produced:**
{artifacts_json}

**Your Task:**
Evaluate if the artifacts satisfy the acceptance criteria. Be strict but fair.

Return ONLY a JSON object:
{{
  "accepted": true/false,
  "feedback": "detailed explanation of why accepted or rejected",
  "score": 0.0-1.0,
  "required_changes": ["specific change 1", "specific change 2"]  // only if not accepted
}}

Rules:
- accepted=true only if ALL criteria are clearly met
- score reflects completeness (1.0 = perfect, 0.0 = nothing done)
- required_changes must be specific and actionable
- If artifacts are empty or missing, reject with score < 0.3"""

    async def review(self, subtask: Subtask, artifacts: list[Artifact]) -> ReviewResult:
        """Review artifacts against acceptance criteria using LLM."""
        if not artifacts:
            return ReviewResult(
                accepted=False,
                feedback="No artifacts produced",
                score=0.0,
                required_changes=["Produce at least one artifact"],
            )

        if self.model_router is None:
            # Fallback: basic heuristic
            has_content = any(art.content_json for art in artifacts)
            if has_content:
                return ReviewResult(
                    accepted=True,
                    feedback="Artifacts produced with content (heuristic)",
                    score=0.6,
                )
            else:
                return ReviewResult(
                    accepted=False,
                    feedback="Artifacts produced but empty (heuristic)",
                    score=0.2,
                    required_changes=["Ensure artifacts contain meaningful content"],
                )

        # Use LLM for review
        prompt = self._build_review_prompt(subtask, artifacts)

        try:
            response = await self.model_router.complete_simple(
                system_prompt="You are a strict, precise reviewer. Return only valid JSON.",
                user_prompt=prompt,
                task_type="review",
            )

            result = json.loads(response)
            return ReviewResult(
                accepted=result.get("accepted", False),
                feedback=result.get("feedback", "No feedback provided"),
                score=result.get("score", 0.0),
                required_changes=result.get("required_changes"),
            )

        except json.JSONDecodeError as e:
            return ReviewResult(
                accepted=False,
                feedback=f"Reviewer returned invalid JSON: {e}",
                score=0.0,
                required_changes=["Fix reviewer output format"],
            )
        except Exception as e:
            return ReviewResult(
                accepted=False,
                feedback=f"Reviewer error: {e}",
                score=0.0,
                required_changes=["Reviewer unavailable"],
            )

    async def review_with_criteria(self, subtask: Subtask, artifacts: list[Artifact], criteria: str) -> ReviewResult:
        """Review with explicit criteria (for LLM-based review)."""
        return await self.review(subtask, artifacts)
