"""Planner - decomposes a user goal into a task graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from desman.core.models import SubtaskSpec, TaskPriority, TaskSpec


@dataclass
class TemplateSubtask:
    """Template for a subtask with placeholder dependencies."""

    description: str
    inputs: dict[str, str] = field(default_factory=dict)  # template strings
    acceptance_criteria: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    depends_on_previous: bool = False


class Planner:
    """Plans a goal into a structured task graph.

    In Phase 1, this uses template-based decomposition.
    Later phases can use LLM-based planning.
    """

    def __init__(self) -> None:
        self.templates: dict[str, list[TemplateSubtask]] = {
            "research": [
                TemplateSubtask(
                    description="Search for relevant information on the topic",
                    inputs={"query": "{goal}"},
                    acceptance_criteria="Found at least 5 relevant sources with URLs",
                    priority=TaskPriority.HIGH,
                ),
                TemplateSubtask(
                    description="Extract key facts and summarize findings",
                    inputs={"sources": "{previous_artifacts}"},
                    acceptance_criteria="Produced structured summary with citations",
                    priority=TaskPriority.NORMAL,
                    depends_on_previous=True,
                ),
            ],
            "code": [
                TemplateSubtask(
                    description="Analyze requirements and create specification",
                    inputs={"goal": "{goal}"},
                    acceptance_criteria="Created detailed spec with interfaces and data models",
                    priority=TaskPriority.HIGH,
                ),
                TemplateSubtask(
                    description="Implement core functionality",
                    inputs={"spec": "{previous_artifacts}"},
                    acceptance_criteria="Code passes tests and matches spec",
                    priority=TaskPriority.NORMAL,
                    depends_on_previous=True,
                ),
                TemplateSubtask(
                    description="Write tests and verify implementation",
                    inputs={"code": "{previous_artifacts}"},
                    acceptance_criteria="Tests cover main functionality, all pass",
                    priority=TaskPriority.NORMAL,
                    depends_on_previous=True,
                ),
            ],
            "default": [
                TemplateSubtask(
                    description="Analyze goal and break into steps",
                    inputs={"goal": "{goal}"},
                    acceptance_criteria="Produced step-by-step plan",
                    priority=TaskPriority.HIGH,
                ),
                TemplateSubtask(
                    description="Execute first step",
                    inputs={"plan": "{previous_artifacts}"},
                    acceptance_criteria="Completed first step with artifact",
                    priority=TaskPriority.NORMAL,
                    depends_on_previous=True,
                ),
            ],
        }

    def plan(self, goal: str, priority: TaskPriority = TaskPriority.NORMAL) -> TaskSpec:
        """Create a task specification from a goal."""
        # Simple keyword-based template selection
        goal_lower = goal.lower()
        if any(kw in goal_lower for kw in ["research", "find", "search", "investigate"]):
            template = self.templates["research"]
        elif any(kw in goal_lower for kw in ["code", "implement", "build", "create", "write"]):
            template = self.templates["code"]
        else:
            template = self.templates["default"]

        # Build subtasks with proper UUID dependencies
        subtasks: list[SubtaskSpec] = []
        previous_subtask_id: UUID | None = None

        for i, tmpl in enumerate(template):
            # Substitute goal into inputs
            inputs = {}
            for k, v in tmpl.inputs.items():
                if isinstance(v, str):
                    inputs[k] = v.format(goal=goal, previous_artifacts="TBD", previous_subtask_id="TBD")
                else:
                    inputs[k] = v

            # Handle dependencies - only reference previous subtask if it exists
            dependencies: list[UUID] = []
            if tmpl.depends_on_previous and previous_subtask_id is not None:
                dependencies.append(previous_subtask_id)

            subtask_spec = SubtaskSpec(
                description=tmpl.description,
                inputs=inputs,
                dependencies=dependencies,
                acceptance_criteria=tmpl.acceptance_criteria,
                priority=tmpl.priority,
                max_retries=tmpl.max_retries,
            )
            subtasks.append(subtask_spec)

            # Generate a placeholder UUID for the next subtask to depend on
            # The actual UUID will be assigned when the task is created in the store
            # For now, we'll use a deterministic approach based on index
            # This is a limitation of the template approach - real UUIDs come from DB
            # The coordinator will need to wire up actual UUIDs after creation
            # For Phase 1, we'll just track the index and the store will assign real IDs

        # We can't create real UUIDs here since they come from the DB
        # The TaskStore.create_task will assign them
        # For template purposes, we leave dependencies empty and the coordinator
        # will link them after creation

        return TaskSpec(goal=goal, priority=priority, subtasks=subtasks)
