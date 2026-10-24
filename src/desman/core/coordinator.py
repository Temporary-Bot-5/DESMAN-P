"""Coordinator - main loop managing task execution."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from uuid import UUID

from desman.core.models import Artifact, Task, TaskPriority, TaskStatus
from desman.core.store import TaskStore
from desman.review.validator import Reviewer
from desman.router import ModelRouter
from desman.runtime.executor import Executor
from desman.runtime.planner import Planner


class Coordinator:
    """Main coordination loop: plan → lease → execute → review → repeat."""

    def __init__(
        self,
        store: TaskStore,
        executor_id: str | None = None,
        workspace_root: Path | None = None,
        max_concurrent: int = 1,
        model_router: ModelRouter | None = None,
    ) -> None:
        self.store = store
        self.executor_id = executor_id or f"executor-{uuid.uuid4().hex[:8]}"
        self.workspace_root = workspace_root or Path("workspace") / self.executor_id
        self.max_concurrent = max_concurrent
        self.planner = Planner()
        self.model_router = model_router
        self.executor = Executor(self.executor_id, self.workspace_root, model_router)
        self.reviewer = Reviewer(model_router)
        self._running = False
        self._tasks: dict[UUID, asyncio.Task] = {}

    async def run_task(self, goal: str, priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        """Create and run a task to completion."""
        # Plan
        spec = self.planner.plan(goal, priority)
        task = await self.store.create_task(spec)
        await self.store.update_task_status(task.id, TaskStatus.READY)

        # Run coordination loop
        self._running = True
        try:
            while self._running:
                await self._tick(task.id)
                await asyncio.sleep(1)  # Poll interval

                # Check if task is complete
                updated_task = await self.store.get_task(task.id)
                if updated_task and all(
                    s.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                    for s in updated_task.subtasks
                ):
                    final_status = TaskStatus.COMPLETED
                    if any(s.status == TaskStatus.FAILED for s in updated_task.subtasks):
                        final_status = TaskStatus.FAILED
                    await self.store.update_task_status(task.id, final_status)
                    break
        finally:
            self._running = False

        final_task = await self.store.get_task(task.id)
        assert final_task is not None, "Task should exist after completion"
        return final_task

    async def _tick(self, task_id: UUID) -> None:
        """Single coordination tick."""
        # Release stale leases
        await self.store.release_stale_leases()

        # Get ready subtasks for this task
        ready = await self.store.get_ready_subtasks(limit=self.max_concurrent, task_id=task_id)

        for subtask in ready:
            # Lease
            leased = await self.store.lease_subtask(subtask.id, self.executor_id)
            if not leased:
                continue

            # Start
            started = await self.store.start_subtask(subtask.id, self.executor_id)
            if not started:
                continue

            # Execute (with checkpoint support)
            checkpoint = started.checkpoint_json
            try:
                artifacts, new_checkpoint = await self.executor.execute(started, checkpoint)
                await self.store.save_checkpoint(started.id, self.executor_id, new_checkpoint)

                # Review
                art_objects = [
                    Artifact(
                        id=uuid.uuid4(),  # temporary ID for review
                        subtask_id=started.id,
                        type=a["type"],
                        content_json=a.get("content", {}),
                        path=a.get("path"),
                    )
                    for a in artifacts
                ]
                review = await self.reviewer.review(started, art_objects)

                if review.accepted:
                    await self.store.complete_subtask(started.id, self.executor_id, artifacts)
                else:
                    await self.store.fail_subtask(started.id, self.executor_id, review.feedback)

            except Exception as e:
                await self.store.fail_subtask(started.id, self.executor_id, str(e))

    async def resume_task(self, task_id: UUID) -> Task:
        """Resume a task from checkpoints."""
        task = await self.store.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        await self.store.update_task_status(task_id, TaskStatus.READY)
        self._running = True
        try:
            while self._running:
                await self._tick(task_id)
                await asyncio.sleep(1)

                updated = await self.store.get_task(task_id)
                if updated and all(
                    s.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                    for s in updated.subtasks
                ):
                    final_status = TaskStatus.COMPLETED
                    if any(s.status == TaskStatus.FAILED for s in updated.subtasks):
                        final_status = TaskStatus.FAILED
                    await self.store.update_task_status(task_id, final_status)
                    break
        finally:
            self._running = False

        final_task = await self.store.get_task(task_id)
        assert final_task is not None, "Task should exist after resume"
        return final_task

    def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
