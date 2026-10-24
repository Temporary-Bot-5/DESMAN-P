"""Tests for DESMAN(P) core functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from desman.core.coordinator import Coordinator
from desman.core.models import SubtaskSpec, TaskPriority, TaskSpec, TaskStatus
from desman.core.store import TaskStore
from desman.runtime.executor import Executor
from desman.runtime.planner import Planner


@pytest.fixture
async def store() -> TaskStore:
    """Create a test store with in-memory database."""
    test_store = TaskStore("sqlite+aiosqlite:///:memory:")
    await test_store.initialize()
    yield test_store
    await test_store.close()


@pytest.fixture
def planner() -> Planner:
    return Planner()


@pytest.fixture
def executor() -> Executor:
    return Executor("test-executor", Path("/tmp/desman_test"))


class TestTaskStore:
    """Test the TaskStore CRUD operations."""

    async def test_create_task(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            priority=TaskPriority.HIGH,
            subtasks=[
                SubtaskSpec(
                    description="First subtask",
                    acceptance_criteria="Done",
                    priority=TaskPriority.NORMAL,
                ),
                SubtaskSpec(
                    description="Second subtask",
                    acceptance_criteria="Done",
                    priority=TaskPriority.NORMAL,
                ),
            ],
        )
        task = await store.create_task(spec)
        assert task.goal == "Test goal"
        assert task.priority == TaskPriority.HIGH.value
        assert task.status == TaskStatus.PENDING

        # Reload to get subtasks
        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        assert len(reloaded.subtasks) == 2

    async def test_create_task_sequential_dependencies(self, store: TaskStore) -> None:
        """Test that subtasks without explicit deps get sequential dependencies."""
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[
                SubtaskSpec(description="First", acceptance_criteria="Done"),
                SubtaskSpec(description="Second", acceptance_criteria="Done"),
                SubtaskSpec(description="Third", acceptance_criteria="Done"),
            ],
        )
        task = await store.create_task(spec)
        # Reload to get persisted data
        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtasks = reloaded.subtasks
        # First subtask has no deps
        assert subtasks[0].dependencies_json == []
        # Second depends on first
        assert subtasks[1].dependencies_json == [str(subtasks[0].id)]
        # Third depends on second
        assert subtasks[2].dependencies_json == [str(subtasks[1].id)]

    async def test_get_ready_subtasks(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[
                SubtaskSpec(description="First", acceptance_criteria="Done", priority=TaskPriority.HIGH),
                SubtaskSpec(description="Second", acceptance_criteria="Done", priority=TaskPriority.NORMAL),
            ],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        # get_ready_subtasks returns all PENDING subtasks without lease
        # Dependency check happens in lease_subtask
        ready = await store.get_ready_subtasks(limit=10, task_id=task.id)
        assert len(ready) == 2  # Both are PENDING and not leased
        # First should be HIGH priority
        assert ready[0].priority == TaskPriority.HIGH.value

    async def test_lease_subtask(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[SubtaskSpec(description="Only task", acceptance_criteria="Done")],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        # Reload to get subtasks
        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtask = reloaded.subtasks[0]

        leased = await store.lease_subtask(subtask.id, "executor-1")
        assert leased is not None
        assert leased.status == TaskStatus.LEASED
        assert leased.executor_id == "executor-1"
        assert leased.lease_until is not None

    async def test_lease_subtask_dependency_not_met(self, store: TaskStore) -> None:
        """Test that leasing fails when dependencies aren't met."""
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[
                SubtaskSpec(description="First", acceptance_criteria="Done"),
                SubtaskSpec(description="Second", acceptance_criteria="Done"),
            ],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtasks = reloaded.subtasks

        # Try to lease second subtask (depends on first)
        leased = await store.lease_subtask(subtasks[1].id, "executor-1")
        assert leased is None  # Should fail because first not completed

    async def test_complete_subtask(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[SubtaskSpec(description="Only task", acceptance_criteria="Done")],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtask = reloaded.subtasks[0]

        await store.lease_subtask(subtask.id, "executor-1")
        await store.start_subtask(subtask.id, "executor-1")

        completed = await store.complete_subtask(
            subtask.id,
            "executor-1",
            [{"type": "result", "content": {"output": "test"}, "path": None}],
        )
        assert completed is not None
        assert completed.status == TaskStatus.COMPLETED

        # Verify artifact was created
        artifacts = await store.get_artifacts(subtask.id)
        assert len(artifacts) == 1
        assert artifacts[0].type == "result"

    async def test_fail_and_retry(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[
                SubtaskSpec(description="Flaky task", acceptance_criteria="Done", max_retries=3)
            ],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtask = reloaded.subtasks[0]

        await store.lease_subtask(subtask.id, "executor-1")
        await store.start_subtask(subtask.id, "executor-1")

        # First failure (retry_count=1, max_retries=3 -> PENDING)
        failed = await store.fail_subtask(subtask.id, "executor-1", "Error 1")
        assert failed is not None
        assert failed.status == TaskStatus.PENDING
        assert failed.retry_count == 1

        # Second failure (retry_count=2, max_retries=3 -> PENDING)
        await store.lease_subtask(subtask.id, "executor-1")
        await store.start_subtask(subtask.id, "executor-1")
        failed = await store.fail_subtask(subtask.id, "executor-1", "Error 2")
        assert failed.retry_count == 2
        assert failed.status == TaskStatus.PENDING

        # Third failure (retry_count=3, max_retries=3 -> FAILED)
        await store.lease_subtask(subtask.id, "executor-1")
        await store.start_subtask(subtask.id, "executor-1")
        failed = await store.fail_subtask(subtask.id, "executor-1", "Error 3")
        assert failed.retry_count == 3
        assert failed.status == TaskStatus.FAILED

    async def test_checkpoint_save_and_resume(self, store: TaskStore) -> None:
        spec = TaskSpec(
            goal="Test goal",
            subtasks=[SubtaskSpec(description="Long task", acceptance_criteria="Done")],
        )
        task = await store.create_task(spec)
        await store.update_task_status(task.id, TaskStatus.READY)

        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        subtask = reloaded.subtasks[0]

        await store.lease_subtask(subtask.id, "executor-1")
        await store.start_subtask(subtask.id, "executor-1")

        # Save checkpoint
        checkpoint = {"step": 5, "data": "intermediate"}
        result = await store.save_checkpoint(subtask.id, "executor-1", checkpoint)
        assert result is True

        # Verify checkpoint saved
        reloaded = await store.get_task(task.id)
        assert reloaded is not None
        assert reloaded.subtasks[0].checkpoint_json == checkpoint
        assert reloaded.subtasks[0].status == TaskStatus.CHECKPOINTED

    async def test_audit_log(self, store: TaskStore) -> None:
        spec = TaskSpec(goal="Test goal", subtasks=[SubtaskSpec(description="Task", acceptance_criteria="Done")])
        task = await store.create_task(spec)

        logs = await store.get_audit_log("task", task.id)
        assert len(logs) >= 1
        assert logs[0].action == "create"
        assert logs[0].entity_type == "task"
        assert logs[0].entity_id == task.id


class TestPlanner:
    """Test the Planner."""

    def test_plan_research(self, planner: Planner) -> None:
        spec = planner.plan("Research AI agents", TaskPriority.NORMAL)
        assert spec.goal == "Research AI agents"
        assert len(spec.subtasks) == 2
        assert spec.subtasks[0].description == "Search for relevant information on the topic"
        assert spec.subtasks[1].description == "Extract key facts and summarize findings"
        # Planner doesn't set dependencies; store wires them up sequentially

    def test_plan_code(self, planner: Planner) -> None:
        spec = planner.plan("Implement a new feature", TaskPriority.HIGH)
        assert spec.goal == "Implement a new feature"
        assert len(spec.subtasks) == 3
        # Planner doesn't set dependencies; store wires them up sequentially

    def test_plan_default(self, planner: Planner) -> None:
        spec = planner.plan("Do something random", TaskPriority.LOW)
        assert spec.goal == "Do something random"
        assert len(spec.subtasks) == 2


class TestExecutor:
    """Test the Executor."""

    @pytest.mark.asyncio
    async def test_execute_placeholder(self, executor: Executor) -> None:
        from uuid import uuid4

        from desman.core.models import Subtask, TaskStatus

        subtask = Subtask(
            id=uuid4(),
            task_id=uuid4(),
            description="Test task",
            inputs_json={},
            dependencies_json=[],
            acceptance_criteria="Done",
            status=TaskStatus.RUNNING,
            priority=TaskPriority.NORMAL,
        )

        artifacts, checkpoint = await executor.execute(subtask)
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "result"
        assert "description" in artifacts[0]["content"]
        assert checkpoint["step"] == 1
        assert "subtask_id" in checkpoint


class TestCoordinator:
    """Test the Coordinator integration."""

    @pytest.mark.asyncio
    async def test_run_simple_task(self, store: TaskStore) -> None:
        """Test running a simple task to completion."""
        coordinator = Coordinator(store=store, workspace_root=Path("/tmp/desman_test"))

        task = await coordinator.run_task("Simple test task", TaskPriority.NORMAL)

        assert task.status == TaskStatus.COMPLETED
        assert len(task.subtasks) == 2
        for subtask in task.subtasks:
            assert subtask.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_completed_task(self, store: TaskStore) -> None:
        """Test resuming an already completed task."""
        coordinator = Coordinator(store=store, workspace_root=Path("/tmp/desman_test"))

        task = await coordinator.run_task("Task to resume", TaskPriority.NORMAL)
        assert task.status == TaskStatus.COMPLETED

        # Resume should work without errors
        resumed = await coordinator.resume_task(task.id)
        assert resumed.status == TaskStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
