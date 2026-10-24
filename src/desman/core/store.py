"""Task Store - persistent storage for tasks, subtasks, artifacts, and audit logs."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from desman.core.models import (
    Artifact,
    AuditLog,
    Base,
    Subtask,
    Task,
    TaskSpec,
    TaskStatus,
)


class TaskStore:
    """Async task store with SQLite backend."""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///data/desman.db"):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close the engine."""
        await self.engine.dispose()

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get a database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # Task operations
    async def create_task(self, spec: TaskSpec) -> Task:
        """Create a new task with optional subtasks.

        If subtasks have no explicit dependencies, they are linked sequentially
        (each depends on the previous one).
        """
        async with self.session() as session:
            task = Task(goal=spec.goal, priority=spec.priority.value, status=TaskStatus.PENDING)
            session.add(task)
            await session.flush()

            created_subtasks: list[Subtask] = []
            for sub_spec in spec.subtasks:
                subtask = Subtask(
                    task_id=task.id,
                    description=sub_spec.description,
                    inputs_json=sub_spec.inputs,
                    dependencies_json=[str(d) for d in sub_spec.dependencies],
                    acceptance_criteria=sub_spec.acceptance_criteria,
                    priority=sub_spec.priority.value,
                    max_retries=sub_spec.max_retries,
                    status=TaskStatus.PENDING,
                )
                session.add(subtask)
                created_subtasks.append(subtask)

            await session.flush()  # Get UUIDs assigned

            # Wire up sequential dependencies for subtasks without explicit deps
            for i, subtask in enumerate(created_subtasks):
                if not subtask.dependencies_json and i > 0:
                    subtask.dependencies_json = [str(created_subtasks[i - 1].id)]

            await self._audit(session, "task", task.id, "create", {"goal": spec.goal})
            return task

    async def get_task(self, task_id: UUID) -> Task | None:
        """Get task by ID with subtasks loaded."""
        async with self.session() as session:
            result = await session.execute(
                select(Task)
                .options(selectinload(Task.subtasks).selectinload(Subtask.artifacts))
                .where(Task.id == task_id)
            )
            return result.scalar_one_or_none()

    async def get_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Get all tasks with given status."""
        async with self.session() as session:
            result = await session.execute(select(Task).where(Task.status == status))
            return list(result.scalars().all())

    async def update_task_status(self, task_id: UUID, status: TaskStatus) -> bool:
        """Update task status."""
        async with self.session() as session:
            result = await session.execute(
                update(Task).where(Task.id == task_id).values(status=status, updated_at=datetime.now(UTC))
            )
            if result.rowcount:  # type: ignore[attr-defined]
                await self._audit(session, "task", task_id, "status_change", {"new_status": status.value})
            return (result.rowcount or 0) > 0  # type: ignore[attr-defined,return-value]

    # Subtask operations
    async def get_ready_subtasks(self, limit: int = 10, task_id: UUID | None = None) -> list[Subtask]:
        """Get subtasks that are ready to execute (dependencies met, not leased)."""
        async with self.session() as session:
            # Subtasks with PENDING status whose dependencies are all COMPLETED
            # and not currently leased
            now = datetime.now(UTC)
            query = select(Subtask).where(
                Subtask.status == TaskStatus.PENDING,
                Subtask.lease_until.is_(None) | (Subtask.lease_until < now),
            )
            if task_id is not None:
                query = query.where(Subtask.task_id == task_id)
            query = query.order_by(Subtask.priority.desc(), Subtask.created_at.asc()).limit(limit)
            result = await session.execute(query)
            return list(result.scalars().all())

    async def lease_subtask(self, subtask_id: UUID, executor_id: str, lease_seconds: int = 300) -> Subtask | None:
        """Lease a subtask to an executor. Returns subtask if successful."""
        async with self.session() as session:
            result = await session.execute(
                select(Subtask).where(Subtask.id == subtask_id, Subtask.status == TaskStatus.PENDING)
            )
            subtask = result.scalar_one_or_none()
            if not subtask:
                return None

            # Check dependencies
            if subtask.dependencies_json:
                dep_uuids = [UUID(d) for d in subtask.dependencies_json]
                dep_result = await session.execute(
                    select(Subtask).where(Subtask.id.in_(dep_uuids))
                )
                deps = dep_result.scalars().all()
                if any(d.status != TaskStatus.COMPLETED for d in deps):
                    return None  # Dependencies not met

            lease_until = datetime.now(UTC) + timedelta(seconds=lease_seconds)
            subtask.status = TaskStatus.LEASED
            subtask.executor_id = executor_id
            subtask.lease_until = lease_until
            await self._audit(session, "subtask", subtask_id, "lease", {"executor": executor_id, "lease_seconds": lease_seconds})
            return subtask

    async def start_subtask(self, subtask_id: UUID, executor_id: str) -> Subtask | None:
        """Mark subtask as running."""
        async with self.session() as session:
            result = await session.execute(
                select(Subtask).where(Subtask.id == subtask_id, Subtask.executor_id == executor_id)
            )
            subtask = result.scalar_one_or_none()
            if not subtask or subtask.status != TaskStatus.LEASED:
                return None
            subtask.status = TaskStatus.RUNNING
            await self._audit(session, "subtask", subtask_id, "start", {})
            return subtask

    async def save_checkpoint(self, subtask_id: UUID, executor_id: str, checkpoint: dict[str, Any]) -> bool:
        """Save a checkpoint for a running subtask."""
        async with self.session() as session:
            result = await session.execute(
                select(Subtask).where(Subtask.id == subtask_id, Subtask.executor_id == executor_id)
            )
            subtask = result.scalar_one_or_none()
            if not subtask or subtask.status != TaskStatus.RUNNING:
                return False
            subtask.checkpoint_json = checkpoint
            subtask.status = TaskStatus.CHECKPOINTED
            await self._audit(session, "subtask", subtask_id, "checkpoint", {"keys": list(checkpoint.keys())})
            return True

    async def complete_subtask(self, subtask_id: UUID, executor_id: str, artifacts: list[dict[str, Any]]) -> Subtask | None:
        """Mark subtask as completed with artifacts."""
        async with self.session() as session:
            result = await session.execute(
                select(Subtask).where(Subtask.id == subtask_id, Subtask.executor_id == executor_id)
            )
            subtask = result.scalar_one_or_none()
            if not subtask or subtask.status not in (TaskStatus.RUNNING, TaskStatus.CHECKPOINTED):
                return None

            subtask.status = TaskStatus.COMPLETED
            subtask.lease_until = None
            for art in artifacts:
                artifact = Artifact(
                    subtask_id=subtask_id,
                    type=art["type"],
                    content_json=art.get("content", {}),
                    path=art.get("path"),
                )
                session.add(artifact)
            await self._audit(session, "subtask", subtask_id, "complete", {"artifact_count": len(artifacts)})
            return subtask

    async def fail_subtask(self, subtask_id: UUID, executor_id: str, error: str) -> Subtask | None:
        """Mark subtask as failed, increment retry count."""
        async with self.session() as session:
            result = await session.execute(
                select(Subtask).where(Subtask.id == subtask_id, Subtask.executor_id == executor_id)
            )
            subtask = result.scalar_one_or_none()
            if not subtask:
                return None

            subtask.retry_count += 1
            if subtask.retry_count >= subtask.max_retries:
                subtask.status = TaskStatus.FAILED
            else:
                subtask.status = TaskStatus.PENDING
                subtask.lease_until = None
                subtask.executor_id = None
            await self._audit(session, "subtask", subtask_id, "fail", {"error": error, "retry_count": subtask.retry_count})
            return subtask

    async def release_stale_leases(self, older_than: datetime | None = None) -> int:
        """Release leases that have expired."""
        if older_than is None:
            older_than = datetime.now(UTC)
        async with self.session() as session:
            result = await session.execute(
                update(Subtask)
                .where(Subtask.lease_until < older_than, Subtask.status == TaskStatus.LEASED)
                .values(status=TaskStatus.PENDING, lease_until=None, executor_id=None)
            )
            rowcount = result.rowcount or 0  # type: ignore[attr-defined]
            if rowcount:
                await self._audit(session, "system", UUID(int=0), "release_stale", {"count": rowcount})
            return rowcount  # type: ignore[return-value]

    # Artifact operations
    async def get_artifacts(self, subtask_id: UUID) -> list[Artifact]:
        """Get all artifacts for a subtask."""
        async with self.session() as session:
            result = await session.execute(select(Artifact).where(Artifact.subtask_id == subtask_id))
            return list(result.scalars().all())

    # Audit operations
    async def _audit(self, session: AsyncSession, entity_type: str, entity_id: UUID, action: str, details: dict[str, Any]) -> None:
        """Record an audit log entry."""
        log = AuditLog(entity_type=entity_type, entity_id=entity_id, action=action, details_json=details)
        session.add(log)

    async def get_audit_log(self, entity_type: str, entity_id: UUID) -> list[AuditLog]:
        """Get audit log for an entity."""
        async with self.session() as session:
            result = await session.execute(
                select(AuditLog)
                .where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
                .order_by(AuditLog.timestamp.asc())
            )
            return list(result.scalars().all())
