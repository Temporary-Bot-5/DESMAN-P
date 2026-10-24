"""Core data models for DESMAN(P) task system."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class TaskStatus(enum.StrEnum):
    """Status of a task or subtask."""

    PENDING = "pending"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, enum.Enum):
    """Priority levels (higher = more urgent)."""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class Task(Base):
    """Top-level task representing a user goal."""

    __tablename__ = "tasks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=TaskPriority.NORMAL.value, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False), default=TaskStatus.PENDING, nullable=False
    )
    parent_task_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("tasks.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    subtasks: Mapped[list[Subtask]] = relationship(
        back_populates="task", cascade="all, delete-orphan"
    )
    parent_task: Mapped[Task | None] = relationship(
        back_populates="child_tasks", remote_side=[id]
    )
    child_tasks: Mapped[list[Task]] = relationship(
        back_populates="parent_task", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_tasks_status_priority", "status", "priority"),
        Index("ix_tasks_parent", "parent_task_id"),
    )

    @property
    def priority_enum(self) -> TaskPriority:
        return TaskPriority(self.priority)

    @priority_enum.setter
    def priority_enum(self, value: TaskPriority) -> None:
        self.priority = value.value


class Subtask(Base):
    """Individual unit of work within a task."""

    __tablename__ = "subtasks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    task_id: Mapped[UUID] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    inputs_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    dependencies_json: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    acceptance_criteria: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus, native_enum=False), default=TaskStatus.PENDING, nullable=False
    )
    priority: Mapped[int] = mapped_column(Integer, default=TaskPriority.NORMAL.value, nullable=False)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    executor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    retry_count: Mapped[int] = mapped_column(default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(default=3, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    task: Mapped[Task] = relationship(back_populates="subtasks")
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="subtask", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_subtasks_task_status", "task_id", "status"),
        Index("ix_subtasks_lease", "lease_until"),
        Index("ix_subtasks_dependencies", "dependencies_json"),
    )

    @property
    def priority_enum(self) -> TaskPriority:
        return TaskPriority(self.priority)

    @priority_enum.setter
    def priority_enum(self, value: TaskPriority) -> None:
        self.priority = value.value


class Artifact(Base):
    """Output artifact produced by a subtask."""

    __tablename__ = "artifacts"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    subtask_id: Mapped[UUID] = mapped_column(
        ForeignKey("subtasks.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    subtask: Mapped[Subtask] = relationship(back_populates="artifacts")

    __table_args__ = (Index("ix_artifacts_subtask", "subtask_id"),)


class AuditLog(Base):
    """Immutable audit log of all significant events."""

    __tablename__ = "audit_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)  # task, subtask, artifact
    entity_id: Mapped[UUID] = mapped_column(nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_entity", "entity_type", "entity_id"),
        Index("ix_audit_timestamp", "timestamp"),
    )


# Pydantic models for API/validation
class SubtaskSpec(BaseModel):
    """Specification for creating a subtask."""

    description: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[UUID] = Field(default_factory=list)
    acceptance_criteria: str
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3


class TaskSpec(BaseModel):
    """Specification for creating a task."""

    goal: str
    priority: TaskPriority = TaskPriority.NORMAL
    subtasks: list[SubtaskSpec] = Field(default_factory=list)
