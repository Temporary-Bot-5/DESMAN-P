"""CLI for DESMAN(P)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import click
from rich.console import Console
from rich.table import Table

from desman.core.coordinator import Coordinator
from desman.core.models import TaskPriority
from desman.core.store import TaskStore
from desman.router import ModelRouter

console = Console()


@click.group()
@click.option("--db", default="sqlite+aiosqlite:///data/desman.db", help="Database URL")
@click.option("--workspace", default="workspace", help="Workspace root directory")
@click.pass_context
def cli(ctx: click.Context, db: str, workspace: str) -> None:
    """DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)"""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["workspace"] = Path(workspace)


@cli.command()
@click.argument("goal")
@click.option("--priority", type=click.Choice(["low", "normal", "high", "critical"]), default="normal")
@click.pass_context
def run(ctx: click.Context, goal: str, priority: str) -> None:
    """Run a new task with the given goal."""
    priority_map = {
        "low": TaskPriority.LOW,
        "normal": TaskPriority.NORMAL,
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL,
    }

    async def _run() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()

        # Initialize model router (will load from .env)
        try:
            model_router = ModelRouter()
            console.print(f"[green]Model router initialized with {len(model_router.providers)} provider(s)[/green]")
        except RuntimeError as e:
            console.print(f"[yellow]Warning: {e}. Running without LLM.[/yellow]")
            model_router = None

        coordinator = Coordinator(
            store=store,
            workspace_root=ctx.obj["workspace"],
            model_router=model_router,
        )

        console.print(f"[bold green]Starting task:[/bold green] {goal}")
        task = await coordinator.run_task(goal, priority_map[priority])

        console.print(f"[bold]Task {task.id} finished with status:[/bold] {task.status.value}")
        await store.close()

    asyncio.run(_run())


@cli.command()
@click.argument("task_id")
@click.pass_context
def resume(ctx: click.Context, task_id: str) -> None:
    """Resume a task from checkpoints."""
    async def _resume() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()

        # Initialize model router
        try:
            model_router = ModelRouter()
        except RuntimeError:
            model_router = None

        coordinator = Coordinator(
            store=store,
            workspace_root=ctx.obj["workspace"],
            model_router=model_router,
        )

        console.print(f"[bold yellow]Resuming task:[/bold yellow] {task_id}")
        task = await coordinator.resume_task(UUID(task_id))

        console.print(f"[bold]Task {task.id} finished with status:[/bold] {task.status.value}")
        await store.close()

    asyncio.run(_resume())


@cli.command()
@click.argument("task_id")
@click.pass_context
def status(ctx: click.Context, task_id: str) -> None:
    """Show task status and subtasks."""
    async def _status() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()

        task = await store.get_task(UUID(task_id))
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            return

        console.print(f"[bold]Task:[/bold] {task.id}")
        console.print(f"[bold]Goal:[/bold] {task.goal}")
        console.print(f"[bold]Status:[/bold] {task.status.value}")
        console.print(f"[bold]Priority:[/bold] {task.priority_enum.name}")
        console.print()

        table = Table(title="Subtasks")
        table.add_column("ID", style="cyan")
        table.add_column("Description")
        table.add_column("Status", style="magenta")
        table.add_column("Priority")
        table.add_column("Retries")

        for subtask in task.subtasks:
            table.add_row(
                str(subtask.id)[:8],
                subtask.description[:60],
                subtask.status.value,
                subtask.priority_enum.name,
                f"{subtask.retry_count}/{subtask.max_retries}",
            )

        console.print(table)

        # Show artifacts
        for subtask in task.subtasks:
            if subtask.artifacts:
                console.print(f"\n[bold]Artifacts for {subtask.id}:[/bold]")
                for art in subtask.artifacts:
                    console.print(f"  - {art.type}: {art.path or 'inline'}")

        await store.close()

    asyncio.run(_status())


@cli.command()
@click.pass_context
def list_tasks(ctx: click.Context) -> None:
    """List all tasks."""
    async def _list() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()

        # Get all tasks
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload

        from desman.core.models import Task

        async with store.session() as session:
            result = await session.execute(
                select(Task)
                .options(selectinload(Task.subtasks))
                .order_by(Task.created_at.desc())
            )
            tasks = list(result.scalars().all())

        if not tasks:
            console.print("[yellow]No tasks found[/yellow]")
            return

        table = Table(title="All Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Goal")
        table.add_column("Status", style="magenta")
        table.add_column("Priority")
        table.add_column("Subtasks")
        table.add_column("Created")

        for task in tasks:
            table.add_row(
                str(task.id)[:8],
                task.goal[:50],
                task.status.value,
                task.priority_enum.name,
                str(len(task.subtasks)),
                task.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        await store.close()

    asyncio.run(_list())


@cli.command()
@click.argument("task_id")
@click.pass_context
def audit(ctx: click.Context, task_id: str) -> None:
    """Show audit log for a task."""
    async def _audit() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()

        logs = await store.get_audit_log("task", UUID(task_id))
        logs += await store.get_audit_log("subtask", UUID(task_id))

        if not logs:
            console.print("[yellow]No audit logs found[/yellow]")
            return

        table = Table(title=f"Audit Log for {task_id}")
        table.add_column("Time")
        table.add_column("Entity")
        table.add_column("Action")
        table.add_column("Details")

        for log in logs:
            table.add_row(
                log.timestamp.strftime("%H:%M:%S"),
                f"{log.entity_type}:{str(log.entity_id)[:8]}",
                log.action,
                str(log.details_json)[:60],
            )

        console.print(table)
        await store.close()

    asyncio.run(_audit())


@cli.command()
@click.pass_context
def init_db(ctx: click.Context) -> None:
    """Initialize the database."""
    async def _init() -> None:
        store = TaskStore(ctx.obj["db"])
        await store.initialize()
        console.print("[green]Database initialized[/green]")
        await store.close()

    asyncio.run(_init())


if __name__ == "__main__":
    cli()
