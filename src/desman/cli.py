"""CLI for DESMAN(P)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import UUID

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from desman.config import ConfigManager, reconfigure, run_initial_config
from desman.core.coordinator import Coordinator
from desman.core.models import TaskPriority
from desman.core.store import TaskStore
from desman.info import print_project_info
from desman.router import ModelRouter

console = Console()

CONFIG_DIR = Path.home() / ".config" / "desman"
CONFIG_FILE = CONFIG_DIR / "config.json"


def ensure_configured() -> bool:
    """Ensure DESMAN is configured, run wizard if not. Returns True if ready."""
    config_manager = ConfigManager(CONFIG_FILE)
    if not config_manager.is_configured():
        console.print("[yellow]DESMAN is not configured. Starting initial configuration...[/yellow]\n")
        run_initial_config(config_manager)
        return True
    return True


# ──────────────────────────────────────────────────────────────────────────────
# First-run setup wizard — multimodal weak-agent structure needs many API keys
# ──────────────────────────────────────────────────────────────────────────────

FREE_PROVIDERS = [
    {
        "name": "OpenRouter",
        "env_var": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/keys",
        "free_models": "Llama-3.1-8B, Gemma-2-9B, Phi-3-mini, Qwen2-7B (малое количество запросов в день)",
        "notes": "Один ключ — доступ к десяткам бесплатных моделей. Лучший старт.",
    },
    {
        "name": "Google AI Studio (Gemini)",
        "env_var": "GEMINI_API_KEY",
        "url": "https://aistudio.google.com/apikey",
        "free_models": "Gemini 1.5 Flash, Gemini 1.5 Pro (1500 запросов/день на Flash)",
        "notes": "Гугл даёт щедрые бесплатные лимиты. Нужен Google-аккаунт.",
    },
    {
        "name": "Groq",
        "env_var": "GROQ_API_KEY",
        "url": "https://console.groq.com/keys",
        "free_models": "Llama-3.1-70B/8B, Mixtral-8x7B, Gemma-7B (14400 запросов/день)",
        "notes": "Очень быстрый inference. Лимиты по токенам в минуту.",
    },
    {
        "name": "Mistral AI",
        "env_var": "MISTRAL_API_KEY",
        "url": "https://console.mistral.ai/api-keys/",
        "free_models": "Mistral-7B, Mixtral-8x7B (бесплатный тир с лимитами)",
        "notes": "Европейский провайдер, открытые веса.",
    },
    {
        "name": "Cloudflare Workers AI",
        "env_var": "CLOUDFLARE_API_KEY",
        "url": "https://dash.cloudflare.com/profile/api-tokens",
        "free_models": "Llama-3.1-8B, Mistral-7B, Gemma-7B (100k запросов/день бесплатно)",
        "notes": "Нужен Cloudflare-аккаунт. Account ID тоже понадобится (CLOUDFLARE_ACCOUNT_ID).",
    },
    {
        "name": "Cohere",
        "env_var": "COHERE_API_KEY",
        "url": "https://dashboard.cohere.com/api-keys",
        "free_models": "Command R/R+ (бесплатный тир для разработки)",
        "notes": "Хорош для RAG и эмбеддингов.",
    },
    {
        "name": "NVIDIA NIM",
        "env_var": "NVIDIA_API_KEY",
        "url": "https://build.nvidia.com/explore/discover",
        "free_models": "Nemotron-3-Ultra, Llama-3.1-Nemotron-70B (бесплатные кредиты при регистрации)",
        "notes": "Мощные модели от NVIDIA. Кредиты заканчиваются — потом платные.",
    },
    {
        "name": "Vercel AI Gateway",
        "env_var": "VERCEL_AI_GATEWAY_KEY",
        "url": "https://vercel.com/dashboard/stores/ai-gateway",
        "free_models": "Проксирует OpenRouter/Anthropic/OpenAI с логированием и кэшем",
        "notes": "Не провайдер моделей, а шлюз. Удобно для наблюдаемости и фоллбеков.",
    },
]


def _print_setup_intro() -> None:
    intro = Panel.fit(
        "[bold cyan]DESMAN(P) — Multimodal Weak-Agent Network[/bold cyan]\n\n"
        "Это не один большой LLM. Это сеть множества маленьких, короткоживущих агентов:\n"
        "  • Planner — разбивает цель на граф подзадач\n"
        "  • Executor — выполняет одну подзадачу\n"
        "  • Reviewer — проверяет результат, решает: принять / повторить / эскалировать\n"
        "  • Coordinator — управляет приоритетами, зависимостями, ретраями\n"
        "  • Improvement Analyst — предлагает улучшения (позже)\n\n"
        "Каждый агент может использовать свою модель. Чем больше у тебя ключей — тем устойчивее сеть:\n"
        "  ✓ Фоллбеки при перегрузках/банах\n"
        "  ✓ Разные модели под разные роли (планирование ≠ код ≠ ревью)\n"
        "  ✓ Распределение нагрузки по бесплатным лимитам\n\n"
        "[yellow]Тебе не нужны ВСЕ ключи.[/yellow] Минимум — один (OpenRouter). Остальные — по желанию.\n"
        "Просто нажимай Enter, чтобы пропустить провайдера.",
        title="🔧 First-run Setup",
        border_style="cyan",
    )
    console.print(intro)


def _print_providers_table() -> None:
    table = Table(title="Бесплатные источники API ключей", show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Провайдер", style="bold")
    table.add_column("Бесплатные модели / лимиты")
    table.add_column("Где взять ключ", style="blue")
    table.add_column("Примечание", style="dim")

    for i, p in enumerate(FREE_PROVIDERS, 1):
        table.add_row(str(i), p["name"], p["free_models"], p["url"], p["notes"])

    console.print(table)


def _load_existing_env() -> dict[str, str]:
    """Load existing .env file if present, return dict of key -> value."""
    env_path = Path(".env")
    if not env_path.exists():
        return {}

    existing = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            existing[key.strip()] = value.strip()
    return existing


def _prompt_keys(existing: dict[str, str]) -> dict[str, str]:
    """Interactively ask for API keys. Returns dict env_var -> key (non-empty only).

    Shows which keys are already configured and only prompts for missing ones.
    """
    keys = dict(existing)  # Start with existing keys

    # Show summary of already configured keys
    seen_vars = set()
    configured = []
    for p in FREE_PROVIDERS:
        if p["env_var"] in existing and p["env_var"] not in seen_vars:
            configured.append(p)
            seen_vars.add(p["env_var"])
    if configured:
        console.print("[bold green]Уже настроено:[/bold green]")
        for p in configured:
            val = existing[p["env_var"]]
            console.print(f"  ✓ {p['name']} ({p['env_var']}) = {val[:8]}...")
        console.print()

    # Track which env_vars we've already prompted for in this loop
    prompted_in_loop = set()

    for p in FREE_PROVIDERS:
        # Skip if already configured (from .env or earlier prompt)
        if p["env_var"] in existing:
            continue

        # Skip duplicates (same env_var appears multiple times in FREE_PROVIDERS)
        if p["env_var"] in prompted_in_loop:
            continue
        prompted_in_loop.add(p["env_var"])

        default = os.getenv(p["env_var"], "")
        prompt_text = f"[bold]{p['name']}[/bold] ({p['env_var']})"
        if default:
            prompt_text += f" [dim](текущее: {default[:8]}...)[/dim]"

        key = Prompt.ask(prompt_text, default="", show_default=False)
        if key.strip():
            keys[p["env_var"]] = key.strip()
            console.print(f"  [green]✓[/green] {p['env_var']} сохранён")
        else:
            console.print("  [dim]⊘ пропущен[/dim]")

    # Cloudflare needs Account ID too
    if "CLOUDFLARE_API_KEY" in keys and "CLOUDFLARE_ACCOUNT_ID" not in keys:
        account_id = existing.get("CLOUDFLARE_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        if account_id:
            keys["CLOUDFLARE_ACCOUNT_ID"] = account_id
            console.print("  [green]✓[/green] CLOUDFLARE_ACCOUNT_ID уже был в .env")
        else:
            account_id = Prompt.ask(
                "[bold]Cloudflare Account ID[/bold] (CLOUDFLARE_ACCOUNT_ID)", default="", show_default=False
            )
            if account_id.strip():
                keys["CLOUDFLARE_ACCOUNT_ID"] = account_id.strip()
                console.print("  [green]✓[/green] CLOUDFLARE_ACCOUNT_ID сохранён")

    return keys


def _write_env_file(keys: dict[str, str]) -> Path:
    env_path = Path(".env")
    lines = ["# DESMAN(P) Environment Configuration", "# Generated by `desman setup`", ""]

    # Group by provider for readability, deduplicate by env_var
    seen = set()
    for p in FREE_PROVIDERS:
        var = p["env_var"]
        if var in keys and var not in seen:
            lines.append(f"{var}={keys[var]}")
            seen.add(var)
    # Cloudflare Account ID
    if "CLOUDFLARE_ACCOUNT_ID" in keys and "CLOUDFLARE_ACCOUNT_ID" not in seen:
        lines.append(f"CLOUDFLARE_ACCOUNT_ID={keys['CLOUDFLARE_ACCOUNT_ID']}")

    lines.append("")
    lines.append("# Database (default: sqlite+aiosqlite:///data/desman.db)")
    lines.append("# DATABASE_URL=sqlite+aiosqlite:///data/desman.db")

    env_path.write_text("\n".join(lines) + "\n")
    return env_path


@click.group(invoke_without_command=True)
@click.option("--db", default="sqlite+aiosqlite:///data/desman.db", help="Database URL")
@click.option("--workspace", default="workspace", help="Workspace root directory")
@click.pass_context
def cli(ctx: click.Context, db: str, workspace: str) -> None:
    """DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)"""
    # Auto-run config if not configured (for any invocation including no subcommand)
    ensure_configured()

    ctx.ensure_object(dict)
    ctx.obj["db"] = db
    ctx.obj["workspace"] = Path(workspace)

    if ctx.invoked_subcommand is None:
        print_project_info()
        return


@cli.command()
@click.pass_context
def help(ctx: click.Context) -> None:
    """Show help for commands."""
    if ctx.parent:
        click.echo(ctx.parent.get_help())


@cli.command()
def config() -> None:
    """Run/re-run DESMAN configuration."""
    config_manager = ConfigManager(CONFIG_FILE)
    reconfigure(config_manager)


@cli.command()
def setup() -> None:
    """Interactive first-run setup: configure API keys for all free LLM providers."""
    _print_setup_intro()
    console.print()
    _print_providers_table()
    console.print()

    existing = _load_existing_env()
    keys = _prompt_keys(existing)

    # Check if we have at least one provider configured (existing or new)
    provider_keys = [k for k in keys if k in [p["env_var"] for p in FREE_PROVIDERS]]
    if not provider_keys:
        console.print("[yellow]Ни один ключ провайдера не настроен. Минимум нужен один (например, OPENROUTER_API_KEY).[/yellow]")
        return

    env_path = _write_env_file(keys)
    console.print(f"\n[green]Готово![/green] Ключи записаны в [bold]{env_path.absolute()}[/bold]")
    console.print("\nТеперь можешь запускать:  [cyan]desman run \"твоя цель\"[/cyan]")


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