# CLAUDE.md - Guidance for Claude Code on DESMAN(P)

## Project Context

DESMAN(P) is a **prototype** for a decentralized, fault-tolerant, self-evolving network of small AI agents.

- `PROJECT_PLAN.md` — full specification (authoritative on goals and principles)
- `ROADMAP.md` — ordered task chain; work proceeds one task (≈ one PR) at a time. Tick the box when a task is done.
- `docs/adr/` — architecture decisions. Read `0001-core-rewrite.md` before touching the core.

**Key constraints:**
- Zero budget - free-tier LLM APIs (local models via Ollama/llama.cpp are an optional extra)
- Every provider is unreliable and rate-limited
- Must handle intermittent execution (sessions stop/start)
- Human control is mandatory - no silent self-modification

## Architecture Principles (from PROJECT_PLAN.md)

1. **Durable state first** - SQLite persistence for all tasks, subtasks, artifacts, logs
2. **Resumability** - Every nontrivial task = checkpoints; crash → resume from last valid checkpoint
3. **Fault tolerance** - Detect stale tasks via leases/timeouts; reassign safely; avoid duplicate side effects
4. **Prioritization/preemption** - Explicit priority levels; critical tasks interrupt lower-priority work
5. **Minimal agent communication** - Structured task records only; no free-form agent chat
6. **Human control** - Human = final authority; no silent rewrites of security rules/permissions/evaluation
7. **Controlled self-evolution** - Propose → test in isolation → compare → record → approve → apply; all reversible via git
8. **Budget awareness** - Degrade gracefully; pause when quotas exhausted

## Current State

The old core was removed (still in git history at `3d10a5e`). What remains:

```
src/desman/
├── cli.py         # Click entry point: `desman`, `desman config`, `desman help`, `desman init-db`
├── config.py      # ConfigManager + bilingual (en/ru) setup wizard
├── info.py        # Banner / project info
└── core/
    ├── states.py  # Subtask state machine: statuses, who may take which edge, transition()
    ├── retry.py   # Failure kinds and retry policy (escalate to human, never give up silently)
    ├── db.py      # sqlite3 connection, transactions, append-only migrations
    ├── models.py  # TaskSpec/SubtaskSpec (validated DAG), Subtask read model, Priority
    ├── store.py   # Store: create_task, claim (atomic, lease token), transition (CAS + audit),
                   #        record_failure (retry policy applied atomically)
                   #        heartbeat/checkpoint, recover_expired, resume_due, sweep,
                   #        submit, task_status, cancel_task, failure propagation
    └── coordinator.py # Loop driving one task; executor/reviewer are pluggable callables
```

Database: `~/.local/share/desman/desman.db` (override with `DESMAN_DATA_DIR`). Never edit a
released migration in `db.MIGRATIONS`; append a new one. Keep the `subtasks.status` CHECK in
sync with `SubtaskStatus` (a test enforces it). All status changes go through `Store._apply()` (state-machine check, CAS,
audit in one place) — public entry points are `claim()`, `transition()`, `record_failure()`,
`submit()`, `cancel_task()` and the housekeeping methods. Never `UPDATE subtasks SET status`
anywhere else; `claim()` is the one other writer. Contracts crossing a boundary (executor /
reviewer results, provider data) are Pydantic models validated where they enter.

The core is rebuilt following `ROADMAP.md` phase by phase. Do not jump ahead of the
current phase (no multiple workers, routing, preemption or self-improvement before their phase).

## Configuration

- Single source: `~/.config/desman/config.json` via `ConfigManager` (`language`, `api_keys`,
  `local_settings`, `permissions`). Override the directory with `DESMAN_CONFIG_DIR`.
- No `.env`. Do not add a second configuration path.
- `permissions` are set by the human only. Agents must never write them; they become the
  upper bound of the tool policy.
- Subcommands must stay non-interactive; only bare `desman` and `desman config` may prompt.

## Code Style & Conventions (ADR 0001)

- Python 3.11+ (dev on 3.12), type hints everywhere, `mypy --strict`
- **Synchronous** code. No asyncio/anyio. Concurrency later = separate worker processes.
- Storage: stdlib `sqlite3`, explicit SQL, numbered migrations, WAL + busy_timeout
- Pydantic v2 for all contracts (specs, LLM responses, review verdicts)
- Click for CLI; `click.style` helpers from `config.py` for colours
- `httpx` for LLM calls (OpenAI-compatible protocol), no `openai` SDK
- Ruff for lint/format, pytest for tests
- Keep dependencies minimal: adding one needs a reason (ADR if non-trivial)

## Testing Requirements

- Every task in ROADMAP lands with tests
- Tests never touch the real config (autouse fixture sets `DESMAN_CONFIG_DIR`) or real LLM APIs
- Time in tests is `conftest.FakeClock` (pass `clock.sleep` to anything that waits); no real sleeps
  except in lock-contention tests
- LLM behaviour is tested through a scripted fake provider (ROADMAP 1.8)
- Fault test: kill executor mid-task → verify resume from checkpoint
- Idempotency test: re-run same subtask → no duplicate side effects

## What NOT To Do

- ❌ Don't add Kubernetes, p2p, message brokers, blockchain
- ❌ Don't let agents modify their own permissions/safety/evaluation
- ❌ Don't run generated code with broad system access
- ❌ Don't implement silent self-modification
- ❌ Don't assume free API quotas are stable
- ❌ Don't build complex UI before CLI is stable
- ❌ Don't create hundreds of agents - start with 1 planner, 1 executor, 1 reviewer, 1 coordinator

## Commands for Development

```bash
# Setup
uv venv -p python3.12 .venv && uv pip install -e ".[dev]"
pre-commit install

# Checks (same as CI)
ruff check .
ruff format --check .
mypy
pytest
```
