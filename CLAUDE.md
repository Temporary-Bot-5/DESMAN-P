# CLAUDE.md - Guidance for Claude Code on DESMAN(P)

## Project Context

DESMAN(P) is a **prototype** for a decentralized, fault-tolerant, self-evolving network of small AI agents. Read `PROJECT_PLAN.md` for the full specification.

**Key constraints:**
- Zero budget - free-tier LLM APIs only (Gemini, Groq, OpenRouter, Cloudflare Workers AI, etc.)
- No local GPU - all inference via cloud APIs
- Every provider is unreliable and rate-limited
- Must handle intermittent execution (sessions stop/start)
- Human control is mandatory - no silent self-modification

## Architecture Principles (from PROJECT_PLAN.md)

1. **Durable state first** - SQLite/file-based persistence for all tasks, subtasks, artifacts, logs
2. **Resumability** - Every nontrivial task = checkpoints; crash → resume from last valid checkpoint
3. **Fault tolerance** - Detect stale tasks via leases/timeouts; reassign safely; avoid duplicate side effects
4. **Prioritization/preemption** - Explicit priority levels; critical tasks interrupt lower-priority work
5. **Minimal agent communication** - Structured task records only; no free-form agent chat
6. **Human control** - Human = final authority; no silent rewrites of security rules/permissions/evaluation
7. **Controlled self-evolution** - Propose → test in isolation → compare → record → approve → apply; all reversible via git
8. **Budget awareness** - Degrade gracefully; pause when quotas exhausted

## Current Phase: Single-Agent Execution Loop (Milestone 1)

**Goal:** Reliable single-agent loop that can create, execute, checkpoint, resume, and audit a task.

**Do NOT build yet:**
- Multiple executors / swarms
- Model routing (single provider first)
- Priority preemption
- Fault injection tests
- Self-improvement workflow
- Web UI

## Code Style & Conventions

- Python 3.11+, type hints required (mypy strict)
- Pydantic v2 for all data models
- SQLAlchemy 2.x for ORM
- Click for CLI, Rich for output
- httpx + tenacity for resilient HTTP
- anyio for async (not asyncio directly)
- Ruff for linting/format, pytest for tests

## Directory Structure

```
src/desman/
├── core/           # Task store, scheduler, coordinator
│   ├── models.py   # SQLAlchemy models (Task, Subtask, Artifact, AuditLog)
│   ├── store.py    # TaskStore class - all DB operations
│   └── scheduler.py# Scheduler/Coordinator - assigns, leases, retries
├── runtime/        # Agent runtime
│   ├── executor.py # Executor - runs one bounded subtask
│   └── planner.py  # Planner - decomposes goal into task graph
├── review/         # Reviewer/Validator
│   └── validator.py
├── router/         # Model Router (Phase 2)
├── tools/          # Tool Registry (Phase 1: file ops, code exec sandbox, search)
└── evolution/      # Improvement Analyst (Phase 3)
```

## Database Schema (Initial)

**Task:** id, goal, priority, status, created_at, updated_at, parent_task_id
**Subtask:** id, task_id, description, inputs_json, dependencies_json, acceptance_criteria, status, priority, lease_until, executor_id, checkpoint_json, created_at, updated_at
**Artifact:** id, subtask_id, type, content_json, path, created_at
**AuditLog:** id, entity_type, entity_id, action, details_json, timestamp

## Implementation Order (Smallest Milestones)

1. **Task Store** - SQLAlchemy models + TaskStore CRUD + migrations (alembic or raw SQL)
2. **Planner** - Takes goal → returns list of Subtask specs (no LLM yet, can be template-based)
3. **Executor** - Claims subtask → does bounded work → writes artifact → updates checkpoint
3. **Reviewer** - Validates artifact against acceptance criteria → accept/retry/escalate
4. **Coordinator** - Main loop: pick ready subtask → lease → dispatch executor → review → repeat
5. **CLI** - `desman run <goal>`, `desman status`, `desman resume <task_id>`
6. **Checkpoint/Resume** - Executor saves checkpoint every N steps; on restart, continue from checkpoint

## LLM Integration (When Ready)

- Model Router selects provider based on task type, availability, rate limits
- Start with ONE provider (e.g., Gemini free tier or OpenRouter free models)
- Implement: request throttling, exponential backoff, provider fallback, response caching
- Never send secrets/private code to free APIs
- Log every model call with prompt hash, response hash, tokens, latency, cost estimate

## Tool Registry (Phase 1)

Permitted tools (sandboxed):
- File read/write (restricted to workspace/)
- Code execution in isolated subprocess (timeout, no network, limited fs)
- Web search (via API)
- Simple utilities (json, text processing)

## Testing Requirements

- Unit tests for each component (store, planner, executor, reviewer)
- Integration test: full loop with simulated LLM (mock)
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
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest -v

# Lint/format
ruff check src tests
ruff format src tests
mypy src

# Run CLI (when available)
python -m desman --help
```

## References

- `PROJECT_PLAN.md` - Full specification (authoritative)
- `README.md` - Quick reference
- `pyproject.toml` - Dependencies and tool config