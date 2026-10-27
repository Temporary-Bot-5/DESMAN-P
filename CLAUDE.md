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
├── cli.py         # Click entry point: `desman`, `config`, `help`, `init-db`, and the task
│                  #   commands `run`, `resume`, `status`, `list`, `audit` (--json, exit codes),
│                  #   operator decisions `inbox`, `retry`, `accept`, `fail`, `cancel`
├── agents.py      # Minimal LLM executor/reviewer (JSON replies) behind `desman run`
├── config.py      # Config model (schema_version, several keys per provider), ConfigManager
├── config_cli.py  # `config` (wizard), `config show|reset`, `keys list|add|remove|import`
├── wizard.py      # en/ru setup wizard: provider menu loop, permissions
├── providers.py   # known model providers, readiness, key masking, key-file parsing
├── ui.py          # rich output + prompts (arrow menus in a terminal, numbered otherwise)
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
    ├── context.py # What an executor sees: inputs, checkpoint, accepted dependency results
                   #   fitted to a token budget, feedback on the previous attempt
    ├── owner.py   # Lease owner "host:pid"; owner_is_gone() lets resume take a killed run's
                   #   leases at once (only when certain)
    ├── report.py  # TaskReport/SubtaskReport read models for status/list (one snapshot)
    ├── audit.py   # Task history as AuditEntry list; check_history() replays it to prove
                   #   every status change was recorded (D7)
    ├── calls.py   # model_calls log (per provider/key usage, limits seen); `desman usage`
    ├── workspace.py   # Per-task directory + path guard: all agent file access goes through
    │                  #   Workspace (no absolute paths, `..`, symlinks; atomic writes)
    └── coordinator.py # Loop driving one task; executor/reviewer are pluggable callables
└── llm/
    ├── base.py    # Provider interface: Request/Completion, ProviderError subclasses carry a
                   #   FailureKind (the coordinator maps them; agents just let them propagate)
    ├── fake.py    # FakeLLM: scripted steps (replies, garbage, timeouts, 429, quota, 5xx, 4xx),
                   #   routes by prompt, JSON scripts, seeded chaos. All model tests use it.
    ├── openai_compat.py # HTTP client for every cloud provider: typed errors, key rotation,
                   #   CallRecord per attempt (hashes and key fingerprint, never the key)
    ├── router.py  # provider/model for a run (--provider, PREFERENCE), built from the config
    └── check.py   # `desman keys check`: is a key accepted, limits reported; spends no tokens
```

Database: `~/.local/share/desman/desman.db`, task workspaces in `workspaces/` next to it
(override the directory with `DESMAN_DATA_DIR`). Never touch files on an agent's behalf except
through `Workspace`. Never edit a
migration in `db.MIGRATIONS` once it has been pushed to any branch; append a new one (with a
backfill if needed) and test upgrading a database built with the previous migrations. Keep the `subtasks.status` CHECK in
sync with `SubtaskStatus` (a test enforces it). All status changes go through `Store._apply()` (state-machine check, CAS,
audit in one place) — public entry points are `claim()`, `transition()`, `record_failure()`,
`submit()`, `cancel_task()` and the housekeeping methods. Never `UPDATE subtasks SET status`
anywhere else; `claim()` is the one other writer. Give each audit entry a meaningful `action`;
record events that change no status (e.g. discarded work) with `Store.note()`. Contracts crossing a boundary (executor /
reviewer results, provider data) are Pydantic models validated where they enter.

The core is rebuilt following `ROADMAP.md` phase by phase. Do not jump ahead of the
current phase (no multiple workers, routing, preemption or self-improvement before their phase).

## Configuration

- Single source: `config.json` via `ConfigManager` (`language`, `api_keys`, `settings`,
  `permissions`), opened with `open_config()`. It lives in the project folder (git-ignored,
  0600); an installed package without one uses `~/.config/desman/config.json`. Override the
  directory with `DESMAN_CONFIG_DIR`.
- No `.env`. Do not add a second configuration path.
- The file is the `Config` model; a schema change bumps `schema_version` and adds a step to
  `upgrade()`. A file that does not load is a `ConfigError`, never "unconfigured".
- Never print or log an API key; show it through `providers.mask()`.
- `permissions` are set by the human only. Agents must never write them; they become the
  upper bound of the tool policy.
- Subcommands must stay non-interactive; only bare `desman`, `desman config`, `config reset`
  (confirmation, `--yes` skips it) and `keys add` (the hidden key; `--stdin` for scripts) may
  prompt. A key is never accepted as a command-line argument.

## Code Style & Conventions (ADR 0001)

- Python 3.11+ (dev on 3.12), type hints everywhere, `mypy --strict`
- **Synchronous** code. No asyncio/anyio. Concurrency later = separate worker processes.
- Storage: stdlib `sqlite3`, explicit SQL, numbered migrations, WAL + busy_timeout
- Pydantic v2 for all contracts (specs, LLM responses, review verdicts)
- Click for CLI; all human-facing output and prompts through `ui.py` (rich, one theme).
  Prompts must keep working without a terminal (numbered fallback) so tests can drive them.
  `--json` output stays plain `json.dumps`.
- `httpx` for LLM calls (OpenAI-compatible protocol), no `openai` SDK
- Ruff for lint/format, pytest for tests
- Keep dependencies minimal: adding one needs a reason (ADR if non-trivial)

## Testing Requirements

- Every task in ROADMAP lands with tests
- Tests never touch the real config (autouse fixture sets `DESMAN_CONFIG_DIR`) or real LLM APIs
  (HTTP goes through `httpx.MockTransport`)
- Time in tests is `conftest.FakeClock` (pass `clock.sleep` to anything that waits); no real sleeps
  except in lock-contention tests and the subprocess crash tests (`tests/test_crash.py`)
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
