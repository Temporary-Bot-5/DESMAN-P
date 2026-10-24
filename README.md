# DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)

A research prototype for a decentralized, fault-tolerant, self-evolving network of small AI agents.

## Overview

DESMAN(P) coordinates many short-lived, specialized agents (LLM-based workers, tools, lightweight services) to decompose large tasks, recover from agent failures, prioritize critical work, and gradually improve its own components under human supervision.

## Core Principles

- **Durable state first** - No agent holds important state alone; SQLite/file-based persistence
- **Resumability** - Every task decomposed into checkpoints for crash recovery
- **Fault tolerance** - Agent failure is normal; stale tasks detected via leases/timeouts
- **Prioritization & preemption** - Critical tasks interrupt lower-priority work
- **Minimal agent communication** - Structured task records, not free-form chat
- **Human control** - Human operator is final authority
- **Controlled self-evolution** - Changes through reviewable pipeline only
- **Budget awareness** - Designed for free-tier LLM APIs with rate limits

## Architecture (v1)

```
Task Store (SQLite) ←→ Scheduler/Coordinator ←→ Agent Runtime ←→ Model Router
                                                          ↓
                                                  Tool Registry
                                                          ↓
                                                  Reviewer/Validator
                                                          ↓
                                                  Evolution Module (later)
```

**Roles:** Planner, Executor, Reviewer, Coordinator, Improvement Analyst

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run CLI (when implemented)
python -m desman --help
```

## Project Structure

```
desman-p/
├── src/desman/           # Main package
│   ├── core/             # Task store, scheduler, coordinator
│   ├── runtime/          # Agent runtime, executor
│   ├── models/           # Pydantic models for tasks, artifacts
│   ├── router/           # Model router, provider fallback
│   ├── tools/            # Tool registry, sandbox
│   ├── review/           # Reviewer/validator
│   └── evolution/        # Improvement analyst (later)
├── tests/
├── data/                 # SQLite databases, artifacts (gitignored)
├── workspace/            # Agent working directories (gitignored)
├── PROJECT_PLAN.md       # Full project specification
├── CLAUDE.md             # Guidance for Claude Code
├── pyproject.toml
└── .gitignore
```

## Development Status

**Phase 1 (Current):** Single-agent execution loop
- Task store with SQLite
- Basic planner → executor → reviewer loop
- Checkpoint/resume capability
- Audit logging

**Phase 2:** Multi-executor, model routing, priority preemption

**Phase 3:** Fault injection tests, self-improvement workflow

## Free-tier Constraints

- Uses free/trial cloud LLM APIs (Gemini, Groq, OpenRouter, Cloudflare Workers AI, etc.)
- Every provider treated as unreliable and rate-limited
- Implements fallback, throttling, caching, task pausing
- No secrets/personal data sent to free APIs without policy review

## Security Rules

- No agent modifies its own permissions/safety constraints
- No generated code with broad system access
- Sandboxing + least-privilege tools
- All decisions logged
- Self-modifications as reviewable diffs/PRs
- Rollback path preserved

## License

MIT