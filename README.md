# DESMAN(P) - Decentralized Self-Evolving Modular Agents Network (Prototype)

A research prototype for a decentralized, fault-tolerant, self-evolving network of small AI agents.

DESMAN(P) coordinates many short-lived, specialized agents (LLM-based workers, tools, lightweight
services) to decompose large tasks, recover from agent failures, prioritize critical work, and
gradually improve its own components under human supervision.

> **Status:** the core is being rewritten (see `ROADMAP.md`). Right now only the CLI shell and
> the configuration wizard work.

## Core Principles

- **Durable state first** - No agent holds important state alone; SQLite persistence
- **Resumability** - Every task decomposed into checkpoints for crash recovery
- **Fault tolerance** - Agent failure is normal; stale tasks detected via leases/timeouts
- **Prioritization & preemption** - Critical tasks interrupt lower-priority work
- **Minimal agent communication** - Structured task records, not free-form chat
- **Human control** - Human operator is final authority
- **Controlled self-evolution** - Changes through reviewable pipeline only
- **Budget awareness** - Designed for free-tier LLM APIs with rate limits

## Quick Start

```bash
uv venv -p python3.12 .venv
uv pip install -e ".[dev]"

desman                        # first run: configuration wizard, then project info
desman config                 # providers, API keys (several per provider), permissions
desman config show            # what is configured, API keys masked
desman config reset           # delete the configuration, keys included
desman keys                   # how many keys each provider has
desman keys add groq          # add a key (asked for, hidden); --label, --stdin
desman keys remove groq 2     # remove key #2 (or --all)
desman keys import keys.txt   # add keys from a file (JSON or NAME=value lines)
desman help
```

Configuration, API keys included, is stored in `config.json` in the project folder, readable by
the owner only and ignored by git. An installed package (no project folder) uses
`~/.config/desman/config.json`; `DESMAN_CONFIG_DIR` overrides both. Git ignores the file,
but a plain copy or archive of the folder does not: leave `config.json` out when you share it.

## Development

```bash
pre-commit install
ruff check . && ruff format --check . && mypy && pytest
```

## Documents

- `PROJECT_PLAN.md` - full specification
- `ROADMAP.md` - audit of the first prototype and the ordered task chain
- `docs/adr/` - architecture decisions
- `CLAUDE.md` - guidance for AI assistants

## License

MIT
