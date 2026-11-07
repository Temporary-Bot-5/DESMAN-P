# AGENTS.md - Guidance for coding agents (Claude Code, Codex) on DESMAN(P)

`CLAUDE.md` is a symlink to this file: Claude Code and Codex read the same rules.

## Two agents, one repository

**Claude Code plans and reviews; Codex writes the code; Devin reviews the PR.**

- One flow: a branch per ROADMAP task (`task/<id>-<name>`), one PR into `master`. Never
  commit to `master` directly.
- Claude Code reads the task, decides the design (ADR when needed) and writes Codex a brief:
  files, contracts, tests, the rules of this file that matter. It runs Codex itself
  (`codex exec -C <checkout>`), reads the diff, runs the checks below and sends fixes back.
  Codex works only on the branch it was given and does not push, open or merge PRs.
- After a push: wait for Devin Review, read every finding, fix the real ones (through Codex),
  re-check. Claude Code then merges the PR itself (the owner allowed it, 2026-11-04): only with
  the local checks green (ruff, format, mypy normal and `--platform win32`, the full pytest; see
  below) and Devin's review present and handled, never before Devin has posted.
- CI (GitHub Actions: Python 3.11–3.13 and Windows) runs only before a release — a `v*` tag —
  or by hand (`gh workflow run ci.yml --ref BRANCH`), not on every PR (owner, 2026-11-07). Run
  it by hand when a change touches platform code (paths, processes, signals, files, encodings):
  Windows is tested nowhere else.
  It tells the owner what was merged and what they must do after (e.g.
  `desman protected approve`).

Claude Code and Codex do not remember each other's sessions. They talk
through `HANDOFF.md` in the project root (not in git, so it survives branch switches; from a
worktree use the main checkout's copy). Read it at the start of every session, claim your
area there before working, and update it (state, claim removed, one log line) before you stop.
Uncommitted changes that `HANDOFF.md` does not explain are someone else's: leave them alone
and ask the owner. The rules of the file are at its top.

## Project Context

DESMAN(P) is a **prototype** for a decentralized, fault-tolerant, self-evolving network of small AI agents.

- `PROJECT_PLAN.md` — full specification (authoritative on goals and principles)
- `ROADMAP.md` — ordered task chain; work proceeds one task (≈ one PR) at a time. Tick the box when a task is done.
- `docs/adr/` — architecture decisions. Read `0001-core-rewrite.md` before touching the core.

**Key constraints:**
- Zero budget by default - free-tier LLM APIs (local models via Ollama/llama.cpp are an optional
  extra). Paid providers (`providers.PAID`) exist but are never chosen automatically: only by
  name (`--provider`), never add one to `PREFERENCE`
- Every provider is unreliable and rate-limited
- Must handle intermittent execution (sessions stop/start)
- Human control is mandatory - no silent self-modification
- Accessible to non-developers: after configuration, bare `desman` becomes a chat (ROADMAP
  phase C, after phase 3). Requests become ordinary tasks; changes to DESMAN itself asked in
  the chat take effect only after the human confirms them in a field rendered by code. An
  explicit `/COMMAND` (or `!COMMAND`) typed by the person runs immediately, like the CLI.

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
├── cli.py         # Click entry point: `desman` (banner, then the chat), `config`, `help`,
│                  #   `init-db`, and the task
│                  #   commands `run`, `resume`, `status`, `list`, `audit` (--json, exit codes),
│                  #   operator decisions `inbox`, `retry`, `accept`, `fail`, `cancel`; `stats`;
│                  #   `worker`: every task's subtasks until stopped (N at once on one database)
│                  #   `proposals`: what the Improvement Analyst proposed (5.4)
├── actions.py     # Bodies of the task commands (data functions + printers), shared by the CLI
│                  #   and the chat's catalog; problems are click.ClickException
├── commands.py    # The chat's catalog of every CLI command (a test checks both ways): about
│                  #   for the model, Args (Pydantic, extra forbidden), class per call (read |
│                  #   change → Proposal shown in a window | input → hidden field / wizard);
│                  #   config_changes() for the operator's journal
├── agents.py      # LLM executor/reviewer behind `desman run`: retry/escalate need feedback,
│                  #   `{"type": "error"}` = can't do it (never accepted, see coordinator)
│                  #   executor acts only via `{"tool_calls": [...]}` (≤8 rounds per attempt)
├── planner.py     # Goal → subtask graph by the model (Plan, ≤8 nodes, DAG, each tagged with its
│                  #   Kind); template on failure; decomposer(): splits a subtask whose retries
│                  #   failed (2–4 parts, untagged ones keep the subtask's kind)
├── config.py      # Config model (schema_version, several keys per provider), ConfigManager
├── config_cli.py  # `config` (wizard), `config show|reset`, `keys list|add|remove|import|check`,
│                  #   `language [CODE]`, `permissions [--grant|--revoke]`, `personalize [on|off]`,
│                  #   `behaviour [--length|--tone|--style|--custom|--reset]`, `usage`; bodies as
│                  #   functions the chat's catalog calls too
├── models_cli.py  # `models show|add MODEL`: a Hugging Face GGUF model for a local server
├── protected_cli.py # `protected show [--diff]|approve [--note]`: bodies shared with the chat
├── improve.py     # Improvement Analyst (5.4): an ordinary task without tools reads failures,
│                  #   names TARGETS (planner prompts in plain constants; never a protected file)
│                  #   with new texts, code builds the diff; or SETTINGS (`Config.tuning`),
│                  #   checked against what it was shown. Applies nothing (applying.py)
├── bench.py       # The benchmark (5.5): fixed goals planned and worked to the end on a DB of
│                  #   its own, no tools; `python -m desman.bench` reports a Result; compare()
│                  #   says which run went better. Protected, like evaluation.py
├── evaluation.py  # `proposals evaluate`: current version and proposed one (diff + settings)
│                  #   in git worktrees of HEAD, each run as its own process; the diff checked
│                  #   to change only the targets' texts; recorded once in `evaluations`
├── global_cli.py  # `global [status]|join [--registry]|leave|donate P|undonate P|quota P`
│                  #   (G.3): consent to desman-global, the node key, modules lent with daily
│                  #   quotas (`providers.can_donate`: local always, free cloud only if
│                  #   `donatable`, paid never); each change in `network_log`; only `leave`
│                  #   talks to the registry (says goodbye while still on)
├── tuning_cli.py  # `tuning`: the model each role asks first, the limits of the workflow
│                  #   (`runs.TUNING_DEFAULTS` when unset); set by the person only
├── proposals_cli.py # `proposals [list]|show N|evaluate N [--real]|apply N|revert N` (and
│                  #   `analyze` in cli.py)
├── applying.py    # `proposals apply|revert` (5.6): only the person's command; the diff becomes a
│                  #   git commit built by code in a worktree (as evaluation builds it), then the
│                  #   checkout fast-forwarded; settings into `tuning`; revert = `git revert` the
│                  #   same way. Also installs/reverts root-signed releases (`global update`).
│                  #   Recorded in `applications` (in turn). Protected
├── local_models.py # their bodies (look up, size vs free memory, plan, fetch, use), shared by
│                  #   the CLI, the wizard and the chat's catalog
├── wizard.py      # setup wizard (language, provider menu loop, permissions, personalisation
│                  #   consent, how the interlocutor talks, desman-global last)
├── providers.py   # known model providers (cloud; local: Ollama, llama.cpp, ready once
│                  #   their model is set), readiness, key masking, key-file parsing
├── ui.py          # rich output + prompts (arrow menus in a terminal, numbered otherwise);
│                  #   the chat's line (prompt_toolkit: `/` lists the chat's commands, Tab)
├── i18n.py        # t()/tn(): messages, command help and Click's own text in the configured
│                  #   language (ADR 0002); catalogs in locales/ (ru, zh, es, pt, de, fr, ja,
│                  #   ko, pl, tr, uk, hi)
├── info.py        # Banner / project info
├── chat.py        # The chat of bare `desman`: a new session (`-c` continues the last one,
│                  #   `-t` a temporary chat, erased when it ends); /help, /new,
│                  #   /sessions, /rename, /compact, /profile, /delete, /clear, /context, /exit; each message gets the interlocutor's turn; a task turn
│                  #   runs a task of the session while the chat waits, then it is retold; a
│                  #   command turn goes through the catalog: the window and the person's "yes"
│                  #   are code's, the result goes back as an event, changes to operator_log
├── interlocutor.py # The interlocutor: Turn (reply | clarify | task | command), no tools;
│                  #   retell() a task's outcome; the catalog is in its prompt. Prompts
│                  #   "Chat turn", "Retell task". Talks as `Config.behaviour` says (always);
│                  #   with `Config.personalize` it also sees the profile and any turn may edit
│                  #   it (`profile`: add | change | forget, ≤3), the chat applies and shows them
│                  #   A `lookup` turn first (≤2 topics of `manual`) → asked again with their texts
├── manual.py      # The interlocutor's manual (C.10): a short English text per topic of what
│                  #   DESMAN does now (never plans); its prompt lists only the topics. A test
│                  #   checks the commands it names and the size of each topic
├── compaction.py  # A history over the interlocutor's budget: older messages summarised by an
│                  #   ordinary task without tools (`runs.summarizer`), kept as a `summary`
│                  #   message that `covers` them; failed → they fall out of view, with a note
├── runs.py        # A run put together from the config (models: a lineup per role, the
│                  #   executor's per subtask Kind, each model
│                  #   paced, backed off, cached; executor access, web tools, coordinator),
│                  #   shared by `run`/`resume`/`worker` and the chat
├── task_files.py  # A task's files for the chat (through Workspace): listed with their folder
│                  #   by code after a retelling, given to the retelling to copy from, copied
│                  #   into a later task of the same chat (`NewTask.files_from` → from-REF/)
├── live.py        # Live progress of a chat task: subtask bar, a line per module (role, calls,
│                  #   tokens, limits left, state); redrawn on coordinator steps and calls
└── core/
    ├── states.py  # Subtask state machine: statuses, who may take which edge, transition()
    ├── scheduler.py # Urgency = priority aged by the wait in its queue (`queued_at`), up to
    │              #   CRITICAL − 1; claim and reviews follow it across every task (4.3)
    ├── retry.py   # Failure kinds and retry policy (escalate to human, never give up silently);
    │              #   Strategy per attempt number (feedback, rephrase, other model, decompose →
    │              #   combine), audited as attempt_started; Store.decompose() makes the parts
    ├── db.py      # sqlite3 connection, transactions, append-only migrations
    ├── models.py  # TaskSpec/SubtaskSpec (validated DAG; retry budgets and `Tools` kept with
                   #   the task: `none` = no tools for whoever runs it), Subtask, Priority,
                   #   Kind (text | code | analysis: picks the executor's model)
                   #   `runs_code`: done only through a successful run_command (coordinator
                   #   checks; without the tool → needs_human at once; never split)
    ├── store.py   # Store: create_task, claim (atomic, lease token), transition (CAS + audit),
                   #        record_failure (the task's own retry policy, `TaskSpec.retry`,
                   #        applied atomically; no caller brings one), reserve_review
                   #        heartbeat/checkpoint, preempt (critical work takes over, 4.4),
                   #        recover_expired, resume_due, sweep,
                   #        submit, task_status, cancel_task, failure propagation
    ├── context.py # What an executor sees: inputs, checkpoint, accepted dependency results
                   #   fitted to a token budget, feedback on the previous attempt
    ├── sessions.py # Chat sessions and messages (append-only: role, text, kind
    │              #   message | summary | event, task link); a session starts with its first message
    │              #   history(): the latest summary and the messages after those it covers
    │              #   a session may be named; deleted = out of the lists, messages kept
    │              #   temporary (C.11): never listed, erase()d when its chat ends, sweep()
    │              #   after a crash; the database refuses to erase any other session
    ├── netlog.py  # network_log: what desman-global did (joined, left, lent, quota); keys by
    │              #   fingerprint, never a secret; append-only
    ├── operator.py # operator_log: what the person did outside tasks (chat commands confirmed
    │              #   or declined, keys by fingerprint, permissions); append-only
    ├── profile.py # User's profile (C.7): short notes for the interlocutor (≤20, ≤200 chars),
    │              #   changed and erased in place (personal data); each change in operator_log
    │              #   by note number, never the text; `Config.personalize` off → unseen, unkept
    ├── owner.py   # Lease owner "host:pid"; owner_is_gone() lets resume take a killed run's
                   #   leases at once (only when certain)
    ├── report.py  # TaskReport/SubtaskReport read models for status/list (one snapshot)
    │              #   InboxWatch: what came to wait for the person (worker, chat; 5.1)
    ├── stats.py   # `desman stats` (5.2): outcomes, first tries, retries, verdicts, failures,
    │              #   attempts by strategy, calls/tokens by role and model; read only
    ├── audit.py   # Task history as AuditEntry list; check_history() replays it to prove
                   #   every status change was recorded (D7)
    ├── calls.py   # model_calls log (per provider/key usage, limits seen); `desman usage`;
    │              #   calls.for_(Purpose) attributes calls (role, task/subtask or session)
    ├── workspace.py   # Per-task directory + path guard: all agent file access goes through
    │                  #   Workspace (no absolute paths, `..`, symlinks; atomic writes)
    ├── tools.py   # Tool registry: Tool (args schema, Permission), Registry, built-in file
    │              #   tools; Toolbox per subtask (only allowed tools, lease guard, audit);
    │              #   suggest_model: an agent only suggests a HF model (audit `model_suggested`,
    │              #   shown by `status` and the chat; the person runs `models add`)
    ├── policy.py  # Access = role's tools in `src/desman/policy.toml` ∩ human's `permissions`;
    │              #   refusals audited (tool_refused), admin calls marked elevated
    ├── proposals.py # The analyst's proposals: analysing → proposed | nothing | unusable, settled
    │              #   once, never changed; the failures and plan fallbacks an analysis reads;
    │              #   their evaluations (append-only)
    ├── protected.py # Protected files (5.3): `src/desman/protected.toml` lists them with SHA-256;
    │              #   approved snapshots in the DB; a change nobody approved blocks every run
    │              #   (`runs.guard`) until `desman protected approve`; a new DB takes the release
    ├── effects.py # Call journal (D6): intent → run → result per tool call; a rerun of an
    │              #   interrupted execution replays results (from its checkpoint) instead of acting
    ├── web.py     # web_search and web_fetch: GET only, public addresses only (checked, then
    │              #   connected to), text only, limits, web_cache; results marked untrusted
    ├── search.py  # Search engines: open sources (Wikipedia, Stack Overflow, arXiv, GitHub; no
    │              #   key), the human's SearXNG, Tavily. Official APIs only, never pages for people
    ├── sandbox.py # Generated code runs only under bwrap: no network, only /usr (read-only) and
    │              #   the workspace, cleared env, rlimits, timeout; unavailable → nothing runs
    └── coordinator.py # Loop driving one task (`run`) or every task (`work`, a worker);
                       #   executor/reviewer are pluggable callables; a review is reserved first
                       #   (`Store.reserve_review`), so two workers never pay for one verdict
                       #   an executor's step (heartbeat, tool call) yields to waiting critical
                       #   work (`PreemptedError`), which runs at once; the dropped one resumes
└── llm/
    ├── base.py    # Provider interface: Request/Completion, ProviderError subclasses carry a
                   #   FailureKind (the coordinator maps them; agents just let them propagate)
    ├── fake.py    # FakeLLM: scripted steps (replies, garbage, timeouts, 429, quota, 5xx, 4xx),
                   #   routes by prompt, JSON scripts, seeded chaos. All model tests use it.
    ├── openai_compat.py # HTTP client for every cloud provider: typed errors, key rotation,
                   #   CallRecord per attempt (hashes and key fingerprint, never the key)
    ├── structured.py # complete_json(): JSON from a reply, validated by a Pydantic schema,
                   #   one repair request showing the model its errors, then StructuredOutputError
    ├── cache.py   # Cached: accepted replies by exact request (response_cache, 7 days);
                   #   complete_json() accepts, garbage is never kept; --no-cache skips it
    ├── catalog.py # Models per provider (default first): context, strict JSON, strengths
                   #   (code, text, analysis), free-tier rpm; need(): what each role asks (the
                   #   executor: what its subtask's Kind asks)
    ├── router.py  # lineup(): a role's models, best first (need, JSON, PREFERENCE, catalog);
                   #   --provider narrows it, --model pins one; other_lineup() for "other model"
                   #   Fallback: a call moves to the next model on a limit, outage, refused key,
                   #   too small a context (not on the model's own failure, nor a slow one)
                   #   Backoff: quick blips (5xx, drops, limits ≤10 s) retried before they cost an
                   #   attempt; slow failures never (a lease is not renewed while waiting)
    ├── limiter.py # Token bucket per provider/model in the DB (rate_buckets), limits a provider
                   #   reported kept as blocked_until: shared by workers, outlives restarts;
                   #   Local: a local server paced by the machine (one request at a time, a
                   #   file lock; not while the load is high; left alone after no answer)
    ├── check.py   # `desman keys check`: is a key accepted, limits reported, catalog models the
                   #   provider does not list; a local server: running, serves its model;
                   #   spends no tokens
    └── hub.py     # Hugging Face: parse author/model[:quant], inspect (GGUF files, size,
                   #   gated), check_access, free RAM/video memory, Ollama's pull; HF_TOKEN
                   #   goes to huggingface.co only. Downloads nothing unless the person said yes
└── net/           # desman-global, the node's side (ADR 0004); never runs while it is off
    ├── __init__.py # registry(config): the registry's address, None while it is off — the
    │              #   one way to it; DEFAULT_REGISTRY (None until the owner's registry runs)
    ├── sign.py    # Ed25519: node.key (0600, shown by fingerprint only), canonical JSON, Signed
    │              #   Record; Trust.receive(): schema, signer, signature, scope, term, replay;
    │              #   root → operator (assignment) → record; revocations; Refused(Reason).
    │              #   Protected
    ├── request.py # Signed requests to the registry: method, path, body hash, time, nonce by the
    │              #   node key; verify_request() refuses forged, stale, replayed. Protected
    ├── protocol.py # Paths and bodies node ↔ registry (Pydantic): capabilities lent (provider,
    │              #   model, key fingerprint, quotas), never a key
    ├── node.py    # The node in the network (G.5): `Node.poll()` between a worker's steps —
    │              #   register with what is lent, heartbeat, trust list checked by `Trust`;
    │              #   config re-read before each contact; backoff; revoked → silent;
    │              #   `tell_left` for `global leave`; all in `network_log`. G.6b: asks for a
    │              #   job only while idle, checks it, extends its lease, hands in the record
    ├── updates.py # Receives root-signed releases; rechecks signatures before manual installation
    ├── jobs.py    # A job lent (G.6b) as a task of origin `network` (`network_work`): LOW,
    │              #   after and preempted by the person's own work; worked by the crew
    │              #   (`runs.network_crew`: lent models within `Quota`, `network_*` roles);
    │              #   a recheck replays the other node's result for review; `ending()` → a
    │              #   signed result/verdict; seen record ids kept (`Seen`); never in the inbox
    └── root.pub   # The root's public key (its private half: the owner's server). Protected
└── registry/      # desman-registry (G.4), the owner's server; reads no config.json, keeps no key
    ├── store.py   # Its own SQLite and migrations (core.db helpers): nodes, root-signed records
    │              #   handed out, nonces, registry_log (append-only)
    ├── server.py  # Registry.handle() (tests: MockTransport) behind ThreadingHTTPServer on
    │              #   127.0.0.1; every request verified first, refusals journaled
    ├── work.py    # Jobs (G.6): an operator's `task` record lent with a lease and token
    │              #   (claim | extend | hand-in), done, then rechecked by another owner's node;
    │              #   rejected, failed or leases lost → the operator's inbox (`job settle`)
    ├── admin.py   # assign / revoke signed by root.key (checked against root.pub and as a node
    │              #   would), backup (SQLite backup API); job add / settle, owner
    └── cli.py     # `desman-registry serve|backup|assign|revoke|nodes|log|job|jobs|inbox|owner`
                   #   ($DESMAN_REGISTRY_DIR, $DESMAN_REGISTRY_LANGUAGE); docs/registry-deploy.md
```

Database: `~/.local/share/desman/desman.db`, task workspaces in `workspaces/` next to it
(override the directory with `DESMAN_DATA_DIR`). Never touch files on an agent's behalf except
through `Workspace`, and let executors act only through tool calls (`Work.tools`): a new
capability is a new `Tool` with the permission it needs, never a direct call in an agent.
A commit that changes a file listed in `src/desman/protected.toml` updates its hash there
(`tests/test_protected.py` prints the line); agents never touch that list or its approvals. Every tool call is journaled (`core/effects.py`); a tool safe to run twice declares `repeatable`. Generated code runs only through `Sandbox` (the `run_command` tool), never with `subprocess` directly. Never edit a
migration in `db.MIGRATIONS` once it has been pushed to any branch; append a new one (with a
backfill if needed) and test upgrading a database built with the previous migrations. Keep the `subtasks.status` CHECK in
sync with `SubtaskStatus` (a test enforces it). All status changes go through `Store._apply()` (state-machine check, CAS,
audit in one place) — public entry points are `claim()`, `transition()`, `record_failure()`,
`submit()`, `cancel_task()` and the housekeeping methods. Never `UPDATE subtasks SET status`
anywhere else; `claim()` is the one other writer. Give each audit entry a meaningful `action`;
record events that change no status (e.g. discarded work) with `Store.note()`. Contracts crossing a boundary (executor /
reviewer results, provider data) are Pydantic models validated where they enter.

The core is rebuilt following `ROADMAP.md` phase by phase (0–3, then C — chat, then 4–5, then
G — desman-global). Do not jump ahead of the current phase (no chat, multiple workers, routing,
preemption, self-improvement or network features before their phase).

## Configuration

- Single source: `config.json` via `ConfigManager` (`language`, `api_keys`, `settings`,
  `permissions`, `personalize`, `behaviour`, `tuning`, `network`), opened with `open_config()`. It lives in the project folder (git-ignored,
  0600); an installed package without one uses `~/.config/desman/config.json`. Override the
  directory with `DESMAN_CONFIG_DIR`.
- No `.env`. Do not add a second configuration path.
- The file is the `Config` model; a schema change bumps `schema_version` and adds a step to
  `upgrade()`. A file that does not load is a `ConfigError`, never "unconfigured".
- Never print or log an API key; show it through `providers.mask()`.
- `permissions` are set by the human only (`desman config`, `desman permissions`, or a chat
  request the human confirms in the window). Agents must never write them; they are the
  upper bound of the tool policy (`src/desman/policy.toml`, which is code: changed by people
  through git, never by an agent). `network` (desman-global) likewise: set by the human only,
  off by default; anything that talks to the registry gets its address from `net.registry()`.
- A new CLI command also gets an entry in `commands.CATALOG` (a test requires it); the chat
  calls the same body function as the CLI, never the CLI itself.
- A change that alters what DESMAN does for the person also updates the topic of
  `src/desman/manual.py` that describes it, in the same commit (the interlocutor answers from it).
- Subcommands must stay non-interactive; only bare `desman` (the chat, from phase C), `desman config`, `config reset`
  (confirmation, `--yes` skips it), `inbox` in a terminal, and `keys add` (the hidden key;
  `--stdin` for scripts) may
  prompt. A key is never accepted as a command-line argument.

## Code Style & Conventions (ADR 0001)

- Python 3.11+ (dev on 3.12), type hints everywhere, `mypy --strict`
- **Synchronous** code. No asyncio/anyio. Concurrency later = separate worker processes.
- Storage: stdlib `sqlite3`, explicit SQL, numbered migrations, WAL + busy_timeout
- Pydantic v2 for all contracts (specs, LLM responses, review verdicts)
- Click for CLI; all human-facing output and prompts through `ui.py` (rich, one theme).
  Prompts must keep working without a terminal (numbered fallback) so tests can drive them.
  `--json` output stays plain `json.dumps`.
- Every human-facing string goes through `i18n.t()` / `tn()` with a literal English text, and
  gets a translation in every catalog in `locales/` (12 languages; a plural needs each of the
  language's forms; `tests/test_i18n.py` lists what is missing). Not
  translated: `--json` output, audit entries and reasons stored in the DB, prompts to models,
  messages raised in `core/`.
- `httpx` for LLM calls (OpenAI-compatible protocol), no `openai` SDK
- Ruff for lint/format, pytest for tests
- Keep dependencies minimal: adding one needs a reason (ADR if non-trivial)

## Testing Requirements

- Every task in ROADMAP lands with tests
- Tests never touch the real config (autouse fixture sets `DESMAN_CONFIG_DIR`) or real LLM APIs
  (HTTP goes through `httpx.MockTransport`)
- Time in tests is `conftest.FakeClock` (pass `clock.sleep` to anything that waits); no real sleeps
  except in lock-contention tests, the subprocess crash and reassignment tests (`tests/test_crash.py`, `tests/test_reassign.py`) and the
  real sandbox runs (`tests/test_sandbox.py`, skipped without bubblewrap except in CI)
- LLM behaviour is tested through a scripted fake provider (ROADMAP 1.8)
- Fault test: kill executor mid-task → verify resume from checkpoint
- Fault injection (`tests/faults.py`, ROADMAP 4.6): seeded simulation of several workers with
  crashes, hangs and provider failures, then invariants; a failing seed replays with
  `DESMAN_FAULT_SEED=N pytest tests/test_faults.py`. A change to the coordinator or the store
  is worth a longer hunt: `DESMAN_FAULT_SEEDS=500`
- Idempotency test: re-run same subtask → no duplicate side effects

## What NOT To Do

- ❌ Don't add Kubernetes, p2p, message brokers, blockchain
- ❌ Don't let agents modify their own permissions/safety/evaluation (a chat request is applied
  only after the human's confirmation in a field built by code, never on the model's word)
- ❌ Don't let an API key reach a model, the chat history, or the logs
- ❌ Don't run generated code with broad system access
- ❌ Don't implement silent self-modification
- ❌ Don't assume free API quotas are stable
- ❌ Don't build complex UI before CLI is stable
- ❌ Don't create hundreds of agents - start with 1 planner, 1 executor, 1 reviewer, 1 coordinator
  (plus 1 interlocutor for the chat in phase C)

## Commands for Development

```bash
# Setup
uv venv -p python3.12 .venv && uv pip install -e ".[dev]"
pre-commit install

# Checks (before every merge; CI runs the same before a release)
ruff check .
ruff format --check .
mypy
pytest
```
