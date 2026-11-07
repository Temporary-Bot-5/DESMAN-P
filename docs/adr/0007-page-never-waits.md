# ADR 0007: A page that never waits

Status: accepted (owner's request, 2026-11-07)

## Context

On the page (`desman gui`, ADR 0005) one `Chat` runs on the main thread. A message waits for the
interlocutor's reply, and a task asked in the chat is worked by a coordinator inside that same
chat while it waits. Meanwhile the person can do nothing: not write in another chat, not start
another task. The owner: a person with many keys and free modules must be able to keep many chats
and tasks going at once; when modules run short, chats and tasks wait in a queue.

ADR 0001 keeps the code synchronous (no asyncio) and puts concurrency in separate worker
processes. Workers (`desman worker`, 4.2) already share one database safely: claims with leases,
rate buckets and local slots in the DB, reviews reserved first.

## Decision

1. **Tasks are worked by workers, not by the chat.** On the page, a task turn creates the task
   (as today) and returns at once. The page starts and supervises a pool of `desman worker`
   processes for as long as it runs (`Tuning.workers`, default 3, 1–8; a worker that dies is
   restarted with backoff). When the page closes, the workers stop like a worker on Ctrl+C:
   leases are released, the next start continues the tasks. Progress comes from the database
   (the panels already poll it). When a task of a session settles, the retelling is queued as a
   turn of that session (point 2). A worker started by hand beside the page is fine: claims
   decide who works what.
2. **Interlocutor turns run off the main thread, one at a time per session.** The main thread
   only reads input and dispatches. Each session with work has a turn runner: a thread with its
   own SQLite connection and its own `runs.Models` (Fallback state is per thread), running that
   session's turns in order. A message written while the session's turn is in flight is kept and
   answered next (shown as "queued"). Threads only wait on network I/O; every write goes through
   the same Store/Sessions code with its own connection (WAL + busy_timeout).
3. **Nothing pops up in the wrong chat.** A turn's outcome is appended to its session; the page
   gets a "session changed" event. The open session redraws; another one gets an unread mark in
   the sidebar. A turn that needs the person (a command's confirmation window, a clarifying
   question, an input window) waits attached to its session and is shown when the person opens
   it (a pending card in that transcript); the runner of that session waits, others go on.
4. **Queue when modules run short.** A module is (provider, key, model). Cloud calls already wait
   on the shared rate buckets and fall back to other models; local ones on the machine's slot.
   On top: at most `Tuning.chat_turns` (default 4) turns run at once; more wait in FIFO order.
   A waiting session or task shows why ("waiting for a free module: 2 ahead").
5. **Stop.** Stop on a session discards its turn in flight (the thread's call cannot be cut:
   its reply is dropped and noted) and clears that session's queue. A task is stopped by the
   pause from T.2k (`task pause`), which workers honour at their next step.
6. **The terminal chat is unchanged** (one session, synchronous). Only the page gets this.

## Consequences

- Chat turn code must not touch the front directly from a runner thread: it produces session
  messages and events; the main thread (or the page's poll) renders them.
- `ui` prompts from a runner become session-attached requests (point 3): the catalog's windows
  are code-built as before; only where they are shown changes.
- More processes: memory per worker (~60 MB); local models are still one at a time (slot).
- Tests: runner threads with FakeLLM and FakeClock-driven waits; a worker pool started by the
  page replaced by a fake launcher; fault tests unchanged (workers are the same code).

## Split

- T.2l-a: worker pool supervised by the page; task turns return at once; retelling when the
  task settles (still on the main thread); `Tuning.workers`.
- T.2l-b: per-session turn runners, queue and caps, unread marks, session-attached windows,
  per-session Stop; `Tuning.chat_turns`.
