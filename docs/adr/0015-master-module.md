# ADR 0015: Master module

Status: accepted (2026-11-21, ROADMAP M.1–M.3)

## Context

DESMAN is built for weak, unreliable free models: the planner cuts a goal into at most
`MAX_SUBTASKS` small subtasks, each worked in at most `agents.MAX_TOOL_ROUNDS` (8) rounds of
tool calls, reviewed, retried and passed on to the next subtask as a fitted result. For a free
8B model that is what makes the work possible at all.

For a strong model the same shape is a handicap. A goal that one capable agent would do in one
sitting (change a program across several files, research and write one coherent text) is cut
into pieces that each see only part of the picture; every hand-over loses context and costs a
call and a review. Some providers give strong models on good free terms (NVIDIA's API: a 550B
model, 40 requests a minute), and the person may run a capable local model.

Ordinary "one strong agent in a loop" programs exist in plenty. What DESMAN can add is the
infrastructure around such an agent: durable state and resumption, the call journal (no side
effect twice), the workspace guard and the sandbox, the reviewer, retries by strategy, the
person's inbox, the knowledge tree, the rate limiter with fallback between models, and the
cluster of small models for the work that does not need a strong one.

## Decision

**A master module is a mode of a subtask, not a new agent: one subtask holds the whole goal
and is worked by a qualified model in a long loop of tool calls, inside everything DESMAN
already does for a subtask.**

1. **The mode.** `SubtaskSpec.master` (stored with the subtask). A master subtask's executor
   asks the master lineup instead of its kind's, and may take up to
   `agents.MASTER_TOOL_ROUNDS` (24) rounds of tool calls in one attempt instead of 8, told so
   in its instructions (plan your own steps, check the work before handing it in); the person
   may set the number (`desman tuning`, `master_rounds`, M.3). Everything else is the
   ordinary path: the same tools within the same access, workspace, sandbox, call journal,
   heartbeat and checkpoints, the same reviewer, the same retry policy. There is still one
   planner, one executor, one reviewer, one coordinator: the master is the executor with
   other models and a longer leash.
2. **Who may be a master ("qualified").** Decided by code, never by a model:
   - a cloud model the catalog marks `master` (`llm.catalog.Model.master`): a large context
     (≥ 128k tokens), keeps to JSON, strong at analysis, no small per-request cap (such a cap
     ends a long loop at its first big request); `catalog.qualifies` checks that basis and a
     test holds every mark to it. Marking a model is a change to code, reviewed like any
     other. First marked: NVIDIA's Nemotron 3 Ultra, Gemini 3.8 Flash, Mistral Medium,
     Cohere Command A, and the paid defaults;
   - a local model only when the person names it (`tuning.models.master`, or `--provider` /
     `--model` of the run) and its context is at least `catalog.MASTER_LOCAL_CONTEXT`
     (32 768 tokens): what a local machine "can carry" is known only to the person, but a
     short context cannot hold a long loop at all;
   - paid models may be marked, but, as everywhere, are asked only when the person names the
     provider (`--provider`); never chosen automatically.
   The person may put one qualified model first (`desman tuning --model master=...`); an
   unqualified one is never asked (`router.qualified`), and `desman tuning` warns that it is
   not among the models the master may use. The Improvement Analyst never proposes the
   master's model, nor admits one.
   **The measured gate (M.3).** `desman master check PROVIDER/MODEL` runs the benchmark
   (`bench.py`, protected) in master mode: each case is one master subtask (no tools) worked
   by that model alone and judged by the ordinary reviewer, on a database of its own. It
   passes when every case is done with nothing escalated (`bench.master_passed`, in the
   protected file, so no proposal can lower the bar). Each check is kept (`master_checks`,
   append-only). `desman master admit MODEL` (the person, like permissions: `Config.master`,
   never an agent) records an admission only on a passing check of the current benchmark
   version; `desman master revoke MODEL` takes a model out, a catalog-marked one included.
   Qualified is then: (marked in the catalog, or admitted) and not revoked, and the basic
   limits still hold (context ≥ `MASTER_LOCAL_CONTEXT`, no small per-request cap); a local
   model admitted no longer has to be named in tuning. The catalog's mark is only the
   starting list.
3. **Who chooses the mode.** The person: `desman run --master` (M.1); in the chat (M.2) the
   interlocutor's task turn carries `"master": true` when the person asked for the work to be
   done by one agent, and `/master on|off` sets it for the chat session while the chat
   runs. It never switches automatically, since a silent switch would leave the cluster unused
   and hide what a task costs.
   Without a ready qualified model `--master` (or the chat) says so and creates nothing; a
   master subtask that finds none later (a key removed) waits for the person (`needs_human`)
   instead of being worked by an unqualified model.
4. **Falling back to the cluster.** A master's attempts follow the task's retry strategies:
   feedback, rephrase, the other master model (another provider, if one is qualified and
   ready), and then **decompose**: the parts of a split master subtask are ordinary subtasks
   (`master` off) worked by the cluster's models, and the master's own failure becomes
   ordinary DESMAN work instead of a dead end. A subtask that must run code (`--runs-code`)
   is never split, so with it a failing master waits for the person instead. A continuation
   of a master task (`task extend`, `task redo`, M.2) is one master subtask again (the
   planner model is not asked), when a qualified model is ready; otherwise it is planned for
   the cluster and the person is told so.
5. **The master uses the cluster.** Long documents (ADR 0014) are still read in parts by the
   cluster: the master subtask depends on their digests like any planned subtask. M.2 lets the
   master delegate on its own, built on the split (`Store.decompose`) rather than on a
   continuation: a reply `{"type": "delegate", "note": ..., "parts": [...]}` (2 to
   `MAX_DELEGATED_PARTS` parts, each a description, acceptance criteria, an optional kind and
   `depends_on` among the parts) ends the master's execution without a review.
   `Store.delegate` adds the parts as ordinary subtasks (`master` off, `parent_id` the master
   subtask, its inputs, dependencies and knowledge, audited `delegated` with the note), and
   the master subtask goes back to PENDING, depending on them: the existing RUNNING → PENDING
   edge of a split, so the state machine does not change. Once they are done the master runs
   again as a master (not a combine), with their results as dependency results and its notes
   of earlier delegations in its context; its earlier tool results stay readable (L.3).
   A delegation is not a failed attempt. At most `MAX_DELEGATIONS` per master subtask; after a
   delegation the subtask is no longer split (its parts are in place), so a failing master
   waits for the person. Parts are reviewed by the cluster's reviewer like any subtask, and
   cannot be split themselves (one level, as a split's parts). Nothing waits inside a lease,
   and every step stays resumable.
6. **Never for desman-global.** A job lent to the network is never a master subtask (refused
   when the task is created): what a lent module may carry is set by its quota, not this.

## Consequences

- A goal for one capable agent costs one long attempt and one review instead of a plan, up to
  eight subtasks and their reviews; on a free strong model that is fewer calls, not more.
- Each request of a long loop is larger. The folding of old tool results (L.3) keeps it in
  bounds; the master's lineup is limited to models whose context and per-request limits allow
  it.
- The catalog's `master` mark is a judgement made in code review; free lineups change without
  notice, so M.3's benchmark gate is what makes "qualified" measured instead of believed.
- A delegation costs no review and no attempt; its parts are cluster work with their own
  reviews, so `desman stats` shows them under the cluster's models.
- Stats (`desman stats`) already count calls by role and model; a task with a master subtask
  records it in `task_created` (`master`: its ids, `plan`: `master`), so the outcomes can be
  compared with the cluster's: `desman stats` shows tasks with a master subtask beside the
  others (done, escalated, calls and tokens per task; M.3).
