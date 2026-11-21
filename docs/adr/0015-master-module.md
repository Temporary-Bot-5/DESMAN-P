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
   in its instructions (plan your own steps, check the work before handing it in). Everything else is the
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
   master's model. M.3 adds a measured gate: the benchmark (`bench.py`) run by the model.
3. **Who chooses the mode.** The person: `desman run --master` (M.1), in the chat (M.2). Never
   automatic in M.1: a silent switch would leave the cluster unused and hide what a task costs.
   Without a ready qualified model `--master` says so and creates nothing; a master subtask
   that finds none later (a key removed) waits for the person (`needs_human`) instead of
   being worked by an unqualified model.
4. **Falling back to the cluster.** A master's attempts follow the task's retry strategies:
   feedback, rephrase, the other master model (another provider, if one is qualified and
   ready), and then **decompose**: the parts of a split master subtask are ordinary subtasks
   (`master` off) worked by the cluster's models, and the master's own failure becomes
   ordinary DESMAN work instead of a dead end.
5. **The master uses the cluster.** Long documents (ADR 0014) are still read in parts by the
   cluster: the master subtask depends on their digests like any planned subtask. M.2 lets the
   master delegate on its own: a result `{"type": "delegate", ...}` with subtasks for the
   cluster ends the master's subtask; code continues the task (ADR 0010, `Store.extend`) with
   those subtasks and a master subtask after them that gets their results. Nothing waits
   inside a lease, and every step stays resumable.
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
- Stats (`desman stats`) already count calls by role and model; a task with a master subtask
  records it in `task_created` (`master`: its ids, `plan`: `master`), so the outcomes can be
  compared with the cluster's (M.3).
