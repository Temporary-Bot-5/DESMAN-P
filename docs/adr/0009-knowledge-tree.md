# ADR 0009: The knowledge tree — context chosen by its table of contents

Status: accepted (owner's idea and answers, 2026-11-09)

## Context

The owner looked at what a model receives (`/context`) and found it far too large for small
models. Measured on 2026-11-09, the interlocutor's system prompt is about 4.7k tokens before a
single message of the conversation:

| Part | Tokens |
|---|---|
| The catalog of every command, with argument schemas (`commands.for_model()`) | ~3,100 |
| The interlocutor's own instructions | ~800 |
| The manual's list of topics | ~340 |
| Honesty, behaviour, profile rules, lookup rule | ~530 |

Two-thirds of it is the command catalog, and most turns use no command at all. The manual
(`manual.py`, ~5.8k tokens) already works the other way round: the prompt lists only topic names,
and a `lookup` turn asks for at most two topics, whose texts come back in a second call (C.10).
That is a one-level version of what the owner proposes.

The owner's proposal: keep what DESMAN knows as a tree of sections, subsections and files —
for example "Projects / X / Information" and "Personal / Style preferences". For a request like
"make a site about our project X in the style I like", a model sees the tree, picks the two
files that look useful, and only those become the context of the modules that do the work. The
tree also holds DESMAN's whole documentation, so it can explain its own design in detail.

What the tree must not break:
- **Free tiers.** Every selection is a model call: latency and requests per minute. Walking a
  deep tree level by level costs a call per level.
- **Misses are silent.** A file the model did not pick makes the work worse without any error.
- **Personal data.** "Projects" and "Preferences" are about the person: they must stay visible,
  editable and erasable by the person, and exist only with `personalize` on (like the profile,
  C.7).
- **Untrusted text.** What tasks produced or web pages said is data, never instructions, also
  when it later sits in the tree.
- **Present and plans.** The manual describes only what DESMAN does now; PROJECT_PLAN, ROADMAP
  and ADRs describe intentions. A model must not present a plan as a feature.

## Decision

1. **A node** has a path (`Commands/Keys`, `Personal/Style`, `DESMAN/Docs/ADR 0004`), a one-line
   description (≤100 characters, written for choosing), a text (bounded, ~1.5k tokens; longer
   documents are split by their headings), and an origin:
   - `release`: comes with DESMAN, built by code from the repository at runtime (manual topics,
     command groups with their argument schemas, `docs/`, ADRs, PROJECT_PLAN). It is never stored
     in the database and never edited by anyone but people through git. Docs that describe
     intentions (PROJECT_PLAN, ROADMAP, ADRs) are marked `plan`.
   - `person`: written or confirmed by the person (see 6). It is stored in the database.
2. **One table of contents, one choice.** A model is shown the table of contents (paths and
   descriptions, never texts) and names the nodes it needs. While the table of contents fits its
   budget (~800 tokens), it is shown whole and no level-by-level walking happens. When it outgrows
   that, only the top sections are shown with their sizes, and the model may open one section,
   which costs one more call, never more than two levels.
3. **The interlocutor.** Its prompt carries the table of contents instead of the topic list
   and the full catalog. Of the catalog's ~3.1k tokens, the descriptions are ~2.1k, the argument
   schemas ~0.7k and the names ~0.2k (70 commands). The prompt keeps the names with their
   argument schemas, so command turns still get their arguments right without an extra call;
   the descriptions move into the `Commands/GROUP` nodes (K.2, measured: the system prompt went
   from ~4.6k to ~2.8k tokens). A lookup opens up to 3 paths; opening a section (`…/`) shows
   its table of contents and allows one more lookup, never more. Like today's `lookup`, it
   costs one extra call, only when needed. `/context` names the paths the last answer read.
4. **Tasks** pay no extra call: the planner already makes one. It sees the table of contents and
   names the nodes each subtask needs (`context: [paths]` in `PlannedSubtask`). The executor
   gets their texts in `core/context.py`, within the existing token budget and marked by origin.
   During its work, an executor that finds it lacks something calls a new tool `read_node(path)`
   (read permission, repeatable, journaled like every tool), so a miss can be repaired by the
   worker itself.
5. **Misses are visible.** What was chosen is recorded: for tasks an audit entry
   (`context_chosen`, paths only) shown by `status`; for the chat, the nodes of each turn shown by
   `/context`. A later Improvement Analyst run can read them like other failures.
6. **Personal nodes** exist only with `personalize` on. The person makes them (a command and the
   page). An agent — the interlocutor, or a task from its results — may propose a change to the
   tree at any moment, but it is applied only after the person confirms it in a window rendered
   by code (owner, 2026-11-09). Each change goes to `operator_log` by node id, never the text.
   Nodes are changed and erased in place (personal data, not an append-only log). A temporary
   chat never writes one. A node's text that came from a task result keeps the `untrusted` mark.
   The tree belongs to the person's own DESMAN. desman-global's jobs are worked by another crew
   (`runs.network_crew`), an agent unrelated to this one apart from sharing the host's
   resources, so it never reads the tree.
7. **DESMAN's own documentation.** The `DESMAN` section holds the manual ("how it works now")
   and the documents marked `plan`. The interlocutor's rule: a `plan` node is told as a plan
   ("planned", "proposed in ADR 0006"), never as something DESMAN does.

## Consequences

- The interlocutor's system prompt dropped from ~4.6k to ~2.8k tokens (K.2). Small local
  models (the 4k context of llama.cpp, slow CPU Ollama, noted in T.2l) get room for the
  conversation.
- The chat makes a second call in the same cases as before (a lookup), and a third only after
  opening a section. Command turns are unchanged: the schemas stay in the prompt. The benchmark
  (`bench.py`) runs tasks, not the chat, so K.2 is measured by `/context`; K.3 (tasks) is
  measured on the benchmark.
- Tasks gain the person's notes and DESMAN's docs without a new call; the planner's prompt
  grows by the table of contents (bounded by 2).
- One new tool (`read_node`) and one new table (personal nodes, a new migration). The
  `release` part needs no storage and cannot drift from the code: a test builds the tree and
  checks every command and every manual topic is in it.
- The profile (C.7) kept its own table in the first steps. K.5 moved its notes into one node,
  `Personal/Profile` (a note per line; migration 47 moves the old notes and drops the table).
  The interlocutor still gets the notes with every message, so its table of contents leaves
  the node out; the planner may choose it like any personal node. The interlocutor edits the
  notes as before, without a window; the node itself only the person writes, moves or removes,
  and only as notes (answer 3).

## Split (each one PR, measured with `/context`)

- **K.1** The tree from the repository (`release` nodes: manual, command groups, docs split by
  headings, `plan` marks), the table of contents with its budget, a test that nothing is
  missing.
- **K.2** The interlocutor on the tree: the table of contents in its prompt, short catalog,
  "open nodes" in place of `lookup`, chosen nodes in `/context`. Benchmark before and after.
- **K.3** Tasks: `context` paths in the plan, node texts in the executor's context,
  `read_node` tool, `context_chosen` audit.
- **K.4** Personal nodes: table and migration, the command and the page section, interlocutor
  proposals, `operator_log`, erasure; only with `personalize`.
- **K.5** The profile's notes moved into `Personal/` (upgrade of existing notes).

## Owner's answers (2026-11-09)

1. **Order:** before the test with acquaintances: K.1 and K.2 come before T.4.
2. **Who writes:** an agent may change the tree at any moment, but only with the person's
   confirmation (decision 6).
3. **desman-global:** the cluster's agents and the local agent are unrelated; the tree is the
   local agent's only (decision 6).
3. **The profile (2026-11-10, K.5):** its notes become one node, `Personal/Profile`, always
   shown to the interlocutor. Its notes are edited by the interlocutor without a window, as
   before (an exception to decision 6, for this node only); every other node keeps the window.
