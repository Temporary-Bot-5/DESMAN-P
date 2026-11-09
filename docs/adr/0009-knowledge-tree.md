# ADR 0009: The knowledge tree — context chosen by its table of contents

Status: proposed (owner's idea, 2026-11-09; open questions at the end)

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
   and the full catalog. The catalog shrinks to command names with their one-line `about`. The
   argument schemas move into the `Commands/…` nodes. The existing `lookup` turn becomes "open
   nodes" (≤3) and costs the same one extra call as today, only when needed. A command turn with
   wrong arguments is still validated by code and repaired once with the schema shown, so a
   command whose schema was not looked up does not slip through.
4. **Tasks** pay no extra call: the planner already makes one. It sees the table of contents and
   names the nodes each subtask needs (`context: [paths]` in `PlannedSubtask`). The executor
   gets their texts in `core/context.py`, within the existing token budget and marked by origin.
   During its work, an executor that finds it lacks something calls a new tool `read_node(path)`
   (read permission, repeatable, journaled like every tool), so a miss can be repaired by the
   worker itself.
5. **Misses are visible.** What was chosen is recorded: for tasks an audit entry
   (`context_chosen`, paths only) shown by `status`; for the chat, the nodes of each turn shown by
   `/context`. A later Improvement Analyst run can read them like other failures.
6. **Personal nodes** exist only with `personalize` on. They are made by the person (a command
   and the page), or proposed by the interlocutor the way profile edits are today (C.7):
   applied, shown in the chat, each change in `operator_log` by node id, never the text. They are
   changed and erased in place (personal data, not an append-only log). A temporary chat never
   writes one. A node's text that came from a task result keeps the `untrusted` mark.
7. **DESMAN's own documentation.** The `DESMAN` section holds the manual ("how it works now")
   and the documents marked `plan`. The interlocutor's rule: a `plan` node is told as a plan
   ("planned", "proposed in ADR 0006"), never as something DESMAN does.

## Consequences

- The interlocutor's prompt drops from ~4.7k to roughly 2k tokens (to be measured with
  `/context` at each step). Small local models (the 4k context of llama.cpp, slow CPU Ollama,
  noted in T.2l) get room for the conversation.
- The chat makes a second call in the same cases as today's `lookup`. When the needed command
  schema was not opened, it may now also need a repair call. Measured on the benchmark before and
  after; if command turns get noticeably worse, the most used commands keep their schemas in the
  prompt.
- Tasks gain the person's notes and DESMAN's docs without a new call; the planner's prompt
  grows by the table of contents (bounded by 2).
- One new tool (`read_node`) and one new table (personal nodes, a new migration). The
  `release` part needs no storage and cannot drift from the code: a test builds the tree and
  checks every command and every manual topic is in it.
- The profile (C.7) stays as it is in the first steps. Moving its notes into `Personal/` is a
  later step with its own upgrade.

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

## Open questions for the owner

1. **Order.** Phase T (testing with acquaintances, T.4–T.7) is next in ROADMAP. Do K.1–K.2,
   which only shrink the prompt and help every tester, come before T.4, or after the test?
2. **Who writes "Projects".** Proposed: only the person, or the interlocutor with the edit shown
   in the chat, as profile edits are now. Should tasks be able to propose nodes from their
   results too (always marked untrusted, shown to the person first)?
3. **desman-global.** Jobs lent to other nodes never see personal nodes (proposed); `release`
   nodes are the same everywhere. Is that right?
