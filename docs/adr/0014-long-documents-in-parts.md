# ADR 0014: Long documents in parts

Status: accepted (2026-11-19, ROADMAP L.2)

## Context

A trial on 2026-11-19 (owner's free keys) asked for an analysis of Chekhov's «Из Сибири» by
link: 66 000 characters. The executor reads at most `MAX_TOOL_ROUNDS` (8) tool results of
`MAX_TEXT_CHARS` (8 000) per attempt; retries read the page from the start again, the request
grew to ~27 000 tokens and free models timed out, 5 failures out of 5. The planner made
"read everything → combine → analyse": it does not divide a text, and a model asked to read a
long text through a narrow window loses it.

## Decision

**Code, not a model, divides a long text and builds the graph that reads it.**

1. **Which texts.** A text file given to the task (L.1, `input/`) and a page whose address is
   in the goal, when either is longer than `documents.LONG_CHARS`. A page is fetched by code
   before planning (the same `Web` as `web_fetch`: public addresses only, GET, its cache) only
   when the run's executor may use `web_fetch` (tool policy ∩ the person's permissions); a long
   one is saved whole as a task file in `input/`, a short one or a failure is left to the
   executor as before (recorded in the task's audit details). At most `documents.MAX_URLS`
   addresses. `web_fetch` during work also saves a long page whole to `web/` in the task's
   workspace and says where, so a later read or a retry (L.3) needs no new fetch.
2. **How it is divided.** `documents.split()`: by headings first (Markdown `#`, chapter lines
   such as `Глава 3`, `Chapter III`, a lone Roman numeral, a short line in capitals between
   blank lines), a section longer than `PART_CHARS` (≈ 6 000) by paragraphs, then sentences,
   then a hard cut. The parts are exact slices: joined they give the text back. Very long texts
   get larger parts so one document has at most `MAX_PARTS` of them.
3. **The graph.** For each long document: a subtask per part (kind `analysis`) with the part's
   text in its `inputs` (marked untrusted), which notes what the part says on the goal, with
   exact short quotes; then combine subtasks over groups of at most `GROUP` (8) results, level
   after level, until one digest remains. The planner's subtasks that depend on nothing depend
   on every digest, so the notes reach them as results of dependencies. The planner is told in
   the goal (`goal_note`) that the document is read in parts by code and is not to be read
   whole again. Parts are made by code, so `MAX_SUBTASKS` (the planner's limit) does not count
   them.
4. **Checking.** The reviewer of a part sees the part's text (its inputs are part of the
   evidence) and checks every quote against it; a combine step checks quotes against the notes
   it was given.

## Consequences

- A long document costs one call per part plus its review, and the combines; on free keys that
  is slower but each request is small (one part, ≈ 2 000 tokens), which free models answer.
- A page found during work (by search) is still read by offset by the executor; splitting it
  mid-task is not done here (a later task if needed).
- Translating a long text is not served by notes; it is a different shape of work (later).
