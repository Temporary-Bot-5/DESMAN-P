# ADR 0019: The reviewer's own tests decide whether code is accepted

Status: accepted (Claude, 2026-11-27, ROADMAP L.5; the owner reviews it with PR #189)

## Context

A live probe (2026-11-19) accepted a `calc.py` whose fifteen tests, written by the executor,
all passed, although four plain inputs broke it: a trailing space, `2^-1`, `(-8)^0.5` (a complex
result) and `10^1000` (an overflow traceback). The reviewer accepted on the executor's tests and
its own reading. L.5 asks that code is accepted only on the result of a run, not on a model's
word.

Until now generated code ran only as the executor's `run_command` tool call: journaled
(`core/effects.py`), in the task's workspace, under the role's policy and the person's
permissions. The reviewer has no tools.

## Decision

1. **The reviewer writes tests, code runs them.** For a subtask that must run code
   (`runs_code`) or a code subtask that wrote Python files, the reviewer's accept must carry a
   pytest module (validated: parses, has collectable tests, ≤20 000 chars). The coordinator, not
   the model, runs it and decides.
2. **Same sandbox, a throwaway copy.** The run goes through the `Sandbox` the `run_command` tool
   uses, with the same limits, only while the task grants tools and the person's
   `shell_execution` is on. It runs in a copy of the workspace made by `runs.py` under the
   workspaces root (files written through `Workspace`), so the tests cannot change what the
   person gets; the copy is removed after the run and stale copies are swept. pytest is started
   isolated (`-I -B`, its own ini, no `conftest.py`, no `addopts`), and results come from a
   JUnit report cross-checked with pytest's summary.
3. **Not a tool call.** The run is code's own check, recorded as a `review_tests` note
   (counts, the reviewer's module, a bounded output tail, the attempt), not in the call journal:
   nothing is repeated on a rerun, and an interrupted review simply runs it again. This is the one
   exception to "generated code runs only through the `run_command` tool"; AGENTS.md says so.
4. **One decision table.** A verified pass → accept. A test that failed while running, a stop or
   a full workspace → back to the executor with the failing tests. Anything ambiguous (collection
   or setup errors in the reviewer's module, zero tests, no report, the whole run timed out) →
   back to the reviewer, who fixes its test or itself names the executor's defect. An import
   error in the executor's own file → the executor's, except a third-party package the sandbox
   lacks → not run. Not runnable (no permission, no sandbox, a package missing, a copy over the
   quota) → the model's verdict stands with the reason recorded, except for `runs_code`, which
   then goes to the person; sandbox outages pause it (ten pauses → the person).

## Consequences

- A correct program can still be sent back by a wrong reviewer test; the prompt limits tests to
  stated behaviour, and the next reviewer sees its previous tests and their output.
- Code under test shares the process and folder with pytest, so deliberate tampering with the
  report is made harder, not impossible.
- Python only: other languages are tested only when they must run (`runs_code`), by starting
  the program from a test.
- Packages beyond the standard library arrive with L.6.
