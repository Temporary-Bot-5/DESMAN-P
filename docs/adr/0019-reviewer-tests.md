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
   person gets; the copy is removed after the run and stale copies are swept. Each copy limit is the
   greater of the sandbox quota and the initial workspace usage, plus room for reviewer
   files. pytest is started isolated (`-I -B`, its own ini, no `conftest.py`, no `addopts`), and results come from a
   JUnit report cross-checked with pytest's summary.
3. **Not a tool call.** The run is code's own check, recorded as a `review_tests` note
   (counts, the reviewer's module, a bounded output tail, the attempt), not in the call journal:
   the note and the verdict or failure commit together. An interrupted review runs it again. This
   is the one exception to "generated code runs only through the `run_command` tool"; AGENTS.md says so.
4. **One decision table.**
   - A. A verified pass → accept.
   - B. A test failure or an import error in the executor's own file → back to the executor.
     A timed-out test also goes back to the executor when the innermost workspace frame in its
     last faulthandler dump is in a copied executor file. The ini requests a stack dump after 50 s of one test (the whole run has 60 s), so a dump
     belongs to the test still running when the run was stopped; a hang that starts later goes
     to the reviewer (C). A run the produced code kills (a segfault or abort: pytest's
     faulthandler prints the stack, no report is written) goes to the executor the same way.
   - C. Ambiguous results (reviewer collection or setup errors, zero tests, missing reports,
     other timeouts, stopped runs and full workspaces) → back to the reviewer, who fixes its
     test or names the executor's defect. The note keeps the reason.
   - D. Tests that cannot run (permissions, sandbox unavailable, missing package, copy over
     quota) → the model's verdict stands with the reason recorded. Required runs (`runs_code`)
     go to the person. A missing import counts as a package only when its top-level name is
     on the curated common PyPI import list.
   - Low disk space pauses every qualifying review, including when copying fails. Reviews
     skip the model call while free space remains below the recorded threshold. An unavailable
     sandbox also pauses required reviews. After ten consecutive outage pauses, a continuing
     outage goes to the person. A non-outage test note or any verdict resets the count for
     that execution attempt.

## Consequences

- A correct program can still be sent back by a wrong reviewer test; the prompt limits tests to
  stated behaviour, and the next reviewer sees its previous tests and their output.
- A dump at 50 s can still belong to a test that then finished, if the next one hangs in
  the last seconds; the blame is then wrong once and the next review sees why.
- The copy is read from the live workspace without a lock: a sibling subtask running at the
  same time may be changing a file being copied (as it already could during the executor's
  own runs).
- Code under test shares the process and folder with pytest, so deliberate tampering with the
  report is made harder, not impossible.
- Under the WebAssembly sandbox (WASI) faulthandler cannot run (no threads), so a run that hits
  the time limit there always goes back to the reviewer, never to the executor; a standard
  module the sandbox lacks (e.g. `_sqlite3` under WASI) means the tests cannot run (row D).
- Python only: other languages are tested only when they must run (`runs_code`), by starting
  the program from a test.
- Packages beyond the standard library arrive with L.6.
