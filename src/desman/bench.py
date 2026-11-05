"""The benchmark a proposal is evaluated on (ROADMAP 5.5): a fixed set of goals, each planned
by the planner and worked to its end by the executor and the reviewer, on a database of its own.

It is run as `python -m desman.bench` inside a git worktree (`desman.evaluation`), once with the
current version of DESMAN and once with the proposed one, so the code that runs is the code
being judged; this module only reports what happened (`Result`), and `compare` says which run
went better. Both are part of how a proposal is judged, so they are protected files: the analyst
can never propose a change to them.

Two modes. With the scripted fake model (`fake`) nothing leaves the machine and every run is the
same: it shows that the proposed version still works from plan to review, and what its prompts
cost in tokens (the fake answers the format its prompt asks for, and nothing else when the
format is not there). With real models (`real`, the person's config) it shows whether the work
goes better; one run of a real model is noisy, which the person is told.

The goals have no tools (`Tools.NONE`): the benchmark is about prompts and workflow, and runs
nothing on the machine and nothing on the web.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

import desman
from desman import runs
from desman.config import Config, ConfigError, Tuning, open_config
from desman.core import calls, db
from desman.core.models import Kind, Tools
from desman.core.stats import collect
from desman.core.store import Store
from desman.llm import router
from desman.llm.base import LLM, Completion, Request
from desman.llm.fake import FakeLLM, garbage
from desman.llm.openai_compat import CallRecord
from desman.planner import PLAN, SPLIT, plan

VERSION = 1
"""Of the benchmark set and its result; results of different versions are not compared."""

Mode = Literal["fake", "real"]


@dataclass(frozen=True)
class Case:
    id: str
    goal: str
    kind: Kind = Kind.TEXT
    """What its work is: the fake tags its plan's subtasks so, which picks the executor's
    model as a real plan would."""


CASES: tuple[Case, ...] = (
    Case(
        "explain",
        "Explain in three short paragraphs, for a ten-year-old, why the sky is blue and why "
        "sunsets are red.",
    ),
    Case(
        "code",
        "Write a Python function is_palindrome(text) that ignores case, spaces and punctuation, "
        "with three example calls and the result of each.",
        Kind.CODE,
    ),
    Case(
        "compare",
        "Compare an array and a linked list for inserting into the middle and for reading the "
        "n-th element, and say which to choose for each of the two.",
        Kind.ANALYSIS,
    ),
    Case(
        "steps",
        "Make a packing list for a three-day autumn hike, grouped by category, and then a "
        "one-paragraph summary of the five most important items and why.",
    ),
)


class CaseResult(BaseModel):
    id: str
    status: str
    """The task's `TaskStatus` when the run stopped."""
    plan: str
    """"llm", or "template" when the model's plan failed."""
    plan_error: str | None = None
    subtasks: int


class Result(BaseModel):
    """What one run of the benchmark reports (stdout of `python -m desman.bench`)."""

    version: int
    mode: Mode
    source: str
    """Where the package that ran is: the worktree's, checked by the caller."""
    cases: list[CaseResult]
    limits: dict[str, int] | None = None
    """Effective workflow limits without model choices; absent in older results."""
    stats: dict[str, Any]
    """`desman stats` of the run's database (`core.stats.Stats.as_json`)."""


# ── Running ──────────────────────────────────────────────────────────────────


def run(
    store: Store,
    mode: Mode,
    tuning: Tuning,
    *,
    config: Config | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Result:
    """Every case, one after another, on `store` (a database of its own)."""
    if mode == "fake":
        models = runs.Models(Recorded(fake(), store.conn), None)
    else:
        assert config is not None
        # The models each role asks first are part of the tuning being evaluated.
        tuned = config.model_copy(update={"tuning": tuning})
        models = runs.models(tuned, store.conn, provider=provider, model=model, cache=False)
    coordinator = runs.summarizer(store, models, tuning)
    cases = []
    for case in CASES:
        with calls.for_(calls.Purpose(calls.Role.PLANNER)):
            planned = plan(
                models.for_(calls.Role.PLANNER),
                case.goal,
                limit=runs.tuned(tuning, "max_subtasks"),
            )
        spec = runs.tuned_spec(planned.spec, tuning).model_copy(update={"tools": Tools.NONE})
        task_id = store.create_task(spec, details=planned.details() | {"benchmark": case.id})
        # A pause (a provider's limit) ends the run: the task is left as it is, and counted.
        result = coordinator.run(task_id)
        cases.append(
            CaseResult(
                id=case.id,
                status=result.status.value,
                plan=planned.source,
                plan_error=planned.error,
                subtasks=len(store.list_subtasks(task_id)),
            )
        )
    return Result(
        version=VERSION,
        mode=mode,
        source=str(Path(desman.__file__).resolve().parent),
        cases=cases,
        limits={name: runs.tuned(tuning, name) for name in runs.TUNING_DEFAULTS},
        stats=collect(store.conn).as_json(),
    )


_PLAN_FORMAT = ('"subtasks"', '"description"', '"acceptance_criteria"', '"depends_on"')
"""What a planner's prompt asks the reply to hold; the fake plans only when it is all there."""


_LIMIT = re.compile(r"Never more than (\d+) subtasks")
_KINDS = {case.goal: case.kind for case in CASES}


def fake() -> FakeLLM:
    """A model that does what the prompt asks, the same way every time: a plan of two steps
    (one, when the prompt allows only one; tagged with the case's kind; only when the planner's
    prompt asks for the plan's format), an answer, an acceptance."""

    def planned(request: Request) -> str:
        system = next((m.content for m in request.messages if m.role == "system"), "")
        if not all(word in system for word in _PLAN_FORMAT):
            return garbage("prose").text
        goal = request.prompt.removeprefix(f"{PLAN}:\n")
        kind = _KINDS.get(goal, Kind.TEXT).value
        limit = _LIMIT.search(system)
        final = {
            "description": f"From the material, do this: {goal}",
            "acceptance_criteria": "The goal is fully achieved.",
            "kind": kind,
            "depends_on": [0],
        }
        if limit is not None and int(limit.group(1)) < 2:
            return json.dumps({"subtasks": [final | {"depends_on": []}]})
        gather = {
            "description": f"Gather what is needed for: {goal}",
            "acceptance_criteria": "The material needed is listed.",
            "kind": kind,
            "depends_on": [],
        }
        return json.dumps({"subtasks": [gather, final]})

    return (
        FakeLLM(name="fake")
        .on(PLAN, planned, repeat_last=True)
        .on(SPLIT, garbage("prose"), repeat_last=True)
        .on("Review", '{"verdict": "accept"}', repeat_last=True)
        .on("Execute", '{"answer": "done as asked"}', repeat_last=True)
    )


class Recorded:
    """A fake whose calls go into `model_calls` like a real provider's, so `desman stats`
    counts them (calls and tokens by role)."""

    def __init__(self, llm: LLM, conn: Any) -> None:
        self.llm = llm
        self.name = llm.name
        self._record: Callable[[CallRecord], None] = calls.recorder(conn)

    def complete(self, request: Request) -> Completion:
        reply = self.llm.complete(request)
        self._record(
            CallRecord(
                provider=reply.provider,
                model=reply.model,
                key_id=None,
                prompt_hash=_hash(request.model_dump_json()),
                response_hash=_hash(reply.text),
                prompt_tokens=reply.usage.prompt_tokens,
                completion_tokens=reply.usage.completion_tokens,
                latency_ms=0,
                http_status=200,
                error_kind=None,
                error=None,
            )
        )
        return reply


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Comparing ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Metric:
    id: str
    better: Literal[1, -1]
    """1: more is better; -1: less is better."""
    decides: bool = True
    """Whether it decides the verdict (in this order); tokens only decide a tie."""


METRICS: tuple[Metric, ...] = (
    Metric("tasks_done", 1),
    Metric("escalated", -1),
    Metric("plan_fallbacks", -1),
    Metric("rejected", -1),
    Metric("attempts", -1),
    Metric("first_try", 1, decides=False),
    Metric("subtasks", -1, decides=False),
    Metric("calls", -1, decides=False),
    Metric("failed_calls", -1, decides=False),
    Metric("tokens", -1, decides=False),
)
TOKENS_MARGIN = 0.05
"""A difference in tokens smaller than this share is no difference."""

Verdict = Literal["better", "worse", "same"]


def metrics(result: Result) -> dict[str, int]:
    """The numbers runs are compared by."""
    stats = result.stats
    subtasks = stats.get("subtasks", {})
    roles = stats.get("roles", [])
    return {
        "tasks": len(result.cases),
        "tasks_done": sum(c.status == "done" for c in result.cases),
        "escalated": int(subtasks.get("escalated", 0)),
        "plan_fallbacks": sum(c.plan != "llm" for c in result.cases),
        "rejected": int(stats.get("reviews", {}).get("retry", 0)),
        "attempts": sum(int(s.get("started", 0)) for s in stats.get("strategies", [])),
        "first_try": int(subtasks.get("first_try", 0)),
        "subtasks": int(subtasks.get("total", 0)),
        "calls": sum(int(r.get("calls", 0)) for r in roles),
        "failed_calls": sum(int(r.get("failed", 0)) for r in roles),
        "tokens": sum(
            int(r.get("prompt_tokens", 0)) + int(r.get("completion_tokens", 0)) for r in roles
        ),
    }


def compare(base: Result, candidate: Result) -> Verdict:
    """Whether the candidate's run went better than the base's: the first deciding metric that
    differs says so; when none does, tokens do (beyond `TOKENS_MARGIN`)."""
    old, new = metrics(base), metrics(candidate)
    for metric in METRICS:
        if metric.decides and old[metric.id] != new[metric.id]:
            ahead = (new[metric.id] - old[metric.id]) * metric.better > 0
            return "better" if ahead else "worse"
    if abs(new["tokens"] - old["tokens"]) > TOKENS_MARGIN * max(old["tokens"], 1):
        return "better" if new["tokens"] < old["tokens"] else "worse"
    return "same"


# ── The process ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """`python -m desman.bench --mode fake|real [--tuning JSON] [--provider P] [--model M]`:
    runs the benchmark on a new database in a temporary folder and prints its `Result` as JSON.
    Real mode reads the config (`DESMAN_CONFIG_DIR`), and writes nothing to it.

    Not a command for people: `desman.evaluation` runs it in a worktree. What it says on
    stderr when it cannot run becomes the reason an evaluation failed, which the database
    keeps as it is, like every stored reason (untranslated)."""
    from desman.stdio import configure_output

    configure_output()
    parser = argparse.ArgumentParser(prog="desman.bench")
    parser.add_argument("--mode", choices=["fake", "real"], required=True)
    parser.add_argument("--tuning", default="{}", help="The tuning to run with (JSON).")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        tuning = Tuning.model_validate_json(args.tuning)
    except ValidationError as e:
        print(f"bad --tuning: {e}", file=sys.stderr)
        return 2
    config = None
    if args.mode == "real":
        try:
            config = open_config().load()
        except ConfigError as e:
            print(f"cannot read the config: {e}", file=sys.stderr)
            return 2
    with tempfile.TemporaryDirectory(prefix="desman-bench-") as folder:
        store = Store(db.open_db(Path(folder) / "desman.db"))
        try:
            result = run(
                store, args.mode, tuning, config=config, provider=args.provider, model=args.model
            )
        except router.NoProvider as e:
            print(str(e), file=sys.stderr)
            return 2
        finally:
            store.conn.close()
    print(result.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
