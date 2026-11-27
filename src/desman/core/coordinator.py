"""Coordinator: drives one task until the system has nothing left to do on its own, or works
on every task as a worker (`desman worker`, ROADMAP 4.1).

The loop per tick:
    1. housekeeping: requeue expired leases, end pauses that are over
    2. review whatever waits for a verdict (no re-execution after a reviewer crash), each review
       reserved first so that another worker does not pay for the same verdict
    3. claim the most urgent ready subtask, execute it, submit the result or record the failure

While it executes work below CRITICAL, every step of the executor (a heartbeat, a tool call)
looks for critical work waiting in the coordinator's scope. If there is some, the coordinator
takes it and gives its own subtask back (`Store.preempt`, ROADMAP 4.4): the executor is stopped,
the critical subtask runs at once, and the dropped one is claimed again later, resuming from its
checkpoint with its tool calls replayed from the journal.

Any number of coordinators and workers share one database: claims are atomic, status changes
compare-and-swap, and each task's retry policy is stored with it (no caller brings its own).
`run()` stops as soon as the task is no longer ACTIVE (done, failed, cancelled or waiting for the
operator). Termination is guaranteed: every quality or transient failure spends a bounded budget
and ends in NEEDS_HUMAN, and blocked dependents are settled immediately. The one open-ended wait
is a provider pause; by default the coordinator returns instead of sleeping through it.

Jobs lent by desman-global (tasks of origin `network`, ROADMAP G.6b) are worked by a crew of
their own (`Crew`: the models lent, the `network_*` roles of the tool policy, no splitting); a
coordinator without one never takes them. The person's own work goes first and takes over from
a job at the executor's next step, as critical work does from the rest.

Executor and reviewer are plain callables, so the loop is tested with scripted fakes and later
driven by real LLM-backed agents. An executor acts on nothing itself: it asks for tool calls
through `Work.tools` (core/tools.py), which checks, guards, audits and journals each one, so
that a rerun of an interrupted execution does not repeat them (core/effects.py).
"""

from __future__ import annotations

import ast
import functools
import json
import math
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from desman.core import calls, db, webguard
from desman.core.context import (
    DEFAULT_BUDGET_TOKENS,
    Context,
    Knowledge,
    KnowledgeText,
    build_context,
    given,
)
from desman.core.effects import (
    EffectRef,
    Effects,
    earlier_lookups,
    earlier_results,
    has_written_python_files,
    workspace_files,
)
from desman.core.models import (
    MAX_DELEGATED_PARTS,
    MAX_DELEGATIONS,
    Kind,
    Origin,
    Priority,
    Subtask,
    SubtaskSpec,
    Tools,
)
from desman.core.policy import Access, executor_access
from desman.core.retry import FailureKind, Strategies, Strategy, strategy_for
from desman.core.sandbox import SandboxUnavailable
from desman.core.states import Actor, SubtaskStatus, TaskStatus
from desman.core.store import (
    DEFAULT_LEASE_SECONDS,
    DEFAULT_REVIEW_SECONDS,
    PreemptedError,
    StaleStateError,
    Store,
)
from desman.core.tools import RUN_TOOL, Registry, Tool, Toolbox, ToolResult, builtin_registry
from desman.core.workspace import PathEscapeError, Workspace, workspace_root_for
from desman.llm.base import LLM, ProviderError, among, guard_model_calls, noticing_unfit

S = SubtaskStatus
MODEL_CALL_MARGIN = 30
REVIEW_TEST_TIMEOUT = 60

DEFAULT_POLL_SECONDS = 5.0
"""How often a coordinator with nothing to do looks again while others hold work (a worker's
lease is renewed by its heartbeats, so waiting for it to run out would wait far too long)."""


class Success(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: dict[str, Any]
    """Stored as the artifact: must be strict JSON (no NaN/Infinity, only JSON types)."""

    @field_validator("content")
    @classmethod
    def _strict_json(cls, content: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(content, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"result is not storable as JSON: {e}") from e
        return content


class Failure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: FailureKind
    reason: str
    paused_until: str | None = None
    """For LIMIT: when the provider's quota resets. Any timezone-aware ISO-8601 timestamp."""
    unfit_provider: str | None = None
    """The model that cannot answer this kind of request (`ProviderError.unfit_model`), as
    `provider/model` (the provider alone when the error does not say): the task's next attempts
    go to another model, if there is one."""

    @field_validator("paused_until", mode="before")
    @classmethod
    def _normalize_deadline(cls, value: object) -> object:
        # Provider data: reject garbage here rather than crash the loop later.
        if isinstance(value, datetime):
            return db.to_iso(value)
        if isinstance(value, str):
            return db.normalize_iso(value)
        return value


class Delegation(BaseModel):
    model_config = ConfigDict(frozen=True)

    parts: list[SubtaskSpec] = Field(min_length=2, max_length=MAX_DELEGATED_PARTS)
    note: str = Field(min_length=1, max_length=2000, pattern=r"\S")


Outcome = Success | Failure | Delegation

ONE_MODEL: Strategies = functools.partial(strategy_for, other_model=False)
"""The coordinator's default: no other model to switch to, so none is claimed in the audit."""

ERROR_TYPE = "error"
"""A result `{"type": "error", "reason": ...}` is the executor saying it could not do the work."""
REASON_EXCERPT = 300
DELEGATION_NOTE_SHOWN = 600
"""Characters of each earlier delegation note a resumed master is shown (≤ MAX_DELEGATIONS)."""


def is_error_report(content: dict[str, Any] | None) -> bool:
    """An artifact that reports a failure instead of a result: never accepted (ROADMAP 2.8)."""
    return content is not None and content.get("type") in (ERROR_TYPE, "delegate")


def _error_reason(content: dict[str, Any]) -> str:
    if content.get("type") == "delegate":
        return "an unhandled delegation is not an executor result"
    reason = content.get("reason") or content.get("message") or "no reason given"
    text = str(reason)
    prefix = "executor reported an error: "
    while text.startswith(prefix):
        text = text[len(prefix) :]
    return f"{prefix}{text[:REASON_EXCERPT]}"


class Verdict(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    feedback: str = ""
    tests: str | None = Field(default=None, max_length=20_000)

    @field_validator("tests")
    @classmethod
    def _test_module(cls, tests: str | None) -> str | None:
        if tests is not None:
            try:
                module = ast.parse(tests)
            except (SyntaxError, ValueError, RecursionError) as error:
                raise ValueError(f"tests must be valid Python: {error}") from error
            if not any(
                isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
                for node in module.body
            ):
                raise ValueError("tests must define a top-level test_ function")
        return tests


class TestRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    ran: bool
    reason_not_run: str | None = Field(default=None, max_length=1_000)
    missing_packages: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    report_missing: bool = False
    failing: list[Annotated[str, Field(max_length=250)]] = Field(
        default_factory=list, max_length=20
    )
    tail: str = Field(default="", max_length=4_000)
    timed_out: bool = False
    workspace_full: str = ""
    stopped: str = ""
    disk_low: bool = False
    sandbox_unavailable: bool = False
    copy_over_quota: str = ""
    skipped_files: int = Field(default=0, ge=0)
    skipped_names: list[str] = Field(default_factory=list, max_length=5)


RunReviewTests = Callable[[Subtask, Workspace, str, int, Callable[[float], None]], TestRun]


@dataclass(frozen=True)
class ReviewTestRunner:
    run: RunReviewTests
    unavailable: Callable[[], str | None]


ReviewTests = RunReviewTests | ReviewTestRunner

# Scope the test contract to the current call while keeping the two-argument Reviewer API.
review_tests_enabled: ContextVar[bool] = ContextVar("review_tests_enabled", default=False)


def needs_review_tests(subtask: Subtask, *, written_python_files: bool) -> bool:
    return subtask.runs_code or (subtask.kind is Kind.CODE and written_python_files)


class Work(BaseModel):
    """What an executor gets: the subtask, its context and a way to keep its lease alive."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    subtask: Subtask
    context: Context
    """Inputs, checkpoint, accepted results of dependencies (fitted to the budget), feedback on
    the previous attempt. Everything the executor may build on."""
    heartbeat: Callable[[dict[str, Any] | None], None]
    """Call during long work; pass a checkpoint to save progress. Raises StaleStateError if the
    lease was lost or the subtask was given up for critical work (`PreemptedError`, after the
    checkpoint was saved): stop and drop the work."""
    tools: Toolbox
    """The only way to act: tool calls on the task's own workspace (and, once granted, more).
    A refused or mistaken call comes back as a failed ToolResult; every call is audited."""
    execution: int = 0
    previous_starts: int = 0
    unfit: frozenset[str] = frozenset()
    """The models an execution of this task found unfit (`Failure.unfit_provider`): an executor
    with a lineup of models does not ask them (`Request.avoid`)."""


Executor = Callable[[Work], Outcome]
Labels = tuple[str, tuple[str, ...] | None]
"""The main model and alternatives (None: no other), as `provider/model` labels."""
Decomposer = Callable[[Work], list[SubtaskSpec] | None]
"""Splits the subtask of `Work` into smaller ones (ROADMAP 2.9b); None if it cannot. Provider
errors propagate like the executor's."""
Reviewer = Callable[[Subtask, dict[str, Any] | None], Review | Failure]


@dataclass(frozen=True)
class Crew:
    """Who works the subtasks of jobs from desman-global (ROADMAP G.6b): only the models the
    person lent, within their quotas; tools by the `network_*` roles of the tool policy (and
    the person's switches); never split (a job is one self-contained subtask)."""

    executor: Executor
    reviewer: Reviewer
    access: Access | None
    """None: the workspace tools only, as for the coordinator's own `access`."""
    strategies: Strategies = ONE_MODEL
    providers: Labels | Callable[[Subtask], Labels] | None = None
    planner: LLM | None = None
    step_seconds: int = 0


class RunResult(BaseModel):
    task_id: str
    status: TaskStatus
    ticks: int
    paused_until: str | None = None
    """Set when the run returned because only paused work was left."""
    events: list[str] = Field(default_factory=list)


class WorkResult(BaseModel):
    """What `Coordinator.work()` did before it stopped."""

    ticks: int = 0
    executed: int = 0
    """Executions started (whatever came of them)."""
    reviewed: int = 0
    """Verdicts and review failures recorded."""
    preempted: int = 0
    """Executions given up for critical work (each resumes later)."""
    events: list[str] = Field(default_factory=list)
    """What the latest tick that moved something did: a worker runs for days, so its history
    is kept by the counters and the audit log, not here."""


class Coordinator:
    def __init__(
        self,
        store: Store,
        executor: Executor,
        reviewer: Reviewer,
        *,
        owner: str = "coordinator",
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        review_seconds: int = DEFAULT_REVIEW_SECONDS,
        context_budget_tokens: int = DEFAULT_BUDGET_TOKENS,
        workspaces: Path | None = None,
        tools: Registry | None = None,
        access: Access | None = None,
        abandoned: Callable[[str], bool] | None = None,
        strategies: Strategies = ONE_MODEL,
        providers: Labels | Callable[[Subtask], Labels] | None = None,
        decomposer: Decomposer | None = None,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        watch: Callable[[], None] | None = None,
        gate: Callable[[], None] | None = None,
        network: Crew | None = None,
        knowledge: Knowledge | None = None,
        review_tests: ReviewTests | None = None,
    ) -> None:
        self.store = store
        self.knowledge = knowledge
        """Where the knowledge tree's paths a subtask was given are looked up (ADR 0009). None:
        nobody is given any. A job from desman-global never is."""
        self.executor = executor
        self.reviewer = reviewer
        self.review_tests = review_tests
        self.owner = owner
        """Lease and review reservations are taken under this name: unique per process when
        several share a database (`core.owner.current_owner`)."""
        self._lease_seconds = lease_seconds
        self._review_seconds = review_seconds
        self.context_budget_tokens = context_budget_tokens
        self.workspaces = workspaces if workspaces is not None else workspace_root_for(store.conn)
        self.tools = builtin_registry() if tools is None and review_tests is not None else tools
        """The registry executors may call into; the built-in tools by default."""
        self.access = access
        """What the executor may use: its tool policy within the human's permissions. By
        default the shipped policy with every switch off (the workspace tools only). A task
        made with `Tools.NONE` gets no tools at all, whatever this says."""
        self.abandoned = abandoned
        """Tells whether a lease owner is certainly gone, so its work is taken back at once
        instead of after the lease expires (see core/owner.py)."""
        self.strategies = strategies
        """Attempt number -> strategy. The executor's owner says what it can do: by default
        there is no other model to switch to, so the audit never claims a switch that did not
        happen; pass `strategy_for` when the executor has one."""
        self.providers = providers
        """The labels of the models the executor asks (`provider/model`, or a provider's name):
        the main one and the other model of `strategies` (None if there is none); or, when they
        depend on the subtask (its kind, ROADMAP 4.5), a function giving them for one. Without
        them no model is avoided as unfit."""
        self.decomposer = decomposer
        """Without one, subtasks are never split."""
        self.sleep = sleep
        self.poll_seconds = poll_seconds
        self.executed = 0
        self.reviewed = 0
        self.preempted = 0
        """Subtasks this coordinator gave back for critical work."""
        self._handover: Subtask | None = None
        """The critical subtask claimed when the current execution was preempted."""
        self.watch = watch
        """Called whenever the task may look different (housekeeping, a claim, an execution, a
        review), so progress can be shown while `run()` works. It reads the state itself; the
        model calls in between are watched through `calls.watching`."""
        self.gate = gate
        """Called before every tick, before anything is swept, reserved or claimed: whatever it
        raises stops `run()` and `work()` with nothing held (protected files that changed
        since the start, ROADMAP 5.3)."""
        self.network = network
        """Who works jobs from desman-global; None: they are never taken here."""

    @property
    def lease_seconds(self) -> int:
        return max(self._lease_seconds, self.network.step_seconds if self.network else 0)

    @property
    def review_seconds(self) -> int:
        return max(self._review_seconds, self.network.step_seconds if self.network else 0)

    def run(
        self,
        task_id: str,
        *,
        wait_for_pauses: bool = False,
        max_ticks: int = 10_000,
        short_pause_seconds: float = 0,
        on_pause: Callable[[float], None] | None = None,
    ) -> RunResult:
        """Work on `task_id` until it is no longer ACTIVE.

        With `wait_for_pauses` the coordinator sleeps until the earliest pause ends; otherwise it
        returns, and a later `run()` resumes (the state is all in the database). Callers can
        opt into waiting for pauses up to `short_pause_seconds`, announced by `on_pause`.
        `max_ticks` is
        a safety net against bugs, not part of normal termination.
        """
        result = RunResult(task_id=task_id, status=self.store.task_status(task_id), ticks=0)
        while result.status is TaskStatus.ACTIVE:
            if self.store.find_task(task_id)["paused_at"] is not None:
                return result
            if result.ticks >= max_ticks:
                raise RuntimeError(f"task {task_id} still active after {max_ticks} ticks")
            result.ticks += 1
            progressed = self.tick(task_id, result.events)
            result.status = self.store.task_status(task_id)
            if progressed or result.status is not TaskStatus.ACTIVE:
                continue
            # Nothing could move. Either someone else holds work (a worker's lease or review, or
            # our own previous process that crashed) or only paused work is left.
            held_until = self._held_until(task_id)
            pause_until = self._earliest(task_id, "paused", "paused_until")
            if held_until is not None and (pause_until is None or held_until <= pause_until):
                # Look again soon (a live holder finishes long before its lease would run out),
                # at the latest just after the hold ends.
                self.sleep(min(self.poll_seconds, self._seconds_until(held_until) + 0.001))
                continue
            if pause_until is None:
                raise RuntimeError(f"task {task_id} is active but nothing can move")  # a bug
            seconds = self._seconds_until(pause_until)
            if not wait_for_pauses and seconds > short_pause_seconds:
                result.paused_until = pause_until
                return result
            if not wait_for_pauses and on_pause is not None:
                on_pause(seconds)
            self.sleep(seconds)
        return result

    def work(
        self,
        *,
        until_idle: bool = False,
        max_ticks: int | None = None,
        on_tick: Callable[[WorkResult], None] | None = None,
        on_idle: Callable[[], bool | None] | None = None,
    ) -> WorkResult:
        """Work on every task, as a worker: until interrupted, or with `until_idle` until there
        is nothing to do and nobody else holds work (paused work is waited for only without it).
        `on_tick` sees the result so far after every tick that moved something (its `events`
        are that tick's); `on_idle` is called after every tick that moved nothing, before the
        worker waits or stops (others may have changed something meanwhile). It returns True
        when it brought work itself (a job from desman-global): the worker looks again at once,
        and does not stop with it left undone."""
        result = WorkResult()
        while max_ticks is None or result.ticks < max_ticks:
            result.ticks += 1
            executed, reviewed, preempted = self.executed, self.reviewed, self.preempted
            events: list[str] = []
            progressed = self.tick(None, events)
            result.executed += self.executed - executed
            result.reviewed += self.reviewed - reviewed
            result.preempted += self.preempted - preempted
            if progressed:
                result.events = events
                if on_tick is not None:
                    on_tick(result)
                continue
            if on_idle is not None and on_idle():
                continue
            if until_idle and self._held_until(None) is None:
                break
            pause_until = self._earliest(None, "paused", "paused_until")
            wait = self.poll_seconds
            if pause_until is not None:
                wait = min(wait, self._seconds_until(pause_until) + 0.001)
            self.sleep(wait)
        return result

    def tick(self, task_id: str | None, events: list[str]) -> bool:
        """One pass over the task (or every task, for None); returns whether anything moved."""
        if self.gate is not None:
            self.gate()
        recovered, resumed = self.store.sweep(task_id=task_id, abandoned=self.abandoned)
        events += [f"recovered {s.id}" for s in recovered]
        events += [f"resumed {s.id}" for s in resumed]
        progressed = bool(recovered or resumed)
        if progressed:
            self._changed()

        # Reviews and executions wait in one order (core/scheduler.py): results are judged first
        # unless work more urgent than they are is ready to run; that runs first, they wait for
        # the next tick. Equal urgency favours the review: its execution is paid for already.
        # Ready work is looked at again before each review: a verdict may have unlocked some.
        network = self.network is not None
        for subtask in self.store.pending_reviews(task_id=task_id, network=network):
            ready = self.store.next_ready(task_id=task_id, network=network)
            if ready is not None and self.store.rank(subtask) < self.store.rank(ready):
                break
            token = self.store.reserve_review(
                subtask.id, self.owner, seconds=self.review_seconds, abandoned=self.abandoned
            )
            if token is None:
                continue  # another worker is on it, or it was decided meanwhile
            try:
                progressed |= self._review(subtask, token, events)
            finally:
                self.store.release_review(subtask.id, token)  # a no-op after a verdict
            self._changed()

        claimed = self.store.claim(
            self.owner, task_id=task_id, lease_seconds=self.lease_seconds, network=network
        )
        while claimed is not None:
            self._changed()
            self.executed += 1
            self._execute(claimed, task_id, events)
            self._changed()
            progressed = True
            # Preempted: the critical subtask is claimed already. It is never preempted itself.
            claimed, self._handover = self._handover, None
        return progressed

    def _changed(self) -> None:
        if self.watch is not None:
            self.watch()

    def _crew(self, subtask: Subtask) -> Crew:
        """Who works `subtask`: the network's crew for a job from desman-global, else ours."""
        if self.network is not None and self.store.origin(subtask.task_id) is Origin.NETWORK:
            return self.network
        return Crew(self.executor, self.reviewer, self.access, self.strategies, self.providers)

    def _for(self, role: calls.Role, subtask: Subtask) -> calls.Purpose:
        return calls.Purpose(role, task_id=subtask.task_id, subtask_id=subtask.id)

    def _execute(self, subtask: Subtask, scope: str | None, events: list[str]) -> None:
        """Run `subtask`; critical work waiting in `scope` (a task, or every task for None)
        preempts it at the executor's next step, leaving the claimed one in `_handover`."""
        lease = subtask.lease_token
        assert lease is not None
        journal = Effects(self.store, subtask.id, lease)  # a rerun replays, never repeats

        def heartbeat(checkpoint: dict[str, Any] | None = None) -> None:
            self.store.heartbeat(
                subtask.id,
                lease,
                lease_seconds=self.lease_seconds,
                checkpoint=checkpoint,
                # A run resuming from the checkpoint skips the calls before it: so does the replay.
                effects=None if checkpoint is None else journal.position,
            )
            if self.store.yield_paused(subtask.id, lease):
                events.append(f"task {subtask.task_id} paused")
                raise PreemptedError(subtask.id, "operator pause")
            if subtask.priority >= Priority.CRITICAL:
                return
            critical = self.store.preempt(
                subtask.id, lease, self.owner, task_id=scope, lease_seconds=self.lease_seconds
            )
            if critical is not None:  # audited as `preempted`; the checkpoint above is kept
                self._handover = critical
                self.preempted += 1
                events.append(f"preempted {subtask.id} for {critical.id}")
                raise PreemptedError(subtask.id, critical.id)

        def cover_call(timeout: float) -> None:
            self.store.heartbeat(
                subtask.id,
                lease,
                lease_seconds=math.ceil(timeout) + MODEL_CALL_MARGIN,
                only_extend=True,
            )

        outcome: Outcome
        crew = self._crew(subtask)
        own = crew is not self.network  # a job from the network is never split
        try:
            attempt = subtask.attempts.quality + 1
            unfit = frozenset(self.store.unfit_providers(subtask.task_id))
            avoid = self._avoid(crew, subtask, unfit)
            can_delegate = (
                subtask.master
                and own
                and subtask.parent_id is None
                and self.store.origin(subtask.task_id) is Origin.LOCAL
            )
            delegations = self.store.delegation_notes(subtask.id) if can_delegate else []
            if self.store.was_decomposed(subtask.id):
                strategy = Strategy.COMBINE  # its parts are done: their results are its inputs
            else:
                # A split would let parts hand in what only running the code can tell.
                can_split = (
                    own
                    and self.decomposer is not None
                    and subtask.parent_id is None
                    and not delegations
                    and not subtask.runs_code
                    # Parts of a long document and their combines are cut by code (ADR 0014):
                    # splitting them again only muddles which part a note is from.
                    and "document" not in subtask.inputs
                )
                strategy = crew.strategies(attempt, split=can_split, **avoid)
            tools = self._toolbox(subtask, journal, heartbeat, crew.access)
            if subtask.runs_code and strategy is not Strategy.COMBINE:
                missing = tools.refusal(RUN_TOOL)
                if missing is not None:
                    # No model is asked: whatever it handed in could not rest on a run.
                    self._cannot_run(subtask, lease, missing, events)
                    return
            goal, is_final = self.store.subtask_goal(subtask)
            offered = [str(spec["name"]) for spec in tools.specs()]
            continuation_rows: list[EffectRef] = []
            context = build_context(
                subtask,
                self.store.dependency_results(subtask.id),
                self.store.last_feedback(subtask.id),
                self.context_budget_tokens,
                strategy,
                self._knowledge(subtask) if own else None,
                goal=goal,
                final_goal=goal if is_final else None,
                files=(lambda chars: workspace_files(self.store, subtask, tools.workspace, chars))
                if {"read_file", "write_file", "edit_file", "list_files"}.intersection(offered)
                else None,
                continuation=(
                    lambda chars: earlier_results(
                        journal, subtask, tools.workspace, chars, continuation_rows
                    )
                )
                if "read_file" in offered and strategy is not Strategy.COMBINE
                else None,
                earlier=lambda chars: earlier_lookups(
                    self.store, subtask, self.store.executions(subtask.id), chars
                ),
            )
            if can_delegate:
                context = context.model_copy(
                    update={
                        # Outside the budget the dependencies are fitted to, so kept short:
                        # the parts' results matter more than the master's notes on them.
                        "delegations": [
                            {**d, "note": d["note"][:DELEGATION_NOTE_SHOWN]} for d in delegations
                        ],
                        "delegations_left": MAX_DELEGATIONS - len(delegations),
                    }
                )
            previous_starts = self.store.execution_starts(subtask.id, journal.execution)
            self._started(subtask, context, offered, continuation_rows)
            work = Work(
                subtask=subtask,
                context=context,
                heartbeat=heartbeat,
                tools=tools,
                unfit=unfit,
                execution=journal.execution,
                previous_starts=previous_starts,
            )
            if strategy is Strategy.DECOMPOSE:
                with guard_model_calls(cover_call):
                    if self._decompose(work, lease, events):
                        return
                # No usable split: try once more the way the attempt before did.
                context = context.model_copy(
                    update={"strategy": crew.strategies(attempt, split=False, **avoid)}
                )
                work = work.model_copy(update={"context": context})
                self._started(subtask, context, offered, continuation_rows)

            def unfit_heard(label: str) -> None:
                self.store.note(
                    subtask.id, Actor.EXECUTOR, "model_unfit", {"unfit_provider": label}
                )

            with (
                calls.for_(self._for(calls.Role.EXECUTOR, subtask)),
                noticing_unfit(unfit_heard),
                guard_model_calls(cover_call),
            ):
                returned = crew.executor(work)
            if not isinstance(returned, Success | Failure | Delegation):
                raise TypeError(
                    f"executor returned {type(returned).__name__}, not Success/Failure/Delegation"
                )
            outcome = returned
        except PreemptedError:
            return  # given back, and recorded, by the heartbeat
        except StaleStateError:
            self._discard(subtask, "lease lost during execution", events)
            return
        except ProviderError as e:  # agents may let these propagate: the error knows its kind
            outcome = self._provider_failure(e)
        except PathEscapeError as e:
            # The attempt asked for a forbidden path: try again, differently.
            outcome = Failure(kind=FailureKind.QUALITY, reason=f"path refused: {e}")
        except Exception as e:  # an executor bug or crash: the attempt was interrupted
            outcome = Failure(
                kind=FailureKind.TRANSIENT, reason=f"executor raised {type(e).__name__}: {e}"
            )

        if isinstance(outcome, Delegation):
            try:
                if not can_delegate:
                    raise ValueError("only a local master may delegate")
                if strategy is Strategy.COMBINE:
                    raise ValueError("this master was already split")
                if len(delegations) >= MAX_DELEGATIONS:
                    raise ValueError("this master has reached its delegation limit")
                created = self.store.delegate(subtask.id, lease, outcome.parts, outcome.note)
                events.append(f"delegated {subtask.id} into {len(created)}")
                return
            except StaleStateError:
                self._discard(subtask, "lease lost before delegation", events)
                return
            except ValueError as e:
                outcome = Failure(kind=FailureKind.QUALITY, reason=f"delegation not possible: {e}")

        if isinstance(outcome, Success) and is_error_report(outcome.content):
            # Not a result: a failed attempt, never something a reviewer could accept.
            outcome = Failure(kind=FailureKind.QUALITY, reason=_error_reason(outcome.content))
        elif isinstance(outcome, Success) and subtask.runs_code:
            run = tools.last_run
            if run is None:
                run = self._journal_run(journal)
            output = run.output if run is not None and run.ok else None
            if (
                not isinstance(output, dict)
                or output.get("exit_code") != 0
                or output.get("workspace_full")
            ):
                kind = FailureKind.QUALITY
                if isinstance(output, dict) and output.get("workspace_full"):
                    reason = (
                        f"the last {RUN_TOOL} left the workspace over its quota "
                        f"({output['workspace_full']}), then run the code again and report "
                        "what it printed"
                    )
                elif isinstance(output, dict) and output.get("disk_low"):
                    # Other programs' writes lower the free space too: not the code's fault the
                    # first time; again (e.g. writing to files it deleted), it is the code's.
                    if subtask.attempts.transient == 0:
                        kind = FailureKind.TRANSIENT
                    reason = (
                        f"the last {RUN_TOOL} was stopped ({output.get('stopped', '')}): "
                        "run the code again and report what it printed"
                    )
                elif isinstance(output, dict) and output.get("stopped"):
                    reason = (
                        f"the last {RUN_TOOL} was stopped ({output['stopped']}): run the code "
                        "again and report what it printed"
                    )
                else:
                    reason = (
                        f"the result was handed in without a successful {RUN_TOOL} call"
                        if run is None
                        else f"the last {RUN_TOOL} failed"
                    ) + ": run the code again and report what it printed"
                outcome = Failure(kind=kind, reason=reason)
        if (
            isinstance(outcome, Failure)
            and outcome.kind is FailureKind.QUALITY
            and self._missing_search(journal)
        ):
            self._cannot_search(subtask, lease, events)
            return
        try:
            if isinstance(outcome, Success):
                self.store.submit(subtask.id, lease, outcome.content)
                events.append(f"submitted {subtask.id}")
            else:
                moved = self.store.record_failure(
                    subtask.id,
                    S.RUNNING,
                    Actor.EXECUTOR,
                    outcome.kind,
                    lease=lease,
                    reason=outcome.reason,
                    paused_until=outcome.paused_until,
                    action="execution_failed",
                    details=(
                        None
                        if outcome.unfit_provider is None
                        else {"unfit_provider": outcome.unfit_provider}
                    ),
                )
                events.append(f"{outcome.kind} {subtask.id} -> {moved.status}")
        except StaleStateError:  # e.g. cancelled while running
            # (An executor that swallowed its preemption lands here too: its work is dropped.)
            self._discard(subtask, f"lease lost before {type(outcome).__name__.lower()}", events)

    def _toolbox(
        self,
        subtask: Subtask,
        journal: Effects,
        heartbeat: Callable[[dict[str, Any] | None], None],
        access: Access | None,
    ) -> Toolbox:
        def record(action: str, details: dict[str, Any]) -> None:
            self.store.note(
                subtask.id, Actor.EXECUTOR, action, {**details, "execution": journal.execution}
            )

        # A part of a long document and its combines work from their inputs and dependencies
        # only (ADR 0014): with tools, notes went to a file no check reads (L.2 live run).
        granted = (
            self.store.tools(subtask.task_id) is Tools.GRANTED and "document" not in subtask.inputs
        )
        workspace = Workspace(self.workspaces, subtask.task_id)

        def screen(tool: Tool[Any], args: BaseModel) -> str | None:
            if tool.name != "web_fetch" or not isinstance(url := getattr(args, "url", None), str):
                return None
            return webguard.screen(self.store, subtask.task_id, workspace, url)

        return Toolbox(
            workspace,
            self.tools if granted else Registry(),
            access=access if granted else None,
            record=record,
            guard=lambda: heartbeat(None),  # a lost lease stops the work before any effect
            journal=journal,
            screen=screen,  # what web_fetch may open (ROADMAP S.1)
        )

    def _execution_results(self, journal: Effects) -> list[tuple[str, ToolResult]]:
        """Finished tool calls of this execution, in their original order."""
        return [
            (row["tool"], ToolResult.model_validate_json(row["result_json"]))
            for row in self.store.effects(journal.subtask_id, journal.execution)
            if row["state"] == "done"
        ]

    def _journal_run(self, journal: Effects) -> ToolResult | None:
        """The most recent run in this execution, including calls after a checkpoint."""
        for row in reversed(self.store.effects(journal.subtask_id, journal.execution)):
            if row["tool"] == RUN_TOOL:
                if row["state"] == "done":
                    return ToolResult.model_validate_json(row["result_json"])
                return ToolResult(ok=False, error="interrupted run_command")
        return None

    def _missing_search(self, journal: Effects) -> bool:
        """A missing whole-web engine was the last kind of tool failure."""
        missing = False
        for tool, result in self._execution_results(journal):
            if not result.ok:
                missing = tool == "web_search" and result.missing == "search_engine"
        if not missing:
            return False
        # Refusals and invalid arguments have no effects to replay, but still stop work.
        last = self.store.conn.execute(
            "SELECT details_json FROM audit_log WHERE subtask_id = ? "
            "AND json_extract(details_json, '$.execution') = ? "
            "AND (action IN ('tool_refused', 'web_refused') "
            "OR (action IN ('tool_called', 'tool_replayed') "
            "AND json_extract(details_json, '$.ok') = 0)) ORDER BY seq DESC LIMIT 1",
            (journal.subtask_id, journal.execution),
        ).fetchone()
        return last is None or json.loads(last[0]).get("missing") == "search_engine"

    def _cannot_search(self, subtask: Subtask, lease: str, events: list[str]) -> None:
        try:
            moved = self.store.record_failure(
                subtask.id,
                S.RUNNING,
                Actor.EXECUTOR,
                FailureKind.PERMANENT,
                lease=lease,
                reason=(
                    "A search engine for the whole web is needed; "
                    "run desman config to add a Tavily key or SearXNG."
                ),
                action="tool_unavailable",
                details={"tool": "web_search", "missing": "search_engine"},
            )
        except StaleStateError:
            self._discard(
                subtask, "lease lost before the missing search engine was recorded", events
            )
            return
        events.append(f"permanent {subtask.id} -> {moved.status}")

    def _cannot_run(self, subtask: Subtask, lease: str, why: str, events: list[str]) -> None:
        """A subtask that must run code, without the tool to run it: the person decides (turn
        the permission on and retry, or settle it otherwise). No attempt is spent."""
        try:
            moved = self.store.record_failure(
                subtask.id,
                S.RUNNING,
                Actor.EXECUTOR,
                FailureKind.PERMANENT,
                lease=lease,
                reason=f"this subtask must run code, but {why}",
                action="tool_unavailable",
                details={"tool": RUN_TOOL},
            )
        except StaleStateError:
            self._discard(subtask, "lease lost before the missing tool was recorded", events)
            return
        events.append(f"permanent {subtask.id} -> {moved.status}")

    def _knowledge(self, subtask: Subtask) -> list[KnowledgeText]:
        """The texts of the nodes the subtask was given. A path the tree no longer has is left
        out and noted (`context_missing`): the executor can still look with `read_node`."""
        if not subtask.knowledge:
            return []
        found, missing = given(self.knowledge, subtask.knowledge)
        if missing:
            self.store.note(subtask.id, Actor.EXECUTOR, "context_missing", {"paths": missing})
        return found

    def _started(
        self, subtask: Subtask, context: Context, tools: list[str], continuation: list[EffectRef]
    ) -> None:
        self.store.note(
            subtask.id,
            Actor.EXECUTOR,
            "attempt_started",
            {
                "attempt": context.attempt,
                "strategy": context.strategy.value,
                "execution": self.store.executions(subtask.id),
                "tools": tools,
                "continuation": continuation,
            },
        )

    def _decompose(self, work: Work, lease: str, events: list[str]) -> bool:
        """Split the subtask; False (recorded) if the decomposer could not. A lost lease or a
        provider error propagates to `_execute`, as from the executor."""
        assert self.decomposer is not None
        subtask = work.subtask
        with calls.for_(self._for(calls.Role.DECOMPOSER, subtask)):
            parts = self.decomposer(work)
        if parts:
            try:
                created = self.store.decompose(subtask.id, lease, parts)
            except ValueError as e:  # not a valid graph of parts
                reason = str(e)
            else:
                events.append(f"split {subtask.id} into {len(created)}")
                return True
        else:
            reason = "the decomposer returned no parts"
        self.store.note(subtask.id, Actor.EXECUTOR, "decomposition_failed", {"reason": reason})
        return False

    def _review_test_skip(
        self, subtask: Subtask, cover_call: Callable[[float], None]
    ) -> TestRun | None:
        if self.review_tests is None:
            return TestRun(ran=False, reason_not_run="review test runner unavailable")
        if self.store.tools(subtask.task_id) is not Tools.GRANTED or "document" in subtask.inputs:
            return TestRun(ran=False, reason_not_run="task does not grant tools")
        tool = self.tools.get(RUN_TOOL) if self.tools is not None else None
        if tool is None:
            return TestRun(ran=False, reason_not_run="run_command unavailable")
        access = self._crew(subtask).access or executor_access()
        if reason := access.refusal(tool):
            return TestRun(ran=False, reason_not_run=reason)
        if isinstance(self.review_tests, ReviewTestRunner):
            cover_call(120)
            if reason := self.review_tests.unavailable():
                return TestRun(ran=False, reason_not_run=reason[:1_000], sandbox_unavailable=True)
        return None

    def _check_review_tests(
        self,
        subtask: Subtask,
        review: Review,
        skipped: TestRun | None,
        cover_call: Callable[[float], None],
    ) -> tuple[Review | Failure, dict[str, Any] | None]:
        attempt = subtask.attempts.quality + 1
        if skipped is not None:
            result = skipped
        elif review.tests is None:
            return Failure(kind=FailureKind.TRANSIENT, reason="reviewer gave no tests"), None
        else:
            runner = self.review_tests
            assert runner is not None
            run = runner.run if isinstance(runner, ReviewTestRunner) else runner
            workspace = Workspace(self.workspaces, subtask.task_id)
            try:
                result = TestRun.model_validate(
                    run(subtask, workspace, review.tests, attempt, cover_call)
                )
            except SandboxUnavailable as error:
                result = TestRun(
                    ran=False, reason_not_run=str(error)[:1_000], sandbox_unavailable=True
                )

        note = {
            **result.model_dump(exclude={"tail"}),
            "tail": result.tail[-2_000:],
            "attempt": attempt,
        }
        if skipped is None:
            note["source"] = review.tests
        return self._review_test_verdict(subtask, review, result), note

    @staticmethod
    def _review_test_verdict(subtask: Subtask, review: Review, result: TestRun) -> Review | Failure:
        if result.copy_over_quota:
            return Failure(kind=FailureKind.PERMANENT, reason=result.copy_over_quota)
        elif result.disk_low:
            return Failure(
                kind=FailureKind.TRANSIENT, reason=result.stopped or "disk space too low"
            )
        elif result.sandbox_unavailable and subtask.runs_code:
            return Failure(
                kind=FailureKind.TRANSIENT,
                reason=result.reason_not_run or "review sandbox unavailable",
            )
        elif result.stopped or result.workspace_full:
            feedback = f"reviewer's tests failed: {result.stopped or result.workspace_full}"
        elif not result.ran:
            return review
        elif result.timed_out:
            feedback = "reviewer's tests failed: timed out"
        elif result.report_missing:
            feedback = "reviewer's tests produced no report"
        elif result.exit_code == 0 and result.failed == result.errors == 0:
            if result.passed > 0:
                return review
            feedback = "reviewer's tests verified nothing: 0 passed"
        else:
            reason = ", ".join(result.failing) or f"pytest exited {result.exit_code}"
            feedback = f"reviewer's tests failed: {reason}"
        if result.skipped_files:
            names = ", ".join(repr(name) for name in result.skipped_names)
            feedback += f"\n{result.skipped_files} files could not be copied"
            feedback += f": {names}" if names else "."
        return Review(
            verdict=Verdict.RETRY,
            feedback=feedback + (f"\n{result.tail}" if result.tail else ""),
        )

    def _review(self, subtask: Subtask, token: str, events: list[str]) -> bool:
        """Review the submitted result under the reservation `token`; the verdict is written
        only while the reservation is still ours."""
        review: Review | Failure
        needs_tests = False
        skipped = None
        artifact = self.store.latest_artifact(subtask.id)

        def cover_call(timeout: float) -> None:
            self.store.extend_review(subtask.id, token, math.ceil(timeout) + MODEL_CALL_MARGIN)

        try:
            needs_tests = not is_error_report(artifact) and needs_review_tests(
                subtask,
                written_python_files=subtask.kind is Kind.CODE
                and not subtask.runs_code
                and has_written_python_files(self.store, subtask, root=self.workspaces),
            )
            skipped = self._review_test_skip(subtask, cover_call) if needs_tests else None
            if skipped and skipped.sandbox_unavailable and subtask.runs_code:
                review = Failure(
                    kind=FailureKind.TRANSIENT,
                    reason=skipped.reason_not_run or "review sandbox unavailable",
                )
            else:
                reviewer = self._crew(subtask).reviewer
                setting = review_tests_enabled.set(needs_tests and skipped is None)
                try:
                    with (
                        calls.for_(self._for(calls.Role.REVIEWER, subtask)),
                        guard_model_calls(cover_call),
                    ):
                        returned = reviewer(subtask, artifact)
                finally:
                    review_tests_enabled.reset(setting)
                # Validate at the boundary: e.g. a verdict parsed from an LLM arrives as a string.
                review = (
                    returned if isinstance(returned, Failure) else Review.model_validate(returned)
                )
        except ProviderError as e:
            review = self._provider_failure(e)
            if review.kind is FailureKind.QUALITY:
                review = review.model_copy(update={"kind": FailureKind.TRANSIENT})
        except (Exception, ValidationError) as e:
            review = Failure(
                kind=FailureKind.TRANSIENT, reason=f"reviewer raised {type(e).__name__}: {e}"
            )
        if (
            isinstance(review, Review)
            and review.verdict is Verdict.ACCEPT
            and is_error_report(artifact)
        ):
            # Whatever the reviewer says: an error report is not a result.
            assert artifact is not None
            review = Review(
                verdict=Verdict.RETRY,
                feedback=f"{_error_reason(artifact)} (accepted by the reviewer, overruled)",
            )
        try:
            note = None
            if skipped and skipped.sandbox_unavailable and subtask.runs_code:
                note = {**skipped.model_dump(), "attempt": subtask.attempts.quality + 1}
            elif (
                isinstance(review, Review)
                and needs_tests
                and (skipped is not None or review.verdict is Verdict.ACCEPT)
            ):
                try:
                    review, note = self._check_review_tests(subtask, review, skipped, cover_call)
                except StaleStateError:
                    raise
                except Exception as error:
                    review = Failure(
                        kind=FailureKind.TRANSIENT,
                        reason=f"review tests raised {type(error).__name__}: {error}",
                    )
            if note is not None:
                with db.transaction(self.store.conn):
                    self.store.extend_review(subtask.id, token, MODEL_CALL_MARGIN)
                    self.store.note(subtask.id, Actor.REVIEWER, "review_tests", note)
            if isinstance(review, Review):
                details = {"verdict": review.verdict.value, "feedback": review.feedback}
            if isinstance(review, Failure):
                # Counts against the subtask's budget, so a reviewer that keeps failing cannot
                # loop forever. A transient one keeps the subtask in REVIEW: the submitted work is
                # reviewed again, not run again.
                moved = self.store.record_failure(
                    subtask.id,
                    S.REVIEW,
                    Actor.REVIEWER,
                    review.kind,
                    reason=review.reason,
                    paused_until=review.paused_until,
                    action="review_failed",
                    review=token,
                )
            elif review.verdict is Verdict.ACCEPT:
                moved = self.store.transition(
                    subtask.id,
                    S.REVIEW,
                    S.DONE,
                    Actor.REVIEWER,
                    reason=review.feedback or None,
                    details=details,
                    action="verdict",
                    review=token,
                )
            elif review.verdict is Verdict.RETRY:
                moved = self.store.record_failure(
                    subtask.id,
                    S.REVIEW,
                    Actor.REVIEWER,
                    FailureKind.QUALITY,
                    reason=review.feedback or "rejected by reviewer",
                    details=details,
                    action="verdict",
                    review=token,
                )
            elif review.verdict is Verdict.ESCALATE:
                moved = self.store.transition(
                    subtask.id,
                    S.REVIEW,
                    S.NEEDS_HUMAN,
                    Actor.REVIEWER,
                    reason=review.feedback or "escalated by reviewer",
                    details=details,
                    action="verdict",
                    review=token,
                )
        except StaleStateError as e:
            # Decided by someone else, or our reservation ran out and was taken over: the
            # verdict is thrown away, and the history says so.
            events.append(f"lost review {subtask.id}")
            self.store.note(
                subtask.id,
                Actor.REVIEWER,
                "review_discarded",
                {
                    "reason": str(e),
                    "owner": self.owner,
                    **(
                        {"verdict": review.verdict.value, "feedback": review.feedback}
                        if isinstance(review, Review)
                        else {"failure": review.model_dump(mode="json")}
                    ),
                },
            )
            return False
        self.reviewed += 1
        events.append(f"reviewed {subtask.id} -> {moved.status}")
        return True

    def _discard(self, subtask: Subtask, why: str, events: list[str]) -> None:
        """The executor's work is thrown away: say so in the history, not just in `events`."""
        events.append(f"lost lease {subtask.id}")
        self.store.note(
            subtask.id, Actor.EXECUTOR, "work_discarded", {"reason": why, "owner": self.owner}
        )

    def _provider_failure(self, error: ProviderError) -> Failure:
        until = error.paused_until(self.store.clock())
        return Failure(
            kind=error.kind,
            reason=f"{type(error).__name__}: {error}",
            paused_until=None if until is None else db.to_iso(until),
            unfit_provider=error.label if error.unfit_model else None,
        )

    def _avoid(self, crew: Crew, subtask: Subtask, unfit: frozenset[str]) -> dict[str, bool]:
        """Which of the models the executor asks for `subtask` an execution of the task found
        unfit (C.9): switch to another model while any alternative is still fit."""
        if crew.providers is None:
            return {}
        main, other = crew.providers(subtask) if callable(crew.providers) else crew.providers
        avoid_other = other is not None and all(among(candidate, unfit) for candidate in other)
        return {"avoid_main": among(main, unfit) and not avoid_other, "avoid_other": avoid_other}

    def _earliest(self, task_id: str | None, status: str, column: str) -> str | None:
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        row = self.store.conn.execute(
            f"SELECT MIN({column}) FROM subtasks WHERE task_id IN (SELECT id FROM tasks WHERE paused_at IS NULL) AND status = ? {scope}",
            (status, *params),
        ).fetchone()
        moment: str | None = row[0]
        return moment

    def _held_until(self, task_id: str | None) -> str | None:
        """When the earliest hold of someone else ends: a running lease or a review reserved by
        another owner (ours would be taken over, not waited for). None if nobody holds work."""
        scope, params = ("AND task_id = ?", (task_id,)) if task_id else ("", ())
        row = self.store.conn.execute(
            f"""SELECT MIN(CASE WHEN status = 'running' THEN lease_until ELSE review_until END)
                FROM subtasks
                WHERE task_id IN (SELECT id FROM tasks WHERE paused_at IS NULL) AND (status = 'running' OR (status = 'review' AND review_owner != ?)) {scope}""",
            (self.owner, *params),
        ).fetchone()
        moment: str | None = row[0]
        return moment

    def _seconds_until(self, moment: str) -> float:
        target = db.parse_iso(moment)
        return max(0.0, (target - self.store.clock()).total_seconds())
