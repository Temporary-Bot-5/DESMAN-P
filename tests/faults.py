"""Fault-injection harness (ROADMAP 4.6): workers and providers fail at random, repeatably.

A simulation runs a few workers on one database, in one process, on a `FakeClock`. Each worker
is a real `Coordinator` with the real LLM executor and reviewer, its own connection and its own
owner name; only the model behind them is scripted. One random generator, seeded, decides
everything: which worker takes the next step, and where something goes wrong.

- A worker **dies** (like SIGKILL): before or after a model call, in the middle of a tool call
  (after its effect, before its result is recorded), or between two steps of its loop. Its
  connection is closed at that instant, so nothing it had not committed survives, and no
  `finally` of its own can tidy up after it. Its owner is then certainly gone (as
  `owner_is_gone` would say for a dead process on this machine), and a new worker starts.
- A worker **hangs** in a model call (like SIGSTOP): the clock runs past its lease and review
  reservation while the others keep working; then its reply arrives, late.
- The **provider** fails: garbage instead of JSON, a timeout, a rate limit, a server error
  (`FakeLLM.chaos`, seeded from the same generator).
- The **reviewer** asks for another attempt now and then.

The tasks have random priorities; the last one arrives later, critical, while the others run,
so preemption (4.4) happens under the same faults.

The executor's model always asks for one tool call first: `note`, an effect that is visibly not
idempotent (each real run is counted in a ledger, with the execution it belongs to), then
hands in a result. When nothing is left to do, `check()` lists what went wrong:

- a task left ACTIVE (stuck), or in a state no fault should lead to (failed, cancelled);
- a history that does not replay (`check_history`);
- a subtask accepted more than once, done without being accepted, or with another result;
- an effect run twice in one execution (the journal must replay it instead, D6).

The same seed gives the same run, step for step (`Report.trace`), so a failing seed is a bug
that can be replayed: `DESMAN_FAULT_SEED=<seed> pytest tests/test_faults.py`.
"""

from __future__ import annotations

import functools
import itertools
import json
import random
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

from pydantic import BaseModel, ConfigDict

from conftest import FakeClock
from desman.agents import EXECUTE, REVIEW, TOOL_RESULTS, llm_executor, llm_reviewer
from desman.core import db
from desman.core import store as store_module
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator
from desman.core.effects import evidence
from desman.core.models import Priority, SubtaskSpec, TaskSpec
from desman.core.policy import Policy, Role, RolePolicy
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import (
    LIST_FILES,
    READ_FILE,
    WRITE_FILE,
    Access,
    Permission,
    Registry,
    Tool,
)
from desman.core.workspace import Workspace
from desman.llm.base import Completion, Request
from desman.llm.fake import GARBAGE, FakeLLM, StepLike

ACCESS = Access(
    Policy(
        roles={Role.EXECUTOR: RolePolicy(tools=["note", "read_file", "write_file", "list_files"])}
    ),
    Role.EXECUTOR,
)
LATE = 30
"""Steps after which the urgent task arrives anyway, without an execution to preempt."""
ENDED: frozenset[TaskStatus] = frozenset({TaskStatus.DONE, TaskStatus.NEEDS_HUMAN})
"""What every task must come to: done, or waiting for the person (a budget ran out)."""


@dataclass(frozen=True)
class Faults:
    """How often each fault happens, as a chance per opportunity."""

    crash: float = 0.03
    """At each crash point (a model call, its reply, a tool call, a step), the worker dies."""
    hang: float = 0.02
    """At each model call, the worker freezes past its lease while the others go on."""
    provider: float = 0.25
    """At each model call that reaches the provider, it fails (spread over the failure kinds)."""
    retry: float = 0.2
    """At each review, the reviewer asks for another attempt."""


@dataclass(frozen=True)
class Shape:
    """What is simulated."""

    tasks: int = 3
    subtasks: int = 3
    """Per task; in every other task they form a chain (each depends on the one before)."""
    urgent: bool = True
    """The last task arrives later, critical, while the others run (preemption, 4.4); the
    others get random priorities below it."""
    workers: int = 3
    """Kept alive: a dead one is replaced at the next step."""
    lease_seconds: int = 60
    review_seconds: int = 60
    idle_seconds: float = 20.0
    """How far the clock moves when no worker could do anything (pauses, a lease to wait out)."""
    hang_steps: int = 12
    """At most so many steps of the others while one worker hangs."""
    planted_bug: bool = False
    """Declare the `note` effect safe to run twice (it is not): a call cut off midway is then
    run again. Only to show that the harness catches what it is meant to."""
    max_steps: int = 4000
    """A safety net: a run still going after this is stuck (a bug), never a normal end."""


@dataclass
class Report:
    seed: int
    replay: str | None = None
    """The command that reruns this exact simulation (default: the seed in the ordinary test,
    which is right only for the default faults and shape)."""
    trace: list[str] = field(default_factory=list)
    """What happened, in order: the harness's faults and every audit entry (ids replaced by
    names, so two runs of one seed compare equal)."""
    faults: Counter[str] = field(default_factory=Counter)
    """Faults injected, by kind ("crash: model call", "hang", "provider: timeout", ...)."""
    actions: Counter[str] = field(default_factory=Counter)
    """Audit actions recorded, by name."""
    statuses: dict[str, TaskStatus] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    """What `check()` found; empty when every invariant holds."""

    def explain(self, lines: int = 60) -> str:
        tail = "\n".join(self.trace[-lines:])
        return (
            f"seed {self.seed}: " + "; ".join(self.problems) + "\n"
            f"replay: {self.replay or f'DESMAN_FAULT_SEED={self.seed} pytest tests/test_faults.py'}\n"
            f"faults: {dict(self.faults)}\nlast {lines} steps:\n{tail}"
        )


class Crash(BaseException):
    """The simulated process dies here: nothing in DESMAN catches it (like SIGKILL)."""


class Stuck(AssertionError):
    pass


class NoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class _Worker:
    def __init__(self, sim: Simulation, number: int) -> None:
        self.sim = sim
        self.owner = f"sim-{number}"
        self.name = self.owner  # as an `LLM`
        self.dead = False
        self.conn = db.connect(sim.path)
        self.store = Store(self.conn, clock=sim.clock)
        self.rng = random.Random(sim.rng.getrandbits(64))
        self.fake = FakeLLM.chaos(
            self.rng.getrandbits(64),
            self._reply,
            garbage_rate=sim.faults.provider * 0.4,
            timeout_rate=sim.faults.provider * 0.2,
            rate_limit_rate=sim.faults.provider * 0.2,
            server_error_rate=sim.faults.provider * 0.2,
            name=self.owner,
            clock=sim.clock,
            sleep=sim.clock.sleep,
        )
        note = Tool(
            "note",
            "Record a note (not idempotent).",
            NoteArgs,
            Permission.WORKSPACE,
            self._note,
            repeatable=sim.shape.planted_bug,
        )
        self.coordinator = Coordinator(
            self.store,
            llm_executor(self),
            llm_reviewer(self, evidence=functools.partial(evidence, self.store)),
            owner=self.owner,
            lease_seconds=sim.shape.lease_seconds,
            review_seconds=sim.shape.review_seconds,
            tools=Registry([note, READ_FILE, WRITE_FILE, LIST_FILES]),
            access=ACCESS,
            abandoned=sim.dead.__contains__,
            sleep=sim.clock.sleep,
            watch=lambda: self.maybe_crash("step"),
        )

    # The worker is its own model (`LLM`): the faults of the call, then the chaos fake.
    def complete(self, request: Request) -> Completion:
        if request.prompt.startswith(EXECUTE) and len(request.messages) == 2:
            self.sim.arrive()  # before the executor's tool call, which can then be preempted
        self.maybe_crash("model call")
        if self.sim.roll(self.sim.faults.hang) and self.sim.freeze(self):
            self.maybe_crash("model call")  # woken up: it may die right away, like anyone
        served = len(self.fake.served)
        try:
            return self.fake.complete(request)
        finally:
            step = self.fake.served[served] if len(self.fake.served) > served else None
            if step is not None and (step.type != "reply" or step.text in GARBAGE.values()):
                self.sim.fault(
                    self, f"provider: {'garbage' if step.type == 'reply' else step.type}"
                )
            self.maybe_crash("model reply")

    def _reply(self, request: Request) -> StepLike:
        prompt = request.prompt
        body = _body(request)
        if prompt.startswith(REVIEW):
            # A careful reviewer: only the expected result is accepted, and not always at once.
            if body["result"] != {"done": body["subtask"]}:
                return json.dumps({"verdict": "retry", "feedback": "not the expected result"})
            if self.rng.random() < self.sim.faults.retry:
                return json.dumps({"verdict": "retry", "feedback": "do it again"})
            return json.dumps({"verdict": "accept"})
        if not prompt.startswith(EXECUTE):
            raise AssertionError(f"unexpected prompt: {prompt[:80]!r}")
        description = body["subtask"]
        if any(m.role == "user" and m.content.startswith(TOOL_RESULTS) for m in request.messages):
            return json.dumps({"done": description})
        return json.dumps({"tool_calls": [{"tool": "note", "args": {"text": description}}]})

    def _note(self, args: NoteArgs, workspace: Workspace) -> dict[str, Any]:
        subtask_id = self.sim.ids[args.text]
        self.sim.ledger.append((subtask_id, self.sim.observer.executions(subtask_id)))
        self.maybe_crash("tool call")  # after the effect, before its result is recorded
        return {"noted": args.text}

    def maybe_crash(self, point: str) -> None:
        if self.dead or not self.sim.roll(self.sim.faults.crash):
            return
        self.dead = True
        self.sim.dead.add(self.owner)
        self.sim.fault(self, f"crash: {point}")
        self.conn.close()  # whatever it had not committed is lost, and it can write nothing more
        raise Crash(point)


class Simulation:
    def __init__(
        self, path: Path, seed: int, faults: Faults, shape: Shape, replay: str | None = None
    ) -> None:
        self.path = path
        self.seed = seed
        self.faults = faults
        self.shape = shape
        self.rng = random.Random(seed)
        self.clock = FakeClock()
        self.observer = Store(db.open_db(path), clock=self.clock)
        self.workers: list[_Worker] = []
        self.dead: set[str] = set()
        self.started = 0
        self.frozen = False
        self.steps = 0
        self.arrival: int | None = None
        """The step at which the urgent task arrives (None: it did, or there is none)."""
        self.ids: dict[str, str] = {}
        """Subtask description -> id (descriptions are unique)."""
        self.names: dict[str, str] = {}
        """Task and subtask ids -> names for the trace."""
        self.ledger: list[tuple[str, int]] = []
        """(subtask id, execution) for every real run of the `note` effect."""
        self.report = Report(seed, replay=replay)
        self.seen = 0
        """The last audit entry already in the trace."""

    def roll(self, chance: float) -> bool:
        return self.rng.random() < chance

    def fault(self, worker: _Worker, kind: str) -> None:
        self.record()  # what was committed before it, first
        self.report.faults[kind] += 1
        self.report.trace.append(f"{self.clock.now:%H:%M:%S} {worker.owner}: {kind}")

    def create(self, t: int, priority: int) -> None:
        chain = t % 2 == 1
        subtasks = [
            SubtaskSpec(
                description=f"t{t}s{s}",
                acceptance_criteria="noted",
                depends_on=[s - 1] if chain and s else [],
            )
            for s in range(self.shape.subtasks)
        ]
        spec = TaskSpec(goal=f"task {t}", subtasks=subtasks, priority=priority)
        task_id = self.observer.create_task(spec)
        self.names[task_id] = f"t{t}"
        for sub in self.observer.list_subtasks(task_id):
            self.ids[sub.description] = sub.id
            self.names[sub.id] = sub.description

    def arrive(self, *, late: bool = False) -> None:
        """The urgent task comes in once its step is reached, at the start of an execution (to
        preempt it); `late`: at a step, when that has not happened for a while."""
        if self.arrival is not None and self.steps >= self.arrival + (LATE if late else 0):
            self.arrival = None
            self.create(self.shape.tasks - 1, Priority.CRITICAL)

    def live(self) -> list[_Worker]:
        return [w for w in self.workers if not w.dead]

    def spawn(self) -> None:
        while len(self.live()) < self.shape.workers:
            self.started += 1
            self.workers.append(_Worker(self, self.started))

    def step(self, worker: _Worker) -> bool:
        self.steps += 1
        self.arrive(late=True)
        if self.steps > self.shape.max_steps:
            raise Stuck(f"still going after {self.shape.max_steps} steps")
        try:
            return worker.coordinator.tick(None, [])
        except BaseException:
            # Its own death, possibly turned into the error of a write on its closed connection.
            if worker.dead:
                return True
            raise
        finally:
            self.record()

    def freeze(self, worker: _Worker) -> bool:
        """`worker` hangs: its lease and review reservation run out while the others work."""
        if self.frozen:
            return False  # one hang at a time
        self.frozen = True
        self.fault(worker, "hang")
        try:
            self.clock.advance(max(self.shape.lease_seconds, self.shape.review_seconds) + 1)
            for _ in range(self.rng.randint(1, self.shape.hang_steps)):
                self.spawn()
                others = [w for w in self.live() if w is not worker]
                if not self.active():
                    break
                self.step(self.rng.choice(others))
        finally:
            self.frozen = False
        return True

    def active(self) -> bool:
        return self.arrival is not None or any(
            self.observer.task_status(task_id) is TaskStatus.ACTIVE for task_id in self._task_ids()
        )

    def run(self) -> Report:
        ordinary = self.shape.tasks - (1 if self.shape.urgent else 0)
        for t in range(ordinary):
            self.create(t, self.rng.choice([Priority.LOW, Priority.NORMAL, Priority.HIGH]))
        if self.shape.urgent:
            self.arrival = self.rng.randint(1, 60)
        idle = 0
        while self.active():
            self.spawn()
            live = self.live()
            if self.step(self.rng.choice(live)):
                idle = 0
                continue
            idle += 1
            if idle >= len(live):  # nobody can move: pauses or another's hold to wait out
                self.clock.advance(self.shape.idle_seconds)
                idle = 0
        self.check()
        return self.report

    def _task_ids(self) -> list[str]:
        return [r["id"] for r in self.observer.conn.execute("SELECT id FROM tasks ORDER BY rowid")]

    def record(self) -> None:
        """Audit entries committed since the last look go to the trace, so that it shows them
        in order with the faults (called after each step and before each fault)."""
        rows = self.observer.conn.execute(
            "SELECT * FROM audit_log WHERE seq > ? ORDER BY seq", (self.seen,)
        ).fetchall()
        for row in rows:
            self.seen = row["seq"]
            self.report.actions[row["action"]] += 1
            who = self.names.get(row["subtask_id"]) or self.names.get(row["task_id"], "?")
            moved = f" {row['from_status']}->{row['to_status']}" if row["to_status"] else ""
            line = f"{row['ts'][11:19]} {who} {row['actor']} {row['action']}{moved}"
            self.report.trace.append(line)

    def check(self) -> None:
        """Every invariant, after the run: what is wrong goes to `report.problems`."""
        self.record()
        store, report = self.observer, self.report
        for task_id in self._task_ids():
            name = self.names[task_id]
            status = store.task_status(task_id)
            report.statuses[name] = status
            if status not in ENDED:
                report.problems.append(f"{name} ended {status.value}")
            report.problems += [f"{name}: {p}" for p in check_history(store.conn, task_id)]
            for sub in store.list_subtasks(task_id):
                report.problems += self._check_subtask(store, sub.id, sub.description, sub.status)
        twice = [key for key, n in Counter(self.ledger).items() if n > 1]
        report.problems += [
            f"{self.names[sid]}: effect ran twice in execution {execution}"
            for sid, execution in twice
        ]

    def _check_subtask(
        self, store: Store, subtask_id: str, name: str, status: SubtaskStatus
    ) -> list[str]:
        accepted = store.conn.execute(
            """SELECT COUNT(*) FROM audit_log WHERE subtask_id = ? AND action = 'verdict'
               AND to_status = 'done'""",
            (subtask_id,),
        ).fetchone()[0]
        problems = []
        if accepted > 1:
            problems.append(f"{name} accepted {accepted} times")
        if status is SubtaskStatus.DONE:
            if accepted != 1:
                problems.append(f"{name} done without its verdict")
            if store.latest_artifact(subtask_id) != {"done": name}:
                problems.append(f"{name} done with {store.latest_artifact(subtask_id)}")
        return problems


def _body(request: Request) -> dict[str, Any]:
    """What the agent's first prompt holds ("Execute subtask:\\n{...}", "Review subtask:...")."""
    first = next(m.content for m in request.messages if m.role == "user")
    body: dict[str, Any] = json.loads(first.split(":\n", 1)[1])
    return body


@contextmanager
def _counted_ids() -> Iterator[None]:
    """Ids from a counter, not uuid4: the database's order of equal rows stays the same."""
    counter = itertools.count(1)
    with mock.patch.object(store_module, "_new_id", lambda: f"{next(counter):032x}"):
        yield


def simulate(
    directory: Path,
    seed: int,
    faults: Faults | None = None,
    shape: Shape | None = None,
    *,
    replay: str | None = None,
) -> Report:
    """One run in `directory` (its database and workspaces), checked: see `Report.problems`.
    `replay`: how to rerun it, for the report (see `Report.replay`)."""
    with _counted_ids():
        db_path = directory / "desman.db"
        sim = Simulation(db_path, seed, faults or Faults(), shape or Shape(), replay)
        try:
            return sim.run()
        finally:
            sim.observer.conn.close()
            for worker in sim.workers:
                worker.conn.close()  # a no-op for the dead ones
