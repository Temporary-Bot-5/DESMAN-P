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
import hashlib
import itertools
import json
import random
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict

from conftest import FakeClock
from desman import runs
from desman.agents import EXECUTE, REVIEW, TOOL_RESULTS, llm_executor, llm_reviewer
from desman.config import Donation, Network
from desman.core import db
from desman.core import store as store_module
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Success, Verdict
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
    builtin_registry,
)
from desman.core.workspace import Workspace
from desman.llm.base import Completion, Request
from desman.llm.fake import GARBAGE, FakeLLM, RateLimit, ServerError, StepLike, Timeout
from desman.net import protocol, sign
from desman.net.node import Node
from desman.registry import admin, clusters
from desman.registry import store as registry_store
from desman.registry import work as registry_work
from desman.registry.server import Registry as RegistryServer
from helpers import make_config

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


# ── desman-global (ROADMAP G.6c) ─────────────────────────────────────────────
#
# A registry and a few nodes, each with its own database, key and owner, work an operator's
# jobs through the real protocol (`httpx.MockTransport` into `Registry.handle`), the real
# `Node`, `net.jobs` and a worker's `Coordinator` with its network crew. One seeded generator
# decides which node moves and what goes wrong:
#
# - the registry cannot be reached for a while;
# - a node dies between two steps and starts again on its database (its held jobs, its key);
#   what it knew in memory — registration, trust — is gone;
# - the clock jumps, sometimes past a lease;
# - the model fails (garbage, timeouts, limits, 5xx) or the reviewer rejects;
# - the registry hides a revocation for a while (G.6c), and a node starts meanwhile: a node
#   whose trust list is not fresh must not take work.
#
# Then the faults stop and the nodes work until every job is settled. `check()`: every job
# accepted or waiting for the operator, or still lent only when no node may take it any more; an accepted job's result rechecked and
# accepted in its round by a node of another owner; everything handed in under the hander's
# own lease; no node lent one job twice; no lease taken while its node's trust list was not
# fresh; every node's task history replays, nothing of the network waits for the person, and
# no node still works on a job at the end.


@dataclass(frozen=True)
class NetFaults:
    down: float = 0.08
    """At each step, the registry goes unreachable for a few steps."""
    crash: float = 0.05
    """At each step, the node that moves dies first, then starts again."""
    jump: float = 0.1
    """At each step, the clock jumps up to 1.5 leases ahead."""
    provider: float = 0.2
    """At each model call, it fails."""
    reject: float = 0.1
    """At each review of another node's result, the reviewer rejects it."""
    hide: float = 0.04
    """At each step, the registry starts hiding revocations for a few steps."""


@dataclass(frozen=True)
class NetShape:
    jobs: int = 3
    """At the start; then the operator adds one at `arrive` per step, up to `max_jobs`."""
    arrive: float = 0.2
    max_jobs: int = 30
    nodes: int = 3
    steps: int = 150
    calm_steps: int = 400
    """After the faults: steps to settle every job, the clock moving a few minutes each."""


def _net_model(rng: random.Random, faults: NetFaults, report: Report, clock: FakeClock) -> FakeLLM:
    """Executor and reviewer of every node's crew: seeded failures, then a reply by role."""

    def respond(request: Request) -> StepLike:
        if rng.random() < faults.provider:
            kind = rng.choice(["garbage", "timeout", "rate limit", "server error"])
            report.faults[f"provider: {kind}"] += 1
            if kind == "garbage":
                return GARBAGE[rng.choice(sorted(GARBAGE))]
            if kind == "timeout":
                return Timeout(after=30)
            return RateLimit(retry_after=5) if kind == "rate limit" else ServerError()
        if not request.prompt.startswith(REVIEW):
            return '{"answer": "5"}'
        if rng.random() < faults.reject:
            report.faults["reject"] += 1
            return '{"verdict": "retry", "feedback": "It says 6"}'
        return '{"verdict": "accept"}'

    return FakeLLM(respond, repeat_last=True, clock=clock, sleep=clock.sleep)


class _NetSettings:
    def __init__(self, registry: str) -> None:
        self.config = make_config(
            language="en",
            configured=True,
            settings={"OLLAMA_MODEL": "llama3.2"},
            network=Network(
                enabled=True,
                registry=registry,
                donated={"ollama": Donation(calls_per_day=10_000, tokens_per_day=10_000_000)},
            ),
        )

    def __call__(self) -> Any:
        return self.config.model_copy(deep=True)


class _NetNode:
    def __init__(self, sim: NetSimulation, name: str, lives: int = 1) -> None:
        self.name = name
        self.home = sim.directory / name
        self.home.mkdir(exist_ok=True)
        self.lives = lives
        if not (self.home / "node.key").exists():
            sign.create_node_key(self.home / "node.key")
        self.conn = db.open_db(self.home / "desman.db")
        self.store = Store(self.conn, sim.clock)
        settings = _NetSettings(sim.address)
        crew = runs.network_crew(
            self.store, settings(), settings, given=runs.Models(sim.model, None), cache=False
        )
        assert crew is not None
        self.coordinator = Coordinator(
            self.store,
            lambda work: Success(content={}),
            lambda *_: Review(verdict=Verdict.ACCEPT),
            owner=f"{name}#{self.lives}",
            sleep=sim.clock.sleep,
            tools=builtin_registry(),
            network=crew,
        )
        self.node = Node(
            self.conn,
            settings,
            key_path=self.home / "node.key",
            transport=sim.transport,
            clock=sim.clock,
            root=sim.root_pub,
            work=True,
            cluster_model=crew.planner if isinstance(sim, ClusterSimulation) else None,
        )

    def close(self) -> None:
        self.node.close()
        self.conn.close()


class NetSimulation:
    def __init__(self, directory: Path, seed: int, faults: NetFaults, shape: NetShape) -> None:
        self.directory = directory
        self.rng = random.Random(seed)
        self.faults = faults
        self.shape = shape
        self.report = Report(
            seed, replay=f"DESMAN_FAULT_SEED={seed} pytest tests/test_faults.py -k network"
        )
        self.clock = FakeClock()
        self.address = "https://registry.test"
        root = sign.NodeKey.generate()
        self.root = root
        self.root_pub = directory / "root.pub"
        self.root_pub.write_bytes(sign.public_pem(root.public))
        self.registry_db = directory / "registry.db"
        registry_store.open_registry(self.registry_db).close()
        self.operator = sign.NodeKey.generate()
        with registry_store.connection(self.registry_db) as conn:
            admin.assign(
                conn, root, self.operator.public, {sign.Scope.TASKS}, days=30, now=self.clock()
            )
            # a revocation in the trust list, for the registry to hide
            admin.revoke(conn, root, sign.NodeKey.generate().fingerprint, "", now=self.clock())
        self.jobs = 0
        for _ in range(shape.jobs):
            self._add_job()
        self.registry = RegistryServer(self.registry_db, clock=self.clock, root=self.root_pub)
        self.down_for = 0
        self.hide_for = 0
        self.transport = httpx.MockTransport(self._handle)
        self.model = _net_model(self.rng, faults, self.report, self.clock)
        self.nodes = [_NetNode(self, f"n{i}") for i in range(shape.nodes)]

    def _add_job(self) -> None:
        job = protocol.Job(description=f"Sum 2 and 3 ({self.jobs})", acceptance_criteria="Says 5")
        with registry_store.connection(self.registry_db) as conn:
            admin.add_job(
                conn, self.operator, job, days=7, now=self.clock(), root_pub=self.root_pub
            )
        self.jobs += 1

    def _handle(self, req: Any) -> Any:
        if self.down_for > 0:
            raise httpx.ConnectError("unreachable", request=req)
        real = registry_store.handed_out

        def hiding(conn: Any, now: Any) -> list[sign.Signed]:
            return [s for s in real(conn, now) if s.record.kind is not sign.RecordKind.REVOCATION]

        with mock.patch.object(registry_store, "handed_out", hiding if self.hide_for > 0 else real):
            reply = self.registry.handle(
                req.method, req.url.raw_path.decode("ascii"), req.headers.multi_items(), req.content
            )
        return httpx.Response(reply.status, json=reply.body)

    def _leases(self, m: _NetNode) -> int:
        """Leases the registry ever granted to this node (0 before its key exists)."""
        path = m.home / "node.key"
        if not path.exists():
            return 0
        conn = db.connect(self.registry_db)
        try:
            found = conn.execute(
                "SELECT count(*) FROM leases WHERE node = ?",
                (sign.load_node_key(path).fingerprint,),
            ).fetchone()[0]
            return int(found)
        finally:
            conn.close()

    def _step(self, n: int, faulty: bool) -> None:
        f = self.faults
        if faulty:
            if self.jobs < self.shape.max_jobs and self.rng.random() < self.shape.arrive:
                self._add_job()
            if self.rng.random() < f.down:
                self.down_for = self.rng.randint(1, 6)
                self.report.faults["registry down"] += 1
            if self.rng.random() < f.hide:
                self.hide_for = self.rng.randint(4, 12)
                self.report.faults["revocation hidden"] += 1
                # a node starting meanwhile knows only what this registry tells it
                j = self.rng.randrange(len(self.nodes))
                self._restart(j)
            if self.rng.random() < f.jump:
                self.clock.advance(self.rng.uniform(0, 900))
                self.report.faults["clock jump"] += 1
        i = self.rng.randrange(len(self.nodes))
        if faulty and self.rng.random() < f.crash:
            self.report.faults["node crash"] += 1
            self._restart(i)
        m = self.nodes[i]
        before = self._leases(m)
        m.node.poll(idle=True)
        if self._leases(m) > before:
            trust = m.node._trust
            why = "no trust" if trust is None else trust.fresh()
            if why is not None:
                self.report.problems.append(f"{m.name} took work while not fresh: {why}")
        m.coordinator.work(until_idle=True, max_ticks=4)
        m.node.poll(idle=False)
        self.report.trace.append(f"{n}: {m.name}#{m.lives}")
        self.down_for = max(0, self.down_for - 1)
        self.hide_for = max(0, self.hide_for - 1)

    def _restart(self, i: int) -> None:
        old = self.nodes[i]
        old.close()
        self.nodes[i] = _NetNode(self, old.name, old.lives + 1)

    def _settled(self) -> bool:
        """No lease held, and no job left that some node may still take."""
        conn = db.connect(self.registry_db)
        try:
            if conn.execute("SELECT 1 FROM leases WHERE ended IS NULL").fetchone():
                return False
            lent = conn.execute("SELECT * FROM jobs WHERE status IN ('execute', 'recheck')")
            return not any(_takers(conn, job) for job in lent.fetchall())
        finally:
            conn.close()

    def run(self) -> Report:
        for n in range(self.shape.steps):
            self._step(n, faulty=True)
        self.down_for = self.hide_for = 0
        for n in range(self.shape.calm_steps):
            if self._settled():
                break
            self._step(self.shape.steps + n, faulty=False)
            self.clock.advance(120)
        for m in self.nodes:  # hand in whatever is finished
            m.node.poll(idle=False)
        self.check()
        return self.report

    def check(self) -> None:
        problems = self.report.problems
        conn = db.connect(self.registry_db)
        try:
            for job in conn.execute("SELECT * FROM jobs").fetchall():
                if job["status"] in ("execute", "recheck"):
                    if _takers(conn, job):
                        problems.append(f"job {job['id'][:8]} left {job['status']}")
                elif job["status"] not in ("accepted", "operator"):
                    problems.append(f"job {job['id'][:8]} left {job['status']}")
                if job["status"] != "accepted":
                    continue
                result = conn.execute(
                    "SELECT * FROM results WHERE id = ?", (job["result"],)
                ).fetchone()
                verdicts = conn.execute(
                    "SELECT r.*, coalesce(n.owner, n.fingerprint) AS by_owner FROM results r"
                    " JOIN nodes n ON n.fingerprint = r.node"
                    " WHERE r.job = ? AND r.kind = 'verdict' AND r.round = ? AND r.outcome = 'accept'",
                    (job["id"], job["round"]),
                ).fetchall()
                done_by = result["owner"] or result["node"]
                if not any(v["by_owner"] != done_by for v in verdicts):
                    problems.append(f"job {job['id'][:8]} accepted without another owner's verdict")
            for row in conn.execute(
                "SELECT r.id, r.node, r.kind, l.node AS holder, l.stage FROM results r"
                " JOIN leases l ON l.token = r.lease"
            ).fetchall():
                stage = "execute" if row["kind"] == "result" else "recheck"
                if row["node"] != row["holder"] or row["stage"] != stage:
                    problems.append(f"{row['kind']} {row['id'][:8]} handed in under another lease")
            for row in conn.execute(
                "SELECT job, node, count(*) AS n FROM leases GROUP BY job, node HAVING n > 1"
            ).fetchall():
                problems.append(f"job {row['job'][:8]} lent to one node {row['n']} times")
        finally:
            conn.close()
        for m in self.nodes:
            for (task_id,) in m.conn.execute("SELECT id FROM tasks").fetchall():
                for p in check_history(m.conn, task_id):
                    problems.append(f"{m.name}: {p}")
                for sub in m.store.list_subtasks(task_id):
                    if sub.status is SubtaskStatus.NEEDS_HUMAN:
                        problems.append(f"{m.name}: a network subtask waits for the person")
            working = m.conn.execute(
                "SELECT count(*) FROM network_work WHERE state = 'working'"
            ).fetchone()[0]
            if working:
                problems.append(f"{m.name} still works on {working} job(s)")

    def close(self) -> None:
        for m in self.nodes:
            m.close()


def _takers(conn: Any, job: Any) -> bool:
    """Some node may still be lent this job: one never lent it, and, for a recheck, not of the
    owner who did the work. Without one the job waits for its term (`expired`, journaled)."""
    found = conn.execute(
        "SELECT 1 FROM nodes n WHERE NOT EXISTS (SELECT 1 FROM leases"
        " WHERE leases.job = ? AND leases.node = n.fingerprint)"
        " AND (? = 'execute' OR coalesce(n.owner, n.fingerprint) != (SELECT"
        " coalesce(r.owner, r.node) FROM results r WHERE r.id = ?))",
        (job["id"], job["status"], job["result"]),
    ).fetchone()
    return found is not None


def simulate_network(
    directory: Path, seed: int, faults: NetFaults | None = None, shape: NetShape | None = None
) -> Report:
    sim = NetSimulation(directory, seed, faults or NetFaults(), shape or NetShape())
    try:
        return sim.run()
    finally:
        sim.close()


# ── Cluster operators (ROADMAP T.4d; ADR 0006 decision 11) ───────────────────


@dataclass(frozen=True)
class ClusterFaults(NetFaults):
    reject: float = 0.5
    operator: float = 0.3
    member: float = 0.6


@dataclass(frozen=True)
class ClusterShape:
    nodes: int = 5
    tasks: int = 2
    steps: int = 45
    calm_steps: int = 240


CLUSTER_PLAN = json.dumps(
    {
        "subtasks": [
            {"description": "Write a poem", "acceptance_criteria": "Four lines"},
            {
                "description": "Polish the poem",
                "acceptance_criteria": "Polished",
                "depends_on": [0],
            },
        ]
    }
)
CLUSTER_SPLIT = json.dumps(
    {
        "subtasks": [
            {"description": "First half", "acceptance_criteria": "First"},
            {"description": "Second half", "acceptance_criteria": "Second", "depends_on": [0]},
        ]
    }
)


def _cluster_planning(model: FakeLLM) -> FakeLLM:

    longer = json.loads(CLUSTER_PLAN)
    longer["subtasks"].append(
        {"description": "Combine the results", "acceptance_criteria": "Says 5", "depends_on": [1]}
    )
    return model.on("Plan task", CLUSTER_PLAN, json.dumps(longer), repeat_last=True).on(
        "Split subtask", CLUSTER_SPLIT, repeat_last=True
    )


class ClusterSimulation(NetSimulation):
    """Real registry/node loops; observations retain authority at receipt and lease time.

    Faults stop before a bounded recovery phase. Only the global operator's final acceptance
    is driven directly: appointments, plans, lending and results all use the node protocol.
    """

    def __init__(
        self, directory: Path, seed: int, faults: ClusterFaults, shape: ClusterShape
    ) -> None:
        super().__init__(directory, seed, faults, NetShape(jobs=0, max_jobs=0, nodes=shape.nodes))
        self.cluster_shape = shape
        self.cluster_faults = faults
        self.report.replay = f"DESMAN_FAULT_SEED={seed} pytest tests/test_faults.py -k cluster"
        appointer_path = directory / registry_store.APPOINTER_KEY
        appointer = sign.create_node_key(appointer_path)
        with registry_store.connection(self.registry_db) as conn:
            admin.assign(
                conn, self.root, appointer.public, {sign.Scope.APPOINT}, days=30, now=self.clock()
            )
        _cluster_planning(self.model)
        self.silent: dict[int, float] = {}
        self.old: list[tuple[int, clusters.AppointmentRow]] = []
        self.observed_leases: dict[str, clusters.AppointmentRow | None] = {}
        self.observed_records: set[str] = set()
        self.late_sent: set[str] = set()
        self.registry_seen = 0
        self.lease_seen = 0
        self.verdict_seen = 0
        self.record_seen = 0
        self.rechecked_by: dict[str, str | None] = {}
        for _ in range(shape.tasks):
            self._add_cluster()

    def _add_cluster(self) -> None:
        with registry_store.connection(self.registry_db) as conn:
            admin.add_cluster_task(
                conn,
                self.operator,
                protocol.ClusterTask(
                    goal=f"Sum 2 and 3 ({self.jobs})", acceptance_criteria="Says 5"
                ),
                days=7,
                now=self.clock(),
                root_pub=self.root_pub,
            )
        self.jobs += 1
        self.report.trace.append(f"task added {self.jobs}")

    def _handle(self, req: Any) -> Any:
        reply = super()._handle(req)
        # Observe after every request, before another poll can renew/replace an appointment.
        if hasattr(self, "observed_leases"):
            self.observe()
        return reply

    def observe(self) -> None:
        with registry_store.connection(self.registry_db) as conn:
            for event in conn.execute(
                "SELECT seq, action FROM registry_log WHERE seq > ? ORDER BY seq",
                (self.registry_seen,),
            ).fetchall():
                self.registry_seen = event["seq"]
                self.report.actions[event["action"]] += 1
                self.report.trace.append(f"registry: {event['action']}")
            for row in conn.execute(
                "SELECT l.rowid AS observation_id, l.*, j.cluster FROM leases l JOIN jobs j "
                "ON j.id=l.job WHERE j.cluster IS NOT NULL AND l.rowid > ? ORDER BY l.rowid",
                (self.lease_seen,),
            ).fetchall():
                self.lease_seen = row["observation_id"]
                task = clusters.cluster_task(conn, row["cluster"])
                assert task is not None
                appointment = clusters.newest(conn, task.cluster)
                self.observed_leases[row["token"]] = appointment
                if appointment is None or not (
                    appointment.issued_at <= self.clock() < appointment.expires_at
                ):
                    self.report.problems.append("lease without an appointment in force")
                    continue
                if row["node"] == appointment.operator:
                    self.report.problems.append("operator took a lease on its own cluster")
                if row["stage"] == "execute" and row["node"] not in appointment.members:
                    self.report.problems.append("executor was not an appointed member")
                if row["stage"] == "recheck":
                    result = conn.execute(
                        "SELECT r.node FROM results r JOIN jobs j ON j.result=r.id WHERE j.id=?",
                        (row["job"],),
                    ).fetchone()
                    if result is None or row["node"] in (result["node"], appointment.operator):
                        self.report.problems.append("recheck lent to executor or current operator")
            for verdict in conn.execute(
                "SELECT r.rowid AS observation_id, r.id, j.cluster FROM results r JOIN jobs j ON j.id=r.job "
                "WHERE r.kind='verdict' AND j.cluster IS NOT NULL AND r.rowid > ? ORDER BY r.rowid",
                (self.verdict_seen,),
            ).fetchall():
                self.verdict_seen = verdict["observation_id"]
                if verdict["id"] not in self.rechecked_by:
                    task = clusters.cluster_task(conn, verdict["cluster"])
                    assert task is not None
                    current = clusters.newest(conn, task.cluster)
                    self.rechecked_by[verdict["id"]] = current.operator if current else None
            for row in conn.execute(
                "SELECT * FROM cluster_records WHERE seq > ? ORDER BY seq", (self.record_seen,)
            ).fetchall():
                self.record_seen = row["seq"]
                self.observed_records.add(row["id"])
                a = clusters.newest(conn, row["cluster"])
                signer = registry_store.node(conn, row["signer"])
                if signer is None:
                    self.report.problems.append("cluster record has an unknown signer")
                else:
                    try:
                        sign.check_signature(
                            sign.Signed.model_validate_json(row["signed_json"]),
                            sign.decode_public(signer.public_key),
                        )
                    except sign.Refused:
                        self.report.problems.append("cluster record has a forged signature")
                if (
                    a is None
                    or row["signer"] != a.operator
                    or not (a.issued_at <= db.parse_iso(row["received_at"]) < a.expires_at)
                ):
                    self.report.problems.append("late cluster record accepted")
                self.report.actions[row["kind"]] += 1

    def _late(self) -> None:
        from time import monotonic

        from desman.net.node import _call, _Problem

        for i, old in self.old:
            if old.id in self.late_sent:
                continue
            with registry_store.connection(self.registry_db) as conn:
                current = clusters.newest(conn, old.cluster)
                task = clusters.by_cluster(conn, old.cluster)
            if current is None or current.operator == old.operator:
                continue
            assert task is not None
            m = self.nodes[i]
            key = sign.load_node_key(m.home / "node.key")
            signed = sign.sign(
                key,
                sign.RecordKind.CLUSTER_PLAN,
                protocol.ClusterPlan(
                    cluster=old.cluster,
                    task=task.id,
                    version=1,
                    nodes=[protocol.PlanNode(description="late", acceptance_criteria="x")],
                ).model_dump(mode="json"),
                issued_at=old.issued_at,
                lifetime=sign.APPOINTMENT_TERM,
            )
            self.late_sent.add(old.id)
            self.report.faults["late record"] += 1
            try:
                _call(
                    m.node._http(),
                    key,
                    self.address,
                    "POST",
                    protocol.CLUSTER_RECORDS,
                    protocol.ClusterHandIn(record=signed),
                    protocol.ClusterReceived,
                    self.clock(),
                    monotonic() + 10,
                )
            except _Problem as e:
                # Refusals pass through Registry.handle, which journals them.
                self.report.trace.append(f"late refused {e.error}")
                with registry_store.connection(self.registry_db) as conn:
                    if not conn.execute(
                        "SELECT 1 FROM registry_log WHERE action='refused' AND node=?",
                        (key.fingerprint,),
                    ).fetchone():
                        self.report.problems.append("late refusal not journaled")
            else:
                self.report.problems.append("late cluster record accepted")

    def _cluster_step(self, n: int, faulty: bool) -> None:
        f = self.cluster_faults
        before_faults = self.report.faults.copy()
        if faulty:
            if n == self.cluster_shape.steps // 2:
                self._add_cluster()
            if self.rng.random() < f.down:
                self.down_for = self.rng.randint(1, 3)
                self.report.faults["registry down"] += 1
            if self.rng.random() < f.jump:
                self.clock.advance(self.rng.uniform(1, 700))
                self.report.faults["clock jump"] += 1
            with registry_store.connection(self.registry_db) as conn:
                active = [
                    clusters.newest(conn, t.cluster)
                    for t in clusters.cluster_tasks(conn, "running")
                ]
                appointments = [
                    a
                    for a in active
                    if a is not None
                    and a.issued_at <= self.clock() < a.expires_at
                    and clusters.latest_plan(conn, a.cluster) is not None
                ]
                if appointments and not self.old and self.rng.random() < f.operator:
                    a = self.rng.choice(appointments)
                    i = next(
                        i
                        for i, m in enumerate(self.nodes)
                        if sign.load_node_key(m.home / "node.key").fingerprint == a.operator
                    )
                    self.old.append((i, a))
                    self.silent[i] = self.clock().timestamp() + 1900
                    self._restart(i)
                    self.report.faults["operator killed"] += 1
        i = self.rng.randrange(len(self.nodes)) if faulty else n % len(self.nodes)
        if self.silent.get(i, 0) > self.clock().timestamp():
            self.report.trace.append(f"{n}: n{i} silent")
        else:
            m = self.nodes[i]
            before = self._leases(m)
            m.node.poll(idle=True)
            if self._leases(m) > before:
                trust = m.node._trust
                if trust is None or trust.fresh() is not None:
                    self.report.problems.append(f"{m.name} took work while not fresh")
            # Stop after a real execute lease: its expiry must let another member take it.
            if (
                faulty
                and not self.report.faults["member gone"]
                and self.rng.random() < f.member
                and m.conn.execute(
                    "SELECT 1 FROM network_work WHERE state='working' AND stage='execute'"
                ).fetchone()
            ):
                self.silent[i] = self.clock().timestamp() + 750
                self.report.faults["member gone"] += 1
            if self.silent.get(i, 0) <= self.clock().timestamp():
                m.coordinator.work(until_idle=True, max_ticks=4)
                m.node.poll(idle=False)
            self.report.trace.append(f"{n}: {m.name}#{m.lives}")
        if not self.down_for:
            self._late()
        for kind, count in sorted((self.report.faults - before_faults).items()):
            self.report.trace.append(f"fault: {kind} ({count})")
        self.down_for = max(0, self.down_for - 1)
        self.clock.advance(120)

    def _settled(self) -> bool:
        with registry_store.connection(self.registry_db) as conn:
            return (
                all(
                    t.status in ("operator", "accepted", "dropped", "expired")
                    for t in clusters.cluster_tasks(conn)
                )
                and not conn.execute("SELECT 1 FROM leases WHERE ended IS NULL").fetchone()
            )

    def run(self) -> Report:
        for n in range(self.cluster_shape.steps):
            self._cluster_step(n, True)
        self.down_for = 0
        # Existing disappearances last their full duration, including into recovery.
        # Disable model faults too: the recovery phase must make progress.
        self.model = _cluster_planning(
            _net_model(self.rng, NetFaults(provider=0, reject=0), self.report, self.clock)
        )
        for i in range(len(self.nodes)):
            self._restart(i)
        for n in range(self.cluster_shape.calm_steps):
            self._cluster_step(self.cluster_shape.steps + n, False)
            if self._settled():
                # Every node must notice leases withdrawn when a task enters the inbox.
                for m in self.nodes:
                    m.node.poll(idle=False)
                if not any(
                    m.conn.execute("SELECT 1 FROM network_work WHERE state='working'").fetchone()
                    for m in self.nodes
                ):
                    break
        self.check()
        return self.report

    def check(self) -> None:
        self.observe()
        problems = self.report.problems
        with registry_store.connection(self.registry_db) as conn:
            finished = 0
            for t in clusters.cluster_tasks(conn):
                if t.status == "operator" and t.reason == "result":
                    try:
                        clusters.settle(conn, t.id, "accept", self.clock())
                        finished += 1
                    except clusters.SettleError as e:
                        problems.append(f"result could not be accepted: {e}")
                elif (
                    t.status == "operator"
                    and t.reason == "job_given_up"
                    and self.report.faults["reject"]
                ):
                    self.report.trace.append("inbox job_given_up explained by injected reject")
                else:
                    problems.append(f"cluster left {t.status}: {t.reason}")
            if not finished:
                problems.append("no cluster result accepted")
            accepted: Counter[tuple[str, int, str]] = Counter()
            for j in registry_work.jobs(conn, "accepted"):
                b = j.cluster_job
                assert b is not None
                plan = next(
                    p
                    for p in clusters.records(conn, b.cluster, sign.RecordKind.CLUSTER_PLAN)
                    if p.record.id == b.plan
                )
                content = protocol.ClusterPlan.model_validate(plan.record.body).nodes[b.node]
                accepted[(b.cluster, b.node, content.model_dump_json())] += 1
                result = conn.execute("SELECT node FROM results WHERE id=?", (j.result,)).fetchone()
                verdicts = conn.execute(
                    "SELECT * FROM results WHERE job=? AND kind='verdict' "
                    "AND outcome='accept' AND round=?",
                    (j.id, j.round),
                ).fetchall()
                valid = False
                for v in verdicts:
                    a = self.observed_leases.get(v["lease"])
                    operator = self.rechecked_by.get(v["id"])
                    if (
                        a
                        and result
                        and operator is not None
                        and v["node"] not in (result["node"], a.operator, operator)
                    ):
                        valid = True
                if not valid:
                    problems.append("accepted node without independent recheck")
            if any(n > 1 for n in accepted.values()):
                problems.append("plan node accepted twice")
        for m in self.nodes:
            for (task_id,) in m.conn.execute("SELECT id FROM tasks").fetchall():
                problems.extend(f"{m.name}: {p}" for p in check_history(m.conn, task_id))
                if any(
                    s.status is SubtaskStatus.NEEDS_HUMAN for s in m.store.list_subtasks(task_id)
                ):
                    problems.append(f"{m.name}: network waits for the person")
            if m.conn.execute("SELECT 1 FROM network_work WHERE state='working'").fetchone():
                problems.append(f"{m.name}: still working")


def simulate_clusters(
    directory: Path,
    seed: int,
    faults: ClusterFaults | None = None,
    shape: ClusterShape | None = None,
) -> Report:
    ids, keys = itertools.count(1), itertools.count(1)

    def key() -> sign.NodeKey:
        material = hashlib.sha256(f"cluster-test:{seed}:{next(keys)}".encode()).digest()
        return sign.NodeKey(Ed25519PrivateKey.from_private_bytes(material))

    with (
        _counted_ids(),
        mock.patch.object(uuid, "uuid4", lambda: uuid.UUID(int=next(ids))),
        mock.patch.object(sign.NodeKey, "generate", key),
    ):
        sim = ClusterSimulation(directory, seed, faults or ClusterFaults(), shape or ClusterShape())
        try:
            return sim.run()
        finally:
            sim.close()
