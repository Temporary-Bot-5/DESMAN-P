"""Jobs from desman-global on a node (ROADMAP G.6b): a task of origin `network` that never
outranks the person's own work, worked by the network's crew (lent models, quotas, the
`network_*` roles), handed in as a signed result or verdict. The registry is the real one, in
process, behind `httpx.MockTransport`, as in tests/test_node.py."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import FakeClock
from desman import runs
from desman.config import Config, ConfigError, Donation, Network
from desman.core import calls, db
from desman.core.coordinator import Coordinator, Crew, Failure, Review, Success, Verdict, Work
from desman.core.models import Origin, Priority, SubtaskSpec, TaskSpec
from desman.core.policy import Role
from desman.core.report import inbox
from desman.core.retry import FailureKind
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.tools import builtin_registry
from desman.llm.base import Message, QuotaExhausted, Request
from desman.llm.fake import FakeLLM
from desman.llm.openai_compat import CallRecord
from desman.net import jobs, node, protocol, sign
from desman.net.node import Change, Node
from desman.registry import admin, store, work
from desman.registry.server import Registry
from helpers import make_config

REGISTRY = "https://registry.test"
OLLAMA = {"OLLAMA_MODEL": "llama3.2"}
LENT = Donation(calls_per_day=100, tokens_per_day=100_000)
LEASE = work.LEASE.total_seconds()
DONE = '{"answer": "5"}'
ACCEPT = '{"verdict": "accept"}'
REJECT = '{"verdict": "retry", "feedback": "It says 6"}'


# ── The world: a registry, an operator, nodes ────────────────────────────────


@pytest.fixture
def root() -> sign.NodeKey:
    return sign.NodeKey.generate()


@pytest.fixture
def root_pub(tmp_path: Path, root: sign.NodeKey) -> Path:
    path = tmp_path / "root.pub"
    path.write_bytes(sign.public_pem(root.public))
    return path


@pytest.fixture
def registry_db(tmp_path: Path) -> Path:
    path = tmp_path / "registry" / store.DB_NAME
    path.parent.mkdir()
    store.open_registry(path).close()
    return path


class World:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.requests: list[tuple[str, str]] = []
        self.down = False

    def handle(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, req.url.path))
        if self.down:
            raise httpx.ConnectError("unreachable", request=req)
        reply = self.registry.handle(
            req.method, req.url.raw_path.decode("ascii"), req.headers.multi_items(), req.content
        )
        return httpx.Response(reply.status, json=reply.body)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


@pytest.fixture
def world(registry_db: Path, clock: FakeClock, root_pub: Path) -> World:
    return World(Registry(registry_db, clock=clock, root=root_pub))


@pytest.fixture
def operator(registry_db: Path, root: sign.NodeKey, clock: FakeClock) -> sign.NodeKey:
    key = sign.NodeKey.generate()
    with store.connection(registry_db) as conn:
        admin.assign(conn, root, key.public, {sign.Scope.TASKS}, days=30, now=clock())
    return key


@pytest.fixture
def add_job(
    registry_db: Path, operator: sign.NodeKey, clock: FakeClock, root_pub: Path
) -> Callable[..., str]:
    def add(**spec: Any) -> str:
        job = protocol.Job(
            description=spec.pop("description", "Sum 2 and 3"),
            acceptance_criteria=spec.pop("acceptance_criteria", "Says 5"),
            **spec,
        )
        with store.connection(registry_db) as conn:
            return admin.add_job(conn, operator, job, days=7, now=clock(), root_pub=root_pub).id

    return add


def job_status(registry_db: Path, job_id: str) -> tuple[str, str | None]:
    with store.connection(registry_db) as conn:
        found = work.job(conn, job_id)
    assert found is not None
    return found.status, found.reason


class Settings:
    def __init__(self, donation: Donation = LENT) -> None:
        self.config = make_config(
            language="en",
            configured=True,
            settings=OLLAMA,
            network=Network(enabled=True, registry=REGISTRY, donated={"ollama": donation}),
        )

        self.broken = False

    def __call__(self) -> Config:
        if self.broken:
            raise ConfigError(Path("config.json"), "does not load")
        return self.config.model_copy(deep=True)


def own_executor(work_: Work) -> Success:
    return Success(content={"own": work_.subtask.description})


def own_reviewer(*_: Any) -> Review:
    return Review(verdict=Verdict.ACCEPT)


class Machine:
    """One node: its database, key, config, the network's crew, a worker's coordinator."""

    def __init__(
        self,
        tmp_path: Path,
        name: str,
        world: World,
        clock: FakeClock,
        root_pub: Path,
        llm: FakeLLM,
    ) -> None:
        home = tmp_path / name
        home.mkdir()
        self.conn = db.open_db(home / "desman.db")
        self.store = Store(self.conn, clock)
        self.key = sign.create_node_key(home / "node.key")
        self.settings = Settings()
        crew = runs.network_crew(
            self.store, self.settings(), self.settings, given=runs.Models(llm, None)
        )
        assert crew is not None
        self.crew = crew
        self.coordinator = Coordinator(
            self.store,
            own_executor,
            own_reviewer,
            owner=name,
            sleep=clock.sleep,
            tools=builtin_registry(),
            network=crew,
        )
        self.node = Node(
            self.conn,
            self.settings,
            key_path=home / "node.key",
            transport=world.transport,
            clock=clock,
            root=root_pub,
            work=True,
        )

    def poll(self, *, idle: bool = True) -> node.Event | None:
        return self.node.poll(idle=idle)

    def work(self) -> None:
        self.coordinator.work(until_idle=True)

    def held(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM network_work ORDER BY created_at").fetchall()

    def log(self) -> list[tuple[str, dict[str, Any]]]:
        return [(e.action, e.details) for e in netlog_entries(self.conn)]

    def close(self) -> None:
        self.node.close()
        self.conn.close()


def netlog_entries(conn: sqlite3.Connection) -> list[Any]:
    from desman.core import netlog

    return netlog.entries(conn)


@pytest.fixture
def machine(
    tmp_path: Path, world: World, clock: FakeClock, root_pub: Path
) -> Iterator[Callable[[str, FakeLLM], Machine]]:
    made: list[Machine] = []

    def make(name: str, llm: FakeLLM) -> Machine:
        m = Machine(tmp_path, name, world, clock, root_pub, llm)
        made.append(m)
        return m

    yield make
    for m in made:
        m.close()


def doer() -> FakeLLM:
    return FakeLLM().on("Execute", DONE, repeat_last=True).on("Review", ACCEPT, repeat_last=True)


# ── The way through ──────────────────────────────────────────────────────────


def test_a_job_is_done_here_and_rechecked_by_another_node(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
    world: World,
    clock: FakeClock,
) -> None:
    job_id = add_job()
    a = machine("a", doer())
    b = machine("b", FakeLLM().on("Review", ACCEPT, repeat_last=True))

    taken = a.poll()
    assert taken is not None and taken.change is Change.WORK_TAKEN and taken.detail == "execute"
    [held] = a.held()
    assert held["job"] == job_id and held["state"] == "working"
    task_id = held["task_id"]
    assert a.store.origin(task_id) is Origin.NETWORK
    [sub] = a.store.list_subtasks(task_id)
    assert sub.priority == Priority.LOW and sub.description == "Sum 2 and 3"
    assert b.poll() is not None  # connected, but no work: one lease per job
    assert b.held() == []

    a.work()
    assert a.store.task_status(task_id) is TaskStatus.DONE
    handed = a.poll(idle=False)
    assert handed is not None and handed.change is Change.WORK_HANDED_IN
    assert handed.detail == "done"
    assert job_status(registry_db, job_id) == ("recheck", None)
    assert a.held()[0]["state"] == "handed_in"
    record = sign.Signed.model_validate_json(a.held()[0]["record_json"])
    assert record.record.signer == a.key.fingerprint
    assert json.loads(record.record.body["output"]) == {"answer": "5"}

    clock.advance(node.CLAIM_EVERY.total_seconds())
    checking = b.poll()
    assert checking is not None and checking.change is Change.WORK_TAKEN
    assert checking.detail == "recheck"
    b.work()
    verdict = b.poll(idle=False)
    assert verdict is not None and verdict.detail == "accept"
    assert job_status(registry_db, job_id) == ("accepted", None)

    assert [action for action, _ in a.log()].count("work_taken") == 1
    assert "handed_in" in [action for action, _ in b.log()]
    assert ("POST", protocol.HAND_IN) in world.requests


def test_a_rejected_recheck_goes_to_the_operator_and_waits_for_nobody_here(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
) -> None:
    job_id = add_job()
    a = machine("a", doer())
    b = machine("b", FakeLLM().on("Review", REJECT, repeat_last=True))
    a.poll()
    a.work()
    a.poll(idle=False)
    b.poll()
    b.work()
    task_id = b.held()[0]["task_id"]
    # One attempt: the reviewer's rejection is the verdict, never a retry of someone's work.
    assert b.store.task_status(task_id) is TaskStatus.NEEDS_HUMAN
    assert inbox(b.store) == []  # a job never waits for the person
    sent = b.poll(idle=False)
    assert sent is not None and sent.detail == "reject"
    assert job_status(registry_db, job_id) == ("operator", "disputed")
    assert b.store.task_status(task_id) is TaskStatus.CANCELLED
    body = sign.Signed.model_validate_json(b.held()[0]["record_json"]).record.body
    assert body["accept"] is False and body["reason"] == "It says 6"


def test_a_job_its_models_cannot_do_is_handed_in_as_failed(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
) -> None:
    job_id = add_job()
    a = machine("a", FakeLLM().on("Execute", '{"type": "error", "reason": "no"}', repeat_last=True))
    a.poll()
    a.work()
    sent = a.poll(idle=False)
    assert sent is not None and sent.detail == "failed"
    assert job_status(registry_db, job_id) == ("execute", None)  # another node may try
    assert a.store.task_status(a.held()[0]["task_id"]) is TaskStatus.CANCELLED


def test_a_record_is_signed_once_and_sent_again_until_the_registry_answers(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
    world: World,
) -> None:
    job_id = add_job()
    a = machine("a", doer())
    a.poll()
    a.work()
    world.down = True
    event = a.poll(idle=False)
    assert event is not None and event.change is Change.UNREACHABLE
    first = a.held()[0]["record_json"]
    assert first is not None and a.held()[0]["state"] == "working"

    world.down = False
    assert a.node.next_at is not None
    a.node.next_at = None  # the backoff is over
    a.node.failures = 0
    event = a.poll(idle=False)
    assert event is not None and event.change is Change.WORK_HANDED_IN
    assert a.held()[0]["record_json"] == first
    assert job_status(registry_db, job_id) == ("recheck", None)


# ── Leases ───────────────────────────────────────────────────────────────────


def test_the_lease_is_extended_while_the_job_is_worked_on(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    clock: FakeClock,
) -> None:
    add_job()
    a = machine("a", doer())
    a.poll()
    before = db.parse_iso(a.held()[0]["expires_at"])
    clock.advance(LEASE / 2 - 1)
    assert a.poll(idle=False) is None  # more than EXTEND_WHEN left: not yet
    clock.advance(2 * 60)
    assert a.poll(idle=False) is None
    after = db.parse_iso(a.held()[0]["expires_at"])
    assert after > before and after == clock().replace(microsecond=0) + work.LEASE


def test_a_lease_lost_drops_the_job(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    clock: FakeClock,
) -> None:
    add_job()
    a = machine("a", doer())
    a.poll()
    task_id = a.held()[0]["task_id"]
    clock.advance(LEASE + 1)
    dropped = a.poll(idle=False)
    assert dropped is not None and dropped.change is Change.WORK_DROPPED
    assert dropped.detail == node.LEASE_LOST
    assert a.held()[0]["state"] == "lost"
    assert a.store.task_status(task_id) is TaskStatus.CANCELLED
    assert "work_dropped" in [action for action, _ in a.log()]


def test_off_drops_what_is_held_without_a_word(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    world: World,
) -> None:
    add_job()
    a = machine("a", doer())
    a.poll()
    task_id = a.held()[0]["task_id"]
    a.settings.config.network.enabled = False
    sent = len(world.requests)
    a.poll(idle=False)
    assert len(world.requests) == sent
    assert a.held()[0]["state"] == "abandoned"
    assert a.store.task_status(task_id) is TaskStatus.CANCELLED


# ── Asking for work ──────────────────────────────────────────────────────────


def test_work_is_asked_for_only_when_idle_and_not_too_often(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    world: World,
    clock: FakeClock,
) -> None:
    a = machine("a", doer())
    assert a.poll(idle=False) is not None  # connected
    assert ("POST", protocol.CLAIM) not in world.requests
    a.poll()
    assert world.requests.count(("POST", protocol.CLAIM)) == 1  # nothing offered
    add_job()
    a.poll()
    assert world.requests.count(("POST", protocol.CLAIM)) == 1  # not again so soon
    clock.advance(node.CLAIM_EVERY.total_seconds())
    taken = a.poll()
    assert taken is not None and taken.change is Change.WORK_TAKEN
    clock.advance(node.CLAIM_EVERY.total_seconds())
    a.poll()
    assert world.requests.count(("POST", protocol.CLAIM)) == 2  # one job at a time


def test_a_worker_without_a_crew_asks_for_nothing(
    tmp_path: Path, world: World, clock: FakeClock, root_pub: Path, add_job: Callable[..., str]
) -> None:
    add_job()
    conn = db.open_db(tmp_path / "desman.db")
    sign.create_node_key(tmp_path / "node.key")
    n = Node(
        conn,
        Settings(),
        key_path=tmp_path / "node.key",
        transport=world.transport,
        clock=clock,
        root=root_pub,
    )
    n.poll(idle=True)
    assert ("POST", protocol.CLAIM) not in world.requests
    n.close()
    conn.close()


def test_no_work_is_asked_for_once_the_day_s_quota_is_spent(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    world: World,
    clock: FakeClock,
) -> None:
    add_job()
    a = machine("a", doer())
    a.settings.config.network.donated["ollama"] = Donation(calls_per_day=1, tokens_per_day=1000)
    spent = network_task(a.store)
    a.conn.execute(
        "INSERT INTO model_calls (at, provider, model, task_id, prompt_hash, latency_ms)"
        " VALUES (?, 'ollama', 'llama3.2', ?, 'h', 1)",
        (db.to_iso(clock()), spent),
    )
    a.poll()
    assert ("POST", protocol.CLAIM) not in world.requests
    clock.advance(24 * 3600)  # a new day
    a.poll()
    assert ("POST", protocol.CLAIM) in world.requests


def network_task(s: Store) -> str:
    return s.create_task(
        TaskSpec(
            goal="g",
            priority=Priority.LOW,
            subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")],
        ),  # fmt: skip
        origin=Origin.NETWORK,
    )


def test_a_lent_model_is_refused_past_its_quota(clock: FakeClock, store: Store) -> None:
    settings = Settings(Donation(calls_per_day=1, tokens_per_day=1000))
    llm = FakeLLM(DONE, DONE)
    quota = jobs.Quota(llm, store.conn, "ollama", settings, clock)
    request = Request(messages=[Message(role="user", content="hi")])
    assert quota.complete(request).text == DONE
    task = network_task(store)
    store.conn.execute(
        "INSERT INTO model_calls (at, provider, model, task_id, prompt_hash, latency_ms)"
        " VALUES (?, 'ollama', 'llama3.2', ?, 'h', 1)",
        (db.to_iso(clock()), task),
    )
    with pytest.raises(QuotaExhausted) as raised:
        quota.complete(request)
    assert raised.value.resets_at == jobs.day_start(clock()) + timedelta(days=1)
    settings.config.network.donated.clear()  # no longer lent
    clock.advance(24 * 3600)
    with pytest.raises(QuotaExhausted):
        quota.complete(request)
    # The person's own calls never count against what they lent.
    assert jobs.used_today(store.conn, "ollama", clock()) == (0, 0)


# ── What a node accepts ──────────────────────────────────────────────────────


def test_a_record_seen_once_is_refused_after_a_restart(
    tmp_path: Path, clock: FakeClock, root: sign.NodeKey, operator: sign.NodeKey
) -> None:
    conn = db.open_db(tmp_path / "desman.db")
    assignment = sign.assign(
        root, operator.public, {sign.Scope.TASKS}, issued_at=clock(), lifetime=timedelta(days=30)
    )
    task = sign.sign(
        operator,
        sign.RecordKind.TASK,
        protocol.Job(description="d", acceptance_criteria="c").model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )

    def trust() -> sign.Trust:
        t = sign.Trust(root.public, clock=clock, seen=jobs.Seen(conn))
        t.receive(assignment.model_dump(mode="json"))
        return t

    trust().receive(task.model_dump(mode="json"))
    with pytest.raises(sign.Refused) as refused:
        trust().receive(task.model_dump(mode="json"))  # another process, the same database
    assert refused.value.reason is sign.Reason.REPLAY
    clock.advance(2 * 24 * 3600)
    assert jobs.Seen(conn).add("ab" * 16, clock() + timedelta(days=1), clock())
    assert conn.execute("SELECT count(*) FROM seen_records").fetchone()[0] == 1  # expired go
    conn.close()


def test_a_recheck_s_result_must_be_signed_by_the_key_that_came_with_it(
    clock: FakeClock, operator: sign.NodeKey
) -> None:
    doer_key, stranger = sign.NodeKey.generate(), sign.NodeKey.generate()
    task = sign.sign(
        operator,
        sign.RecordKind.TASK,
        protocol.Job(description="d", acceptance_criteria="c").model_dump(mode="json"),
        issued_at=clock(),
        lifetime=timedelta(days=1),
    )
    lease = "0" * 32

    def lent(result: sign.Signed, key: sign.NodeKey) -> protocol.Work:
        return protocol.Work(
            lease=lease,
            expires_at=clock() + work.LEASE,
            stage="recheck",
            task=task,
            result=result,
            result_key=sign.encode_public(key.public),
        )

    def result(key: sign.NodeKey, **body: Any) -> sign.Signed:
        content = {"job": task.record.id, "lease": "1" * 32, "outcome": "done"} | body
        return sign.sign(
            key, sign.RecordKind.RESULT, content, issued_at=clock(), lifetime=timedelta(days=1)
        )

    node._check_result(lent(result(doer_key), doer_key), task.record.id, clock())
    with pytest.raises(sign.Refused):
        node._check_result(lent(result(stranger), doer_key), task.record.id, clock())
    with pytest.raises(sign.Refused):
        node._check_result(lent(result(doer_key, job="ab" * 16), doer_key), task.record.id, clock())
    with pytest.raises(sign.Refused):
        node._check_result(
            lent(result(doer_key, outcome="failed"), doer_key), task.record.id, clock()
        )
    with pytest.raises(ValueError, match="recheck"):
        protocol.Work(lease=lease, expires_at=clock(), stage="recheck", task=task, result=None)


def test_a_job_s_tools_are_the_network_roles(
    machine: Callable[[str, FakeLLM], Machine],
) -> None:
    a = machine("a", doer())
    a.settings.config.permissions.network_access = True
    crew = runs.network_crew(a.store, a.settings(), a.settings, given=runs.Models(FakeLLM(), None))
    assert crew is not None and crew.access is not None
    assert crew.access.role is Role.NETWORK_EXECUTOR
    tools = builtin_registry()

    def tool(name: str) -> Any:
        found = tools.get(name)
        assert found is not None
        return found

    for name in ("web_search", "web_fetch", "suggest_model"):
        refusal = crew.access.refusal(tool(name))
        assert refusal is not None and "tool policy" in refusal
    assert crew.access.refusal(tool("write_file")) is None


def test_no_crew_while_off_or_nothing_lent(store: Store) -> None:
    off = Settings()
    off.config.network.enabled = False
    assert runs.network_crew(store, off(), off, given=runs.Models(FakeLLM(), None)) is None
    nothing = Settings()
    nothing.config.network.donated.clear()
    assert runs.network_crew(store, nothing(), nothing, given=runs.Models(FakeLLM(), None)) is None


# ── The person's own work first ──────────────────────────────────────────────


def own_task(s: Store, priority: int = Priority.LOW) -> str:
    return s.create_task(
        TaskSpec(
            goal="mine",
            priority=priority,
            subtasks=[SubtaskSpec(description="mine", acceptance_criteria="c")],
        ),  # fmt: skip
    )


def test_a_job_never_outranks_the_person_s_work(store: Store, clock: FakeClock) -> None:
    job = network_task(store)
    clock.advance(24 * 3600)  # however long it waited
    [waiting] = store.list_subtasks(job)
    assert store.urgency(waiting) == Priority.LOW
    mine = own_task(store, priority=-10)  # even below LOW, just made
    claimed = store.claim("w")
    assert claimed is not None and claimed.task_id == mine
    claimed = store.claim("w")
    assert claimed is not None and claimed.task_id == job


def test_a_job_s_priority_is_capped_at_low(store: Store, clock: FakeClock) -> None:
    high = store.create_task(
        TaskSpec(
            goal="g",
            priority=Priority.CRITICAL,
            subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")],
        ),  # fmt: skip
        origin=Origin.NETWORK,
    )
    [sub] = store.list_subtasks(high)
    assert store.urgency(sub) == Priority.LOW


def test_a_run_without_a_crew_never_takes_a_job(store: Store) -> None:
    job = network_task(store)
    assert store.claim("w", network=False) is None
    coordinator = Coordinator(store, own_executor, own_reviewer)
    coordinator.work(until_idle=True)
    assert store.task_status(job) is TaskStatus.ACTIVE
    assert store.list_subtasks(job)[0].status is SubtaskStatus.READY


def test_the_person_s_work_takes_over_from_a_job(store: Store, clock: FakeClock) -> None:
    job = network_task(store)
    claimed = store.claim("w")
    assert claimed is not None and claimed.task_id == job
    assert claimed.lease_token is not None
    assert store.preempt(claimed.id, claimed.lease_token, "w") is None  # nothing waits
    mine = own_task(store, priority=-10)
    took = store.preempt(claimed.id, claimed.lease_token, "w")
    assert took is not None and took.task_id == mine
    assert store.get_subtask(claimed.id).status is SubtaskStatus.READY
    # and never the other way round
    assert took.lease_token is not None
    other = network_task(store)
    assert store.preempt(took.id, took.lease_token, "w") is None
    assert store.list_subtasks(other)[0].status is SubtaskStatus.READY


def test_a_worker_runs_the_person_s_work_before_a_job(
    machine: Callable[[str, FakeLLM], Machine], add_job: Callable[..., str]
) -> None:
    add_job()
    a = machine("a", doer())
    a.poll()
    mine = own_task(a.store, priority=-10)
    order: list[str] = []
    a.coordinator.watch = None
    original = a.crew.executor

    def network_executor(work_: Work) -> Any:
        order.append("job")
        return original(work_)

    def mine_first(work_: Work) -> Success:
        order.append("mine")
        return own_executor(work_)

    a.coordinator.executor = mine_first
    a.coordinator.network = Crew(
        network_executor, a.crew.reviewer, a.crew.access, a.crew.strategies, a.crew.providers
    )
    a.work()
    assert order == ["mine", "job"]
    assert a.store.task_status(mine) is TaskStatus.DONE


def test_reviews_of_the_person_s_work_come_before_a_job_s(store: Store) -> None:
    job, mine = network_task(store), own_task(store, priority=-10)
    for task in (job, mine):
        sub = store.claim("w", task_id=task)
        assert sub is not None and sub.lease_token is not None
        store.submit(sub.id, sub.lease_token, {"r": 1})
    first = store.pending_reviews()[0]
    assert first.task_id == mine
    assert [s.task_id for s in store.pending_reviews(network=False)] == [mine]


# ── The database ─────────────────────────────────────────────────────────────


def test_a_task_keeps_its_origin_and_only_a_job_holds_network_work(store: Store) -> None:
    mine = own_task(store)
    with pytest.raises(sqlite3.IntegrityError, match="origin"):
        store.conn.execute("UPDATE tasks SET origin = 'network' WHERE id = ?", (mine,))
    with pytest.raises(sqlite3.IntegrityError, match="network task"):
        store.conn.execute(
            "INSERT INTO network_work (lease, registry, job, stage, task_id, task_json,"
            " expires_at) VALUES ('l', 'r', 'j', 'execute', ?, '{}', 'now')",
            (mine,),
        )
    audit = store.audit_log(network_task(store))
    assert json.loads(audit[0]["details_json"])["origin"] == "network"


def test_an_older_database_upgrades_with_every_task_its_own(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    conn = db.connect(path)
    db.migrate(conn, db.MIGRATIONS[:27])
    conn.execute(
        "INSERT INTO tasks (id, goal, priority, created_at, max_attempts, max_transient)"
        " VALUES ('t', 'g', 0, '2026-01-01T00:00:00.000Z', 3, 3)"
    )
    db.migrate(conn)
    assert conn.execute("SELECT origin FROM tasks").fetchone()[0] == "local"
    conn.close()


def test_a_database_with_network_work_gets_quota_holds(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "28.db")
    db.migrate(conn, db.MIGRATIONS[:28])
    db.migrate(conn)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT count(*) FROM quota_holds").fetchone()[0] == 0
    conn.close()


def test_a_failed_review_of_a_recheck_hands_in_nothing(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
) -> None:
    job_id = add_job()
    a = machine("a", doer())
    a.poll()
    a.work()
    a.poll(idle=False)
    b = machine("b", FakeLLM())
    b.poll()
    task_id = b.held()[0]["task_id"]
    [sub] = b.store.list_subtasks(task_id)
    b.coordinator.network = Crew(
        b.crew.executor,
        lambda *_: Failure(kind=FailureKind.PERMANENT, reason="no model answers"),
        b.crew.access,
    )
    b.work()
    assert b.store.task_status(task_id) is TaskStatus.NEEDS_HUMAN
    dropped = b.poll(idle=False)
    assert dropped is not None and dropped.change is Change.WORK_DROPPED
    assert b.held()[0]["state"] == "abandoned" and b.held()[0]["record_json"] is None
    assert job_status(registry_db, job_id) == ("recheck", None)  # its lease will run out
    assert sub.id


def test_resume_refuses_a_job(tmp_path: Path, store: Store) -> None:
    from click.testing import CliRunner

    from desman.cli import cli

    path = tmp_path / "jobs.db"
    s = Store(db.open_db(path))
    task = network_task(s)
    s.conn.close()
    script = tmp_path / "script.json"
    script.write_text('{"steps": []}')
    result = CliRunner().invoke(
        cli, ["resume", task[:8], "--db", str(path), "--fake-llm", str(script)]
    )
    assert result.exit_code != 0
    assert "desman worker" in result.output


# ── Review fixes (Devin, PR #79) ─────────────────────────────────────────────


class Slow:
    """A lent model that takes `minutes` per call, recorded as a real client records one (so
    `calls.watching` sees it)."""

    def __init__(self, llm: FakeLLM, conn: sqlite3.Connection, clock: FakeClock, minutes: int):
        self.llm = llm
        self.name = llm.name
        self.record = calls.recorder(conn)
        self.clock = clock
        self.minutes = minutes

    def complete(self, request: Request) -> Any:
        self.clock.advance(self.minutes * 60)
        reply = self.llm.complete(request)
        self.record(
            CallRecord(
                provider="ollama",
                model="llama3.2",
                key_id=None,
                prompt_hash="h",
                response_hash="r",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1,
                http_status=200,
                error_kind=None,
                error=None,
            )  # fmt: skip
        )
        return reply


def tend(m: Machine) -> None:
    m.poll(idle=False)


def test_the_lease_is_extended_between_the_model_calls_of_one_step(
    tmp_path: Path,
    world: World,
    clock: FakeClock,
    root_pub: Path,
    add_job: Callable[..., str],
    registry_db: Path,
) -> None:
    """Execution and review take six minutes each, the lease ten: only extended while the
    models are asked (as `desman worker` does) is the result still taken."""
    job_id = add_job()
    a = Machine(tmp_path, "a", world, clock, root_pub, doer())
    slow = Slow(doer(), a.conn, clock, minutes=6)
    crew = runs.network_crew(a.store, a.settings(), a.settings, given=runs.Models(slow, None))
    assert crew is not None
    a.coordinator.network = crew
    try:
        a.poll()
        with calls.watching(lambda: tend(a)):
            a.work()
        handed = a.poll(idle=False)
        assert handed is not None and handed.change is Change.WORK_HANDED_IN
        assert job_status(registry_db, job_id) == ("recheck", None)
        assert world.requests.count(("POST", protocol.EXTEND)) >= 2
    finally:
        a.close()


def test_two_workers_never_take_the_last_call_of_a_quota(clock: FakeClock, store: Store) -> None:
    settings = Settings(Donation(calls_per_day=1, tokens_per_day=1000))
    request = Request(messages=[Message(role="user", content="hi")])
    second = jobs.Quota(FakeLLM(DONE), store.conn, "ollama", settings, clock)
    refused: list[BaseException] = []

    class Meanwhile:
        """The first worker's call: while it is out, the other worker asks."""

        name = "fake"

        def complete(self, _: Request) -> Any:
            try:
                second.complete(request)
            except QuotaExhausted as e:
                refused.append(e)
            return FakeLLM(DONE).complete(request)

    first = jobs.Quota(Meanwhile(), store.conn, "ollama", settings, clock)
    assert first.complete(request).text == DONE
    assert len(refused) == 1
    assert store.conn.execute("SELECT count(*) FROM quota_holds").fetchone()[0] == 0


def test_a_hold_left_by_a_dead_worker_runs_out(clock: FakeClock, store: Store) -> None:
    settings = Settings(Donation(calls_per_day=1, tokens_per_day=1000))
    assert jobs.hold_call(store.conn, settings(), "ollama", clock()) is not None  # never released
    assert jobs.hold_call(store.conn, settings(), "ollama", clock()) is None
    clock.advance(jobs.HOLD_FOR.total_seconds())
    assert jobs.hold_call(store.conn, settings(), "ollama", clock()) is not None


def test_a_result_holds_as_long_as_its_job_and_a_late_recheck_accepts_it(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
    registry_db: Path,
    clock: FakeClock,
) -> None:
    job_id = add_job()  # seven days
    a = machine("a", doer())
    b = machine("b", FakeLLM().on("Review", ACCEPT, repeat_last=True))
    a.poll()
    a.work()
    a.poll(idle=False)
    record = sign.Signed.model_validate_json(a.held()[0]["record_json"]).record
    task = sign.Signed.model_validate_json(a.held()[0]["task_json"]).record
    assert record.expires_at >= task.expires_at
    clock.advance(3 * 24 * 3600)  # the other node comes days later
    checking = b.poll()
    assert checking is not None and checking.change is Change.WORK_TAKEN
    b.work()
    b.poll(idle=False)
    assert job_status(registry_db, job_id) == ("accepted", None)


def test_an_until_idle_worker_does_the_job_it_took_before_it_stops(
    machine: Callable[[str, FakeLLM], Machine],
    add_job: Callable[..., str],
) -> None:
    add_job()
    a = machine("a", doer())

    def idle() -> bool:
        event = a.poll()
        return event is not None and event.change is Change.WORK_TAKEN

    a.coordinator.work(until_idle=True, on_idle=idle)
    [held] = a.held()
    assert a.store.task_status(held["task_id"]) is TaskStatus.DONE


def test_a_permission_turned_off_holds_for_a_running_worker_s_jobs(
    machine: Callable[[str, FakeLLM], Machine],
) -> None:
    a = machine("a", doer())
    a.settings.config.permissions.shell_execution = True
    crew = runs.network_crew(a.store, a.settings(), a.settings, given=runs.Models(FakeLLM(), None))
    assert crew is not None and crew.access is not None
    run = builtin_registry().get("run_command")
    assert run is not None
    assert crew.access.refusal(run) is None
    a.settings.config.permissions.shell_execution = False  # while the worker runs
    refusal = crew.access.refusal(run)
    assert refusal is not None and "shell_execution" in refusal


def test_a_config_that_fails_turns_every_switch_off(
    machine: Callable[[str, FakeLLM], Machine],
) -> None:
    a = machine("a", doer())
    a.settings.config.permissions.shell_execution = True
    crew = runs.network_crew(a.store, a.settings(), a.settings, given=runs.Models(FakeLLM(), None))
    assert crew is not None and crew.access is not None
    run = builtin_registry().get("run_command")
    assert run is not None
    a.settings.broken = True
    assert crew.access.refusal(run) is not None


def test_a_call_out_at_midnight_counts_against_the_new_day(clock: FakeClock, store: Store) -> None:
    """It is logged when it ends, today: the quota never lets one more through meanwhile."""
    settings = Settings(Donation(calls_per_day=1, tokens_per_day=1000))
    clock.advance((jobs.day_start(clock()) + timedelta(days=1) - clock()).total_seconds() - 1)
    hold = jobs.hold_call(store.conn, settings(), "ollama", clock())  # 23:59:59
    assert hold is not None
    clock.advance(2)  # a new day, the call still out
    assert jobs.hold_call(store.conn, settings(), "ollama", clock()) is None
    jobs.release_call(store.conn, hold)
    assert jobs.hold_call(store.conn, settings(), "ollama", clock()) is not None
