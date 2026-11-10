"""Jobs from desman-global on this node (ADR 0004, decisions 5–7; ROADMAP G.6b).

The registry lends a job (`protocol.Work`): a `task` record an operator signed, to be done
(stage `execute`) or, with another node's signed result, to be rechecked (`recheck`). The node
(`net.node.Node`) checks the record against the root (`sign.Trust`; the record ids it has seen
are kept in the database, `Seen`, so a replay is refused after a restart too) and takes it here:

- It becomes an ordinary task of origin `network` (`take`): one subtask at priority LOW, a
  retry budget of its own, its own workspace, with the lease, the record and (for a recheck)
  the result kept in `network_work` in the same transaction. The worker's coordinator runs it
  with the network's crew (`runs.network_crew`): only the models the person lent, each within
  its day's quota (`Quota`, counted from `model_calls` and the calls being made, each held in
  the transaction that checks it), the tool policy's `network_*` roles.
  The person's own work goes first and takes over from it (`core.scheduler`).
- A recheck's execution is the other node's result as it came (`executor`): the work is the
  review. The reviewer judges it without the evidence of tool calls (they stayed on the node
  that made them), and with one attempt only: a rejection is the verdict.
- Its end is read from the task (`ending`): done → a `result` with the output and the files of
  its workspace (as many as fit), or for a recheck a `verdict` accepting it; the reviewer's
  rejection of a recheck → a verdict rejecting it, with why; anything else that stops it (its
  retries ran out, it would wait for a person, a pause longer than the lease) → a `failed`
  result, or for a recheck nothing (the node could not judge: its lease runs out and another
  node gets the job). The record is signed once and kept (`keep`), and sent again as it is
  until the registry answers.
- Nothing of it waits for the person: once it ends, a job still open is cancelled (`close`), and
  the inbox never lists a job (`core.report.inbox`).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ValidationError

from desman.config import Config
from desman.core import db
from desman.core.coordinator import Executor, Reviewer, Success, Work
from desman.core.models import Origin, Priority, Subtask, SubtaskSpec, TaskSpec, Tools
from desman.core.retry import RetryPolicy
from desman.core.states import SubtaskStatus, TaskStatus
from desman.core.store import Store
from desman.core.workspace import PathEscapeError, Workspace
from desman.llm.base import LLM, Completion, QuotaExhausted, Request
from desman.net import protocol, sign

EXECUTE_RETRY: Final = RetryPolicy(max_attempts=2, max_transient=3)
RECHECK_RETRY: Final = RetryPolicy(max_attempts=1, max_transient=3)
"""One attempt: the reviewer's rejection of the other node's result is the verdict."""
RECORD_LIFETIME: Final = timedelta(days=1)
"""The least term of a result or verdict this node signs: long enough to be sent again for a
while. A result lasts as long as its job at least (`record_lifetime`)."""
HOLD_FOR: Final = timedelta(minutes=30)
"""A call held of a quota counts this long at most: longer than any call takes, so a hold is
left behind only by a worker that died during the call."""
GOAL_CHARS: Final = 120
MAX_FILES: Final = 16
RECHECK: Final = "recheck"
EXECUTE: Final = "execute"

State = Literal["working", "handed_in", "lost", "abandoned"]


class Seen:
    """`sign.Seen` in the node's database: a record id once, forgotten when its record expired
    (it is refused as expired from then on anyway)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._inspecting = False

    @contextmanager
    def inspect(self) -> Iterator[None]:
        """Verify a signed cluster snapshot without reserving its jobs as taken.

        The same Trust still checks signatures, revocations and terms. Only its replay
        reservation is omitted: an operator reading a job has not executed/rechecked it.
        """
        before = self._inspecting
        self._inspecting = True
        try:
            yield
        finally:
            self._inspecting = before

    def add(self, id: str, expires_at: datetime, now: datetime) -> bool:
        if self._inspecting:
            return True
        with db.transaction(self.conn):
            self.conn.execute("DELETE FROM seen_records WHERE expires_at <= ?", (db.to_iso(now),))
            added = self.conn.execute(
                "INSERT OR IGNORE INTO seen_records (id, expires_at) VALUES (?, ?)",
                (id, db.to_iso(expires_at)),
            ).rowcount
        return added == 1


@dataclass(frozen=True)
class Held:
    """A job this node works on (a `network_work` row)."""

    lease: str
    registry: str
    job: str
    stage: protocol.Stage
    task_id: str
    task: sign.Signed
    result: sign.Signed | None
    expires_at: datetime
    state: State
    record: sign.Signed | None
    """What it hands in, once signed."""


def _held(row: sqlite3.Row) -> Held:
    return Held(
        lease=row["lease"],
        registry=row["registry"],
        job=row["job"],
        stage=row["stage"],
        task_id=row["task_id"],
        task=sign.Signed.model_validate_json(row["task_json"]),
        result=(
            None
            if row["result_json"] is None
            else sign.Signed.model_validate_json(row["result_json"])
        ),
        expires_at=db.parse_iso(row["expires_at"]),
        state=row["state"],
        record=(
            None
            if row["record_json"] is None
            else sign.Signed.model_validate_json(row["record_json"])
        ),
    )


def holding(conn: sqlite3.Connection) -> list[Held]:
    """The jobs being worked on, oldest first."""
    rows = conn.execute(
        "SELECT * FROM network_work WHERE state = 'working' ORDER BY created_at, lease"
    ).fetchall()
    return [_held(r) for r in rows]


def held_by_task(conn: sqlite3.Connection, task_id: str) -> Held | None:
    row = conn.execute("SELECT * FROM network_work WHERE task_id = ?", (task_id,)).fetchone()
    return None if row is None else _held(row)


def _dump(signed: sign.Signed) -> str:
    return json.dumps(signed.model_dump(mode="json"), sort_keys=True)


def take(store: Store, registry: str, work: protocol.Work, job: protocol.Job) -> str:
    """The job lent as a task of this node (its id), with the lease, in one transaction."""
    recheck = work.stage == RECHECK
    record = work.task.record
    goal = " ".join(job.description.split())
    if len(goal) > GOAL_CHARS:
        goal = goal[: GOAL_CHARS - 1] + "…"
    spec = TaskSpec(
        goal=f"desman-global {work.stage}: {goal}",
        priority=Priority.LOW,
        subtasks=[
            SubtaskSpec(
                description=job.description,
                acceptance_criteria=job.acceptance_criteria,
                inputs=dict(job.inputs),
                kind=job.kind,
                # A recheck runs nothing: it judges what another node handed in.
                runs_code=job.runs_code and not recheck,
            )
        ],
        retry=RECHECK_RETRY if recheck else EXECUTE_RETRY,
        tools=Tools.NONE if recheck else Tools.GRANTED,
    )

    def keep(conn: sqlite3.Connection, task_id: str) -> None:
        conn.execute(
            "INSERT INTO network_work (lease, registry, job, stage, task_id, task_json,"
            " result_json, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                work.lease,
                registry,
                record.id,
                work.stage,
                task_id,
                _dump(work.task),
                None if work.result is None else _dump(work.result),
                db.to_iso(work.expires_at),
            ),
        )

    return store.create_task(
        spec,
        details={"job": record.id, "stage": work.stage, "signer": record.signer},
        also=keep,
        origin=Origin.NETWORK,
    )


def record_lifetime(held: Held, now: datetime) -> timedelta:
    """The term of what `held` hands in: a result holds until the job's own term ends, since
    another node may recheck it as late as that (the registry refuses one that ends sooner)."""
    return max(RECORD_LIFETIME, held.task.record.expires_at - now)


def extended(conn: sqlite3.Connection, lease: str, until: datetime) -> None:
    with db.transaction(conn):
        conn.execute(
            "UPDATE network_work SET expires_at = ?, updated_at = ? WHERE lease = ?",
            (db.to_iso(until), db.now_iso(), lease),
        )


def keep(conn: sqlite3.Connection, lease: str, record: sign.Signed) -> None:
    """The record to hand in, signed once: a later try sends this one, never a new one."""
    with db.transaction(conn):
        conn.execute(
            "UPDATE network_work SET record_json = ?, updated_at = ?"
            " WHERE lease = ? AND record_json IS NULL",
            (_dump(record), db.now_iso(), lease),
        )


def close(store: Store, held: Held, state: State, why: str) -> None:
    """The job ends here (`handed_in`, `lost`, `abandoned`); its task, if still open, is
    cancelled, so nothing of it waits for anyone."""
    with db.transaction(store.conn):
        store.conn.execute(
            "UPDATE network_work SET state = ?, updated_at = ? WHERE lease = ? AND state = 'working'",
            (state, db.now_iso(), held.lease),
        )
    if store.task_status(held.task_id) in (TaskStatus.ACTIVE, TaskStatus.NEEDS_HUMAN):
        store.cancel_task(held.task_id, f"desman-global: {why}")


# ── The crew's executor and reviewer ─────────────────────────────────────────


def replay(conn: sqlite3.Connection, task_id: str) -> dict[str, Any] | None:
    """For a recheck: the other node's result, as the reviewer sees it; None otherwise."""
    found = held_by_task(conn, task_id)
    if found is None or found.result is None:
        return None
    body = protocol.Result.model_validate(found.result.record.body)
    return {"output": body.output, "files": {f.path: f.text for f in body.files}}


def executor(conn: sqlite3.Connection, executor: Executor) -> Executor:
    """The network crew's executor: a recheck hands in the other node's result untouched (the
    work is the review); a job to do goes to `executor`."""

    def run(work: Work) -> Any:
        content = replay(conn, work.subtask.task_id)
        if content is not None:
            return Success(content=content)
        return executor(work)

    return run


def reviewer(conn: sqlite3.Connection, own: Reviewer, recheck: Reviewer) -> Reviewer:
    """`own` (with the evidence of this node's tool calls) for a job this node did; `recheck`
    (without: they stayed with the node that did the work) for another node's result."""

    def review(subtask: Subtask, artifact: dict[str, Any] | None) -> Any:
        found = held_by_task(conn, subtask.task_id)
        if found is not None and found.stage == RECHECK:
            return recheck(subtask, artifact)
        return own(subtask, artifact)

    return review


# ── Quotas ───────────────────────────────────────────────────────────────────


def day_start(now: datetime) -> datetime:
    """Quotas are per UTC day."""
    moment = now.astimezone(UTC)
    return moment.replace(hour=0, minute=0, second=0, microsecond=0)


def used_today(conn: sqlite3.Connection, provider: str, now: datetime) -> tuple[int, int]:
    """Calls and tokens jobs from the network spent today on `provider`'s models."""
    row = conn.execute(
        """SELECT count(*), COALESCE(SUM(m.prompt_tokens + m.completion_tokens), 0)
           FROM model_calls m JOIN tasks t ON t.id = m.task_id
           WHERE t.origin = 'network' AND m.provider = ? AND m.at >= ?""",
        (provider, db.to_iso(day_start(now))),
    ).fetchone()
    return int(row[0]), int(row[1])


def _held_calls(conn: sqlite3.Connection, provider: str, now: datetime) -> int:
    """Calls being made, all counted against today: one taken before midnight is logged when it
    ends, maybe today. The quota errs on the side of the person who lent it: right after
    midnight a call still out may keep a job waiting, never let one through past the limit."""
    row = conn.execute(
        "SELECT count(*) FROM quota_holds WHERE provider = ? AND expires_at > ?",
        (provider, db.to_iso(now)),
    ).fetchone()
    return int(row[0])


def quota_left(conn: sqlite3.Connection, config: Config, provider: str, now: datetime) -> bool:
    """Whether `provider` is lent and its day's quota is not spent (calls being made count)."""
    donation = config.network.donated.get(provider)
    if donation is None:
        return False
    calls, tokens = used_today(conn, provider, now)
    calls += _held_calls(conn, provider, now)
    return calls < donation.calls_per_day and tokens < donation.tokens_per_day


def hold_call(conn: sqlite3.Connection, config: Config, provider: str, now: datetime) -> int | None:
    """A call of the day's quota taken for one request (its hold's id), checked and held in one
    transaction: workers on one database never take the same last call. None: none left."""
    with db.transaction(conn):
        conn.execute("DELETE FROM quota_holds WHERE expires_at <= ?", (db.to_iso(now),))
        if not quota_left(conn, config, provider, now):
            return None
        cursor = conn.execute(
            "INSERT INTO quota_holds (provider, expires_at) VALUES (?, ?)",
            (provider, db.to_iso(now + HOLD_FOR)),
        )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def release_call(conn: sqlite3.Connection, hold: int) -> None:
    """The call is over: what it spent is in model_calls now."""
    with db.transaction(conn):
        conn.execute("DELETE FROM quota_holds WHERE id = ?", (hold,))


class Quota:
    """A lent model as the network's crew asks it: refused (`QuotaExhausted`, until the next
    UTC day) once the person's quota for its provider is spent, or when it is no longer lent.
    The config is read at every call: the person may change the quota while a worker runs."""

    def __init__(
        self,
        llm: LLM,
        conn: sqlite3.Connection,
        provider: str,
        config: Callable[[], Config],
        clock: Callable[[], datetime],
    ) -> None:
        self.llm = llm
        self.name = llm.name
        self.conn = conn
        self.provider = provider
        self._config = config
        self._clock = clock

    def complete(self, request: Request) -> Completion:
        now = self._clock()
        hold = hold_call(self.conn, self._config(), self.provider, now)
        if hold is None:
            raise QuotaExhausted(self.provider, day_start(now) + timedelta(days=1))
        try:
            return self.llm.complete(request)
        finally:
            release_call(self.conn, hold)

    def accept(self, request: Request, reply: Completion) -> None:
        accept = getattr(self.llm, "accept", None)
        if accept is not None:
            accept(request, reply)


# ── The end of a job ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Ending:
    body: protocol.Result | protocol.Verdict | None
    """What to sign and hand in; None: nothing (a recheck that could not be judged)."""
    why: str


def ending(store: Store, held: Held, workspaces: Path, now: datetime) -> Ending | None:
    """How the job ended, read from its task; None while it is still being worked on."""
    status = store.task_status(held.task_id)
    subtask = _the_subtask(store, held.task_id)
    if status is TaskStatus.DONE:
        if held.stage == RECHECK:
            return Ending(_verdict(held, accept=True), "accepted")
        return Ending(_result(store, held, workspaces), "done")
    if status is TaskStatus.ACTIVE:
        paused = subtask.paused_until
        if (
            subtask.status is SubtaskStatus.PAUSED
            and paused
            and db.parse_iso(paused) > (held.expires_at)
        ):
            return _gave_up(held, "paused for longer than the lease")
        return None
    rejected = _rejection(store, subtask)
    if held.stage == RECHECK and rejected is not None:
        return Ending(_verdict(held, accept=False, reason=rejected), "rejected")
    return _gave_up(held, f"the job ended {status.value}")


def _the_subtask(store: Store, task_id: str) -> Subtask:
    return next(s for s in store.list_subtasks(task_id) if s.parent_id is None)


def _gave_up(held: Held, why: str) -> Ending:
    if held.stage == RECHECK:
        return Ending(None, why)
    return Ending(
        protocol.Result(job=held.job, lease=held.lease, outcome="failed"),
        why,
    )


def _rejection(store: Store, subtask: Subtask) -> str | None:
    """The reviewer's feedback if its last verdict on `subtask` was not to accept it."""
    row = store.conn.execute(
        """SELECT details_json FROM audit_log WHERE subtask_id = ? AND action = 'verdict'
           ORDER BY seq DESC LIMIT 1""",
        (subtask.id,),
    ).fetchone()
    if row is None:
        return None
    details = json.loads(row[0])
    if details.get("verdict") in (None, "accept"):
        return None
    feedback = details.get("feedback")
    return (feedback if isinstance(feedback, str) and feedback else "rejected")[:2000]


def _verdict(held: Held, *, accept: bool, reason: str = "") -> protocol.Verdict:
    assert held.result is not None
    return protocol.Verdict(
        job=held.job,
        lease=held.lease,
        result=held.result.record.id,
        accept=accept,
        reason=reason,
    )


def _encoded(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=True))


def _result(store: Store, held: Held, workspaces: Path) -> protocol.Result:
    """The job's output (its accepted result, as JSON) and the files of its workspace, as many
    as fit (text files with names the protocol carries, smallest names first)."""
    content = store.final_result(held.task_id)
    output = "" if content is None else json.dumps(content, ensure_ascii=False)
    room = protocol.MAX_RESULT_BYTES - _encoded(output)
    if room < 0:
        return protocol.Result(job=held.job, lease=held.lease, outcome="failed")
    files: list[protocol.ResultFile] = []
    space = Workspace(workspaces, held.task_id)
    for path in space.files():
        if len(files) >= MAX_FILES:
            break
        try:
            text = space.read_text(path)
            file = protocol.ResultFile(path=path, text=text)
        except (PathEscapeError, OSError, UnicodeDecodeError, ValidationError):
            continue  # binary, gone, or a name the protocol cannot carry
        size = _encoded(text)
        if size > room:
            continue
        room -= size
        files.append(file)
    return protocol.Result(
        job=held.job, lease=held.lease, outcome="done", output=output, files=files
    )


def open_jobs(conn: sqlite3.Connection) -> int:
    """Jobs being worked on (for the person: `desman global status`)."""
    return int(
        conn.execute("SELECT count(*) FROM network_work WHERE state = 'working'").fetchone()[0]
    )
