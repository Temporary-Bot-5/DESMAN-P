"""Demo D10 (ROADMAP G.11): a leased job survives a node crash, another owner rechecks it,
and a confirmed proposal becomes a root-signed release installed and rolled back by a person.

Three nodes are needed: a node is never lent a job it held before, so crashed A cannot
recheck B's result; C supplies the independent verdict. Nothing above HTTP is faked except
the model (FakeLLM) and time (FakeClock). Registry.handle runs behind MockTransport; crews,
coordinators, signatures, databases, benchmark subprocesses and temporary git repos are real.
The manual check is in docs/demo-d10.md.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman import evaluation, improve, runs
from desman.agents import REVIEW
from desman.cli import cli
from desman.config import Config, ConfigManager, Donation, Network, Tuning
from desman.core import db, netlog, proposals
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Success, Verdict
from desman.core.store import Store
from desman.core.tools import builtin_registry
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from desman.net import node, protocol, sign
from desman.registry import admin, checks, releases, work
from desman.registry import store as registry_store
from desman.registry.server import Registry
from test_evaluation import checkout, git, origin, package, proposal, rules_change, worktrees
from test_registry import _write_key

__all__ = ["checkout", "origin"]
ADDRESS = "https://registry.test"


class Machine:
    """One persistent identity/home, with a new connection and crew on restart."""

    def __init__(
        self, home: Path, root: Path, transport: httpx.MockTransport, clock: FakeClock
    ) -> None:
        home.mkdir(exist_ok=True)
        self.closed = False
        self.key_path = home / "node.key"
        self.key = sign.node_key(self.key_path)
        self.conn = db.open_db(home / "desman.db")
        self.store = Store(self.conn, clock)
        self.config = Config(
            language="en",
            settings={"OLLAMA_MODEL": "llama3.2"},
            network=Network(
                enabled=True,
                registry=ADDRESS,
                donated={"ollama": Donation(calls_per_day=1000, tokens_per_day=1_000_000)},
            ),
        )
        self.manager = ConfigManager(home / "config.json")
        self.manager.save(self.config)

        def respond(request: Request) -> str:
            return (
                '{"verdict": "accept"}' if request.prompt.startswith(REVIEW) else '{"answer": "5"}'
            )

        model = FakeLLM(respond, repeat_last=True, clock=clock, sleep=clock.sleep)
        crew = runs.network_crew(
            self.store, self.config, self.manager.load, given=runs.Models(model, None), cache=False
        )
        assert crew is not None
        self.coordinator = Coordinator(
            self.store,
            lambda _: Success(content={}),
            lambda *_: Review(verdict=Verdict.ACCEPT),
            owner=home.name,
            sleep=clock.sleep,
            tools=builtin_registry(),
            network=crew,
        )
        self.node = node.Node(
            self.conn,
            self.manager.load,
            key_path=self.key_path,
            root=root,
            transport=transport,
            clock=clock,
            work=True,
        )

    def finish(self) -> None:
        self.coordinator.work(until_idle=True, max_ticks=10)
        self.node.poll(idle=False)

    def close(self) -> None:
        if not self.closed:
            self.node.close()
            self.conn.close()
            self.closed = True


def test_d10(
    checkout: Path, tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `desman global update` checks the release with a Trust on the wall clock; the trust list
    # and records here are dated by FakeClock (2026-01-01), so give that Trust the same clock.
    init = sign.Trust.__init__

    def timed_trust(self: sign.Trust, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **{"clock": clock, **kwargs})

    monkeypatch.setattr(sign.Trust, "__init__", timed_trust)
    directory = tmp_path / "registry"
    directory.mkdir()
    root = sign.NodeKey.generate()
    root_pub = directory / "root.pub"
    root_pub.write_bytes(sign.public_pem(root.public))
    monkeypatch.setattr(sign, "ROOT_PUB", root_pub)
    _write_key(directory / registry_store.ROOT_KEY, root)
    registry_db = directory / registry_store.DB_NAME
    registry_store.open_registry(registry_db).close()
    operator = sign.NodeKey.generate()
    with registry_store.connection(registry_db) as conn:
        admin.assign(conn, root, operator.public, {sign.Scope.TASKS}, days=30, now=clock())
        admin.sign_trust_list(conn, root, days=7, now=clock())
        admin.add_job(
            conn,
            operator,
            protocol.Job(description="Sum 2 and 3", acceptance_criteria="Says 5"),
            days=7,
            now=clock(),
            root_pub=root_pub,
        )
    server = Registry(registry_db, root=root_pub, clock=clock)
    exchanges: list[tuple[str, int, dict[str, Any]]] = []

    def handle(req: httpx.Request) -> httpx.Response:
        reply = server.handle(
            req.method, req.url.raw_path.decode(), req.headers.multi_items(), req.content
        )
        exchanges.append((req.url.path, reply.status, reply.body))
        return httpx.Response(reply.status, json=reply.body)

    transport = httpx.MockTransport(handle)
    # Both the operator and A clone the same bare origin; release delivery never supplies code.
    remote = tmp_path / "remote.git"
    git(checkout, "clone", "--bare", str(checkout), str(remote))
    git(checkout, "remote", "set-url", "origin", str(remote))
    a_repo = tmp_path / "a-checkout"
    git(checkout, "clone", str(remote), str(a_repo))
    base = git(a_repo, "rev-parse", "HEAD").strip()
    old_rules = improve.current(improve.BY_ID["planner.rules"], package(a_repo))

    with ExitStack() as stack:

        def machine(name: str) -> Machine:
            m = Machine(tmp_path / name, root_pub, transport, clock)
            stack.callback(m.close)
            return m

        a, b, c = machine("A"), machine("B"), machine("C")
        a.node.poll(idle=True)
        assert a.conn.execute("SELECT state FROM network_work").fetchone()[0] == "working"
        a.coordinator.work(until_idle=True, max_ticks=10)
        # Crash after durable local completion, before the next poll signs/sends the result.
        # The registry still owns an active lease and has received no result.
        with registry_store.connection(registry_db) as conn:
            assert conn.execute("SELECT count(*) FROM results").fetchone()[0] == 0
        a.close()
        clock.advance(work.LEASE.total_seconds() + 1)
        with registry_store.connection(registry_db) as conn:
            work.expire(conn, clock())
            assert conn.execute("SELECT ended FROM leases").fetchone()[0] == "expired"
            assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "execute"
        b.node.poll(idle=True)
        b.finish()
        a = machine("A")
        restart_at = len(exchanges)
        a.node.poll(idle=True)
        assert any(
            path == protocol.CLAIM and status == 200 and body.get("work") is None
            for path, status, body in exchanges[restart_at:]
        )
        assert any(
            path == protocol.HAND_IN and body.get("error") == "lease_lost"
            for path, _, body in exchanges
        )
        with registry_store.connection(registry_db) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM leases WHERE node = ?", (a.key.fingerprint,)
                ).fetchone()[0]
                == 1
            )
            assert conn.execute("SELECT status FROM jobs").fetchone()[0] == "recheck"
        c.node.poll(idle=True)
        c.finish()

        with registry_store.connection(registry_db) as conn:
            accepted = conn.execute("SELECT * FROM jobs").fetchone()
            assert accepted["status"] == "accepted"
            result = conn.execute(
                "SELECT * FROM results WHERE id = ?", (accepted["result"],)
            ).fetchone()
            verdict = conn.execute("SELECT * FROM results WHERE kind = 'verdict'").fetchone()
            assert result["node"] == b.key.fingerprint and verdict["node"] == c.key.fingerprint
            assert verdict["outcome"] == "accept"
            assert (result["owner"] or result["node"]) != (verdict["owner"] or verdict["node"])
            for row in conn.execute(
                "SELECT r.node, r.kind, l.node AS holder, l.stage FROM results r JOIN leases l ON l.token = r.lease"
            ):
                assert row["node"] == row["holder"]
                assert row["stage"] == ("execute" if row["kind"] == "result" else "recheck")
            assert not conn.execute(
                "SELECT job, node FROM leases GROUP BY job, node HAVING count(*) > 1"
            ).fetchall()
        for m in (a, b, c):
            for (task_id,) in m.conn.execute("SELECT id FROM tasks"):
                assert check_history(m.conn, task_id) == []
            assert not m.conn.execute(
                "SELECT 1 FROM subtasks WHERE status = 'needs_human'"
            ).fetchall()
            assert not m.conn.execute(
                "SELECT 1 FROM network_work WHERE state = 'working'"
            ).fetchall()
            actions = [e.action for e in netlog.entries(m.conn)]
            assert "record_received" in actions
            assert ("work_dropped" if m is a else "handed_in") in actions

        # A local proposal with a real completed fake benchmark, then the exact upload window.
        diff, details = improve.render([rules_change(a_repo)], package(a_repo))
        p = proposal(a.store, diff, details)
        evaluated = evaluation.evaluate(a.conn, p, mode="fake", tuning=Tuning(), repo=a_repo)
        assert evaluated.verdict == "same"
        before = len(exchanges)
        submit = node.submit_proposal

        def send(
            address: str,
            key: sign.NodeKey,
            signed: protocol.SignedProposalSubmission,
            **kwargs: Any,
        ) -> protocol.ProposalReceived:
            return submit(
                address, key, signed, **{**kwargs, "transport": transport, "clock": clock}
            )

        with monkeypatch.context() as context:
            context.setenv("DESMAN_CONFIG_DIR", str(a.key_path.parent))
            context.setenv("DESMAN_DATA_DIR", str(a.key_path.parent))
            context.setattr(node, "submit_proposal", send)
            runner = CliRunner()
            args = ["global", "submit-proposal", str(p.id)]
            shown_upload = runner.invoke(cli, [*args, "--json"])
            assert shown_upload.exit_code == 0, shown_upload.output
            preview = json.loads(shown_upload.output)
            assert preview["sent"] is False and len(exchanges) == before
            with registry_store.connection(registry_db) as conn:
                assert conn.execute("SELECT count(*) FROM proposals").fetchone()[0] == 0
            refused = runner.invoke(cli, [*args, "--confirm", "0" * 64])
            assert refused.exit_code == 1 and len(exchanges) == before
            sent = runner.invoke(cli, [*args, "--confirm", preview["confirmation"], "--json"])
            assert sent.exit_code == 0, sent.output
            submission_id = json.loads(sent.output)["id"]
        clock.advance(120)  # the next idle claim, paced by Node
        with monkeypatch.context() as context:
            context.setattr(evaluation, "repository", lambda: checkout)
            b.node.poll(idle=True)
        assert len(worktrees(checkout)) == 1
        with registry_store.connection(registry_db) as conn:
            found = registry_store.proposal(conn, submission_id)
            assert found is not None and found.status == "reviewed"
            [checked] = checks.results(conn, submission_id)
            assert checked["outcome"] == "checked" and checked["matches_submission"] is True
            snapshot = list(conn.iterdump())
            plan = releases.plan(conn, submission_id, checkout)
            assert list(conn.iterdump()) == snapshot
            assert git(checkout, "rev-parse", "HEAD").strip() == base
            released = releases.accept(
                conn,
                submission_id,
                checkout,
                plan["confirmation"],
                directory=directory,
                now=clock(),
            )
        signed = sign.Signed.model_validate(released["signed"])
        sign.check_signature(signed, root.public)
        assert released["commit"] == git(checkout, "rev-parse", "HEAD").strip()
        git(checkout, "push", "origin", "main")
        assert a.node.next_at is not None
        clock.advance(max(0, (a.node.next_at - clock()).total_seconds()))
        a.node.poll(idle=False)
        assert "release_received" in [e.action for e in netlog.entries(a.conn)]
        # The person's own commands: look first, install exactly release 1, then go back.
        with monkeypatch.context() as context:
            context.setenv("DESMAN_CONFIG_DIR", str(a.key_path.parent))
            context.setenv("DESMAN_DATA_DIR", str(a.key_path.parent))
            context.setattr(evaluation, "repository", lambda: a_repo)
            runner = CliRunner()
            shown = runner.invoke(cli, ["global", "update", "--show"])
            assert shown.exit_code == 0, shown.output
            assert "planner.py" in shown.output
            assert git(a_repo, "rev-parse", "HEAD").strip() == base
            assert proposals.updates(a.conn) == []
            applied = runner.invoke(cli, ["global", "update", "--apply", "1"])
            assert applied.exit_code == 0, applied.output
            assert [u.action for u in proposals.updates(a.conn)] == ["applied"]
            assert git(a_repo, "rev-parse", "HEAD").strip() == released["commit"]
            assert (
                improve.current(improve.BY_ID["planner.rules"], package(a_repo))
                == old_rules + " Prefer one subtask."
            )
            reverted = runner.invoke(cli, ["global", "update", "--revert"])
            assert reverted.exit_code == 0, reverted.output
        assert git(a_repo, "rev-parse", "HEAD").strip() == base
        assert improve.current(improve.BY_ID["planner.rules"], package(a_repo)) == old_rules
        assert [u.action for u in reversed(proposals.updates(a.conn))] == ["applied", "reverted"]
        assert {"proposal_submitted", "update_applied", "update_reverted"} <= {
            e.action for e in netlog.entries(a.conn)
        }
