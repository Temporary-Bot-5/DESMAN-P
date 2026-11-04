"""Confirmed one-shot reports: exact bodies, privacy, signatures and durable receipts."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click import ClickException
from click.testing import CliRunner

from conftest import FakeClock
from desman import commands, global_cli
from desman.cli import cli
from desman.config import ApiKey, Config, ConfigManager, Network, default_config_path, open_config
from desman.core import db, profile, sessions
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.retry import FailureKind
from desman.core.states import Actor, SubtaskStatus
from desman.core.store import Store
from desman.core.workspace import Workspace, workspace_root_for
from desman.net import node, protocol, sign
from desman.registry import store as registry_store
from desman.registry.cli import cli as registry_cli
from desman.registry.server import Registry
from test_registry import Node


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    opened = db.open_db(":memory:")
    yield opened
    opened.close()


@pytest.fixture
def joined() -> Config:
    config = Config(language="en", network=Network(enabled=True, registry="https://registry.test"))
    ConfigManager(default_config_path()).save(config)
    sign.node_key(sign.default_node_key_path())
    return config


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def respond(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"id": "receipt", "kind": body["signed"]["report"]["kind"]})

    original = node.submit_report

    def submit(
        address: str, key: sign.NodeKey, signed: protocol.SignedReport, **kwargs: Any
    ) -> protocol.ReportReceived:
        kwargs["transport"] = httpx.MockTransport(respond)
        return original(address, key, signed, **kwargs)

    monkeypatch.setattr(node, "submit_report", submit)
    return seen


@pytest.mark.parametrize(
    "args", [["report", "It crashes"], ["share"], ["share", "--no-stats", "--failures"]]
)
def test_cli_preview_and_confirmation(
    joined: Config, sent: list[dict[str, Any]], args: list[str]
) -> None:
    runner = CliRunner()
    shown = runner.invoke(cli, ["global", *args, "--json"])
    assert shown.exit_code == 0, shown.output
    preview = json.loads(shown.output)
    assert preview["sent"] is False and sent == []
    wrong = runner.invoke(cli, ["global", *args, "--confirm", "0" * 64])
    assert wrong.exit_code == 1 and "preview changed" in wrong.output and sent == []
    result = runner.invoke(cli, ["global", *args, "--confirm", preview["confirmation"], "--json"])
    assert result.exit_code == 0, result.output
    assert sent == [preview["submission"]]
    assert json.loads(result.output)["sent"] is True
    opened = db.open_db()
    try:
        entry = opened.execute(
            "SELECT details_json FROM network_log WHERE action = 'report_sent'"
        ).fetchone()
        assert set(json.loads(entry[0])) == {"registry", "node", "id", "kind"}
        assert "It crashes" not in entry[0]
    finally:
        opened.close()


def test_off_and_empty_selection_send_nothing(sent: list[dict[str, Any]], joined: Config) -> None:
    runner = CliRunner()
    empty = runner.invoke(cli, ["global", "share", "--no-stats", "--no-failures"])
    assert empty.exit_code == 1 and "Choose statistics" in empty.output
    joined.network.enabled = False
    ConfigManager(default_config_path()).save(joined)
    for args in (["report", "problem"], ["share"]):
        result = runner.invoke(cli, ["global", *args])
        assert result.exit_code == 1 and "is off" in result.output
    assert sent == []


@pytest.mark.parametrize("secret", ["sk-plain-test", 'sk-test"secret', "sk-test\\secret"])
def test_bug_secrets_are_never_printed(
    joined: Config, sent: list[dict[str, Any]], secret: str
) -> None:
    joined.api_keys = {"GROQ_API_KEY": [ApiKey(value=secret)]}
    ConfigManager(default_config_path()).save(joined)
    result = CliRunner().invoke(cli, ["global", "report", f"Failure with {secret}", "--json"])
    assert result.exit_code == 1
    assert "contains an API key" in result.output
    assert secret not in result.output and json.dumps(secret) not in result.output
    assert sent == []


def failed_task(conn: sqlite3.Connection, clock: FakeClock, cause: str) -> str:
    store = Store(conn, clock=clock)
    task = store.create_task(
        TaskSpec(
            goal="task goal",
            subtasks=[SubtaskSpec(description="description", acceptance_criteria="ok")],
        )
    )
    sub = store.claim("test")
    assert sub is not None
    store.record_failure(
        sub.id,
        SubtaskStatus.RUNNING,
        Actor.EXECUTOR,
        FailureKind.PERMANENT,
        lease=sub.lease_token,
        reason=cause,
    )
    return task


def test_memory_secrets_and_locked_config_recheck(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock
) -> None:
    secret = 'sk-test"\\secret'
    # Keep the failure inside the deterministic sharing window.
    clock.now = global_cli._utc_now()
    failed_task(conn, clock, secret)
    signed = global_cli.memory_upload(conn, joined, stats=False, failures=True, days=30)
    assert secret in signed.report.model_dump_json().replace('\\"', '"').replace("\\\\", "\\")
    joined.api_keys = {"GROQ_API_KEY": [ApiKey(value=secret)]}
    manager = ConfigManager(default_config_path())
    manager.save(joined)
    with pytest.raises(ClickException, match="contains an API key"):
        global_cli.memory_upload(conn, joined, stats=False, failures=True, days=30)
    with pytest.raises(ClickException, match="contains an API key"):
        global_cli.send_report(manager, conn, signed)


def test_task_snapshot_and_private_data(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(global_cli, "_utc_now", clock)
    task = failed_task(conn, clock, "failure cause")
    profile.Profile(conn).add("private profile marker", profile.Author.PERSON)
    sessions.Sessions(conn).add(None, sessions.Role.USER, "private chat marker")
    Workspace(workspace_root_for(conn), task).write("secret.txt", "workspace contents marker")
    bug = global_cli.report_upload(conn, joined, "problem", task)
    assert isinstance(bug.report, protocol.BugReport) and bug.report.task is not None
    assert bug.report.task.status == "needs_human"
    sub = bug.report.task.subtasks[0]
    assert sub.kind == "text" and sub.failure == "permanent" and sub.cause == "failure cause"
    memory = global_cli.memory_upload(conn, joined, stats=True, failures=True, days=30)
    for signed in (bug, memory):
        body = signed.model_dump_json()
        assert all(
            marker not in body
            for marker in (
                "private profile marker",
                "private chat marker",
                "workspace contents marker",
            )
        )
    assert isinstance(memory.report, protocol.MemoryShare)
    assert memory.report.failures and memory.report.failures[0]["said"] == "failure cause"
    assert memory.report.since.endswith("T00:00:00.000Z")


@pytest.mark.parametrize(
    "name,args", [("global report", {"text": "problem"}), ("global share", {"failures": True})]
)
def test_chat_window_contains_exact_body(
    conn: sqlite3.Connection, joined: Config, name: str, args: dict[str, Any]
) -> None:
    context = commands.Context(
        Store(conn), open_config(), lambda *_: None, lambda _: None, lambda: None
    )
    entry = commands.find(name)
    assert entry is not None
    parsed = entry.args.model_validate(args)
    assert entry.kind_of(parsed) is commands.Kind.CHANGE
    window = entry.handler(context, parsed)
    assert isinstance(window, commands.Proposal)
    exact = json.loads(window.lines[1])
    signed = protocol.SubmitReport.model_validate(exact).signed
    key = sign.node_key(sign.default_node_key_path())
    key.public.verify(base64.b64decode(signed.signature), protocol.report_content(signed.report))
    expected = "Send this bug report" if name == "global report" else "Send this memory snapshot"
    assert expected in window.lines[0]
    assert "https://registry.test" in window.lines[0]
    assert window.warnings


def test_memory_window_is_stable_within_day(
    conn: sqlite3.Connection, joined: Config, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(global_cli, "_utc_now", clock)
    first = global_cli.memory_upload(conn, joined, stats=True, failures=False, days=30)
    clock.advance(60)
    second = global_cli.memory_upload(conn, joined, stats=True, failures=False, days=30)
    assert first == second


def signed_bug(key: sign.NodeKey, text: str = "problem") -> protocol.SignedReport:
    report = protocol.BugReport(
        text=text,
        version="0.2",
        commit=None,
        python="3.12",
        system="Linux",
        language="en",
        task=None,
    )
    return protocol.SignedReport(
        report=report,
        signer=key.fingerprint,
        signature=base64.b64encode(key.sign(protocol.report_content(report))).decode(),
    )


@pytest.fixture
def registry(tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> Registry:
    root = sign.NodeKey.generate()
    root_path = tmp_path / "root.pub"
    root_path.write_bytes(sign.public_pem(root.public))
    monkeypatch.setattr(sign, "ROOT_PUB", root_path)
    path = tmp_path / "registry" / "registry.db"
    monkeypatch.setenv(registry_store.DIR_ENV, str(path.parent))
    registry_store.open_registry(path).close()
    return Registry(path, clock=clock)


def test_registry_accepts_retries_limits_and_signatures(
    registry: Registry, clock: FakeClock
) -> None:
    caller = Node(registry, clock)
    assert caller.register().status_code == 200
    first_id = None
    for i in range(5):
        body = protocol.SubmitReport(signed=signed_bug(caller.key, f"problem {i}")).model_dump(
            mode="json"
        )
        reply = caller.call("POST", protocol.REPORTS, body)
        assert reply.status_code == 200, reply.text
        if i == 0:
            first_id = reply.json()["id"]
    body = protocol.SubmitReport(signed=signed_bug(caller.key, "problem 0")).model_dump(mode="json")
    assert caller.call("POST", protocol.REPORTS, body).json()["id"] == first_id
    body = protocol.SubmitReport(signed=signed_bug(caller.key, "sixth")).model_dump(mode="json")
    reply = caller.call("POST", protocol.REPORTS, body)
    assert reply.status_code == 403 and reply.json()["error"] == "report_limit"
    memory = protocol.MemoryShare(
        since="2026-01-01T00:00:00Z", stats={}, failures=None, plan_fallbacks=None
    )
    signed = protocol.SignedReport(
        report=memory,
        signer=caller.fingerprint,
        signature=base64.b64encode(caller.key.sign(protocol.report_content(memory))).decode(),
    )
    assert (
        caller.call(
            "POST", protocol.REPORTS, protocol.SubmitReport(signed=signed).model_dump(mode="json")
        ).status_code
        == 200
    )
    forged = protocol.SubmitReport(signed=signed_bug(caller.key)).model_dump(mode="json")
    forged["signed"]["report"]["text"] = "tampered"
    assert caller.call("POST", protocol.REPORTS, forged).status_code == 401
    other = protocol.SubmitReport(signed=signed_bug(sign.NodeKey.generate())).model_dump(
        mode="json"
    )
    assert caller.call("POST", protocol.REPORTS, other).status_code == 401
    clock.advance(24 * 60 * 60 + 1)
    assert caller.call("POST", protocol.REPORTS, body).status_code == 200
    listing = CliRunner().invoke(registry_cli, ["reports", "--kind", "bug", "--json"])
    assert listing.exit_code == 0, listing.output
    assert len(json.loads(listing.output)) == 6
    shown = CliRunner().invoke(registry_cli, ["report", "show", str(first_id), "--json"])
    assert shown.exit_code == 0 and json.loads(shown.output)["id"] == first_id


def test_registry_upgrade_and_append_only(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "old.db"
    old = db.connect(path)
    db.migrate(old, registry_store.MIGRATIONS[:7])
    old.close()
    conn = registry_store.open_registry(path)
    try:
        received, created = registry_store.submit_report(
            conn, "node", signed_bug(sign.NodeKey.generate()), clock()
        )
        assert created
        for sql in ("UPDATE reports SET kind = 'memory'", "DELETE FROM reports"):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(sql)
        assert conn.execute("SELECT id FROM reports").fetchone()[0] == received.id
        event = conn.execute(
            "SELECT details_json FROM registry_log WHERE action = 'report_received'"
        ).fetchone()[0]
        assert set(json.loads(event)) == {"id", "kind"}
    finally:
        conn.close()


@pytest.mark.parametrize("memory", [False, True])
def test_node_to_registry_wire_round_trip(
    registry: Registry, clock: FakeClock, conn: sqlite3.Connection, joined: Config, memory: bool
) -> None:
    key = sign.node_key(sign.default_node_key_path())
    caller = Node(registry, clock, key=key)
    assert caller.register().status_code == 200
    signed = (
        global_cli.memory_upload(conn, joined, stats=True, failures=False, days=30)
        if memory
        else global_cli.report_upload(conn, joined, "problem", None)
    )
    wire = protocol.SubmitReport(signed=signed).model_dump(mode="json", exclude_none=True)
    received = global_cli.send_report(
        ConfigManager(default_config_path()),
        conn,
        signed,
        transport=httpx.MockTransport(caller._handle),
        clock=clock,
    )
    assert caller.last is not None and json.loads(caller.last.content) == wire
    assert received.kind == signed.report.kind
    # Neither the HTTP signature nor omitted null fields change the durable identity.
    again = global_cli.send_report(
        ConfigManager(default_config_path()),
        conn,
        signed,
        transport=httpx.MockTransport(caller._handle),
        clock=clock,
    )
    assert received == again


def test_confirmation_refuses_changed_memory_and_registry(
    joined: Config, sent: list[dict[str, Any]]
) -> None:
    runner = CliRunner()
    shown = json.loads(runner.invoke(cli, ["global", "share", "--json"]).output)
    conn = db.open_db()
    try:
        Store(conn).create_task(
            TaskSpec(
                goal="new work",
                subtasks=[SubtaskSpec(description="work", acceptance_criteria="ok")],
            )
        )
    finally:
        conn.close()
    changed = runner.invoke(cli, ["global", "share", "--confirm", shown["confirmation"]])
    assert changed.exit_code == 1 and "preview changed" in changed.output
    preview = json.loads(runner.invoke(cli, ["global", "report", "problem", "--json"]).output)
    joined.network.registry = "https://other.test"
    ConfigManager(default_config_path()).save(joined)
    changed = runner.invoke(
        cli, ["global", "report", "problem", "--confirm", preview["confirmation"]]
    )
    assert changed.exit_code == 1 and "preview changed" in changed.output
    assert sent == []


def test_invalid_contents_and_report_size_are_generic(
    conn: sqlite3.Connection, joined: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "private invalid text" * 400
    with pytest.raises(ClickException, match="invalid contents") as raised:
        global_cli.report_upload(conn, joined, secret, None)
    assert secret not in str(raised.value)
    monkeypatch.setattr(protocol, "MAX_REPORT_BYTES", 100)
    with pytest.raises(ClickException, match="too large"):
        global_cli.report_upload(conn, joined, "problem", None)


def test_registry_refuses_inactive_and_oversized_reports(
    registry: Registry, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = Node(registry, clock)
    assert caller.register().status_code == 200
    body = protocol.SubmitReport(signed=signed_bug(caller.key)).model_dump(mode="json")
    with monkeypatch.context() as limited:
        limited.setattr(protocol, "MAX_REPORT_BYTES", 100)
        reply = caller.call("POST", protocol.REPORTS, body)
        assert reply.status_code == 400
    assert caller.call("POST", protocol.LEAVE).status_code == 200
    assert caller.call("POST", protocol.REPORTS, body).status_code == 403


def test_failure_secret_is_refused_by_cli_without_output(
    joined: Config,
    sent: list[dict[str, Any]],
    clock: FakeClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = 'sk-private"\\failure'
    monkeypatch.setattr(global_cli, "_utc_now", clock)
    conn = db.open_db()
    try:
        failed_task(conn, clock, secret)
    finally:
        conn.close()
    joined.api_keys = {"GROQ_API_KEY": [ApiKey(value="unrelated"), ApiKey(value=secret)]}
    ConfigManager(default_config_path()).save(joined)
    result = CliRunner().invoke(cli, ["global", "share", "--failures", "--json"])
    assert result.exit_code == 1 and "contains an API key" in result.output
    assert secret not in result.output and json.dumps(secret) not in result.output
    assert sent == []


def test_report_signature_has_a_separate_domain(registry: Registry, clock: FakeClock) -> None:
    caller = Node(registry, clock)
    assert caller.register().status_code == 200
    signed = signed_bug(caller.key)
    signature = base64.b64encode(
        caller.key.sign(sign.canonical(signed.report.model_dump(mode="json")))
    ).decode()
    wrong_domain = signed.model_copy(update={"signature": signature})
    reply = caller.call(
        "POST", protocol.REPORTS, protocol.SubmitReport(signed=wrong_domain).model_dump(mode="json")
    )
    assert reply.status_code == 401
