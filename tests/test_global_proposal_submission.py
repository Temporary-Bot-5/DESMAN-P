"""The person sees and confirms the exact proposal payload before it leaves the node."""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click import ClickException
from click.testing import CliRunner

from desman import commands, global_cli
from desman.cli import cli
from desman.config import ApiKey, Config, ConfigManager, Network, default_config_path, open_config
from desman.core import db, proposals
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.proposals import Status
from desman.core.store import Store
from desman.net import node, protocol, sign


@pytest.fixture
def conn() -> Generator[sqlite3.Connection, None, None]:
    opened = db.open_db(":memory:")
    yield opened
    opened.close()


DETAILS: dict[str, Any] = {
    "changes": [
        {
            "target": "planner.rules",
            "file": "src/desman/planner.py",
            "old": "rules",
            "new": "clearer rules",
        }
    ],
    "base": {"src/desman/planner.py": "c" * 64},
    "evidence": ["task 12 'my tax return for 2025' failed twice"],
    "expected_effect": "Fewer plans fall back to the template.",
}
DIFF = "--- a/src/desman/planner.py\n+++ b/src/desman/planner.py\n"


def proposed(
    conn: sqlite3.Connection,
    *,
    details: dict[str, Any] | None = None,
    summary: str = "Improve the planner prompt.",
    diff: str | None = DIFF,
) -> int:
    store = Store(conn)
    started = proposals.start(
        store,
        TaskSpec(
            goal="analysis",
            subtasks=[SubtaskSpec(description="analyze", acceptance_criteria="valid")],
        ),
        None,
    )
    proposals.settle(
        conn,
        started.id,
        Status.PROPOSED,
        summary=summary,
        details=details or DETAILS,
        diff=diff,
    )
    base = {
        "version": 1,
        "mode": "fake",
        "source": "/checkout/src/desman",
        "cases": [{"id": "case", "status": "done", "plan": "llm", "subtasks": 1}],
        "stats": {"tasks": {"done": 1}},
    }
    proposals.record_evaluation(
        conn,
        started.id,
        mode="fake",
        commit="a" * 40,
        verdict="better",
        base=base,
        candidate=base,
    )
    return started.id


def make_manager(tmp_path: Path, config: Config) -> ConfigManager:
    found = ConfigManager(tmp_path / "config" / "config.json")
    found.save(config)
    return found


def test_preview_is_signed_and_contains_both_benchmark_runs(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    proposal_id = proposed(conn)
    key_path = tmp_path / "node.key"
    key = sign.node_key(key_path)
    config = Config(network=Network(enabled=True, registry="https://registry.test"))

    preview = global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)

    assert preview.signer == key.fingerprint
    sign_bytes = sign.canonical(preview.submission.model_dump(mode="json"))
    key.public.verify(base64.b64decode(preview.signature), sign_bytes)
    assert preview.submission.base.cases[0].id == "case"
    assert preview.submission.candidate.cases[0].id == "case"


def test_a_configured_api_key_in_the_payload_cancels_submission(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    secret = "sk-configured-test-secret-0123456789"
    proposal_id = proposed(conn, summary=f"copied {secret}")
    key_path = tmp_path / "node.key"
    sign.node_key(key_path)
    config = Config(
        network=Network(enabled=True, registry="https://registry.test"),
        api_keys={"GROQ_API_KEY": [ApiKey(value=secret)]},
    )

    with pytest.raises(ClickException) as raised:
        global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)

    assert secret not in str(raised.value)
    assert "nothing was sent" in str(raised.value)


def test_send_uses_the_previewed_signed_payload_and_records_the_receipt(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    proposal_id = proposed(conn)
    key_path = tmp_path / "node.key"
    key = sign.node_key(key_path)
    config = Config(network=Network(enabled=True, registry="https://registry.test"))
    manager = make_manager(tmp_path, config)
    preview = global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)
    seen: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "b" * 32,
                "proposal": proposal_id,
                "evaluation": preview.submission.evaluation,
                "status": "submitted",
            },
        )

    received = global_cli.send_proposal(
        manager,
        conn,
        preview,
        key_path=key_path,
        transport=httpx.MockTransport(respond),
    )

    assert received.id == "b" * 32
    assert seen[0] == json.loads(
        protocol.SubmitProposal(signed=preview).model_dump_json(exclude_none=True)
    )
    rows = conn.execute("SELECT action, details_json FROM network_log").fetchall()
    assert len(rows) == 1 and rows[0]["action"] == "proposal_submitted"
    details = json.loads(rows[0]["details_json"])
    assert details["node"] == key.fingerprint
    assert "sk-configured-test-secret" not in rows[0]["details_json"]


def test_the_analysts_evidence_never_leaves(conn: sqlite3.Connection, tmp_path: Path) -> None:
    proposal_id = proposed(conn)
    key_path = tmp_path / "node.key"
    sign.node_key(key_path)
    config = Config(network=Network(enabled=True, registry="https://registry.test"))

    preview = global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)

    wire = protocol.SubmitProposal(signed=preview).model_dump_json()
    assert "tax return" not in wire and "evidence" not in wire
    assert preview.submission.details.expected_effect == DETAILS["expected_effect"]
    assert preview.submission.kind == protocol.PROPOSAL_KIND


def test_a_proposal_of_settings_stays_here(conn: sqlite3.Connection, tmp_path: Path) -> None:
    settings = [{"setting": "planner_model", "value": "x"}]
    proposal_id = proposed(conn, details=DETAILS | {"settings": settings})
    key_path = tmp_path / "node.key"
    sign.node_key(key_path)
    config = Config(network=Network(enabled=True, registry="https://registry.test"))

    with pytest.raises(ClickException, match="only proposals that change texts"):
        global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)


def test_a_missing_summary_is_the_same_in_every_language(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    proposal_id = proposed(conn, summary="")
    key_path = tmp_path / "node.key"
    sign.node_key(key_path)
    config = Config(language="ru", network=Network(enabled=True, registry="https://registry.test"))

    preview = global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)

    assert preview.submission.summary == global_cli.NO_SUMMARY


def test_the_registry_limit_is_told_in_words(conn: sqlite3.Connection, tmp_path: Path) -> None:
    proposal_id = proposed(conn)
    key_path = tmp_path / "node.key"
    sign.node_key(key_path)
    config = Config(network=Network(enabled=True, registry="https://registry.test"))
    manager = make_manager(tmp_path, config)
    preview = global_cli.proposal_upload(conn, config, proposal_id, key_path=key_path)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": protocol.PROPOSAL_LIMIT})

    with pytest.raises(ClickException, match="at most 5 proposals a day"):
        global_cli.send_proposal(
            manager, conn, preview, key_path=key_path, transport=httpx.MockTransport(respond)
        )
    assert conn.execute("SELECT count(*) FROM network_log").fetchone()[0] == 0


# ── The command and the chat ─────────────────────────────────────────────────


@pytest.fixture
def on_disk(monkeypatch: pytest.MonkeyPatch) -> tuple[int, list[protocol.SignedProposalSubmission]]:
    """A proposal in the default database, a joined node, and a registry that records what
    reaches it."""
    ConfigManager(default_config_path()).save(
        Config(
            language="en",
            network=Network(enabled=True, registry="https://registry.test"),
        )
    )
    sign.node_key(sign.default_node_key_path())
    opened = db.open_db()
    try:
        proposal_id = proposed(opened)
    finally:
        opened.close()
    sent: list[protocol.SignedProposalSubmission] = []

    def submit(
        registry: str, key: sign.NodeKey, signed: protocol.SignedProposalSubmission, **_: Any
    ) -> protocol.ProposalReceived:
        sent.append(signed)
        return protocol.ProposalReceived(
            id="d" * 32,
            proposal=signed.submission.proposal,
            evaluation=signed.submission.evaluation,
            status="submitted",
        )

    monkeypatch.setattr(node, "submit_proposal", submit)
    return proposal_id, sent


def test_the_command_sends_only_the_previewed_hash(
    on_disk: tuple[int, list[protocol.SignedProposalSubmission]],
) -> None:
    proposal_id, sent = on_disk
    runner = CliRunner()
    shown = runner.invoke(cli, ["global", "submit-proposal", str(proposal_id), "--json"])
    assert shown.exit_code == 0, shown.output
    preview = json.loads(shown.output)
    assert preview["sent"] is False and sent == []
    text = runner.invoke(cli, ["global", "submit-proposal", str(proposal_id)])
    assert "may quote those tasks" in text.output.replace("\n", " ")

    wrong = runner.invoke(
        cli, ["global", "submit-proposal", str(proposal_id), "--confirm", "0" * 64]
    )
    assert wrong.exit_code == 1
    assert "nothing was sent" in wrong.output and sent == []

    done = runner.invoke(
        cli,
        [
            "global",
            "submit-proposal",
            str(proposal_id),
            "--confirm",
            preview["confirmation"].upper(),
            "--json",
        ],
    )
    assert done.exit_code == 0, done.output
    assert json.loads(done.output)["sent"] is True
    [signed] = sent
    assert signed.model_dump(mode="json", exclude_none=True) == preview["submission"]["signed"]


def test_the_chat_sends_only_after_the_window(
    on_disk: tuple[int, list[protocol.SignedProposalSubmission]],
) -> None:
    proposal_id, sent = on_disk
    opened = db.open_db()
    try:
        ctx = commands.Context(
            Store(opened), open_config(), lambda *_: None, lambda task: None, lambda: None
        )
        entry = commands.find("global submit-proposal")
        assert entry is not None
        args = entry.args.model_validate({"proposal": proposal_id})
        assert entry.kind_of(args) is commands.Kind.CHANGE
        window = entry.handler(ctx, args)
        assert isinstance(window, commands.Proposal)
        assert '"signature"' in window.lines[1]
        assert any("may quote those tasks" in w for w in window.warnings)
        assert sent == []  # nothing until the "yes"
        outcome = window.apply()
    finally:
        opened.close()
    assert outcome.data["id"] == "d" * 32
    assert len(sent) == 1
