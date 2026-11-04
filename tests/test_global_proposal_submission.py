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

from desman import global_cli
from desman.config import ApiKey, Config, ConfigManager, Network
from desman.core import db, proposals
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.proposals import Status
from desman.core.store import Store
from desman.net import protocol, sign


@pytest.fixture
def conn() -> Generator[sqlite3.Connection, None, None]:
    opened = db.open_db(":memory:")
    yield opened
    opened.close()


def proposed(conn: sqlite3.Connection, *, details: dict[str, Any] | None = None) -> int:
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
        summary="Improve the planner prompt.",
        details=details or {"changes": [{"target": "planner.rules", "new": "clearer"}]},
        diff="--- a/src/desman/planner.py\n+++ b/src/desman/planner.py\n",
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
    proposal_id = proposed(conn, details={"evidence": [f"copied {secret}"]})
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
