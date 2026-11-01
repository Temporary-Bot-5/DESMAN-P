"""An agent may only *suggest* a model (ROADMAP 4.5b): `suggest_model` records a suggestion in
the audit and nothing else; the person reads it in `desman status` (or the chat) and adds the
model with `desman models add`. Hugging Face is an `httpx.MockTransport` handler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from conftest import FakeClock
from desman.actions import suggestion_lines
from desman.agents import llm_executor
from desman.cli import cli
from desman.core import db
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator
from desman.core.policy import Permission, executor_access
from desman.core.report import ModelSuggestion, task_report
from desman.core.states import TaskStatus
from desman.core.store import StaleStateError, Store
from desman.core.tools import (
    MAX_SUGGESTIONS,
    Registry,
    Toolbox,
    ToolCall,
    builtin_registry,
    suggest_model_tool,
)
from desman.core.workspace import Workspace
from desman.llm import hub
from desman.llm.fake import FakeLLM
from test_hub import GB, PLENTY, REPO, client, listing
from test_tools import accept, one_step, tool_calls

SUGGESTION = {"model": REPO, "reason": "a small model that writes code well"}


class Hub:
    """Hugging Face's API: the listing of REPO, and what was asked."""

    def __init__(self, response: httpx.Response | None = None) -> None:
        self.response = response if response is not None else listing()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    def look_up(self, ref: hub.Ref) -> hub.Gguf:
        return hub.inspect(ref, client(self))


def registry(api: Hub, machine: hub.Memory = PLENTY) -> Registry:
    return Registry([suggest_model_tool(api.look_up, lambda: machine)])


def toolbox(
    tmp_path: Path, tools: Registry, *switches: Permission, recorded: list[Any] | None = None
) -> Toolbox:
    return Toolbox(
        Workspace(tmp_path, "t1"),
        tools,
        access=executor_access(switches),
        record=lambda action, details: (recorded if recorded is not None else []).append(
            (action, details)
        ),
    )


def test_a_suggestion_needs_network_access(tmp_path: Path) -> None:
    api = Hub()
    box = toolbox(tmp_path, registry(api))
    assert box.specs() == []
    refused = box.call(ToolCall(tool="suggest_model", args=SUGGESTION))
    assert not refused.ok and "network_access" in (refused.error or "")
    assert api.requests == []


def test_a_suggestion_is_recorded_and_changes_nothing(tmp_path: Path) -> None:
    api, recorded = Hub(), list[Any]()
    box = toolbox(tmp_path, registry(api), Permission.NETWORK, recorded=recorded)
    result = box.call(ToolCall(tool="suggest_model", args=SUGGESTION))
    assert result.ok
    assert result.output == {
        "model": f"{REPO}:Q4_K_M",
        "server": "ollama",
        "reason": "a small model that writes code well",
        "size": 2 * GB,
        "memory": int(2 * GB * hub.OVERHEAD),
        "fits": True,
        "gated": False,
        "note": "Suggested to the person; nothing was downloaded or changed.",
    }
    assert [action for action, _ in recorded] == ["tool_called", "model_suggested"]
    assert recorded[1][1] == result.output
    # Only the listing was read, without a token: nothing was pulled or downloaded.
    [request] = api.requests
    assert request.method == "GET" and request.url.path == f"/api/models/{REPO}"
    assert "authorization" not in request.headers
    assert list((tmp_path / "t1").iterdir()) == []  # nothing in the workspace either


def test_whether_it_fits_is_said(tmp_path: Path) -> None:
    small = hub.Memory(ram=GB, video=None)
    box = toolbox(tmp_path, registry(Hub(), small), Permission.NETWORK)
    result = box.call(ToolCall(tool="suggest_model", args={**SUGGESTION, "server": "llamacpp"}))
    assert result.ok and result.output["fits"] is False
    assert result.output["server"] == "llamacpp"


@pytest.mark.parametrize(
    ("model", "response", "error"),
    [
        ("not a model", None, "is not a Hugging Face model"),
        (REPO, httpx.Response(404), "is not on Hugging Face"),
        (REPO, listing([{"rfilename": "model.safetensors", "size": GB}]), "no GGUF files"),
        (f"{REPO}:Q2_K", None, "name one of its quantizations"),
        (REPO, httpx.Response(503), "cannot be reached"),
    ],
)
def test_a_model_that_cannot_be_used_goes_back_to_the_model(
    tmp_path: Path, model: str, response: httpx.Response | None, error: str
) -> None:
    recorded: list[Any] = []
    box = toolbox(tmp_path, registry(Hub(response)), Permission.NETWORK, recorded=recorded)
    result = box.call(ToolCall(tool="suggest_model", args={**SUGGESTION, "model": model}))
    assert not result.ok and error in (result.error or "")
    assert "model_suggested" not in [action for action, _ in recorded]


def test_suggestions_have_a_budget_and_a_reason(tmp_path: Path) -> None:
    box = toolbox(tmp_path, registry(Hub()), Permission.NETWORK)
    results = [
        box.call(ToolCall(tool="suggest_model", args=SUGGESTION))
        for _ in range(MAX_SUGGESTIONS + 1)
    ]
    assert all(r.ok for r in results[:MAX_SUGGESTIONS])
    assert "may be called" in (results[-1].error or "")
    no_reason = toolbox(tmp_path, registry(Hub()), Permission.NETWORK).call(
        ToolCall(tool="suggest_model", args={"model": REPO})
    )
    assert not no_reason.ok and "reason" in (no_reason.error or "")


def test_the_executor_suggests_and_the_person_reads_it_in_status(
    clock: FakeClock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "desman.db"
    store = Store(db.open_db(db_path), clock=clock)
    api = Hub(listing(gated=True))
    llm = FakeLLM().on(
        "Execute",
        tool_calls(("suggest_model", SUGGESTION)),
        tool_calls(("suggest_model", SUGGESTION)),  # the same again: shown once
        '{"answer": "done with what I have"}',
    )
    task_id = store.create_task(one_step())
    coordinator = Coordinator(
        store,
        llm_executor(llm),
        accept,
        workspaces=tmp_path / "ws",
        sleep=clock.sleep,
        tools=builtin_registry(models=api.look_up),
        access=executor_access([Permission.NETWORK]),
    )
    monkeypatch.setattr(hub, "memory", lambda: PLENTY)
    assert coordinator.run(task_id).status is TaskStatus.DONE
    assert check_history(store.conn, task_id) == []

    report = task_report(store, task_id)
    assert report.suggestions == [
        ModelSuggestion(
            seq=0,
            model=f"{REPO}:Q4_K_M",
            server="ollama",
            reason="a small model that writes code well",
            size=2 * GB,
            memory=int(2 * GB * hub.OVERHEAD),
            fits=True,
            gated=True,
        )
    ]
    lines = suggestion_lines(report.suggestions)
    assert lines[-1].strip() == f"To use it: desman models add {REPO}:Q4_K_M --server ollama"
    assert any("is gated" in line and f"https://huggingface.co/{REPO}" in line for line in lines)

    shown = CliRunner().invoke(cli, ["status", task_id, "--db", str(db_path)])
    assert "Suggested models" in shown.output and "Ollama" in shown.output
    as_json = CliRunner().invoke(cli, ["status", task_id, "--db", str(db_path), "--json"])
    assert json.loads(as_json.output)["suggestions"][0]["model"] == f"{REPO}:Q4_K_M"


def test_no_suggestion_no_lines() -> None:
    assert suggestion_lines([]) == []


def test_a_suggestion_is_not_recorded_once_the_lease_is_lost(tmp_path: Path) -> None:
    """A lookup may outlast the lease: work another worker took over leaves no suggestion."""
    lost = False

    def look_up(ref: hub.Ref) -> hub.Gguf:
        nonlocal lost
        found = Hub().look_up(ref)
        lost = True  # the lease expired while Hugging Face answered
        return found

    def guard() -> None:
        if lost:
            raise StaleStateError("lease lost")

    recorded: list[Any] = []
    box = Toolbox(
        Workspace(tmp_path, "t1"),
        Registry([suggest_model_tool(look_up, lambda: PLENTY)]),
        access=executor_access([Permission.NETWORK]),
        record=lambda action, details: recorded.append((action, details)),
        guard=guard,
    )
    with pytest.raises(StaleStateError):
        box.call(ToolCall(tool="suggest_model", args=SUGGESTION))
    assert "model_suggested" not in [action for action, _ in recorded]


def test_a_blank_reason_is_refused(tmp_path: Path) -> None:
    box = toolbox(tmp_path, registry(Hub()), Permission.NETWORK)
    result = box.call(ToolCall(tool="suggest_model", args={**SUGGESTION, "reason": "   "}))
    assert not result.ok and "reason" in (result.error or "")
