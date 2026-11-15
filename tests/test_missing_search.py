"""Missing whole-web setup is an operator decision, only when the executor gives up."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import FakeClock
from desman.actions import print_inbox, print_report
from desman.agents import llm_executor
from desman.chat import Chat
from desman.core.audit import check_history
from desman.core.coordinator import Coordinator, Review, Verdict
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.policy import Access, Policy, Role, RolePolicy
from desman.core.report import inbox, task_report
from desman.core.sessions import Role as SessionRole
from desman.core.sessions import Sessions
from desman.core.states import TaskStatus
from desman.core.store import Store
from desman.core.tools import Permission, Registry, web_tools
from desman.core.web import Web
from desman.interlocutor import Interlocutor
from desman.llm.fake import FakeLLM


@pytest.mark.parametrize(
    "search,success,source,earlier",
    [
        (True, False, "web", False),
        (False, False, "web", False),
        (True, True, "web", False),
        (True, False, "wikipedia", False),
        (True, False, "web", True),
    ],
)
def test_missing_search_engine(
    store: Store,
    clock: FakeClock,
    tmp_path: Path,
    search: bool,
    success: bool,
    capsys: pytest.CaptureFixture[str],
    source: str,
    earlier: bool,
) -> None:
    steps = []
    if search:
        steps.append(
            json.dumps(
                {
                    "tool_calls": [
                        {"tool": "web_search", "args": {"query": "price today", "source": source}}
                    ]
                }
            )
        )
    if earlier:
        steps.append(json.dumps({"text": "Initial findings."}))
    steps.append(
        json.dumps(
            {"text": "Price unavailable."}
            if success
            else {"type": "error", "reason": "Cannot find price."}
        )
    )
    # Ordinary errors still retry; a later answer can finish them.
    steps.append(json.dumps({"text": "Price unavailable."}))
    fake = FakeLLM().on("Execute", *steps)
    split = []
    task_id = store.create_task(
        TaskSpec(
            goal="Current price",
            subtasks=[SubtaskSpec(description="Find price", acceptance_criteria="Report findings")],
        )
    )
    reviewed = 0

    def review(subtask: object, artifact: object) -> Review:
        nonlocal reviewed
        reviewed += 1
        if earlier and reviewed == 1:
            return Review(verdict=Verdict.RETRY, feedback="Find more evidence.")
        return Review(verdict=Verdict.ACCEPT)

    runner = Coordinator(
        store,
        llm_executor(fake),
        review,
        workspaces=tmp_path,
        tools=Registry(web_tools(Web())),
        access=Access(
            Policy(roles={Role.EXECUTOR: RolePolicy(tools=["web_search"])}),
            Role.EXECUTOR,
            [Permission.NETWORK],
        ),
        sleep=clock.sleep,
        decomposer=lambda work: split.append(work),
    )
    result = runner.run(task_id)
    [subtask] = store.list_subtasks(task_id)
    unavailable = [row for row in store.audit_log(task_id) if row["action"] == "tool_unavailable"]
    if search and not success and source == "web" and not earlier:
        assert result.status is TaskStatus.NEEDS_HUMAN
        assert subtask.attempts.quality == subtask.attempts.transient == 0
        assert not split and not store.parts(subtask.id)
        [entry] = unavailable
        details = json.loads(entry["details_json"])
        assert details["tool"] == "web_search" and details["missing"] == "search engine"
        assert len(fake.calls) == 2
        hint = "Add a Tavily key or SearXNG in desman config to search the whole web."
        print_report(task_report(store, task_id))
        assert hint in capsys.readouterr().out
        print_inbox(inbox(store))
        assert hint in capsys.readouterr().out
        sessions = Sessions(store.conn, clock=clock)
        chat = Chat(sessions)
        chat.session_id = sessions.add(None, SessionRole.USER, "Find price").session_id
        reteller = FakeLLM().on("Retell task", '{"text": "Please set up search."}')
        chat._outcome(task_id, task_id[:8], Interlocutor(reteller, "en"))
        assert any(hint in message.text for message in sessions.messages(chat.session_id))
    else:
        assert result.status is TaskStatus.DONE
        assert subtask.attempts.quality == (2 if earlier else 0 if success else 1)
        assert not unavailable
    assert check_history(store.conn, task_id) == []
