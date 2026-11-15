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
        assert details["tool"] == "web_search" and details["missing"] == "search_engine"
        assert len(fake.calls) == 2
        hint = "Set up a search engine for the whole web: `desman search setup` (SearXNG, on this computer) or a Tavily key in `desman config`."
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


@pytest.mark.parametrize(
    "ending", ["rounds", "unusable", "write_refused", "write_path", "write_args"]
)
def test_search_failure_followed_by_giving_up(
    store: Store, clock: FakeClock, tmp_path: Path, ending: str
) -> None:
    from desman.agents import MAX_TOOL_ROUNDS
    from desman.core.tools import WRITE_FILE

    search = json.dumps({"tool_calls": [{"tool": "web_search", "args": {"query": "price"}}]})
    if ending == "rounds":
        steps = [search] * (MAX_TOOL_ROUNDS + 1)
    elif ending == "unusable":
        steps = [search, '{"text":', '{"text":']
    else:
        steps = [
            search,
            json.dumps(
                {
                    "tool_calls": [
                        {
                            "tool": "write_file",
                            "args": (
                                {"path": "../x", "content": "x"}
                                if ending == "write_path"
                                else {"path": "x"}
                                if ending == "write_args"
                                else {"path": "x", "content": "x"}
                            ),
                        }
                    ]
                }
            ),
            '{"type":"error","reason":"Cannot write"}',
            '{"text":"Finished on retry"}',
        ]
    fake = FakeLLM().on("Execute", *steps)
    task_id = store.create_task(
        TaskSpec(
            goal="Find price",
            subtasks=[SubtaskSpec(description="Find", acceptance_criteria="Report")],
        )
    )
    runner = Coordinator(
        store,
        llm_executor(fake),
        lambda s, a: Review(verdict=Verdict.ACCEPT),
        workspaces=tmp_path,
        tools=Registry([*web_tools(Web()), WRITE_FILE]),
        access=Access(
            Policy(
                roles={
                    Role.EXECUTOR: RolePolicy(
                        tools=["web_search"]
                        if ending == "write_refused"
                        else ["web_search", "write_file"]
                    )
                }
            ),
            Role.EXECUTOR,
            [Permission.NETWORK],
        ),
        sleep=clock.sleep,
    )
    result = runner.run(task_id)
    [sub] = store.list_subtasks(task_id)
    if ending.startswith("write_"):
        assert result.status is TaskStatus.DONE
        assert sub.attempts.quality == 1
        assert not [r for r in store.audit_log(task_id) if r["action"] == "tool_unavailable"]
    else:
        assert result.status is TaskStatus.NEEDS_HUMAN
        assert sub.attempts.quality == 0
        assert store.executions(sub.id) == 1
    assert check_history(store.conn, task_id) == []


@pytest.mark.parametrize("note", [None, "Try again"])
def test_retry_after_engine_setup_starts_new_search(
    store: Store, clock: FakeClock, tmp_path: Path, note: str | None
) -> None:
    from desman.core.states import Actor, SubtaskStatus
    from desman.core.web import SearchResult
    from desman.llm.cache import Cached

    class Engine:
        name = "test"
        calls = 0

        def search(
            self, query: str, max_results: int, language: str | None = None
        ) -> list[SearchResult]:
            self.calls += 1
            return [SearchResult(title="Price", url="https://example.com", snippet="10")]

    search = json.dumps({"tool_calls": [{"tool": "web_search", "args": {"query": "price"}}]})
    fake = FakeLLM().on(
        "Execute",
        search,
        '{"type":"error","reason":"No search"}',
        search,
        '{"type":"error","reason":"No search"}',
        search,
        '{"text":"Price is 10"}',
    )
    cached = Cached(fake, store.conn, "test", clock=clock)
    web = Web()
    task_id = store.create_task(
        TaskSpec(
            goal="Find price",
            subtasks=[SubtaskSpec(description="Find", acceptance_criteria="Report")],
        )
    )
    runner = Coordinator(
        store,
        llm_executor(cached),
        lambda s, a: Review(verdict=Verdict.ACCEPT),
        workspaces=tmp_path,
        tools=Registry(web_tools(web)),
        access=Access(
            Policy(roles={Role.EXECUTOR: RolePolicy(tools=["web_search"])}),
            Role.EXECUTOR,
            [Permission.NETWORK],
        ),
        sleep=clock.sleep,
    )
    assert runner.run(task_id).status is TaskStatus.NEEDS_HUMAN
    [sub] = store.list_subtasks(task_id)
    assert store.executions(sub.id) == 1
    store.transition(
        sub.id, SubtaskStatus.NEEDS_HUMAN, SubtaskStatus.READY, Actor.HUMAN, reason=note
    )
    assert runner.run(task_id).status is TaskStatus.NEEDS_HUMAN
    assert store.executions(sub.id) == 2
    engine = Engine()
    web.engines.append(engine)
    store.transition(
        sub.id, SubtaskStatus.NEEDS_HUMAN, SubtaskStatus.READY, Actor.HUMAN, reason=note
    )
    assert runner.run(task_id).status is TaskStatus.DONE
    assert engine.calls == 1
    assert len(fake.calls) == 6 and cached.hits == 0
    assert json.loads(store.effects(sub.id, 2)[0]["result_json"])["ok"] is True
    assert not [r for r in store.audit_log(task_id) if r["action"] == "tool_replayed"]
    assert check_history(store.conn, task_id) == []


def test_chat_hint_once_after_all_waiting_rows(
    store: Store, clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    from desman.actions import search_setup_hint
    from desman.core.retry import FailureKind
    from desman.core.states import Actor, SubtaskStatus

    task_id = store.create_task(
        TaskSpec(
            goal="Prices",
            subtasks=[
                SubtaskSpec(description=description, acceptance_criteria="Report")
                for description in ["First waiting item", "Second waiting item"]
            ],
        )
    )
    for _ in range(2):
        sub = store.claim("test", task_id=task_id)
        assert sub is not None and sub.lease_token is not None
        store.record_failure(
            sub.id,
            SubtaskStatus.RUNNING,
            Actor.EXECUTOR,
            FailureKind.PERMANENT,
            lease=sub.lease_token,
            action="tool_unavailable",
            reason="Search needs setup",
            details={"tool": "web_search", "missing": "search_engine"},
        )
    sessions = Sessions(store.conn, clock=clock)
    chat = Chat(sessions)
    chat.session_id = sessions.add(None, SessionRole.USER, "Prices").session_id
    hint = search_setup_hint(inbox(store)[0])
    assert hint is not None
    reteller = FakeLLM().on("Retell task", '{"text":"Waiting"}', '{"text":"Still waiting"}')
    for _ in range(2):
        chat._outcome(task_id, task_id[:8], Interlocutor(reteller, "en"))
        output = capsys.readouterr().out
        if _ == 0:
            # Rich wraps the long hint on terminal width: check its opening text.
            assert output.index("Second waiting item") < output.index("Set up a search engine")
        else:
            assert "Set up a search engine" not in output
    assert sum(message.text == hint for message in sessions.messages(chat.session_id)) == 1
    # Prose alone never selects the setup hint.
    item = inbox(store)[0].model_copy(
        update={"action": "execution_failed", "details": {}, "reason": "[missing_search_engine]"}
    )
    assert search_setup_hint(item) is None
