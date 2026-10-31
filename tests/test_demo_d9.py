"""Demo D9 (ROADMAP C.8): a request in ordinary words in the chat becomes a task whose progress
is seen, and its result comes back to the same chat. On the way the person adds a key and
grants a permission, both through the chat's windows, and the task uses them.

Nothing is faked above HTTP: the router, the call log, the cache, the coordinator, the tools
and the web search run as they do for a person. The provider and Tavily answer through
`httpx.MockTransport`. The manual check on real providers is in `docs/demo-d9.md`.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from desman import chat as chat_module
from desman.agents import EXECUTE, REVIEW, TOOL_RESULTS
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db, operator
from desman.core.sessions import Kind, Role, Sessions
from desman.interlocutor import CHAT, RETELL
from desman.llm import router
from desman.llm.openai_compat import key_id
from desman.planner import PLAN
from helpers import make_config

GROQ_KEY = "gsk_" + "Groq42" * 8
TAVILY_KEY = "tvly-" + "Tav1ly" * 6

FLIGHT = {
    "title": "Lisbon flights from 61 EUR",
    "url": "https://flights.example/lisbon",
    "content": "TP 1351 Berlin → Lisbon, 12 Nov, 09:40, 61 EUR.",
}

# What the model answers, by the heading its prompt starts with, in order.
REPLIES: dict[str, list[dict[str, Any]]] = {
    CHAT: [
        {"type": "command", "name": "keys add", "args": {"provider": "tavily"}},
        {"type": "command", "name": "permissions", "args": {"grant": ["network_access"]}},
        {
            "type": "task",
            "title": "Cheap flights to Lisbon",
            "goal": "Find the cheapest flight from Berlin to Lisbon on 12 November",
        },
    ],
    PLAN: [
        {
            "subtasks": [
                {
                    "description": "Search the web for flights Berlin → Lisbon on 12 November",
                    "acceptance_criteria": "The cheapest flight found, with its price and source",
                }
            ]
        }
    ],
    TOOL_RESULTS: [
        {"flight": "TP 1351, 12 Nov, 09:40", "price": "61 EUR", "source": FLIGHT["url"]}
    ],
    EXECUTE: [
        {
            "tool_calls": [
                {"tool": "web_search", "args": {"query": "cheap flights Berlin Lisbon 12 November"}}
            ]
        }
    ],
    REVIEW: [{"verdict": "accept"}],
    RETELL: [{"text": "The cheapest is TP 1351 on 12 November at 09:40, for 61 EUR."}],
}


class Internet:
    """The provider and Tavily, as far as the demo reaches them."""

    def __init__(self) -> None:
        self.replies = {
            heading: deque(json.dumps(r) for r in rs) for heading, rs in REPLIES.items()
        }
        self.prompts: dict[str, list[str]] = defaultdict(list)
        self.searches: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            self.searches.append(request)
            return httpx.Response(200, json={"results": [FLIGHT]})
        assert request.url.host == "api.groq.com"
        assert request.headers["Authorization"] == f"Bearer {GROQ_KEY}"
        prompt = json.loads(request.content)["messages"][-1]["content"]
        heading = next(h for h in self.replies if prompt.startswith(h))  # TOOL_RESULTS first
        self.prompts[heading].append(prompt)
        return httpx.Response(
            200,
            json={
                "model": "llama",
                "choices": [{"message": {"content": self.replies[heading].popleft()}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
            headers={"x-ratelimit-remaining-requests": "999"},
        )


@pytest.fixture
def internet(monkeypatch: pytest.MonkeyPatch) -> Internet:
    world = Internet()

    def client() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(world))

    build, alternative = router.build, router.alternative
    monkeypatch.setattr(router, "build", lambda *a, **kw: build(*a, **kw, client=client()))
    monkeypatch.setattr(
        router, "alternative", lambda *a, **kw: alternative(*a, **kw, client=client())
    )
    monkeypatch.setattr(chat_module, "client", client)  # the web tools' engines
    return world


def test_a_request_in_ordinary_words_becomes_a_task_seen_working_and_retold_there(
    internet: Internet,
) -> None:
    config = make_config(language="en", configured=True, api_keys={"GROQ_API_KEY": GROQ_KEY})
    ConfigManager(default_config_path()).save(config)
    assert not config.permissions.network_access  # the person grants it below

    typed = [
        "I want you to search the web, here is my Tavily key",
        TAVILY_KEY,  # into the hidden field the window opens
        "and let your agents go online",
        "y",  # the person's "yes" in the window
        "find me the cheapest flight from Berlin to Lisbon on 12 November",
    ]
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in typed))
    assert result.exit_code == 0, result.output
    shown = result.output

    # The key: typed into the hidden field, saved, never shown.
    saved = ConfigManager(default_config_path()).load()
    assert [k.value for k in saved.keys("TAVILY_API_KEY")] == [TAVILY_KEY]
    assert TAVILY_KEY not in shown and GROQ_KEY not in shown
    # The permission: granted by the person's "yes" in the window built by code.
    assert "Apply this change?" in shown
    assert saved.permissions.network_access

    # The task: started, seen working, done, and its result retold in the same chat.
    assert "started: Cheap flights to Lisbon (1 subtask)" in shown
    assert "[0] running  Search the web for flights" in shown
    assert "groq llama (#1 gsk_…" in shown and "executor · [0] · working" in shown
    assert "is done" in shown
    assert "The cheapest is TP 1351 on 12 November at 09:40, for 61 EUR." in shown

    # The work used both: Tavily was asked with the new key, and what it found reached the
    # executor (marked untrusted), whose result the retelling saw.
    [search] = internet.searches
    assert search.headers["Authorization"] == f"Bearer {TAVILY_KEY}"
    assert json.loads(search.content)["query"] == "cheap flights Berlin Lisbon 12 November"
    [tool_results] = internet.prompts[TOOL_RESULTS]
    assert FLIGHT["content"] in tool_results and "untrusted" in tool_results
    # The reviewer checked the result against what was found, not the result alone.
    [review] = internet.prompts[REVIEW]
    assert FLIGHT["content"] in review and '"evidence"' in review
    [retell] = internet.prompts[RETELL]
    assert "61 EUR" in retell and "Find the cheapest flight" in retell
    assert all(not replies for replies in internet.replies.values())  # every step was taken

    conn = db.open_db(db.default_db_path())
    try:
        # One session holds it all, in order.
        sessions = Sessions(conn)
        [session] = sessions.recent()
        said = [(m.role, m.kind) for m in sessions.messages(session.id)]
        [(task_id,)] = conn.execute("SELECT id FROM tasks WHERE session_id = ?", (session.id,))
        assert said == [
            (Role.USER, Kind.MESSAGE),  # "here is my Tavily key" (the key is not in it)
            (Role.DESMAN, Kind.EVENT),  # keys add ran, by fingerprint
            (Role.USER, Kind.MESSAGE),  # "let your agents go online"
            (Role.DESMAN, Kind.EVENT),  # permissions ran
            (Role.USER, Kind.MESSAGE),  # the request
            (Role.DESMAN, Kind.EVENT),  # task started
            (Role.DESMAN, Kind.EVENT),  # task done
            (Role.INTERLOCUTOR, Kind.MESSAGE),  # the retelling
        ]
        # The operator's journal has what the person did, the key by its fingerprint.
        done = [(e.action, e.details) for e in operator.entries(conn)]
        assert ("operator_key_added", {"provider": "tavily", "key_id": key_id(TAVILY_KEY)}) in done
        assert ("operator_permission_grant", {"permission": "network_access"}) in done
        # The tool call is in the task's audit, and every model call is attributed.
        [(details,)] = conn.execute(
            "SELECT details_json FROM audit_log WHERE task_id = ? AND action = 'tool_called'",
            (task_id,),
        )
        assert json.loads(details)["tool"] == "web_search" and json.loads(details)["ok"]
        roles = [r for (r,) in conn.execute("SELECT role FROM model_calls ORDER BY seq")]
        assert roles == [
            "interlocutor",  # keys add
            "interlocutor",  # permissions
            "interlocutor",  # the request → a task
            "planner",
            "executor",  # asks for web_search
            "executor",  # the result, from what was found
            "reviewer",
            "interlocutor",  # the retelling
        ]
        # Neither key is kept anywhere in the database (fingerprints aside).
        dump = "\n".join(conn.iterdump()).replace(key_id(TAVILY_KEY), "")
        assert TAVILY_KEY not in dump and GROQ_KEY not in dump
    finally:
        conn.close()
