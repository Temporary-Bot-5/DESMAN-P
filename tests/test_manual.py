"""Tests for the interlocutor's manual (ROADMAP C.10): its texts against the code, and the
lookup turn that reads them."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

import click
import pytest
from click.testing import CliRunner, Result

from desman import chat, commands, manual, runs
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core import db
from desman.core.profile import Profile
from desman.core.sessions import Role, Sessions
from desman.interlocutor import CHAT, LOOKED_UP, LOOKUP, Interlocutor
from desman.llm.base import Request
from desman.llm.fake import FakeLLM
from helpers import make_config

# ── The texts ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("topic", manual.TOPICS, ids=lambda topic: topic.name)
def test_every_command_a_topic_names_exists(topic: manual.Topic) -> None:
    """Whatever a topic writes in backticks is a command of the CLI and the catalog, typed as
    it would work (`desman models show MODEL`, `desman fail SUBTASK --note TEXT`), one of the
    chat's (`/compact`), or bare `desman`: nothing it names is made up or incomplete."""
    chat_commands = set(chat._commands())
    for quoted in re.findall(r"`([^`]*)`", topic.text):
        if quoted == "desman" or quoted in chat_commands:
            continue
        if re.fullmatch(r"desman( -\w)+", quoted):  # the chat's own options: `desman -t`
            assert typed(cli, quoted.split()[1:]), f"{topic.name}: `{quoted}`"
            continue
        assert quoted.startswith("desman "), f"{topic.name}: `{quoted}` is not a command"
        words = quoted.split()[1:]
        name = list(itertools.takewhile(lambda w: not w.startswith("-") and not w.isupper(), words))
        entry = commands.find(" ".join(name))
        assert entry is not None, f"{topic.name}: no command `{quoted}`"
        command = commands.cli_command(cli, entry.name)
        assert command is not None, f"{topic.name}: `{quoted}` is not in the CLI"
        assert typed(command, words[len(name) :]), f"{topic.name}: `{quoted}` would not run"
    assert topic.text.count("`") % 2 == 0, f"{topic.name}: an unclosed backtick"


def typed(command: click.Command, words: list[str]) -> bool:
    """Whether `words` (options; upper-case placeholders for values) are what `command` takes:
    known options, each given its value, every required argument and option there."""
    by_opt = {o: p for p in command.params for o in (*p.opts, *p.secondary_opts)}
    arguments = [p for p in command.params if isinstance(p, click.Argument)]
    seen: set[str | None] = set()
    positional = 0
    rest = iter(words)
    for word in rest:
        if word.isupper():
            positional += 1
        elif word in by_opt:
            option = by_opt[word]
            seen.add(option.name)
            takes_value = isinstance(option, click.Option) and not option.is_flag
            if takes_value and not next(rest, "").isupper():
                return False  # an option that takes a value, without one
        else:
            return False
    needed = [p.name for p in command.params if isinstance(p, click.Option) and p.required]
    return sum(a.required for a in arguments) <= positional <= len(arguments) and all(
        name in seen for name in needed
    )


@pytest.mark.parametrize("topic", manual.TOPICS, ids=lambda topic: topic.name)
def test_every_topic_is_short_and_listed_in_one_line(topic: manual.Topic) -> None:
    assert len(topic.text) <= manual.TOPIC_CHARS, f"{topic.name}: {len(topic.text)} chars"
    assert re.fullmatch(r"[a-z]+", topic.name)
    assert "\n" not in topic.about and len(topic.about) <= 100


def test_topics_have_unique_names_and_the_prompt_lists_each() -> None:
    assert len(manual.BY_NAME) == len(manual.TOPICS)
    listed = manual.contents().splitlines()
    assert [line.split(":")[0] for line in listed] == [f"- {t.name}" for t in manual.TOPICS]


# ── The lookup turn ──────────────────────────────────────────────────────────

REPLY = '{"type": "reply", "text": "A subtask that runs out of attempts waits for you."}'


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def talk(*lines: str) -> Result:
    result = CliRunner().invoke(cli, [], input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    return result


def turns(llm: FakeLLM) -> list[Request]:
    return [r for r in llm.calls if r.prompt.startswith(CHAT)]


def payload(request: Request) -> Any:
    return json.loads(request.prompt.split("\n", 1)[1])


def system(request: Request) -> str:
    return request.messages[0].content


def said() -> list[tuple[Role, str]]:
    sessions = Sessions(db.open_db(db.default_db_path()))
    latest = sessions.latest()
    assert latest is not None
    return [(m.role, m.text) for m in sessions.messages(latest.id)]


def test_the_prompt_lists_the_topics_not_their_texts(fake: FakeLLM) -> None:
    fake.on(CHAT, REPLY)
    talk("hi")
    (request,) = turns(fake)
    assert manual.contents() in system(request)
    assert LOOKUP.split("{topics}")[0] in system(request)
    assert manual.BY_NAME["retries"].text not in request.prompt + system(request)


def test_a_lookup_asks_again_with_the_topics_texts(fake: FakeLLM) -> None:
    fake.on(CHAT, '{"type": "lookup", "topics": ["retries", " Tasks"]}', REPLY)
    result = talk("what happens when a task keeps failing?")
    first, second = turns(fake)
    assert "manual" not in payload(first)
    assert payload(second)["manual"] == {
        "retries": manual.BY_NAME["retries"].text,
        "tasks": manual.BY_NAME["tasks"].text,
    }
    assert payload(second)["conversation"] == payload(first)["conversation"]
    assert LOOKED_UP in system(second) and manual.contents() not in system(second)
    assert "A subtask that runs out of attempts waits for you." in result.output
    assert said() == [  # the lookup itself is not part of the conversation
        (Role.USER, "what happens when a task keeps failing?"),
        (Role.INTERLOCUTOR, "A subtask that runs out of attempts waits for you."),
    ]


@pytest.mark.parametrize(
    "lookup",
    [
        '{"type": "lookup", "topics": ["weather"]}',
        '{"type": "lookup", "topics": []}',
        '{"type": "lookup", "topics": ["tasks", "retries", "chat"]}',
    ],
    ids=["unknown topic", "no topic", "too many"],
)
def test_a_bad_lookup_is_repaired_once(fake: FakeLLM, lookup: str) -> None:
    fake.on(CHAT, lookup, '{"type": "lookup", "topics": ["chat"]}', REPLY)
    talk("how do sessions work?")
    first, repair, second = turns(fake)
    assert "topics" in repair.prompt  # the model was shown what was wrong
    assert list(payload(second)["manual"]) == ["chat"]


def test_after_a_lookup_another_one_is_refused(fake: FakeLLM) -> None:
    again = '{"type": "lookup", "topics": ["chat"]}'
    fake.on(CHAT, '{"type": "lookup", "topics": ["tasks"]}', again, again)
    result = talk("tell me about tasks")
    assert len(turns(fake)) == 3  # the lookup, the turn with the texts, its one repair
    assert "made no sense" in result.output
    assert said() == [(Role.USER, "tell me about tasks")]  # the message is kept


def test_profile_edits_given_with_a_lookup_are_kept(fake: FakeLLM) -> None:
    config = make_config(language="en", configured=True)
    config.personalize = True
    ConfigManager(default_config_path()).save(config)
    lookup = {"type": "lookup", "topics": ["profile"], "profile": [{"add": "Likes it short"}]}
    answer = {
        "type": "reply",
        "text": "I keep notes.",
        "profile": [{"add": "Likes it short"}, {"add": "Asks about DESMAN"}],
    }
    fake.on(CHAT, json.dumps(lookup), json.dumps(answer))
    result = talk("forget nothing; how do you note things? and be short")
    second = turns(fake)[1]
    assert payload(second)["profile_edits_given"] == [{"add": "Likes it short"}]
    notes = Profile(db.open_db(db.default_db_path())).notes()
    assert [n.text for n in notes] == ["Likes it short", "Asks about DESMAN"]  # once each
    assert "I keep notes." in result.output


def test_the_topics_texts_take_the_place_of_older_messages() -> None:
    sessions = Sessions(db.open_db(":memory:"))
    session = None
    for n in range(40):  # far more than the interlocutor's budget
        session = sessions.add(session, Role.USER, f"message {n} " + "word " * 80).session_id
    assert session is not None
    llm = FakeLLM().on(CHAT, '{"type": "lookup", "topics": ["providers", "tasks"]}', REPLY)
    Interlocutor(llm, "en").turn(sessions.history(session, last=100))
    first, second = turns(llm)
    assert len(second.prompt) <= len(first.prompt) + 400  # about as large as the first
    assert len(payload(second)["conversation"]) < len(payload(first)["conversation"])
    assert payload(second)["conversation"][-1] == payload(first)["conversation"][-1]
