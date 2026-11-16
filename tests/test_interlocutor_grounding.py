"""A guessed command explanation gets one chance to read the knowledge tree."""

import json

import pytest

from desman.core.sessions import History
from desman.interlocutor import Interlocutor, Reply, names_desman_commands
from desman.llm.fake import FakeLLM, Rejected


def reply(text: str) -> str:
    return json.dumps({"type": "reply", "text": text})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Try `keys check`.", True),
        ("It becomes `needs_human`.", True),
        ("Use desman worker.", True),
        ("Use /inbox", True),
        ("Use `desman resume TASK`", True),
        ("Use `/global quota`", True),
        ("у моделей закончились ключи", False),
        ("Check your keys and models", False),
        ("`unknown command`", False),
        ("https://example.com/inbox", False),
        ("Use `list` and `run` in your loop.", False),
        ("```\ndesman worker\n```", False),
        ("Call `status` on the object.", False),
    ],
)
def test_explicit_mentions(text: str, expected: bool) -> None:
    assert names_desman_commands(text) is expected


def test_correction_reads_node_before_answering() -> None:
    def answer(request: object) -> str:
        from desman.llm.base import Request

        assert isinstance(request, Request)
        assert '"knowledge"' in request.prompt
        assert "≤15 min" in request.prompt
        return reply("Use `desman resume --wait` to wait longer.")

    llm = FakeLLM(
        reply("Try `keys check`."),
        '{"type": "lookup", "nodes": ["DESMAN/Manual/retries"]}',
        answer,
    )
    interlocutor = Interlocutor(llm, "en")
    result = interlocutor.turn(History(summary=None, messages=[]))
    assert isinstance(result, Reply)
    assert result.text == "Use `desman resume --wait` to wait longer."
    assert interlocutor.opened == ["DESMAN/Manual/retries"]
    assert len(llm.calls) == 3
    assert "knowledge correction" in llm.calls[1].prompt


@pytest.mark.parametrize("failure", [False, True])
def test_plain_correction_or_provider_failure_keeps_first(failure: bool) -> None:
    llm = FakeLLM(reply("Try `keys check`."), Rejected() if failure else reply("Second guess"))
    result = Interlocutor(llm, "en").turn(History(summary=None, messages=[]))
    assert isinstance(result, Reply)
    assert result.text == "Try `keys check`."
    assert len(llm.calls) == 2


def test_invalid_correction_keeps_first() -> None:
    llm = FakeLLM(reply("Try `keys check`."), "{}", "{}")
    result = Interlocutor(llm, "en").turn(History(summary=None, messages=[]))
    assert isinstance(result, Reply)
    assert result.text == "Try `keys check`."
    assert len(llm.calls) == 3  # complete_json's single schema repair


@pytest.mark.parametrize(
    "turn",
    [
        reply("Hello there"),
        '{"type": "task", "title": "keys check", "goal": "Explain `keys check`"}',
        '{"type": "command", "name": "keys check", "args": {}}',
    ],
)
def test_other_turns_need_no_correction(turn: str) -> None:
    llm = FakeLLM(turn)
    Interlocutor(llm, "en").turn(History(summary=None, messages=[]))
    assert len(llm.calls) == 1


def test_clarification_also_gets_a_lookup_correction() -> None:
    llm = FakeLLM(
        '{"type": "clarify", "question": "Did it become `needs_human`?"}',
        '{"type": "lookup", "nodes": ["Commands/keys"]}',
        reply("Read the key check result."),
    )
    interlocutor = Interlocutor(llm, "en")
    assert isinstance(interlocutor.turn(History(summary=None, messages=[])), Reply)
    assert interlocutor.opened == ["Commands/keys"]
    assert len(llm.calls) == 3


def test_normal_lookup_answer_is_never_rechecked() -> None:
    llm = FakeLLM(
        '{"type": "lookup", "nodes": ["DESMAN/Manual/retries"]}',
        reply("Try `desman resume --wait`."),
    )
    result = Interlocutor(llm, "en").turn(History(summary=None, messages=[]))
    assert isinstance(result, Reply)
    assert result.text == "Try `desman resume --wait`."
    assert len(llm.calls) == 2
