"""Tests for structured output: schema validation and the single repair request (2.3)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from desman.llm.base import Message, RateLimited, Request
from desman.llm.fake import FakeLLM, RateLimit, garbage
from desman.llm.structured import MAX_PROBLEMS, StructuredOutputError, complete_json


class Answer(BaseModel):
    answer: int
    unit: str = Field(min_length=1)


def ask(llm: FakeLLM) -> Answer:
    request = Request(
        messages=[Message(role="system", content="s"), Message(role="user", content="Ask:\n{}")]
    )
    return complete_json(llm, request, Answer, heading="Ask")


def test_a_valid_reply_needs_no_repair() -> None:
    llm = FakeLLM('```json\n{"answer": 42, "unit": "m"}\n```')
    assert ask(llm) == Answer(answer=42, unit="m")
    assert len(llm.calls) == 1


def test_one_repair_fixes_the_reply() -> None:
    llm = FakeLLM('{"answer": "many"}', '{"answer": 42, "unit": "m"}')
    assert ask(llm) == Answer(answer=42, unit="m")
    repair = llm.calls[1]
    assert repair.prompt.startswith("Ask (repair):\n")
    assert "- answer: Input should be a valid integer" in repair.prompt
    assert "- unit: Field required" in repair.prompt
    assert "many" not in repair.prompt  # input values stay out; the model has its reply
    assert repair.temperature == llm.calls[0].temperature


def test_a_second_failure_is_raised_and_no_third_call_is_made() -> None:
    llm = FakeLLM(garbage("prose"), garbage("trailing_comma"))
    with pytest.raises(StructuredOutputError) as caught:
        ask(llm)
    assert caught.value.problem == "The reply is not a JSON object."
    assert caught.value.reply.text == garbage("trailing_comma").text
    assert "after a repair request" in str(caught.value)
    assert len(llm.calls) == 2


def test_a_cut_off_reply_is_told_to_be_shorter() -> None:
    long = '{"answer": 42, "unit": "metres, and then a lot more metres"}'
    llm = FakeLLM(long, '{"answer": 42, "unit": "m"}')  # the second fits in 8 tokens
    request = Request(messages=[Message(role="user", content="Ask:\n{}")], max_tokens=8)
    assert complete_json(llm, request, Answer, heading="Ask").unit == "m"
    assert "cut off by the token limit" in llm.calls[1].prompt


def test_long_error_lists_are_capped() -> None:
    class Many(BaseModel):
        a: int
        b: int
        c: int
        d: int
        e: int
        f: int
        g: int

    llm = FakeLLM("{}", '{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}')
    request = Request(messages=[Message(role="user", content="Ask:\n{}")])
    complete_json(llm, request, Many, heading="Ask")
    problems = [line for line in llm.calls[1].prompt.splitlines() if line.startswith("- ")]
    assert len(problems) == MAX_PROBLEMS + 1
    assert problems[-1] == "- and 2 more"


def test_provider_errors_during_repair_propagate() -> None:
    llm = FakeLLM(garbage("prose"), RateLimit(retry_after=30))
    with pytest.raises(RateLimited):
        ask(llm)
