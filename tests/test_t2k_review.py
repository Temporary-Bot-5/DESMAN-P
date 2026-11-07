"""Review regressions: native Ollama, ordered fallback and finished paused work."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from desman.core.calls import Role
from desman.core.coordinator import Coordinator, Review, Success, Work
from desman.core.models import Subtask, SubtaskSpec, TaskSpec
from desman.core.states import SubtaskStatus as S
from desman.core.store import Store
from desman.llm import router
from desman.llm.base import Message, ProviderRejected, ProviderUnavailable, Request
from desman.llm.openai_compat import CallRecord
from helpers import make_config

REQUEST = Request(messages=[Message(role="user", content="hello")])


def test_native_ollama_reply_and_journal() -> None:
    seen = []
    records: list[CallRecord] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "tiny",
                "message": {"content": "answer"},
                "prompt_eval_count": 21,
                "eval_count": 5,
                "done_reason": "length",
            },
        )

    config = make_config(settings={"OLLAMA_MODEL": "tiny", "OLLAMA_CONTEXT": "8192"})
    [choice] = router.lineup(config, Role.INTERLOCUTOR)
    llm = router.connect(
        config,
        choice,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        on_call=records.append,
    )
    result = llm.complete(REQUEST)
    assert str(seen[0].url) == "http://localhost:11434/api/chat"
    body = json.loads(seen[0].content)
    assert body["stream"] is False
    assert body["options"]["num_ctx"] == 8192
    assert "format" not in body
    assert result.text == "answer" and result.finish_reason == "length"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (21, 5)
    [record] = records
    assert record.provider == "ollama" and record.key_id is None
    assert record.prompt_hash and record.response_hash
    assert (record.prompt_tokens, record.completion_tokens) == (21, 5)


@pytest.mark.parametrize("status", [404, None])
def test_native_ollama_errors(status: int | None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if status is None:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(status, json={"error": "model not found"})

    config = make_config(settings={"OLLAMA_MODEL": "tiny"})
    [choice] = router.lineup(config, Role.INTERLOCUTOR)
    records: list[CallRecord] = []
    llm = router.connect(
        config,
        choice,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        on_call=records.append,
    )
    with pytest.raises(ProviderUnavailable if status is None else ProviderRejected) as caught:
        llm.complete(REQUEST)
    assert isinstance(caught.value, (ProviderRejected, ProviderUnavailable))
    assert caught.value.status == status
    assert records[0].http_status == status


def test_pause_after_execution_submits_finished_result(store: Store) -> None:
    ref = store.create_task(
        TaskSpec(
            goal="goal",
            subtasks=[
                SubtaskSpec(description="first", acceptance_criteria="ok"),
                SubtaskSpec(description="second", acceptance_criteria="ok"),
            ],
        )
    )
    seen = []

    def execute(work: Work) -> Success:
        seen.append(work.subtask.id)
        store.pause_task(ref)
        return Success(content={"answer": "done"})

    def review(subtask: Subtask, content: dict[str, Any] | None) -> Review:
        pytest.fail("paused reviews must wait")

    coordinator = Coordinator(store, execute, review)
    coordinator.run(ref)
    subs = store.list_subtasks(ref)
    assert len(seen) == 1
    assert subs[0].status is S.REVIEW
    assert subs[1].status is S.READY
    assert store.claim("next", task_id=ref) is None
    store.resume_task(ref)
    claimed = store.claim("next", task_id=ref)
    assert claimed and claimed.id == subs[1].id


@pytest.mark.parametrize("small_first", [True, False])
def test_fallback_only_reports_preceding_models_in_order(small_first: bool) -> None:
    from desman import providers
    from desman.llm import catalog
    from desman.llm.fake import FakeLLM, Rejected

    provider = providers.find("ollama")
    small = (
        FakeLLM("never", name="small"),
        router.Choice(provider, "small", catalog.local_model("small", 100)),
    )
    refused = (
        FakeLLM(Rejected(status=404), name="refused"),
        router.Choice(provider, "refused", catalog.local_model("refused", 16384)),
    )
    answer = (
        FakeLLM("answer", name="answer"),
        router.Choice(provider, "answer", catalog.local_model("answer", 16384)),
    )
    lineup = [small, refused, answer] if small_first else [refused, answer, small]
    llm = router.Fallback(lineup)
    llm.complete(REQUEST)
    assert [s.model for s in llm.last_call.skipped] == (
        ["ollama/small", "ollama/refused"] if small_first else ["ollama/refused"]
    )
    assert llm.last_call.skipped[-1].reason == "refused"
    if small_first:
        assert llm.last_call.skipped[0].needed == router.estimated_tokens(REQUEST)
        assert llm.last_call.skipped[0].context == 100


def test_cached_ollama_context_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from desman import local_models, providers
    from desman.llm.hub import InstalledDetails, InstalledModel

    monkeypatch.setattr(
        local_models,
        "_tags_cache",
        {
            "http://localhost:11434": (
                0,
                [InstalledModel(name="tiny", details=InstalledDetails(context_length=8192))],
                True,
            )
        },
    )
    config = make_config(settings={"OLLAMA_MODEL": "tiny"})
    [choice] = router.lineup(config, Role.INTERLOCUTOR)
    assert choice.spec.context == 8192
    assert router.connect(config, choice).context == 8192
    config.settings["OLLAMA_CONTEXT"] = "32768"
    [choice] = router.lineup(config, Role.INTERLOCUTOR)
    assert choice.spec.context == 32768
    assert router.connect(config, choice).context == 32768
    assert providers.find("llamacpp").context(config) == 4096


def test_groq_tts_models_are_filtered() -> None:
    from desman import providers
    from desman.llm.check import chat_models

    assert chat_models(
        providers.find("groq"), ("canopylabs/orpheus-v1-english", "playai-tts", "chat")
    ) == ("chat",)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, "limit reached"),
        (404, "no such model or no access"),
        (401, "key refused"),
        (403, "key refused"),
        (503, "unavailable"),
    ],
)
def test_chat_fallback_reasons_are_plain_words(store: Store, status: int, expected: str) -> None:
    from desman import providers, ui
    from desman.chat import Chat
    from desman.core.sessions import Sessions
    from desman.gui.front import WebFront
    from desman.interlocutor import Interlocutor
    from desman.llm.fake import FakeLLM
    from desman.llm.openai_compat import OpenAICompatible

    first = OpenAICompatible(
        "groq",
        "https://example.test/v1",
        ["secret"],
        "first",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    status,
                    headers={"x-ratelimit-limit-requests": "0"},
                    json={"error": "account limit 0"},
                )
            )
        ),
    )
    provider = providers.find("groq")
    llm = router.Fallback(
        [
            (first, router.Choice(provider, "first")),
            (FakeLLM("hello"), router.Choice(provider, "second")),
        ]
    )
    llm.complete(REQUEST)
    chat = Chat(Sessions(store.conn))
    front = WebFront()
    with ui.use(front):
        chat._fallback_note(Interlocutor(llm, "en"))
    rendered = str(front.events)
    assert f"groq/first: {expected}" in rendered
    assert "secret" not in rendered and "account limit 0" not in rendered


def test_tags_keep_model_context_metadata() -> None:
    from desman.llm.hub import ollama_tags

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "tiny", "details": {"context_length": 2048, "family": "llama"}},
                    {"name": "unknown"},
                ]
            },
        )

    tags = ollama_tags(
        "http://localhost:11434", httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert tags[0].details and tags[0].details.context_length == 2048
    assert tags[1].details is None


@pytest.mark.parametrize(
    "body",
    [{}, {"message": {"content": None}}, {"message": {"content": ""}, "prompt_eval_count": 2}],
)
def test_native_ollama_unusable_reply_is_journaled(body: dict[str, Any]) -> None:
    from desman.llm.openai_compat import OpenAICompatible

    records: list[CallRecord] = []
    llm = OpenAICompatible(
        "ollama",
        "http://localhost:11434/v1",
        [""],
        "tiny",
        on_call=records.append,
        client=httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    )
    if body.get("message", {}).get("content") == "":
        assert llm.complete(REQUEST).text == ""
        assert records[0].prompt_tokens == 2
    else:
        with pytest.raises(ProviderUnavailable):
            llm.complete(REQUEST)
        assert records[0].error_kind == "transient"
