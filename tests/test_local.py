"""Local models: Ollama and llama.cpp behind the router, paced by the machine (ROADMAP 4.5a)."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from conftest import FakeClock
from desman import providers, runs
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.core.calls import Role
from desman.core.models import SubtaskSpec, TaskSpec
from desman.core.states import TaskStatus
from desman.core.store import DEFAULT_LEASE_SECONDS, DEFAULT_REVIEW_SECONDS, Store
from desman.llm import check, limiter, router
from desman.llm.base import (
    Completion,
    Message,
    ProviderRejected,
    ProviderUnavailable,
    RateLimited,
    Request,
)
from desman.llm.fake import FakeLLM
from desman.llm.limiter import Buckets, Busy, Local
from helpers import make_config

GROQ = "gsk_" + "a" * 48
OLLAMA = {"OLLAMA_MODEL": "llama3.2"}
REQUEST = Request(messages=[Message(role="user", content="hi")])
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture(autouse=True)
def quiet_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slots in the test's own directory, and a machine with room to spare."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(limiter, "machine_load", lambda: 0.1)


def ok(text: str = '{"v": 1}', model: str = "llama3.2") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


# ── Configuration ────────────────────────────────────────────────────────────


def test_a_local_provider_is_ready_once_its_model_is_set() -> None:
    ollama = providers.find("ollama")
    assert not ollama.is_ready(make_config())
    assert providers.ready(make_config()) == []
    config = make_config(settings=OLLAMA)
    assert [p.id for p in providers.ready(config)] == ["ollama"]
    assert ollama.endpoint(config) == "http://localhost:11434/v1"  # the default address
    assert ollama.model(config) == "llama3.2" and ollama.context(config) == 4096
    described = {row["id"]: row for row in providers.describe(config)["providers"]}
    assert described["ollama"]["ready"] and not described["llamacpp"]["ready"]


def test_a_host_without_a_scheme_and_a_bad_context_are_understood() -> None:
    config = make_config(settings={**OLLAMA, "OLLAMA_HOST": "box:11434", "OLLAMA_CONTEXT": "lots"})
    ollama = providers.find("ollama")
    assert ollama.endpoint(config) == "http://box:11434/v1"
    assert ollama.context(config) is None
    llamacpp = providers.find("llamacpp")
    assert llamacpp.endpoint(config) == "http://localhost:8080/v1"


# ── Routing ──────────────────────────────────────────────────────────────────


def test_local_models_come_after_every_cloud_one() -> None:
    config = make_config(api_keys={"GROQ_API_KEY": GROQ}, settings=OLLAMA)
    for role in Role:
        lineup = router.lineup(config, role)
        assert lineup[-1].label == "ollama/llama3.2"
        assert all(not c.provider.local for c in lineup[:-1])
    assert router.choose(config).id == "groq"


def test_a_local_model_alone_is_a_lineup() -> None:
    config = make_config(settings={**OLLAMA, "OLLAMA_CONTEXT": "16384"})
    [choice] = router.lineup(config, Role.PLANNER)
    assert choice.label == "ollama/llama3.2"
    assert choice.spec.context == 16_384 and not choice.spec.strict_json
    assert router.other_lineup([choice]) == []
    [pinned] = router.lineup(config, Role.EXECUTOR, "ollama", "qwen3:8b")
    assert pinned.model == "qwen3:8b" and pinned.spec.context == 16_384


def test_a_local_provider_without_a_model_says_where_to_set_it() -> None:
    with pytest.raises(router.NoProvider, match="llama.cpp has no model set: choose one"):
        router.choose(make_config(), "llamacpp")


def test_a_local_server_is_called_without_a_key_and_with_a_long_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok()

    config = make_config(settings={"LLAMA_CPP_MODEL": "qwen", "LLAMA_CPP_URL": "http://gpu:9000"})
    records: list[Any] = []
    [choice] = router.lineup(config, Role.EXECUTOR)
    llm = router.connect(
        config,
        choice,
        on_call=records.append,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert llm.timeout == providers.LOCAL_TIMEOUT
    assert llm.complete(REQUEST).text == '{"v": 1}'
    [request] = seen
    assert str(request.url) == "http://gpu:9000/v1/chat/completions"
    assert "authorization" not in request.headers
    assert json.loads(request.content)["model"] == "qwen"
    [record] = records
    assert record.key_id is None and record.provider == "llamacpp"


def test_a_model_the_server_does_not_have_moves_the_call_on() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": 'model "x" not found'}})

    config = make_config(settings={"OLLAMA_MODEL": "x"})
    [choice] = router.lineup(config, Role.EXECUTOR)
    llm = router.connect(
        config, choice, client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ProviderRejected, match="not found"):
        llm.complete(REQUEST)


# ── Paced by the machine ─────────────────────────────────────────────────────


class Down:
    """A server that is not running: every call is refused at once."""

    name = "ollama"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: Request) -> Completion:
        self.calls += 1
        raise ProviderUnavailable("ollama", None, "ConnectError")


def local(store: Store, clock: FakeClock, llm: Any, tmp_path: Path, **kw: Any) -> Local:
    return Local(
        llm,
        Buckets(store.conn, clock),
        "ollama/llama3.2",
        "ollama@http://localhost:11434/v1",
        locks=tmp_path / "locks",
        sleep=clock.sleep,
        **kw,
    )


def test_a_server_that_does_not_answer_is_left_alone_for_a_while(
    store: Store, clock: FakeClock, tmp_path: Path
) -> None:
    down = Down()
    paced = local(store, clock, down, tmp_path)
    with pytest.raises(ProviderUnavailable):
        paced.complete(REQUEST)
    # Every worker on the database knows: nothing is sent until the pause is over.
    other = local(store, clock, down, tmp_path)
    with pytest.raises(Busy, match="did not answer") as e:
        other.complete(REQUEST)
    assert e.value.retry_after == limiter.DOWN_PAUSE and down.calls == 1
    clock.advance(limiter.DOWN_PAUSE)
    with pytest.raises(ProviderUnavailable):
        paced.complete(REQUEST)
    assert down.calls == 2


def test_a_busy_machine_gets_no_request(store: Store, clock: FakeClock, tmp_path: Path) -> None:
    fake = FakeLLM("ok", repeat_last=True)
    paced = local(store, clock, fake, tmp_path, load=lambda: limiter.MAX_LOAD + 0.5)
    with pytest.raises(Busy, match="machine is busy") as e:
        paced.complete(REQUEST)
    assert isinstance(e.value, RateLimited) and e.value.retry_after == limiter.LOAD_PAUSE
    assert fake.calls == []
    assert local(store, clock, fake, tmp_path, load=lambda: None).complete(REQUEST)


@pytest.fixture
def held(tmp_path: Path) -> Iterator[Callable[[str], None]]:
    """Takes the server's slot as another process would."""
    handles: list[Any] = []

    def take(server: str) -> None:
        import hashlib

        folder = tmp_path / "locks"
        folder.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(server.encode()).hexdigest()[:16]
        handle = open(folder / f"{name}.0.lock", "a")  # noqa: SIM115
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)

    yield take
    for handle in handles:
        handle.close()


def test_one_request_at_a_time_on_a_server(
    store: Store, clock: FakeClock, tmp_path: Path, held: Callable[[str], None]
) -> None:
    fake = FakeLLM("ok", repeat_last=True)
    paced = local(store, clock, fake, tmp_path)
    paced.complete(REQUEST)
    paced.complete(REQUEST)  # the slot is let go after each call
    assert len(fake.calls) == 2
    held("ollama@http://localhost:11434/v1")
    started = clock()
    with pytest.raises(Busy, match="answering another request") as e:
        paced.complete(REQUEST)
    assert e.value.retry_after == limiter.BUSY_PAUSE
    assert (clock() - started).total_seconds() == limiter.MAX_WAIT  # waited, then gave up
    assert len(fake.calls) == 2
    # Another server on the same machine has a slot of its own.
    elsewhere = Local(
        fake, Buckets(store.conn, clock), "s", "llamacpp@http://x/v1", locks=tmp_path / "locks"
    )
    elsewhere.complete(REQUEST)
    assert len(fake.calls) == 3


# ── keys check ───────────────────────────────────────────────────────────────


def checked(handler: Handler, **settings: str) -> check.KeyCheck:
    config = make_config(settings=settings)
    p = providers.find("llamacpp" if "LLAMA_CPP_MODEL" in settings else "ollama")
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        [result] = check.check(config, p, http)
    return result


def models(*ids: str) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET" and request.url.path == "/v1/models"
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})

    return handler


def test_check_finds_the_model_on_a_running_server() -> None:
    result = checked(models("llama3.2:latest", "qwen3:8b"), **OLLAMA)
    assert result.ok and result.default_model_listed and result.models == 2
    assert result.model == "llama3.2" and result.key_id == ""
    assert checked(models("qwen3:8b"), OLLAMA_MODEL="qwen3:8b").ok
    # llama-server lists the file it loaded.
    assert checked(models("/models/qwen.gguf"), LLAMA_CPP_MODEL="qwen.gguf").ok


def test_check_says_what_is_wrong_with_a_local_server() -> None:
    missing = checked(models("qwen3:8b"), **OLLAMA)
    assert not missing.ok and missing.default_model_listed is False
    assert missing.served == ("qwen3:8b",) and "does not serve llama3.2" in missing.detail

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    down = checked(refused, **OLLAMA)
    assert not down.ok and down.http_status is None
    assert "not running at http://localhost:11434/v1" in down.detail
    page = checked(lambda r: httpx.Response(200, text="<html>"), **OLLAMA)
    assert not page.ok and "not an Ollama" in page.detail


def test_keys_check_reports_local_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    ConfigManager(default_config_path()).save(make_config(settings=OLLAMA))
    real = httpx.Client
    handler = models("qwen3:8b")
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    result = desman("keys", "check")
    assert result.exit_code == 1, result.output
    assert "It serves: qwen3:8b" in result.output
    assert "ollama pull llama3.2" in result.output
    handler = models("llama3.2:latest")
    assert desman("keys", "check", "ollama").exit_code == 0
    [row] = json.loads(desman("keys", "check", "--json").output)
    assert row["ok"] and row["provider"] == "ollama" and row["model"] == "llama3.2"
    assert desman("keys", "check", "llamacpp").exit_code == 1  # no model set


# ── A run on a local model ───────────────────────────────────────────────────


def test_a_task_runs_on_a_local_model_alone(store: Store, clock: FakeClock) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(f"{request.url.host}:{request.url.port} {body['model']}")
        prompt = body["messages"][-1]["content"]
        return ok('{"verdict": "accept"}' if prompt.startswith("Review") else '{"v": 1}')

    models = runs.models(
        make_config(settings=OLLAMA),
        store.conn,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock,
        sleep=clock.sleep,
        cache=False,
    )
    assert models.choice is not None and models.choice.label == "ollama/llama3.2"
    assert models.other is None
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    assert seen == ["localhost:11434 llama3.2"] * 2


def test_when_the_cloud_is_limited_the_local_model_answers(store: Store, clock: FakeClock) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url.host))
        if request.url.host == "api.groq.com":
            return httpx.Response(429, headers={"retry-after": "3600"})
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return ok('{"verdict": "accept"}' if prompt.startswith("Review") else '{"v": 1}')

    models = runs.models(
        make_config(api_keys={"GROQ_API_KEY": GROQ}, settings=OLLAMA),
        store.conn,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock,
        sleep=clock.sleep,
        cache=False,
    )
    task_id = store.create_task(
        TaskSpec(goal="g", subtasks=[SubtaskSpec(description="d", acceptance_criteria="c")])
    )
    assert runs.summarizer(store, models).run(task_id).status is TaskStatus.DONE
    assert seen.count("localhost") == 2


def test_run_names_the_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    ConfigManager(default_config_path()).save(make_config(settings=OLLAMA))
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return ok('{"verdict": "accept"}' if prompt.startswith("Review") else '{"answer": 42}')

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    result = desman("run", "answer the question", "--plan", "template")
    assert result.exit_code == 0, result.output
    assert "Using Ollama · llama3.2 (on this machine)" in result.output


# ── Devin's review of #56 ────────────────────────────────────────────────────


def test_a_slow_model_gets_a_lease_and_review_reservation_that_outlast_its_step(
    store: Store, clock: FakeClock
) -> None:
    def build(config: Any) -> Any:
        models = runs.models(config, store.conn, clock=clock, sleep=clock.sleep, cache=False)
        return models, runs.summarizer(store, models)

    models, coordinator = build(make_config(api_keys={"GROQ_API_KEY": GROQ}, settings=OLLAMA))
    # Two calls (a request and its repair), each with its waits, and a margin.
    assert models.step_seconds == 2 * (providers.LOCAL_TIMEOUT + 30) + 30
    assert coordinator.lease_seconds == models.step_seconds
    assert coordinator.review_seconds == models.step_seconds
    models, coordinator = build(make_config(api_keys={"GROQ_API_KEY": GROQ}))
    assert coordinator.lease_seconds == DEFAULT_LEASE_SECONDS  # already long enough
    assert coordinator.review_seconds == models.step_seconds > DEFAULT_REVIEW_SECONDS
    fake = runs.summarizer(store, runs.Models(FakeLLM(), None))
    assert fake.review_seconds == DEFAULT_REVIEW_SECONDS


def test_another_server_does_not_get_the_replies_of_the_first(
    store: Store, clock: FakeClock
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url.host))
        return ok()

    def ask(url: str) -> None:
        config = make_config(settings={"LLAMA_CPP_MODEL": "assistant", "LLAMA_CPP_URL": url})
        models = runs.models(
            config,
            store.conn,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            clock=clock,
            sleep=clock.sleep,
        )
        llm: Any = models.for_(Role.PLANNER)
        reply = llm.complete(REQUEST)
        llm.accept(REQUEST, reply)

    ask("http://localhost:8080")
    ask("http://localhost:8080")  # kept
    ask("http://gpu-box:8080")
    assert seen == ["localhost", "gpu-box"]


def test_only_a_loopback_address_is_this_machine() -> None:
    for url in ("http://localhost:11434", "127.0.0.1:8080", "http://[::1]:8080", "a.localhost"):
        assert providers.on_this_machine(url), url
    for url in ("http://gpu-box:8080", "http://192.168.1.5:11434", "10.0.0.2:8080", ""):
        assert not providers.on_this_machine(url), url


def test_run_says_where_a_remote_local_server_is(monkeypatch: pytest.MonkeyPatch) -> None:
    ConfigManager(default_config_path()).save(
        make_config(settings={**OLLAMA, "OLLAMA_HOST": "gpu-box:11434"})
    )
    real = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return ok('{"verdict": "accept"}' if prompt.startswith("Review") else '{"answer": 42}')

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    result = desman("run", "answer the question", "--plan", "template")
    assert result.exit_code == 0, result.output
    assert "Using Ollama · llama3.2 (server at http://gpu-box:11434/v1)" in result.output


def test_keys_check_words_local_problems_in_the_persons_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ConfigManager(default_config_path()).save(make_config(language="ru", settings=OLLAMA))
    real = httpx.Client

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(refused), **kw)
    )
    result = desman("keys", "check", "ollama")
    assert result.exit_code == 1
    assert "не запущен по адресу http://localhost:11434/v1" in result.output
    [row] = json.loads(desman("keys", "check", "--json").output)
    assert row["detail"].startswith("not running at")  # --json stays English
