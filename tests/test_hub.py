"""Models from Hugging Face for the local servers (ROADMAP 4.5b). No real downloads: the hub's
API and Ollama's are `httpx.MockTransport` handlers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from click.testing import CliRunner, Result

from desman import providers, runs
from desman.cli import cli
from desman.config import ConfigManager, default_config_path
from desman.interlocutor import CHAT
from desman.llm import hub
from desman.llm.fake import FakeLLM
from helpers import make_config
from test_commands import asks, chat, journal, saved

Handler = Callable[[httpx.Request], httpx.Response]
TOKEN = "hf_" + "Tok3n" * 8
REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
GB = 1_000_000_000
PLENTY = hub.Memory(ram=16 * GB, video=None)
SIBLINGS = [
    {"rfilename": "README.md", "size": 10},
    {"rfilename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size": 2 * GB},
    {"rfilename": "Llama-3.2-3B-Instruct-Q8_0.gguf", "size": 3 * GB},
    {"rfilename": "Llama-3.2-3B-Instruct-f16.gguf", "size": 6 * GB},
    {"rfilename": "mmproj-model-f16.gguf", "size": GB},
]


def listing(siblings: list[dict[str, Any]] = SIBLINGS, **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"id": REPO, "siblings": siblings, **extra})


def client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def inspect(handler: Handler, text: str = REPO, token: str | None = None) -> hub.Gguf:
    return hub.inspect(hub.parse(text), client(handler), token)


# ── Names ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "repo", "quant"),
    [
        (REPO, REPO, None),
        (f"{REPO}:Q8_0", REPO, "Q8_0"),
        (f"hf.co/{REPO}:Q4_K_M", REPO, "Q4_K_M"),
        (f"https://huggingface.co/{REPO}", REPO, None),
        (f"https://huggingface.co/{REPO}/blob/main/x.gguf", REPO, None),
    ],
)
def test_a_model_is_named_as_people_copy_it(text: str, repo: str, quant: str | None) -> None:
    ref = hub.parse(text)
    assert (ref.repo, ref.quant) == (repo, quant)


@pytest.mark.parametrize(
    "text",
    ["llama3", "a/b/c", "https://example.com/a/b", "a/b:Q4 K", "../x/y", "x/..", "a/b; rm -rf /"],
)
def test_anything_else_is_not_a_model(text: str) -> None:
    with pytest.raises(ValueError):
        hub.parse(text)


def test_each_server_names_it_its_own_way() -> None:
    ref = hub.Ref(REPO, "Q4_K_M")
    assert ref.ollama == f"hf.co/{REPO}:Q4_K_M"
    assert hub.llama_server_command(ref, 9000) == (
        f"llama-server -hf {REPO}:Q4_K_M --alias {REPO}:Q4_K_M --port 9000"
    )


# ── What a model is ──────────────────────────────────────────────────────────


def test_the_default_quantization_its_size_and_the_others() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return listing()

    model = inspect(handler)
    assert model.ref == hub.Ref(REPO, "Q4_K_M")
    assert model.files == ("Llama-3.2-3B-Instruct-Q4_K_M.gguf",) and model.size == 2 * GB
    assert model.quants == ("F16", "Q4_K_M", "Q8_0")  # the projector is not a model
    assert not model.gated
    [request] = seen
    assert str(request.url) == f"https://huggingface.co/api/models/{REPO}?blobs=true"
    assert "authorization" not in request.headers  # a public model needs no token


def test_a_quantization_asked_for_and_a_split_file() -> None:
    parts = [
        {"rfilename": f"Q8_0/model-Q8_0-0000{i}-of-00002.gguf", "size": 4 * GB} for i in (1, 2)
    ]
    model = inspect(lambda r: listing(SIBLINGS[:2] + parts), f"{REPO}:q8_0")
    assert model.ref.quant == "Q8_0"
    assert model.size == 8 * GB and model.files == tuple(p["rfilename"] for p in parts)


def test_alternative_files_of_a_quantization_are_not_added_up() -> None:
    """Devin's review of #57: a server takes one of them; memory must hold the largest."""
    parts = [
        {"rfilename": f"Q8_0/model-Q8_0-0000{i}-of-00002.gguf", "size": 4 * GB} for i in (1, 2)
    ]
    model = inspect(lambda r: listing(SIBLINGS + parts), f"{REPO}:Q8_0")
    assert model.size == 8 * GB and len(model.files) == 2  # not 3 + 8
    small = [{"rfilename": "other/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "size": GB}]
    model = inspect(lambda r: listing(SIBLINGS + small))
    assert model.size == 2 * GB and model.files == ("Llama-3.2-3B-Instruct-Q4_K_M.gguf",)


@pytest.mark.parametrize("size", ["2GB", None, -1, True])
def test_a_listing_without_a_usable_size_is_unexpected(size: object) -> None:
    siblings = [{"rfilename": "model-Q4_K_M.gguf", "size": size}]
    with pytest.raises(hub.HubError) as e:
        inspect(lambda r: listing(siblings))
    assert e.value.problem is hub.Problem.UNEXPECTED
    with pytest.raises(hub.HubError) as e:
        inspect(lambda r: httpx.Response(200, json={"siblings": "none"}))
    assert e.value.problem is hub.Problem.UNEXPECTED


@pytest.mark.parametrize(
    ("siblings", "text", "problem", "quants"),
    [
        ([{"rfilename": "model.safetensors", "size": 1}], REPO, hub.Problem.NOT_GGUF, ()),
        (SIBLINGS, f"{REPO}:Q2_K", hub.Problem.NO_QUANT, ("F16", "Q4_K_M", "Q8_0")),
        (SIBLINGS[2:4], REPO, hub.Problem.CHOOSE_QUANT, ("F16", "Q8_0")),
    ],
)
def test_what_cannot_be_used_says_why(
    siblings: list[dict[str, Any]], text: str, problem: hub.Problem, quants: tuple[str, ...]
) -> None:
    with pytest.raises(hub.HubError) as e:
        inspect(lambda r: listing(siblings), text)
    assert e.value.problem is problem and e.value.quants == quants


@pytest.mark.parametrize(
    ("status", "problem"),
    [(401, hub.Problem.NOT_FOUND), (404, hub.Problem.NOT_FOUND), (403, hub.Problem.NO_ACCESS)],
)
def test_a_model_that_cannot_be_read(status: int, problem: hub.Problem) -> None:
    with pytest.raises(hub.HubError) as e:
        inspect(lambda r: httpx.Response(status))
    assert e.value.problem is problem


def test_a_gated_model_needs_a_token_and_its_license_accepted() -> None:
    heads: list[httpx.Request] = []

    def handler(status: int) -> Handler:
        def handle(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD":
                heads.append(request)
                return httpx.Response(status, headers={"location": "https://cdn/x"})
            return listing(gated="manual")

        return handle

    model = inspect(handler(302), token=TOKEN)
    assert model.gated
    with pytest.raises(hub.HubError) as e:
        hub.check_access(model, client(handler(302)), None)
    assert e.value.problem is hub.Problem.TOKEN and heads == []
    with pytest.raises(hub.HubError) as e:
        hub.check_access(model, client(handler(403)), TOKEN)
    assert e.value.problem is hub.Problem.LICENSE
    for status, problem in ((404, hub.Problem.NOT_FOUND), (500, hub.Problem.UNREACHABLE)):
        with pytest.raises(hub.HubError) as e:  # Devin's review of #57: not let through
            hub.check_access(model, client(handler(status)), TOKEN)
        assert e.value.problem is problem
    hub.check_access(model, client(handler(200)), TOKEN)
    hub.check_access(model, client(handler(302)), TOKEN)  # a redirect to the file: allowed
    assert heads[-1].url.path == f"/{REPO}/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    assert heads[-1].headers["authorization"] == f"Bearer {TOKEN}"


def test_whether_it_fits_in_the_free_memory() -> None:
    model = inspect(lambda r: listing())  # 2 GB on disk, 2.4 GB to run
    assert hub.fits(model, hub.Memory(ram=3 * GB, video=None))
    assert not hub.fits(model, hub.Memory(ram=GB, video=None))
    assert hub.fits(model, hub.Memory(ram=GB, video=2 * GB))  # with the video memory
    assert hub.fits(model, hub.Memory(ram=None, video=None)) is None
    assert hub.size_text(2_400_000_000) == "2.4 GB" and hub.size_text(820_000_000) == "820 MB"


# ── Ollama pulls it ──────────────────────────────────────────────────────────


def pull_lines(*lines: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, text="\n".join(json.dumps(line) for line in lines))


def test_ollama_pulls_it_and_reports_how_it_goes() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/pull" and "authorization" not in request.headers
        seen.append(json.loads(request.content))
        return pull_lines(
            {"status": "pulling manifest"},
            {"status": "pulling abc", "total": 100, "completed": 50},
            {"status": "success"},
        )

    steps: list[hub.Step] = []
    hub.pull("http://localhost:11434", hub.Ref(REPO, "Q4_K_M"), client(handler), steps.append)
    assert seen == [{"model": f"hf.co/{REPO}:Q4_K_M", "stream": True}]
    assert [s.status for s in steps] == ["pulling manifest", "pulling abc", "success"]
    assert steps[1].completed == 50


@pytest.mark.parametrize(
    "response",
    [
        pull_lines({"status": "pulling manifest"}, {"error": "pull model manifest: 401"}),
        pull_lines({"status": "pulling manifest"}),  # broke off
        httpx.Response(500, json={"error": "out of disk"}),
    ],
)
def test_a_pull_that_fails_says_so(response: httpx.Response) -> None:
    with pytest.raises(hub.PullFailed):
        hub.pull("http://localhost:11434", hub.Ref(REPO, "Q4_K_M"), client(lambda r: response))


# ── desman models ────────────────────────────────────────────────────────────


@pytest.fixture
def web(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Hugging Face and Ollama behind a mock; a machine with 16 GB free."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "huggingface.co":
            return listing()
        assert request.url.host == "localhost"
        return pull_lines(
            {"status": "pulling x", "total": 10, "completed": 10}, {"status": "success"}
        )

    real = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    monkeypatch.setattr(hub, "memory", lambda: PLENTY)
    return seen


def desman(*args: str) -> Result:
    return CliRunner().invoke(cli, list(args))


def test_models_show(web: list[httpx.Request]) -> None:
    result = desman("models", "show", REPO)
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert f"{REPO}:Q4_K_M: 2.0 GB to download, about 2.4 GB of memory to run" in out
    assert "Free memory: 16.0 GB RAM" in out and "Other quantizations: F16, Q4_K_M, Q8_0" in out
    data = json.loads(desman("models", "show", REPO, "--json").output)
    assert data["fits"] and data["size"] == 2 * GB and data["files"]
    assert all(r.url.host == "huggingface.co" for r in web)  # nothing downloaded


def test_models_add_has_ollama_pull_it_then_uses_it(web: list[httpx.Request]) -> None:
    ConfigManager(default_config_path()).save(make_config(api_keys={"HF_TOKEN": TOKEN}))
    result = desman("models", "add", REPO)
    assert result.exit_code == 0, result.output
    assert "Downloaded 100% of 1 MB" in result.output  # Ollama's progress, in tenths
    assert "Ollama model set to hf.co/" in result.output
    assert saved().settings["OLLAMA_MODEL"] == f"hf.co/{REPO}:Q4_K_M"
    hf = [r for r in web if r.url.host == "huggingface.co"]
    ollama = [r for r in web if r.url.host == "localhost"]
    assert hf and all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in hf)
    assert ollama and all("authorization" not in r.headers for r in ollama)  # never the token
    assert TOKEN not in result.output
    assert [p.id for p in providers.ready(saved())] == ["ollama"]


def test_models_add_for_llama_cpp_says_how_to_start_it(web: list[httpx.Request]) -> None:
    result = desman("models", "add", f"{REPO}:Q8_0", "--server", "llamacpp")
    assert result.exit_code == 0, result.output
    assert f"llama-server -hf {REPO}:Q8_0 --alias {REPO}:Q8_0 --port 8080" in " ".join(
        result.output.split()
    )
    assert saved().settings["LLAMA_CPP_MODEL"] == f"{REPO}:Q8_0"
    assert all(r.url.host == "huggingface.co" for r in web)  # llama.cpp downloads it itself


def test_a_model_too_large_for_the_machine_needs_force(
    web: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(ram=GB, video=None))
    result = desman("models", "add", REPO)
    assert result.exit_code == 1 and "does not fit in the free memory" in result.output
    assert not any(r.url.host == "localhost" for r in web)
    assert "OLLAMA_MODEL" not in saved().settings
    assert desman("models", "add", REPO, "--force").exit_code == 0


def test_a_problem_is_worded_for_the_person(monkeypatch: pytest.MonkeyPatch) -> None:
    real = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real(transport=httpx.MockTransport(lambda r: httpx.Response(401)), **kw),
    )
    result = desman("models", "show", "someone/secret-GGUF")
    assert result.exit_code == 1
    assert "add a token with `desman keys add huggingface`" in " ".join(result.output.split())
    assert desman("models", "show", "not a model").exit_code == 2


def test_the_hugging_face_token_is_a_key_like_the_others() -> None:
    result = CliRunner().invoke(cli, ["keys", "add", "huggingface", "--stdin"], input=TOKEN)
    assert result.exit_code == 0, result.output
    assert [k.value for k in saved().keys("HF_TOKEN")] == [TOKEN]
    listed = desman("keys", "list").output
    assert "Hugging Face" in listed and TOKEN not in listed
    assert providers.redact_keys(f"my token {TOKEN}")[1] == 1


# ── In the chat: a change the person confirms ────────────────────────────────


@pytest.fixture
def interlocutor(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(runs, "models", lambda config, conn, **_: runs.Models(llm, None))
    ConfigManager(default_config_path()).save(make_config(language="en", configured=True))
    return llm


def test_the_chat_shows_the_size_and_downloads_only_after_yes(
    interlocutor: FakeLLM, web: list[httpx.Request]
) -> None:
    ask = asks("models add", model=REPO)
    interlocutor.on(CHAT, ask, ask)  # a declined change asks the interlocutor nothing more
    result = chat("get me a small local model", "n")
    out = " ".join(result.output.split())
    assert "2.0 GB to download" in out and "Ollama at http://localhost:11434 downloads it" in out
    assert not any(r.url.host == "localhost" for r in web)
    assert "OLLAMA_MODEL" not in saved().settings
    chat("get me a small local model", "y")
    assert saved().settings["OLLAMA_MODEL"] == f"hf.co/{REPO}:Q4_K_M"
    assert journal()[-2:] == [
        (
            "operator_command",
            {"command": "models add", "args": {"model": REPO, "server": "ollama"}},
        ),
        ("operator_setting", {"name": "OLLAMA_MODEL"}),
    ]


def test_the_wizard_downloads_a_model_after_yes(web: list[httpx.Request]) -> None:
    from test_wizard import DONE, KEEP_PERMISSIONS, KEEP_PERSONAL, menu, wizard

    # Ollama's menu: 4 = download from Hugging Face, 5 = Back.
    result = wizard(
        "1", menu("ollama"), "4", REPO, "y", "5", DONE, *KEEP_PERMISSIONS, *KEEP_PERSONAL
    )
    assert result.exit_code == 0, result.output
    assert "2.0 GB to download" in " ".join(result.output.split())
    assert saved().settings["OLLAMA_MODEL"] == f"hf.co/{REPO}:Q4_K_M"


# ── Devin's review of #57 ────────────────────────────────────────────────────


def test_what_ollama_prints_is_shown_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    import click

    from desman import local_models

    leaked = "hf_" + "Leak3d" * 7

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co":
            return listing()
        return pull_lines({"status": f"pulling with {TOKEN}"}, {"error": f"denied {leaked}"})

    config = make_config(api_keys={"HF_TOKEN": TOKEN})
    model = hub.Gguf(hub.Ref(REPO, "Q4_K_M"), ("f.gguf",), GB, False, ("Q4_K_M",))
    shown: list[str] = []
    with pytest.raises(click.ClickException) as e:
        local_models.fetch(
            config, model, providers.find("ollama"), client(handler), show=shown.append
        )
    assert shown and TOKEN not in shown[0] and providers.REDACTED in shown[0]
    assert leaked not in e.value.message and providers.REDACTED in e.value.message


def test_a_remote_ollama_over_plain_http_is_warned_about() -> None:
    from desman import local_models

    model = hub.Gguf(hub.Ref(REPO, "Q4_K_M"), ("f.gguf",), GB, False, ("Q4_K_M",))
    ollama = providers.find("ollama")
    for host, warned in (
        ("http://gpu-box:11434", True),
        ("gpu-box:11434", True),
        ("https://gpu-box:11434", False),
        ("http://127.0.0.1:11434", False),
    ):
        _, warnings = local_models.plan(make_config(settings={"OLLAMA_HOST": host}), model, ollama)
        assert any("without encryption" in w for w in warnings) is warned, host


def test_too_large_for_the_memory_the_window_asks_to_download_anyway(
    interlocutor: FakeLLM, web: list[httpx.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hub, "memory", lambda: hub.Memory(ram=GB, video=None))
    interlocutor.on(CHAT, asks("models add", model=REPO))
    result = chat("get me a model", "n")
    out = " ".join(result.output.split())
    assert "does not fit in the free memory" in out and "Download it anyway?" in out
    assert "Apply this change?" not in out
    assert not any(r.url.host == "localhost" for r in web)


def test_the_wizard_says_ollama_keeps_what_it_downloaded(web: list[httpx.Request]) -> None:
    from test_wizard import DONE, KEEP_PERMISSIONS, KEEP_PERSONAL, menu, wizard

    result = wizard(
        "1", menu("ollama"), "4", REPO, "y", "5", DONE, *KEEP_PERMISSIONS, *KEEP_PERSONAL
    )
    assert result.exit_code == 0, result.output
    assert "Ollama keeps the downloaded model even if you leave without saving." in " ".join(
        result.output.split()
    )
