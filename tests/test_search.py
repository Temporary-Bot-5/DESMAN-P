"""Search engines of web_search (ROADMAP 3.5): open sources, SearXNG, Tavily, DuckDuckGo."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from conftest import FakeClock
from desman import providers
from desman.core import search
from desman.core.search import (
    DUCKDUCKGO_INTERVAL,
    Arxiv,
    DuckDuckGo,
    GitHub,
    SearXNG,
    StackOverflow,
    Tavily,
    Wikipedia,
    check_searxng,
)
from desman.core.store import Store
from desman.core.web import SearchResult, WebError
from helpers import make_config
from test_wizard import DONE, KEEP_PERMISSIONS, menu, saved, wizard

Handler = Callable[[httpx.Request], httpx.Response]


def http(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def dumped(results: list[SearchResult]) -> list[dict[str, str]]:
    return [r.model_dump() for r in results]


# ── The whole web ────────────────────────────────────────────────────────────

RESULTS = {"results": [{"title": "T", "url": "https://example.org", "content": "snippet"}]}


def test_tavily_tries_the_next_key_when_one_is_refused() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        if request.headers["authorization"] == "Bearer used-up":
            return httpx.Response(432, json={"detail": "plan limit"})
        assert json.loads(request.content)["query"] == "desman"
        return httpx.Response(200, json=RESULTS)

    found = Tavily(["used-up", "fresh"], http(handler)).search("desman", 5)
    assert dumped(found) == [{"title": "T", "url": "https://example.org", "snippet": "snippet"}]
    assert seen == ["Bearer used-up", "Bearer fresh"]


def test_tavily_without_a_working_key_says_so() -> None:
    with pytest.raises(WebError, match="HTTP 401"):
        Tavily(["bad"], http(lambda r: httpx.Response(401))).search("q", 5)


def test_searxng_is_asked_for_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json" and request.url.params["q"] == "q"
        assert request.url.params["language"] == "ru"
        return httpx.Response(200, json=RESULTS)

    engine = SearXNG("http://localhost:8888/", http(handler))
    assert [r.url for r in engine.search("q", 5, "ru")] == ["https://example.org"]


def test_a_searxng_without_json_is_explained() -> None:
    assert "JSON format" in (check_searxng("http://s", http(lambda r: httpx.Response(403))) or "")
    assert "not a list" in (check_searxng("http://s", http(lambda r: httpx.Response(200))) or "")
    assert check_searxng("http://s", http(lambda r: httpx.Response(200, json=RESULTS))) is None


DUCKDUCKGO_PAGE = """<html><body><div id="links" class="results">
<div class="result results_links web-result"><div class="links_main result__body">
<h2 class="result__title"><a rel="nofollow" class="result__a"
 href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fx%3D1&amp;rut=abc">Example <b>A</b></a></h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">First <b>result</b> text.</a>
</div></div>
<div class="result result--ad"><div class="links_main result__body">
<h2 class="result__title"><a class="result__a" href="https://duckduckgo.com/y.js?ad_domain=shop">Ad</a></h2>
</div></div>
<div class="result results_links web-result"><div class="links_main result__body">
<h2 class="result__title"><a class="result__a" href="https://example.net/b">B</a></h2>
<a class="result__snippet">Second.</a>
</div></div>
</div></body></html>"""  # noqa: E501


def test_duckduckgo_results_page_is_read_without_ads() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST" and request.content == b"q=desman+agents"
        return httpx.Response(200, text=DUCKDUCKGO_PAGE)

    found = DuckDuckGo(http(handler)).search("desman agents", 5)
    assert dumped(found) == [
        {"title": "Example A", "url": "https://example.org/a?x=1", "snippet": "First result text."},
        {"title": "B", "url": "https://example.net/b", "snippet": "Second."},
    ]


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(202, text="<html></html>"),
        httpx.Response(200, text='<div class="anomaly-modal">Select all squares</div>'),
    ],
)
def test_duckduckgo_asking_for_a_captcha_is_reported_not_retried(answer: httpx.Response) -> None:
    asked: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request)
        return answer

    with pytest.raises(WebError, match="not a robot"):
        DuckDuckGo(http(handler)).search("q", 5)
    assert len(asked) == 1


def test_a_changed_duckduckgo_page_is_an_error_not_an_empty_answer() -> None:
    engine = DuckDuckGo(http(lambda r: httpx.Response(200, text="<html>new design</html>")))
    with pytest.raises(WebError, match="not understood"):
        engine.search("q", 5)
    empty = '<div id="links"><div class="no-results">No results.</div></div>'
    assert DuckDuckGo(http(lambda r: httpx.Response(200, text=empty))).search("q", 5) == []


def test_duckduckgo_is_asked_at_most_every_few_seconds() -> None:
    clock = FakeClock()

    def now() -> float:
        return clock().timestamp()

    engine = DuckDuckGo(
        http(lambda r: httpx.Response(200, text=DUCKDUCKGO_PAGE)), clock=now, sleep=clock.sleep
    )
    engine.search("a", 5)
    engine.search("b", 5)
    assert clock.slept == DUCKDUCKGO_INTERVAL
    clock.advance(60)
    engine.search("c", 5)
    assert clock.slept == DUCKDUCKGO_INTERVAL  # long enough since the last one


# ── Open sources ─────────────────────────────────────────────────────────────


def test_wikipedia_in_the_language_asked_for() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "ru.wikipedia.org"
        assert request.url.params["srsearch"] == "агент" and request.headers["user-agent"]
        hits = [{"title": "Интеллектуальный агент", "snippet": '<span class="m">агент</span>'}]
        return httpx.Response(200, json={"query": {"search": hits}})

    found = Wikipedia(_with_agent(handler)).search("агент", 3, "ru")
    assert dumped(found) == [
        {
            "title": "Интеллектуальный агент",
            "url": "https://ru.wikipedia.org/wiki/Интеллектуальный_агент",
            "snippet": "агент",
        }
    ]


def _with_agent(handler: Handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": "DESMAN test"}
    )


def test_stack_overflow_questions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["site"] == "stackoverflow"
        item = {
            "title": "How to read a file in Python &amp; close it?",
            "link": "https://stackoverflow.com/q/1",
            "score": 42,
            "is_answered": True,
            "tags": ["python", "file"],
        }
        return httpx.Response(200, json={"items": [item]})

    [found] = StackOverflow(http(handler)).search("read file", 3)
    assert found.title == "How to read a file in Python & close it?"
    assert found.snippet == "score 42, answered, tags: python, file"


def test_arxiv_papers() -> None:
    feed = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.00001v1</id><title>Small   agents,
 big networks</title><summary>We study networks.</summary></entry></feed>"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search_query"] == "all:agents"
        return httpx.Response(200, text=feed)

    assert dumped(Arxiv(http(handler)).search("agents", 3)) == [
        {
            "title": "Small agents, big networks",
            "url": "http://arxiv.org/abs/2401.00001v1",
            "snippet": "We study networks.",
        }
    ]
    with pytest.raises(WebError, match="not understood"):
        Arxiv(http(lambda r: httpx.Response(200, text="<feed"))).search("x", 3)


def test_github_repositories() -> None:
    item = {
        "full_name": "searxng/searxng",
        "html_url": "https://github.com/searxng/searxng",
        "description": "Metasearch engine",
        "stargazers_count": 20000,
    }
    found = GitHub(http(lambda r: httpx.Response(200, json={"items": [item]}))).search("x", 3)
    assert dumped(found) == [
        {
            "title": "searxng/searxng",
            "url": "https://github.com/searxng/searxng",
            "snippet": "Metasearch engine (★ 20000)",
        }
    ]


@pytest.mark.parametrize("engine", [Wikipedia, StackOverflow, GitHub])
def test_an_open_source_that_fails_says_so(engine: type) -> None:
    with pytest.raises(WebError, match="HTTP 503"):
        engine(http(lambda r: httpx.Response(503))).search("q", 3)
    with pytest.raises(WebError, match="not understood"):
        engine(http(lambda r: httpx.Response(200, json={"odd": 1}))).search("q", 3)


# ── Configuration ────────────────────────────────────────────────────────────

TAVILY = {"TAVILY_API_KEY": "tvly-" + "k" * 30}


def test_engines_come_from_the_config_in_order() -> None:
    assert providers.search_engines(make_config()) == []
    every = make_config(
        api_keys=TAVILY, settings={"SEARXNG_URL": "http://s", "DUCKDUCKGO_SEARCH": "on"}
    )
    assert [p.id for p in providers.search_engines(every)] == ["searxng", "tavily", "duckduckgo"]
    off = make_config(settings={"DUCKDUCKGO_SEARCH": "off"})
    assert providers.search_engines(off) == []  # opt-in: only "on" turns it on
    assert providers.find_keyed("tavily").search and providers.by_key_env("TAVILY_API_KEY")
    with pytest.raises(providers.UnknownProvider):
        providers.find("tavily")  # never a model provider


def test_a_run_gets_the_open_sources_and_the_engines_set_up(store: Store) -> None:
    from desman.cli import _web

    bare = _web(make_config(), store)
    assert bare.engines == [] and set(bare.sources) == {
        "wikipedia",
        "stackoverflow",
        "arxiv",
        "github",
    }
    every = _web(
        make_config(
            api_keys=TAVILY, settings={"SEARXNG_URL": "http://s", "DUCKDUCKGO_SEARCH": "on"}
        ),
        store,
    )
    assert [type(e) for e in every.engines] == [SearXNG, Tavily, DuckDuckGo]
    assert every.cache is not None


def test_the_wizard_turns_duckduckgo_on_only_after_the_warning() -> None:
    result = wizard("1", menu("duckduckgo"), "y", DONE, *KEEP_PERMISSIONS)
    assert result.exit_code == 0, result.output
    assert (
        "Unofficial search" in result.output and "Use DuckDuckGo for web search?" in result.output
    )
    assert saved().load().settings["DUCKDUCKGO_SEARCH"] == "on"
    result = wizard("1", menu("duckduckgo"), "n", DONE, *KEEP_PERMISSIONS)
    assert saved().load().settings["DUCKDUCKGO_SEARCH"] == "off"


def test_the_wizard_checks_a_searxng_url(monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[str] = []

    def check(url: str, client: httpx.Client) -> str | None:
        checked.append(url)
        return None if url.endswith("8888") else "SearXNG cannot be reached: ConnectError"

    monkeypatch.setattr(search, "check_searxng", check)
    # SearXNG's menu: 1 = set URL, 2 = Back.
    good = wizard("1", menu("searxng"), "1", "http://localhost:8888", "2", DONE, *KEEP_PERMISSIONS)
    assert good.exit_code == 0, good.output
    assert "docs/searxng.md" in good.output and "web search works" in good.output
    bad = wizard("1", menu("searxng"), "1", "http://localhost:9", "2", DONE, *KEEP_PERMISSIONS)
    assert "cannot be used yet" in bad.output
    assert checked == ["http://localhost:8888", "http://localhost:9"]
    assert saved().load().settings["SEARXNG_URL"] == "http://localhost:9"  # kept, with a warning
