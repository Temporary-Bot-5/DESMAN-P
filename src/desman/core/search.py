"""Search engines for `web_search` (ROADMAP 3.5; core/web.py runs them).

Three ways in, none of which needs a paid service:

1. Open sources, no setup: Wikipedia, Stack Overflow (Stack Exchange), arXiv and GitHub have
   official APIs that answer without a key (with small per-address quotas). Not the whole web,
   but where most answers an agent needs are.
2. The human's own SearXNG: a metasearch server running on their machine (docs/searxng.md) that
   asks the big engines itself. No key, no account. Or Tavily, with a free key.
3. DuckDuckGo's HTML results page, read like a browser would. Unofficial: there is no API, the
   page can change, too many searches get a captcha, and its terms may not allow it. So it is
   off until the human turns it on, searches at most once every few seconds, and says plainly
   when it is refused instead of retrying.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from typing import Any, Final
from urllib.parse import parse_qs, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from desman.core.web import TIMEOUT, USER_AGENT, SearchEngine, SearchResult, WebError, html_text

DUCKDUCKGO_INTERVAL: Final = 5.0
"""Seconds between two DuckDuckGo searches of this process at least."""


def client() -> httpx.Client:
    """For the engines' fixed hosts (not for pages: core/web.Fetcher checks every address)."""
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)


def _get(http: httpx.Client, engine: str, url: str, **kwargs: Any) -> httpx.Response:
    try:
        response = http.get(url, **kwargs)
    except httpx.HTTPError as e:
        raise WebError(f"{engine} cannot be reached: {type(e).__name__}") from e
    if response.status_code != 200:
        raise WebError(f"{engine} answered HTTP {response.status_code}")
    return response


class _Results(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[SearchResult] = []


def parse_results(response: httpx.Response, engine: str) -> list[SearchResult]:
    """A JSON answer {"results": [{title, url, content}, ...]} (Tavily, SearXNG)."""
    if response.status_code != 200:
        raise WebError(f"{engine} answered HTTP {response.status_code}")
    try:
        return _Results.model_validate_json(response.content).results
    except ValidationError as e:
        raise WebError(f"{engine} sent an answer that is not a list of results") from e


# ── The whole web ────────────────────────────────────────────────────────────


class SearXNG:
    """A SearXNG server with the JSON format on (docs/searxng.md)."""

    name = "searxng"

    def __init__(self, url: str, http: httpx.Client) -> None:
        self.url = url.rstrip("/")
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        try:
            response = self.http.get(
                f"{self.url}/search",
                params={
                    "q": query,
                    "format": "json",
                    **({"language": language} if language else {}),
                },
                timeout=TIMEOUT,
            )
        except httpx.HTTPError as e:
            raise WebError(f"SearXNG cannot be reached: {type(e).__name__}") from e
        if response.status_code == 403:
            raise WebError("SearXNG refuses the JSON format (see docs/searxng.md)")
        return parse_results(response, "SearXNG")[:max_results]


def check_searxng(url: str, http: httpx.Client) -> str | None:
    """Why a SearXNG server at `url` cannot be used, or None when it answers searches."""
    try:
        SearXNG(url, http).search("desman", 1)
    except WebError as e:
        return str(e)
    return None


class Tavily:
    """https://tavily.com: free monthly credits, one per basic search. Keys are tried in order:
    one that is refused or out of credits gives way to the next."""

    name = "tavily"
    URL: Final = "https://api.tavily.com/search"

    def __init__(self, keys: Sequence[str], http: httpx.Client) -> None:
        self.keys = list(keys)
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        problem = "no Tavily key"
        for key in self.keys:
            try:
                response = self.http.post(
                    self.URL,
                    json={"query": query, "max_results": max_results, "search_depth": "basic"},
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=TIMEOUT,
                )
            except httpx.HTTPError as e:
                raise WebError(f"search failed: {type(e).__name__}") from e
            if response.status_code in (401, 403, 429, 432, 433):  # bad key, limit, no credits
                problem = f"Tavily refused the search (HTTP {response.status_code})"
                continue
            return parse_results(response, "Tavily")
        raise WebError(problem)


class _DuckDuckGoPage(HTMLParser):
    """Results of html.duckduckgo.com: links `a.result__a`, text `.result__snippet`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = (dict(attrs).get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self.results.append(
                {"title": "", "url": _target(dict(attrs).get("href") or ""), "snippet": ""}
            )
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "td", "div"):
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field is not None:
            self.results[-1][self._field] += data


def _target(href: str) -> str:
    """The real URL of a result: DuckDuckGo links through //duckduckgo.com/l/?uddg=<url>."""
    parts = urlsplit(href)
    if parts.path == "/l/":
        return parse_qs(parts.query).get("uddg", [""])[0]
    if parts.path.startswith("/y.js"):  # an ad
        return ""
    return href


class DuckDuckGo:
    """DuckDuckGo's results page for browsers without JavaScript. Unofficial (module docs)."""

    name = "duckduckgo"
    URL: Final = "https://html.duckduckgo.com/html/"

    def __init__(
        self,
        http: httpx.Client,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self._last: float | None = None

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        if self._last is not None:
            self.sleep(max(0.0, self._last + DUCKDUCKGO_INTERVAL - self.clock()))
        self._last = self.clock()
        try:
            response = self.http.post(self.URL, data={"q": query}, timeout=TIMEOUT)
        except httpx.HTTPError as e:
            raise WebError(f"DuckDuckGo cannot be reached: {type(e).__name__}") from e
        page = response.text
        if response.status_code in (202, 403, 429) or "anomaly" in page:
            raise WebError(
                "DuckDuckGo asks to prove this is not a robot (too many searches): try again "
                "later, or use another source"
            )
        if response.status_code != 200:
            raise WebError(f"DuckDuckGo answered HTTP {response.status_code}")
        parser = _DuckDuckGoPage()
        parser.feed(page)
        found = [
            SearchResult(title=r["title"].strip(), url=r["url"], snippet=r["snippet"].strip())
            for r in parser.results
            if r["url"].startswith(("http://", "https://"))
        ]
        if not found and "result__a" not in page and "no-results" not in page:
            raise WebError("DuckDuckGo's page was not understood (it may have changed)")
        return found[:max_results]


# ── Open sources: official APIs, no key ─────────────────────────────────────


class Wikipedia:
    name = "wikipedia"

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        base = f"https://{language or 'en'}.wikipedia.org"
        response = _get(
            self.http,
            "Wikipedia",
            f"{base}/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
        )
        try:
            hits = response.json()["query"]["search"]
            return [
                SearchResult(
                    title=hit["title"],
                    url=f"{base}/wiki/{hit['title'].replace(' ', '_')}",
                    snippet=html_text(hit.get("snippet", ""))[1],
                )
                for hit in hits
            ]
        except (ValueError, KeyError, TypeError) as e:
            raise WebError("Wikipedia sent an answer that was not understood") from e


class StackOverflow:
    name = "stackoverflow"
    URL: Final = "https://api.stackexchange.com/2.3/search/advanced"

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        response = _get(
            self.http,
            "Stack Overflow",
            self.URL,
            params={
                "q": query,
                "site": "stackoverflow",
                "order": "desc",
                "sort": "relevance",
                "pagesize": max_results,
            },
        )
        try:
            return [
                SearchResult(
                    title=html_text(item["title"])[1],
                    url=item["link"],
                    snippet=(
                        f"score {item.get('score', 0)}, "
                        f"{'answered' if item.get('is_answered') else 'not answered'}, "
                        f"tags: {', '.join(item.get('tags', []))}"
                    ),
                )
                for item in response.json()["items"]
            ]
        except (ValueError, KeyError, TypeError) as e:
            raise WebError("Stack Overflow sent an answer that was not understood") from e


class Arxiv:
    name = "arxiv"
    URL: Final = "https://export.arxiv.org/api/query"
    _ATOM: Final = "{http://www.w3.org/2005/Atom}"

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        response = _get(
            self.http,
            "arXiv",
            self.URL,
            params={"search_query": f"all:{query}", "max_results": max_results},
        )
        try:
            feed = ET.fromstring(response.content)
        except ET.ParseError as e:
            raise WebError("arXiv sent an answer that was not understood") from e
        results = []
        for entry in feed.iter(f"{self._ATOM}entry"):
            title = " ".join((entry.findtext(f"{self._ATOM}title") or "").split())
            summary = " ".join((entry.findtext(f"{self._ATOM}summary") or "").split())
            url = (entry.findtext(f"{self._ATOM}id") or "").strip()
            if url:
                results.append(SearchResult(title=title, url=url, snippet=summary[:500]))
        return results


class GitHub:
    name = "github"
    URL: Final = "https://api.github.com/search/repositories"

    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def search(
        self, query: str, max_results: int, language: str | None = None
    ) -> list[SearchResult]:
        response = _get(
            self.http,
            "GitHub",
            self.URL,
            params={"q": query, "per_page": max_results},
            headers={"Accept": "application/vnd.github+json"},
        )
        try:
            return [
                SearchResult(
                    title=item["full_name"],
                    url=item["html_url"],
                    snippet=(
                        f"{item.get('description') or ''} (★ {item.get('stargazers_count', 0)})"
                    ).strip(),
                )
                for item in response.json()["items"]
            ]
        except (ValueError, KeyError, TypeError) as e:
            raise WebError("GitHub sent an answer that was not understood") from e


def open_sources(http: httpx.Client) -> dict[str, SearchEngine]:
    """Every source of web_search besides the whole web: no setup needed."""
    return {
        "wikipedia": Wikipedia(http),
        "stackoverflow": StackOverflow(http),
        "arxiv": Arxiv(http),
        "github": GitHub(http),
    }
