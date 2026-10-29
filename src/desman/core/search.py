"""Search engines for `web_search` (ROADMAP 3.5; core/web.py runs them).

Two ways in, neither of which needs a paid service:

1. Open sources, no setup: Wikipedia, Stack Overflow (Stack Exchange), arXiv and GitHub have
   official APIs that answer without a key (with small per-address quotas). Not the whole web,
   but where most answers an agent needs are.
2. The whole web through the human's own SearXNG: a metasearch server running on their machine
   (docs/searxng.md) that asks the big engines itself. No key, no account. Or Tavily, with a
   free key.

Only official APIs and the human's own server: DESMAN never reads a search engine's pages
meant for people, which they refuse to robots.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from desman.core.web import (
    MAX_BYTES,
    TIMEOUT,
    USER_AGENT,
    SearchEngine,
    SearchResult,
    WebError,
    html_text,
)


def client() -> httpx.Client:
    """For the engines' fixed hosts (not for pages: core/web.Fetcher checks every address)."""
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)


def _send(http: httpx.Client, engine: str, method: str, url: str, **kwargs: Any) -> httpx.Response:
    """The answer, read up to MAX_BYTES: a bigger one is an error, not a full memory."""
    body = bytearray()
    try:
        with http.stream(method, url, **kwargs) as response:
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > MAX_BYTES:
                    raise WebError(f"{engine} sent an answer that is too big")
    except httpx.HTTPError as e:
        raise WebError(f"{engine} cannot be reached: {type(e).__name__}") from e
    return httpx.Response(response.status_code, content=bytes(body))


def _get(http: httpx.Client, engine: str, url: str, **kwargs: Any) -> httpx.Response:
    response = _send(http, engine, "GET", url, **kwargs)
    if response.status_code != 200:
        raise WebError(f"{engine} answered HTTP {response.status_code}")
    return response


class _Results(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[SearchResult]  # required: {} or {"error": ...} is not "nothing found"


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
        response = _send(
            self.http,
            "SearXNG",
            "GET",
            f"{self.url}/search",
            params={"q": query, "format": "json", **({"language": language} if language else {})},
            timeout=TIMEOUT,
        )
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
            response = _send(
                self.http,
                "Tavily",
                "POST",
                self.URL,
                json={"query": query, "max_results": max_results, "search_depth": "basic"},
                headers={"Authorization": f"Bearer {key}"},
                timeout=TIMEOUT,
            )
            if response.status_code in (401, 403, 429, 432, 433):  # bad key, limit, no credits
                problem = f"Tavily refused the search (HTTP {response.status_code})"
                continue
            return parse_results(response, "Tavily")
        raise WebError(problem)


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
