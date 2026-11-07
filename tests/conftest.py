"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from desman import i18n
from desman.config import CONFIG_DIR_ENV
from desman.core import db
from desman.core.db import DATA_DIR_ENV
from desman.core.store import Store


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DESMAN at throwaway config and data dirs so tests never touch the real ones."""
    config_dir = tmp_path / "config"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config_dir))
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "data"))
    return config_dir


_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


@pytest.fixture(autouse=True)
def no_internet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real HTTP request leaves this machine in no test: it fails as a dropped connection
    would. Tests pass `httpx.MockTransport`; a few talk to a server on 127.0.0.1."""
    real = httpx.HTTPTransport.handle_request

    def guarded(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.host not in _LOOPBACK:
            raise httpx.ConnectError(f"no internet in tests: {request.url.host}", request=request)
        return real(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)


@pytest.fixture(autouse=True)
def no_real_ollama(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ollama discovery uses mocks only, including on loopback; isolate its cache too."""
    from desman import local_models

    local_models._tags_cache.clear()
    original = httpx.HTTPTransport.handle_request

    def guarded(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            raise httpx.ConnectError("no real Ollama in tests", request=request)
        return original(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)
    yield
    local_models._tags_cache.clear()


@pytest.fixture(autouse=True)
def english() -> Iterator[None]:
    """Every test starts and ends in English (the wizard and the CLI switch the language)."""
    i18n.set_language("en")
    yield
    i18n.set_language("en")


class FakeClock:
    """Deterministic time; `sleep` advances it, so loops that wait run instantly in tests."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)
        self.slept = 0.0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.advance(seconds)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(clock: FakeClock) -> Iterator[Store]:
    conn = db.open_db(":memory:")
    yield Store(conn, clock=clock)
    conn.close()
