"""Root-signed registry discovery, requested by the person."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Final, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from desman import net
from desman.core import web
from desman.i18n import t
from desman.net import RegistryUrl, sign

MAX_BYTES: Final = 64 * 1024


class CatalogError(ValueError):
    pass


class Unreachable(CatalogError):
    pass


class Refused(CatalogError):
    pass


class Expired(CatalogError):
    pass


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: RegistryUrl
    name: str = Field(min_length=1, max_length=60)
    note: str | None = Field(default=None, max_length=200)

    @field_validator("url")
    @classmethod
    def https_only(cls, value: str) -> str:
        if not value.startswith("https://") or "?" in value or "#" in value or "\\" in value:
            raise ValueError("expected an HTTPS registry base URL")
        # Parse the authority too, including the port range.
        try:
            url = httpx.URL(value)
        except httpx.InvalidURL as e:
            raise ValueError("invalid registry URL") from e
        if not url.host or (url.port is not None and not 1 <= url.port <= 65535):
            raise ValueError("expected an HTTPS host and a valid port")
        return value


class RegistryCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registries: list[CatalogEntry] = Field(max_length=50)

    @model_validator(mode="after")
    def unique_urls(self) -> Self:
        urls = {str(httpx.URL(entry.url)).rstrip("/") for entry in self.registries}
        if len(urls) != len(self.registries):
            raise ValueError("duplicate registry URL")
        return self


def verify(data: bytes, now: datetime) -> RegistryCatalog:
    if len(data) > MAX_BYTES:
        raise Refused(t("The registry catalog is too large."))
    try:
        signed = sign.parse(data)
        sign.check_signature(signed, sign.load_root())
        record = signed.record
        if (
            record.kind is not sign.RecordKind.REGISTRY_CATALOG
            or record.issued_at > now + sign.SKEW
        ):
            raise ValueError("unexpected kind or issue time")
        if now >= record.expires_at:
            raise Expired(t("The registry catalog has expired."))
        return RegistryCatalog.model_validate(record.body)
    except Expired:
        raise
    except (ValueError, OSError, RecursionError) as e:
        raise Refused(t("The registry catalog could not be verified.")) from e


def _read(response: httpx.Response, limit: int, deadline: float) -> bytes:
    # Ask for identity encoding and cap the bytes before any decompression.
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise Refused(t("The registry catalog could not be verified."))
    body = bytearray()
    chunks = (response.content,) if response.is_stream_consumed else response.iter_raw()
    for chunk in chunks:
        if perf_counter() > deadline:
            raise httpx.ReadTimeout("response deadline exceeded")
        if len(body) + len(chunk) > limit:
            raise Refused(t("The registry catalog is too large."))
        body.extend(chunk)
    return bytes(body)


def fetch(client: httpx.Client, now: datetime) -> RegistryCatalog:
    start = perf_counter()
    fetcher = web.Fetcher(client, resolver=web.resolve)
    try:
        with closing(
            fetcher.get(
                net.CATALOG_URL, timeout=10, accept="application/json", accept_encoding="identity"
            )
        ) as response:
            response.raise_for_status()
            data = _read(response, MAX_BYTES, start + 10)
    except (web.WebError, httpx.HTTPError) as e:
        raise Unreachable(t("The registry catalog is unreachable.")) from e
    return verify(data, now)


@dataclass(frozen=True)
class Probe:
    entry: CatalogEntry
    latency_ms: int | None

    @property
    def answers(self) -> bool:
        return self.latency_ms is not None


def probe(entries: list[CatalogEntry], client: httpx.Client) -> list[Probe]:
    """Ask each listed registry whether it is up. The root signed these addresses, and a node
    reaches its registry the same way (no public-address pinning): a registry may resolve to a
    private address on the owner's own network (Tailscale's MagicDNS gives 100.64.0.0/10)."""

    def check(entry: CatalogEntry) -> Probe:
        start = perf_counter()
        target = entry.url.rstrip("/") + "/healthz"
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        try:
            request = client.build_request("GET", target, headers=headers, timeout=5)
            with closing(client.send(request, stream=True, follow_redirects=False)) as response:
                if response.status_code == 200:
                    body = json.loads(_read(response, 4096, start + 5))
                    if isinstance(body, dict) and body.get("status") == "ok":
                        return Probe(entry, round((perf_counter() - start) * 1000))
        except (web.WebError, httpx.HTTPError, ValueError, RecursionError):
            pass
        return Probe(entry, None)

    with ThreadPoolExecutor(max_workers=50) as pool:
        return list(pool.map(check, entries))


def available(probes: list[Probe]) -> list[Probe]:
    return sorted((item for item in probes if item.answers), key=lambda item: item.latency_ms or 0)


def discover(now: datetime) -> list[Probe]:
    with web.fetch_client() as client:
        return probe(fetch(client, now).registries, client)
