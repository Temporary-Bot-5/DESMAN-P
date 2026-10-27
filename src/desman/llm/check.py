"""`desman keys check`: is each key accepted, and what limits does the provider report?

Spends no tokens: it lists the provider's models (and, for OpenRouter, reads the key's own
limits). A key is never sent anywhere but its own provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import httpx

from desman.config import Config
from desman.llm.openai_compat import key_id
from desman.providers import Provider


@dataclass(frozen=True)
class KeyCheck:
    provider: str
    number: int  # 1-based, as `desman keys` lists it
    key_id: str
    ok: bool
    http_status: int | None
    detail: str  # short, human-readable
    models: int | None = None
    default_model_listed: bool | None = None
    limits: dict[str, Any] = field(default_factory=dict)


def check(
    config: Config, provider: Provider, client: httpx.Client, timeout: float = 20.0
) -> list[KeyCheck]:
    assert provider.key_env is not None
    out = []
    for number, key in enumerate(config.keys(provider.key_env), 1):
        out.append(_check_key(config, provider, number, key.value, client, timeout))
    return out


def _check_key(
    config: Config,
    provider: Provider,
    number: int,
    key: str,
    client: httpx.Client,
    timeout: float,
) -> KeyCheck:
    base = provider.endpoint(config)
    headers = {"Authorization": f"Bearer {key}"}
    result = partial(KeyCheck, provider.id, number, key_id(key))
    try:
        if provider.id == "openrouter":
            # /models is public there: /key is what tells a valid key from a bad one.
            response = client.get(f"{base}/key", headers=headers, timeout=timeout)
        else:
            response = client.get(f"{base}/models", headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        return result(ok=False, http_status=None, detail=type(e).__name__)

    limits = {
        k.lower(): v
        for k, v in response.headers.items()
        if "limit" in k.lower() and k.lower().startswith("x-")
    }
    if response.status_code != 200:
        return result(
            ok=False,
            http_status=response.status_code,
            detail=_refusal(response.status_code),
            limits=limits,
        )
    try:
        data = response.json()
    except ValueError:
        data = {}
    if provider.id == "openrouter":
        info = data.get("data") if isinstance(data, dict) else None
        if not isinstance(info, dict):
            return result(ok=False, http_status=200, detail=_UNEXPECTED, limits=limits)
        limits.update(_openrouter_limits(info))
        return result(ok=True, http_status=200, detail="accepted", limits=limits)
    models = data.get("data") if isinstance(data, dict) else None
    if not isinstance(models, list):  # an HTML page from a proxy is not an answer
        return result(ok=False, http_status=200, detail=_UNEXPECTED, limits=limits)
    ids = [str(m.get("id", "")) for m in models if isinstance(m, dict)]
    listed = any(i.removeprefix("models/") == provider.default_model for i in ids)
    return result(
        ok=True,
        http_status=200,
        detail="accepted",
        models=len(ids),
        default_model_listed=listed,
        limits=limits,
    )


_UNEXPECTED = "unexpected reply (not the provider's API?): the key could not be verified"


def _refusal(status: int) -> str:
    if status in (401, 403):
        return "refused: the key is wrong, revoked or lacks access"
    if status == 429:
        return "rate limited right now"
    if status == 402:
        return "no credits left"
    return f"unexpected HTTP {status}"


def _openrouter_limits(info: dict[str, Any]) -> dict[str, Any]:
    """What OpenRouter reports about a key: free-tier daily requests, credit limit, usage."""
    out: dict[str, Any] = {}
    daily = info.get("free_model_daily_requests")
    if isinstance(daily, dict):
        out["free requests today"] = f"{daily.get('used', '?')} of {daily.get('limit', '?')}"
    if info.get("limit") is not None:
        out["credit limit"] = info["limit"]
        out["credit left"] = info.get("limit_remaining")
    if "usage_daily" in info:
        out["spent today (credits)"] = info["usage_daily"]
    if "is_free_tier" in info:
        out["free tier"] = bool(info["is_free_tier"])
    return out
