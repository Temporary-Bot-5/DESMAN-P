"""The model providers DESMAN knows: what each needs from `config.json`, and which are ready.

One list for the wizard, `desman config show`, `desman keys` and the router (2.2+). Variable
names follow each provider's own documentation, so a key copied from its console goes under the
name the provider uses. Secrets go to `api_keys` (a provider may have several keys); values that
are not secret (an account id, a local host) go to `settings`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from desman.i18n import t

if TYPE_CHECKING:
    from desman.config import Config


@dataclass(frozen=True)
class Setting:
    env: str
    label: str
    default: str = ""


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    url: str
    key_env: str | None = None  # the secret, kept in api_keys; None for local providers
    settings: tuple[Setting, ...] = field(default=())
    local: bool = False
    # OpenAI-compatible endpoint; {NAME} is filled from settings (Cloudflare's account id).
    base_url: str = ""
    # A model the free tier serves. Free lineups change: `desman run --model` overrides it.
    default_model: str = ""
    limits_url: str = ""  # the provider's page on rate limits and quotas
    # Billed per use, no free tier: never chosen automatically (only with --provider).
    paid: bool = False
    # How the key is sent; {key} is replaced by it.
    auth: tuple[tuple[str, str], ...] = (("Authorization", "Bearer {key}"),)
    # False for providers that reject a temperature the model does not support (Claude 4.7+
    # accepts only the default): it is left out of the request.
    send_temperature: bool = True
    # Sent when a request sets no max_tokens, for providers that need one.
    max_tokens: int | None = None

    def headers(self, key: str) -> dict[str, str]:
        return {name: value.format(key=key) for name, value in self.auth}

    def endpoint(self, config: Config) -> str:
        return self.base_url.format(**config.settings).rstrip("/")

    def is_ready(self, config: Config) -> bool:
        """Everything it needs is configured."""
        if self.key_env is not None and not config.keys(self.key_env):
            return False
        return all(config.settings.get(s.env) for s in self.settings if not s.default)


FREE = (
    Provider(
        "nvidia",
        "NVIDIA NIM",
        "https://build.nvidia.com/explore/discover",
        "NVIDIA_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        default_model="nvidia/llama-3.1-nemotron-70b-instruct",
        limits_url="https://docs.api.nvidia.com/nim/docs/api-limits",
    ),
    Provider(
        "google",
        "Google AI Studio",
        "https://aistudio.google.com/apikey",
        "GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.5-flash-lite",
        limits_url="https://ai.google.dev/gemini-api/docs/rate-limits",
    ),
    Provider(
        "groq",
        "GROQ",
        "https://console.groq.com/keys",
        "GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-20b",
        limits_url="https://console.groq.com/docs/rate-limits",
    ),
    Provider(
        "mistral",
        "Mistral AI",
        "https://console.mistral.ai/api-keys/",
        "MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        limits_url="https://admin.mistral.ai/plateforme/limits",
    ),
    Provider(
        "cloudflare",
        "Cloudflare Workers AI",
        "https://dash.cloudflare.com/profile/api-tokens",
        "CLOUDFLARE_API_TOKEN",
        settings=(Setting("CLOUDFLARE_ACCOUNT_ID", "Account ID"),),
        base_url="https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        default_model="@cf/meta/llama-3.1-8b-instruct",
        limits_url="https://developers.cloudflare.com/workers-ai/platform/limits/",
    ),
    Provider(
        "cohere",
        "Cohere",
        "https://dashboard.cohere.com/api-keys",
        "COHERE_API_KEY",
        base_url="https://api.cohere.ai/compatibility/v1",
        default_model="command-r7b-12-2024",
        limits_url="https://docs.cohere.com/docs/rate-limits",
    ),
    Provider(
        "aion",
        "Aion Labs",
        "https://aionlabs.ai/",
        "AION_API_KEY",
        base_url="https://api.aionlabs.ai/v1",
        default_model="aion-labs/aion-3.0-mini",
        limits_url="https://aionlabs.ai/",
    ),
    Provider(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/keys",
        "OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="nvidia/nemotron-3.5-lightning:free",
        limits_url="https://openrouter.ai/docs/api-reference/limits",
    ),
    Provider(
        "vercel",
        "Vercel AI Gateway",
        "https://vercel.com/dashboard/ai-gateway",
        "AI_GATEWAY_API_KEY",
        base_url="https://ai-gateway.vercel.sh/v1",
        default_model="google/gemini-2.5-flash-lite",
        limits_url="https://vercel.com/docs/ai-gateway/pricing",
    ),
)

# Billed per use. The default models are the cheapest current ones; check them with
# `desman keys check` (lineups change) and set a spending limit in each console. Not tested
# against the real APIs in the prototype (owner's decision): `untested_note()` says so in the CLI.
PAID = (
    Provider(
        "openai",
        "OpenAI",
        "https://platform.openai.com/api-keys",
        "OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-6-luna",
        limits_url="https://platform.openai.com/settings/organization/limits",
        paid=True,
    ),
    Provider(
        "anthropic",
        "Anthropic (Claude)",
        "https://platform.claude.com/settings/keys",
        "ANTHROPIC_API_KEY",
        # The OpenAI-compatible layer of the Claude API.
        base_url="https://api.anthropic.com/v1",
        default_model="claude-haiku-5-5",
        limits_url="https://platform.claude.com/docs/en/api/rate-limits",
        paid=True,
        # Bearer for chat completions; x-api-key and a version for the native /models.
        auth=(
            ("Authorization", "Bearer {key}"),
            ("x-api-key", "{key}"),
            ("anthropic-version", "2023-06-01"),
        ),
        send_temperature=False,
        max_tokens=4096,
    ),
    Provider(
        "deepseek",
        "DeepSeek",
        "https://platform.deepseek.com/api_keys",
        "DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-flash",
        limits_url="https://api-docs.deepseek.com/quick_start/rate_limit",
        paid=True,
    ),
    Provider(
        "xai",
        "xAI (Grok)",
        "https://console.x.ai",
        "XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        default_model="grok-4.3",
        limits_url="https://docs.x.ai/docs/models",
        paid=True,
    ),
)

CLOUD = FREE + PAID


def untested_note() -> str:
    return t("Paid providers are not tested in the prototype: they will be tested outside it.")


# Tried in this order when no --provider is given: generous free tiers first. Paid providers
# are never on it: they are used only when asked for by name.
PREFERENCE = (
    "groq",
    "google",
    "mistral",
    "openrouter",
    "nvidia",
    "cohere",
    "aion",
    "cloudflare",
    "vercel",
)

LOCAL = (
    Provider(
        "ollama",
        "Ollama",
        "https://ollama.com",
        settings=(Setting("OLLAMA_HOST", "Host", "http://localhost:11434"),),
        local=True,
    ),
    Provider(
        "llamacpp",
        "llama.cpp",
        "https://github.com/ggml-org/llama.cpp",
        settings=(Setting("LLAMA_CPP_PATH", "Path"),),
        local=True,
    ),
)

PROVIDERS = CLOUD + LOCAL


class UnknownProvider(LookupError):
    pass


def find(name: str) -> Provider:
    """A provider by id (`groq`), name (`GROQ`) or key variable (`GROQ_API_KEY`), any case."""
    wanted = name.strip().lower()
    for p in PROVIDERS:
        if wanted in (p.id, p.name.lower(), (p.key_env or "").lower()):
            return p
    known = ", ".join(p.id for p in PROVIDERS)
    raise UnknownProvider(t("unknown provider {name!r} (known: {known})", name=name, known=known))


def by_key_env(env: str) -> Provider | None:
    return next((p for p in CLOUD if p.key_env == env), None)


def ready(config: Config) -> list[Provider]:
    """Cloud providers with everything they need, in list order. Local ones are not offered
    yet: nothing checks that a local server is running."""
    return [p for p in CLOUD if p.is_ready(config)]


def label(text: str) -> str:
    """A key's label for display: as is, unless it looks like a key itself."""
    from desman.config import looks_like_secret

    return mask(text) if looks_like_secret(text) else text


def mask(secret: str) -> str:
    """Enough to tell keys apart, never enough to use one: a short key shows nothing."""
    if len(secret) < 20:
        return "****"
    return f"{secret[:4]}…{secret[-4:]}"


# From the start of a line: `"NAME": "value"`, `NAME = 'value'`, `export NAME=value`. A quoted
# value may be followed by anything (a comma, a comment); an unquoted one by a `# comment`.
_PAIR = re.compile(
    r"""^\s*[{,]?\s*(?:export\s+)?["']?([A-Za-z][A-Za-z0-9_]*)["']?\s*[:=]\s*["']([^"'\n]+)["']"""
)
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^\s#\"']+)\s*(?:#.*)?$")


def parse_key_file(text: str) -> list[tuple[str, str]]:
    """(variable, value) pairs from a key file: JSON (a value may be a list of keys), or
    `NAME=value` / `"NAME": "value"` lines, also when the JSON is not quite valid (a missing
    comma is the usual slip). Order and duplicates are kept; the caller decides."""
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict):
        pairs: list[tuple[str, str]] = []
        for env, value in data.items():
            values = value if isinstance(value, list) else [value]
            pairs += [(str(env), v.strip()) for v in values if isinstance(v, str) and v.strip()]
        return pairs
    pairs = []
    for line in text.splitlines():
        if line.lstrip().startswith(("#", "//")):  # a disabled key stays disabled
            continue
        found = _PAIR.match(line) or _LINE.match(line)
        if found and found[2].strip():
            pairs.append((found[1], found[2].strip()))
    return pairs


def describe(config: Config) -> dict[str, Any]:
    """The config with every secret masked: what `desman config show` prints."""
    known = {p.key_env for p in PROVIDERS} | {s.env for p in PROVIDERS for s in p.settings}
    rows = []
    for p in PROVIDERS:
        row: dict[str, Any] = {"id": p.id, "name": p.name, "local": p.local, "paid": p.paid}
        if p.key_env is not None:
            row["key_env"] = p.key_env
            row["keys"] = [
                {"value": mask(k.value), "label": label(k.label)} for k in config.keys(p.key_env)
            ]
        row["settings"] = {s.env: config.settings.get(s.env) or None for s in p.settings}
        row["ready"] = p.is_ready(config) if not p.local else None
        rows.append(row)
    return {
        "schema_version": config.schema_version,
        "configured": config.configured,
        "language": config.language,
        "permissions": config.permissions.model_dump(),
        "providers": rows,
        # Saved under a name no provider uses (typo, or a provider not supported yet).
        "unknown_keys": {
            env: [mask(k.value) for k in keys]
            for env, keys in config.api_keys.items()
            if env not in known
        },
        # Masked too: nothing says a value saved under an unknown name is not a secret.
        "unknown_settings": {k: mask(v) for k, v in config.settings.items() if k not in known},
    }
