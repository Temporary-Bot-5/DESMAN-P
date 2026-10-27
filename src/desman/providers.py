"""The model providers DESMAN knows: what each needs from `config.json`, and which are ready.

One list for the wizard, `desman config show` and the router (2.2+). Variable names follow
each provider's own documentation, so a key copied from its console goes under the name the
provider uses. Secrets go to `api_keys`; values that are not secret (an account id, a local
host) go to `settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

    def is_ready(self, config: Config) -> bool:
        """Everything it needs is configured."""
        if self.key_env is not None and not config.api_keys.get(self.key_env):
            return False
        return all(config.settings.get(s.env) for s in self.settings if not s.default)


CLOUD = (
    Provider("nvidia", "NVIDIA NIM", "https://build.nvidia.com/explore/discover", "NVIDIA_API_KEY"),
    Provider("google", "Google AI Studio", "https://aistudio.google.com/apikey", "GOOGLE_API_KEY"),
    Provider("groq", "GROQ", "https://console.groq.com/keys", "GROQ_API_KEY"),
    Provider("mistral", "Mistral AI", "https://console.mistral.ai/api-keys/", "MISTRAL_API_KEY"),
    Provider(
        "cloudflare",
        "Cloudflare Workers AI",
        "https://dash.cloudflare.com/profile/api-tokens",
        "CLOUDFLARE_API_TOKEN",
        settings=(Setting("CLOUDFLARE_ACCOUNT_ID", "Account ID"),),
    ),
    Provider("cohere", "Cohere", "https://dashboard.cohere.com/api-keys", "COHERE_API_KEY"),
    Provider("aion", "Aion Labs", "https://aionlabs.ai/", "AION_API_KEY"),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/keys", "OPENROUTER_API_KEY"),
    Provider(
        "vercel",
        "Vercel AI Gateway",
        "https://vercel.com/dashboard/ai-gateway",
        "AI_GATEWAY_API_KEY",
    ),
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


def ready(config: Config) -> list[Provider]:
    """Cloud providers with everything they need, in list order. Local ones are not offered
    yet: nothing checks that a local server is running."""
    return [p for p in CLOUD if p.is_ready(config)]


def mask(secret: str) -> str:
    """Enough to tell keys apart, never enough to use one: a short key shows nothing."""
    if len(secret) < 20:
        return "****"
    return f"{secret[:4]}…{secret[-4:]}"


def describe(config: Config) -> dict[str, Any]:
    """The config with every secret masked: what `desman config show` prints."""
    known = {p.key_env for p in PROVIDERS} | {s.env for p in PROVIDERS for s in p.settings}
    rows = []
    for p in PROVIDERS:
        row: dict[str, Any] = {"id": p.id, "name": p.name, "local": p.local}
        if p.key_env is not None:
            key = config.api_keys.get(p.key_env)
            row["key"] = {"env": p.key_env, "value": mask(key) if key else None}
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
        "unknown_keys": {k: mask(v) for k, v in config.api_keys.items() if k not in known},
        # Masked too: nothing says a value saved under an unknown name is not a secret.
        "unknown_settings": {k: mask(v) for k, v in config.settings.items() if k not in known},
    }
