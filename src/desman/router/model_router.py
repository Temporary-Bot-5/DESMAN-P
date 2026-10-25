"""Model Router — selects LLM provider, handles fallbacks, retries, rate limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from desman.core.models import TaskPriority


@dataclass
class ModelConfig:
    """Configuration for a model provider."""

    name: str
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 4000
    temperature: float = 0.3
    timeout: float = 60.0


PROVIDER_SPECS: list[dict[str, Any]] = [
    {
        "name": "openrouter",
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.1-8b-instruct:free",
        "env_model": "OPENROUTER_MODEL",
    },
    {
        "name": "gemini",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-1.5-flash",
        "env_model": "GEMINI_MODEL",
    },
    {
        "name": "groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
        "env_model": "GROQ_MODEL",
    },
    {
        "name": "mistral",
        "env_key": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "env_model": "MISTRAL_MODEL",
    },
    {
        "name": "cloudflare",
        "env_key": "CLOUDFLARE_API_KEY",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/",
        "default_model": "@cf/meta/llama-3.1-8b-instruct",
        "env_model": "CLOUDFLARE_MODEL",
        "needs_account_id": True,
    },
    {
        "name": "cohere",
        "env_key": "COHERE_API_KEY",
        "base_url": "https://api.cohere.com/v1",
        "default_model": "command-r",
        "env_model": "COHERE_MODEL",
    },
    {
        "name": "nvidia",
        "env_key": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "nvidia/nemotron-3-ultra",
        "env_model": "NVIDIA_MODEL",
    },
    {
        "name": "vercel",
        "env_key": "VERCEL_AI_GATEWAY_KEY",
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "default_model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
        "env_model": "VERCEL_MODEL",
    },
]


class ModelRouter:
    """Routes LLM requests to available providers with fallback."""

    def __init__(self) -> None:
        self.providers: list[ModelConfig] = []
        self._clients: dict[str, AsyncOpenAI] = {}
        self._load_providers()

    def _load_providers(self) -> None:
        """Load providers from environment variables."""
        for spec in PROVIDER_SPECS:
            api_key = os.getenv(spec["env_key"])
            if not api_key:
                continue

            # Cloudflare needs account ID in base_url
            base_url = spec["base_url"]
            if spec.get("needs_account_id"):
                account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
                if not account_id:
                    continue
                base_url = base_url.format(account_id=account_id)

            model = os.getenv(spec["env_model"], spec["default_model"])

            self.providers.append(
                ModelConfig(
                    name=spec["name"],
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                )
            )

        if not self.providers:
            raise RuntimeError(
                "No LLM providers configured. Run `desman setup` to configure API keys, "
                "or set at least OPENROUTER_API_KEY in .env"
            )

        # Create clients
        for p in self.providers:
            self._clients[p.name] = AsyncOpenAI(
                base_url=p.base_url,
                api_key=p.api_key,
                timeout=p.timeout,
            )

    def _get_client(self, provider_name: str) -> AsyncOpenAI:
        return self._clients[provider_name]

    @retry(
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_provider(
        self,
        provider: ModelConfig,
        messages: list[ChatCompletionMessageParam],  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> str:
        """Call a single provider with retries."""
        client = self._get_client(provider.name)
        response = await client.chat.completions.create(
            model=provider.model,
            messages=messages,
            max_tokens=provider.max_tokens,
            temperature=provider.temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from LLM")
        return content.strip()  # type: ignore[no-any-return]

    async def complete(
        self,
        messages: list[ChatCompletionMessageParam],  # type: ignore[type-arg]
        task_type: str = "general",
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ) -> str:
        """
        Complete with fallback across providers.

        Args:
            messages: OpenAI-format messages
            task_type: "planning", "execution", "review", "general"
            priority: higher priority may use better models
        """
        last_error = None

        for provider in self.providers:
            try:
                return await self._call_provider(provider, messages, **kwargs)
            except Exception as e:
                last_error = e
                # Log and try next provider
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def complete_simple(
        self,
        system_prompt: str,
        user_prompt: str,
        task_type: str = "general",
        **kwargs: Any,
    ) -> str:
        """Convenience method for simple system+user prompts."""
        return await self.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            task_type=task_type,
            **kwargs,
        )
