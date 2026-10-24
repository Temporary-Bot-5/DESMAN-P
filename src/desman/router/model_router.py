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


class ModelRouter:
    """Routes LLM requests to available providers with fallback."""

    def __init__(self) -> None:
        self.providers: list[ModelConfig] = []
        self._clients: dict[str, AsyncOpenAI] = {}
        self._load_providers()

    def _load_providers(self) -> None:
        """Load providers from environment variables."""
        # OpenRouter (primary - many free models)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            self.providers.append(
                ModelConfig(
                    name="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"),
                )
            )

        # Google Gemini (via OpenAI-compatible endpoint or direct)
        # For now, OpenRouter covers Gemini too

        # Add more providers here (Groq, Together, etc.)

        if not self.providers:
            raise RuntimeError(
                "No LLM providers configured. Set OPENROUTER_API_KEY in .env"
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
        messages: list[ChatCompletionMessageParam],
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
        messages: list[ChatCompletionMessageParam],
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
