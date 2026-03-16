"""LLM provider abstraction for agents that use language model reasoning."""

from __future__ import annotations

import os
from typing import Any

from ..core.config import LLMConfig


class LLMProvider:
    """Abstraction over LLM providers (Anthropic, OpenAI, etc.)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Any = None

    def _init_client(self) -> Any:
        """Lazily initialize the LLM client."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found. Set {self.config.api_key_env} environment variable."
            )

        if self.config.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install ml-agent-team[llm]")
        elif self.config.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("Install openai: pip install ml-agent-team[llm]")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

        return self._client

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a text response from the LLM."""
        client = self._init_client()

        if self.config.provider == "anthropic":
            message = client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        elif self.config.provider == "openai":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""

        raise ValueError(f"Unsupported provider: {self.config.provider}")
