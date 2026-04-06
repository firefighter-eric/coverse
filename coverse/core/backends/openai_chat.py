from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import dotenv
from openai import OpenAI


@dataclass(slots=True)
class ModelBackendConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenAIChatBackend:
    """Thin OpenAI-compatible backend for research experiments."""

    def __init__(self, config: ModelBackendConfig):
        self.config = config
        self._client: OpenAI | None = None

    def _resolve_api_key(self) -> str:
        dotenv.load_dotenv()

        if self.config.provider == "ollama":
            return "ollama"

        if self.config.api_key_env is None:
            raise ValueError(
                "api_key_env is required for provider "
                f"{self.config.provider!r}. Example: ARK_API_KEY."
            )

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise ValueError(
                f"Environment variable {self.config.api_key_env!r} is not set for "
                f"provider {self.config.provider!r}."
            )
        return api_key

    def _resolve_base_url(self) -> str | None:
        if self.config.provider == "ollama":
            return self.config.base_url or "http://127.0.0.1:11434/v1"
        return self.config.base_url

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=self._resolve_api_key(),
                base_url=self._resolve_base_url(),
            )
        return self._client

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        response = self._get_client().chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("Model returned empty content.")
        return answer
