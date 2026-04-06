from __future__ import annotations

import re

from coverse.core.backends import OpenAIChatBackend
from coverse.core.types import AgentConfig, ConversationMessage


class ConversationAgent:
    def __init__(self, config: AgentConfig, backend: OpenAIChatBackend):
        self.config = config
        self.backend = backend

    def _normalize_messages(
        self, messages: list[ConversationMessage | dict[str, str]]
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, ConversationMessage):
                normalized.append(message.to_dict())
            else:
                normalized.append({"role": message["role"], "content": message["content"]})
        return normalized

    def build_messages(
        self, messages: list[ConversationMessage | dict[str, str]]
    ) -> list[dict[str, str]]:
        normalized = self._normalize_messages(messages)
        if not normalized or normalized[0]["role"] != "system":
            return [{"role": "system", "content": self.config.system_prompt}, *normalized]
        return normalized

    def postprocess(self, answer: str) -> str:
        answer = re.sub(r"<think>.+?</think>", "", answer, flags=re.MULTILINE | re.DOTALL)
        answer = answer.strip()
        if not answer:
            raise ValueError(f"Agent {self.config.name!r} returned empty content.")
        return answer

    def respond(self, messages: list[ConversationMessage | dict[str, str]]) -> str:
        payload = self.build_messages(messages)
        raw_answer = self.backend.generate(
            payload,
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            max_tokens=self.config.generation.max_tokens,
        )
        return self.postprocess(raw_answer)
