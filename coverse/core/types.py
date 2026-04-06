from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024


@dataclass(slots=True)
class AgentConfig:
    name: str
    system_prompt: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass(slots=True)
class ExperimentMetadata:
    topic: str
    command: str
    args: dict[str, Any]
    model: dict[str, Any]
    output_dir: str
    input_source: str | None = None
    created_at: str = field(default_factory=utc_timestamp)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
