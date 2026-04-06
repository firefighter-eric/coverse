from __future__ import annotations

# 这个脚本放第一启动句课题的公共能力：
# 包括 prompt 加载、采样清洗、通用数据结构和 JSON 读写辅助函数。

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coverse.config import (
    DEFAULT_EMBEDDING_MODEL_PATH,
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from coverse.core.backends import ModelBackendConfig, OpenAIChatBackend
from coverse.topics.first_sentence_baseline.prompts import (
    DEFAULT_PROMPTS_PATH,
    PromptSpec,
    load_prompt_specs_from_file,
)

GENERATION_SYSTEM_PROMPT = """
# 角色
你是一名中文故事续写参与者。

# 任务
我会给你一句故事的开头，请你只续写紧接着的下一句。

# 规则
- 只能输出一句中文续写
- 长度必须在7到20字之间
- 不允许输出解释 标题 序号或多句内容
- 不允许换行

# 输出
只输出下一句本身
""".strip()


@dataclass(slots=True)
class SampleRecord:
    prompt_id: int
    scenario: str
    prompt: str
    sample_index: int
    raw_response: str
    cleaned_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_prompts_file(prompts_file: str | None) -> str:
    return prompts_file or str(DEFAULT_PROMPTS_PATH)


def load_prompt_specs(prompts_file: str | None) -> list[PromptSpec]:
    return load_prompt_specs_from_file(resolve_prompts_file(prompts_file))


def clean_generated_sentence(text: str) -> str:
    text = re.sub(r"<think>.+?</think>", "", text, flags=re.MULTILINE | re.DOTALL)
    text = text.strip()
    text = text.splitlines()[0].strip() if text else ""
    for separator in ["。", "！", "？", ".", "!", "?", ";", "；"]:
        if separator in text:
            text = text.split(separator)[0].strip()
    return text


class NextSentenceGenerator:
    def __init__(
        self,
        *,
        provider: str = DEFAULT_LLM_PROVIDER,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str | None = DEFAULT_LLM_BASE_URL,
        api_key_env: str | None = DEFAULT_LLM_API_KEY_ENV,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 128,
    ):
        self.backend = OpenAIChatBackend(
            ModelBackendConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
            )
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def generate_one(self, prompt: str) -> dict[str, str]:
        raw = self.backend.generate(
            [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return {"raw": raw, "cleaned": clean_generated_sentence(raw)}


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    deduped = []
    for record in records:
        key = (record["prompt_id"], record["cleaned_response"])
        if not record["cleaned_response"] or key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def default_embedding_model_path() -> str:
    return DEFAULT_EMBEDDING_MODEL_PATH
