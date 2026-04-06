from __future__ import annotations

# 这个脚本负责管理第一启动句课题的材料文件入口：
# 默认从 data/first_sentence_baseline/prompt.json 读取启动句。

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_PROMPTS_PATH = Path("data/first_sentence_baseline/prompt.json")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    scenario: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_prompt_specs_from_file(path: str | Path) -> list[PromptSpec]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Prompt file not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    specs = [PromptSpec(scenario=item["scenario"], text=item["text"]) for item in payload]
    if not specs:
        raise ValueError(f"No prompts found in {resolved}")
    return specs
