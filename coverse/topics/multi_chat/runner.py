from __future__ import annotations

# 这个脚本用于批量运行多 Agent 故事续写实验。
# 作用：
# 1. 读取一组故事开头 prompt。
# 2. 构造多个 Agent 按轮次接续发言。
# 3. 保存完整 transcript 和扁平化结果文件。
#
# 原理：
# - 每个 Agent 在自己的视角下接收历史消息。
# - 多个 Agent 按固定顺序轮流续写，形成一段故事对话。
# - 适合用于快速构造故事样本或比较不同模型/参数下的共创结果。
#
# 主要输入：
# - prompts_path: 文本 prompt 文件
# - agent_names: 参与对话的 Agent 名称
# - n_turns: 轮数
#
# 主要输出：
# - results.json: 完整 transcript
# - results.csv: 扁平化结果
# - metadata.json: 参数与模型配置
#
# 直接运行示例：
# python coverse/topics/multi_chat/runner.py --prompts-path data/coverse_pe/story_prompt.txt

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from coverse.core.agents import AgentConfig, ConversationAgent, GenerationConfig, MultiAgentRunner
from coverse.core.backends import ModelBackendConfig, OpenAIChatBackend
from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.multi_chat.prompts import DEFAULT_STORY_SYSTEM_PROMPT


def load_prompts(path: str | Path) -> list[str]:
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            prompts.append(stripped)
    if not prompts:
        raise ValueError(f"No prompts found in {path!s}.")
    return prompts


def build_runner(
    *,
    provider: str,
    model: str,
    agent_names: list[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    base_url: str | None,
    api_key_env: str | None,
) -> MultiAgentRunner:
    resolved_system_prompt = system_prompt or DEFAULT_STORY_SYSTEM_PROMPT
    backend = OpenAIChatBackend(
        ModelBackendConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    )
    generation = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    agents = [
        ConversationAgent(
            AgentConfig(name=name, system_prompt=resolved_system_prompt, generation=generation),
            backend,
        )
        for name in agent_names
    ]
    return MultiAgentRunner(agents)


def run_one_prompt(
    first_message: str,
    *,
    runner: MultiAgentRunner,
    n_turns: int,
) -> dict[str, Any]:
    transcript = runner.run(first_message=first_message, n_turns=n_turns)
    story_lines = [
        message["content"]
        for message in transcript
        if message["role"] not in {"system", "user", "assistant"}
    ]
    return {
        "first_message": first_message,
        "messages": transcript,
        "story": "\n".join(story_lines),
    }


def run_batch_multi_chat(
    *,
    provider: str,
    model: str,
    prompts_path: str,
    output_dir: str,
    command: str,
    agent_names: list[str],
    n_turns: int,
    concurrency: int,
    tag: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 1024,
    system_prompt: str | None = DEFAULT_STORY_SYSTEM_PROMPT,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, str]:
    prompts = load_prompts(prompts_path)
    runner = build_runner(
        provider=provider,
        model=model,
        agent_names=agent_names,
        system_prompt=system_prompt or DEFAULT_STORY_SYSTEM_PROMPT,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        base_url=base_url,
        api_key_env=api_key_env,
    )
    experiment_io = ExperimentIO(output_dir)
    run_name = f"{utc_timestamp().replace(':', '-')}-{tag}"
    run_dir = experiment_io.prepare_run_dir("multi_chat", run_name)

    metadata = ExperimentMetadata(
        topic="multi_chat",
        command=command,
        args={
            "prompts_path": prompts_path,
            "agent_names": agent_names,
            "n_turns": n_turns,
            "concurrency": concurrency,
            "tag": tag,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        },
        model={
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
        },
        output_dir=str(run_dir),
        input_source=str(prompts_path),
    )
    experiment_io.write_metadata(run_dir, metadata)

    rows: list[dict[str, Any]] = []
    worker_count = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_one_prompt, prompt, runner=runner, n_turns=n_turns)
            for prompt in prompts
        ]
        for future in tqdm(futures, total=len(futures)):
            rows.append(future.result())

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "first_message": row["first_message"],
                "story": row["story"],
                "messages_json": json.dumps(row["messages"], ensure_ascii=False),
            }
        )

    json_path = experiment_io.write_json(run_dir, "results.json", rows)
    csv_path = experiment_io.write_csv(run_dir, "results.csv", normalized_rows)
    return {
        "run_dir": str(run_dir),
        "metadata_path": str(Path(run_dir) / "metadata.json"),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量运行多 Agent 故事续写实验。")
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--n-turns", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--agent-name", dest="agent_names", action="append", default=[])
    parser.add_argument("--system-prompt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_batch_multi_chat(
        provider=args.provider,
        model=args.model,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        command="python coverse/topics/multi_chat/runner.py",
        agent_names=args.agent_names or ["agent_1", "agent_2"],
        n_turns=args.n_turns,
        concurrency=args.concurrency,
        tag=args.tag,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
