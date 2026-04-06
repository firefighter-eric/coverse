from __future__ import annotations

# 这个脚本用于第一启动句基线课题的第一阶段采样。
# 作用：
# 1. 读取启动句材料文件。
# 2. 对每个 prompt 调用 LLM 多次，生成“紧接着的下一句”。
# 3. 保存原始回答和清洗后的回答，供后续 embedding 与分析阶段继续使用。
#
# 原理：
# - 把每个启动句视为独立刺激材料。
# - 通过重复采样观察同一个 prompt 能引出多少不同方向的下一句。
# - 这一阶段不做语义相似度计算，只负责把语言模型输出稳定落盘。
#
# 主要输入：
# - prompts_file: 启动句 JSON 文件
# - system_prompt_file: system prompt Markdown 文件
# - samples_per_prompt: 每个 prompt 采样次数
# - temperature: LLM 温度
#
# 主要输出：
# - output_path 指定的 llm_samples.json: 每条 prompt 的原始回答和清洗后回答
# - llm_sample_metadata.json: 与输出文件同目录，记录本次采样的参数与模型配置
#
# 直接运行示例：
# python coverse/topics/first_sentence_analysis/llm_sample.py --output-path data/first_sentence_analysis/v1/llm_samples.json

import argparse
import concurrent.futures
import json
import random
import sys
import threading
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from tqdm import tqdm

from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from coverse.core.types import ExperimentMetadata
from coverse.topics.first_sentence_analysis.common import (
    NextSentenceGenerator,
    SampleRecord,
    load_prompt_specs,
    load_system_prompt,
    resolve_prompts_file,
)


def run_llm_sample(
    *,
    output_path: str,
    prompts_file: str | None = None,
    system_prompt_file: str | None = None,
    samples_per_prompt: int = 30,
    temperature: float = 1.0,
    concurrency: int = 16,
    seed: int | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key_env: str | None = DEFAULT_LLM_API_KEY_ENV,
) -> dict[str, str]:
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be greater than zero.")
    if concurrency <= 0:
        raise ValueError("concurrency must be greater than zero.")

    resolved_prompts_file = resolve_prompts_file(prompts_file)
    prompt_specs = load_prompt_specs(prompts_file)
    random.Random(seed).shuffle(prompt_specs)
    system_prompt = load_system_prompt(system_prompt_file)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_file.parent / "llm_sample_metadata.json"

    metadata = ExperimentMetadata(
        topic="first_sentence_analysis",
        command=f"python {Path(__file__).as_posix()}",
        args={
            "stage": "llm_sample",
            "prompts_file": resolved_prompts_file,
            "system_prompt_file": system_prompt_file,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "concurrency": concurrency,
            "seed": seed,
            "output_path": str(output_file),
        },
        model={
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
        },
        output_dir=str(output_file.parent),
        input_source=resolved_prompts_file,
        notes={"topic_readme": "coverse/topics/first_sentence_analysis/README.md"},
    )
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    thread_local = threading.local()

    def get_generator() -> NextSentenceGenerator:
        generator = getattr(thread_local, "generator", None)
        if generator is None:
            generator = NextSentenceGenerator(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                temperature=temperature,
                system_prompt=system_prompt,
            )
            thread_local.generator = generator
        return generator

    def sample_once(prompt_id: int, prompt_spec, sample_index: int) -> dict[str, str | int]:
        sample = get_generator().generate_one(prompt_spec.text)
        return SampleRecord(
            prompt_id=prompt_id,
            scenario=prompt_spec.scenario,
            prompt=prompt_spec.text,
            sample_index=sample_index,
            raw_response=sample["raw"],
            cleaned_response=sample["cleaned"],
        ).to_dict()

    tasks = [
        (prompt_id, prompt_spec, sample_index)
        for prompt_id, prompt_spec in enumerate(prompt_specs, start=1)
        for sample_index in range(1, samples_per_prompt + 1)
    ]
    total_samples = len(prompt_specs) * samples_per_prompt
    with tqdm(total=total_samples, desc="Sampling next sentences", unit="sample") as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(sample_once, prompt_id, prompt_spec, sample_index)
                for prompt_id, prompt_spec, sample_index in tasks
            ]
            rows = []
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                progress.update(1)
    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row["sample_index"])))

    output_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metadata_path": str(metadata_path),
        "output_path": str(output_file),
        "samples_path": str(output_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对每个启动句采样多个下一句，并输出 llm_samples.json。"
    )
    parser.add_argument("--prompts-file", default="data/first_sentence_analysis/v0/prompt.json")
    parser.add_argument("--system-prompt-file", default="data/first_sentence_analysis/v0/system_prompt.md")
    parser.add_argument("--output-path", default="data/first_sentence_analysis/v0/llm_samples.json")
    parser.add_argument("--samples-per-prompt", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_llm_sample(
        output_path=args.output_path,
        prompts_file=args.prompts_file,
        system_prompt_file=args.system_prompt_file,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        concurrency=args.concurrency,
        seed=args.seed,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
