from __future__ import annotations

# 这个脚本负责第一阶段采样：
# 对每个启动句调用 LLM 多次，生成下一句，并保存原始回答与清洗后的回答。

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)
from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.first_sentence_baseline.common import (
    NextSentenceGenerator,
    SampleRecord,
    load_prompt_specs,
    resolve_prompts_file,
)


def run_llm_sample(
    *,
    output_dir: str,
    command: str,
    prompts_file: str | None = None,
    samples_per_prompt: int = 30,
    temperature: float = 1.0,
    seed: int | None = None,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key_env: str | None = DEFAULT_LLM_API_KEY_ENV,
) -> dict[str, str]:
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be greater than zero.")

    resolved_prompts_file = resolve_prompts_file(prompts_file)
    prompt_specs = load_prompt_specs(prompts_file)
    random.Random(seed).shuffle(prompt_specs)

    generator = NextSentenceGenerator(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature,
    )
    experiment_io = ExperimentIO(output_dir)
    run_name = utc_timestamp().replace(":", "-")
    run_dir = experiment_io.prepare_run_dir("first_sentence_baseline", run_name)

    metadata = ExperimentMetadata(
        topic="first_sentence_baseline",
        command=command,
        args={
            "stage": "llm_sample",
            "prompts_file": resolved_prompts_file,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "seed": seed,
        },
        model={
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key_env": api_key_env,
        },
        output_dir=str(run_dir),
        input_source=resolved_prompts_file,
        notes={"topic_readme": "coverse/topics/first_sentence_baseline/README.md"},
    )
    experiment_io.write_metadata(run_dir, metadata)

    rows = []
    for prompt_id, prompt_spec in enumerate(prompt_specs, start=1):
        for sample_index in range(1, samples_per_prompt + 1):
            sample = generator.generate_one(prompt_spec.text)
            rows.append(
                SampleRecord(
                    prompt_id=prompt_id,
                    scenario=prompt_spec.scenario,
                    prompt=prompt_spec.text,
                    sample_index=sample_index,
                    raw_response=sample["raw"],
                    cleaned_response=sample["cleaned"],
                ).to_dict()
            )

    samples_path = experiment_io.write_json(run_dir, "llm_samples.json", rows)
    return {
        "run_dir": str(run_dir),
        "metadata_path": str(Path(run_dir) / "metadata.json"),
        "samples_path": str(samples_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对每个启动句采样多个下一句，并输出 llm_samples.json。"
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prompts-file")
    parser.add_argument("--samples-per-prompt", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_llm_sample(
        output_dir=args.output_dir,
        command="python coverse/topics/first_sentence_baseline/llm_sample.py",
        prompts_file=args.prompts_file,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
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
