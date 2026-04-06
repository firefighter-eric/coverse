from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
