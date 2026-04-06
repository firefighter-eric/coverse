from __future__ import annotations

import argparse
import json

from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)


def run_batch_multi_chat(**kwargs):
    from coverse.topics.multi_chat.runner import run_batch_multi_chat as runner

    return runner(**kwargs)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("multi-chat")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--n-turns", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--agent-name",
        dest="agent_names",
        action="append",
        default=[],
        help="Repeat this flag to add multiple agents.",
    )
    parser.add_argument("--system-prompt")
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    result = run_batch_multi_chat(
        provider=args.provider,
        model=args.model,
        prompts_path=args.prompts_path,
        output_dir=args.output_dir,
        command="coverse topic multi-chat",
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
