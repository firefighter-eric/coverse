from __future__ import annotations

import argparse

from coverse.config import (
    DEFAULT_LLM_API_KEY_ENV,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
)


def launch_app(**kwargs):
    from coverse.apps.gradio_app import launch_app as app_launcher

    return app_launcher(**kwargs)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("serve")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key-env", default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument("--min-latency", type=float, default=0.0)
    parser.add_argument("--max-latency", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--system-prompt")
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    launch_app(
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        system_prompt=args.system_prompt,
        min_latency=args.min_latency,
        max_latency=args.max_latency,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    return 0
