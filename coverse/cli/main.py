from __future__ import annotations

import argparse

from coverse.cli.app import serve
from coverse.cli.topic import multi_chat, prob_detect


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coverse")
    subparsers = parser.add_subparsers(dest="command_group", required=True)

    topic_parser = subparsers.add_parser("topic")
    topic_subparsers = topic_parser.add_subparsers(dest="topic_command", required=True)
    multi_chat.register(topic_subparsers)
    prob_detect.register(topic_subparsers)

    app_parser = subparsers.add_parser("app")
    app_subparsers = app_parser.add_subparsers(dest="app_command", required=True)
    serve.register(app_subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Unsupported command.")
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
