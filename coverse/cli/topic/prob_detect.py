from __future__ import annotations

import argparse
import json


def run_probability_experiment(**kwargs):
    from coverse.topics.prob_detect.runner import run_probability_experiment as runner

    return runner(**kwargs)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("prob-detect")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--text-file")
    parser.set_defaults(handler=handle)


def handle(args: argparse.Namespace) -> int:
    result = run_probability_experiment(
        model_path=args.model_path,
        target=args.target,
        output_dir=args.output_dir,
        command="coverse topic prob-detect",
        texts=args.text,
        text_file=args.text_file,
        device_map=args.device_map,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
