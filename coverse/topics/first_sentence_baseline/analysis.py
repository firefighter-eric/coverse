from __future__ import annotations

# 这个脚本负责第三阶段统计分析：
# 按 prompt 聚合相似度结果，计算均值、方差并输出最终排序。

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pvariance

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.first_sentence_baseline.common import load_json_records


def run_analysis(
    *,
    similarities_path: str,
    output_dir: str,
    command: str,
) -> dict[str, str]:
    rows = load_json_records(similarities_path)
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(row)

    ranking_rows = []
    for prompt_rows in grouped.values():
        similarities = [row["prompt_response_cosine_similarity"] for row in prompt_rows]
        distances = [row["prompt_response_cosine_distance"] for row in prompt_rows]
        first = prompt_rows[0]
        ranking_rows.append(
            {
                "prompt_id": first["prompt_id"],
                "scenario": first["scenario"],
                "prompt": first["prompt"],
                "unique_count": len(prompt_rows),
                "avg_prompt_response_cosine_similarity": mean(similarities),
                "prompt_response_similarity_variance": pvariance(similarities)
                if len(similarities) >= 2
                else None,
                "avg_prompt_response_cosine_distance": mean(distances),
                "prompt_response_distance_variance": pvariance(distances)
                if len(distances) >= 2
                else None,
                "computable": len(prompt_rows) >= 2,
            }
        )

    ranking_rows.sort(
        key=lambda item: (
            item["prompt_response_distance_variance"] is None,
            -(item["prompt_response_distance_variance"] or float("-inf")),
        )
    )
    for rank, row in enumerate(ranking_rows, start=1):
        row["rank"] = rank

    experiment_io = ExperimentIO(output_dir)
    run_name = utc_timestamp().replace(":", "-")
    run_dir = experiment_io.prepare_run_dir("first_sentence_baseline_analysis", run_name)
    metadata = ExperimentMetadata(
        topic="first_sentence_baseline",
        command=command,
        args={
            "stage": "analysis",
            "similarities_path": similarities_path,
            "ranking_key": "prompt_response_distance_variance_desc",
        },
        model={},
        output_dir=str(run_dir),
        input_source=similarities_path,
    )
    experiment_io.write_metadata(run_dir, metadata)
    ranking_path = experiment_io.write_csv(run_dir, "analysis_ranking.csv", ranking_rows)
    details_path = experiment_io.write_json(run_dir, "analysis_details.json", ranking_rows)
    return {
        "run_dir": str(run_dir),
        "metadata_path": str(Path(run_dir) / "metadata.json"),
        "ranking_path": str(ranking_path),
        "details_path": str(details_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 embedding 相似度结果，按 prompt 聚合并输出分析排序。"
    )
    parser.add_argument("--similarities-path", required=True)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        similarities_path=args.similarities_path,
        output_dir=args.output_dir,
        command="python coverse/topics/first_sentence_baseline/analysis.py",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
