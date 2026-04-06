from __future__ import annotations

# 这个脚本用于第一启动句基线课题的第三阶段统计分析。
# 作用：
# 1. 读取第二阶段生成的 embedding 相似度结果。
# 2. 按 prompt 分组聚合每个回答对应的 similarity / distance。
# 3. 计算均值与方差，并输出最终排序表。
#
# 原理：
# - 对每个 prompt，统计 prompt-response cosine similarity 和 distance 的分布。
# - 当前默认按 prompt_response_distance_variance 从高到低排序。
# - 距离方差越大，说明同一启动句引出的回答相对 prompt 的语义偏移波动越大，可视为更有发散潜力。
#
# 主要输入：
# - similarities_path: 第二阶段输出的 embedding_similarity.json
#
# 主要输出：
# - output_path 指定的 analysis_details.json
# - 与输出文件同目录的 analysis_ranking.csv
# - metadata.json
#
# 直接运行示例：
# python coverse/topics/first_sentence_analysis/analysis.py --similarities-path xxx/embedding_similarity.json --output-path data/first_sentence_analysis/v1/analysis_details.json

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, pvariance

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from coverse.core.types import ExperimentMetadata
from coverse.topics.first_sentence_analysis.common import load_json_records


def run_analysis(
    *,
    similarities_path: str,
    output_path: str,
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

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_file.parent / "metadata.json"
    ranking_path = output_file.parent / "analysis_ranking.csv"
    metadata = ExperimentMetadata(
        topic="first_sentence_analysis",
        command=f"python {Path(__file__).as_posix()}",
        args={
            "stage": "analysis",
            "similarities_path": similarities_path,
            "output_path": str(output_file),
            "ranking_key": "prompt_response_distance_variance_desc",
        },
        model={},
        output_dir=str(output_file.parent),
        input_source=similarities_path,
    )
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if ranking_rows:
        with ranking_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(ranking_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ranking_rows)
    else:
        ranking_path.write_text("", encoding="utf-8")
    output_file.write_text(
        json.dumps(ranking_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metadata_path": str(metadata_path),
        "ranking_path": str(ranking_path),
        "details_path": str(output_file),
        "output_path": str(output_file),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 embedding 相似度结果，按 prompt 聚合并输出分析排序。"
    )
    parser.add_argument("--similarities-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_analysis(
        similarities_path=args.similarities_path,
        output_path=args.output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
