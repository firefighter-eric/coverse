from __future__ import annotations

# 这个脚本用于第一启动句基线课题的第二阶段 embedding 相似度计算。
# 作用：
# 1. 读取第一阶段生成的 llm_samples.json。
# 2. 对完全重复的回答先去重。
# 3. 分别编码“原始 prompt”和“回答”。
# 4. 计算 prompt-response 之间的余弦相似度与余弦距离。
#
# 原理：
# - 这里不比较“回答和回答之间”的距离，而是比较“prompt 和回答”的语义接近程度。
# - 如果同一个 prompt 产生的回答对 prompt 的语义距离波动较大，后续分析里通常更容易体现发散性。
#
# 主要输入：
# - samples_path: 第一阶段的 llm_samples.json
# - embedding_model_path: 本地 embedding 模型目录
#
# 主要输出：
# - output_path 指定的 embedding_similarity.json
# - 与输出文件同目录的 embedding_similarity.csv
# - embedding_similarity_metadata.json
#
# 直接运行示例：
# python coverse/topics/first_sentence_analysis/embedding_similarity.py --samples-path xxx/llm_samples.json --output-path data/first_sentence_analysis/v1/embedding_similarity.json

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from coverse.config import DEFAULT_EMBEDDING_MODEL_PATH
from coverse.core.types import ExperimentMetadata
from coverse.topics.first_sentence_analysis.common import (
    dedupe_records,
    load_json_records,
)


class SentenceEmbeddingModel:
    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path)
        self._model.to(self.device)
        self._model.eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_loaded()
        chunks = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(texts), batch_size),
            total=total_batches,
            desc="Encoding texts",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            embeddings = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            chunks.append(embeddings.cpu().numpy())
        return np.vstack(chunks)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def run_embedding_similarity(
    *,
    samples_path: str,
    embedding_model_path: str,
    output_path: str,
) -> dict[str, str]:
    records = dedupe_records(load_json_records(samples_path))
    embedding_model = SentenceEmbeddingModel(embedding_model_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_file.parent / "embedding_similarity_metadata.json"
    csv_path = output_file.with_suffix(".csv")

    metadata = ExperimentMetadata(
        topic="first_sentence_analysis",
        command=f"python {Path(__file__).as_posix()}",
        args={
            "stage": "embedding_similarity",
            "samples_path": samples_path,
            "output_path": str(output_file),
            "dedupe_policy": "exact_text_before_embedding",
            "similarity_definition": "cosine_similarity(prompt,response)",
        },
        model={"embedding_model_path": embedding_model_path},
        output_dir=str(output_file.parent),
        input_source=samples_path,
    )
    metadata_path.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    texts = []
    for record in records:
        texts.extend([record["prompt"], record["cleaned_response"]])
    embeddings = embedding_model.encode(texts)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(
        tqdm(records, total=len(records), desc="Computing similarities", unit="record")
    ):
        prompt_vec = embeddings[index * 2]
        response_vec = embeddings[index * 2 + 1]
        similarity = cosine_similarity(prompt_vec, response_vec)
        rows.append(
            {
                **record,
                "prompt_response_cosine_similarity": similarity,
                "prompt_response_cosine_distance": 1.0 - similarity,
            }
        )

    output_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        import csv

        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return {
        "metadata_path": str(metadata_path),
        "output_path": str(output_file),
        "similarities_path": str(output_file),
        "csv_path": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 llm 采样结果，计算 prompt 与回答的 embedding 相似度。"
    )
    parser.add_argument(
        "--samples-path", default="data/first_sentence_analysis/v0/llm_samples.json"
    )
    parser.add_argument("--embedding-model-path", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument(
        "--output-path", default="data/first_sentence_analysis/v0/embedding_similarities.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_embedding_similarity(
        samples_path=args.samples_path,
        embedding_model_path=args.embedding_model_path,
        output_path=args.output_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
