from __future__ import annotations

# 这个脚本负责第二阶段 embedding 相似度计算：
# 读取 LLM 采样结果，分别编码 prompt 和回答，并计算二者的余弦相似度与距离。

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from coverse.config import DEFAULT_EMBEDDING_MODEL_PATH
from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.first_sentence_baseline.common import (
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
        for start in range(0, len(texts), batch_size):
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
    output_dir: str,
    command: str,
) -> dict[str, str]:
    records = dedupe_records(load_json_records(samples_path))
    embedding_model = SentenceEmbeddingModel(embedding_model_path)
    experiment_io = ExperimentIO(output_dir)
    run_name = utc_timestamp().replace(":", "-")
    run_dir = experiment_io.prepare_run_dir("first_sentence_baseline_embeddings", run_name)

    metadata = ExperimentMetadata(
        topic="first_sentence_baseline",
        command=command,
        args={
            "stage": "embedding_similarity",
            "samples_path": samples_path,
            "dedupe_policy": "exact_text_before_embedding",
            "similarity_definition": "cosine_similarity(prompt,response)",
        },
        model={"embedding_model_path": embedding_model_path},
        output_dir=str(run_dir),
        input_source=samples_path,
    )
    experiment_io.write_metadata(run_dir, metadata)

    texts = []
    for record in records:
        texts.extend([record["prompt"], record["cleaned_response"]])
    embeddings = embedding_model.encode(texts)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
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

    similarities_path = experiment_io.write_json(run_dir, "embedding_similarity.json", rows)
    experiment_io.write_csv(run_dir, "embedding_similarity.csv", rows)
    return {
        "run_dir": str(run_dir),
        "metadata_path": str(Path(run_dir) / "metadata.json"),
        "similarities_path": str(similarities_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="读取 llm 采样结果，计算 prompt 与回答的 embedding 相似度。"
    )
    parser.add_argument("--samples-path", required=True)
    parser.add_argument("--embedding-model-path", default=DEFAULT_EMBEDDING_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_embedding_similarity(
        samples_path=args.samples_path,
        embedding_model_path=args.embedding_model_path,
        output_dir=args.output_dir,
        command="python coverse/topics/first_sentence_baseline/embedding_similarity.py",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
