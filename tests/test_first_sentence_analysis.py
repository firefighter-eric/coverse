from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from coverse.topics.first_sentence_analysis.analysis import run_analysis
from coverse.topics.first_sentence_analysis.common import clean_generated_sentence
from coverse.topics.first_sentence_analysis.common import load_system_prompt
from coverse.topics.first_sentence_analysis.embedding_similarity import (
    cosine_similarity,
    run_embedding_similarity,
)
from coverse.topics.first_sentence_analysis.llm_sample import run_llm_sample
from coverse.topics.first_sentence_analysis.prompts import (
    DEFAULT_PROMPTS_PATH,
    load_prompt_specs_from_file,
)


class FakeGenerator:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def generate_one(self, prompt: str):
        self.calls.append(prompt)
        if prompt.endswith("红绿灯"):
            values = ["第一句回答", "第一句回答", "第二句回答"]
        else:
            values = ["第三句回答", "第四句回答", "第五句回答"]
        value = values[(len(self.calls) - 1) % len(values)]
        return {"raw": value + "。更多内容", "cleaned": clean_generated_sentence(value + "。更多内容")}


class FakeEmbeddingModel:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def encode(self, texts: list[str]):
        mapping = {
            "我在路边等红绿灯": np.array([1.0, 0.0]),
            "今天我收到了一封电子邮件": np.array([0.0, 1.0]),
            "第一句回答": np.array([1.0, 0.0]),
            "第二句回答": np.array([0.0, 1.0]),
            "第三句回答": np.array([1.0, 0.0]),
            "第四句回答": np.array([0.5, 0.5]),
            "第五句回答": np.array([0.0, 1.0]),
        }
        return np.vstack([mapping[text] for text in texts])


class FirstSentenceAnalysisTests(unittest.TestCase):
    def test_clean_generated_sentence_keeps_single_sentence(self):
        cleaned = clean_generated_sentence("<think>x</think> 这是第一句。\n这是第二句")
        self.assertEqual(cleaned, "这是第一句")

    def test_default_prompt_file_exists_and_loads(self):
        self.assertTrue(DEFAULT_PROMPTS_PATH.exists())
        specs = load_prompt_specs_from_file(DEFAULT_PROMPTS_PATH)
        self.assertEqual(len(specs), 20)

    def test_default_system_prompt_file_exists_and_loads(self):
        prompt = load_system_prompt("data/first_sentence_analysis/v0/system_prompt.md")
        self.assertIn("你是一名中文故事续写参与者", prompt)

    def test_cosine_similarity_for_small_vectors(self):
        value = cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        self.assertEqual(value, 0.0)

    def test_llm_sample_writes_outputs(self):
        prompts_payload = [
            {"scenario": "公共空间与街道", "text": "我在路边等红绿灯"},
            {"scenario": "室内与办公场景", "text": "今天我收到了一封电子邮件"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_path = Path(tmp_dir) / "prompts.json"
            prompts_path.write_text(json.dumps(prompts_payload, ensure_ascii=False), encoding="utf-8")

            with mock.patch(
                "coverse.topics.first_sentence_analysis.llm_sample.NextSentenceGenerator",
                FakeGenerator,
            ):
                result = run_llm_sample(
                    output_path=str(Path(tmp_dir) / "llm_samples.json"),
                    prompts_file=str(prompts_path),
                    system_prompt_file="data/first_sentence_analysis/v0/system_prompt.md",
                    samples_per_prompt=3,
                    seed=7,
                )

            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            samples = json.loads(Path(result["samples_path"]).read_text(encoding="utf-8"))

            self.assertEqual(metadata["topic"], "first_sentence_analysis")
            self.assertEqual(metadata["args"]["stage"], "llm_sample")
            self.assertEqual(
                metadata["args"]["system_prompt_file"],
                "data/first_sentence_analysis/v0/system_prompt.md",
            )
            self.assertEqual(metadata["args"]["concurrency"], 16)
            self.assertEqual(len(samples), 6)

    def test_embedding_similarity_writes_outputs(self):
        records = [
            {
                "prompt_id": 1,
                "scenario": "公共空间与街道",
                "prompt": "我在路边等红绿灯",
                "sample_index": 1,
                "raw_response": "第一句回答",
                "cleaned_response": "第一句回答",
            },
            {
                "prompt_id": 1,
                "scenario": "公共空间与街道",
                "prompt": "我在路边等红绿灯",
                "sample_index": 2,
                "raw_response": "第一句回答",
                "cleaned_response": "第一句回答",
            },
            {
                "prompt_id": 2,
                "scenario": "室内与办公场景",
                "prompt": "今天我收到了一封电子邮件",
                "sample_index": 1,
                "raw_response": "第四句回答",
                "cleaned_response": "第四句回答",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = Path(tmp_dir) / "llm_samples.json"
            samples_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

            with mock.patch(
                "coverse.topics.first_sentence_analysis.embedding_similarity.SentenceEmbeddingModel",
                FakeEmbeddingModel,
            ):
                result = run_embedding_similarity(
                    samples_path=str(samples_path),
                    embedding_model_path="/tmp/fake-model",
                    output_path=str(Path(tmp_dir) / "embedding_similarity.json"),
                )

            rows = json.loads(Path(result["similarities_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertIn("prompt_response_cosine_similarity", rows[0])

    def test_analysis_writes_outputs(self):
        rows = [
            {
                "prompt_id": 1,
                "scenario": "公共空间与街道",
                "prompt": "我在路边等红绿灯",
                "sample_index": 1,
                "cleaned_response": "第一句回答",
                "prompt_response_cosine_similarity": 1.0,
                "prompt_response_cosine_distance": 0.0,
            },
            {
                "prompt_id": 1,
                "scenario": "公共空间与街道",
                "prompt": "我在路边等红绿灯",
                "sample_index": 2,
                "cleaned_response": "第二句回答",
                "prompt_response_cosine_similarity": 0.0,
                "prompt_response_cosine_distance": 1.0,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            similarities_path = Path(tmp_dir) / "embedding_similarity.json"
            similarities_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

            result = run_analysis(
                similarities_path=str(similarities_path),
                output_path=str(Path(tmp_dir) / "analysis_details.json"),
            )

            with Path(result["ranking_path"]).open(encoding="utf-8") as file:
                ranking_rows = list(csv.DictReader(file))
            self.assertEqual(len(ranking_rows), 1)
            self.assertIn("prompt_response_distance_variance", ranking_rows[0])

    def test_analysis_marks_non_computable_when_unique_count_below_two(self):
        rows = [
            {
                "prompt_id": 1,
                "scenario": "公共空间与街道",
                "prompt": "我在路边等红绿灯",
                "sample_index": 1,
                "cleaned_response": "第一句回答",
                "prompt_response_cosine_similarity": 1.0,
                "prompt_response_cosine_distance": 0.0,
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            similarities_path = Path(tmp_dir) / "embedding_similarity.json"
            similarities_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

            result = run_analysis(
                similarities_path=str(similarities_path),
                output_path=str(Path(tmp_dir) / "analysis_details.json"),
            )

            details = json.loads(Path(result["details_path"]).read_text(encoding="utf-8"))
            self.assertFalse(details[0]["computable"])

    def test_topic_readme_mentions_new_pipeline(self):
        readme = Path("coverse/topics/first_sentence_analysis/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("llm_sample.py", readme)
        self.assertIn("embedding_similarity.py", readme)
        self.assertIn("analysis.py", readme)


if __name__ == "__main__":
    unittest.main()
