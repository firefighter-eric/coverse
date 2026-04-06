from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from coverse.cli.main import main
from coverse.topics.multi_chat.runner import run_batch_multi_chat
from coverse.topics.prob_detect.runner import run_probability_experiment


class FakeRunner:
    def run(self, *, first_message: str, n_turns: int):
        return [
            {"role": "user", "content": first_message},
            {"role": "agent_1", "content": f"{first_message} reply 1"},
            {"role": "agent_2", "content": f"{first_message} reply 2"},
        ]


class FakeProbabilityScorer:
    def __init__(self, model_path: str, device_map: str = "auto"):
        self.model_path = model_path
        self.device_map = device_map

    def score(self, text: str, target: str):
        return type(
            "Score",
            (),
            {
                "to_dict": lambda self: {
                    "text": text,
                    "target": target,
                    "token_results": [{"score": 0.5}],
                    "prob": 0.5,
                    "log_prob": 0.6931471805599453,
                }
            },
        )()


class TopicRunnerTests(unittest.TestCase):
    def test_multi_chat_runner_writes_metadata_and_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_path = Path(tmp_dir) / "prompts.txt"
            prompts_path.write_text("第一句\n第二句\n", encoding="utf-8")

            with mock.patch(
                "coverse.topics.multi_chat.runner.build_runner",
                return_value=FakeRunner(),
            ):
                result = run_batch_multi_chat(
                    provider="ollama",
                    model="qwen",
                    prompts_path=str(prompts_path),
                    output_dir=tmp_dir,
                    command="coverse topic multi-chat",
                    agent_names=["agent_1", "agent_2"],
                    n_turns=2,
                    concurrency=1,
                    tag="test",
                )

            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            rows = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

            self.assertEqual(metadata["topic"], "multi_chat")
            self.assertEqual(metadata["args"]["n_turns"], 2)
            self.assertEqual(len(rows), 2)
            self.assertIn("messages_json", Path(result["csv_path"]).read_text(encoding="utf-8"))

    def test_probability_runner_writes_metadata_and_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "coverse.topics.prob_detect.runner.ProbabilityScorer",
                FakeProbabilityScorer,
            ):
                result = run_probability_experiment(
                    model_path="fake-model",
                    target="目标",
                    output_dir=tmp_dir,
                    command="coverse topic prob-detect",
                    texts=["文本一", "文本二"],
                )

            metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            rows = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))

            self.assertEqual(metadata["topic"], "prob_detect")
            self.assertEqual(metadata["args"]["text_count"], 2)
            self.assertEqual(rows[0]["target"], "目标")


class CLITests(unittest.TestCase):
    def test_multi_chat_cli_smoke(self):
        with mock.patch(
            "coverse.cli.topic.multi_chat.run_batch_multi_chat",
            return_value={"ok": True},
        ) as patched, redirect_stdout(StringIO()):
            exit_code = main(
                [
                    "topic",
                    "multi-chat",
                    "--prompts-path",
                    "prompts.txt",
                ]
            )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        self.assertEqual(patched.call_args.kwargs["provider"], "deepseek")
        self.assertEqual(patched.call_args.kwargs["model"], "deepseek-chat")

    def test_prob_detect_cli_smoke(self):
        with mock.patch(
            "coverse.cli.topic.prob_detect.run_probability_experiment",
            return_value={"ok": True},
        ) as patched, redirect_stdout(StringIO()):
            exit_code = main(
                [
                    "topic",
                    "prob-detect",
                    "--model-path",
                    "bert",
                    "--target",
                    "目标",
                    "--text",
                    "文本",
                ]
            )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()

    def test_serve_cli_smoke(self):
        with mock.patch("coverse.cli.app.serve.launch_app") as patched:
            exit_code = main(
                [
                    "app",
                    "serve",
                ]
            )

        self.assertEqual(exit_code, 0)
        patched.assert_called_once()
        self.assertEqual(patched.call_args.kwargs["provider"], "deepseek")
        self.assertEqual(patched.call_args.kwargs["model"], "deepseek-chat")


if __name__ == "__main__":
    unittest.main()
