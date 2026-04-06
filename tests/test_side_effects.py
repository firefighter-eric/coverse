from __future__ import annotations

import importlib
import unittest
from unittest import mock

from coverse.topics.prob_detect.scorer import ProbabilityScorer


class SideEffectTests(unittest.TestCase):
    def test_probability_scorer_is_lazy(self):
        with mock.patch(
            "coverse.topics.prob_detect.scorer.AutoTokenizer.from_pretrained"
        ) as tokenizer_loader, mock.patch(
            "coverse.topics.prob_detect.scorer.AutoModelForMaskedLM.from_pretrained"
        ) as model_loader:
            ProbabilityScorer(model_path="fake-model")

        tokenizer_loader.assert_not_called()
        model_loader.assert_not_called()

    def test_importing_topic_modules_does_not_run_experiments(self):
        importlib.import_module("coverse.topics.multi_chat.runner")
        importlib.import_module("coverse.topics.prob_detect.runner")
        importlib.import_module("coverse.apps.gradio_app")


if __name__ == "__main__":
    unittest.main()
