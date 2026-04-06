from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline


@dataclass(slots=True)
class ProbabilityScore:
    text: str
    target: str
    token_results: list[dict[str, Any]]
    prob: float
    log_prob: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProbabilityScorer:
    def __init__(self, model_path: str, device_map: str = "auto"):
        self.model_path = model_path
        self.device_map = device_map
        self._tokenizer = None
        self._model = None
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
        )
        self._pipeline = FillMaskPipeline(model=self._model, tokenizer=self._tokenizer)

    @property
    def tokenizer(self):
        self._ensure_pipeline()
        return self._tokenizer

    @property
    def pipeline(self):
        self._ensure_pipeline()
        return self._pipeline

    def score_one_token(self, text: str, target: str, index: int) -> dict[str, Any]:
        results = self.pipeline(text, targets=[target], top_k=1)
        return results[index][0]

    def score(self, text: str, target: str) -> ProbabilityScore:
        target_token_ids = self.tokenizer(target)["input_ids"][1:-1]
        target_tokens = [self.tokenizer.decode([token_id]) for token_id in target_token_ids]

        token_results = []
        for index, token in enumerate(target_tokens):
            token_results.append(self.score_one_token(text, target=token, index=index))

        prob = 1.0
        log_prob = 0.0
        for token_result in token_results:
            prob *= token_result["score"]
            log_prob += -1.0 * np.log(token_result["score"] + 1e-12)

        return ProbabilityScore(
            text=text,
            target=target,
            token_results=token_results,
            prob=prob,
            log_prob=log_prob,
        )
