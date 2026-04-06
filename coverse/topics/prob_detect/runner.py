from __future__ import annotations

from pathlib import Path

from coverse.core.io import ExperimentIO
from coverse.core.types import ExperimentMetadata, utc_timestamp
from coverse.topics.prob_detect.scorer import ProbabilityScorer


def load_texts(texts: list[str] | None, text_file: str | None) -> list[str]:
    values = [text.strip() for text in texts or [] if text.strip()]
    if text_file:
        file_texts = Path(text_file).read_text(encoding="utf-8").splitlines()
        values.extend(text.strip() for text in file_texts if text.strip())
    if not values:
        raise ValueError("At least one text or a text file is required.")
    return values


def run_probability_experiment(
    *,
    model_path: str,
    target: str,
    output_dir: str,
    command: str,
    texts: list[str] | None = None,
    text_file: str | None = None,
    device_map: str = "auto",
) -> dict[str, str]:
    scorer = ProbabilityScorer(model_path=model_path, device_map=device_map)
    samples = load_texts(texts, text_file)
    experiment_io = ExperimentIO(output_dir)
    run_name = utc_timestamp().replace(":", "-")
    run_dir = experiment_io.prepare_run_dir("prob_detect", run_name)

    metadata = ExperimentMetadata(
        topic="prob_detect",
        command=command,
        args={
            "target": target,
            "text_file": text_file,
            "device_map": device_map,
            "text_count": len(samples),
        },
        model={"model_path": model_path},
        output_dir=str(run_dir),
        input_source=text_file,
    )
    experiment_io.write_metadata(run_dir, metadata)

    results = [scorer.score(text, target).to_dict() for text in samples]
    json_path = experiment_io.write_json(run_dir, "results.json", results)
    csv_rows = [
        {
            "text": item["text"],
            "target": item["target"],
            "prob": item["prob"],
            "log_prob": item["log_prob"],
        }
        for item in results
    ]
    csv_path = experiment_io.write_csv(run_dir, "results.csv", csv_rows)
    return {
        "run_dir": str(run_dir),
        "metadata_path": str(Path(run_dir) / "metadata.json"),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
