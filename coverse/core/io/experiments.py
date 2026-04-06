from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from coverse.core.types import ExperimentMetadata


class ExperimentIO:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def prepare_run_dir(self, topic: str, run_name: str) -> Path:
        run_dir = self.root_dir / topic / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def write_metadata(self, run_dir: str | Path, metadata: ExperimentMetadata) -> Path:
        path = Path(run_dir) / "metadata.json"
        path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_json(self, run_dir: str | Path, filename: str, payload: Any) -> Path:
        path = Path(run_dir) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_csv(self, run_dir: str | Path, filename: str, rows: list[dict[str, Any]]) -> Path:
        path = Path(run_dir) / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path
