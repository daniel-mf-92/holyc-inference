#!/usr/bin/env python3
"""Host-side checks for eval dataset contamination audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_contamination_audit.py"
spec = importlib.util.spec_from_file_location("dataset_contamination_audit", AUDIT_PATH)
dataset_contamination_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_contamination_audit"] = dataset_contamination_audit
spec.loader.exec_module(dataset_contamination_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_cross_dataset_payload_and_answer_conflict_gates() -> None:
    rows = [
        {
            "id": "arc-a",
            "dataset": "arc-local",
            "split": "validation",
            "prompt": "Pick the warm object",
            "choices": ["ice", "fire", "snow", "rain"],
            "answer_index": 1,
            "provenance": "synthetic contamination test",
        },
        {
            "id": "truth-a",
            "dataset": "truthfulqa-local",
            "split": "validation",
            "prompt": " pick  the warm object ",
            "choices": ["Ice", "fire", "snow", "rain"],
            "answer_index": 0,
            "provenance": "synthetic contamination test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "contamination.jsonl"
        output = Path(tmp) / "contamination.json"
        write_jsonl(source, rows)

        status = dataset_contamination_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--fail-on-contamination",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["dataset_count"] == 2
        assert report["unique_prompt_count"] == 1
        assert report["unique_payload_count"] == 1
        assert kinds == {
            "cross_dataset_answer_conflict",
            "cross_dataset_payload_reuse",
            "cross_dataset_prompt_reuse",
        }


if __name__ == "__main__":
    test_cross_dataset_payload_and_answer_conflict_gates()
    print("dataset_contamination_audit_tests=ok")
