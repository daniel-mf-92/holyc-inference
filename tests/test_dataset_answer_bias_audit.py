#!/usr/bin/env python3
"""Host-side checks for answer-length bias audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_answer_bias_audit.py"
spec = importlib.util.spec_from_file_location("dataset_answer_bias_audit", AUDIT_PATH)
dataset_answer_bias_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_answer_bias_audit"] = dataset_answer_bias_audit
spec.loader.exec_module(dataset_answer_bias_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_answer_length_bias_gates() -> None:
    rows = [
        {
            "id": "bias-a",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Pick the answer",
            "choices": ["no", "also no", "a deliberately long correct answer", "nope"],
            "answer_index": 2,
            "provenance": "synthetic answer bias test",
        },
        {
            "id": "bias-b",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Pick the answer again",
            "choices": ["red", "blue", "another deliberately long correct answer", "green"],
            "answer_index": 2,
            "provenance": "synthetic answer bias test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "biased.jsonl"
        output = Path(tmp) / "answer_bias.json"
        record_csv = Path(tmp) / "answer_bias_records.csv"
        write_jsonl(source, rows)

        status = dataset_answer_bias_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--record-csv",
                str(record_csv),
                "--max-answer-longest-pct",
                "50",
                "--max-mean-answer-distractor-ratio",
                "2",
                "--check-dataset-splits",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["summary"]["position_histogram"] == {"longest": 2}
        assert report["summary"]["answer_longest_or_tied_pct"] == 100.0
        assert report["record_telemetry"][0]["answer_length_position"] == "longest"
        assert {"answer_longest_bias", "mean_answer_too_long"} <= kinds
        assert {"overall", "unit:validation"} <= {finding["scope"] for finding in report["findings"]}
        assert report["dataset_split_summaries"]["unit:validation"]["position_histogram"] == {"longest": 2}
        assert "answer_length_position" in record_csv.read_text(encoding="utf-8").splitlines()[0]


if __name__ == "__main__":
    test_answer_length_bias_gates()
    print("dataset_answer_bias_audit_tests=ok")
