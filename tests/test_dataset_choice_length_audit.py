#!/usr/bin/env python3
"""Host-side checks for eval dataset choice-length audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_choice_length_audit.py"
spec = importlib.util.spec_from_file_location("dataset_choice_length_audit", AUDIT_PATH)
dataset_choice_length_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_choice_length_audit"] = dataset_choice_length_audit
spec.loader.exec_module(dataset_choice_length_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_dataset_choice_length_audit_passes_balanced_rows_and_writes_record_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        record_csv = root / "records.csv"
        write_jsonl(
            sample,
            [
                {
                    "id": "balanced",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which tool measures temperature?",
                    "choices": ["thermometer", "barometer", "micrometer", "speedometer"],
                    "answer_index": 0,
                }
            ],
        )

        status = dataset_choice_length_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--record-csv",
                str(record_csv),
                "--max-choice-byte-span",
                "16",
                "--max-answer-delta-bytes",
                "16",
                "--max-answer-to-mean-other-ratio",
                "2.0",
                "--min-answer-to-mean-other-ratio",
                "0.5",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 1
        assert report["findings"] == []
        assert report["answer_length_rank_histogram"] == {"middle_or_tied": 1}
        assert record_csv.read_text(encoding="utf-8").count("\n") == 2


def test_dataset_choice_length_audit_flags_answer_length_cues() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "long-answer",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Choose one.",
                    "choices": ["one", "two", "three", "the uniquely long correct answer"],
                    "answer_index": 3,
                }
            ],
        )

        status = dataset_choice_length_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--max-choice-byte-span",
                "8",
                "--max-answer-delta-bytes",
                "8",
                "--max-answer-to-mean-other-ratio",
                "2.0",
                "--fail-on-unique-longest-answer",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert {"choice_byte_span_exceeded", "answer_length_delta_exceeded", "answer_length_ratio_high", "answer_unique_longest"}.issubset(kinds)


if __name__ == "__main__":
    test_dataset_choice_length_audit_passes_balanced_rows_and_writes_record_csv()
    test_dataset_choice_length_audit_flags_answer_length_cues()
    print("dataset_choice_length_audit_tests=ok")
