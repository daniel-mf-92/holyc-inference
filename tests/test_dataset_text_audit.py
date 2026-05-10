#!/usr/bin/env python3
"""Host-side checks for eval dataset text quality audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_text_audit.py"
spec = importlib.util.spec_from_file_location("dataset_text_audit", AUDIT_PATH)
dataset_text_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_text_audit"] = dataset_text_audit
spec.loader.exec_module(dataset_text_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_dataset_text_audit_passes_clean_rows_and_writes_record_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        record_csv = root / "records.csv"
        write_jsonl(
            sample,
            [
                {
                    "id": "clean",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which tool measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                }
            ],
        )

        status = dataset_text_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--record-csv",
                str(record_csv),
                "--max-prompt-bytes",
                "256",
                "--max-choice-bytes",
                "64",
                "--fail-on-control-chars",
                "--fail-on-replacement-chars",
                "--fail-on-blank-text",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 1
        assert report["text_field_count"] == 5
        assert report["choice_label_prefix_field_count"] == 0
        assert report["findings"] == []
        assert record_csv.read_text(encoding="utf-8").count("\n") == 6


def test_dataset_text_audit_flags_controls_replacement_byte_budgets_and_choice_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "bad",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "bad\x00prompt",
                    "choices": ["ok", "A. bad\ufffdchoice"],
                    "answer_index": 0,
                }
            ],
        )

        status = dataset_text_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--max-prompt-bytes",
                "4",
                "--max-choice-bytes",
                "4",
                "--fail-on-control-chars",
                "--fail-on-replacement-chars",
                "--fail-on-choice-label-prefixes",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["choice_label_prefix_field_count"] == 1
        assert {"control_character", "unicode_replacement_character", "field_byte_budget_exceeded", "choice_label_prefix"}.issubset(kinds)


if __name__ == "__main__":
    test_dataset_text_audit_passes_clean_rows_and_writes_record_csv()
    test_dataset_text_audit_flags_controls_replacement_byte_budgets_and_choice_labels()
    print("dataset_text_audit_tests=ok")
