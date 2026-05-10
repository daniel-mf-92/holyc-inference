#!/usr/bin/env python3
"""Host-side checks for eval dataset split-overlap audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_split_overlap_audit.py"
spec = importlib.util.spec_from_file_location("dataset_split_overlap_audit", AUDIT_PATH)
dataset_split_overlap_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_split_overlap_audit"] = dataset_split_overlap_audit
spec.loader.exec_module(dataset_split_overlap_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_dataset_split_overlap_audit_passes_unique_splits_and_writes_record_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        record_csv = root / "records.csv"
        write_jsonl(
            sample,
            [
                {
                    "id": "train-one",
                    "dataset": "unit",
                    "split": "train",
                    "prompt": "Which tool measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
                {
                    "id": "validation-one",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which tool measures distance?",
                    "choices": ["ruler", "thermometer", "scale", "compass"],
                    "answer_index": 0,
                },
            ],
        )

        status = dataset_split_overlap_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--record-csv",
                str(record_csv),
                "--fail-on-prompt-overlap",
                "--fail-on-payload-overlap",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 2
        assert report["prompt_overlap_record_count"] == 0
        assert report["payload_overlap_record_count"] == 0
        assert report["findings"] == []
        assert record_csv.read_text(encoding="utf-8").count("\n") == 3


def test_dataset_split_overlap_audit_flags_cross_split_prompt_and_payload_reuse() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "train-copy",
                    "dataset": "unit",
                    "split": "train",
                    "prompt": "Which object measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
                {
                    "id": "validation-copy",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which object measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
            ],
        )

        status = dataset_split_overlap_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--fail-on-prompt-overlap",
                "--fail-on-payload-overlap",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert {"prompt_split_overlap", "payload_split_overlap"}.issubset(kinds)
        assert report["prompt_overlap_record_count"] == 2
        assert report["payload_overlap_record_count"] == 2


if __name__ == "__main__":
    test_dataset_split_overlap_audit_passes_unique_splits_and_writes_record_csv()
    test_dataset_split_overlap_audit_flags_cross_split_prompt_and_payload_reuse()
    print("dataset_split_overlap_audit_tests=ok")
