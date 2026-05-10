#!/usr/bin/env python3
"""Host-side checks for eval dataset schema audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "bench" / "dataset_schema_audit.py"
spec = importlib.util.spec_from_file_location("dataset_schema_audit", SCHEMA_PATH)
dataset_schema_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_schema_audit"] = dataset_schema_audit
spec.loader.exec_module(dataset_schema_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_duplicate_payload_and_conflicting_answer_gates() -> None:
    rows = [
        {
            "id": "dup-a",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Pick the color",
            "choices": ["red", "blue", "green", "yellow"],
            "answer_index": 0,
            "provenance": "synthetic duplicate payload test",
        },
        {
            "id": "dup-b",
            "dataset": "unit",
            "split": "validation",
            "prompt": " Pick  the color ",
            "choices": ["Red", "blue", "green", "yellow"],
            "answer_index": 1,
            "provenance": "synthetic duplicate payload test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dups.jsonl"
        output = Path(tmp) / "schema.json"
        write_jsonl(source, rows)

        status = dataset_schema_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--fail-on-duplicate-payloads",
                "--fail-on-conflicting-payload-answers",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["duplicate_payloads"][0]["conflicting_answers"] is True
        assert report["duplicate_payloads"][0]["answer_histogram"] == {"0": 1, "1": 1}
        assert len(report["record_telemetry"]) == 2
        assert report["record_telemetry"][0]["choice_count"] == 4
        assert report["record_telemetry"][0]["record_payload_bytes"] > report["record_telemetry"][0]["prompt_bytes"]
        assert report["record_telemetry"][0]["payload_key_sha256"] == report["duplicate_payloads"][0]["key_sha256"]
        assert kinds == {"duplicate_payload", "conflicting_payload_answers"}


def test_answer_label_coverage_gates() -> None:
    rows = [
        {
            "id": "arc-a",
            "dataset": "arc",
            "split": "validation",
            "prompt": "ARC question A",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic label coverage test",
        },
        {
            "id": "arc-b",
            "dataset": "arc",
            "split": "validation",
            "prompt": "ARC question B",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 1,
            "provenance": "synthetic label coverage test",
        },
        {
            "id": "truthfulqa-a",
            "dataset": "truthfulqa",
            "split": "validation",
            "prompt": "TruthfulQA question A",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic label coverage test",
        },
        {
            "id": "truthfulqa-b",
            "dataset": "truthfulqa",
            "split": "validation",
            "prompt": "TruthfulQA question B",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic label coverage test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "labels.jsonl"
        output = Path(tmp) / "schema.json"
        write_jsonl(source, rows)

        status = dataset_schema_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--min-answer-labels",
                "3",
                "--min-dataset-split-answer-labels",
                "2",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["answer_label_count"] == 2
        assert report["dataset_split_answer_label_counts"] == {
            "arc": {"validation": 2},
            "truthfulqa": {"validation": 1},
        }
        assert kinds == {"too_few_answer_labels", "dataset_split_too_few_answer_labels"}


if __name__ == "__main__":
    test_duplicate_payload_and_conflicting_answer_gates()
    test_answer_label_coverage_gates()
    print("dataset_schema_audit_tests=ok")
