#!/usr/bin/env python3
"""Host-side checks for eval dataset row fingerprints."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = ROOT / "bench" / "dataset_fingerprint.py"
spec = importlib.util.spec_from_file_location("dataset_fingerprint", FINGERPRINT_PATH)
dataset_fingerprint = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_fingerprint"] = dataset_fingerprint
spec.loader.exec_module(dataset_fingerprint)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_fingerprint_emits_stable_input_and_answer_hashes() -> None:
    rows = [
        {
            "id": "fingerprint-a",
            "dataset": "arc",
            "split": "validation",
            "prompt": "Pick the letter A.",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "provenance": "synthetic fingerprint unit test",
        },
        {
            "id": "fingerprint-b",
            "dataset": "truthfulqa",
            "split": "validation",
            "prompt": "Pick the letter C.",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 2,
            "provenance": "synthetic fingerprint unit test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dataset.jsonl"
        output = Path(tmp) / "fingerprints.json"
        jsonl = Path(tmp) / "fingerprints.jsonl"
        write_jsonl(source, rows)

        status = dataset_fingerprint.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--jsonl",
                str(jsonl),
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        fingerprints = report["fingerprints"]
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 2
        assert report["dataset_split_counts"] == {"arc": {"validation": 1}, "truthfulqa": {"validation": 1}}
        assert report["answer_histogram"] == {"0": 1, "2": 1}
        assert len({row["input_sha256"] for row in fingerprints}) == 2
        assert all(len(row["answer_payload_sha256"]) == 64 for row in fingerprints)
        assert jsonl.read_text(encoding="utf-8").count("\n") == 2


def test_fingerprint_gates_duplicate_ids_and_conflicting_inputs() -> None:
    rows = [
        {
            "id": "duplicate-id",
            "dataset": "arc",
            "split": "validation",
            "prompt": "Same normalized prompt?",
            "choices": ["yes", "no"],
            "answer_index": 0,
            "provenance": "synthetic fingerprint unit test",
        },
        {
            "id": "duplicate-id",
            "dataset": "arc",
            "split": "validation",
            "prompt": " Same   normalized prompt? ",
            "choices": ["yes", "no"],
            "answer_index": 1,
            "provenance": "synthetic fingerprint unit test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "dataset.jsonl"
        output = Path(tmp) / "fingerprints.json"
        write_jsonl(source, rows)

        status = dataset_fingerprint.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--fail-on-duplicate-ids",
                "--fail-on-conflicting-input-answers",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert kinds == {"conflicting_input_answers", "duplicate_record_id"}


if __name__ == "__main__":
    test_fingerprint_emits_stable_input_and_answer_hashes()
    test_fingerprint_gates_duplicate_ids_and_conflicting_inputs()
    print("dataset_fingerprint_tests=ok")
