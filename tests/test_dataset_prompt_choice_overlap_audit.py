#!/usr/bin/env python3
"""Host-side checks for eval dataset prompt/choice overlap audit gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_prompt_choice_overlap_audit.py"
spec = importlib.util.spec_from_file_location("dataset_prompt_choice_overlap_audit", AUDIT_PATH)
dataset_prompt_choice_overlap_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_prompt_choice_overlap_audit"] = dataset_prompt_choice_overlap_audit
spec.loader.exec_module(dataset_prompt_choice_overlap_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_prompt_choice_overlap_audit_passes_clean_rows_and_writes_record_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        records = root / "records.csv"
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

        status = dataset_prompt_choice_overlap_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--record-csv",
                str(records),
                "--fail-on-any-overlap",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 1
        assert report["overlap_record_count"] == 0
        assert records.read_text(encoding="utf-8").count("\n") == 2


def test_prompt_choice_overlap_audit_flags_answer_and_distractor_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "answer-overlap",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Pick the option that means fire.",
                    "choices": ["ice", "fire", "snow", "rain"],
                    "answer_index": 1,
                },
                {
                    "id": "distractor-overlap",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "The clue mentions snow but asks for the hot item.",
                    "choices": ["warmth", "flame", "snow", "steam"],
                    "answer_index": 0,
                },
            ],
        )

        status = dataset_prompt_choice_overlap_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--fail-on-answer-overlap",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert report["overlap_record_count"] == 2
        assert report["answer_overlap_record_count"] == 1
        assert {"answer_choice_overlap", "distractor_choice_overlap"}.issubset(kinds)


if __name__ == "__main__":
    test_prompt_choice_overlap_audit_passes_clean_rows_and_writes_record_csv()
    test_prompt_choice_overlap_audit_flags_answer_and_distractor_overlap()
    print("dataset_prompt_choice_overlap_audit_tests=ok")
