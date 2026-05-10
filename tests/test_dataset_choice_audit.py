#!/usr/bin/env python3
"""Host-side checks for dataset choice audit telemetry."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHOICE_PATH = ROOT / "bench" / "dataset_choice_audit.py"
spec = importlib.util.spec_from_file_location("dataset_choice_audit", CHOICE_PATH)
dataset_choice_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_choice_audit"] = dataset_choice_audit
spec.loader.exec_module(dataset_choice_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_answer_choice_byte_telemetry() -> None:
    rows = [
        {
            "id": "choice-bytes",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Pick the answer.",
            "choices": ["aa", "bbbb", "cccccc", "dddddddd"],
            "answer_index": 2,
            "provenance": "synthetic choice byte telemetry test",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "choices.jsonl"
        output = Path(tmp) / "choice.json"
        record_csv = Path(tmp) / "choice_records.csv"
        write_jsonl(source, rows)

        status = dataset_choice_audit.main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--record-csv",
                str(record_csv),
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        telemetry = report["record_telemetry"][0]
        csv_text = record_csv.read_text(encoding="utf-8")
        assert status == 0
        assert telemetry["answer_choice_bytes"] == 6
        assert telemetry["total_choice_bytes"] == 20
        assert telemetry["answer_choice_byte_pct"] == 30.0
        assert "answer_choice_bytes" in csv_text
        assert "answer_choice_byte_pct" in csv_text


if __name__ == "__main__":
    test_answer_choice_byte_telemetry()
    print("dataset_choice_audit_tests=ok")
