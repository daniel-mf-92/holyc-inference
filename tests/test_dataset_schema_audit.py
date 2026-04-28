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
        assert kinds == {"duplicate_payload", "conflicting_payload_answers"}


if __name__ == "__main__":
    test_duplicate_payload_and_conflicting_answer_gates()
    print("dataset_schema_audit_tests=ok")
