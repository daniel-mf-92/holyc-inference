#!/usr/bin/env python3
"""Host-side checks for eval dataset duplicate audit gates."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "dataset_duplicate_audit.py"
spec = importlib.util.spec_from_file_location("dataset_duplicate_audit", AUDIT_PATH)
dataset_duplicate_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_duplicate_audit"] = dataset_duplicate_audit
spec.loader.exec_module(dataset_duplicate_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_dataset_duplicate_audit_passes_unique_rows_and_writes_sidecars() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        markdown = root / "audit.md"
        findings_csv = root / "findings.csv"
        record_csv = root / "records.csv"
        junit = root / "audit.xml"
        write_jsonl(
            sample,
            [
                {
                    "id": "one",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which tool measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
                {
                    "id": "two",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which tool measures distance?",
                    "choices": ["ruler", "thermometer", "scale", "compass"],
                    "answer_index": 0,
                },
            ],
        )

        status = dataset_duplicate_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--csv",
                str(findings_csv),
                "--record-csv",
                str(record_csv),
                "--junit",
                str(junit),
                "--fail-on-duplicate-prompts",
                "--fail-on-duplicate-payloads",
                "--fail-on-findings",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        assert status == 0
        assert report["status"] == "pass"
        assert report["record_count"] == 2
        assert report["duplicate_prompt_record_count"] == 0
        assert report["duplicate_payload_record_count"] == 0
        assert "Dataset Duplicate Audit" in markdown.read_text(encoding="utf-8")
        assert list(csv.DictReader(findings_csv.open(encoding="utf-8"))) == []
        rows = list(csv.DictReader(record_csv.open(encoding="utf-8")))
        assert rows[0]["payload_duplicate_count"] == "1"
        junit_root = ET.parse(junit).getroot()
        assert junit_root.attrib["name"] == "dataset_duplicate_audit"
        assert junit_root.attrib["failures"] == "0"


def test_dataset_duplicate_audit_flags_duplicate_payloads_and_conflicting_answers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sample = root / "sample.jsonl"
        output = root / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "copy-a",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which object measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
                {
                    "id": "copy-b",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": " Which   object measures temperature? ",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 1,
                },
            ],
        )

        status = dataset_duplicate_audit.main(
            [
                "--input",
                str(sample),
                "--output",
                str(output),
                "--fail-on-duplicate-payloads",
                "--fail-on-conflicting-answers",
            ]
        )

        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert status == 1
        assert report["status"] == "fail"
        assert {"conflicting_duplicate_prompt", "conflicting_duplicate_payload"}.issubset(kinds)
        assert report["duplicate_prompt_record_count"] == 2
        assert report["duplicate_payload_record_count"] == 2


if __name__ == "__main__":
    test_dataset_duplicate_audit_passes_unique_rows_and_writes_sidecars()
    test_dataset_duplicate_audit_flags_duplicate_payloads_and_conflicting_answers()
    print("dataset_duplicate_audit_tests=ok")
