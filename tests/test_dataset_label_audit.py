#!/usr/bin/env python3
"""Host-side checks for dataset raw label audit."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "bench"
if str(BENCH_PATH) not in sys.path:
    sys.path.insert(0, str(BENCH_PATH))

AUDIT_PATH = BENCH_PATH / "dataset_label_audit.py"
spec = importlib.util.spec_from_file_location("dataset_label_audit", AUDIT_PATH)
dataset_label_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["dataset_label_audit"] = dataset_label_audit
spec.loader.exec_module(dataset_label_audit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_valid_arc_and_truthfulqa_labels_pass(tmp_path: Path) -> None:
    source = tmp_path / "good.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "arc-good",
                "question": "Which object measures temperature?",
                "choices": [
                    {"label": "A", "text": "thermometer"},
                    {"label": "B", "text": "ruler"},
                ],
                "answerKey": "A",
            },
            {
                "id": "truthfulqa-good",
                "question": "Pick one.",
                "mc1_targets": {"choices": ["correct", "incorrect"], "labels": [1, 0]},
            },
        ],
    )

    report = dataset_label_audit.audit_dataset(source, require_contiguous_arc_labels=True)

    assert report["status"] == "pass"
    assert report["rows"] == 2
    assert report["shape_counts"] == {"arc": 1, "truthfulqa": 1}
    assert report["findings"] == []


def test_bad_raw_labels_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "arc-bad",
                "question": "Which letter is first?",
                "choices": [{"label": "A", "text": "A"}, {"label": "A", "text": "duplicate"}],
                "answerKey": "Z",
            },
            {"id": "hs-bad", "ctx": "A person starts", "endings": ["x", "y"], "label": "2"},
            {
                "id": "tqa-bad",
                "question": "Pick one.",
                "mc1_targets": {"choices": ["one", "two"], "labels": [1, 1]},
            },
        ],
    )

    report = dataset_label_audit.audit_dataset(source)

    kinds = {finding["kind"] for finding in report["findings"]}
    assert report["status"] == "fail"
    assert "duplicate_choice_labels" in kinds
    assert "answer_label_missing" in kinds
    assert "hellaswag_label_out_of_range" in kinds
    assert "truthfulqa_correct_label_count" in kinds


def test_cli_writes_outputs(tmp_path: Path) -> None:
    source = tmp_path / "good.jsonl"
    output = tmp_path / "label.json"
    markdown = tmp_path / "label.md"
    csv_path = tmp_path / "label.csv"
    record_csv = tmp_path / "label_records.csv"
    junit = tmp_path / "label.xml"
    write_jsonl(
        source,
        [
            {
                "id": "arc-good",
                "question": "Which letter is first?",
                "choices": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
                "answerKey": "A",
            }
        ],
    )

    status = dataset_label_audit.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--record-csv",
            str(record_csv),
            "--junit",
            str(junit),
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    finding_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    record_rows = list(csv.DictReader(record_csv.open(encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert "Dataset Label Audit" in markdown.read_text(encoding="utf-8")
    assert finding_rows == []
    assert record_rows[0]["record_id"] == "arc-good"
    assert junit_root.attrib["name"] == "holyc_dataset_label_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-label-audit-tests-") as tmp:
        tmp_path = Path(tmp)
        test_valid_arc_and_truthfulqa_labels_pass(tmp_path)
        test_bad_raw_labels_are_reported(tmp_path)
        test_cli_writes_outputs(tmp_path)
    print("dataset_label_audit_tests=ok")
