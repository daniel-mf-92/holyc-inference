#!/usr/bin/env python3
"""Tests for normalized dataset stats reporting."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_stats_report


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sample_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "arc-1",
            "dataset": "arc-smoke",
            "split": "validation",
            "question": "Which tool measures temperature?",
            "choices": [
                {"label": "A", "text": "thermometer"},
                {"label": "B", "text": "ruler"},
                {"label": "C", "text": "scale"},
                {"label": "D", "text": "compass"},
            ],
            "answerKey": "A",
        },
        {
            "id": "arc-2",
            "dataset": "arc-smoke",
            "split": "validation",
            "question": "Which object is magnetic?",
            "choices": ["iron nail", "paper cup", "glass jar", "wood spoon"],
            "answer_index": 0,
        },
    ]


def test_stats_report_groups_dataset_split(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, sample_rows())
    args = dataset_stats_report.build_parser().parse_args([str(path)])

    loaded, inputs, findings = dataset_stats_report.load_records(args.inputs, args.default_dataset, args.default_split)
    rows = [dataset_stats_report.record_stats(record) for record in loaded]
    stats = dataset_stats_report.build_scope_stats("arc-smoke:validation", rows)

    assert findings == []
    assert inputs[0]["rows"] == 2
    assert stats.records == 2
    assert stats.answer_index_histogram == "0:2"
    assert stats.choice_count_histogram == "4:2"
    assert stats.prompt_bytes_p95 is not None


def test_stats_report_flags_threshold_failures(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, sample_rows()[:1])
    output_dir = tmp_path / "out"

    status = dataset_stats_report.main(
        [
            str(path),
            "--min-records",
            "2",
            "--min-records-per-scope",
            "2",
            "--max-prompt-p95-bytes",
            "4",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "stats",
        ]
    )

    assert status == 1
    report = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert {"min_records", "min_records_per_scope", "prompt_p95_bytes"} <= kinds
    junit_root = ET.parse(output_dir / "stats_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "1"


def test_cli_writes_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, sample_rows())
    output_dir = tmp_path / "out"

    status = dataset_stats_report.main([str(path), "--output-dir", str(output_dir), "--output-stem", "stats"])

    assert status == 0
    report = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["summary"]["records"] == 2
    assert "Dataset Stats Report" in (output_dir / "stats.md").read_text(encoding="utf-8")
    assert "prompt_bytes_p95" in (output_dir / "stats.csv").read_text(encoding="utf-8")
    assert "answer_choice_bytes" in (output_dir / "stats_records.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "stats_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "stats_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_dataset_stats_report"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-stats-test-") as tmp:
        test_stats_report_groups_dataset_split(Path(tmp) / "groups")
        test_stats_report_flags_threshold_failures(Path(tmp) / "failures")
        test_cli_writes_sidecars(Path(tmp) / "cli")
    print("test_dataset_stats_report=ok")
