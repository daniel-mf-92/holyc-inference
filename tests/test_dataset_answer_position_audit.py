#!/usr/bin/env python3
"""Host-side tests for dataset answer-position distribution audit."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import dataset_answer_position_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def rows() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Question A?",
            "choices": ["right", "wrong", "wronger", "wrongest"],
            "answer_index": 0,
        },
        {
            "id": "b",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Question B?",
            "choices": ["wrong", "right", "wronger", "wrongest"],
            "answer_index": 1,
        },
        {
            "id": "c",
            "dataset": "unit",
            "split": "validation",
            "prompt": "Question C?",
            "choices": ["wrong", "right", "wronger", "wrongest"],
            "answer_index": 1,
        },
    ]


def test_answer_position_summary_counts_dominant_position(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    write_jsonl(path, rows())

    loaded, inputs, findings = dataset_answer_position_audit.load_records([path], "eval", "validation")
    summaries = dataset_answer_position_audit.build_summaries(loaded)

    assert findings == []
    assert inputs[0]["rows"] == 3
    assert summaries[0].record_count == 3
    assert summaries[0].distinct_answer_positions == 2
    assert summaries[0].dominant_answer_index == 1
    assert summaries[0].dominant_answer_pct == (2 / 3 * 100.0)
    assert summaries[0].answer_index_histogram == "0:1,1:2"


def test_answer_position_gates_fail_concentrated_subset(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    out = tmp_path / "out"
    write_jsonl(path, rows())

    status = dataset_answer_position_audit.main(
        [
            str(path),
            "--output-dir",
            str(out),
            "--output-stem",
            "answer_position",
            "--min-records",
            "3",
            "--min-distinct-answer-positions",
            "3",
            "--max-dominant-answer-pct",
            "60",
        ]
    )

    assert status == 1
    report = json.loads((out / "answer_position.json").read_text(encoding="utf-8"))
    kinds = {finding["kind"] for finding in report["findings"]}
    assert {"answer_position_coverage", "dominant_answer_position"} <= kinds
    junit_root = ET.parse(out / "answer_position_junit.xml").getroot()
    assert junit_root.attrib["failures"] == "1"


def test_answer_position_cli_writes_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    out = tmp_path / "out"
    write_jsonl(path, rows())

    status = dataset_answer_position_audit.main([str(path), "--output-dir", str(out), "--output-stem", "answer_position"])

    assert status == 0
    report = json.loads((out / "answer_position.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert "Dataset Answer Position Audit" in (out / "answer_position.md").read_text(encoding="utf-8")
    assert "dominant_answer_pct" in (out / "answer_position.csv").read_text(encoding="utf-8")
    assert "kind" in (out / "answer_position_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(out / "answer_position_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_dataset_answer_position_audit"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-answer-position-test-") as tmp:
        test_answer_position_summary_counts_dominant_position(Path(tmp) / "summary")
        test_answer_position_gates_fail_concentrated_subset(Path(tmp) / "gates")
        test_answer_position_cli_writes_sidecars(Path(tmp) / "cli")
    print("test_dataset_answer_position_audit=ok")
