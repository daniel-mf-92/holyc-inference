#!/usr/bin/env python3
"""Tests for eval score order audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_score_order_audit


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return eval_score_order_audit.build_parser().parse_args(extra)


def test_score_order_audit_passes_matching_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"id": "a", "prediction": 0, "scores": [5, 1]}, {"id": "b", "prediction": "B", "scores": [0, 3]}])
    args = parse_args(["--predictions", f"holyc={predictions}", "--require-both", "--min-checked-records", "2"])

    records, summaries, findings = eval_score_order_audit.audit(args)

    assert findings == []
    assert len(records) == 2
    assert summaries[0].checked_count == 2
    assert summaries[0].match_rate == 1.0


def test_score_order_audit_flags_mismatch_tie_missing_and_range(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {"id": "mismatch", "prediction": 1, "scores": [4, 1]},
            {"id": "tie", "prediction": 0, "scores": [2, 2]},
            {"id": "missing-scores", "prediction": 0},
            {"id": "range", "prediction": 4, "scores": [1, 0]},
        ],
    )
    args = parse_args(["--predictions", f"llama={predictions}", "--require-both"])

    _, _, findings = eval_score_order_audit.audit(args)

    metrics = {finding.metric for finding in findings}
    assert {"prediction_score_mismatch", "top_score_tie", "missing_scores", "prediction_out_of_range"} <= metrics


def test_score_order_cli_writes_sidecars(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "out"
    write_jsonl(predictions, [{"id": "a", "prediction": 0, "scores": [5, 1]}, {"id": "b", "prediction": 1, "scores": [0, 3]}])

    status = eval_score_order_audit.main(
        [
            "--predictions",
            f"holyc={predictions}",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "score_order",
            "--require-both",
            "--min-checked-records",
            "2",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "score_order.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["checked_count"] == 2
    assert "No score order findings." in (output_dir / "score_order.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "score_order_records.csv").open(encoding="utf-8")))
    assert rows[0]["record_id"] == "a"
    finding_rows = list(csv.DictReader((output_dir / "score_order_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "score_order_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_score_order_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_score_order_audit_passes_matching_predictions(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_score_order_audit_flags_mismatch_tie_missing_and_range(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_score_order_cli_writes_sidecars(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
