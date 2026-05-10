#!/usr/bin/env python3
"""Tests for eval choice distribution audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_choice_distribution_audit


def write_gold(path: Path) -> None:
    rows = [
        {
            "id": "case-1",
            "dataset": "smoke",
            "split": "validation",
            "prompt": "one",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 0,
        },
        {
            "id": "case-2",
            "dataset": "smoke",
            "split": "validation",
            "prompt": "two",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return eval_choice_distribution_audit.build_parser().parse_args(extra)


def test_audit_summarizes_choice_distribution_from_scores(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.jsonl"
    write_gold(gold)
    write_predictions(preds, [{"id": "case-1", "scores": [4, 1, 0, 0]}, {"id": "case-2", "scores": [0, 3, 0, 0]}])
    args = parse_args(["--gold", str(gold), "--predictions", f"holyc={preds}"])

    rows, findings = eval_choice_distribution_audit.audit(args)

    assert findings == []
    by_choice = {row.choice_index: row for row in rows}
    assert by_choice[0].gold_count == 1
    assert by_choice[0].predicted_count == 1
    assert by_choice[1].correct_count == 1


def test_audit_flags_missing_unknown_out_of_range_and_collapse(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.jsonl"
    write_gold(gold)
    write_predictions(preds, [{"id": "case-1", "prediction": 4}, {"id": "unknown", "prediction": 0}])
    args = parse_args(
        [
            "--gold",
            str(gold),
            "--predictions",
            f"llama={preds}",
            "--choice-collapse-min-predictions",
            "1",
            "--max-choice-share",
            "0.75",
        ]
    )

    rows, findings = eval_choice_distribution_audit.audit(args)

    assert rows
    kinds = {finding.kind for finding in findings}
    assert {"missing_prediction", "unknown_prediction", "out_of_range_prediction"} <= kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.jsonl"
    write_gold(gold)
    write_predictions(preds, [{"id": "case-1", "prediction": 0}, {"id": "case-2", "prediction": 1}])
    output_dir = tmp_path / "out"

    status = eval_choice_distribution_audit.main(
        [
            "--gold",
            str(gold),
            "--predictions",
            f"holyc={preds}",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "choice_distribution",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "choice_distribution.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["engines"] == ["holyc"]
    assert "No choice distribution findings." in (output_dir / "choice_distribution.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "choice_distribution.csv").open(encoding="utf-8")))
    assert rows[0]["engine"] == "holyc"
    finding_rows = list(csv.DictReader((output_dir / "choice_distribution_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "choice_distribution_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_choice_distribution_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_summarizes_choice_distribution_from_scores(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_unknown_out_of_range_and_collapse(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
