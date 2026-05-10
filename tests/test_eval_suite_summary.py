#!/usr/bin/env python3
"""Tests for host-side eval suite summaries."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_suite_summary


def write_report(path: Path, *, status: str = "pass", records: int = 4, accuracy: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q8_0",
                "gold_sha256": "a" * 64,
                "holyc_predictions_sha256": "b" * 64,
                "llama_predictions_sha256": "c" * 64,
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": records,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "accuracy_delta_holyc_minus_llama": accuracy - 1.0,
                    "agreement": accuracy,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_record_extracts_eval_compare_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_report(artifact, records=7, accuracy=0.75)

    record = eval_suite_summary.load_record(artifact)

    assert record.status == "pass"
    assert record.dataset == "smoke-eval"
    assert record.record_count == 7
    assert record.holyc_accuracy == 0.75
    assert record.agreement == 0.75


def test_evaluate_flags_failed_reports_and_low_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_report(artifact, status="fail", records=2, accuracy=0.5)
    record = eval_suite_summary.load_record(artifact)
    args = eval_suite_summary.parse_args(
        [
            str(artifact),
            "--min-reports",
            "2",
            "--min-records",
            "4",
            "--min-holyc-accuracy",
            "0.9",
            "--min-agreement",
            "0.9",
            "--fail-on-failed-reports",
            "--fail-on-regressions",
            "--require-dataset",
            "arc",
            "--require-split",
            "test",
            "--require-model",
            "other-model",
            "--require-quantization",
            "Q4_0",
        ]
    )

    findings = eval_suite_summary.evaluate([record], args)

    assert {finding.gate for finding in findings} == {
        "min_reports",
        "min_records",
        "required_dataset",
        "required_split",
        "required_model",
        "required_quantization",
        "report_status",
        "regressions",
        "min_holyc_accuracy",
        "min_agreement",
    }


def test_cli_writes_suite_summary_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output_dir = tmp_path / "out"
    write_report(first, records=2, accuracy=1.0)
    write_report(second, records=2, accuracy=0.5)

    status = eval_suite_summary.main(
        [
            str(first),
            str(second),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "suite",
            "--min-holyc-accuracy",
            "0.9",
            "--min-agreement",
            "0.9",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "suite.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["summary"]["reports"] == 2
    assert payload["summary"]["records"] == 4
    assert payload["summary"]["weighted_holyc_accuracy"] == 0.75
    rows = list(csv.DictReader((output_dir / "suite.csv").open(encoding="utf-8")))
    assert [row["source"] for row in rows] == [str(first), str(second)]
    finding_rows = list(csv.DictReader((output_dir / "suite_findings.csv").open(encoding="utf-8")))
    assert {row["gate"] for row in finding_rows} == {"min_holyc_accuracy", "min_agreement"}
    markdown = (output_dir / "suite.md").read_text(encoding="utf-8")
    assert "Eval Suite Summary" in markdown
    assert "HolyC-vs-llama agreement" in markdown
    junit_root = ET.parse(output_dir / "suite_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_suite_summary"
    assert junit_root.attrib["failures"] == "1"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "load"
        path.mkdir()
        test_load_record_extracts_eval_compare_summary(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluate"
        path.mkdir()
        test_evaluate_flags_failed_reports_and_low_metrics(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_suite_summary_artifacts(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
