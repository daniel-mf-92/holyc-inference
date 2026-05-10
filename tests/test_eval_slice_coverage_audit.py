#!/usr/bin/env python3
"""Tests for host-side eval slice coverage audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_slice_coverage_audit


def write_report(path: Path, *, status: str = "pass", records: int = 2, accuracy: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "dataset": "smoke-suite",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q8_0",
                "regressions": [] if status == "pass" else [{"metric": "accuracy"}],
                "summary": {
                    "record_count": records,
                    "holyc_accuracy": accuracy,
                    "llama_accuracy": 1.0,
                    "agreement": accuracy,
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "record_count": records,
                            "holyc_accuracy": accuracy,
                            "llama_accuracy": 1.0,
                            "agreement": accuracy,
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_records_extracts_dataset_breakdown(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_report(artifact, records=5, accuracy=0.75)

    records, findings = eval_slice_coverage_audit.load_records(artifact)

    assert findings == []
    assert len(records) == 1
    assert records[0].dataset == "arc-smoke"
    assert records[0].record_count == 5
    assert records[0].holyc_accuracy == 0.75


def test_evaluate_flags_missing_and_undersized_slices(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_report(artifact, status="fail", records=1, accuracy=0.5)
    records, findings = eval_slice_coverage_audit.load_records(artifact)
    args = eval_slice_coverage_audit.parse_args(
        [
            str(artifact),
            "--require-slice",
            "truthfulqa-smoke:validation",
            "--min-slices",
            "2",
            "--min-records-per-slice",
            "2",
            "--min-slice-holyc-accuracy",
            "0.9",
            "--min-slice-agreement",
            "0.9",
            "--fail-on-failed-reports",
            "--fail-on-regressions",
        ]
    )

    audit_findings = eval_slice_coverage_audit.evaluate(records, findings, args)

    assert {finding.gate for finding in audit_findings} == {
        "min_slices",
        "required_slice",
        "min_records_per_slice",
        "min_slice_holyc_accuracy",
        "min_slice_agreement",
        "report_status",
        "regressions",
    }


def test_cli_writes_slice_coverage_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    output_dir = tmp_path / "out"
    write_report(artifact)

    status = eval_slice_coverage_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "slice_audit",
            "--require-slice",
            "arc-smoke:validation",
            "--min-slices",
            "1",
            "--min-records-per-slice",
            "2",
            "--min-slice-holyc-accuracy",
            "0.95",
            "--min-slice-agreement",
            "0.95",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "slice_audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["slice_counts"] == {"arc-smoke:validation": 2}
    rows = list(csv.DictReader((output_dir / "slice_audit.csv").open(encoding="utf-8")))
    assert rows[0]["dataset"] == "arc-smoke"
    finding_rows = list(csv.DictReader((output_dir / "slice_audit_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    markdown = (output_dir / "slice_audit.md").read_text(encoding="utf-8")
    assert "Eval Slice Coverage Audit" in markdown
    junit_root = ET.parse(output_dir / "slice_audit_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_slice_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "load"
        path.mkdir()
        test_load_records_extracts_dataset_breakdown(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluate"
        path.mkdir()
        test_evaluate_flags_missing_and_undersized_slices(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_slice_coverage_artifacts(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
