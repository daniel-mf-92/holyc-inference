#!/usr/bin/env python3
"""Tests for host-side eval result indexing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import eval_result_index


def write_eval_compare(path: Path, *, status: str = "pass", records: int = 4, accuracy: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
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


def write_suite_summary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:01:00Z",
                "status": "pass",
                "summary": {
                    "reports": 1,
                    "records": 4,
                    "findings": 0,
                    "regressions": 0,
                    "weighted_holyc_accuracy": 0.75,
                    "weighted_agreement": 0.5,
                },
                "reports": [],
                "findings": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_artifact_extracts_eval_compare_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_eval_compare(artifact, records=7, accuracy=0.75)

    summary = eval_result_index.load_artifact(artifact)

    assert summary is not None
    assert summary.artifact_type == "eval_compare"
    assert summary.status == "pass"
    assert summary.dataset == "smoke-eval"
    assert summary.quantization == "Q8_0"
    assert summary.record_count == 7
    assert summary.holyc_accuracy == 0.75
    assert summary.agreement == 0.75


def test_load_artifact_extracts_suite_summary_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "suite.json"
    write_suite_summary(artifact)

    summary = eval_result_index.load_artifact(artifact)

    assert summary is not None
    assert summary.artifact_type == "eval_suite_summary"
    assert summary.record_count == 4
    assert summary.suite_reports == 1
    assert summary.holyc_accuracy == 0.75
    assert summary.agreement == 0.5


def test_evaluate_flags_coverage_and_quality_gates(tmp_path: Path) -> None:
    artifact = tmp_path / "eval_compare.json"
    write_eval_compare(artifact, status="fail", records=2, accuracy=0.5)
    summary = eval_result_index.load_artifact(artifact)
    assert summary is not None
    args = eval_result_index.parse_args(
        [
            str(artifact),
            "--min-artifacts",
            "2",
            "--min-records",
            "4",
            "--require-dataset",
            "arc",
            "--require-quantization",
            "Q4_0",
            "--min-holyc-accuracy",
            "0.9",
            "--min-agreement",
            "0.9",
            "--fail-on-failed",
            "--fail-on-regressions",
        ]
    )

    findings = eval_result_index.evaluate([summary], args)

    assert {finding.gate for finding in findings} == {
        "min_artifacts",
        "min_records",
        "required_dataset",
        "required_quantization",
        "status",
        "regressions",
        "min_holyc_accuracy",
        "min_agreement",
    }


def test_cli_writes_index_artifacts(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output_dir = tmp_path / "out"
    write_eval_compare(first, records=2, accuracy=1.0)
    write_eval_compare(second, records=2, accuracy=0.5)

    status = eval_result_index.main(
        [
            str(first),
            str(second),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "index",
            "--min-holyc-accuracy",
            "0.9",
            "--min-agreement",
            "0.9",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert payload["summary"]["artifacts"] == 2
    assert payload["summary"]["records"] == 4
    assert payload["summary"]["weighted_holyc_accuracy"] == 0.75
    rows = list(csv.DictReader((output_dir / "index.csv").open(encoding="utf-8")))
    assert [row["source"] for row in rows] == [str(first), str(second)]
    finding_rows = list(csv.DictReader((output_dir / "index_findings.csv").open(encoding="utf-8")))
    assert {row["gate"] for row in finding_rows} == {"min_holyc_accuracy", "min_agreement"}
    markdown = (output_dir / "index.md").read_text(encoding="utf-8")
    assert "Eval Result Index" in markdown
    junit_root = ET.parse(output_dir / "index_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_eval_result_index"
    assert junit_root.attrib["failures"] == "1"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "load"
        path.mkdir()
        test_load_artifact_extracts_eval_compare_metrics(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "suite"
        path.mkdir()
        test_load_artifact_extracts_suite_summary_metrics(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluate"
        path.mkdir()
        test_evaluate_flags_coverage_and_quality_gates(path)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cli"
        path.mkdir()
        test_cli_writes_index_artifacts(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
