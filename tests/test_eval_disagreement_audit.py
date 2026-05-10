#!/usr/bin/env python3
"""Host-side checks for eval disagreement audits."""

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

AUDIT_PATH = BENCH_PATH / "eval_disagreement_audit.py"
spec = importlib.util.spec_from_file_location("eval_disagreement_audit", AUDIT_PATH)
eval_disagreement_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_disagreement_audit"] = eval_disagreement_audit
spec.loader.exec_module(eval_disagreement_audit)


def write_eval_report(path: Path, agreement_count: int = 3) -> None:
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "summary": {
            "record_count": 4,
            "agreement_count": agreement_count,
            "agreement": agreement_count / 4,
            "holyc_accuracy": 0.75,
            "llama_accuracy": 0.5,
            "accuracy_delta_holyc_minus_llama": 0.25,
            "dataset_breakdown": [
                {
                    "dataset": "arc-smoke",
                    "split": "validation",
                    "record_count": 2,
                    "agreement_count": 2,
                    "agreement": 1.0,
                    "holyc_accuracy": 1.0,
                    "llama_accuracy": 0.5,
                    "accuracy_delta_holyc_minus_llama": 0.5,
                },
                {
                    "dataset": "truthfulqa-smoke",
                    "split": "validation",
                    "record_count": 2,
                    "agreement_count": max(0, agreement_count - 2),
                    "agreement": max(0, agreement_count - 2) / 2,
                    "holyc_accuracy": 0.5,
                    "llama_accuracy": 0.5,
                    "accuracy_delta_holyc_minus_llama": 0.0,
                },
            ],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_audit_reports_overall_and_dataset_split_scopes(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_eval_report(report_path, agreement_count=3)

    report = eval_disagreement_audit.audit_reports(
        [report_path],
        min_records=4,
        max_disagreement_pct=25,
        max_dataset_split_disagreement_pct=50,
    )

    assert report["status"] == "pass"
    assert report["scope_count"] == 3
    assert report["scopes"][0]["scope"] == "overall"
    assert report["scopes"][0]["disagreement_count"] == 1
    assert report["scopes"][0]["disagreement_pct"] == 25.0
    assert report["findings"] == []


def test_audit_gates_disagreement_pct(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_eval_report(report_path, agreement_count=2)

    report = eval_disagreement_audit.audit_reports(
        [report_path],
        min_records=4,
        max_disagreement_pct=25,
        max_dataset_split_disagreement_pct=25,
    )

    kinds = [finding["kind"] for finding in report["findings"]]
    assert report["status"] == "fail"
    assert kinds.count("disagreement_pct_exceeded") == 2


def test_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    findings_csv = tmp_path / "findings.csv"
    junit = tmp_path / "audit.xml"
    write_eval_report(report_path, agreement_count=3)

    status = eval_disagreement_audit.main(
        [
            str(report_path),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--findings-csv",
            str(findings_csv),
            "--junit",
            str(junit),
            "--min-records",
            "4",
            "--max-disagreement-pct",
            "25",
            "--max-dataset-split-disagreement-pct",
            "50",
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    finding_rows = list(csv.DictReader(findings_csv.open(newline="", encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows) == 3
    assert rows[0]["scope"] == "overall"
    assert finding_rows == []
    assert "Eval Disagreement Audit" in markdown.read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_disagreement_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-disagreement-audit-tests-") as tmp:
        tmp_path = Path(tmp)
        test_audit_reports_overall_and_dataset_split_scopes(tmp_path)
        test_audit_gates_disagreement_pct(tmp_path)
        test_cli_writes_outputs(tmp_path)
    print("eval_disagreement_audit_tests=ok")
