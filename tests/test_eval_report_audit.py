#!/usr/bin/env python3
"""Host-side checks for eval report consistency audits."""

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

AUDIT_PATH = BENCH_PATH / "eval_report_audit.py"
spec = importlib.util.spec_from_file_location("eval_report_audit", AUDIT_PATH)
eval_report_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["eval_report_audit"] = eval_report_audit
spec.loader.exec_module(eval_report_audit)


def eval_row(record_id: str, holyc: int, llama: int, answer: int) -> dict[str, object]:
    return {
        "record_id": record_id,
        "dataset": "arc-smoke",
        "split": "validation",
        "answer_index": answer,
        "holyc_prediction": holyc,
        "llama_prediction": llama,
        "holyc_correct": holyc == answer,
        "llama_correct": llama == answer,
        "engines_agree": holyc == llama,
    }


def write_report(path: Path, *, stale_summary: bool = False, status: str = "pass", regressions: list[dict[str, object]] | None = None) -> None:
    rows = [
        eval_row("a", 0, 0, 0),
        eval_row("b", 1, 0, 1),
        eval_row("c", 2, 3, 0),
        eval_row("d", 0, 0, 1),
    ]
    summary = {
        "record_count": 4,
        "holyc_accuracy": 0.5,
        "llama_accuracy": 0.25,
        "agreement": 0.5,
    }
    if stale_summary:
        summary["holyc_accuracy"] = 1.0
        summary["record_count"] = 5
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "status": status,
        "rows": rows,
        "summary": summary,
        "regressions": regressions or [],
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_eval_report_audit_accepts_consistent_summary(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_report(report_path)

    report = eval_report_audit.build_report([report_path], tolerance=1e-9, min_reports=1)

    assert report["status"] == "pass"
    assert report["summary"]["reports"] == 1
    assert report["summary"]["rows"] == 4
    assert report["findings"] == []
    assert report["reports"][0]["recomputed_holyc_accuracy"] == 0.5


def test_eval_report_audit_flags_stale_summary_and_status(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    write_report(report_path, stale_summary=True, regressions=[{"record_id": "b"}])

    report = eval_report_audit.build_report([report_path], tolerance=1e-9, min_reports=2)
    kinds = [finding["kind"] for finding in report["findings"]]

    assert report["status"] == "fail"
    assert "summary_mismatch" in kinds
    assert "status_regression_mismatch" in kinds
    assert "min_reports" in kinds


def test_eval_report_audit_cli_writes_outputs(tmp_path: Path) -> None:
    report_path = tmp_path / "eval.json"
    output = tmp_path / "audit.json"
    markdown = tmp_path / "audit.md"
    csv_path = tmp_path / "audit.csv"
    findings_csv = tmp_path / "findings.csv"
    junit = tmp_path / "audit.xml"
    write_report(report_path)

    status = eval_report_audit.main(
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
            "--fail-on-findings",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    findings = list(csv.DictReader(findings_csv.open(newline="", encoding="utf-8")))
    junit_root = ET.parse(junit).getroot()
    assert status == 0
    assert payload["status"] == "pass"
    assert len(rows) == 1
    assert findings == []
    assert "Eval Report Audit" in markdown.read_text(encoding="utf-8")
    assert junit_root.attrib["name"] == "holyc_eval_report_audit"
    assert junit_root.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-eval-report-audit-tests-") as tmp:
        tmp_path = Path(tmp)
        test_eval_report_audit_accepts_consistent_summary(tmp_path)
        test_eval_report_audit_flags_stale_summary_and_status(tmp_path)
        test_eval_report_audit_cli_writes_outputs(tmp_path)
    print("eval_report_audit_tests=ok")
