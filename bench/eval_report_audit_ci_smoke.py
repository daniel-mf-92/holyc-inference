#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval report consistency audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def row(record_id: str, holyc: int, llama: int, answer: int) -> dict[str, object]:
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


def write_report(path: Path, *, stale: bool = False) -> None:
    rows = [
        row("a", 0, 0, 0),
        row("b", 1, 0, 1),
        row("c", 2, 3, 0),
        row("d", 0, 0, 1),
    ]
    summary = {"record_count": 4, "holyc_accuracy": 0.5, "llama_accuracy": 0.25, "agreement": 0.5}
    regressions: list[dict[str, object]] = []
    if stale:
        summary["agreement"] = 1.0
        regressions = [{"record_id": "c", "kind": "llama_only_correct"}]
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "status": "pass",
        "rows": rows,
        "summary": summary,
        "regressions": regressions,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-report-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "eval_compare_pass.json"
        write_report(passing)
        output_dir = tmp_path / "out"

        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "eval_report_audit.py"),
                str(passing),
                "--output",
                str(output_dir / "eval_report_audit_smoke.json"),
                "--markdown",
                str(output_dir / "eval_report_audit_smoke.md"),
                "--csv",
                str(output_dir / "eval_report_audit_smoke.csv"),
                "--findings-csv",
                str(output_dir / "eval_report_audit_smoke_findings.csv"),
                "--junit",
                str(output_dir / "eval_report_audit_smoke_junit.xml"),
                "--fail-on-findings",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode
        report = json.loads((output_dir / "eval_report_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 4, "unexpected_row_count"):
            return rc
        if rc := require(
            "Eval Report Audit" in (output_dir / "eval_report_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_markdown",
        ):
            return rc
        junit = ET.parse(output_dir / "eval_report_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_report_audit", "missing_junit"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failing = tmp_path / "eval_compare_fail.json"
        write_report(failing, stale=True)
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "eval_report_audit.py"),
                str(failing),
                "--output",
                str(output_dir / "eval_report_audit_fail.json"),
                "--markdown",
                str(output_dir / "eval_report_audit_fail.md"),
                "--csv",
                str(output_dir / "eval_report_audit_fail.csv"),
                "--findings-csv",
                str(output_dir / "eval_report_audit_fail_findings.csv"),
                "--junit",
                str(output_dir / "eval_report_audit_fail_junit.xml"),
                "--fail-on-findings",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_failure"):
            return rc
        failed_report = json.loads((output_dir / "eval_report_audit_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("summary_mismatch" in kinds, "missing_summary_mismatch"):
            return rc
        if rc := require("status_regression_mismatch" in kinds, "missing_status_regression_mismatch"):
            return rc

    print("eval_report_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
