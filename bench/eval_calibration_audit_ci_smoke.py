#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval calibration audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_eval_report(path: Path, *, holyc_ece: float, llama_ece: float, holyc_coverage: float = 1.0) -> None:
    def calibration(ece: float, coverage: float = 1.0) -> dict[str, object]:
        return {
            "accuracy_when_scored": 1.0,
            "brier_score": ece / 2.0,
            "calibration_bins": [],
            "ece": ece,
            "mean_confidence": 1.0 - ece,
            "score_coverage": coverage,
            "scored_count": int(10 * coverage),
            "total_count": 10,
        }

    path.write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_calibration": calibration(holyc_ece, holyc_coverage),
                    "llama_calibration": calibration(llama_ece),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_audit(output_dir: Path, report: Path, stem: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_calibration_audit.py"),
            str(report),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-calibration-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "eval_compare_pass.json"
        write_eval_report(passing_report, holyc_ece=0.02, llama_ece=0.03)
        passed = run_audit(
            tmp_path,
            passing_report,
            "calibration_pass",
            "--min-score-coverage",
            "1.0",
            "--max-ece",
            "0.05",
            "--max-holyc-ece-delta",
            "0.01",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode

        payload = json.loads((tmp_path / "calibration_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["engine_summary_count"] == 2, "unexpected_engine_summary_count"):
            return rc
        if rc := require((tmp_path / "calibration_pass.csv").read_text(encoding="utf-8").startswith("severity,"), "missing_csv_header"):
            return rc
        summary_csv = (tmp_path / "calibration_pass_summaries.csv").read_text(encoding="utf-8")
        if rc := require(summary_csv.startswith("source,engine,"), "missing_summary_csv_header"):
            return rc
        if rc := require("holyc" in summary_csv and "llama" in summary_csv, "missing_summary_csv_engines"):
            return rc
        junit = ET.parse(tmp_path / "calibration_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_calibration_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        failing_report = tmp_path / "eval_compare_fail.json"
        write_eval_report(failing_report, holyc_ece=0.20, llama_ece=0.01, holyc_coverage=0.5)
        failed = run_audit(
            tmp_path,
            failing_report,
            "calibration_fail",
            "--min-score-coverage",
            "1.0",
            "--max-ece",
            "0.05",
            "--max-holyc-ece-delta",
            "0.05",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "calibration_fail.json").read_text(encoding="utf-8"))
        if rc := require(failed_payload["status"] == "fail", "unexpected_failed_report_status"):
            return rc
        messages = [finding["message"] for finding in failed_payload["findings"]]
        if rc := require(any("score_coverage" in message for message in messages), "missing_coverage_finding"):
            return rc
        if rc := require(any("HolyC ECE delta" in message for message in messages), "missing_delta_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "calibration_fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_calibration_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
