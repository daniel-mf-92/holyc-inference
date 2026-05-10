#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset_stats_report.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-stats-ci-") as tmp:
        out = Path(tmp) / "out"
        passed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_stats_report.py"),
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "dataset_stats_report_smoke",
                "--min-records",
                "3",
                "--max-prompt-p95-bytes",
                "256",
                "--max-choice-p95-bytes",
                "128",
            ]
        )
        if passed.returncode != 0:
            return passed.returncode

        report = json.loads((out / "dataset_stats_report_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_dataset_stats_status"):
            return rc
        if rc := require(report["summary"]["records"] == 3, "unexpected_dataset_stats_record_count"):
            return rc
        if rc := require(report["summary"]["scopes"] == 4, "unexpected_dataset_stats_scope_count"):
            return rc
        if rc := require(report["scopes"][0]["scope"] == "all:all", "missing_dataset_stats_global_scope"):
            return rc
        if rc := require(report["scopes"][0]["records"] == 3, "unexpected_dataset_stats_global_count"):
            return rc
        if rc := require("prompt_bytes_p95" in (out / "dataset_stats_report_smoke.csv").read_text(encoding="utf-8"), "missing_stats_csv"):
            return rc
        if rc := require("answer_choice_bytes" in (out / "dataset_stats_report_smoke_records.csv").read_text(encoding="utf-8"), "missing_stats_records_csv"):
            return rc
        if rc := require("No dataset stats findings." in (out / "dataset_stats_report_smoke.md").read_text(encoding="utf-8"), "missing_stats_markdown"):
            return rc
        junit_text = (out / "dataset_stats_report_smoke_junit.xml").read_text(encoding="utf-8")
        if rc := require('name="holyc_dataset_stats_report"' in junit_text, "missing_stats_junit"):
            return rc
        if rc := require('failures="0"' in junit_text, "unexpected_stats_junit_failure"):
            return rc

        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_stats_report.py"),
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "dataset_stats_report_failing",
                "--min-records",
                "4",
                "--max-prompt-p95-bytes",
                "4",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "dataset_stats_bad_thresholds_not_rejected"):
            return rc
        failed_report = json.loads((out / "dataset_stats_report_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require({"min_records", "prompt_p95_bytes"} <= kinds, "dataset_stats_findings_not_reported"):
            return rc

    print("dataset_stats_report_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
