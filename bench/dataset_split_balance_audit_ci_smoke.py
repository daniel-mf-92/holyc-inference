#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset_split_balance_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
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
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-split-balance-ci-") as tmp:
        out = Path(tmp) / "out"
        passed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_split_balance_audit.py"),
                "--input",
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "dataset_split_balance_audit_smoke",
                "--min-records",
                "3",
                "--require-split",
                "validation",
                "--require-dataset-split",
                "arc-smoke:validation",
                "--max-largest-split-pct",
                "100",
            ]
        )
        if passed.returncode != 0:
            return passed.returncode

        report = json.loads((out / "dataset_split_balance_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_split_balance_status"):
            return rc
        if rc := require(report["summary"]["records"] == 3, "unexpected_split_balance_record_count"):
            return rc
        if rc := require(report["summary"]["dataset_splits"] == 3, "unexpected_split_balance_bucket_count"):
            return rc
        if rc := require("pct_of_dataset" in (out / "dataset_split_balance_audit_smoke.csv").read_text(encoding="utf-8"), "missing_split_csv"):
            return rc
        if rc := require("largest_split_pct" in (out / "dataset_split_balance_audit_smoke_datasets.csv").read_text(encoding="utf-8"), "missing_dataset_csv"):
            return rc
        if rc := require("No dataset split-balance findings." in (out / "dataset_split_balance_audit_smoke.md").read_text(encoding="utf-8"), "missing_split_balance_markdown"):
            return rc
        junit = ET.parse(out / "dataset_split_balance_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_dataset_split_balance_audit", "missing_split_balance_junit"):
            return rc

        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_split_balance_audit.py"),
                "--input",
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "dataset_split_balance_audit_failing",
                "--require-split",
                "test",
                "--min-splits-per-dataset",
                "2",
                "--max-largest-split-pct",
                "50",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "dataset_split_balance_bad_thresholds_not_rejected"):
            return rc
        failed_report = json.loads((out / "dataset_split_balance_audit_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require(
            {"required_split_missing", "min_splits_per_dataset", "largest_split_pct"} <= kinds,
            "dataset_split_balance_findings_not_reported",
        ):
            return rc

    print("dataset_split_balance_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
