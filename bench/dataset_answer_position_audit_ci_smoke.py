#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset_answer_position_audit.py."""

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
    with tempfile.TemporaryDirectory(prefix="holyc-answer-position-ci-") as tmp:
        out = Path(tmp) / "out"
        passed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_answer_position_audit.py"),
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "answer_position_smoke",
                "--min-records",
                "3",
                "--max-dominant-answer-pct",
                "100",
            ]
        )
        if passed.returncode != 0:
            return passed.returncode

        report = json.loads((out / "answer_position_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_answer_position_status"):
            return rc
        if rc := require(report["summary"]["records"] == 3, "unexpected_answer_position_record_count"):
            return rc
        if rc := require("dominant_answer_pct" in (out / "answer_position_smoke.csv").read_text(encoding="utf-8"), "missing_answer_position_csv"):
            return rc
        if rc := require("No dataset answer-position findings." in (out / "answer_position_smoke.md").read_text(encoding="utf-8"), "missing_answer_position_markdown"):
            return rc
        junit = ET.parse(out / "answer_position_smoke_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_dataset_answer_position_audit", "missing_answer_position_junit"):
            return rc

        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "dataset_answer_position_audit.py"),
                str(SAMPLE),
                "--output-dir",
                str(out),
                "--output-stem",
                "answer_position_failing",
                "--min-records",
                "3",
                "--min-distinct-answer-positions",
                "2",
                "--max-dominant-answer-pct",
                "50",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "answer_position_bad_distribution_not_rejected"):
            return rc
        failed_report = json.loads((out / "answer_position_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require({"answer_position_coverage", "dominant_answer_position"} <= kinds, "answer_position_findings_not_reported"):
            return rc

    print("dataset_answer_position_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
