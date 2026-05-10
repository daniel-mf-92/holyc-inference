#!/usr/bin/env python3
"""Stdlib-only smoke gate for offline eval workload estimates."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "bench" / "fixtures" / "eval_workload_estimate" / "smoke.jsonl"


def run_estimate(output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_workload_estimate.py"),
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "eval_workload_estimate_latest",
            "--tok-per-s",
            "100",
            "--qemu-launch-overhead-s",
            "0.25",
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-workload-") as tmp:
        tmp_path = Path(tmp)

        pass_dir = tmp_path / "pass"
        completed = run_estimate(
            pass_dir,
            "--max-record-scored-tokens",
            "4096",
            "--max-record-launches",
            "8",
            "--max-choices-per-record",
            "8",
            "--max-scored-tokens",
            "65536",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "eval_workload_estimate_latest.json").read_text(encoding="utf-8"))
        junit = (pass_dir / "eval_workload_estimate_latest_junit.xml").read_text(encoding="utf-8")
        checks = [
            require(report["status"] == "pass", "workload_pass_status=false"),
            require(report["summary"]["records"] == 2, "workload_pass_records!=2"),
            require(report["summary"]["choices"] == 8, "workload_pass_choices!=8"),
            require(report["summary"]["launches_est"] == 2, "workload_pass_launches!=2"),
            require(report["summary"]["scored_tokens_est"] == 157, "workload_pass_tokens!=157"),
            require('failures="0"' in junit, "workload_pass_junit_failures=true"),
            require(
                "scope,dataset,split,records,choices,prompt_bytes"
                in (pass_dir / "eval_workload_estimate_latest.csv").read_text(encoding="utf-8"),
                "workload_pass_missing_scope_csv=true",
            ),
            require(
                "source,row,dataset,split,record_id,choices"
                in (pass_dir / "eval_workload_estimate_latest_rows.csv").read_text(encoding="utf-8"),
                "workload_pass_missing_rows_csv=true",
            ),
            require(
                "Eval Workload Estimate"
                in (pass_dir / "eval_workload_estimate_latest.md").read_text(encoding="utf-8"),
                "workload_pass_missing_markdown=true",
            ),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_estimate(
            fail_dir,
            "--max-record-scored-tokens",
            "64",
            "--max-record-launches",
            "0",
            "--max-choices-per-record",
            "3",
            "--max-scored-tokens",
            "100",
            "--max-launches",
            "1",
        )
        if completed.returncode == 0:
            print("workload_failure_budget_not_rejected=true", file=sys.stderr)
            return 1

        fail_report = json.loads((fail_dir / "eval_workload_estimate_latest.json").read_text(encoding="utf-8"))
        findings = "\n".join(finding["kind"] for finding in fail_report["findings"])
        fail_junit = (fail_dir / "eval_workload_estimate_latest_junit.xml").read_text(encoding="utf-8")
        checks = [
            require(fail_report["status"] == "fail", "workload_fail_status=false"),
            require("choices_per_record_budget" in findings, "workload_fail_missing_choices=true"),
            require("record_scored_tokens_budget" in findings, "workload_fail_missing_record_tokens=true"),
            require("record_launch_budget" in findings, "workload_fail_missing_record_launches=true"),
            require("scored_tokens_budget" in findings, "workload_fail_missing_scope_tokens=true"),
            require("launch_budget" in findings, "workload_fail_missing_scope_launches=true"),
            require('failures="1"' in fail_junit, "workload_fail_junit_failures=false"),
            require(
                "record estimated scored tokens exceed budget"
                in (fail_dir / "eval_workload_estimate_latest_findings.csv").read_text(encoding="utf-8"),
                "workload_fail_missing_findings_csv=true",
            ),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
