#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side perf regression dashboards."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "bench" / "fixtures" / "perf_regression" / "ci_pass.jsonl"
RESULTS = ROOT / "bench" / "results"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = Path(tmp) / "dashboards"
        command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(RESULTS),
            "--input",
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = output_dir / "perf_regression_latest.json"
        markdown_path = output_dir / "perf_regression_latest.md"
        sample_violations_path = output_dir / "perf_regression_sample_violations_latest.csv"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_status={report['status']}", file=sys.stderr)
            return 1
        if report["regressions"]:
            print("unexpected_regressions=true", file=sys.stderr)
            return 1
        if report["sample_violations"]:
            print("unexpected_sample_violations=true", file=sys.stderr)
            return 1
        if "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short" not in report["summaries"]:
            print("missing_ci_fixture_summary=true", file=sys.stderr)
            return 1
        if "Perf Regression Dashboard" not in markdown_path.read_text(encoding="utf-8"):
            print("missing_markdown_dashboard=true", file=sys.stderr)
            return 1
        if "key,commit,records,minimum_records" not in sample_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_sample_violations_csv=true", file=sys.stderr)
            return 1

        audit_output = tmp_path / "airgap_audit.json"
        audit_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(RESULTS),
            "--output",
            str(audit_output),
        ]
        completed = subprocess.run(
            audit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        audit_report = json.loads(audit_output.read_text(encoding="utf-8"))
        if audit_report["status"] != "pass":
            print(f"unexpected_airgap_status={audit_report['status']}", file=sys.stderr)
            return 1
        if audit_report["commands_checked"] < 1:
            print("missing_qemu_command_audit=true", file=sys.stderr)
            return 1

        unsafe_fixture = tmp_path / "unsafe_qemu.json"
        unsafe_fixture.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "benchmark": "qemu_prompt",
                            "command": [
                                "qemu-system-x86_64",
                                "-drive",
                                "file=/tmp/TempleOS.img,format=raw,if=ide",
                                "-device",
                                "e1000",
                            ],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(unsafe_fixture),
            "--output",
            str(tmp_path / "unsafe_airgap_audit.json"),
        ]
        completed = subprocess.run(
            unsafe_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unsafe_qemu_command_not_rejected=true", file=sys.stderr)
            return 1

    print("perf_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
