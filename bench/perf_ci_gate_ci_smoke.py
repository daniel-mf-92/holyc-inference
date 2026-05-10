#!/usr/bin/env python3
"""Stdlib-only smoke test for perf_ci_gate.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_dashboard(path: Path, *, status: str = "pass", findings: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": status,
                "summary": {"rows": 2, "findings": findings},
            }
        ),
        encoding="utf-8",
    )
    path.with_suffix(".md").write_text("# dashboard\n", encoding="utf-8")
    path.with_suffix(".csv").write_text("name,status\nsmoke,pass\n", encoding="utf-8")
    path.with_name(f"{path.with_suffix('').name}_junit.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuite name="smoke" tests="1" failures="0" />',
        encoding="utf-8",
    )


def run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "bench" / "perf_ci_gate.py"), *args]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-ci-gate-") as tmp:
        tmp_path = Path(tmp)
        dashboard = tmp_path / "perf_regression_latest.json"
        output_dir = tmp_path / "out"
        write_dashboard(dashboard)

        completed = run_gate(str(dashboard), "--output-dir", str(output_dir), "--output-stem", "gate")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((output_dir / "gate.json").read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print("unexpected_pass_gate_failure=true", file=sys.stderr)
            return 1
        if report["summary"]["sidecars_checked"] != 3:
            print("missing_sidecar_checks=true", file=sys.stderr)
            return 1
        if "No perf CI gate findings." not in (output_dir / "gate.md").read_text(encoding="utf-8"):
            print("missing_markdown_success=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(output_dir / "gate_junit.xml").getroot()
        if junit_root.attrib.get("name") != "holyc_perf_ci_gate":
            print("missing_junit_suite=true", file=sys.stderr)
            return 1

        failed = tmp_path / "perf_slo_audit_latest.json"
        write_dashboard(failed, status="fail", findings=1)
        failed_completed = run_gate(str(failed), "--output-dir", str(output_dir), "--output-stem", "gate_fail")
        if failed_completed.returncode != 1:
            print("expected_failed_gate=true", file=sys.stderr)
            return 1
        fail_report = json.loads((output_dir / "gate_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if {"dashboard_status", "dashboard_findings"} - kinds:
            print("missing_expected_findings=true", file=sys.stderr)
            return 1
        findings_csv = (output_dir / "gate_fail_findings.csv").read_text(encoding="utf-8")
        if "dashboard_status" not in findings_csv:
            print("missing_findings_csv=true", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
