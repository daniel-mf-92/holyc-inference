#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_serial_accounting_audit.py."""

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


def row(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "stdout_bytes": 32,
        "stderr_bytes": 8,
        "serial_output_bytes": 40,
        "stdout_lines": 2,
        "stderr_lines": 1,
        "serial_output_lines": 3,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_serial_accounting_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_serial_accounting_audit_latest",
            "--min-rows",
            "1",
            "--require-metrics",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-serial-accounting-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_serial_accounting_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_serial_accounting_pass_status"):
            return rc
        if rc := require(report["summary"]["checks"] == 2, "missing_serial_accounting_checks"):
            return rc
        if rc := require(
            "No serial accounting findings." in (pass_dir / "qemu_serial_accounting_audit_latest.md").read_text(encoding="utf-8"),
            "missing_serial_accounting_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_serial_accounting_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_serial_accounting_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(serial_output_bytes=41),
                        row(serial_output_lines=2),
                        row(stdout_bytes=-1),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "serial_accounting_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_serial_accounting_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"metric_drift", "invalid_metric"} <= kinds, "serial_accounting_findings_not_reported"):
            return rc
        metrics = {finding["metric"] for finding in fail_report["findings"]}
        if rc := require({"serial_output_bytes", "serial_output_lines", "stdout_bytes"} <= metrics, "serial_metrics_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
