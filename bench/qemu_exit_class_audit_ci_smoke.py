#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_exit_class_audit.py."""

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


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
        "failure_reason": None,
        "tokens": 32,
        "elapsed_us": 200000,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_exit_class_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_exit_class_audit_latest",
            "--require-success-telemetry",
            "--require-failure-reason",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-exit-class-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("ok"),
                        row("timeout", returncode=-9, timed_out=True, exit_class="timeout", failure_reason="timeout", tokens=None, elapsed_us=0),
                    ]
                }
            ),
            encoding="utf-8",
        )
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_exit_class_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_exit_class_pass_status"):
            return rc
        if rc := require(report["summary"]["exit_class_timeout_rows"] == 1, "missing_exit_class_summary"):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_exit_class_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_exit_class_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("bad-class", returncode=0, timed_out=False, exit_class="timeout", failure_reason="timeout"),
                        row("bad-reason", returncode=127, exit_class="launch_error", failure_reason="launch_error"),
                        row("missing-telemetry", tokens=0, elapsed_us=0),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "bad_exit_class_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_exit_class_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require(
            {"exit_class_mismatch", "failure_reason_mismatch", "ok_row_missing_tokens", "ok_row_missing_elapsed_us"} <= kinds,
            "exit_class_findings_not_reported",
        ):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
