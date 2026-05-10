#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_failure_reason_audit.py."""

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
        "returncode": 0,
        "timed_out": False,
        "failure_reason": None,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_failure_reason_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_failure_reason_audit_latest",
            "--min-rows",
            "1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-failure-reason-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps(
                {
                    "warmups": [row(phase="warmup")],
                    "benchmarks": [
                        row(),
                        row(exit_class="timeout", returncode=124, timed_out=True, failure_reason="timeout after 1.0s"),
                        row(exit_class="nonzero_exit", returncode=2, timed_out=False, failure_reason="qemu exited 2"),
                        row(exit_class="launch_error", returncode=None, timed_out=False, failure_reason="missing qemu binary"),
                    ],
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

        report = json.loads((pass_dir / "qemu_failure_reason_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_failure_reason_pass_status"):
            return rc
        if rc := require(report["summary"]["failure_rows"] == 3, "missing_failure_reason_failure_rows"):
            return rc
        if rc := require(
            "No failure-reason findings." in (pass_dir / "qemu_failure_reason_audit_latest.md").read_text(encoding="utf-8"),
            "missing_failure_reason_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_failure_reason_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_failure_reason_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(exit_class="ok", returncode=7, failure_reason="stale error"),
                        row(exit_class="timeout", returncode=0, timed_out=False, failure_reason=""),
                        row(exit_class="nonzero_exit", returncode=0, timed_out=True, failure_reason=""),
                        row(exit_class="mystery", timed_out="maybe"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "failure_reason_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_failure_reason_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"ok_nonzero_returncode", "timeout_without_timed_out", "invalid_exit_class"} <= kinds, "failure_reason_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
