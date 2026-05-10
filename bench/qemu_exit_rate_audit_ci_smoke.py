#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_exit_rate_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def row(prompt: str = "smoke-short", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "returncode": 0,
        "timed_out": False,
        "exit_class": "ok",
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmarks": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-exit-rate-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        passing = tmp_path / "passing" / "qemu_prompt_bench_latest.json"
        write_artifact(passing, [row("a"), row("b")])
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_exit_rate_audit.py"),
                str(passing.parent),
                "--min-rows",
                "2",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_exit_rate_audit_smoke",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode
        report = json.loads((output_dir / "qemu_exit_rate_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_exit_rate_pass_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 2, "unexpected_exit_rate_row_count"):
            return rc
        if rc := require("No exit-rate findings." in (output_dir / "qemu_exit_rate_audit_smoke.md").read_text(encoding="utf-8"), "missing_exit_rate_markdown"):
            return rc
        junit = ET.parse(output_dir / "qemu_exit_rate_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_qemu_exit_rate_audit", "missing_exit_rate_junit"):
            return rc

        failing = tmp_path / "failing" / "qemu_prompt_bench_latest.json"
        write_artifact(
            failing,
            [
                row("ok"),
                row("timeout", returncode=124, timed_out=True, exit_class="timeout"),
                row("nonzero", returncode=2, exit_class="nonzero_exit"),
                row("launch", returncode=None, exit_class="launch_error"),
            ],
        )
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_exit_rate_audit.py"),
                str(failing.parent),
                "--max-failure-pct",
                "10",
                "--max-timeout-pct",
                "10",
                "--max-nonzero-exit-pct",
                "10",
                "--max-launch-error-pct",
                "10",
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_exit_rate_audit_failing",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "exit_rate_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((output_dir / "qemu_exit_rate_audit_failing.json").read_text(encoding="utf-8"))
        expected = {"max_failure_pct", "max_timeout_pct", "max_nonzero_exit_pct", "max_launch_error_pct"}
        if rc := require(expected <= {finding["kind"] for finding in fail_report["findings"]}, "exit_rate_findings_not_reported"):
            return rc

    print("qemu_exit_rate_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
