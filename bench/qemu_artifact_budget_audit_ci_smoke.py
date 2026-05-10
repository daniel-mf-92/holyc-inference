#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU artifact budget audits."""

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


def write_artifact(path: Path, *, serial_bytes: int, stdout_tail: str = "", stderr_tail: str = "") -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-29T23:01:49Z",
                "benchmarks": [
                    {
                        "prompt": "smoke",
                        "phase": "measured",
                        "exit_class": "ok",
                        "serial_output_bytes": serial_bytes,
                        "stdout_tail": stdout_tail,
                        "stderr_tail": stderr_tail,
                        "failure_reason": "",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-artifact-budget-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
        write_artifact(passing, serial_bytes=256, stdout_tail="ok", stderr_tail="")
        output_dir = tmp_path / "out"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_artifact_budget_audit.py"),
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_artifact_budget_audit_smoke",
                "--max-file-bytes",
                "4096",
                "--max-serial-output-bytes",
                "1024",
                "--max-stdout-tail-bytes",
                "64",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / "qemu_artifact_budget_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_budget_status"):
            return rc
        if rc := require(report["summary"]["rows"] == 1, "unexpected_row_count"):
            return rc
        if rc := require(
            "QEMU Artifact Budget Audit"
            in (output_dir / "qemu_artifact_budget_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_markdown",
        ):
            return rc
        if rc := require(
            "max_stdout_tail_bytes"
            in (output_dir / "qemu_artifact_budget_audit_smoke.csv").read_text(encoding="utf-8"),
            "missing_csv",
        ):
            return rc
        junit = ET.parse(output_dir / "qemu_artifact_budget_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_qemu_artifact_budget_audit", "missing_junit"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failing_dir = tmp_path / "failing"
        failing_dir.mkdir()
        write_artifact(
            failing_dir / "qemu_prompt_bench_20260429T230149Z.json",
            serial_bytes=2048,
            stdout_tail="x" * 80,
            stderr_tail="",
        )
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_artifact_budget_audit.py"),
                str(failing_dir),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_artifact_budget_audit_failing",
                "--max-file-bytes",
                "4096",
                "--max-serial-output-bytes",
                "1024",
                "--max-stdout-tail-bytes",
                "64",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_budget_failure"):
            return rc
        failed_report = json.loads((output_dir / "qemu_artifact_budget_audit_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("serial_output_budget_exceeded" in kinds, "missing_serial_budget_finding"):
            return rc
        if rc := require("stdout_tail_budget_exceeded" in kinds, "missing_stdout_tail_finding"):
            return rc

    print("qemu_artifact_budget_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
