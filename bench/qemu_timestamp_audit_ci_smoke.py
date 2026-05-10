#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU timestamp audits."""

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


def write_artifact(path: Path, generated_at: str, timestamps: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": "pass",
                "benchmarks": [
                    {"launch_index": index + 1, "phase": "measured", "prompt": f"p{index}", "timestamp": timestamp}
                    for index, timestamp in enumerate(timestamps)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-timestamp-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
        write_artifact(
            passing,
            "2026-04-29T23:01:49Z",
            ["2026-04-29T23:01:48Z", "2026-04-29T23:01:49Z"],
        )
        output_dir = tmp_path / "out"
        completed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_timestamp_audit.py"),
                str(tmp_path),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timestamp_audit_smoke",
                "--require-rows",
            ]
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / "qemu_timestamp_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_timestamp_status"):
            return rc
        if rc := require(report["summary"]["parsed_row_timestamps"] == 2, "unexpected_row_timestamp_count"):
            return rc
        if rc := require(
            "QEMU Timestamp Audit" in (output_dir / "qemu_timestamp_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_markdown",
        ):
            return rc
        if rc := require(
            "max_row_skew_seconds" in (output_dir / "qemu_timestamp_audit_smoke.csv").read_text(encoding="utf-8"),
            "missing_csv",
        ):
            return rc
        junit = ET.parse(output_dir / "qemu_timestamp_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_qemu_timestamp_audit", "missing_junit"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        failing_dir = tmp_path / "failing"
        failing_dir.mkdir()
        write_artifact(
            failing_dir / "qemu_prompt_bench_20260429T230149Z.json",
            "2026-04-29T23:01:48Z",
            ["2026-04-29T23:01:49Z", "2026-04-29T23:01:47Z"],
        )
        failed = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_timestamp_audit.py"),
                str(failing_dir),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timestamp_audit_failing",
                "--max-row-after-generated-at-seconds",
                "0",
                "--require-rows",
            ],
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_timestamp_failure"):
            return rc
        failed_report = json.loads((output_dir / "qemu_timestamp_audit_failing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_report["findings"]}
        if rc := require("filename_stamp_mismatch" in kinds, "missing_filename_stamp_finding"):
            return rc
        if rc := require("row_timestamp_regressed" in kinds, "missing_regression_finding"):
            return rc
        if rc := require("row_after_generated_at" in kinds, "missing_skew_finding"):
            return rc

        stale_dir = tmp_path / "stale"
        stale_dir.mkdir()
        write_artifact(
            stale_dir / "qemu_prompt_bench_20260429T230149Z.json",
            "2026-04-29T23:01:49Z",
            ["2026-04-29T21:01:49Z"],
        )
        stale = run_command(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_timestamp_audit.py"),
                str(stale_dir),
                "--output-dir",
                str(output_dir),
                "--output-stem",
                "qemu_timestamp_audit_stale",
                "--max-row-before-generated-at-seconds",
                "60",
                "--require-rows",
            ],
            expected_failure=True,
        )
        if rc := require(stale.returncode == 1, "expected_stale_row_failure"):
            return rc
        stale_report = json.loads((output_dir / "qemu_timestamp_audit_stale.json").read_text(encoding="utf-8"))
        stale_kinds = {finding["kind"] for finding in stale_report["findings"]}
        if rc := require("row_before_generated_at" in stale_kinds, "missing_stale_row_finding"):
            return rc

    print("qemu_timestamp_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
