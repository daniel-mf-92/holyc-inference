#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_host_overhead_audit.py."""

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
        "phase": "measured",
        "iteration": 1,
        "exit_class": "ok",
        "elapsed_us": 100_000,
        "wall_elapsed_us": 125_000,
        "host_overhead_us": 25_000,
        "host_overhead_pct": 25.0,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_host_overhead_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_host_overhead_audit_latest",
            "--min-rows",
            "1",
            "--max-ok-host-overhead-pct",
            "40",
            "--fail-negative-host-overhead",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-host-overhead-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_host_overhead_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_host_overhead_pass_status"):
            return rc
        if rc := require(report["summary"]["max_host_overhead_pct"] == 25.0, "missing_host_overhead_summary"):
            return rc
        if rc := require(report["summary"]["median_host_overhead_pct"] == 25.0, "missing_host_overhead_median"):
            return rc
        if rc := require(report["summary"]["p95_host_overhead_pct"] == 25.0, "missing_host_overhead_p95"):
            return rc
        if rc := require(
            "No host overhead findings." in (pass_dir / "qemu_host_overhead_audit_latest.md").read_text(encoding="utf-8"),
            "missing_host_overhead_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_host_overhead_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_host_overhead_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(prompt="pct-drift", host_overhead_pct=1.0),
                        row(prompt="us-drift", host_overhead_us=10),
                        row(prompt="inverted", elapsed_us=200_000, wall_elapsed_us=100_000),
                        row(prompt="too-high", host_overhead_pct=45.0, wall_elapsed_us=145_000, host_overhead_us=45_000),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "host_overhead_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_host_overhead_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"host_overhead_pct_drift", "host_overhead_us_drift", "wall_elapsed_before_guest_elapsed", "ok_host_overhead_too_high"}
        if rc := require(expected <= kinds, "host_overhead_findings_not_reported"):
            return rc

        negative_allowed = root / "qemu_prompt_bench_negative_allowed.json"
        negative_allowed.write_text(json.dumps({"benchmarks": [row(elapsed_us=200_000, wall_elapsed_us=100_000, host_overhead_us=-100_000, host_overhead_pct=-50.0)]}), encoding="utf-8")
        allowed_dir = root / "negative_allowed"
        allowed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_host_overhead_audit.py"),
                str(negative_allowed),
                "--output-dir",
                str(allowed_dir),
                "--output-stem",
                "qemu_host_overhead_audit_latest",
                "--min-rows",
                "1",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc := require(allowed.returncode == 0, "negative_host_overhead_not_allowed_by_default"):
            sys.stdout.write(allowed.stdout)
            sys.stderr.write(allowed.stderr)
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
