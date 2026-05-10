#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_memory_accounting_audit.py."""

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
        "tokens": 32,
        "memory_bytes": 1024,
        "memory_bytes_per_token": 32.0,
        "host_child_peak_rss_bytes": 4096,
        "host_rss_bytes_per_token": 128.0,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_memory_accounting_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_memory_accounting_audit_latest",
            "--min-rows",
            "1",
            "--require-memory-bytes",
            "--require-host-rss",
            "--require-guest-memory-within-host-rss",
            "--max-host-rss-over-guest-ratio",
            "4",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-memory-accounting-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((pass_dir / "qemu_memory_accounting_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_memory_accounting_pass_status"):
            return rc
        if rc := require(report["summary"]["checks"] >= 3, "missing_memory_accounting_checks"):
            return rc
        if rc := require(report["summary"]["host_rss_over_guest_ratio_max"] == 4.0, "missing_host_rss_ratio_summary"):
            return rc
        if rc := require(
            "No memory accounting findings." in (pass_dir / "qemu_memory_accounting_audit_latest.md").read_text(encoding="utf-8"),
            "missing_memory_accounting_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_memory_accounting_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_memory_accounting_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(memory_bytes_per_token=1.0, host_rss_bytes_per_token=1.0),
                        row(memory_bytes=8192, host_child_peak_rss_bytes=4096),
                        row(host_child_peak_rss_bytes=8192, host_rss_bytes_per_token=256.0),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "memory_accounting_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_memory_accounting_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require(
            {"metric_drift", "memory_bound_violation", "host_rss_over_guest_ratio"} <= kinds,
            "memory_accounting_findings_not_reported",
        ):
            return rc
        metrics = {finding["metric"] for finding in fail_report["findings"]}
        if rc := require(
            {"memory_bytes_per_token", "host_rss_bytes_per_token", "memory_bytes", "host_rss_over_guest_ratio"} <= metrics,
            "memory_metrics_not_reported",
        ):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
