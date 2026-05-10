#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_resource_coverage_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def artifact(*, missing_memory: bool = False) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "memory_bytes": 67_174_400,
        "memory_bytes_per_token": 2_099_200.0,
        "host_child_peak_rss_bytes": 458_752,
        "host_child_cpu_us": 67_610,
        "host_child_cpu_pct": 28.5,
        "host_child_tok_per_cpu_s": 473.3,
    }
    if missing_memory:
        row.pop("host_child_peak_rss_bytes")
    return {"status": "pass", "benchmarks": [row]}


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_resource_coverage_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_resource_coverage_audit_latest",
            "--min-rows",
            "1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-resource-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_resource_coverage_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_resource_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["rows_with_all_metrics"] == 1, "missing_resource_coverage_rollup"):
            return rc
        if rc := require((pass_dir / "qemu_resource_coverage_audit_latest.md").exists(), "missing_resource_markdown"):
            return rc
        if rc := require((pass_dir / "qemu_resource_coverage_audit_latest.csv").exists(), "missing_resource_csv"):
            return rc
        if rc := require(
            (pass_dir / "qemu_resource_coverage_audit_latest_junit.xml").exists(),
            "missing_resource_junit",
        ):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(missing_memory=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "resource_audit_missing_metric_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_resource_coverage_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require("missing_metric" in kinds, "resource_audit_missing_metric_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
