#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_iteration_coverage_audit.py."""

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


def artifact(*, gap: bool = False, duplicate: bool = False) -> dict[str, object]:
    measured_iterations = [1, 2]
    if gap:
        measured_iterations = [1, 3]
    if duplicate:
        measured_iterations = [1, 1]
    return {
        "status": "pass",
        "warmups": [
            {"prompt": "smoke-short", "phase": "warmup", "iteration": 1, "launch_index": 1, "exit_class": "ok"},
        ],
        "benchmarks": [
            {
                "prompt": "smoke-short",
                "phase": "measured",
                "iteration": iteration,
                "launch_index": index + 2,
                "exit_class": "ok",
            }
            for index, iteration in enumerate(measured_iterations)
        ],
    }


def run_audit(input_path: Path, output_dir: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_iteration_coverage_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_iteration_coverage_audit_latest",
            "--min-measured-iterations-per-prompt",
            "2",
            *(extra or []),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-iteration-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_iteration_coverage_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_iteration_audit_pass_status"):
            return rc
        if rc := require((pass_dir / "qemu_iteration_coverage_audit_latest.md").exists(), "missing_iteration_markdown"):
            return rc
        if rc := require((pass_dir / "qemu_iteration_coverage_audit_latest.csv").exists(), "missing_iteration_csv"):
            return rc
        if rc := require((pass_dir / "qemu_iteration_coverage_audit_latest_findings.csv").exists(), "missing_iteration_findings_csv"):
            return rc
        if rc := require((pass_dir / "qemu_iteration_coverage_audit_latest_junit.xml").exists(), "missing_iteration_junit"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(gap=True, duplicate=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "iteration_audit_gap_or_duplicate_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_iteration_coverage_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"duplicate_iteration"} <= kinds, "iteration_audit_duplicate_not_reported"):
            return rc

        gap_only = root / "qemu_prompt_bench_gap.json"
        gap_only.write_text(json.dumps(artifact(gap=True)), encoding="utf-8")
        gap_dir = root / "gap"
        gap_failed = run_audit(gap_only, gap_dir)
        if rc := require(gap_failed.returncode == 1, "iteration_audit_gap_not_rejected"):
            return rc
        gap_report = json.loads((gap_dir / "qemu_iteration_coverage_audit_latest.json").read_text(encoding="utf-8"))
        gap_kinds = {finding["kind"] for finding in gap_report["findings"]}
        if rc := require({"iteration_gap"} <= gap_kinds, "iteration_audit_gap_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
