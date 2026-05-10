#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_runtime_budget_audit.py."""

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


def artifact(*, slow: bool = False, missing: bool = False, failed: bool = False) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    durations = [100_000, 120_000, 40_000]
    if slow:
        durations = [900_000, 700_000, 500_000]
    phases = ["measured", "measured", "warmup"]
    for index, (phase, wall_elapsed_us) in enumerate(zip(phases, durations, strict=True), 1):
        row: dict[str, object] = {
            "profile": "ci-airgap-smoke",
            "model": "synthetic-smoke",
            "quantization": "Q4_0",
            "prompt": f"smoke-{index}",
            "phase": phase,
            "exit_class": "ok",
            "returncode": 0,
            "timed_out": False,
            "tokens": 16,
            "wall_elapsed_us": wall_elapsed_us,
        }
        if missing and index == 1:
            row.pop("wall_elapsed_us")
        if failed and index == 2:
            row["exit_class"] = "timeout"
            row["timed_out"] = True
            row["returncode"] = 124
        rows.append(row)
    return {"status": "pass", "benchmarks": rows}


def run_audit(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_runtime_budget_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_runtime_budget_audit_latest",
            "--min-rows",
            "3",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-runtime-budget-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(
            passing,
            pass_dir,
            "--max-total-wall-seconds",
            "0.30",
            "--max-measured-wall-seconds",
            "0.25",
            "--max-warmup-wall-seconds",
            "0.05",
            "--max-group-wall-seconds",
            "0.30",
            "--max-row-wall-seconds",
            "0.13",
            "--fail-on-failures",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_runtime_budget_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "runtime_budget_pass_status_not_pass"):
            return rc
        if rc := require(report["summary"]["measured_wall_seconds"] == 0.22, "runtime_budget_measured_seconds_wrong"):
            return rc
        if rc := require((pass_dir / "qemu_runtime_budget_audit_latest.csv").exists(), "missing_runtime_budget_csv"):
            return rc
        if rc := require((pass_dir / "qemu_runtime_budget_audit_latest.md").exists(), "missing_runtime_budget_markdown"):
            return rc
        if rc := require((pass_dir / "qemu_runtime_budget_audit_latest_junit.xml").exists(), "missing_runtime_budget_junit"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(slow=True, missing=True, failed=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed_run = run_audit(
            failing,
            fail_dir,
            "--max-total-wall-seconds",
            "1.00",
            "--max-measured-wall-seconds",
            "0.90",
            "--max-warmup-wall-seconds",
            "0.10",
            "--max-group-wall-seconds",
            "1.00",
            "--max-row-wall-seconds",
            "0.80",
            "--fail-on-failures",
        )
        if rc := require(failed_run.returncode == 1, "runtime_budget_failures_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_runtime_budget_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"max_group_wall_seconds", "max_warmup_wall_seconds", "missing_wall_elapsed_us", "failed_rows"}
        if rc := require(expected <= kinds, "runtime_budget_findings_missing"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
