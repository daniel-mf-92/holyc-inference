#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_launch_order_audit.py."""

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


def row(*, phase: str, launch_index: int, timestamp: str) -> dict[str, object]:
    return {
        "phase": phase,
        "launch_index": launch_index,
        "prompt": "smoke-short",
        "iteration": 1,
        "timestamp": timestamp,
        "wall_elapsed_us": 100000,
        "exit_class": "ok",
    }


def artifact(*, broken: bool = False) -> dict[str, object]:
    if broken:
        warmups = [row(phase="warmup", launch_index=2, timestamp="2026-05-01T00:00:02Z")]
        measured = [row(phase="measured", launch_index=2, timestamp="2026-05-01T00:00:01Z")]
    else:
        warmups = [row(phase="warmup", launch_index=1, timestamp="2026-05-01T00:00:01Z")]
        measured = [
            row(phase="measured", launch_index=2, timestamp="2026-05-01T00:00:02Z"),
            row(phase="measured", launch_index=3, timestamp="2026-05-01T00:00:03Z"),
        ]
    return {
        "planned_total_launches": 3,
        "planned_warmup_launches": 1,
        "planned_measured_launches": 2,
        "warmups": warmups,
        "benchmarks": measured,
    }


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_launch_order_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_launch_order_audit_latest",
            "--timestamp-tolerance-us",
            "0",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-launch-order-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_launch_order_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_launch_order_pass_status"):
            return rc
        if rc := require(report["summary"]["launch_rows"] == 3, "missing_launch_rows"):
            return rc
        for suffix in (".md", ".csv", "_rows.csv", "_findings.csv", "_junit.xml"):
            if rc := require((pass_dir / f"qemu_launch_order_audit_latest{suffix}").exists(), f"missing_launch_order{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(broken=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "launch_order_drift_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_launch_order_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"duplicate_launch_index", "launch_index_start", "planned_total_drift", "warmup_after_measured"} <= kinds, "launch_order_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
