#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_phase_sequence_audit.py."""

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


def row(prompt: str, phase: str, iteration: int, exit_class: str = "ok") -> dict[str, object]:
    return {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": phase,
        "iteration": iteration,
        "exit_class": exit_class,
    }


def passing_artifact() -> dict[str, object]:
    return {
        "warmups": [row("smoke-short", "warmup", 1), row("smoke-code", "warmup", 1)],
        "benchmarks": [
            row("smoke-short", "measured", 1),
            row("smoke-short", "measured", 2),
            row("smoke-code", "measured", 1),
            row("smoke-code", "measured", 2),
        ],
    }


def failing_artifact() -> dict[str, object]:
    return {
        "benchmarks": [
            row("smoke-short", "measured", 1, "timeout"),
            row("smoke-short", "warmup", 1),
            row("smoke-short", "measured", 1),
        ],
    }


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_phase_sequence_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_phase_sequence_audit_latest",
            "--min-rows",
            "3",
            "--min-warmups-per-group",
            "1",
            "--min-measured-per-group",
            "2",
            "--require-measured-ok",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-phase-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(passing_artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_phase_sequence_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_phase_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["groups"] == 2, "missing_phase_group_rollups"):
            return rc
        for suffix in (".md", ".csv", "_runs.csv", "_findings.csv", "_junit.xml"):
            if rc := require((pass_dir / f"qemu_phase_sequence_audit_latest{suffix}").exists(), f"missing_phase_sidecar_{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(failing_artifact()), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "phase_audit_bad_sequence_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_phase_sequence_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"warmup_after_measured", "duplicate_iteration", "measured_not_ok"}
        if rc := require(expected <= kinds, "phase_audit_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
