#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_warmup_isolation_audit.py."""

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


def row(*, phase: str, launch_index: int, tokens: int) -> dict[str, object]:
    return {
        "prompt": "smoke-short",
        "phase": phase,
        "launch_index": launch_index,
        "tokens": tokens,
        "exit_class": "ok",
    }


def artifact(*, leaking: bool = False) -> dict[str, object]:
    warmups = [row(phase="warmup", launch_index=1, tokens=8)]
    measured = [row(phase="measured", launch_index=2, tokens=16), row(phase="measured", launch_index=3, tokens=16)]
    if leaking:
        measured[0] = row(phase="warmup", launch_index=1, tokens=16)
    return {
        "planned_warmup_launches": 1,
        "planned_measured_launches": 2,
        "warmups": warmups,
        "benchmarks": measured,
        "suite_summary": {"total_tokens": 40 if leaking else 32},
        "phase_summaries": {"warmup": {"runs": 1}, "measured": {"runs": 2}},
    }


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_warmup_isolation_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_warmup_isolation_audit_latest",
            "--min-warmup-rows",
            "1",
            "--min-measured-rows",
            "2",
            "--require-phase-summaries",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-warmup-isolation-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps(artifact()), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_warmup_isolation_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_warmup_isolation_pass_status"):
            return rc
        if rc := require(report["summary"]["warmup_rows"] == 1, "missing_warmup_rows"):
            return rc
        if rc := require(report["summary"]["measured_tokens_total"] == 32, "missing_measured_token_rollup"):
            return rc
        for suffix in (".md", ".csv", "_findings.csv", "_junit.xml"):
            if rc := require((pass_dir / f"qemu_warmup_isolation_audit_latest{suffix}").exists(), f"missing_warmup_isolation{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(json.dumps(artifact(leaking=True)), encoding="utf-8")
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "warmup_isolation_leak_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_warmup_isolation_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"measured_phase_drift", "launch_index_overlap", "suite_summary_warmup_leak"} <= kinds, "warmup_isolation_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
