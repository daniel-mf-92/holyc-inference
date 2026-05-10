#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_failure_audit.py."""

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


def row(prompt: str, exit_class: str = "ok", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": prompt,
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "iteration": 1,
        "exit_class": exit_class,
        "timed_out": exit_class == "timeout",
        "returncode": 0 if exit_class == "ok" else 1,
        "failure_reason": "" if exit_class == "ok" else exit_class,
        "tokens": 32 if exit_class == "ok" else None,
        "wall_elapsed_us": 100000 if exit_class == "ok" else None,
        "wall_tok_per_s": 320.0 if exit_class == "ok" else None,
    }
    payload.update(overrides)
    return payload


def run_audit(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_failure_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_failure_audit_latest",
            "--min-rows",
            "3",
            "--max-failure-pct",
            "50",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-failure-audit-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("smoke-short"),
                        row("smoke-code"),
                        row("smoke-timeout", "timeout"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((pass_dir / "qemu_failure_audit_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_failure_audit_pass_status"):
            return rc
        if rc := require(report["summary"]["failure_rows"] == 1, "missing_failure_row_count"):
            return rc
        for suffix in (".md", ".csv", "_rows.csv", "_findings.csv", "_junit.xml"):
            if rc := require((pass_dir / f"qemu_failure_audit_latest{suffix}").exists(), f"missing_failure_sidecar_{suffix}"):
                return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row("bad-timeout", "ok", timed_out=True),
                        row("bad-nonzero", "nonzero_exit", returncode=0),
                        row("bad-ok", "ok", tokens=0, failure_reason="unexpected"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "failure_audit_bad_rows_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_failure_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {"timeout_exit_class_mismatch", "nonzero_returncode_mismatch", "ok_has_failure_reason", "ok_missing_tokens"}
        if rc := require(expected <= kinds, "failure_audit_findings_not_reported"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
