#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_token_accounting_audit.py."""

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
        "expected_tokens": 32,
        "expected_tokens_match": True,
        "prompt_bytes": 16,
        "elapsed_us": 30_000,
        "wall_elapsed_us": 32_000,
        "timeout_seconds": 1.0,
        "wall_timeout_pct": 3.2,
        "host_overhead_us": 2_000,
        "host_overhead_pct": 2_000 * 100.0 / 30_000,
        "tok_per_s": 32 * 1_000_000.0 / 30_000,
        "wall_tok_per_s": 32 * 1_000_000.0 / 32_000,
        "prompt_bytes_per_s": 16 * 1_000_000.0 / 30_000,
        "wall_prompt_bytes_per_s": 16 * 1_000_000.0 / 32_000,
        "us_per_token": 30_000 / 32,
        "wall_us_per_token": 32_000 / 32,
        "tokens_per_prompt_byte": 2.0,
        "memory_bytes": 1024,
        "memory_bytes_per_token": 32.0,
    }
    value.update(overrides)
    return value


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_token_accounting_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_token_accounting_audit_latest",
            "--min-rows",
            "1",
            "--require-expected-tokens",
            "--require-expected-tokens-match",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-token-accounting-") as tmp:
        root = Path(tmp)
        passing = root / "qemu_prompt_bench_pass.json"
        passing.write_text(json.dumps({"benchmarks": [row()]}), encoding="utf-8")
        pass_dir = root / "pass"
        completed = run_audit(passing, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = pass_dir / "qemu_token_accounting_audit_latest.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_token_accounting_pass_status"):
            return rc
        if rc := require(report["summary"]["checks"] >= 12, "missing_token_accounting_checks"):
            return rc
        if rc := require(
            "No token accounting findings." in (pass_dir / "qemu_token_accounting_audit_latest.md").read_text(encoding="utf-8"),
            "missing_token_accounting_markdown",
        ):
            return rc
        junit_root = ET.parse(pass_dir / "qemu_token_accounting_audit_latest_junit.xml").getroot()
        if rc := require(junit_root.attrib["failures"] == "0", "unexpected_token_accounting_junit_failure"):
            return rc

        failing = root / "qemu_prompt_bench_fail.json"
        failing.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        row(
                            expected_tokens=31,
                            expected_tokens_match=True,
                            wall_tok_per_s=1.0,
                            wall_us_per_token=1.0,
                            host_overhead_us=1.0,
                            host_overhead_pct=1.0,
                            wall_timeout_pct=1.0,
                            prompt_bytes_per_s=1.0,
                            wall_prompt_bytes_per_s=1.0,
                            memory_bytes_per_token=1.0,
                        )
                    ]
                }
            ),
            encoding="utf-8",
        )
        fail_dir = root / "fail"
        failed = run_audit(failing, fail_dir)
        if rc := require(failed.returncode == 1, "token_accounting_bad_artifact_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "qemu_token_accounting_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require(
            {"metric_drift", "expected_tokens_match_drift", "expected_tokens_mismatch"} <= kinds,
            "token_accounting_findings_not_reported",
        ):
            return rc
        metrics = {finding["metric"] for finding in fail_report["findings"]}
        if rc := require(
            {"host_overhead_us", "host_overhead_pct", "wall_timeout_pct", "prompt_bytes_per_s", "wall_prompt_bytes_per_s"} <= metrics,
            "host_overhead_findings_not_reported",
        ):
            return rc

        recorded_mismatch = root / "qemu_prompt_bench_recorded_mismatch.json"
        recorded_mismatch.write_text(
            json.dumps({"benchmarks": [row(expected_tokens=31, expected_tokens_match=False)]}),
            encoding="utf-8",
        )
        mismatch_dir = root / "recorded-mismatch"
        mismatch = run_audit(recorded_mismatch, mismatch_dir)
        if rc := require(mismatch.returncode == 1, "token_accounting_recorded_mismatch_not_rejected"):
            return rc
        mismatch_report = json.loads((mismatch_dir / "qemu_token_accounting_audit_latest.json").read_text(encoding="utf-8"))
        mismatch_kinds = {finding["kind"] for finding in mismatch_report["findings"]}
        if rc := require(mismatch_kinds == {"expected_tokens_mismatch"}, "token_accounting_recorded_mismatch_findings_drift"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
