#!/usr/bin/env python3
"""Stdlib-only smoke test for perf_regression.py."""

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


def bench_row(commit: str, timestamp: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": timestamp,
        "commit": commit,
        "benchmark": "qemu_prompt",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "ci-short",
        "tok_per_s": 100.0,
        "wall_tok_per_s": 95.0,
        "elapsed_us": 1_000_000,
        "wall_elapsed_us": 1_052_632,
        "us_per_token": 10_000.0,
        "wall_us_per_token": 10_526.315789473685,
        "tokens": 100,
        "prompt_bytes": 50,
        "prompt_bytes_per_s": 500.0,
        "wall_prompt_bytes_per_s": 475.0,
        "tokens_per_prompt_byte": 2.0,
        "serial_output_bytes": 2048,
        "memory_bytes": 1_000_000,
        "host_child_peak_rss_bytes": 2_200_000,
        "host_child_cpu_us": 400_000,
        "host_child_cpu_pct": 55.0,
        "host_child_tok_per_cpu_s": 80.0,
        "ttft_us": 50_000,
        "host_overhead_pct": 5.0,
        "environment_sha256": "ci-env",
        "host_platform": "macOS-ci",
        "host_machine": "arm64",
        "qemu_version": "QEMU emulator version synthetic",
        "qemu_bin": "qemu-system-x86_64",
    }
    row.update(overrides)
    return row


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def run_regression(input_path: Path, output_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--baseline-commit",
            "ci-base",
            "--candidate-commit",
            "ci-head",
            "--fail-on-regression",
            *extra,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-regression-") as tmp:
        root = Path(tmp)

        passing_input = root / "pass.jsonl"
        write_jsonl(
            passing_input,
            [
                bench_row("ci-base", "2026-05-01T00:00:00Z"),
                bench_row(
                    "ci-head",
                    "2026-05-01T00:05:00Z",
                    tok_per_s=101.0,
                    wall_tok_per_s=96.0,
                    tokens=101,
                    host_child_peak_rss_bytes=2_100_000,
                    host_child_cpu_us=390_000,
                    host_child_cpu_pct=54.0,
                    host_child_tok_per_cpu_s=82.0,
                    ttft_us=49_000,
                    host_overhead_pct=4.0,
                ),
            ],
        )
        pass_dir = root / "pass"
        completed = run_regression(
            passing_input,
            pass_dir,
            "--tok-regression-pct",
            "5",
            "--wall-tok-regression-pct",
            "5",
            "--memory-regression-pct",
            "10",
            "--require-tok-per-s",
            "--require-wall-tok-per-s",
            "--require-memory",
            "--require-tokens",
            "--require-ttft-us",
            "--require-environment-sha256",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        pass_report = json.loads((pass_dir / "perf_regression_latest.json").read_text(encoding="utf-8"))
        if rc := require(pass_report["status"] == "pass", "unexpected_perf_regression_pass_status"):
            return rc
        if rc := require(pass_report["record_count"] == 2, "missing_perf_regression_records"):
            return rc
        if rc := require(len(pass_report["comparisons"]) == 1, "missing_perf_regression_comparison"):
            return rc
        if rc := require(
            "No regressions detected." in (pass_dir / "perf_regression_latest.md").read_text(encoding="utf-8"),
            "missing_perf_regression_markdown_success",
        ):
            return rc
        junit_text = (pass_dir / "perf_regression_junit_latest.xml").read_text(encoding="utf-8")
        if rc := require('name="holyc_perf_regression"' in junit_text, "missing_perf_regression_junit_suite"):
            return rc
        if rc := require('failures="0"' in junit_text, "unexpected_perf_regression_junit_failure"):
            return rc
        if rc := require((pass_dir / "perf_regression_comparisons_latest.csv").exists(), "missing_comparisons_csv"):
            return rc

        failing_input = root / "fail.jsonl"
        write_jsonl(
            failing_input,
            [
                bench_row("ci-base", "2026-05-01T00:00:00Z"),
                bench_row(
                    "ci-head",
                    "2026-05-01T00:05:00Z",
                    tok_per_s=80.0,
                    wall_tok_per_s=70.0,
                    elapsed_us=1_350_000,
                    wall_elapsed_us=1_500_000,
                    us_per_token=12_500.0,
                    wall_us_per_token=14_285.714285714286,
                    memory_bytes=1_250_000,
                    host_child_peak_rss_bytes=2_800_000,
                    ttft_us=75_000,
                ),
            ],
        )
        fail_dir = root / "fail"
        failed = run_regression(
            failing_input,
            fail_dir,
            "--tok-regression-pct",
            "5",
            "--wall-tok-regression-pct",
            "5",
            "--memory-regression-pct",
            "10",
            "--host-child-peak-rss-regression-pct",
            "10",
            "--ttft-regression-pct",
            "10",
        )
        if rc := require(failed.returncode == 1, "bad_perf_regression_not_rejected"):
            return rc
        fail_report = json.loads((fail_dir / "perf_regression_latest.json").read_text(encoding="utf-8"))
        metrics = {regression["metric"] for regression in fail_report["regressions"]}
        if rc := require(
            {"tok_per_s", "wall_tok_per_s", "memory_bytes", "host_child_peak_rss_bytes", "ttft_us"} <= metrics,
            "missing_expected_perf_regression_metrics",
        ):
            return rc
        regressions_csv = (fail_dir / "perf_regression_regressions_latest.csv").read_text(encoding="utf-8")
        if rc := require("tok_per_s" in regressions_csv, "missing_regressions_csv_metric"):
            return rc

        missing_telemetry_input = root / "missing_telemetry.jsonl"
        write_jsonl(
            missing_telemetry_input,
            [
                bench_row("ci-base", "2026-05-01T00:00:00Z"),
                bench_row(
                    "ci-head",
                    "2026-05-01T00:05:00Z",
                    environment_sha256="",
                    host_platform="",
                    host_machine="",
                    qemu_version="",
                    qemu_bin="",
                ),
            ],
        )
        coverage_dir = root / "coverage"
        coverage = run_regression(missing_telemetry_input, coverage_dir, "--require-environment-sha256")
        if rc := require(coverage.returncode == 1, "missing_environment_coverage_not_rejected"):
            return rc
        coverage_report = json.loads((coverage_dir / "perf_regression_latest.json").read_text(encoding="utf-8"))
        fields = {violation["field"] for violation in coverage_report["environment_coverage_violations"]}
        if rc := require("environment_sha256" in fields, "missing_environment_coverage_violation"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
