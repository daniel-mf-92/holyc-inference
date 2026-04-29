#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side build benchmark comparisons."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def qemu_report(
    *,
    commit: str,
    tok_per_s_values: list[float],
    wall_tok_per_s_values: list[float],
    ttft_us_values: list[int],
    memory_bytes: int,
    host_child_peak_rss_bytes: int,
    host_child_tok_per_cpu_s_values: list[float],
    serial_output_bytes: int,
    command_sha256: str = "ci-command",
    prompt_suite_sha256: str = "ci-suite",
    environment: dict[str, Any] | None = None,
    prompt: str = "ci-short",
) -> dict[str, Any]:
    environment = environment or {
        "platform": "ci-platform",
        "machine": "ci-machine",
        "python": "3.14",
        "qemu_bin": "synthetic-qemu",
        "qemu_version": "synthetic-qemu 1.0",
    }
    rows: list[dict[str, Any]] = []
    for index, tok_per_s in enumerate(tok_per_s_values):
        rows.append(
            {
                "commit": commit,
                "benchmark": "qemu_prompt",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "prompt": prompt,
                "tokens": 32,
                "elapsed_us": int(32_000_000 / tok_per_s),
                "tok_per_s": tok_per_s,
                "wall_tok_per_s": wall_tok_per_s_values[index],
                "ttft_us": ttft_us_values[index],
                "memory_bytes": memory_bytes,
                "host_child_cpu_us": 395000,
                "host_child_cpu_pct": 87.5,
                "host_child_tok_per_cpu_s": host_child_tok_per_cpu_s_values[index],
                "host_child_peak_rss_bytes": host_child_peak_rss_bytes,
                "serial_output_bytes": serial_output_bytes,
                "command_sha256": command_sha256,
                "returncode": 0,
                "timed_out": False,
            }
        )
    return {
        "generated_at": "2026-04-29T04:00:00Z",
        "status": "pass",
        "environment": environment,
        "prompt_suite": {"suite_sha256": prompt_suite_sha256, "prompt_count": 1},
        "benchmarks": rows,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_pass_case(tmp_path: Path) -> None:
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "head.json"
    output_dir = tmp_path / "pass_dashboard"
    write_json(
        baseline,
        qemu_report(
            commit="ci-base",
            tok_per_s_values=[100.0, 102.0],
            wall_tok_per_s_values=[94.0, 96.0],
            ttft_us_values=[50000, 49000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_200_000,
            host_child_tok_per_cpu_s_values=[80.0, 82.0],
            serial_output_bytes=4096,
        ),
    )
    write_json(
        candidate,
        qemu_report(
            commit="ci-head",
            tok_per_s_values=[101.0, 103.0],
            wall_tok_per_s_values=[95.0, 97.0],
            ttft_us_values=[49500, 48500],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_100_000,
            host_child_tok_per_cpu_s_values=[81.0, 83.0],
            serial_output_bytes=4096,
        ),
    )

    completed = run_command(
        [
            sys.executable,
            str(ROOT / "bench" / "build_compare.py"),
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--baseline",
            "base",
            "--output-dir",
            str(output_dir),
            "--max-tok-regression-pct",
            "5",
            "--max-p05-tok-regression-pct",
            "5",
            "--max-wall-tok-regression-pct",
            "5",
            "--max-p05-wall-tok-regression-pct",
            "5",
            "--max-ttft-growth-pct",
            "10",
            "--max-host-child-tok-per-cpu-s-regression-pct",
            "5",
            "--max-host-child-rss-growth-pct",
            "10",
            "--max-serial-output-growth-pct",
            "10",
            "--max-memory-growth-pct",
            "10",
            "--min-ok-runs-per-build",
            "2",
            "--fail-on-regression",
            "--fail-on-coverage",
            "--fail-on-comparison-coverage",
            "--fail-on-prompt-suite-drift",
            "--fail-on-command-drift",
            "--fail-on-environment-drift",
        ]
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    assert_true(completed.returncode == 0, "build_compare pass fixture failed")

    report = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    assert_true(report["status"] == "pass", "unexpected pass report status")
    assert_true(len(report["deltas"]) == 1, "missing build delta")
    assert_true(not report["regressions"], "unexpected regression")
    assert_true(not report["coverage_violations"], "unexpected coverage violation")
    assert_true(not report["comparison_coverage_violations"], "unexpected comparison coverage violation")
    assert_true(not report["prompt_suite_drift"], "unexpected prompt-suite drift")
    assert_true(not report["command_drift"], "unexpected command drift")
    assert_true(not report["environment_drift"], "unexpected environment drift")
    delta = report["deltas"][0]
    assert_true(delta["tok_per_s_delta_pct"] > 0, "missing positive tok/s delta")
    assert_true(delta["host_child_peak_rss_delta_pct"] < 0, "missing host RSS improvement")

    rows = list(csv.DictReader((output_dir / "build_compare_latest.csv").open(encoding="utf-8")))
    assert_true(rows[0]["candidate_build"] == "head", "missing CSV candidate build")
    assert_true(rows[0]["candidate_host_child_peak_rss_bytes"] == "2100000", "missing CSV host RSS")
    assert_true(rows[0]["baseline_environment_sha256"], "missing CSV baseline environment hash")
    comparison_coverage_csv = (
        output_dir / "build_compare_comparison_coverage_latest.csv"
    ).read_text(encoding="utf-8")
    assert_true("key,build,role,reason" in comparison_coverage_csv, "missing comparison coverage CSV")
    assert_true(
        "Build Benchmark Compare" in (output_dir / "build_compare_latest.md").read_text(encoding="utf-8"),
        "missing Markdown title",
    )
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    assert_true(junit_root.attrib.get("name") == "holyc_build_compare", "missing JUnit suite")
    assert_true(junit_root.attrib.get("failures") == "0", "unexpected JUnit failure")


def check_command_drift_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "drift_base.json"
    candidate = tmp_path / "drift_head.json"
    output_dir = tmp_path / "command_drift_dashboard"
    write_json(
        baseline,
        qemu_report(
            commit="ci-base",
            tok_per_s_values=[100.0, 101.0],
            wall_tok_per_s_values=[95.0, 96.0],
            ttft_us_values=[50000, 49000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0, 81.0],
            serial_output_bytes=4096,
            command_sha256="command-a",
        ),
    )
    write_json(
        candidate,
        qemu_report(
            commit="ci-head",
            tok_per_s_values=[100.0, 101.0],
            wall_tok_per_s_values=[95.0, 96.0],
            ttft_us_values=[50000, 49000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0, 81.0],
            serial_output_bytes=4096,
            command_sha256="command-b",
        ),
    )

    completed = run_command(
        [
            sys.executable,
            str(ROOT / "bench" / "build_compare.py"),
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--baseline",
            "base",
            "--output-dir",
            str(output_dir),
            "--fail-on-command-drift",
        ]
    )
    assert_true(completed.returncode == 1, "command drift gate did not fail")
    report = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    assert_true(report["command_drift"], "missing command drift report")
    drift_csv = (output_dir / "build_compare_command_drift_latest.csv").read_text(encoding="utf-8")
    assert_true("command-a,command-b" in drift_csv, "missing command drift CSV row")


def check_environment_drift_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "environment_base.json"
    candidate = tmp_path / "environment_head.json"
    output_dir = tmp_path / "environment_drift_dashboard"
    write_json(
        baseline,
        qemu_report(
            commit="ci-base",
            tok_per_s_values=[100.0, 101.0],
            wall_tok_per_s_values=[95.0, 96.0],
            ttft_us_values=[50000, 49000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0, 81.0],
            serial_output_bytes=4096,
            environment={
                "platform": "ci-platform-a",
                "machine": "ci-machine",
                "python": "3.14",
                "qemu_bin": "synthetic-qemu",
                "qemu_version": "synthetic-qemu 1.0",
            },
        ),
    )
    write_json(
        candidate,
        qemu_report(
            commit="ci-head",
            tok_per_s_values=[100.0, 101.0],
            wall_tok_per_s_values=[95.0, 96.0],
            ttft_us_values=[50000, 49000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0, 81.0],
            serial_output_bytes=4096,
            environment={
                "platform": "ci-platform-b",
                "machine": "ci-machine",
                "python": "3.14",
                "qemu_bin": "synthetic-qemu",
                "qemu_version": "synthetic-qemu 1.0",
            },
        ),
    )

    completed = run_command(
        [
            sys.executable,
            str(ROOT / "bench" / "build_compare.py"),
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--baseline",
            "base",
            "--output-dir",
            str(output_dir),
            "--fail-on-environment-drift",
        ]
    )
    assert_true(completed.returncode == 1, "environment drift gate did not fail")
    report = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    assert_true(report["environment_drift"], "missing environment drift report")
    drift_csv = (output_dir / "build_compare_environment_drift_latest.csv").read_text(encoding="utf-8")
    assert_true("baseline_environment_sha256" in drift_csv, "missing environment drift CSV header")
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    assert_true(junit_root.attrib.get("failures") == "1", "environment drift JUnit did not fail")


def check_coverage_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "coverage_base.json"
    candidate = tmp_path / "coverage_head.json"
    output_dir = tmp_path / "coverage_dashboard"
    write_json(
        baseline,
        qemu_report(
            commit="ci-base",
            tok_per_s_values=[100.0],
            wall_tok_per_s_values=[95.0],
            ttft_us_values=[50000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0],
            serial_output_bytes=4096,
        ),
    )
    write_json(
        candidate,
        qemu_report(
            commit="ci-head",
            tok_per_s_values=[100.0],
            wall_tok_per_s_values=[95.0],
            ttft_us_values=[50000],
            memory_bytes=1_000_000,
            host_child_peak_rss_bytes=2_000_000,
            host_child_tok_per_cpu_s_values=[80.0],
            serial_output_bytes=4096,
        ),
    )

    completed = run_command(
        [
            sys.executable,
            str(ROOT / "bench" / "build_compare.py"),
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--baseline",
            "base",
            "--output-dir",
            str(output_dir),
            "--min-ok-runs-per-build",
            "2",
            "--fail-on-coverage",
        ]
    )
    assert_true(completed.returncode == 1, "coverage gate did not fail")
    report = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    assert_true(len(report["coverage_violations"]) == 2, "missing baseline/candidate coverage violations")
    coverage_csv = (output_dir / "build_compare_coverage_violations_latest.csv").read_text(
        encoding="utf-8"
    )
    assert_true("minimum_ok_runs" in coverage_csv, "missing coverage CSV header")


def check_comparison_coverage_gate(tmp_path: Path) -> None:
    baseline = tmp_path / "comparison_base.json"
    candidate = tmp_path / "comparison_head.json"
    output_dir = tmp_path / "comparison_coverage_dashboard"
    write_json(
        baseline,
        {
            "generated_at": "2026-04-29T04:00:00Z",
            "status": "pass",
            "benchmarks": [
                *qemu_report(
                    commit="ci-base",
                    tok_per_s_values=[100.0, 101.0],
                    wall_tok_per_s_values=[95.0, 96.0],
                    ttft_us_values=[50000, 49000],
                    memory_bytes=1_000_000,
                    host_child_peak_rss_bytes=2_000_000,
                    host_child_tok_per_cpu_s_values=[80.0, 81.0],
                    serial_output_bytes=4096,
                    prompt="ci-short",
                )["benchmarks"],
                *qemu_report(
                    commit="ci-base",
                    tok_per_s_values=[98.0, 99.0],
                    wall_tok_per_s_values=[93.0, 94.0],
                    ttft_us_values=[51000, 50000],
                    memory_bytes=1_000_000,
                    host_child_peak_rss_bytes=2_000_000,
                    host_child_tok_per_cpu_s_values=[78.0, 79.0],
                    serial_output_bytes=4096,
                    prompt="baseline-only",
                )["benchmarks"],
            ],
        },
    )
    write_json(
        candidate,
        {
            "generated_at": "2026-04-29T04:00:00Z",
            "status": "pass",
            "benchmarks": [
                *qemu_report(
                    commit="ci-head",
                    tok_per_s_values=[100.0, 101.0],
                    wall_tok_per_s_values=[95.0, 96.0],
                    ttft_us_values=[50000, 49000],
                    memory_bytes=1_000_000,
                    host_child_peak_rss_bytes=2_000_000,
                    host_child_tok_per_cpu_s_values=[80.0, 81.0],
                    serial_output_bytes=4096,
                    prompt="ci-short",
                )["benchmarks"],
                *qemu_report(
                    commit="ci-head",
                    tok_per_s_values=[97.0, 98.0],
                    wall_tok_per_s_values=[92.0, 93.0],
                    ttft_us_values=[52000, 51000],
                    memory_bytes=1_000_000,
                    host_child_peak_rss_bytes=2_000_000,
                    host_child_tok_per_cpu_s_values=[77.0, 78.0],
                    serial_output_bytes=4096,
                    prompt="candidate-only",
                )["benchmarks"],
            ],
        },
    )

    completed = run_command(
        [
            sys.executable,
            str(ROOT / "bench" / "build_compare.py"),
            "--input",
            f"base={baseline}",
            "--input",
            f"head={candidate}",
            "--baseline",
            "base",
            "--output-dir",
            str(output_dir),
            "--fail-on-comparison-coverage",
        ]
    )
    assert_true(completed.returncode == 1, "comparison coverage gate did not fail")
    report = json.loads((output_dir / "build_compare_latest.json").read_text(encoding="utf-8"))
    violations = report["comparison_coverage_violations"]
    assert_true(len(violations) == 2, "missing comparison coverage violations")
    reasons = {violation["reason"] for violation in violations}
    assert_true(
        reasons == {"missing_candidate_key", "missing_baseline_key"},
        "wrong comparison coverage reasons",
    )
    comparison_csv = (output_dir / "build_compare_comparison_coverage_latest.csv").read_text(
        encoding="utf-8"
    )
    assert_true("missing_candidate_key" in comparison_csv, "missing comparison coverage CSV row")
    assert_true("missing_baseline_key" in comparison_csv, "missing baseline comparison coverage CSV row")
    junit_root = ET.parse(output_dir / "build_compare_junit_latest.xml").getroot()
    assert_true(junit_root.attrib.get("failures") == "2", "comparison coverage JUnit did not fail")


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="holyc-build-compare-ci-") as tmp:
            tmp_path = Path(tmp)
            check_pass_case(tmp_path)
            check_command_drift_gate(tmp_path)
            check_environment_drift_gate(tmp_path)
            check_coverage_gate(tmp_path)
            check_comparison_coverage_gate(tmp_path)
    except AssertionError as exc:
        print(f"build_compare_ci_smoke_failed={exc}", file=sys.stderr)
        return 1
    print("build_compare_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
