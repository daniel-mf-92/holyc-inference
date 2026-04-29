#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for host-side perf regression dashboards."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "bench" / "fixtures" / "perf_regression" / "ci_pass.jsonl"
RESULTS = ROOT / "bench" / "results"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-perf-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = Path(tmp) / "dashboards"
        command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(RESULTS),
            "--input",
            str(FIXTURE),
            "--output-dir",
            str(output_dir),
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = output_dir / "perf_regression_latest.json"
        markdown_path = output_dir / "perf_regression_latest.md"
        junit_path = output_dir / "perf_regression_junit_latest.xml"
        commit_points_path = output_dir / "perf_regression_commit_points_latest.csv"
        comparisons_path = output_dir / "perf_regression_comparisons_latest.csv"
        sample_violations_path = output_dir / "perf_regression_sample_violations_latest.csv"
        variability_violations_path = output_dir / "perf_regression_variability_violations_latest.csv"
        wall_variability_violations_path = (
            output_dir / "perf_regression_wall_variability_violations_latest.csv"
        )
        commit_coverage_violations_path = (
            output_dir / "perf_regression_commit_coverage_violations_latest.csv"
        )
        comparison_coverage_violations_path = (
            output_dir / "perf_regression_comparison_coverage_violations_latest.csv"
        )
        prompt_suite_drift_path = output_dir / "perf_regression_prompt_suite_drift_latest.csv"
        environment_drift_path = output_dir / "perf_regression_environment_drift_latest.csv"
        environment_coverage_path = (
            output_dir / "perf_regression_environment_coverage_violations_latest.csv"
        )
        telemetry_coverage_path = (
            output_dir / "perf_regression_telemetry_coverage_violations_latest.csv"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_status={report['status']}", file=sys.stderr)
            return 1
        if report["regressions"]:
            print("unexpected_regressions=true", file=sys.stderr)
            return 1
        if report["sample_violations"]:
            print("unexpected_sample_violations=true", file=sys.stderr)
            return 1
        if report["variability_violations"]:
            print("unexpected_variability_violations=true", file=sys.stderr)
            return 1
        if report["wall_variability_violations"]:
            print("unexpected_wall_variability_violations=true", file=sys.stderr)
            return 1
        if report["commit_coverage_violations"]:
            print("unexpected_commit_coverage_violations=true", file=sys.stderr)
            return 1
        if report["comparison_coverage_violations"]:
            print("unexpected_comparison_coverage_violations=true", file=sys.stderr)
            return 1
        if report["prompt_suite_drift_violations"]:
            print("unexpected_prompt_suite_drift=true", file=sys.stderr)
            return 1
        if "environment_drift_violations" not in report:
            print("missing_environment_drift_report=true", file=sys.stderr)
            return 1
        if "environment_coverage_violations" not in report:
            print("missing_environment_coverage_report=true", file=sys.stderr)
            return 1
        if report["environment_coverage_violations"]:
            print("unexpected_environment_coverage_violations=true", file=sys.stderr)
            return 1
        if report["telemetry_coverage_violations"]:
            print("unexpected_telemetry_coverage_violations=true", file=sys.stderr)
            return 1
        if "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short" not in report["summaries"]:
            print("missing_ci_fixture_summary=true", file=sys.stderr)
            return 1
        fixture_summary = report["summaries"][
            "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short"
        ]
        if fixture_summary.get("median_wall_tok_per_s") != 95.5:
            print("missing_wall_tok_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p05_wall_tok_per_s") != 95.05:
            print("missing_p05_wall_tok_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_us_per_token") != 9950.495049504951:
            print("missing_us_per_token_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p95_us_per_token") != 9995.049504950495:
            print("missing_p95_us_per_token_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_wall_us_per_token") != 10471.491228070176:
            print("missing_wall_us_per_token_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p95_wall_us_per_token") != 10520.833333333334:
            print("missing_p95_wall_us_per_token_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p05_tok_per_s") != 100.05:
            print("missing_p05_tok_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_ttft_us") != 49500.0:
            print("missing_ttft_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p95_ttft_us") != 49950.0:
            print("missing_p95_ttft_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_host_overhead_pct") != 4.5:
            print("missing_host_overhead_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("max_host_child_peak_rss_bytes") != 2200000:
            print("missing_host_child_peak_rss_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_host_child_cpu_us") != 395000.0:
            print("missing_host_child_cpu_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("p95_host_child_cpu_us") != 399500.0:
            print("missing_p95_host_child_cpu_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_host_child_cpu_pct") != 54.5:
            print("missing_host_child_cpu_pct_summary=true", file=sys.stderr)
            return 1
        if fixture_summary.get("median_host_child_tok_per_cpu_s") != 81.0:
            print("missing_host_child_tok_per_cpu_s_summary=true", file=sys.stderr)
            return 1
        if "Perf Regression Dashboard" not in markdown_path.read_text(encoding="utf-8"):
            print("missing_markdown_dashboard=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(junit_path).getroot()
        if junit_root.attrib.get("name") != "holyc_perf_regression":
            print("missing_junit_suite=true", file=sys.stderr)
            return 1
        if junit_root.attrib.get("failures") != "0":
            print("unexpected_junit_failures=true", file=sys.stderr)
            return 1
        commit_points_header = commit_points_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        required_commit_points_columns = {
            "key", "commit", "latest_timestamp", "records",
            "tok_per_s_records", "wall_tok_per_s_records",
            "us_per_token_records", "wall_us_per_token_records",
            "memory_records", "memory_bytes_per_token_records",
            "ttft_us_records", "host_overhead_records",
            "host_child_peak_rss_records",
            "host_child_cpu_us_records", "host_child_cpu_pct_records",
            "host_child_tok_per_cpu_s_records",
            "p05_tok_per_s", "median_tok_per_s", "median_wall_tok_per_s",
            "p05_wall_tok_per_s",
            "median_us_per_token", "p95_us_per_token",
            "median_wall_us_per_token", "p95_wall_us_per_token",
            "median_ttft_us", "p95_ttft_us", "median_host_overhead_pct",
            "median_host_child_cpu_us", "p95_host_child_cpu_us",
            "median_host_child_cpu_pct",
            "median_host_child_tok_per_cpu_s",
            "wall_tok_per_s_cv_pct", "max_memory_bytes",
            "median_memory_bytes_per_token", "max_memory_bytes_per_token",
            "max_host_child_peak_rss_bytes",
            "environment_sha256", "host_platform", "host_machine", "qemu_version", "qemu_bin",
        }
        if not required_commit_points_columns.issubset(commit_points_header):
            missing = sorted(required_commit_points_columns - set(commit_points_header))
            print(f"missing_commit_points_csv=true missing={missing}", file=sys.stderr)
            return 1
        comparisons_csv = comparisons_path.read_text(encoding="utf-8")
        if "key,baseline_commit,candidate_commit,baseline_latest_timestamp" not in comparisons_csv:
            print("missing_comparisons_csv=true", file=sys.stderr)
            return 1
        if "qemu_prompt/ci-airgap-smoke/synthetic-smoke/Q4_0/ci-short,ci-base,ci-head" not in comparisons_csv:
            print("missing_ci_fixture_comparison=true", file=sys.stderr)
            return 1
        if "median_host_overhead_pct_baseline,median_host_overhead_pct_candidate,median_host_overhead_pct_delta_pct" not in comparisons_csv:
            print("missing_host_overhead_comparison=true", file=sys.stderr)
            return 1
        if "median_memory_bytes_per_token_baseline,median_memory_bytes_per_token_candidate,median_memory_bytes_per_token_delta_pct" not in comparisons_csv:
            print("missing_memory_per_token_comparison=true", file=sys.stderr)
            return 1
        if "max_host_child_peak_rss_bytes_baseline,max_host_child_peak_rss_bytes_candidate,max_host_child_peak_rss_bytes_delta_pct" not in comparisons_csv:
            print("missing_host_child_peak_rss_comparison=true", file=sys.stderr)
            return 1
        if "median_host_child_cpu_us_baseline,median_host_child_cpu_us_candidate,median_host_child_cpu_us_delta_pct" not in comparisons_csv:
            print("missing_host_child_cpu_comparison=true", file=sys.stderr)
            return 1
        if "p95_host_child_cpu_us_baseline,p95_host_child_cpu_us_candidate,p95_host_child_cpu_us_delta_pct" not in comparisons_csv:
            print("missing_p95_host_child_cpu_comparison=true", file=sys.stderr)
            return 1
        if "median_host_child_cpu_pct_baseline,median_host_child_cpu_pct_candidate,median_host_child_cpu_pct_delta_pct" not in comparisons_csv:
            print("missing_host_child_cpu_pct_comparison=true", file=sys.stderr)
            return 1
        if "median_host_child_tok_per_cpu_s_baseline,median_host_child_tok_per_cpu_s_candidate,median_host_child_tok_per_cpu_s_delta_pct" not in comparisons_csv:
            print("missing_host_child_tok_per_cpu_s_comparison=true", file=sys.stderr)
            return 1
        if "p05_wall_tok_per_s_baseline,p05_wall_tok_per_s_candidate,p05_wall_tok_per_s_delta_pct" not in comparisons_csv:
            print("missing_p05_wall_tok_comparison=true", file=sys.stderr)
            return 1
        if "median_us_per_token_baseline,median_us_per_token_candidate,median_us_per_token_delta_pct" not in comparisons_csv:
            print("missing_us_per_token_comparison=true", file=sys.stderr)
            return 1
        if "median_wall_us_per_token_baseline,median_wall_us_per_token_candidate,median_wall_us_per_token_delta_pct" not in comparisons_csv:
            print("missing_wall_us_per_token_comparison=true", file=sys.stderr)
            return 1
        environment_drift_csv = environment_drift_path.read_text(encoding="utf-8")
        if "key,environment_sha256s,commits,host_platforms,host_machines,qemu_versions,qemu_bins,sources" not in environment_drift_csv:
            print("missing_environment_drift_csv=true", file=sys.stderr)
            return 1
        if "key,commit,field,records,present_records" not in environment_coverage_path.read_text(
            encoding="utf-8"
        ):
            print("missing_environment_coverage_csv=true", file=sys.stderr)
            return 1

        environment_drift_fixture = tmp_path / "environment_drift.jsonl"
        environment_drift_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "env-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-env",
                            "tok_per_s": 100.0,
                            "environment": {
                                "platform": "macOS-A",
                                "machine": "arm64",
                                "qemu_version": "QEMU synthetic A",
                                "qemu_bin": "qemu-system-x86_64",
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "env-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-env",
                            "tok_per_s": 101.0,
                            "environment": {
                                "platform": "macOS-B",
                                "machine": "arm64",
                                "qemu_version": "QEMU synthetic B",
                                "qemu_bin": "qemu-system-x86_64",
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        environment_drift_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(environment_drift_fixture),
            "--output-dir",
            str(tmp_path / "environment_drift_dashboard"),
            "--fail-on-environment-drift",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            environment_drift_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("environment_drift_not_rejected=true", file=sys.stderr)
            return 1
        drift_report = json.loads(
            (tmp_path / "environment_drift_dashboard" / "perf_regression_latest.json").read_text(
                encoding="utf-8"
            )
        )
        if len(drift_report["environment_drift_violations"]) != 1:
            print("missing_environment_drift_violation=true", file=sys.stderr)
            return 1

        environment_coverage_fixture = tmp_path / "environment_coverage.jsonl"
        environment_coverage_fixture.write_text(
            json.dumps(
                {
                    "timestamp": "2026-04-27T15:00:00Z",
                    "commit": "env-coverage-head",
                    "benchmark": "qemu_prompt",
                    "profile": "ci-airgap-smoke",
                    "model": "synthetic-smoke",
                    "quantization": "Q4_0",
                    "prompt": "ci-env-coverage",
                    "tok_per_s": 100.0,
                    "environment": {
                        "platform": "macOS synthetic",
                        "machine": "arm64",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        environment_coverage_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(environment_coverage_fixture),
            "--output-dir",
            str(tmp_path / "environment_coverage_dashboard"),
            "--require-environment-sha256",
            "--require-host-platform",
            "--require-host-machine",
            "--require-qemu-version",
            "--require-qemu-bin",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            environment_coverage_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("environment_coverage_not_rejected=true", file=sys.stderr)
            return 1
        coverage_report = json.loads(
            (
                tmp_path
                / "environment_coverage_dashboard"
                / "perf_regression_latest.json"
            ).read_text(encoding="utf-8")
        )
        if len(coverage_report["environment_coverage_violations"]) != 2:
            print("missing_environment_coverage_violation=true", file=sys.stderr)
            return 1

        ttft_regression_fixture = tmp_path / "p95_ttft_regression.jsonl"
        ttft_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "ttft-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-ttft",
                            "tok_per_s": 100.0,
                            "ttft_us": 100000,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "ttft-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-ttft",
                            "tok_per_s": 100.0,
                            "ttft_us": 125000,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ttft_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(ttft_regression_fixture),
            "--output-dir",
            str(tmp_path / "ttft_regression_dashboard"),
            "--p95-ttft-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            ttft_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("p95_ttft_regression_not_rejected=true", file=sys.stderr)
            return 1

        overhead_regression_fixture = tmp_path / "host_overhead_regression.jsonl"
        overhead_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "overhead-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-overhead",
                            "tok_per_s": 100.0,
                            "host_overhead_pct": 5.0,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "overhead-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-overhead",
                            "tok_per_s": 100.0,
                            "host_overhead_pct": 8.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        overhead_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(overhead_regression_fixture),
            "--output-dir",
            str(tmp_path / "host_overhead_regression_dashboard"),
            "--host-overhead-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            overhead_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("host_overhead_regression_not_rejected=true", file=sys.stderr)
            return 1

        p05_wall_regression_fixture = tmp_path / "p05_wall_tok_regression.jsonl"
        p05_wall_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "p05-wall-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-p05-wall",
                            "tok_per_s": 100.0,
                            "wall_tok_per_s": 100.0,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "p05-wall-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-p05-wall",
                            "tok_per_s": 100.0,
                            "wall_tok_per_s": 84.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        p05_wall_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(p05_wall_regression_fixture),
            "--output-dir",
            str(tmp_path / "p05_wall_regression_dashboard"),
            "--p05-wall-tok-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            p05_wall_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("p05_wall_tok_regression_not_rejected=true", file=sys.stderr)
            return 1

        min_token_drop_fixture = tmp_path / "min_token_drop_regression.jsonl"
        min_token_drop_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "token-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-token",
                            "tok_per_s": 100.0,
                            "tokens": 100,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:01:00Z",
                            "commit": "token-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-token",
                            "tok_per_s": 100.0,
                            "tokens": 100,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "token-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-token",
                            "tok_per_s": 100.0,
                            "tokens": 100,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:06:00Z",
                            "commit": "token-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-token",
                            "tok_per_s": 100.0,
                            "tokens": 50,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        min_token_drop_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(min_token_drop_fixture),
            "--output-dir",
            str(tmp_path / "min_token_drop_dashboard"),
            "--min-token-drop-regression-pct",
            "40",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            min_token_drop_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("min_token_drop_regression_not_rejected=true", file=sys.stderr)
            return 1
        min_token_drop_report = json.loads(
            (tmp_path / "min_token_drop_dashboard" / "perf_regression_latest.json").read_text(
                encoding="utf-8"
            )
        )
        if not any(
            regression.get("metric") == "min_tokens"
            for regression in min_token_drop_report["regressions"]
        ):
            print("missing_min_token_drop_regression=true", file=sys.stderr)
            return 1

        wall_variability_fixture = tmp_path / "wall_variability.jsonl"
        wall_variability_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "wall-var-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-wall-var",
                            "tok_per_s": 100.0,
                            "wall_tok_per_s": 100.0,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:01:00Z",
                            "commit": "wall-var-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-wall-var",
                            "tok_per_s": 100.0,
                            "wall_tok_per_s": 70.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        wall_variability_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(wall_variability_fixture),
            "--output-dir",
            str(tmp_path / "wall_variability_dashboard"),
            "--max-wall-tok-cv-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            wall_variability_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("wall_variability_not_rejected=true", file=sys.stderr)
            return 1

        token_latency_regression_fixture = tmp_path / "token_latency_regression.jsonl"
        token_latency_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "lat-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-lat",
                            "tok_per_s": 100.0,
                            "us_per_token": 10000.0,
                            "wall_us_per_token": 11000.0,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "lat-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-lat",
                            "tok_per_s": 100.0,
                            "us_per_token": 11600.0,
                            "wall_us_per_token": 12700.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        token_latency_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(token_latency_regression_fixture),
            "--output-dir",
            str(tmp_path / "token_latency_regression_dashboard"),
            "--us-per-token-regression-pct",
            "10",
            "--wall-us-per-token-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            token_latency_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("token_latency_regression_not_rejected=true", file=sys.stderr)
            return 1

        host_rss_regression_fixture = tmp_path / "host_child_peak_rss_regression.jsonl"
        host_rss_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "host-rss-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-host-rss",
                            "tok_per_s": 100.0,
                            "host_child_peak_rss_bytes": 1000000,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "host-rss-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-host-rss",
                            "tok_per_s": 100.0,
                            "host_child_peak_rss_bytes": 1300000,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        host_rss_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(host_rss_regression_fixture),
            "--output-dir",
            str(tmp_path / "host_child_peak_rss_regression_dashboard"),
            "--host-child-peak-rss-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            host_rss_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("host_child_peak_rss_regression_not_rejected=true", file=sys.stderr)
            return 1

        host_tok_cpu_regression_fixture = tmp_path / "host_child_tok_per_cpu_s_regression.jsonl"
        host_tok_cpu_regression_fixture.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:00:00Z",
                            "commit": "host-tok-cpu-base",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-host-tok-cpu",
                            "tok_per_s": 100.0,
                            "host_child_tok_per_cpu_s": 80.0,
                        }
                    ),
                    json.dumps(
                        {
                            "timestamp": "2026-04-27T15:05:00Z",
                            "commit": "host-tok-cpu-head",
                            "benchmark": "qemu_prompt",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "prompt": "ci-host-tok-cpu",
                            "tok_per_s": 100.0,
                            "host_child_tok_per_cpu_s": 60.0,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        host_tok_cpu_regression_command = [
            sys.executable,
            str(ROOT / "bench" / "perf_regression.py"),
            "--input",
            str(host_tok_cpu_regression_fixture),
            "--output-dir",
            str(tmp_path / "host_child_tok_per_cpu_s_regression_dashboard"),
            "--host-child-tok-per-cpu-s-regression-pct",
            "10",
            "--fail-on-regression",
        ]
        completed = subprocess.run(
            host_tok_cpu_regression_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("host_child_tok_per_cpu_s_regression_not_rejected=true", file=sys.stderr)
            return 1
        if "key,commit,records,minimum_records" not in sample_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_sample_violations_csv=true", file=sys.stderr)
            return 1
        if "key,commit,records,tok_per_s_cv_pct,threshold_pct" not in variability_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_variability_violations_csv=true", file=sys.stderr)
            return 1
        if (
            "key,commit,records,wall_tok_per_s_cv_pct,threshold_pct"
            not in wall_variability_violations_path.read_text(encoding="utf-8")
        ):
            print("missing_wall_variability_violations_csv=true", file=sys.stderr)
            return 1
        if "key,commits,minimum_commits,latest_commit" not in commit_coverage_violations_path.read_text(
            encoding="utf-8"
        ):
            print("missing_commit_coverage_violations_csv=true", file=sys.stderr)
            return 1
        if (
            "key,baseline_commit,candidate_commit,missing_commits"
            not in comparison_coverage_violations_path.read_text(encoding="utf-8")
        ):
            print("missing_comparison_coverage_violations_csv=true", file=sys.stderr)
            return 1
        if "key,hashes,commits,sources" not in prompt_suite_drift_path.read_text(encoding="utf-8"):
            print("missing_prompt_suite_drift_csv=true", file=sys.stderr)
            return 1
        if "key,commit,metric,records,present_records" not in telemetry_coverage_path.read_text(
            encoding="utf-8"
        ):
            print("missing_telemetry_coverage_csv=true", file=sys.stderr)
            return 1

        audit_output = tmp_path / "airgap_audit.json"
        audit_markdown = tmp_path / "airgap_audit.md"
        audit_csv = tmp_path / "airgap_audit.csv"
        audit_junit = tmp_path / "airgap_audit.xml"
        audit_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(RESULTS),
            "--output",
            str(audit_output),
            "--markdown",
            str(audit_markdown),
            "--csv",
            str(audit_csv),
            "--junit",
            str(audit_junit),
        ]
        completed = subprocess.run(
            audit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        audit_report = json.loads(audit_output.read_text(encoding="utf-8"))
        if audit_report["status"] != "pass":
            print(f"unexpected_airgap_status={audit_report['status']}", file=sys.stderr)
            return 1
        if audit_report["commands_checked"] < 1:
            print("missing_qemu_command_audit=true", file=sys.stderr)
            return 1
        if "Benchmark Air-Gap Audit" not in audit_markdown.read_text(encoding="utf-8"):
            print("missing_airgap_markdown=true", file=sys.stderr)
            return 1
        if "source,row,reason,command" not in audit_csv.read_text(encoding="utf-8"):
            print("missing_airgap_csv=true", file=sys.stderr)
            return 1
        audit_junit_root = ET.parse(audit_junit).getroot()
        if audit_junit_root.attrib.get("name") != "holyc_benchmark_airgap_audit":
            print("missing_airgap_junit_suite=true", file=sys.stderr)
            return 1
        if audit_junit_root.attrib.get("failures") != "0":
            print("unexpected_airgap_junit_failures=true", file=sys.stderr)
            return 1

        unsafe_fixture = tmp_path / "unsafe_qemu.json"
        unsafe_fixture.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "benchmark": "qemu_prompt",
                            "command": [
                                "qemu-system-x86_64",
                                "-drive",
                                "file=/tmp/TempleOS.img,format=raw,if=ide",
                                "-device",
                                "e1000",
                            ],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_command = [
            sys.executable,
            str(ROOT / "bench" / "airgap_audit.py"),
            "--input",
            str(unsafe_fixture),
            "--output",
            str(tmp_path / "unsafe_airgap_audit.json"),
        ]
        completed = subprocess.run(
            unsafe_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unsafe_qemu_command_not_rejected=true", file=sys.stderr)
            return 1

        source_audit_fixture = tmp_path / "qemu_source_audit.md"
        source_audit_fixture.write_text(
            "\n".join(
                [
                    "```bash",
                    "qemu-system-x86_64 \\",
                    "  -nic none \\",
                    "  -m 512M \\",
                    "  -drive file=/tmp/TempleOS.img,format=raw,if=ide",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        source_audit_args_file = tmp_path / "safe_qemu.args"
        source_audit_args_file.write_text("-m 512M -smp 1\n", encoding="utf-8")
        source_audit_matrix_fixture = tmp_path / "qemu_source_audit_matrix.json"
        source_audit_matrix_fixture.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": "safe-fragment",
                            "qemu_args": ["-m", "512M", "-smp", "1"],
                            "qemu_args_files": [source_audit_args_file.name],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        source_audit_yaml_fixture = tmp_path / "qemu_source_audit_matrix.yml"
        source_audit_yaml_fixture.write_text(
            "\n".join(
                [
                    "profiles:",
                    "  - name: safe-yaml-fragment",
                    "    qemu_args:",
                    "      - -m",
                    "      - 512M",
                    f"    qemu_args_file: {source_audit_args_file.name}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        source_audit_output = tmp_path / "qemu_source_audit.json"
        source_audit_markdown = tmp_path / "qemu_source_audit.md.out"
        source_audit_csv = tmp_path / "qemu_source_audit.csv"
        source_audit_junit = tmp_path / "qemu_source_audit.xml"
        source_audit_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_source_audit.py"),
            "--input",
            str(source_audit_fixture),
            "--input",
            str(source_audit_matrix_fixture),
            "--input",
            str(source_audit_yaml_fixture),
            "--output",
            str(source_audit_output),
            "--markdown",
            str(source_audit_markdown),
            "--csv",
            str(source_audit_csv),
            "--junit",
            str(source_audit_junit),
        ]
        completed = subprocess.run(
            source_audit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        source_audit_report = json.loads(source_audit_output.read_text(encoding="utf-8"))
        if source_audit_report["status"] != "pass":
            print(
                f"unexpected_qemu_source_audit_status={source_audit_report['status']}",
                file=sys.stderr,
            )
            return 1
        if source_audit_report["commands_checked"] != 4:
            print("missing_qemu_source_command_audit=true", file=sys.stderr)
            return 1
        if "QEMU Source Air-Gap Audit" not in source_audit_markdown.read_text(encoding="utf-8"):
            print("missing_qemu_source_audit_markdown=true", file=sys.stderr)
            return 1
        if "source,line,reason,command,text" not in source_audit_csv.read_text(encoding="utf-8"):
            print("missing_qemu_source_audit_csv=true", file=sys.stderr)
            return 1
        source_audit_junit_root = ET.parse(source_audit_junit).getroot()
        if source_audit_junit_root.attrib.get("name") != "holyc_qemu_source_airgap_audit":
            print("missing_qemu_source_audit_junit_suite=true", file=sys.stderr)
            return 1
        if source_audit_junit_root.attrib.get("failures") != "0":
            print("unexpected_qemu_source_audit_junit_failures=true", file=sys.stderr)
            return 1

        unsafe_source_fixture = tmp_path / "unsafe_qemu_source.md"
        unsafe_source_fixture.write_text(
            "qemu-system-x86_64 -m 512M -drive file=/tmp/TempleOS.img,format=raw,if=ide\n",
            encoding="utf-8",
        )
        unsafe_source_args_fixture = tmp_path / "unsafe_qemu_source_args.json"
        unsafe_source_args_fixture.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": "unsafe-fragment",
                            "qemu_args": ["-netdev", "user,id=n0", "-device", "e1000,netdev=n0"],
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_source_yaml_fixture = tmp_path / "unsafe_qemu_source_args.yml"
        unsafe_source_yaml_fixture.write_text(
            "\n".join(
                [
                    "profiles:",
                    "  - name: unsafe-yaml-fragment",
                    "    qemu_extra_args: ['-netdev', 'user,id=n0']",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        unsafe_source_ref_args = tmp_path / "unsafe_qemu_source_ref.args"
        unsafe_source_ref_args.write_text("-netdev user,id=n0\n", encoding="utf-8")
        unsafe_source_ref_fixture = tmp_path / "unsafe_qemu_source_ref.json"
        unsafe_source_ref_fixture.write_text(
            json.dumps(
                {
                    "profiles": [
                        {
                            "name": "unsafe-ref-fragment",
                            "qemu_args_file": unsafe_source_ref_args.name,
                        }
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        unsafe_source_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_source_audit.py"),
            "--input",
            str(unsafe_source_fixture),
            "--input",
            str(unsafe_source_args_fixture),
            "--input",
            str(unsafe_source_yaml_fixture),
            "--input",
            str(unsafe_source_ref_fixture),
            "--output",
            str(tmp_path / "unsafe_qemu_source_audit.json"),
        ]
        completed = subprocess.run(
            unsafe_source_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("unsafe_qemu_source_command_not_rejected=true", file=sys.stderr)
            return 1

        dry_run_output_dir = tmp_path / "qemu_prompt_bench_dry_run"
        dry_run_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            "/tmp/TempleOS.synthetic.img",
            "--prompts",
            str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
            "--qemu-bin",
            str(ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"),
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--max-launches",
            "10",
            "--output-dir",
            str(dry_run_output_dir),
            "--dry-run",
        ]
        completed = subprocess.run(
            dry_run_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        dry_run_report = json.loads(
            (dry_run_output_dir / "qemu_prompt_bench_dry_run_latest.json").read_text(encoding="utf-8")
        )
        if dry_run_report.get("planned_total_launches") != 6:
            print("dry_run_launch_count_mismatch=true", file=sys.stderr)
            return 1
        dry_run_csv = (dry_run_output_dir / "qemu_prompt_bench_dry_run_latest.csv").read_text(
            encoding="utf-8"
        )
        if "planned_total_launches" not in dry_run_csv or "-nic" not in dry_run_csv:
            print("missing_dry_run_csv_fields=true", file=sys.stderr)
            return 1
        dry_run_junit = ET.parse(dry_run_output_dir / "qemu_prompt_bench_dry_run_junit_latest.xml").getroot()
        if dry_run_junit.attrib.get("name") != "holyc_qemu_prompt_bench_dry_run":
            print("missing_dry_run_junit_suite=true", file=sys.stderr)
            return 1
        if dry_run_junit.attrib.get("failures") != "0":
            print("unexpected_dry_run_junit_failures=true", file=sys.stderr)
            return 1

        bench_output_dir = tmp_path / "qemu_prompt_bench"
        bench_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            "/tmp/TempleOS.synthetic.img",
            "--prompts",
            str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
            "--qemu-bin",
            str(ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--repeat",
            "3",
            "--max-prompt-cv-pct",
            "0.1",
            "--min-wall-tok-per-s",
            "1",
            "--min-tokens-per-prompt-byte",
            "0.5",
            "--min-total-tokens",
            "100",
            "--max-memory-bytes",
            "100000000",
            "--max-memory-bytes-per-token",
            "3000000",
            "--require-guest-prompt-bytes-match",
            "--output-dir",
            str(bench_output_dir),
        ]
        completed = subprocess.run(
            bench_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        bench_report = json.loads(
            (bench_output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        suite_summary = bench_report["suite_summary"]
        telemetry_gates = bench_report["telemetry_gates"]
        if not bench_report.get("launch_plan_sha256"):
            print("missing_measured_launch_plan_hash=true", file=sys.stderr)
            return 1
        if len(bench_report.get("launch_plan") or []) != 6:
            print("measured_launch_plan_count_mismatch=true", file=sys.stderr)
            return 1
        expected_launch_indexes = list(range(1, 7))
        actual_launch_indexes = [
            row.get("launch_index")
            for row in bench_report["warmups"] + bench_report["benchmarks"]
        ]
        if actual_launch_indexes != expected_launch_indexes:
            print("measured_launch_order_mismatch=true", file=sys.stderr)
            return 1
        measured_launch_csv = (bench_output_dir / "qemu_prompt_bench_launches_latest.csv").read_text(
            encoding="utf-8"
        )
        if "launch_plan_sha256" not in measured_launch_csv or "measured" not in measured_launch_csv:
            print("missing_measured_launch_csv_fields=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("min_wall_tok_per_s") != 1.0:
            print("missing_min_wall_tok_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("max_memory_bytes") != 100000000:
            print("missing_max_memory_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("max_memory_bytes_per_token") != 3000000.0:
            print("missing_max_memory_per_token_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("require_guest_prompt_bytes_match") is not True:
            print("missing_guest_prompt_bytes_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("min_tokens_per_prompt_byte") != 0.5:
            print("missing_min_tokens_per_prompt_byte_gate=true", file=sys.stderr)
            return 1
        if telemetry_gates.get("min_total_tokens") != 100:
            print("missing_min_total_tokens_gate=true", file=sys.stderr)
            return 1
        if suite_summary.get("tok_per_s_stdev") is None:
            print("missing_suite_tok_stdev=true", file=sys.stderr)
            return 1
        if suite_summary.get("failed_runs") != 0:
            print("unexpected_suite_failed_runs=true", file=sys.stderr)
            return 1
        if suite_summary.get("ok_run_pct") != 100.0:
            print("unexpected_suite_ok_run_pct=true", file=sys.stderr)
            return 1
        if suite_summary.get("timed_out_runs") != 0:
            print("unexpected_suite_timed_out_runs=true", file=sys.stderr)
            return 1
        if suite_summary.get("nonzero_exit_runs") != 0:
            print("unexpected_suite_nonzero_exit_runs=true", file=sys.stderr)
            return 1
        if suite_summary.get("tok_per_s_cv_pct") is None:
            print("missing_suite_tok_cv=true", file=sys.stderr)
            return 1
        if suite_summary.get("wall_tok_per_s_p05") is None:
            print("missing_suite_wall_tok_p05=true", file=sys.stderr)
            return 1
        if suite_summary.get("wall_tok_per_s_p05_p95_spread_pct") is None:
            print("missing_suite_wall_tok_spread=true", file=sys.stderr)
            return 1
        if suite_summary.get("ttft_us_median") is None:
            print("missing_suite_ttft_median=true", file=sys.stderr)
            return 1
        if suite_summary.get("ttft_us_p95") is None:
            print("missing_suite_ttft_p95=true", file=sys.stderr)
            return 1
        if suite_summary.get("host_overhead_us_median") is None:
            print("missing_suite_host_overhead=true", file=sys.stderr)
            return 1
        if suite_summary.get("host_child_cpu_us_median") is None:
            print("missing_suite_host_child_cpu=true", file=sys.stderr)
            return 1
        if suite_summary.get("host_child_cpu_pct_median") is None:
            print("missing_suite_host_child_cpu_pct=true", file=sys.stderr)
            return 1
        if suite_summary.get("tokens_per_prompt_byte_median") is None:
            print("missing_suite_tokens_per_prompt_byte=true", file=sys.stderr)
            return 1
        if suite_summary.get("memory_bytes_per_token_median") is None:
            print("missing_suite_memory_per_token=true", file=sys.stderr)
            return 1
        if suite_summary.get("memory_bytes_per_token_max") != 2099200.0:
            print("unexpected_suite_memory_per_token_max=true", file=sys.stderr)
            return 1
        if not all("tok_per_s_cv_pct" in row for row in bench_report["summaries"]):
            print("missing_prompt_tok_cv=true", file=sys.stderr)
            return 1
        if not all(
            row.get("wall_tok_per_s_p05") is not None
            and row.get("wall_tok_per_s_p05_p95_spread_pct") is not None
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_wall_tok_tail=true", file=sys.stderr)
            return 1
        if not all(
            row.get("failed_runs") == 0
            and row.get("ok_run_pct") == 100.0
            and row.get("timed_out_runs") == 0
            and row.get("nonzero_exit_runs") == 0
            for row in bench_report["summaries"]
        ):
            print("unexpected_prompt_failure_counts=true", file=sys.stderr)
            return 1
        if not all(
            "host_overhead_us_median" in row and "host_overhead_pct_median" in row
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_host_overhead=true", file=sys.stderr)
            return 1
        if not all(
            "host_child_cpu_us_median" in row and "host_child_cpu_pct_median" in row
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_host_child_cpu=true", file=sys.stderr)
            return 1
        if not all(
            "host_child_cpu_us" in row and "host_child_cpu_pct" in row
            for row in bench_report["benchmarks"]
        ):
            print("missing_run_host_child_cpu=true", file=sys.stderr)
            return 1
        if not all(
            "ttft_us_median" in row and "ttft_us_p95" in row
            for row in bench_report["summaries"]
        ):
            print("missing_prompt_ttft=true", file=sys.stderr)
            return 1
        if not all(row.get("tokens_per_prompt_byte_median") is not None for row in bench_report["summaries"]):
            print("missing_prompt_tokens_per_prompt_byte=true", file=sys.stderr)
            return 1
        if not all(row.get("tokens_per_prompt_byte") is not None for row in bench_report["benchmarks"]):
            print("missing_run_tokens_per_prompt_byte=true", file=sys.stderr)
            return 1
        if not all(
            row.get("guest_prompt_bytes") == row.get("prompt_bytes")
            and row.get("guest_prompt_bytes_match") is True
            for row in bench_report["benchmarks"]
        ):
            print("missing_run_guest_prompt_bytes_match=true", file=sys.stderr)
            return 1
        if not all(row.get("memory_bytes_per_token_median") is not None for row in bench_report["summaries"]):
            print("missing_prompt_memory_per_token=true", file=sys.stderr)
            return 1
        if not all(row.get("memory_bytes_per_token") is not None for row in bench_report["benchmarks"]):
            print("missing_run_memory_per_token=true", file=sys.stderr)
            return 1
        if bench_report.get("variability_findings"):
            print("unexpected_variability_findings=true", file=sys.stderr)
            return 1
        if bench_report.get("telemetry_findings"):
            print("unexpected_bench_telemetry_findings=true", file=sys.stderr)
            return 1

        bench_gate_fail_dir = tmp_path / "qemu_prompt_bench_gate_fail"
        bench_gate_fail_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            "/tmp/TempleOS.synthetic.img",
            "--prompts",
            str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
            "--qemu-bin",
            str(ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--min-wall-tok-per-s",
            "1000000",
            "--max-memory-bytes",
            "1",
            "--max-memory-bytes-per-token",
            "1",
            "--min-tokens-per-prompt-byte",
            "1000",
            "--min-total-tokens",
            "1000000",
            "--output-dir",
            str(bench_gate_fail_dir),
        ]
        completed = subprocess.run(
            bench_gate_fail_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bench_telemetry_gates_did_not_fail=true", file=sys.stderr)
            return 1
        bench_gate_fail_report = json.loads(
            (bench_gate_fail_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        gate_metrics = {finding.get("metric") for finding in bench_gate_fail_report["telemetry_findings"]}
        if not {
            "wall_tok_per_s",
            "memory_bytes",
            "memory_bytes_per_token",
            "tokens_per_prompt_byte",
            "total_tokens",
        }.issubset(gate_metrics):
            print("missing_bench_gate_failure_metrics=true", file=sys.stderr)
            return 1

        bad_prompt_bytes_qemu = tmp_path / "qemu_bad_prompt_bytes.py"
        bad_prompt_bytes_qemu.write_text(
            """#!/usr/bin/env python3
import hashlib
import json
import sys

prompt = sys.stdin.read()
print("BENCH_RESULT: " + json.dumps({
    "tokens": 8,
    "elapsed_us": 100000,
    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "prompt_bytes": len(prompt.encode("utf-8")) + 1,
}))
""",
            encoding="utf-8",
        )
        bad_prompt_bytes_qemu.chmod(0o755)
        bad_prompt_bytes_dir = tmp_path / "qemu_prompt_bench_bad_prompt_bytes"
        bad_prompt_bytes_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            "/tmp/TempleOS.synthetic.img",
            "--prompts",
            str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
            "--qemu-bin",
            str(bad_prompt_bytes_qemu),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--require-guest-prompt-bytes-match",
            "--output-dir",
            str(bad_prompt_bytes_dir),
        ]
        completed = subprocess.run(
            bad_prompt_bytes_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("guest_prompt_bytes_gate_did_not_fail=true", file=sys.stderr)
            return 1
        bad_prompt_bytes_report = json.loads(
            (bad_prompt_bytes_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        bad_prompt_bytes_metrics = {
            finding.get("metric") for finding in bad_prompt_bytes_report["telemetry_findings"]
        }
        if "guest_prompt_bytes_match" not in bad_prompt_bytes_metrics:
            print("missing_guest_prompt_bytes_gate_failure=true", file=sys.stderr)
            return 1

        matrix_output_dir = tmp_path / "bench_matrix"
        matrix_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_matrix.py"),
            "--matrix",
            str(ROOT / "bench" / "fixtures" / "bench_matrix_smoke.json"),
            "--output-dir",
            str(matrix_output_dir),
            "--max-suite-cv-pct",
            "0.1",
        ]
        completed = subprocess.run(
            matrix_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        matrix_report = json.loads(
            (matrix_output_dir / "bench_matrix_latest.json").read_text(encoding="utf-8")
        )
        if matrix_report["status"] != "pass":
            print(f"unexpected_matrix_status={matrix_report['status']}", file=sys.stderr)
            return 1
        if matrix_report["variability_gates"].get("max_suite_cv_pct") != 0.1:
            print("missing_matrix_suite_cv_gate=true", file=sys.stderr)
            return 1
        if matrix_report["variability_gates"].get("max_prompt_cv_pct") != 0.1:
            print("missing_matrix_prompt_cv_gate=true", file=sys.stderr)
            return 1
        if not matrix_report["cells"] or any(
            "variability_findings" not in cell for cell in matrix_report["cells"]
        ):
            print("missing_matrix_variability_findings=true", file=sys.stderr)
            return 1
        if any(cell["variability_findings"] for cell in matrix_report["cells"]):
            print("unexpected_matrix_variability_findings=true", file=sys.stderr)
            return 1
        if "Variability gates" not in (matrix_output_dir / "bench_matrix_latest.md").read_text(
            encoding="utf-8"
        ):
            print("missing_matrix_variability_markdown=true", file=sys.stderr)
            return 1
        if "variability_findings" not in (
            matrix_output_dir / "bench_matrix_latest.csv"
        ).read_text(encoding="utf-8"):
            print("missing_matrix_variability_csv=true", file=sys.stderr)
            return 1
        matrix_summary_csv = (matrix_output_dir / "bench_matrix_summary_latest.csv").read_text(
            encoding="utf-8"
        )
        matrix_summary_lines = matrix_summary_csv.splitlines()
        matrix_summary_header = set(matrix_summary_lines[0].split(","))
        matrix_summary_required_fields = {
            "scope",
            "status",
            "cells",
            "passing_cells",
            "failing_cells",
            "prompts",
            "prompt_bytes_total",
            "median_tok_per_s_median",
            "wall_tok_per_s_median",
            "ttft_us_p95_max",
            "host_child_peak_rss_bytes_max",
            "variability_findings",
        }
        if not matrix_summary_required_fields.issubset(matrix_summary_header):
            print("missing_matrix_summary_csv_fields=true", file=sys.stderr)
            return 1
        if len(matrix_summary_lines) != len(matrix_report["cells"]) + 2:
            print("unexpected_matrix_summary_csv_rows=true", file=sys.stderr)
            return 1
        if not matrix_summary_lines[1].startswith("matrix,"):
            print("missing_matrix_summary_csv_matrix_row=true", file=sys.stderr)
            return 1
        matrix_metric_fields = {
            "wall_tok_per_s_median",
            "ttft_us_p95",
            "host_overhead_pct_median",
            "us_per_token_median",
            "wall_us_per_token_median",
        }
        matrix_prompt_size_fields = {
            "prompt_bytes_total",
            "prompt_bytes_min",
            "prompt_bytes_max",
        }
        for cell in matrix_report["cells"]:
            missing_fields = [
                field for field in matrix_metric_fields if cell.get(field) is None
            ]
            if missing_fields:
                print(
                    "missing_matrix_rollup_metrics="
                    + ",".join(sorted(missing_fields)),
                    file=sys.stderr,
                )
                return 1
            missing_prompt_size_fields = [
                field for field in matrix_prompt_size_fields if cell.get(field) is None
            ]
            if missing_prompt_size_fields:
                print(
                    "missing_matrix_prompt_size_fields="
                    + ",".join(sorted(missing_prompt_size_fields)),
                    file=sys.stderr,
                )
                return 1
        matrix_csv = (matrix_output_dir / "bench_matrix_latest.csv").read_text(
            encoding="utf-8"
        )
        if not matrix_metric_fields.issubset(set(matrix_csv.splitlines()[0].split(","))):
            print("missing_matrix_metric_csv_fields=true", file=sys.stderr)
            return 1
        if not matrix_prompt_size_fields.issubset(set(matrix_csv.splitlines()[0].split(","))):
            print("missing_matrix_prompt_size_csv_fields=true", file=sys.stderr)
            return 1
        if "Wall tok/s" not in (matrix_output_dir / "bench_matrix_latest.md").read_text(
            encoding="utf-8"
        ):
            print("missing_matrix_metric_markdown=true", file=sys.stderr)
            return 1
        if "Prompt bytes" not in (matrix_output_dir / "bench_matrix_latest.md").read_text(
            encoding="utf-8"
        ):
            print("missing_matrix_prompt_size_markdown=true", file=sys.stderr)
            return 1

        result_index_output_dir = tmp_path / "bench_result_index"
        result_index_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_result_index.py"),
            "--input",
            str(matrix_output_dir),
            "--output-dir",
            str(result_index_output_dir),
            "--fail-on-airgap",
            "--fail-on-telemetry",
            "--fail-on-command-hash-metadata",
        ]
        completed = subprocess.run(
            result_index_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        result_index_report = json.loads(
            (result_index_output_dir / "bench_result_index_latest.json").read_text(
                encoding="utf-8"
            )
        )
        if result_index_report["status"] != "pass":
            print(
                f"unexpected_result_index_status={result_index_report['status']}",
                file=sys.stderr,
            )
            return 1
        if not result_index_report["artifacts"]:
            print("missing_result_index_artifacts=true", file=sys.stderr)
            return 1
        for artifact in result_index_report["artifacts"]:
            missing_fields = [
                field for field in matrix_metric_fields if artifact.get(field) is None
            ]
            if missing_fields:
                print(
                    "missing_result_index_rollup_metrics="
                    + ",".join(sorted(missing_fields)),
                    file=sys.stderr,
                )
                return 1
        result_index_csv = (
            result_index_output_dir / "bench_result_index_latest.csv"
        ).read_text(encoding="utf-8")
        if not matrix_metric_fields.issubset(set(result_index_csv.splitlines()[0].split(","))):
            print("missing_result_index_metric_csv_fields=true", file=sys.stderr)
            return 1
        if "Wall tok/s" not in (
            result_index_output_dir / "bench_result_index_latest.md"
        ).read_text(encoding="utf-8"):
            print("missing_result_index_metric_markdown=true", file=sys.stderr)
            return 1

        manifest_fixture_dir = tmp_path / "manifest_fixture"
        manifest_fixture_dir.mkdir()
        incomplete_report = manifest_fixture_dir / "qemu_prompt_bench_incomplete.json"
        incomplete_report.write_text(
            json.dumps(
                {
                    "generated_at": "2026-04-28T00:00:00Z",
                    "status": "pass",
                    "prompt_suite": {
                        "prompt_count": 1,
                        "suite_sha256": "manifest-smoke-suite",
                    },
                    "benchmarks": [
                        {
                            "benchmark": "qemu_prompt",
                            "profile": "manifest-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "commit": "manifest-smoke-commit",
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=/tmp/TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_output_dir = tmp_path / "manifest_output"
        manifest_airgap_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-airgap",
        ]
        completed = subprocess.run(
            manifest_airgap_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_airgap_gate_failed_on_non_airgap_issue=true", file=sys.stderr)
            return completed.returncode

        manifest_telemetry_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-telemetry",
        ]
        completed = subprocess.run(
            manifest_telemetry_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("manifest_telemetry_gate_did_not_fail=true", file=sys.stderr)
            return 1

        manifest_commit_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(manifest_fixture_dir),
            "--output-dir",
            str(manifest_output_dir),
            "--fail-on-commit-metadata",
        ]
        completed = subprocess.run(
            manifest_commit_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_commit_gate_failed_on_stale_only=true", file=sys.stderr)
            return completed.returncode

        command_hash_fixture_dir = tmp_path / "command_hash_manifest_fixture"
        command_hash_fixture_dir.mkdir()
        command_hash_report = command_hash_fixture_dir / "qemu_prompt_bench_command_hash_mixed.json"
        command_hash_payload = json.loads(incomplete_report.read_text(encoding="utf-8"))
        command_hash_payload["command_sha256"] = "report-command-hash"
        command_hash_payload["benchmarks"][0]["command_sha256"] = "run-command-hash"
        command_hash_report.write_text(json.dumps(command_hash_payload) + "\n", encoding="utf-8")
        manifest_command_hash_output_dir = tmp_path / "manifest_command_hash_output"
        manifest_command_hash_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(command_hash_fixture_dir),
            "--output-dir",
            str(manifest_command_hash_output_dir),
            "--fail-on-command-hash-metadata",
        ]
        completed = subprocess.run(
            manifest_command_hash_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("manifest_command_hash_gate_did_not_fail=true", file=sys.stderr)
            return 1
        command_hash_manifest = json.loads(
            (manifest_command_hash_output_dir / "bench_artifact_manifest_latest.json").read_text(
                encoding="utf-8"
            )
        )
        command_hash_artifact = command_hash_manifest["latest_artifacts"][0]
        if command_hash_artifact.get("command_hash_status") != "fail":
            print("missing_manifest_command_hash_failure=true", file=sys.stderr)
            return 1
        if "Command Hash" not in (
            manifest_command_hash_output_dir / "bench_artifact_manifest_latest.md"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_command_hash_markdown=true", file=sys.stderr)
            return 1
        if "command_hash_status" not in (
            manifest_command_hash_output_dir / "bench_artifact_manifest_latest.csv"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_command_hash_csv=true", file=sys.stderr)
            return 1
        command_hash_junit_root = ET.parse(
            manifest_command_hash_output_dir / "bench_artifact_manifest_junit_latest.xml"
        ).getroot()
        if command_hash_junit_root.attrib.get("failures") == "0":
            print("missing_manifest_command_hash_junit_failure=true", file=sys.stderr)
            return 1

        complete_manifest_fixture_dir = tmp_path / "complete_manifest_fixture"
        complete_manifest_fixture_dir.mkdir()
        complete_report = complete_manifest_fixture_dir / "qemu_prompt_bench_complete.json"
        complete_payload = json.loads(incomplete_report.read_text(encoding="utf-8"))
        complete_payload["suite_summary"] = {
            "prompts": 1,
            "tok_per_s_median": 88.0,
            "memory_bytes_max": 123456,
        }
        complete_payload["benchmarks"][0]["tokens"] = 8
        complete_payload["benchmarks"][0]["elapsed_us"] = 90909
        complete_report.write_text(json.dumps(complete_payload) + "\n", encoding="utf-8")

        manifest_fresh_output_dir = tmp_path / "manifest_fresh_output"
        manifest_fresh_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(complete_manifest_fixture_dir),
            "--output-dir",
            str(manifest_fresh_output_dir),
            "--max-artifact-age-hours",
            "1000000",
            "--fail-on-stale-artifact",
        ]
        completed = subprocess.run(
            manifest_fresh_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            print("manifest_fresh_gate_failed=true", file=sys.stderr)
            return completed.returncode
        manifest_report = json.loads(
            (manifest_fresh_output_dir / "bench_artifact_manifest_latest.json").read_text(
                encoding="utf-8"
            )
        )
        latest_artifact = manifest_report["latest_artifacts"][0]
        if latest_artifact.get("freshness_status") != "pass":
            print("missing_manifest_freshness_pass=true", file=sys.stderr)
            return 1
        if latest_artifact.get("generated_age_seconds") is None:
            print("missing_manifest_artifact_age=true", file=sys.stderr)
            return 1
        if "Freshness" not in (
            manifest_fresh_output_dir / "bench_artifact_manifest_latest.md"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_freshness_markdown=true", file=sys.stderr)
            return 1
        if "freshness_status" not in (
            manifest_fresh_output_dir / "bench_artifact_manifest_latest.csv"
        ).read_text(encoding="utf-8"):
            print("missing_manifest_freshness_csv=true", file=sys.stderr)
            return 1
        manifest_junit_root = ET.parse(
            manifest_fresh_output_dir / "bench_artifact_manifest_junit_latest.xml"
        ).getroot()
        if manifest_junit_root.attrib.get("name") != "holyc_bench_artifact_manifest":
            print("missing_manifest_junit_suite=true", file=sys.stderr)
            return 1
        if manifest_junit_root.attrib.get("failures") != "0":
            print("unexpected_manifest_junit_failures=true", file=sys.stderr)
            return 1

        stale_manifest_fixture_dir = tmp_path / "stale_manifest_fixture"
        stale_manifest_fixture_dir.mkdir()
        stale_report = stale_manifest_fixture_dir / "qemu_prompt_bench_stale.json"
        stale_payload = json.loads(complete_report.read_text(encoding="utf-8"))
        stale_payload["generated_at"] = "2000-01-01T00:00:00Z"
        stale_report.write_text(json.dumps(stale_payload) + "\n", encoding="utf-8")
        stale_manifest_command = [
            sys.executable,
            str(ROOT / "bench" / "bench_artifact_manifest.py"),
            "--input",
            str(stale_manifest_fixture_dir),
            "--output-dir",
            str(tmp_path / "stale_manifest_output"),
            "--max-artifact-age-hours",
            "1",
            "--fail-on-stale-artifact",
        ]
        completed = subprocess.run(
            stale_manifest_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("manifest_stale_gate_did_not_fail=true", file=sys.stderr)
            return 1

    print("perf_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
