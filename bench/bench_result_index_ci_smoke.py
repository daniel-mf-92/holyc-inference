#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for benchmark result indexing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def launch_plan_sha256(launch_plan: list[dict[str, object]]) -> str:
    encoded = json.dumps(launch_plan, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_sha256(environment: dict[str, object]) -> str:
    normalized = {
        key: environment[key]
        for key in (
            "platform",
            "machine",
            "processor",
            "python",
            "cpu_count",
            "qemu_bin",
            "qemu_path",
            "qemu_version",
        )
        if environment.get(key) not in (None, "")
    }
    encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_qemu_report(
    path: Path,
    command: list[str],
    *,
    generated_at: str | None = None,
    suite: str = "index-suite",
    launch_plan: str | None = None,
    environment: dict[str, object] | None = None,
    command_hash: str | None = None,
) -> None:
    generated_at = generated_at or iso_now()
    environment = environment or {
        "platform": "ci-smoke",
        "machine": "host",
        "qemu_bin": command[0],
        "qemu_version": "synthetic",
    }
    command_hash = command_hash or command_sha256(command)
    launch_plan = launch_plan or launch_plan_sha256(
        [
            {"phase": "warmup", "prompt_id": "index-smoke"},
            {"phase": "measured", "prompt_id": "index-smoke"},
        ]
    )
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "generated_at": generated_at,
                "command_sha256": command_hash,
                "launch_plan_sha256": launch_plan,
                "prompt_suite": {
                    "expected_token_prompts": 1,
                    "expected_tokens_total": 32,
                    "name": "index-smoke",
                    "prompt_count": 1,
                    "suite_sha256": suite,
                },
                "environment": environment,
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "command": command,
                        "command_sha256": command_hash,
                        "commit": "index-smoke",
                        "profile": "ci-airgap-smoke",
                        "model": "synthetic-smoke",
                        "quantization": "Q4_0",
                        "prompt": "index-smoke",
                        "prompt_bytes": 80,
                        "tokens": 32,
                        "elapsed_us": 200000,
                        "expected_tokens": 32,
                        "expected_tokens_match": True,
                        "wall_elapsed_us": 220000,
                        "tok_per_s": 160.0,
                        "wall_tok_per_s": 145.45,
                        "prompt_bytes_per_s": 400.0,
                        "wall_prompt_bytes_per_s": 363.64,
                        "tokens_per_prompt_byte": 0.4,
                        "ttft_us": 12000,
                        "memory_bytes": 67174400,
                        "memory_bytes_per_token": 2099200.0,
                        "serial_output_bytes": 4096,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ],
                "summaries": [
                    {
                        "prompt": "index-smoke",
                        "tok_per_s_median": 160.0,
                        "wall_tok_per_s_median": 145.45,
                        "ttft_us_p95": 12000.0,
                        "host_overhead_pct_median": 10.0,
                        "host_child_cpu_us_median": 180000.0,
                        "host_child_cpu_pct_median": 81.8,
                        "host_child_tok_per_cpu_s_median": 177.78,
                        "host_child_peak_rss_bytes_max": 73400320,
                        "prompt_bytes": 80,
                        "prompt_bytes_per_s_median": 400.0,
                        "wall_prompt_bytes_per_s_median": 363.64,
                        "tokens_per_prompt_byte_median": 0.4,
                        "us_per_token_median": 6250.0,
                        "wall_us_per_token_median": 6875.0,
                        "memory_bytes_per_token_median": 2099200.0,
                        "memory_bytes_per_token_max": 2099200.0,
                        "serial_output_bytes_total": 4096,
                        "serial_output_bytes_max": 4096,
                        "memory_bytes_max": 67174400,
                    }
                ],
                "suite_summary": {
                    "prompts": 1,
                    "measured_prompt_bytes_total": 80,
                    "prompt_bytes_min": 80,
                    "prompt_bytes_max": 80,
                    "total_tokens": 32,
                    "total_elapsed_us": 200000,
                    "tok_per_s_median": 160.0,
                    "wall_tok_per_s_median": 145.45,
                    "prompt_bytes_per_s_median": 400.0,
                    "wall_prompt_bytes_per_s_median": 363.64,
                    "tokens_per_prompt_byte_median": 0.4,
                    "ttft_us_p95": 12000.0,
                    "host_overhead_pct_median": 10.0,
                    "memory_bytes_per_token_median": 2099200.0,
                    "memory_bytes_per_token_max": 2099200.0,
                    "serial_output_bytes_total": 4096,
                    "serial_output_bytes_max": 4096,
                    "memory_bytes_max": 67174400,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_dry_run_report(path: Path, command: list[str]) -> None:
    command_hash = command_sha256(command)
    launch_plan = [
        {"phase": "warmup", "prompt_id": "index-smoke"},
        {"phase": "measured", "prompt_id": "index-smoke"},
    ]
    path.write_text(
        json.dumps(
            {
                "status": "planned",
                "generated_at": iso_now(),
                "command": command,
                "command_sha256": command_hash,
                "commit": "index-smoke",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "prompt_suite": {
                    "expected_token_prompts": 1,
                    "expected_tokens_total": 32,
                    "prompt_count": 1,
                    "suite_sha256": "index-suite",
                },
                "launch_plan_sha256": launch_plan_sha256(launch_plan),
                "planned_warmup_launches": 1,
                "planned_measured_launches": 1,
                "planned_total_launches": 2,
                "launch_plan": launch_plan,
                "environment": {
                    "platform": "ci-smoke",
                    "machine": "host",
                    "qemu_bin": command[0],
                    "qemu_version": "synthetic",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_matrix_report(path: Path, command: list[str]) -> None:
    command_hash = command_sha256(command)
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "generated_at": iso_now(),
                "environment": {
                    "platform": "ci-smoke",
                    "machine": "host",
                    "qemu_bin": command[0],
                    "qemu_version": "synthetic",
                },
                "cells": [
                    {
                        "status": "pass",
                        "command": command,
                        "command_sha256": command_hash,
                        "commit": "index-smoke",
                        "profile": "ci-airgap-smoke",
                        "model": "synthetic-smoke",
                        "quantization": "Q8_0",
                        "prompt_suite_sha256": "index-suite",
                        "launch_plan_sha256": "matrix-plan",
                        "prompts": 1,
                        "measured_runs": 1,
                        "warmup_runs": 0,
                        "total_tokens": 64,
                        "total_elapsed_us": 320000,
                        "measured_prompt_bytes_total": 128,
                        "prompt_bytes_min": 128,
                        "prompt_bytes_max": 128,
                        "median_tok_per_s": 200.0,
                        "wall_tok_per_s_median": 190.0,
                        "prompt_bytes_per_s_median": 400.0,
                        "wall_prompt_bytes_per_s_median": 380.0,
                        "tokens_per_prompt_byte_median": 0.5,
                        "ttft_us_p95": 10000.0,
                        "host_child_tok_per_cpu_s_median": 210.0,
                        "host_child_peak_rss_bytes_max": 73400320,
                        "memory_bytes_per_token_median": 1049600.0,
                        "memory_bytes_per_token_max": 1049600.0,
                        "serial_output_bytes_total": 8192,
                        "serial_output_bytes_max": 8192,
                        "max_memory_bytes": 67174400,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_index(input_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "bench_result_index.py"),
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    safe_command = [
        "qemu-system-x86_64",
        "-nic",
        "none",
        "-display",
        "none",
        "-drive",
        "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
    ]

    with tempfile.TemporaryDirectory(prefix="holyc-index-ci-") as tmp:
        tmp_path = Path(tmp)
        safe_source = tmp_path / "safe_source"
        safe_source.mkdir()
        write_qemu_report(safe_source / "qemu_prompt_bench_safe.json", safe_command)
        write_dry_run_report(safe_source / "qemu_prompt_bench_dry_run_safe.json", safe_command)
        write_matrix_report(safe_source / "bench_matrix_safe.json", safe_command)

        safe_output = tmp_path / "safe_index"
        completed = run_index(
            safe_source,
            safe_output,
            "--fail-on-airgap",
            "--fail-on-telemetry",
            "--fail-on-command-hash-metadata",
            "--fail-on-commit-metadata",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = safe_output / "bench_result_index_latest.json"
        markdown_path = safe_output / "bench_result_index_latest.md"
        csv_path = safe_output / "bench_result_index_latest.csv"
        latest_csv_path = safe_output / "bench_result_index_latest_comparable_latest.csv"
        launch_plan_drift_csv_path = safe_output / "bench_result_index_launch_plan_drift_latest.csv"
        dry_run_coverage_csv_path = safe_output / "bench_result_index_dry_run_coverage_latest.csv"
        history_coverage_csv_path = safe_output / "bench_result_index_history_coverage_latest.csv"
        freshness_failures_csv_path = safe_output / "bench_result_index_freshness_failures_latest.csv"
        junit_path = safe_output / "bench_result_index_junit_latest.xml"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_index_status={report['status']}", file=sys.stderr)
            return 1
        if len(report["artifacts"]) != 3:
            print("unexpected_index_artifact_count=true", file=sys.stderr)
            return 1
        if len(report["latest_comparable_artifacts"]) != 2:
            print("unexpected_latest_comparable_count=true", file=sys.stderr)
            return 1
        qemu_artifact = next(row for row in report["artifacts"] if row["artifact_type"] == "qemu_prompt")
        if qemu_artifact.get("memory_bytes_per_token_median") != 2099200.0:
            print("missing_index_memory_per_token=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("expected_tokens_total") != 32:
            print("missing_index_expected_tokens_total=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("expected_tokens_matches") != 1:
            print("missing_index_expected_token_matches=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("measured_prompt_bytes_total") != 80:
            print("missing_index_prompt_bytes_total=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("prompt_bytes_per_s_median") != 400.0:
            print("missing_index_prompt_bytes_per_s=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("tokens_per_prompt_byte_median") != 0.4:
            print("missing_index_tokens_per_prompt_byte=true", file=sys.stderr)
            return 1
        if qemu_artifact.get("serial_output_bytes_total") != 4096:
            print("missing_index_serial_output_bytes=true", file=sys.stderr)
            return 1
        if report["dry_run_coverage_violations"]:
            print("unexpected_dry_run_coverage_violations=true", file=sys.stderr)
            return 1
        if report["history_coverage_violations"]:
            print("unexpected_history_coverage_violations=true", file=sys.stderr)
            return 1
        if any(row["command_airgap_status"] != "pass" for row in report["artifacts"]):
            print("safe_airgap_status_not_pass=true", file=sys.stderr)
            return 1
        if "Benchmark Result Index" not in markdown_path.read_text(encoding="utf-8"):
            print("missing_index_markdown=true", file=sys.stderr)
            return 1
        if "source,artifact_type,status" not in csv_path.read_text(encoding="utf-8"):
            print("missing_index_csv=true", file=sys.stderr)
            return 1
        if "expected_token_prompts,expected_tokens_total,expected_tokens_matches,expected_tokens_mismatches" not in csv_path.read_text(encoding="utf-8"):
            print("missing_index_expected_token_csv=true", file=sys.stderr)
            return 1
        if "measured_prompt_bytes_total,prompt_bytes_min,prompt_bytes_max,prompt_bytes_per_s_median,wall_prompt_bytes_per_s_median,tokens_per_prompt_byte_median" not in csv_path.read_text(encoding="utf-8"):
            print("missing_index_prompt_efficiency_csv=true", file=sys.stderr)
            return 1
        if "memory_bytes_per_token_median,memory_bytes_per_token_max,serial_output_bytes_total,serial_output_bytes_max" not in csv_path.read_text(encoding="utf-8"):
            print("missing_index_resource_density_csv=true", file=sys.stderr)
            return 1
        if "key,history_count,source" not in latest_csv_path.read_text(encoding="utf-8"):
            print("missing_latest_comparable_csv=true", file=sys.stderr)
            return 1
        if "memory_bytes_per_token_median,memory_bytes_per_token_max,serial_output_bytes_total,serial_output_bytes_max" not in latest_csv_path.read_text(encoding="utf-8"):
            print("missing_latest_resource_density_csv=true", file=sys.stderr)
            return 1
        if "expected_token_prompts,expected_tokens_total,expected_tokens_matches,expected_tokens_mismatches" not in latest_csv_path.read_text(encoding="utf-8"):
            print("missing_latest_expected_token_csv=true", file=sys.stderr)
            return 1
        if "measured_prompt_bytes_total,prompt_bytes_min,prompt_bytes_max,prompt_bytes_per_s_median,wall_prompt_bytes_per_s_median,tokens_per_prompt_byte_median" not in latest_csv_path.read_text(encoding="utf-8"):
            print("missing_latest_prompt_efficiency_csv=true", file=sys.stderr)
            return 1
        if "key,hash_count,source_count" not in launch_plan_drift_csv_path.read_text(encoding="utf-8"):
            print("missing_launch_plan_drift_csv=true", file=sys.stderr)
            return 1
        if "key,measured_source,generated_at" not in dry_run_coverage_csv_path.read_text(encoding="utf-8"):
            print("missing_dry_run_coverage_csv=true", file=sys.stderr)
            return 1
        if "key,history_count,min_history" not in history_coverage_csv_path.read_text(encoding="utf-8"):
            print("missing_history_coverage_csv=true", file=sys.stderr)
            return 1
        if (
            "source,artifact_type,generated_at,generated_age_seconds,freshness_status,freshness_findings"
            not in freshness_failures_csv_path.read_text(encoding="utf-8")
        ):
            print("missing_index_freshness_failures_csv=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(junit_path).getroot()
        if junit_root.attrib.get("name") != "holyc_bench_result_index":
            print("missing_index_junit_suite=true", file=sys.stderr)
            return 1
        if junit_root.attrib.get("failures") != "0":
            print("unexpected_index_junit_failures=true", file=sys.stderr)
            return 1

        bad_hash_source = tmp_path / "bad_hash_source"
        bad_hash_source.mkdir()
        write_qemu_report(
            bad_hash_source / "qemu_prompt_bench_bad_hash.json",
            safe_command,
            command_hash="not-the-command-hash",
        )
        bad_hash_output = tmp_path / "bad_hash_index"
        completed = run_index(bad_hash_source, bad_hash_output, "--fail-on-command-hash-metadata")
        if completed.returncode == 0:
            print("bad_hash_index_not_rejected=true", file=sys.stderr)
            return 1
        bad_hash_report = json.loads(
            (bad_hash_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        if bad_hash_report["artifacts"][0]["command_hash_status"] != "fail":
            print("bad_hash_status_not_fail=true", file=sys.stderr)
            return 1

        bad_launch_plan_source = tmp_path / "bad_launch_plan_source"
        bad_launch_plan_source.mkdir()
        write_dry_run_report(
            bad_launch_plan_source / "qemu_prompt_bench_dry_run_bad_launch_plan.json",
            safe_command,
        )
        bad_launch_plan_path = bad_launch_plan_source / "qemu_prompt_bench_dry_run_bad_launch_plan.json"
        bad_launch_plan_report = json.loads(bad_launch_plan_path.read_text(encoding="utf-8"))
        bad_launch_plan_report["launch_plan_sha256"] = "0" * 64
        bad_launch_plan_path.write_text(
            json.dumps(bad_launch_plan_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bad_launch_plan_output = tmp_path / "bad_launch_plan_index"
        completed = run_index(
            bad_launch_plan_source,
            bad_launch_plan_output,
            "--fail-on-launch-plan-hash-metadata",
        )
        if completed.returncode == 0:
            print("bad_launch_plan_hash_not_rejected=true", file=sys.stderr)
            return 1
        bad_launch_plan_index = json.loads(
            (bad_launch_plan_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        if bad_launch_plan_index["artifacts"][0]["launch_plan_hash_status"] != "fail":
            print("bad_launch_plan_hash_status_not_fail=true", file=sys.stderr)
            return 1

        unsafe_source = tmp_path / "unsafe_source"
        unsafe_source.mkdir()
        unsafe_command = [
            "qemu-system-x86_64",
            "-display",
            "none",
            "-drive",
            "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
            "-device",
            "e1000",
        ]
        write_qemu_report(unsafe_source / "qemu_prompt_bench_unsafe.json", unsafe_command)
        unsafe_output = tmp_path / "unsafe_index"
        completed = run_index(unsafe_source, unsafe_output, "--fail-on-airgap")
        if completed.returncode == 0:
            print("unsafe_index_airgap_not_rejected=true", file=sys.stderr)
            return 1
        unsafe_report = json.loads(
            (unsafe_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        if unsafe_report["artifacts"][0]["command_airgap_status"] != "fail":
            print("unsafe_airgap_status_not_fail=true", file=sys.stderr)
            return 1

        stale_source = tmp_path / "stale_source"
        stale_source.mkdir()
        write_qemu_report(
            stale_source / "qemu_prompt_bench_stale.json",
            safe_command,
            generated_at="2026-04-27T00:00:00Z",
        )
        stale_output = tmp_path / "stale_index"
        completed = run_index(
            stale_source,
            stale_output,
            "--max-artifact-age-hours",
            "1",
            "--fail-on-stale-artifact",
        )
        if completed.returncode == 0:
            print("stale_index_not_rejected=true", file=sys.stderr)
            return 1
        stale_report = json.loads((stale_output / "bench_result_index_latest.json").read_text(encoding="utf-8"))
        if stale_report["artifacts"][0]["freshness_status"] != "fail":
            print("stale_status_not_fail=true", file=sys.stderr)
            return 1
        stale_csv = (
            stale_output / "bench_result_index_freshness_failures_latest.csv"
        ).read_text(encoding="utf-8")
        if "qemu_prompt_bench_stale.json" not in stale_csv or "freshness_status" not in stale_csv:
            print("missing_stale_index_freshness_failure_csv_row=true", file=sys.stderr)
            return 1

        token_mismatch_source = tmp_path / "token_mismatch_source"
        token_mismatch_source.mkdir()
        write_qemu_report(token_mismatch_source / "qemu_prompt_bench_token_mismatch.json", safe_command)
        token_mismatch_report = json.loads(
            (token_mismatch_source / "qemu_prompt_bench_token_mismatch.json").read_text(encoding="utf-8")
        )
        token_mismatch_report["benchmarks"][0]["expected_tokens_match"] = False
        (token_mismatch_source / "qemu_prompt_bench_token_mismatch.json").write_text(
            json.dumps(token_mismatch_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        token_mismatch_output = tmp_path / "token_mismatch_index"
        completed = run_index(token_mismatch_source, token_mismatch_output, "--fail-on-telemetry")
        if completed.returncode == 0:
            print("expected_token_mismatch_not_rejected=true", file=sys.stderr)
            return 1
        token_mismatch_index = json.loads(
            (token_mismatch_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        if token_mismatch_index["artifacts"][0]["expected_tokens_mismatches"] != 1:
            print("expected_token_mismatch_count_missing=true", file=sys.stderr)
            return 1

        drift_source = tmp_path / "drift_source"
        drift_source.mkdir()
        write_qemu_report(
            drift_source / "qemu_prompt_bench_env_a.json",
            safe_command,
            generated_at="2026-04-29T00:00:00Z",
        )
        write_qemu_report(
            drift_source / "qemu_prompt_bench_env_b.json",
            safe_command,
            generated_at="2026-04-29T00:01:00Z",
            environment={
                "platform": "ci-smoke",
                "machine": "other-host",
                "qemu_bin": safe_command[0],
                "qemu_version": "synthetic",
            },
        )
        drift_output = tmp_path / "drift_index"
        completed = run_index(drift_source, drift_output, "--fail-on-environment-drift")
        if completed.returncode == 0:
            print("environment_drift_not_rejected=true", file=sys.stderr)
            return 1
        drift_report = json.loads((drift_output / "bench_result_index_latest.json").read_text(encoding="utf-8"))
        if len(drift_report["environment_drift"]) != 1:
            print("unexpected_environment_drift_count=true", file=sys.stderr)
            return 1
        expected_hashes = {
            environment_sha256(
                {
                    "platform": "ci-smoke",
                    "machine": "host",
                    "qemu_bin": safe_command[0],
                    "qemu_version": "synthetic",
                }
            ),
            environment_sha256(
                {
                    "platform": "ci-smoke",
                    "machine": "other-host",
                    "qemu_bin": safe_command[0],
                    "qemu_version": "synthetic",
                }
            ),
        }
        if set(drift_report["environment_drift"][0]["hashes"]) != expected_hashes:
            print("environment_drift_hashes_mismatch=true", file=sys.stderr)
            return 1

        missing_dry_run_source = tmp_path / "missing_dry_run_source"
        missing_dry_run_source.mkdir()
        write_qemu_report(missing_dry_run_source / "qemu_prompt_bench_measured_only.json", safe_command)
        missing_dry_run_output = tmp_path / "missing_dry_run_index"
        completed = run_index(missing_dry_run_source, missing_dry_run_output, "--fail-on-missing-dry-run")
        if completed.returncode == 0:
            print("missing_dry_run_not_rejected=true", file=sys.stderr)
            return 1
        missing_dry_run_report = json.loads(
            (missing_dry_run_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        violations = missing_dry_run_report["dry_run_coverage_violations"]
        if len(violations) != 1:
            print("unexpected_missing_dry_run_violation_count=true", file=sys.stderr)
            return 1
        if "qemu_prompt_bench_measured_only.json" not in violations[0]["measured_source"]:
            print("missing_dry_run_source_not_reported=true", file=sys.stderr)
            return 1
        missing_dry_run_junit = ET.parse(
            missing_dry_run_output / "bench_result_index_junit_latest.xml"
        ).getroot()
        if missing_dry_run_junit.attrib.get("failures") == "0":
            print("missing_dry_run_junit_not_failed=true", file=sys.stderr)
            return 1
        dry_run_case = missing_dry_run_junit.find("./testcase[@name='dry_run_coverage']")
        if dry_run_case is None or dry_run_case.find("failure") is None:
            print("missing_dry_run_junit_failure_case=true", file=sys.stderr)
            return 1

        history_source = tmp_path / "history_source"
        history_source.mkdir()
        write_qemu_report(history_source / "qemu_prompt_bench_single_history.json", safe_command)
        history_output = tmp_path / "history_index"
        completed = run_index(
            history_source,
            history_output,
            "--min-history-per-key",
            "2",
            "--fail-on-history-coverage",
        )
        if completed.returncode == 0:
            print("history_coverage_not_rejected=true", file=sys.stderr)
            return 1
        history_report = json.loads(
            (history_output / "bench_result_index_latest.json").read_text(encoding="utf-8")
        )
        history_violations = history_report["history_coverage_violations"]
        if len(history_violations) != 1 or history_violations[0]["history_count"] != 1:
            print("unexpected_history_coverage_violation=true", file=sys.stderr)
            return 1
        history_junit = ET.parse(history_output / "bench_result_index_junit_latest.xml").getroot()
        history_case = history_junit.find("./testcase[@name='history_coverage']")
        if history_case is None or history_case.find("failure") is None:
            print("missing_history_coverage_junit_failure_case=true", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
