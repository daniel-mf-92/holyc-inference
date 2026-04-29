#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for benchmark artifact manifests."""

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


def write_qemu_report(
    path: Path,
    command: list[str],
    generated_at: str | None = None,
    qemu_version: str = "synthetic",
) -> None:
    command_hash = command_sha256(command)
    generated_at = generated_at or iso_now()
    path.write_text(
        json.dumps(
            {
                "status": "pass",
                "generated_at": generated_at,
                "command_sha256": command_hash,
                "prompt_suite": {
                    "name": "manifest-smoke",
                    "prompt_count": 1,
                    "suite_sha256": "manifest-smoke-suite",
                },
                "environment": {
                    "platform": "ci-smoke",
                    "machine": "host",
                    "qemu_bin": command[0],
                    "qemu_version": qemu_version,
                },
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "command": command,
                        "command_sha256": command_hash,
                        "commit": "manifest-smoke",
                        "profile": "ci-airgap-smoke",
                        "model": "synthetic-smoke",
                        "quantization": "Q4_0",
                        "prompt": "manifest-smoke",
                        "tokens": 32,
                        "elapsed_us": 200000,
                        "tok_per_s": 160.0,
                        "memory_bytes_per_token": 2099200.0,
                        "serial_output_bytes": 4096,
                        "returncode": 0,
                        "timed_out": False,
                    }
                ],
                "summaries": [
                    {
                        "prompt": "manifest-smoke",
                        "tok_per_s_median": 160.0,
                        "wall_tok_per_s_median": 150.0,
                        "ttft_us_p95": 12000.0,
                        "memory_bytes_per_token_median": 2099200.0,
                        "memory_bytes_per_token_max": 2099200.0,
                        "serial_output_bytes_total": 4096,
                        "serial_output_bytes_max": 4096,
                        "memory_bytes_max": 67174400,
                    }
                ],
                "suite_summary": {
                    "prompts": 1,
                    "total_tokens": 32,
                    "total_elapsed_us": 200000,
                    "tok_per_s_median": 160.0,
                    "wall_tok_per_s_median": 150.0,
                    "ttft_us_p95": 12000.0,
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


def write_dry_run_report(path: Path, command: list[str], generated_at: str | None = None) -> None:
    command_hash = command_sha256(command)
    generated_at = generated_at or iso_now()
    path.write_text(
        json.dumps(
            {
                "status": "planned",
                "generated_at": generated_at,
                "command": command,
                "command_sha256": command_hash,
                "commit": "manifest-smoke",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "prompt_suite": {
                    "prompt_count": 1,
                    "suite_sha256": "manifest-smoke-suite",
                },
                "launch_plan_sha256": "",
                "planned_warmup_launches": 0,
                "planned_measured_launches": 1,
                "planned_total_launches": 1,
                "launch_plan": [{"phase": "measured", "prompt_id": "manifest-smoke"}],
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


def run_manifest(input_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "bench_artifact_manifest.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-manifest-ci-") as tmp:
        tmp_path = Path(tmp)
        safe_report = tmp_path / "qemu_prompt_bench_safe.json"
        safe_command = [
            "qemu-system-x86_64",
            "-nic",
            "none",
            "-display",
            "none",
            "-drive",
            "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
        ]
        write_qemu_report(safe_report, safe_command)

        safe_output_dir = tmp_path / "safe_manifest"
        completed = run_manifest(
            safe_report,
            safe_output_dir,
            "--fail-on-airgap",
            "--fail-on-telemetry",
            "--fail-on-command-hash-metadata",
            "--fail-on-commit-metadata",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report_path = safe_output_dir / "bench_artifact_manifest_latest.json"
        markdown_path = safe_output_dir / "bench_artifact_manifest_latest.md"
        csv_path = safe_output_dir / "bench_artifact_manifest_latest.csv"
        history_csv_path = safe_output_dir / "bench_artifact_manifest_history_latest.csv"
        history_coverage_csv_path = safe_output_dir / "bench_artifact_manifest_history_coverage_latest.csv"
        sample_coverage_csv_path = safe_output_dir / "bench_artifact_manifest_sample_coverage_latest.csv"
        dry_run_coverage_csv_path = safe_output_dir / "bench_artifact_manifest_dry_run_coverage_latest.csv"
        environment_drift_csv_path = safe_output_dir / "bench_artifact_manifest_environment_drift_latest.csv"
        junit_path = safe_output_dir / "bench_artifact_manifest_junit_latest.xml"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "pass":
            print(f"unexpected_manifest_status={report['status']}", file=sys.stderr)
            return 1
        if len(report["latest_artifacts"]) != 1 or len(report["history"]) != 1:
            print("unexpected_manifest_artifact_counts=true", file=sys.stderr)
            return 1
        artifact = report["latest_artifacts"][0]
        if artifact.get("command_airgap_status") != "pass":
            print("safe_artifact_airgap_not_pass=true", file=sys.stderr)
            return 1
        if artifact.get("telemetry_status") != "pass":
            print("safe_artifact_telemetry_not_pass=true", file=sys.stderr)
            return 1
        if artifact.get("command_hash_status") != "pass":
            print("safe_artifact_command_hash_not_pass=true", file=sys.stderr)
            return 1
        if artifact.get("sha256") != hashlib.sha256(safe_report.read_bytes()).hexdigest():
            print("manifest_artifact_hash_mismatch=true", file=sys.stderr)
            return 1
        if artifact.get("memory_bytes_per_token_median") != 2099200.0:
            print("missing_manifest_memory_per_token=true", file=sys.stderr)
            return 1
        if artifact.get("serial_output_bytes_total") != 4096:
            print("missing_manifest_serial_output_bytes=true", file=sys.stderr)
            return 1
        if "Benchmark Artifact Manifest" not in markdown_path.read_text(encoding="utf-8"):
            print("missing_manifest_markdown=true", file=sys.stderr)
            return 1
        if "key,source,artifact_type,status" not in csv_path.read_text(encoding="utf-8"):
            print("missing_manifest_csv=true", file=sys.stderr)
            return 1
        if "memory_bytes_per_token_median,memory_bytes_per_token_max,serial_output_bytes_total,serial_output_bytes_max" not in csv_path.read_text(encoding="utf-8"):
            print("missing_manifest_resource_density_csv=true", file=sys.stderr)
            return 1
        if "key,source,artifact_type,status" not in history_csv_path.read_text(encoding="utf-8"):
            print("missing_manifest_history_csv=true", file=sys.stderr)
            return 1
        if (
            "key,history_count,required_history_count,sources"
            not in history_coverage_csv_path.read_text(encoding="utf-8")
        ):
            print("missing_manifest_history_coverage_csv=true", file=sys.stderr)
            return 1
        if (
            "key,source,metric,observed,required"
            not in sample_coverage_csv_path.read_text(encoding="utf-8")
        ):
            print("missing_manifest_sample_coverage_csv=true", file=sys.stderr)
            return 1
        if (
            "key,measured_source,generated_at"
            not in dry_run_coverage_csv_path.read_text(encoding="utf-8")
        ):
            print("missing_manifest_dry_run_coverage_csv=true", file=sys.stderr)
            return 1
        if (
            "key,environment_hashes,environment_hash_count,sources,source_count"
            not in environment_drift_csv_path.read_text(encoding="utf-8")
        ):
            print("missing_manifest_environment_drift_csv=true", file=sys.stderr)
            return 1
        junit_root = ET.parse(junit_path).getroot()
        if junit_root.attrib.get("name") != "holyc_bench_artifact_manifest":
            print("missing_manifest_junit_suite=true", file=sys.stderr)
            return 1
        if junit_root.attrib.get("failures") != "0":
            print("unexpected_manifest_junit_failures=true", file=sys.stderr)
            return 1

        coverage_source_dir = tmp_path / "coverage_sources"
        coverage_source_dir.mkdir()
        write_qemu_report(
            coverage_source_dir / "qemu_prompt_bench_old.json",
            safe_command,
            generated_at="2026-04-27T00:00:00Z",
        )
        write_qemu_report(
            coverage_source_dir / "qemu_prompt_bench_new.json",
            safe_command,
            generated_at="2026-04-28T00:00:00Z",
        )
        coverage_output_dir = tmp_path / "coverage_manifest"
        completed = run_manifest(
            coverage_source_dir,
            coverage_output_dir,
            "--min-history-per-key",
            "2",
            "--fail-on-history-coverage",
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        coverage_report = json.loads(
            (coverage_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        if coverage_report["history_coverage_violations"]:
            print("unexpected_history_coverage_violation=true", file=sys.stderr)
            return 1

        dry_run_source_dir = tmp_path / "dry_run_sources"
        dry_run_source_dir.mkdir()
        write_qemu_report(dry_run_source_dir / "qemu_prompt_bench_measured.json", safe_command)
        write_dry_run_report(dry_run_source_dir / "qemu_prompt_bench_dry_run.json", safe_command)
        dry_run_output_dir = tmp_path / "dry_run_manifest"
        completed = run_manifest(dry_run_source_dir, dry_run_output_dir, "--fail-on-missing-dry-run")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        dry_run_manifest = json.loads(
            (dry_run_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        if dry_run_manifest["dry_run_coverage_violations"]:
            print("unexpected_dry_run_coverage_violation=true", file=sys.stderr)
            return 1

        missing_dry_run_output_dir = tmp_path / "missing_dry_run_manifest"
        completed = run_manifest(safe_report, missing_dry_run_output_dir, "--fail-on-missing-dry-run")
        if completed.returncode == 0:
            print("missing_dry_run_manifest_not_rejected=true", file=sys.stderr)
            return 1
        missing_dry_run_manifest = json.loads(
            (missing_dry_run_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        dry_run_violations = missing_dry_run_manifest["dry_run_coverage_violations"]
        if (
            len(dry_run_violations) != 1
            or "qemu_prompt_bench_safe.json" not in dry_run_violations[0]["measured_source"]
        ):
            print("unexpected_dry_run_coverage_violation_payload=true", file=sys.stderr)
            return 1

        sparse_output_dir = tmp_path / "sparse_manifest"
        completed = run_manifest(
            safe_report,
            sparse_output_dir,
            "--min-history-per-key",
            "2",
            "--fail-on-history-coverage",
        )
        if completed.returncode == 0:
            print("sparse_manifest_history_coverage_not_rejected=true", file=sys.stderr)
            return 1
        sparse_report = json.loads(
            (sparse_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        violations = sparse_report["history_coverage_violations"]
        if len(violations) != 1 or violations[0]["history_count"] != 1:
            print("unexpected_history_coverage_violation_payload=true", file=sys.stderr)
            return 1

        sample_output_dir = tmp_path / "sample_manifest"
        completed = run_manifest(
            safe_report,
            sample_output_dir,
            "--min-measured-runs",
            "2",
            "--min-total-tokens",
            "64",
            "--fail-on-sample-coverage",
        )
        if completed.returncode == 0:
            print("sparse_manifest_sample_coverage_not_rejected=true", file=sys.stderr)
            return 1
        sample_report = json.loads(
            (sample_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        sample_violations = sample_report["sample_coverage_violations"]
        if (
            sample_report["status"] != "fail"
            or [row["metric"] for row in sample_violations] != ["measured_runs", "total_tokens"]
        ):
            print("unexpected_sample_coverage_violation_payload=true", file=sys.stderr)
            return 1
        sample_junit = ET.parse(sample_output_dir / "bench_artifact_manifest_junit_latest.xml").getroot()
        if sample_junit.attrib.get("failures") != "1":
            print("unexpected_sample_coverage_junit_failures=true", file=sys.stderr)
            return 1

        env_drift_source_dir = tmp_path / "env_drift_sources"
        env_drift_source_dir.mkdir()
        write_qemu_report(
            env_drift_source_dir / "qemu_prompt_bench_env_a.json",
            safe_command,
            generated_at="2026-04-27T00:00:00Z",
            qemu_version="synthetic-a",
        )
        write_qemu_report(
            env_drift_source_dir / "qemu_prompt_bench_env_b.json",
            safe_command,
            generated_at="2026-04-28T00:00:00Z",
            qemu_version="synthetic-b",
        )
        env_drift_output_dir = tmp_path / "env_drift_manifest"
        completed = run_manifest(env_drift_source_dir, env_drift_output_dir, "--fail-on-environment-drift")
        if completed.returncode == 0:
            print("environment_drift_manifest_not_rejected=true", file=sys.stderr)
            return 1
        env_drift_report = json.loads(
            (env_drift_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        environment_drift = env_drift_report["environment_drift"]
        if (
            env_drift_report["status"] != "fail"
            or len(environment_drift) != 1
            or len(environment_drift[0]["hashes"]) != 2
            or len(environment_drift[0]["sources"]) != 2
        ):
            print("unexpected_environment_drift_payload=true", file=sys.stderr)
            return 1
        env_drift_junit = ET.parse(env_drift_output_dir / "bench_artifact_manifest_junit_latest.xml").getroot()
        if env_drift_junit.attrib.get("failures") != "1":
            print("unexpected_environment_drift_junit_failures=true", file=sys.stderr)
            return 1

        unsafe_report = tmp_path / "qemu_prompt_bench_unsafe.json"
        unsafe_command = [
            "qemu-system-x86_64",
            "-display",
            "none",
            "-drive",
            "file=/tmp/TempleOS.synthetic.img,format=raw,if=ide",
            "-device",
            "e1000",
        ]
        write_qemu_report(unsafe_report, unsafe_command)
        unsafe_output_dir = tmp_path / "unsafe_manifest"
        completed = run_manifest(unsafe_report, unsafe_output_dir, "--fail-on-airgap")
        if completed.returncode == 0:
            print("unsafe_manifest_airgap_not_rejected=true", file=sys.stderr)
            return 1
        unsafe_manifest = json.loads(
            (unsafe_output_dir / "bench_artifact_manifest_latest.json").read_text(encoding="utf-8")
        )
        unsafe_artifact = unsafe_manifest["history"][0]
        if unsafe_artifact.get("command_airgap_status") != "fail":
            print("unsafe_artifact_airgap_not_fail=true", file=sys.stderr)
            return 1
        findings = "\n".join(unsafe_artifact.get("command_hash_findings", []))
        if findings:
            print("unsafe_artifact_command_hash_unexpected_fail=true", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
