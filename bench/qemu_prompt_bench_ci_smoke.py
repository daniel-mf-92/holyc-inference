#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_prompt_bench.py artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "bench" / "prompts" / "smoke.jsonl"
SYNTHETIC_QEMU = ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"
SYNTHETIC_IMAGE = Path("/tmp/TempleOS.synthetic.img")
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_bench


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    planned = qemu_prompt_bench.dry_run_launch_plan(
        qemu_prompt_bench.load_prompt_cases(PROMPTS),
        warmup=0,
        repeat=1,
    )
    expected_sequence = qemu_prompt_bench.launch_sequence_from_plan(planned)
    observed_sequence = [dict(row) for row in expected_sequence]
    observed_sequence[0]["prompt_bytes"] += 1
    integrity = qemu_prompt_bench.launch_sequence_integrity(expected_sequence, observed_sequence)
    if rc := require(integrity["launch_sequence_match"] is False, "expected_launch_sequence_negative_mismatch"):
        return rc
    if rc := require(
        qemu_prompt_bench.launch_sequence_findings(integrity),
        "missing_launch_sequence_negative_finding",
    ):
        return rc

    with tempfile.TemporaryDirectory(prefix="holyc-qemu-bench-ci-") as tmp:
        output_dir = Path(tmp) / "results"
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            str(SYNTHETIC_IMAGE),
            "--prompts",
            str(PROMPTS),
            "--qemu-bin",
            str(SYNTHETIC_QEMU),
            "--warmup",
            "1",
            "--repeat",
            "2",
            "--timeout",
            "5",
            "--output-dir",
            str(output_dir),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--require-tokens",
            "--require-tok-per-s",
            "--require-memory",
            "--require-ttft-us",
            "--require-guest-prompt-sha256-match",
            "--require-guest-prompt-bytes-match",
            "--require-expected-tokens-match",
            "--max-launches",
            "6",
            "--min-prompt-count",
            "2",
            "--max-serial-output-lines",
            "1",
            "--qemu-arg=-m",
            "--qemu-arg=256M",
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

        report_path = output_dir / "qemu_prompt_bench_latest.json"
        phase_csv_path = output_dir / "qemu_prompt_bench_phases_latest.csv"
        markdown_path = output_dir / "qemu_prompt_bench_latest.md"
        launch_csv_path = output_dir / "qemu_prompt_bench_launches_latest.csv"
        junit_path = output_dir / "qemu_prompt_bench_junit_latest.xml"
        report = json.loads(report_path.read_text(encoding="utf-8"))

        if rc := require(report["status"] == "pass", "unexpected_qemu_bench_status"):
            return rc
        if rc := require(report["profile"] == "ci-airgap-smoke", "missing_top_level_profile"):
            return rc
        if rc := require(report["model"] == "synthetic-smoke", "missing_top_level_model"):
            return rc
        if rc := require(report["quantization"] == "Q4_0", "missing_top_level_quantization"):
            return rc
        if rc := require(report["commit"], "missing_top_level_commit"):
            return rc
        if rc := require(report["command"] == report["benchmarks"][0]["command"], "top_level_command_mismatch"):
            return rc
        if rc := require(
            report["command_sha256"] == report["benchmarks"][0]["command_sha256"],
            "top_level_command_sha256_mismatch",
        ):
            return rc
        if rc := require(
            report["expected_launch_sequence_sha256"] == report["observed_launch_sequence_sha256"],
            "launch_sequence_hash_mismatch",
        ):
            return rc
        if rc := require(
            report["launch_sequence_integrity"]["launch_sequence_match"] is True,
            "launch_sequence_integrity_mismatch",
        ):
            return rc
        if rc := require(
            report["launch_sequence_integrity"]["observed_launches"] == 6,
            "unexpected_observed_launch_count",
        ):
            return rc
        if rc := require(not report["launch_sequence_findings"], "unexpected_launch_sequence_findings"):
            return rc
        if rc := require(report["planned_warmup_launches"] == 2, "unexpected_warmup_launches"):
            return rc
        if rc := require(report["planned_measured_launches"] == 4, "unexpected_measured_launches"):
            return rc
        if rc := require("-nic" in report["warmups"][0]["command"], "missing_airgap_nic_flag"):
            return rc
        if rc := require("none" in report["warmups"][0]["command"], "missing_airgap_nic_none"):
            return rc
        if rc := require(report["suite_summary"]["total_tokens"] == 160, "unexpected_suite_tokens"):
            return rc
        if rc := require(report["suite_summary"]["ok_run_pct"] == 100.0, "unexpected_suite_ok_pct"):
            return rc
        if rc := require(
            report["suite_summary"]["guest_prompt_sha256_matches"] == 4,
            "unexpected_suite_guest_prompt_sha256_matches",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["guest_prompt_bytes_matches"] == 4,
            "unexpected_suite_guest_prompt_bytes_matches",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["serial_output_lines_total"] == 4,
            "unexpected_suite_serial_output_lines",
        ):
            return rc
        if rc := require(
            report["telemetry_gates"]["max_serial_output_lines"] == 1,
            "missing_serial_output_line_gate",
        ):
            return rc

        phases = {row["phase"]: row for row in report["phase_summaries"]}
        if rc := require(set(phases) == {"warmup", "measured", "all"}, "unexpected_phase_rows"):
            return rc
        if rc := require(phases["warmup"]["launches"] == 2, "unexpected_warmup_phase_launches"):
            return rc
        if rc := require(phases["warmup"]["total_tokens"] == 80, "unexpected_warmup_phase_tokens"):
            return rc
        if rc := require(phases["measured"]["launches"] == 4, "unexpected_measured_phase_launches"):
            return rc
        if rc := require(phases["measured"]["total_tokens"] == 160, "unexpected_measured_phase_tokens"):
            return rc
        if rc := require(
            phases["measured"]["serial_output_lines_total"] == 4,
            "unexpected_measured_phase_serial_output_lines",
        ):
            return rc
        if rc := require(
            phases["measured"]["guest_prompt_sha256_mismatches"] == 0,
            "unexpected_measured_prompt_sha256_mismatches",
        ):
            return rc
        if rc := require(
            phases["measured"]["guest_prompt_bytes_mismatches"] == 0,
            "unexpected_measured_prompt_bytes_mismatches",
        ):
            return rc
        if rc := require(phases["all"]["launches"] == 6, "unexpected_all_phase_launches"):
            return rc
        if rc := require(phases["all"]["total_tokens"] == 240, "unexpected_all_phase_tokens"):
            return rc

        phase_rows = list(csv.DictReader(phase_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(phase_rows) == 3, "unexpected_phase_csv_rows"):
            return rc
        if rc := require(
            {
                "phase",
                "launches",
                "total_tokens",
                "tok_per_s_median",
                "serial_output_bytes_total",
                "serial_output_lines_total",
            }.issubset(phase_rows[0].keys()),
            "missing_phase_csv_columns",
        ):
            return rc
        if rc := require(
            "guest_prompt_sha256_mismatches" in phase_rows[0],
            "missing_phase_prompt_integrity_columns",
        ):
            return rc

        launch_rows = list(csv.DictReader(launch_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(launch_rows) == 6, "unexpected_launch_csv_rows"):
            return rc
        if rc := require(
            {row["phase"] for row in launch_rows} == {"warmup", "measured"},
            "unexpected_launch_csv_phases",
        ):
            return rc
        if rc := require(
            {
                "expected_launch_sequence_sha256",
                "observed_launch_sequence_sha256",
                "serial_output_lines",
            }.issubset(launch_rows[0].keys()),
            "missing_launch_sequence_csv_columns",
        ):
            return rc
        if rc := require("Phase Summary" in markdown_path.read_text(encoding="utf-8"), "missing_phase_markdown"):
            return rc
        if rc := require(
            "Launch Sequence Integrity" in markdown_path.read_text(encoding="utf-8"),
            "missing_launch_sequence_markdown",
        ):
            return rc
        junit_root = ET.parse(junit_path).getroot()
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        dry_output_dir = Path(tmp) / "dry_results"
        dry_command = command + ["--dry-run", "--output-dir", str(dry_output_dir)]
        completed = subprocess.run(
            dry_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        dry_report = json.loads((dry_output_dir / "qemu_prompt_bench_dry_run_latest.json").read_text(encoding="utf-8"))
        if rc := require(dry_report["profile"] == "ci-airgap-smoke", "missing_dry_run_profile"):
            return rc
        if rc := require(dry_report["model"] == "synthetic-smoke", "missing_dry_run_model"):
            return rc
        if rc := require(dry_report["quantization"] == "Q4_0", "missing_dry_run_quantization"):
            return rc
        if rc := require(dry_report["commit"], "missing_dry_run_commit"):
            return rc
        if rc := require(dry_report["expected_launch_sequence_sha256"], "missing_dry_run_launch_sequence_hash"):
            return rc
        dry_csv_rows = list(csv.DictReader((dry_output_dir / "qemu_prompt_bench_dry_run_latest.csv").open(encoding="utf-8", newline="")))
        if rc := require(dry_csv_rows[0]["profile"] == "ci-airgap-smoke", "missing_dry_run_csv_profile"):
            return rc
        if rc := require(
            dry_csv_rows[0]["expected_launch_sequence_sha256"] == dry_report["expected_launch_sequence_sha256"],
            "dry_run_csv_launch_sequence_hash_mismatch",
        ):
            return rc
        dry_junit_root = ET.parse(dry_output_dir / "qemu_prompt_bench_dry_run_junit_latest.xml").getroot()
        dry_properties = {
            item.attrib.get("name"): item.attrib.get("value")
            for item in dry_junit_root.findall(".//property")
        }
        if rc := require(dry_properties.get("commit"), "missing_dry_run_junit_commit"):
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
