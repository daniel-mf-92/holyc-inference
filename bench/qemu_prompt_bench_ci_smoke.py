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
    if rc := require(
        qemu_prompt_bench.command_airgap_metadata(["qemu-system-x86_64", "-nic", "none"])["ok"] is True,
        "explicit_nic_none_airgap_metadata_failed",
    ):
        return rc
    missing_nic_metadata = qemu_prompt_bench.command_airgap_metadata(["qemu-system-x86_64", "-net", "none"])
    if rc := require(missing_nic_metadata["ok"] is False, "missing_nic_none_airgap_metadata_passed"):
        return rc
    if rc := require(
        "missing explicit `-nic none`" in missing_nic_metadata["violations"],
        "missing_nic_none_airgap_violation_detail",
    ):
        return rc
    redundant_legacy_net_metadata = qemu_prompt_bench.command_airgap_metadata(
        ["qemu-system-x86_64", "-nic", "none", "-net", "none"]
    )
    if rc := require(
        redundant_legacy_net_metadata["ok"] is False,
        "legacy_net_none_with_nic_airgap_metadata_passed",
    ):
        return rc
    if rc := require(
        "legacy `-net none` present; benchmark artifacts must use `-nic none`"
        in redundant_legacy_net_metadata["violations"],
        "legacy_net_none_airgap_violation_detail",
    ):
        return rc
    nic_user_metadata = qemu_prompt_bench.command_airgap_metadata(["qemu-system-x86_64", "-nic", "user"])
    if rc := require(nic_user_metadata["ok"] is False, "network_nic_airgap_metadata_passed"):
        return rc
    if rc := require(
        "non-air-gapped `-nic user`" in nic_user_metadata["violations"],
        "network_nic_airgap_violation_detail",
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
            "--require-expected-tokens",
            "--require-expected-tokens-match",
            "--max-launches",
            "6",
            "--min-prompt-count",
            "2",
            "--max-serial-output-lines",
            "1",
            "--max-wall-elapsed-us",
            "5000000",
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
        prompt_rank_csv_path = output_dir / "qemu_prompt_bench_prompt_rank_latest.csv"
        prompt_variability_csv_path = output_dir / "qemu_prompt_bench_prompt_variability_latest.csv"
        prompt_efficiency_csv_path = output_dir / "qemu_prompt_bench_prompt_efficiency_latest.csv"
        prompt_serial_output_csv_path = output_dir / "qemu_prompt_bench_prompt_serial_output_latest.csv"
        launch_csv_path = output_dir / "qemu_prompt_bench_launches_latest.csv"
        launch_jsonl_path = output_dir / "qemu_prompt_bench_launches_latest.jsonl"
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
        if rc := require(report["command_airgap"]["ok"] is True, "command_airgap_not_ok"):
            return rc
        if rc := require(
            report["command_airgap"]["explicit_nic_none"] is True,
            "missing_command_airgap_nic_none",
        ):
            return rc
        if rc := require(report["benchmarks"][0]["command_airgap_ok"] is True, "run_airgap_not_ok"):
            return rc
        if rc := require(
            report["benchmarks"][0]["command_has_explicit_nic_none"] is True,
            "run_missing_explicit_nic_none",
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
        if rc := require(report["suite_summary"]["exit_class_ok_runs"] == 4, "unexpected_suite_exit_ok"):
            return rc
        if rc := require(
            report["suite_summary"]["exit_class_timeout_runs"] == 0,
            "unexpected_suite_exit_timeout",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["exit_class_launch_error_runs"] == 0,
            "unexpected_suite_exit_launch_error",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["exit_class_nonzero_exit_runs"] == 0,
            "unexpected_suite_exit_nonzero",
        ):
            return rc
        if rc := require(
            report["telemetry_gates"]["max_serial_output_lines"] == 1,
            "missing_serial_output_line_gate",
        ):
            return rc
        if rc := require(
            report["telemetry_gates"]["require_expected_tokens"] is True,
            "missing_expected_tokens_gate",
        ):
            return rc
        if rc := require(
            report["telemetry_gates"]["max_wall_elapsed_us"] == 5000000,
            "missing_wall_elapsed_gate",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["total_wall_elapsed_us"] is not None,
            "missing_suite_total_wall_elapsed",
        ):
            return rc
        if rc := require(
            report["suite_summary"]["wall_elapsed_us_median"] is not None,
            "missing_suite_wall_elapsed_median",
        ):
            return rc
        if rc := require(
            report["summaries"][0]["wall_elapsed_us_p95"] is not None,
            "missing_prompt_wall_elapsed_p95",
        ):
            return rc
        if rc := require(len(report["prompt_rankings"]) == 2, "unexpected_prompt_rank_count"):
            return rc
        if rc := require(
            report["prompt_rankings"][0]["slowest_rank"] == 1,
            "missing_first_prompt_rank",
        ):
            return rc
        if rc := require(
            report["prompt_rankings"][0]["wall_us_per_token_median"] is not None,
            "missing_prompt_rank_wall_us_per_token",
        ):
            return rc
        if rc := require(
            len(report["prompt_variability_rankings"]) == 2,
            "unexpected_prompt_variability_rank_count",
        ):
            return rc
        if rc := require(
            report["prompt_variability_rankings"][0]["variability_rank"] == 1,
            "missing_first_prompt_variability_rank",
        ):
            return rc
        if rc := require(
            report["prompt_variability_rankings"][0]["wall_tok_per_s_iqr_pct"] is not None,
            "missing_prompt_variability_wall_iqr",
        ):
            return rc
        if rc := require(
            len(report["prompt_efficiency_rankings"]) == 2,
            "unexpected_prompt_efficiency_rank_count",
        ):
            return rc
        if rc := require(
            report["prompt_efficiency_rankings"][0]["efficiency_rank"] == 1,
            "missing_first_prompt_efficiency_rank",
        ):
            return rc
        if rc := require(
            report["prompt_efficiency_rankings"][0]["wall_prompt_bytes_per_s_median"] is not None,
            "missing_prompt_efficiency_wall_prompt_bytes_per_s",
        ):
            return rc
        if rc := require(
            len(report["prompt_serial_output_rankings"]) == 2,
            "unexpected_prompt_serial_output_rank_count",
        ):
            return rc
        if rc := require(
            report["prompt_serial_output_rankings"][0]["serial_output_rank"] == 1,
            "missing_first_prompt_serial_output_rank",
        ):
            return rc
        if rc := require(
            report["prompt_serial_output_rankings"][0]["serial_output_bytes_total"] is not None,
            "missing_prompt_serial_output_bytes_total",
        ):
            return rc

        phases = {row["phase"]: row for row in report["phase_summaries"]}
        if rc := require(set(phases) == {"warmup", "measured", "all"}, "unexpected_phase_rows"):
            return rc
        if rc := require(phases["warmup"]["launches"] == 2, "unexpected_warmup_phase_launches"):
            return rc
        if rc := require(phases["warmup"]["total_tokens"] == 80, "unexpected_warmup_phase_tokens"):
            return rc
        if rc := require(phases["warmup"]["exit_class_ok_runs"] == 2, "unexpected_warmup_exit_ok"):
            return rc
        if rc := require(phases["measured"]["launches"] == 4, "unexpected_measured_phase_launches"):
            return rc
        if rc := require(phases["measured"]["total_tokens"] == 160, "unexpected_measured_phase_tokens"):
            return rc
        if rc := require(phases["measured"]["exit_class_ok_runs"] == 4, "unexpected_measured_exit_ok"):
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
        if rc := require(
            phases["measured"]["wall_elapsed_us_median"] is not None,
            "missing_measured_wall_elapsed_median",
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
                "wall_elapsed_us_median",
                "wall_elapsed_us_p95",
                "serial_output_bytes_total",
                "serial_output_lines_total",
                "exit_class_ok_runs",
                "exit_class_timeout_runs",
                "exit_class_launch_error_runs",
                "exit_class_nonzero_exit_runs",
            }.issubset(phase_rows[0].keys()),
            "missing_phase_csv_columns",
        ):
            return rc
        summary_rows = list(csv.DictReader((output_dir / "qemu_prompt_bench_summary_latest.csv").open(encoding="utf-8", newline="")))
        if rc := require(summary_rows[0]["total_wall_elapsed_us"], "missing_summary_csv_wall_elapsed"):
            return rc
        if rc := require(summary_rows[0]["exit_class_ok_runs"] == "4", "missing_summary_exit_ok"):
            return rc
        if rc := require(
            "guest_prompt_sha256_mismatches" in phase_rows[0],
            "missing_phase_prompt_integrity_columns",
        ):
            return rc
        prompt_rank_rows = list(csv.DictReader(prompt_rank_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(prompt_rank_rows) == 2, "unexpected_prompt_rank_csv_rows"):
            return rc
        if rc := require(prompt_rank_rows[0]["slowest_rank"] == "1", "unexpected_prompt_rank_csv_rank"):
            return rc
        if rc := require(
            {
                "prompt_suite_sha256",
                "command_sha256",
                "wall_us_per_token_median",
                "wall_tok_per_s_median",
                "ttft_us_p95",
            }.issubset(prompt_rank_rows[0].keys()),
            "missing_prompt_rank_csv_columns",
        ):
            return rc
        prompt_variability_rows = list(csv.DictReader(prompt_variability_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(prompt_variability_rows) == 2, "unexpected_prompt_variability_csv_rows"):
            return rc
        if rc := require(
            prompt_variability_rows[0]["variability_rank"] == "1",
            "unexpected_prompt_variability_csv_rank",
        ):
            return rc
        if rc := require(
            {
                "prompt_suite_sha256",
                "command_sha256",
                "wall_tok_per_s_iqr_pct",
                "wall_tok_per_s_p05_p95_spread_pct",
                "tok_per_s_cv_pct",
            }.issubset(prompt_variability_rows[0].keys()),
            "missing_prompt_variability_csv_columns",
        ):
            return rc
        prompt_efficiency_rows = list(csv.DictReader(prompt_efficiency_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(prompt_efficiency_rows) == 2, "unexpected_prompt_efficiency_csv_rows"):
            return rc
        if rc := require(
            prompt_efficiency_rows[0]["efficiency_rank"] == "1",
            "unexpected_prompt_efficiency_csv_rank",
        ):
            return rc
        if rc := require(
            {
                "prompt_suite_sha256",
                "command_sha256",
                "wall_prompt_bytes_per_s_median",
                "tokens_per_prompt_byte_median",
                "wall_tok_per_s_median",
            }.issubset(prompt_efficiency_rows[0].keys()),
            "missing_prompt_efficiency_csv_columns",
        ):
            return rc
        prompt_serial_output_rows = list(csv.DictReader(prompt_serial_output_csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(prompt_serial_output_rows) == 2, "unexpected_prompt_serial_output_csv_rows"):
            return rc
        if rc := require(
            prompt_serial_output_rows[0]["serial_output_rank"] == "1",
            "unexpected_prompt_serial_output_csv_rank",
        ):
            return rc
        if rc := require(
            {
                "prompt_suite_sha256",
                "command_sha256",
                "serial_output_bytes_total",
                "serial_output_bytes_max",
                "serial_output_lines_total",
                "serial_output_lines_max",
            }.issubset(prompt_serial_output_rows[0].keys()),
            "missing_prompt_serial_output_csv_columns",
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
                "command_airgap_ok",
                "command_has_explicit_nic_none",
                "serial_output_lines",
            }.issubset(launch_rows[0].keys()),
            "missing_launch_sequence_csv_columns",
        ):
            return rc
        if rc := require(launch_rows[0]["command_airgap_ok"] == "True", "launch_csv_airgap_not_ok"):
            return rc
        launch_jsonl_rows = [
            json.loads(line)
            for line in launch_jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rc := require(len(launch_jsonl_rows) == 6, "unexpected_launch_jsonl_rows"):
            return rc
        if rc := require(
            {row["phase"] for row in launch_jsonl_rows} == {"warmup", "measured"},
            "unexpected_launch_jsonl_phases",
        ):
            return rc
        if rc := require(
            launch_jsonl_rows[0]["launch_plan_sha256"] == report["launch_plan_sha256"],
            "launch_jsonl_plan_hash_mismatch",
        ):
            return rc
        if rc := require(
            launch_jsonl_rows[0]["command_airgap_ok"] is True,
            "launch_jsonl_airgap_not_ok",
        ):
            return rc
        if rc := require(
            launch_jsonl_rows[0]["prompt_suite_sha256"] == report["prompt_suite"]["suite_sha256"],
            "launch_jsonl_prompt_suite_hash_mismatch",
        ):
            return rc

        no_expected_prompts = Path(tmp) / "missing_expected.jsonl"
        no_expected_prompts.write_text(
            '{"prompt_id":"missing-expected","prompt":"Emit a short deterministic answer."}\n',
            encoding="utf-8",
        )
        missing_expected_output_dir = Path(tmp) / "missing_expected_results"
        missing_expected_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_bench.py"),
            "--image",
            str(SYNTHETIC_IMAGE),
            "--prompts",
            str(no_expected_prompts),
            "--qemu-bin",
            str(SYNTHETIC_QEMU),
            "--repeat",
            "1",
            "--timeout",
            "5",
            "--output-dir",
            str(missing_expected_output_dir),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--require-expected-tokens",
        ]
        missing_expected_completed = subprocess.run(
            missing_expected_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if rc := require(
            missing_expected_completed.returncode == 1,
            "missing_expected_tokens_gate_did_not_fail",
        ):
            sys.stdout.write(missing_expected_completed.stdout)
            sys.stderr.write(missing_expected_completed.stderr)
            return rc
        missing_expected_report = json.loads(
            (missing_expected_output_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8")
        )
        if rc := require(missing_expected_report["status"] == "fail", "missing_expected_report_not_failed"):
            return rc
        if rc := require(
            any(
                finding.get("metric") == "expected_tokens"
                for finding in missing_expected_report["telemetry_findings"]
            ),
            "missing_expected_tokens_finding",
        ):
            return rc
        if rc := require("Phase Summary" in markdown_path.read_text(encoding="utf-8"), "missing_phase_markdown"):
            return rc
        if rc := require(
            "Slowest Prompts" in markdown_path.read_text(encoding="utf-8"),
            "missing_prompt_rank_markdown",
        ):
            return rc
        if rc := require(
            "Prompt Variability" in markdown_path.read_text(encoding="utf-8"),
            "missing_prompt_variability_markdown",
        ):
            return rc
        if rc := require(
            "Prompt Efficiency" in markdown_path.read_text(encoding="utf-8"),
            "missing_prompt_efficiency_markdown",
        ):
            return rc
        if rc := require(
            "Prompt Serial Output Ranking" in markdown_path.read_text(encoding="utf-8"),
            "missing_prompt_serial_output_rank_markdown",
        ):
            return rc
        if rc := require(
            "Launch Sequence Integrity" in markdown_path.read_text(encoding="utf-8"),
            "missing_launch_sequence_markdown",
        ):
            return rc
        if rc := require(
            "Exit class launch error" in markdown_path.read_text(encoding="utf-8"),
            "missing_exit_class_markdown",
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
        if rc := require(dry_report["command_airgap"]["ok"] is True, "missing_dry_run_airgap_ok"):
            return rc
        dry_csv_rows = list(csv.DictReader((dry_output_dir / "qemu_prompt_bench_dry_run_latest.csv").open(encoding="utf-8", newline="")))
        if rc := require(dry_csv_rows[0]["profile"] == "ci-airgap-smoke", "missing_dry_run_csv_profile"):
            return rc
        if rc := require(dry_csv_rows[0]["command_airgap_ok"] == "True", "dry_run_csv_airgap_not_ok"):
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
