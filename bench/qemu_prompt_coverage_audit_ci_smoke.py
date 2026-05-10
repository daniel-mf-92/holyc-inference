#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU prompt coverage audit artifacts."""

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


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-coverage-ci-") as tmp:
        tmp_path = Path(tmp)
        bench_dir = tmp_path / "bench"
        coverage_dir = tmp_path / "coverage"

        bench_command = [
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
            str(bench_dir),
            "--profile",
            "ci-airgap-smoke",
            "--model",
            "synthetic-smoke",
            "--quantization",
            "Q4_0",
            "--require-tokens",
            "--require-expected-tokens",
            "--require-expected-tokens-match",
            "--require-guest-prompt-sha256-match",
            "--require-guest-prompt-bytes-match",
            "--max-launches",
            "6",
            "--min-prompt-count",
            "2",
            "--qemu-arg=-m",
            "--qemu-arg=256M",
        ]
        completed = run(bench_command, cwd=ROOT)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        coverage_command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_prompt_coverage_audit.py"),
            str(bench_dir / "qemu_prompt_bench_latest.json"),
            "--output-dir",
            str(coverage_dir),
            "--output-stem",
            "coverage",
            "--require-suite-file",
            "--require-success",
            "--fail-on-unexpected-prompts",
            "--min-artifacts",
            "1",
            "--min-prompts",
            "2",
            "--min-runs-per-prompt",
            "2",
        ]
        completed = run(coverage_command, cwd=ROOT)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((coverage_dir / "coverage.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_coverage_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_artifact_count"):
            return rc
        if rc := require(report["summary"]["expected_prompts"] == 2, "unexpected_expected_prompt_count"):
            return rc
        if rc := require(report["summary"]["measured_prompts"] == 2, "unexpected_measured_prompt_count"):
            return rc
        if rc := require(report["summary"]["successful_runs"] == 4, "unexpected_successful_run_count"):
            return rc
        if rc := require(report["artifacts"][0]["prompt_suite_file_sha256_matches"] is True, "suite_hash_not_checked"):
            return rc
        if rc := require(report["artifacts"][0]["min_successful_runs_per_expected_prompt"] == 2, "missing_min_success"):
            return rc
        if rc := require({row["prompt"] for row in report["prompt_rows"]} == {"smoke-code", "smoke-short"}, "missing_prompt_rows"):
            return rc
        if rc := require(
            all(row["expected_tokens_mismatches"] == 0 for row in report["prompt_rows"]),
            "unexpected_expected_token_mismatch",
        ):
            return rc
        if rc := require(
            "QEMU Prompt Coverage Audit" in (coverage_dir / "coverage.md").read_text(encoding="utf-8"),
            "missing_markdown_title",
        ):
            return rc
        artifact_rows = list(csv.DictReader((coverage_dir / "coverage.csv").open(encoding="utf-8", newline="")))
        if rc := require(artifact_rows[0]["status"] == "pass", "unexpected_csv_status"):
            return rc
        prompt_rows = list(csv.DictReader((coverage_dir / "coverage_prompts.csv").open(encoding="utf-8", newline="")))
        if rc := require(len(prompt_rows) == 2, "unexpected_prompt_csv_rows"):
            return rc
        junit_root = ET.parse(coverage_dir / "coverage_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_qemu_prompt_coverage_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        fail_dir = tmp_path / "fail"
        fail_command = coverage_command.copy()
        fail_command[fail_command.index("--output-dir") + 1] = str(fail_dir)
        fail_command[fail_command.index("--min-runs-per-prompt") + 1] = "3"
        completed = run(fail_command, cwd=ROOT)
        if completed.returncode == 0:
            print("coverage_min_runs_gate_did_not_fail=true", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_dir / "coverage.json").read_text(encoding="utf-8"))
        if rc := require(fail_report["status"] == "fail", "unexpected_fail_report_status"):
            return rc
        if rc := require(
            {finding["kind"] for finding in fail_report["findings"]} == {"min_runs_per_prompt"},
            "missing_min_runs_finding",
        ):
            return rc

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        empty_output = tmp_path / "empty-output"
        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_prompt_coverage_audit.py"),
                str(empty_dir),
                "--output-dir",
                str(empty_output),
                "--output-stem",
                "empty",
                "--min-artifacts",
                "1",
            ],
            cwd=ROOT,
        )
        if completed.returncode == 0:
            print("coverage_min_artifacts_gate_did_not_fail=true", file=sys.stderr)
            return 1
        empty_report = json.loads((empty_output / "empty.json").read_text(encoding="utf-8"))
        if rc := require(empty_report["findings"][0]["kind"] == "min_artifacts", "missing_min_artifacts_finding"):
            return rc

    print("qemu_prompt_coverage_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
