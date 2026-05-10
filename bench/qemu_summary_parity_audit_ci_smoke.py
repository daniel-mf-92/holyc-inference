#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU summary parity audits."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-summary-parity-ci-") as tmp:
        tmp_path = Path(tmp)
        bench_dir = tmp_path / "bench"
        audit_dir = tmp_path / "audit"
        completed = run(
            [
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
                "1",
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
                "--max-launches",
                "4",
                "--min-prompt-count",
                "2",
                "--qemu-arg=-m",
                "--qemu-arg=256M",
            ]
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_summary_parity_audit.py"),
                str(bench_dir / "qemu_prompt_bench_latest.json"),
                "--output-dir",
                str(audit_dir),
                "--output-stem",
                "summary_parity",
                "--min-artifacts",
                "1",
                "--min-rows",
                "3",
            ]
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((audit_dir / "summary_parity.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_summary_parity_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_artifact_count"):
            return rc
        if rc := require(report["summary"]["suite_rows"] == 1, "unexpected_suite_rows"):
            return rc
        if rc := require(report["summary"]["prompt_rows"] == 2, "unexpected_prompt_rows"):
            return rc
        if rc := require(report["summary"]["compared_values"] >= 100, "too_few_compared_values"):
            return rc
        rows = list(csv.DictReader((audit_dir / "summary_parity.csv").open(encoding="utf-8", newline="")))
        if rc := require(rows[0]["status"] == "pass", "unexpected_csv_status"):
            return rc
        finding_rows = list(csv.DictReader((audit_dir / "summary_parity_findings.csv").open(encoding="utf-8", newline="")))
        if rc := require(finding_rows == [], "unexpected_findings"):
            return rc
        junit_text = (audit_dir / "summary_parity_junit.xml").read_text(encoding="utf-8")
        if rc := require('name="holyc_qemu_summary_parity_audit"' in junit_text, "missing_junit_suite"):
            return rc
        if rc := require('failures="0"' in junit_text, "unexpected_junit_failures"):
            return rc

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        bad_json = bad_dir / "qemu_prompt_bench_latest.json"
        bad_csv = bad_dir / "qemu_prompt_bench_summary_latest.csv"
        bad_json.write_text((bench_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"), encoding="utf-8")
        rows = list(csv.DictReader((bench_dir / "qemu_prompt_bench_summary_latest.csv").open(encoding="utf-8", newline="")))
        rows[0]["total_tokens"] = "0"
        with bad_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        fail_dir = tmp_path / "fail"
        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_summary_parity_audit.py"),
                str(bad_json),
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "summary_parity",
            ]
        )
        if rc := require(completed.returncode == 1, "expected_bad_summary_parity_failure"):
            return rc
        fail_report = json.loads((fail_dir / "summary_parity.json").read_text(encoding="utf-8"))
        kinds = {(finding["field"], finding["kind"]) for finding in fail_report["findings"]}
        if rc := require(("total_tokens", "value_mismatch") in kinds, "missing_total_tokens_mismatch"):
            return rc

    print("qemu_summary_parity_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
