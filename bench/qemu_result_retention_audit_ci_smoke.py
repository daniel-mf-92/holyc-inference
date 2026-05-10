#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for qemu_result_retention_audit.py."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_audit(input_dir: Path, output_dir: Path, stem: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "qemu_result_retention_audit.py"),
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_artifact(path: Path, generated_at: str = "2026-05-01T08:55:00Z", status: str = "pass") -> None:
    payload = {
        "generated_at": generated_at,
        "status": status,
        "command": ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"],
        "benchmarks": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-retention-audit-ci-") as tmp:
        root = Path(tmp)
        output_dir = root / "out"

        passing = root / "passing"
        passing.mkdir()
        latest = passing / "qemu_prompt_bench_latest.json"
        history = passing / "qemu_prompt_bench_20260501T085500Z.json"
        write_artifact(latest)
        history.write_bytes(latest.read_bytes())

        completed = run_audit(passing, output_dir, "qemu_result_retention_audit_smoke")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        report = json.loads((output_dir / "qemu_result_retention_audit_smoke.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_retention_pass_status"):
            return rc
        if rc := require(report["summary"]["latest_artifacts"] == 1, "unexpected_retention_latest_count"):
            return rc
        if rc := require(report["artifacts"][0]["hashes_match"] is True, "missing_retention_hash_match"):
            return rc
        if rc := require(
            "QEMU Result Retention Audit" in (output_dir / "qemu_result_retention_audit_smoke.md").read_text(encoding="utf-8"),
            "missing_retention_markdown",
        ):
            return rc
        if rc := require(
            "history_sha256" in (output_dir / "qemu_result_retention_audit_smoke.csv").read_text(encoding="utf-8"),
            "missing_retention_csv_fields",
        ):
            return rc
        finding_rows = list(
            csv.DictReader((output_dir / "qemu_result_retention_audit_smoke_findings.csv").open(encoding="utf-8"))
        )
        if rc := require(finding_rows == [], "unexpected_retention_finding_rows"):
            return rc
        junit = ET.parse(output_dir / "qemu_result_retention_audit_smoke_junit.xml").getroot()
        if rc := require(junit.attrib["name"] == "holyc_qemu_result_retention_audit", "missing_retention_junit"):
            return rc
        if rc := require(junit.attrib["failures"] == "0", "unexpected_retention_junit_failure"):
            return rc

        missing = root / "missing"
        missing.mkdir()
        write_artifact(missing / "qemu_prompt_bench_latest.json")
        failed = run_audit(missing, output_dir, "qemu_result_retention_audit_missing")
        if rc := require(failed.returncode == 1, "retention_audit_missing_history_not_rejected"):
            return rc
        fail_report = json.loads((output_dir / "qemu_result_retention_audit_missing.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"history_count", "missing_expected_history"} <= kinds, "retention_findings_not_reported"):
            return rc
        fail_rows = list(
            csv.DictReader((output_dir / "qemu_result_retention_audit_missing_findings.csv").open(encoding="utf-8"))
        )
        if rc := require(
            any(row["kind"] == "missing_expected_history" for row in fail_rows),
            "missing_retention_history_finding_csv",
        ):
            return rc

        mismatch = root / "mismatch"
        mismatch.mkdir()
        latest = mismatch / "qemu_prompt_bench_latest.json"
        history = mismatch / "qemu_prompt_bench_20260501T085500Z.json"
        write_artifact(latest)
        write_artifact(history, status="fail")
        failed = run_audit(mismatch, output_dir, "qemu_result_retention_audit_mismatch")
        if rc := require(failed.returncode == 1, "retention_audit_mismatch_not_rejected"):
            return rc
        mismatch_report = json.loads((output_dir / "qemu_result_retention_audit_mismatch.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in mismatch_report["findings"]}
        if rc := require("latest_history_mismatch" in kinds, "retention_mismatch_not_reported"):
            return rc

    print("qemu_result_retention_audit_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
