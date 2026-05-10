#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU summary consistency audits."""

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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-summary-consistency-ci-") as tmp:
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
                str(ROOT / "bench" / "qemu_summary_consistency_audit.py"),
                str(bench_dir / "qemu_prompt_bench_latest.json"),
                "--output-dir",
                str(audit_dir),
                "--output-stem",
                "summary",
                "--min-artifacts",
                "1",
                "--min-measured-rows",
                "2",
            ]
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_summary_status"):
            return rc
        if rc := require(report["summary"]["artifacts"] == 1, "unexpected_artifact_count"):
            return rc
        if rc := require(report["summary"]["measured_rows"] == 2, "unexpected_measured_rows"):
            return rc
        if rc := require(report["summary"]["checked_fields"] > 20, "too_few_checked_fields"):
            return rc
        if rc := require(
            "QEMU Summary Consistency Audit" in (audit_dir / "summary.md").read_text(encoding="utf-8"),
            "missing_markdown_title",
        ):
            return rc
        artifact_rows = list(csv.DictReader((audit_dir / "summary.csv").open(encoding="utf-8", newline="")))
        if rc := require(artifact_rows[0]["status"] == "pass", "unexpected_csv_status"):
            return rc
        finding_rows = list(csv.DictReader((audit_dir / "summary_findings.csv").open(encoding="utf-8", newline="")))
        if rc := require(finding_rows == [], "unexpected_findings_rows"):
            return rc
        junit_root = ET.parse(audit_dir / "summary_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_qemu_summary_consistency_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_path = tmp_path / "bad_summary.json"
        bad_payload = json.loads((bench_dir / "qemu_prompt_bench_latest.json").read_text(encoding="utf-8"))
        bad_payload["suite_summary"]["runs"] += 1
        bad_payload["summaries"][0]["tok_per_s_median"] = 1.0
        bad_path.write_text(json.dumps(bad_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fail_dir = tmp_path / "fail"
        completed = run(
            [
                sys.executable,
                str(ROOT / "bench" / "qemu_summary_consistency_audit.py"),
                str(bad_path),
                "--output-dir",
                str(fail_dir),
                "--output-stem",
                "summary",
            ]
        )
        if rc := require(completed.returncode == 1, "expected_bad_summary_failure"):
            return rc
        fail_report = json.loads((fail_dir / "summary.json").read_text(encoding="utf-8"))
        kinds = {(finding["scope"], finding["field"], finding["kind"]) for finding in fail_report["findings"]}
        if rc := require(("suite", "runs", "value_mismatch") in kinds, "missing_suite_mismatch"):
            return rc
        if rc := require(any(field == "tok_per_s_median" for _, field, _ in kinds), "missing_prompt_mismatch"):
            return rc

    print("qemu_summary_consistency_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
