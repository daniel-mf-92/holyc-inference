#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_roundtrip_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
RESULTS = BENCH / "results" / "datasets"
SAMPLE = BENCH / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "dataset_roundtrip_audit_smoke_latest.json"
    markdown = RESULTS / "dataset_roundtrip_audit_smoke_latest.md"
    findings_csv = RESULTS / "dataset_roundtrip_audit_smoke_latest_findings.csv"
    fingerprints_csv = RESULTS / "dataset_roundtrip_audit_smoke_fingerprints_latest.csv"
    junit = RESULTS / "dataset_roundtrip_audit_smoke_latest_junit.xml"

    command = [
        sys.executable,
        str(BENCH / "dataset_roundtrip_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--findings-csv",
        str(findings_csv),
        "--fingerprints-csv",
        str(fingerprints_csv),
        "--junit",
        str(junit),
        "--dataset",
        "smoke-eval",
        "--split",
        "validation",
        "--max-prompt-bytes",
        "4096",
        "--max-choice-bytes",
        "1024",
        "--max-record-payload-bytes",
        "8192",
        "--fail-on-findings",
    ]
    completed = run_command(command)
    if completed.returncode:
        return completed.returncode

    report = json.loads(output.read_text(encoding="utf-8"))
    if rc := require(report["status"] == "pass", "unexpected_status"):
        return rc
    if rc := require(report["record_count"] == 3, "unexpected_record_count"):
        return rc
    if rc := require(report["expected_source_sha256"] == report["actual_source_sha256"], "source_digest_mismatch"):
        return rc
    if rc := require(report["expected_binary_layout"] == report["actual_binary_layout"], "layout_mismatch"):
        return rc
    if rc := require(
        report["expected_record_fingerprints"] == report["actual_record_fingerprints"],
        "fingerprint_mismatch",
    ):
        return rc
    if rc := require("Dataset Roundtrip Audit" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
        return rc
    if rc := require("severity,kind,scope,detail" in findings_csv.read_text(encoding="utf-8"), "missing_findings_csv"):
        return rc
    fingerprint_rows = list(csv.DictReader(fingerprints_csv.open(encoding="utf-8", newline="")))
    if rc := require(len(fingerprint_rows) == 3, "unexpected_fingerprint_rows"):
        return rc
    if rc := require("full_payload_sha256" in fingerprint_rows[0], "missing_fingerprint_column"):
        return rc
    junit_root = ET.parse(junit).getroot()
    if rc := require(junit_root.attrib.get("name") == "holyc_dataset_roundtrip_audit", "missing_junit_name"):
        return rc
    if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
        return rc

    with tempfile.TemporaryDirectory(prefix="dataset-roundtrip-audit-") as tmp:
        bad_jsonl = Path(tmp) / "too_large.jsonl"
        bad_output = Path(tmp) / "too_large.json"
        write_jsonl(
            bad_jsonl,
            [
                {
                    "id": "large-1",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "This prompt is intentionally long for the byte-limit gate.",
                    "choices": ["yes", "no"],
                    "answer_index": 0,
                    "provenance": "synthetic roundtrip byte gate smoke",
                }
            ],
        )
        bad_command = [
            sys.executable,
            str(BENCH / "dataset_roundtrip_audit.py"),
            "--input",
            str(bad_jsonl),
            "--output",
            str(bad_output),
            "--max-prompt-bytes",
            "8",
            "--fail-on-findings",
        ]
        completed = run_command(bad_command, expected_failure=True)
        if rc := require(completed.returncode == 1, "byte_gate_not_rejected"):
            return rc
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        if rc := require(bad_report["status"] == "fail", "unexpected_bad_status"):
            return rc
        if rc := require(
            any(finding["kind"] == "inspector_finding" for finding in bad_report["findings"]),
            "missing_inspector_finding",
        ):
            return rc

    print("dataset_roundtrip_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
