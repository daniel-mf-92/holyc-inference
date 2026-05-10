#!/usr/bin/env python3
"""Stdlib-only smoke gate for hceval_export_roundtrip_audit.py."""

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
    with tempfile.TemporaryDirectory(prefix="hceval-export-roundtrip-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "mixed.jsonl"
        binary = tmp_path / "mixed.hceval"
        manifest = tmp_path / "mixed.hceval.manifest.json"
        write_jsonl(
            source,
            [
                {
                    "id": "arc-1",
                    "dataset": "arc",
                    "split": "validation",
                    "prompt": "Pick the letter after A.",
                    "choices": ["A", "B", "C", "D"],
                    "answer_index": 1,
                    "provenance": "synthetic arc smoke",
                },
                {
                    "id": "truth-1",
                    "dataset": "truthfulqa",
                    "split": "validation",
                    "prompt": "Which answer is grounded?",
                    "choices": ["claim", "evidence", "rumor", "guess"],
                    "answer_index": 1,
                    "provenance": "synthetic truthfulqa smoke",
                },
            ],
        )
        pack = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_pack.py"),
                "--input",
                str(source),
                "--output",
                str(binary),
                "--manifest",
                str(manifest),
                "--dataset",
                "mixed-smoke",
                "--split",
                "validation",
            ]
        )
        if pack.returncode:
            return pack.returncode

        output = RESULTS / "hceval_export_roundtrip_audit_smoke_latest.json"
        markdown = RESULTS / "hceval_export_roundtrip_audit_smoke_latest.md"
        parity_csv = RESULTS / "hceval_export_roundtrip_audit_smoke_latest.csv"
        junit = RESULTS / "hceval_export_roundtrip_audit_smoke_latest_junit.xml"
        completed = run_command(
            [
                sys.executable,
                str(BENCH / "hceval_export_roundtrip_audit.py"),
                "--input",
                str(binary),
                "--pack-manifest",
                str(manifest),
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--csv",
                str(parity_csv),
                "--junit",
                str(junit),
                "--fail-on-findings",
            ]
        )
        if completed.returncode:
            return completed.returncode

        report = json.loads(output.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_status"):
            return rc
        if rc := require(report["binary_sha256"] == report["repacked_binary_sha256"], "binary_digest_mismatch"):
            return rc
        if rc := require(report["source_sha256"] == report["repacked_source_sha256"], "source_digest_mismatch"):
            return rc
        if rc := require(
            "HCEval Export Roundtrip Audit" in markdown.read_text(encoding="utf-8"),
            "missing_markdown",
        ):
            return rc
        rows = list(csv.DictReader(parity_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(rows) == 4, "unexpected_csv_rows"):
            return rc
        if rc := require(all(row["status"] == "pass" for row in rows), "unexpected_csv_failure"):
            return rc
        junit_root = ET.parse(junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_hceval_export_roundtrip_audit", "bad_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_output = tmp_path / "without_manifest.json"
        bad = run_command(
            [
                sys.executable,
                str(BENCH / "hceval_export_roundtrip_audit.py"),
                "--input",
                str(binary),
                "--output",
                str(bad_output),
                "--fail-on-findings",
            ],
            expected_failure=True,
        )
        if rc := require(bad.returncode == 1, "missing_manifest_not_rejected"):
            return rc
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        if rc := require(bad_report["status"] == "fail", "unexpected_bad_status"):
            return rc
        if rc := require(
            any(finding["kind"] == "source_digest_mismatch" for finding in bad_report["findings"]),
            "missing_source_mismatch",
        ):
            return rc

    print("hceval_export_roundtrip_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
