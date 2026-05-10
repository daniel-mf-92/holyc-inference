#!/usr/bin/env python3
"""Stdlib-only smoke gate for hceval_provenance_audit.py."""

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


def pack_dataset(source: Path, binary: Path, manifest: Path) -> int:
    completed = run_command(
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
            "provenance-smoke",
            "--split",
            "validation",
        ]
    )
    return completed.returncode


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hceval-provenance-audit-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "provenance.jsonl"
        binary = tmp_path / "provenance.hceval"
        manifest = tmp_path / "provenance.hceval.manifest.json"
        write_jsonl(
            source,
            [
                {
                    "id": "prov-1",
                    "dataset": "provenance-smoke",
                    "split": "validation",
                    "prompt": "Choose the documented answer.",
                    "choices": ["guess", "citation", "rumor", "silence"],
                    "answer_index": 1,
                    "provenance": "synthetic provenance smoke source",
                }
            ],
        )
        if rc := pack_dataset(source, binary, manifest):
            return rc

        output = RESULTS / "hceval_provenance_audit_smoke_latest.json"
        markdown = RESULTS / "hceval_provenance_audit_smoke_latest.md"
        csv_path = RESULTS / "hceval_provenance_audit_smoke_latest.csv"
        findings_csv = RESULTS / "hceval_provenance_audit_smoke_latest_findings.csv"
        junit = RESULTS / "hceval_provenance_audit_smoke_latest_junit.xml"
        completed = run_command(
            [
                sys.executable,
                str(BENCH / "hceval_provenance_audit.py"),
                "--input",
                str(tmp_path),
                "--require-manifest",
                "--min-provenance-coverage-pct",
                "100",
                "--min-distinct-provenance",
                "1",
                "--max-provenance-bytes",
                "128",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--csv",
                str(csv_path),
                "--findings-csv",
                str(findings_csv),
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
        if rc := require(report["record_count"] == 1, "unexpected_record_count"):
            return rc
        if rc := require(report["provenance_coverage_pct"] == 100.0, "bad_coverage"):
            return rc
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(rows) == 1 and rows[0]["status"] == "pass", "bad_csv_rows"):
            return rc
        findings = list(csv.DictReader(findings_csv.open(encoding="utf-8", newline="")))
        if rc := require(findings == [], "unexpected_findings"):
            return rc
        if rc := require("HCEval Provenance Audit" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        junit_root = ET.parse(junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_hceval_provenance_audit", "bad_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

    print("hceval_provenance_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
