#!/usr/bin/env python3
"""Stdlib-only smoke gate for hceval_suite_audit.py."""

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


def pack_dataset(source: Path, binary: Path, manifest: Path, dataset: str) -> int:
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
            dataset,
            "--split",
            "validation",
        ]
    )
    return completed.returncode


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hceval-suite-audit-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "suite.jsonl"
        binary = tmp_path / "suite.hceval"
        manifest = tmp_path / "suite.hceval.manifest.json"
        write_jsonl(
            source,
            [
                {
                    "id": "suite-1",
                    "dataset": "suite-smoke",
                    "split": "validation",
                    "prompt": "Choose the grounded answer.",
                    "choices": ["guess", "evidence", "rumor", "silence"],
                    "answer_index": 1,
                    "provenance": "synthetic suite smoke",
                }
            ],
        )
        if rc := pack_dataset(source, binary, manifest, "suite-smoke"):
            return rc

        output = RESULTS / "hceval_suite_audit_smoke_latest.json"
        markdown = RESULTS / "hceval_suite_audit_smoke_latest.md"
        csv_path = RESULTS / "hceval_suite_audit_smoke_latest.csv"
        junit = RESULTS / "hceval_suite_audit_smoke_latest_junit.xml"
        completed = run_command(
            [
                sys.executable,
                str(BENCH / "hceval_suite_audit.py"),
                "--input",
                str(tmp_path),
                "--require-manifest",
                "--max-prompt-bytes",
                "128",
                "--max-choice-bytes",
                "64",
                "--max-record-payload-bytes",
                "256",
                "--output",
                str(output),
                "--markdown",
                str(markdown),
                "--csv",
                str(csv_path),
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
        if rc := require(report["input_count"] == 1, "unexpected_input_count"):
            return rc
        if rc := require(report["record_count"] == 1, "unexpected_record_count"):
            return rc
        if rc := require(report["dataset_counts"] == {"suite-smoke": 1}, "bad_dataset_counts"):
            return rc
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(rows) == 1 and rows[0]["status"] == "pass", "bad_csv_rows"):
            return rc
        if rc := require("HCEval Suite Audit" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        junit_root = ET.parse(junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_hceval_suite_audit", "bad_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        missing_manifest = tmp_path / "missing.hceval"
        missing_manifest.write_bytes(binary.read_bytes())
        bad_output = tmp_path / "missing_manifest_report.json"
        bad = run_command(
            [
                sys.executable,
                str(BENCH / "hceval_suite_audit.py"),
                "--input",
                str(missing_manifest),
                "--require-manifest",
                "--output",
                str(bad_output),
                "--fail-on-findings",
            ],
            expected_failure=True,
        )
        if rc := require(bad.returncode == 1, "missing_manifest_not_rejected"):
            return rc
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        if rc := require(
            any(finding["kind"] == "missing_manifest" for finding in bad_report["findings"]),
            "missing_manifest_finding_absent",
        ):
            return rc

    print("hceval_suite_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
