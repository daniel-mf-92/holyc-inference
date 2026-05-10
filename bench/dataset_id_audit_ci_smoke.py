#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_id_audit.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def run_command(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-id-audit-") as tmp:
        tmp_path = Path(tmp)
        pass_json = tmp_path / "dataset_id_audit_smoke_latest.json"
        pass_md = tmp_path / "dataset_id_audit_smoke_latest.md"
        pass_csv = tmp_path / "dataset_id_audit_smoke_latest.csv"
        pass_record_csv = tmp_path / "dataset_id_audit_smoke_records_latest.csv"
        pass_junit = tmp_path / "dataset_id_audit_smoke_latest_junit.xml"

        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_id_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(pass_json),
            "--markdown",
            str(pass_md),
            "--csv",
            str(pass_csv),
            "--record-csv",
            str(pass_record_csv),
            "--junit",
            str(pass_junit),
            "--require-explicit-id",
            "--max-record-id-bytes",
            "64",
            "--id-pattern",
            r"[a-z0-9-]+",
            "--fail-duplicate-record-ids",
            "--fail-duplicate-dataset-split-record-ids",
            "--fail-on-findings",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads(pass_json.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(report["record_count"] == 3, "unexpected_record_count"):
            return rc
        if rc := require(report["explicit_id_count"] == 3, "unexpected_explicit_id_count"):
            return rc
        if rc := require(report["duplicate_record_id_count"] == 0, "unexpected_duplicate_record_ids"):
            return rc
        if rc := require(report["findings"] == [], "unexpected_pass_findings"):
            return rc
        if rc := require("Dataset ID Audit" in pass_md.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require("severity,kind,scope,detail" in pass_csv.read_text(encoding="utf-8"), "missing_csv_header"):
            return rc
        record_rows = list(csv.DictReader(pass_record_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(record_rows) == 3, "unexpected_record_csv_rows"):
            return rc
        if rc := require(record_rows[0]["record_id_bytes"], "missing_record_id_bytes"):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_id_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_jsonl = tmp_path / "bad_ids.jsonl"
        write_jsonl(
            bad_jsonl,
            [
                {
                    "id": "dup",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "First prompt?",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 0,
                    "provenance": "synthetic id smoke",
                },
                {
                    "id": "dup",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Second prompt?",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 1,
                    "provenance": "synthetic id smoke",
                },
                {
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Missing explicit id?",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 2,
                    "provenance": "synthetic id smoke",
                },
                {
                    "id": "Bad ID With Spaces",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Bad id pattern?",
                    "choices": ["a", "b", "c", "d"],
                    "answer_index": 3,
                    "provenance": "synthetic id smoke",
                },
            ],
        )
        fail_json = tmp_path / "bad_ids_report.json"
        fail_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_id_audit.py"),
            "--input",
            str(bad_jsonl),
            "--output",
            str(fail_json),
            "--require-explicit-id",
            "--max-record-id-bytes",
            "8",
            "--id-pattern",
            r"[a-z0-9-]+",
            "--fail-duplicate-record-ids",
            "--fail-duplicate-dataset-split-record-ids",
            "--fail-on-findings",
        ]
        completed = run_command(fail_command, expected_failure=True)
        if completed.returncode == 0:
            print("bad_ids_not_rejected=true", file=sys.stderr)
            return 1
        fail_report = json.loads(fail_json.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        expected = {
            "duplicate_record_id",
            "duplicate_dataset_split_record_id",
            "implicit_record_id",
            "record_id_pattern_mismatch",
            "record_id_too_long",
        }
        if rc := require(expected <= kinds, "missing_id_findings"):
            return rc

    print("dataset_id_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
