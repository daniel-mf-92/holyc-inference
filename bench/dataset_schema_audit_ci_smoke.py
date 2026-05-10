#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_schema_audit.py."""

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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-schema-audit-") as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "dataset_schema_audit_smoke_latest.json"
        markdown = tmp_path / "dataset_schema_audit_smoke_latest.md"
        csv_path = tmp_path / "dataset_schema_audit_smoke_latest.csv"
        record_csv = tmp_path / "dataset_schema_audit_smoke_records_latest.csv"
        junit = tmp_path / "dataset_schema_audit_smoke_latest_junit.xml"
        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_schema_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--record-csv",
            str(record_csv),
            "--junit",
            str(junit),
            "--require-provenance",
            "--min-choices",
            "4",
            "--max-choices",
            "4",
            "--max-prompt-bytes",
            "4096",
            "--max-choice-bytes",
            "1024",
            "--max-record-payload-bytes",
            "8192",
            "--min-answer-labels",
            "1",
            "--min-dataset-split-answer-labels",
            "1",
            "--fail-on-duplicate-ids",
            "--fail-on-duplicate-payloads",
            "--fail-on-conflicting-payload-answers",
            "--fail-on-findings",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads(output.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_schema_status"):
            return rc
        if rc := require(report["normalized_record_count"] == 3, "unexpected_schema_record_count"):
            return rc
        if rc := require(report["choice_count_histogram"] == {"4": 3}, "unexpected_choice_histogram"):
            return rc
        if rc := require(report["answer_label_count"] == 1, "unexpected_answer_label_count"):
            return rc
        if rc := require(
            report["dataset_split_answer_label_counts"] == {
                "arc-smoke": {"validation": 1},
                "hellaswag-smoke": {"validation": 1},
                "truthfulqa-smoke": {"validation": 1},
            },
            "unexpected_dataset_split_answer_label_counts",
        ):
            return rc
        if rc := require(report["findings"] == [], "unexpected_schema_findings"):
            return rc
        if rc := require(len(report["record_telemetry"]) == 3, "missing_record_telemetry"):
            return rc
        if rc := require(
            all(row["record_payload_bytes"] > row["prompt_bytes"] for row in report["record_telemetry"]),
            "unexpected_record_payload_bytes",
        ):
            return rc
        if rc := require("Eval Dataset Schema Audit" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require(
            "severity,kind,source,detail" in csv_path.read_text(encoding="utf-8"),
            "missing_findings_csv_header",
        ):
            return rc
        record_rows = list(csv.DictReader(record_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(record_rows) == 3, "unexpected_record_csv_rows"):
            return rc
        if rc := require(record_rows[0]["choice_count"] == "4", "unexpected_record_csv_choice_count"):
            return rc
        if rc := require(record_rows[0]["payload_key_sha256"], "missing_record_csv_payload_key"):
            return rc
        junit_root = ET.parse(junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_schema_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_input = tmp_path / "bad_schema.jsonl"
        bad_rows = [
            {
                "id": "duplicate-id",
                "dataset": "schema-smoke",
                "split": "validation",
                "prompt": "Choose the first option.",
                "choices": ["alpha", "bravo", "charlie", "delta"],
                "answer_index": 0,
                "provenance": "synthetic schema audit smoke",
            },
            {
                "id": "duplicate-id",
                "dataset": "schema-smoke",
                "split": "validation",
                "prompt": "Choose the first option.",
                "choices": ["alpha", "bravo", "charlie", "delta"],
                "answer_index": 1,
                "provenance": "synthetic schema audit smoke",
            },
            {
                "id": "missing-provenance",
                "dataset": "schema-smoke",
                "split": "validation",
                "prompt": "Choose from too few options.",
                "choices": ["yes", "no"],
                "answer_index": 0,
            },
        ]
        bad_input.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in bad_rows) + "\n",
            encoding="utf-8",
        )
        bad_output = tmp_path / "bad_schema_audit.json"
        bad_record_csv = tmp_path / "bad_schema_audit_records.csv"
        bad_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_schema_audit.py"),
            "--input",
            str(bad_input),
            "--output",
            str(bad_output),
            "--record-csv",
            str(bad_record_csv),
            "--require-provenance",
            "--min-choices",
            "4",
            "--fail-on-duplicate-ids",
            "--fail-on-duplicate-payloads",
            "--fail-on-conflicting-payload-answers",
            "--fail-on-findings",
        ]
        completed = run_command(bad_command, expected_failure=True)
        if completed.returncode == 0:
            print("bad_schema_input_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in bad_report["findings"]}
        for expected in {
            "missing_provenance",
            "too_few_choices",
            "duplicate_record_id",
            "duplicate_payload",
            "conflicting_payload_answers",
        }:
            if rc := require(expected in kinds, f"missing_{expected}"):
                return rc
        bad_rows_csv = list(csv.DictReader(bad_record_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(bad_rows_csv) == 3, "unexpected_bad_record_csv_rows"):
            return rc

    print("dataset_schema_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
