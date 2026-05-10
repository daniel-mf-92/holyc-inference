#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_leak_audit.py."""

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


def finding_kinds(report_path: Path) -> set[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {finding["kind"] for finding in report["findings"]}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-leak-") as tmp:
        tmp_path = Path(tmp)
        pass_json = tmp_path / "dataset_leak_audit_smoke_latest.json"
        pass_md = tmp_path / "dataset_leak_audit_smoke_latest.md"
        pass_csv = tmp_path / "dataset_leak_audit_smoke_latest.csv"
        pass_junit = tmp_path / "dataset_leak_audit_smoke_latest_junit.xml"

        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(pass_json),
            "--markdown",
            str(pass_md),
            "--csv",
            str(pass_csv),
            "--junit",
            str(pass_junit),
            "--fail-on-leaks",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        pass_report = json.loads(pass_json.read_text(encoding="utf-8"))
        if rc := require(pass_report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(pass_report["record_count"] == 3, "unexpected_pass_record_count"):
            return rc
        if rc := require(pass_report["findings"] == [], "unexpected_pass_findings"):
            return rc
        expected_counts = {
            "arc-smoke": {"validation": 1},
            "hellaswag-smoke": {"validation": 1},
            "truthfulqa-smoke": {"validation": 1},
        }
        if rc := require(pass_report["counts_by_dataset_split"] == expected_counts, "unexpected_dataset_split_counts"):
            return rc
        if rc := require("Dataset Leak Audit" in pass_md.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        csv_rows = list(csv.DictReader(pass_csv.open(encoding="utf-8", newline="")))
        if rc := require(csv_rows == [], "unexpected_pass_csv_findings"):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_leak_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        leak_jsonl = tmp_path / "split_leak.jsonl"
        write_jsonl(
            leak_jsonl,
            [
                {
                    "id": "train-hot",
                    "dataset": "leak-smoke",
                    "split": "train",
                    "prompt": "Which item is hot?",
                    "choices": ["ice", "fire", "snow", "rain"],
                    "answer_index": 1,
                    "provenance": "synthetic leak audit smoke",
                },
                {
                    "id": "valid-hot",
                    "dataset": "leak-smoke",
                    "split": "validation",
                    "prompt": " which item is hot? ",
                    "choices": ["ice", "fire", "snow", "rain"],
                    "answer_index": 1,
                    "provenance": "synthetic leak audit smoke",
                },
            ],
        )
        leak_output = tmp_path / "split_leak_report.json"
        leak_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(leak_jsonl),
            "--output",
            str(leak_output),
            "--fail-on-leaks",
        ]
        completed = run_command(leak_command, expected_failure=True)
        if completed.returncode == 0:
            print("split_leak_not_rejected=true", file=sys.stderr)
            return 1
        if rc := require(
            {"prompt_split_leak", "payload_split_leak"} <= finding_kinds(leak_output),
            "missing_split_leak_findings",
        ):
            return rc

        warning_jsonl = tmp_path / "duplicate_within_split.jsonl"
        write_jsonl(
            warning_jsonl,
            [
                {
                    "id": "duplicate",
                    "dataset": "leak-smoke",
                    "split": "validation",
                    "prompt": "Question A",
                    "choices": ["yes", "no"],
                    "answer_index": 0,
                    "provenance": "synthetic leak audit smoke",
                },
                {
                    "id": "duplicate",
                    "dataset": "leak-smoke",
                    "split": "validation",
                    "prompt": "Question B",
                    "choices": ["yes", "no"],
                    "answer_index": 1,
                    "provenance": "synthetic leak audit smoke",
                },
            ],
        )
        warning_output = tmp_path / "duplicate_within_split_report.json"
        warning_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_leak_audit.py"),
            "--input",
            str(warning_jsonl),
            "--output",
            str(warning_output),
            "--fail-on-leaks",
        ]
        completed = run_command(warning_command)
        if completed.returncode != 0:
            return completed.returncode
        warning_report = json.loads(warning_output.read_text(encoding="utf-8"))
        if rc := require(warning_report["status"] == "pass", "unexpected_warning_status"):
            return rc
        if rc := require(warning_report["warning_count"] == 1, "unexpected_warning_count"):
            return rc
        if rc := require(finding_kinds(warning_output) == {"duplicate_record_id"}, "missing_duplicate_warning"):
            return rc

    print("dataset_leak_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
