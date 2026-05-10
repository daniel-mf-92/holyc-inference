#!/usr/bin/env python3
"""CI smoke test for dataset_prompt_choice_overlap_audit.py."""

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


def run(command: list[str], *, expected_failure: bool = False) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode and not expected_failure:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def assert_pass_outputs() -> int:
    report = json.loads((RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest.json").read_text(encoding="utf-8"))
    if rc := require(report["status"] == "pass", "unexpected_pass_status"):
        return rc
    if rc := require(report["record_count"] == 3, "unexpected_record_count"):
        return rc
    if rc := require(report["overlap_record_count"] == 0, "unexpected_overlap_count"):
        return rc
    if rc := require(report["findings"] == [], "unexpected_findings"):
        return rc
    record_csv = RESULTS / "dataset_prompt_choice_overlap_audit_smoke_records_latest.csv"
    record_rows = list(csv.DictReader(record_csv.open()))
    if rc := require(len(record_rows) == 3, "unexpected_record_csv_rows"):
        return rc
    junit_root = ET.parse(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest_junit.xml").getroot()
    if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
        return rc
    return 0


def assert_failure_gate() -> int:
    with tempfile.TemporaryDirectory(prefix="dataset-prompt-choice-overlap-") as tmp:
        sample = Path(tmp) / "overlap.jsonl"
        output = Path(tmp) / "audit.json"
        write_jsonl(
            sample,
            [
                {
                    "id": "answer-overlap",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Select the item that is fire.",
                    "choices": ["ice", "fire", "snow", "rain"],
                    "answer_index": 1,
                },
                {
                    "id": "distractor-overlap",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Which item is cold like snow?",
                    "choices": ["heat", "fire", "snow", "steam"],
                    "answer_index": 0,
                },
            ],
        )
        command = [
            sys.executable,
            str(BENCH / "dataset_prompt_choice_overlap_audit.py"),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--fail-on-answer-overlap",
        ]
        completed = run(command, expected_failure=True)
        if completed.returncode == 0:
            print("answer_overlap_not_rejected=true", file=sys.stderr)
            return 1
        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in report["findings"]}
        if rc := require("answer_choice_overlap" in kinds, "missing_answer_overlap"):
            return rc
        if rc := require("distractor_choice_overlap" in kinds, "missing_distractor_overlap"):
            return rc
        if rc := require(report["answer_overlap_record_count"] == 1, "unexpected_answer_overlap_count"):
            return rc
    return 0


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(BENCH / "dataset_prompt_choice_overlap_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_records_latest.csv"),
        "--junit",
        str(RESULTS / "dataset_prompt_choice_overlap_audit_smoke_latest_junit.xml"),
        "--fail-on-any-overlap",
    ]
    completed = run(command)
    if completed.returncode:
        return completed.returncode
    if rc := assert_pass_outputs():
        return rc
    return assert_failure_gate()


if __name__ == "__main__":
    raise SystemExit(main())
