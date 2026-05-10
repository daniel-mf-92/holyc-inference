#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_choice_audit.py."""

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


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-choice-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        pass_json = tmp_path / "dataset_choice_audit_smoke_latest.json"
        pass_md = tmp_path / "dataset_choice_audit_smoke_latest.md"
        pass_csv = tmp_path / "dataset_choice_audit_smoke_latest.csv"
        pass_record_csv = tmp_path / "dataset_choice_audit_smoke_records_latest.csv"
        pass_junit = tmp_path / "dataset_choice_audit_smoke_latest_junit.xml"

        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_choice_audit.py"),
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
            "--fail-on-duplicate-choices",
            "--fail-on-choice-overlap",
            "--fail-on-label-prefixes",
            "--fail-on-prompt-answer-leak",
            "--fail-on-prompt-choice-leak",
            "--max-choice-length-ratio",
            "100",
            "--fail-on-length-skew",
            "--fail-on-findings",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        pass_report = json.loads(pass_json.read_text(encoding="utf-8"))
        if rc := require(pass_report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(pass_report["record_count"] == 3, "unexpected_pass_record_count"):
            return rc
        if rc := require(pass_report["choice_count_histogram"] == {"4": 3}, "unexpected_choice_histogram"):
            return rc
        if rc := require(pass_report["findings"] == [], "unexpected_pass_findings"):
            return rc
        if rc := require(len(pass_report["record_telemetry"]) == 3, "missing_record_telemetry"):
            return rc
        if rc := require(
            all(row["choice_count"] == 4 for row in pass_report["record_telemetry"]),
            "unexpected_record_choice_counts",
        ):
            return rc
        if rc := require("Dataset Choice Audit" in pass_md.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require(
            "severity,kind,source,dataset,split,record_id,detail" in pass_csv.read_text(encoding="utf-8"),
            "missing_csv_header",
        ):
            return rc
        if rc := require(
            "source,dataset,split,record_id,choice_count,answer_index" in pass_record_csv.read_text(encoding="utf-8"),
            "missing_record_csv_header",
        ):
            return rc
        record_rows = list(csv.DictReader(pass_record_csv.open(encoding="utf-8", newline="")))
        if rc := require(record_rows[0]["answer_choice_bytes"] == "45", "unexpected_answer_choice_bytes"):
            return rc
        if rc := require(float(record_rows[0]["answer_choice_byte_pct"]) > 25.0, "missing_answer_choice_byte_pct"):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_choice_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        bad_jsonl = tmp_path / "bad_choices.jsonl"
        bad_jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "dup-choice",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "Pick a capital city.",
                            "choices": ["Paris", "Paris", "Berlin", "Rome"],
                            "answer_index": 0,
                            "provenance": "synthetic bad duplicate choice",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "choice-overlap",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "Choose the process named by the teacher.",
                            "choices": ["evaporation", "water evaporation", "condensation", "sublimation"],
                            "answer_index": 0,
                            "provenance": "synthetic bad choice overlap",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "label-prefix",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "Pick the planet closest to the Sun.",
                            "choices": ["A. Mercury", "Venus", "Earth", "Mars"],
                            "answer_index": 0,
                            "provenance": "synthetic bad label prefix",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "answer-leak",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "A thermometer is used to measure air temperature.",
                            "choices": ["thermometer", "ruler", "scale", "compass"],
                            "answer_index": 0,
                            "provenance": "synthetic bad answer leak",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "distractor-leak",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "The sentence mentions Venus but asks for the innermost planet.",
                            "choices": ["Mercury", "Venus", "Earth", "Mars"],
                            "answer_index": 0,
                            "provenance": "synthetic bad distractor leak",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "length-skew",
                            "dataset": "choice-smoke",
                            "split": "validation",
                            "prompt": "Choose the shortest valid answer.",
                            "choices": [
                                "yes",
                                "a much longer distractor answer with many extra words",
                                "no",
                                "maybe",
                            ],
                            "answer_index": 0,
                            "provenance": "synthetic bad length skew",
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        bad_output = tmp_path / "bad_choice_audit.json"
        bad_record_csv = tmp_path / "bad_choice_audit_records.csv"
        bad_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_choice_audit.py"),
            "--input",
            str(bad_jsonl),
            "--output",
            str(bad_output),
            "--record-csv",
            str(bad_record_csv),
            "--fail-on-duplicate-choices",
            "--fail-on-choice-overlap",
            "--fail-on-label-prefixes",
            "--fail-on-prompt-answer-leak",
            "--fail-on-prompt-choice-leak",
            "--min-answer-leak-chars",
            "8",
            "--min-choice-leak-chars",
            "5",
            "--max-choice-length-ratio",
            "8",
            "--fail-on-length-skew",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            bad_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("bad_choice_input_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in bad_report["findings"]}
        for expected in {
            "duplicate_choice_text",
            "choice_text_overlap",
            "choice_label_prefix",
            "prompt_contains_correct_choice",
            "prompt_contains_choice_text",
            "choice_length_skew",
        }:
            if rc := require(expected in kinds, f"missing_{expected}"):
                return rc
        telemetry_by_id = {row["record_id"]: row for row in bad_report["record_telemetry"]}
        if rc := require(
            telemetry_by_id["dup-choice"]["duplicate_choice_group_count"] == 1,
            "missing_duplicate_record_telemetry",
        ):
            return rc
        if rc := require(
            telemetry_by_id["distractor-leak"]["prompt_choice_leak_count"] == 1,
            "missing_leak_record_telemetry",
        ):
            return rc
        if rc := require("prompt_choice_leak_count" in bad_record_csv.read_text(encoding="utf-8"), "missing_bad_record_csv"):
            return rc

    print("dataset_choice_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
