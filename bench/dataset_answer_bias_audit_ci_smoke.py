#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_answer_bias_audit.py."""

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
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-answer-bias-") as tmp:
        tmp_path = Path(tmp)
        pass_json = tmp_path / "dataset_answer_bias_audit_smoke_latest.json"
        pass_md = tmp_path / "dataset_answer_bias_audit_smoke_latest.md"
        pass_csv = tmp_path / "dataset_answer_bias_audit_smoke_latest.csv"
        pass_record_csv = tmp_path / "dataset_answer_bias_audit_smoke_records_latest.csv"
        pass_junit = tmp_path / "dataset_answer_bias_audit_smoke_latest_junit.xml"

        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_answer_bias_audit.py"),
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
            "--max-answer-longest-pct",
            "100",
            "--max-answer-shortest-pct",
            "100",
            "--min-mean-answer-distractor-ratio",
            "0.01",
            "--max-mean-answer-distractor-ratio",
            "100",
            "--check-dataset-splits",
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
        if rc := require(pass_report["findings"] == [], "unexpected_pass_findings"):
            return rc
        if rc := require(pass_report["summary"]["position_histogram"], "missing_position_histogram"):
            return rc
        if rc := require(
            sorted(pass_report["dataset_split_summaries"]) == [
                "arc-smoke:validation",
                "hellaswag-smoke:validation",
                "truthfulqa-smoke:validation",
            ],
            "missing_dataset_split_summaries",
        ):
            return rc
        if rc := require("Dataset Answer Bias Audit" in pass_md.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require("severity,kind,scope,detail" in pass_csv.read_text(encoding="utf-8"), "missing_csv_header"):
            return rc
        record_rows = list(csv.DictReader(pass_record_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(record_rows) == 3, "unexpected_record_csv_rows"):
            return rc
        if rc := require(record_rows[0]["answer_length_position"], "missing_answer_position"):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_answer_bias_audit", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        biased_jsonl = tmp_path / "answer_bias.jsonl"
        write_jsonl(
            biased_jsonl,
            [
                {
                    "id": "bias-1",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Pick the longest answer",
                    "choices": ["a", "bb", "the deliberately longest correct answer", "cc"],
                    "answer_index": 2,
                    "provenance": "synthetic answer bias smoke",
                },
                {
                    "id": "bias-2",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Pick the longest answer again",
                    "choices": ["x", "yy", "another deliberately longest correct answer", "zz"],
                    "answer_index": 2,
                    "provenance": "synthetic answer bias smoke",
                },
            ],
        )
        biased_output = tmp_path / "answer_bias_report.json"
        biased_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_answer_bias_audit.py"),
            "--input",
            str(biased_jsonl),
            "--output",
            str(biased_output),
            "--max-answer-longest-pct",
            "50",
            "--max-mean-answer-distractor-ratio",
            "2",
            "--check-dataset-splits",
            "--fail-on-findings",
        ]
        completed = run_command(biased_command, expected_failure=True)
        if completed.returncode == 0:
            print("answer_bias_not_rejected=true", file=sys.stderr)
            return 1
        biased_report = json.loads(biased_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in biased_report["findings"]}
        if rc := require({"answer_longest_bias", "mean_answer_too_long"} <= kinds, "missing_bias_findings"):
            return rc
        scopes = {finding["scope"] for finding in biased_report["findings"]}
        if rc := require({"overall", "unit:validation"} <= scopes, "missing_scoped_bias_findings"):
            return rc

    print("dataset_answer_bias_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
