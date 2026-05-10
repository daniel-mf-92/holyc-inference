#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_order_audit.py."""

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


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


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


def write_row(record_id: str, answer_index: int) -> str:
    return json.dumps(
        {
            "id": record_id,
            "dataset": "order-smoke",
            "split": "validation",
            "prompt": f"Choose the indexed answer for {record_id}.",
            "choices": ["alpha", "bravo", "charlie", "delta"],
            "answer_index": answer_index,
            "provenance": "synthetic order audit smoke row",
        },
        sort_keys=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-order-audit-") as tmp:
        tmp_path = Path(tmp)
        pass_input = tmp_path / "balanced_order.jsonl"
        pass_input.write_text(
            "\n".join(write_row(f"balanced-{index}", answer) for index, answer in enumerate([0, 1, 2, 3]))
            + "\n",
            encoding="utf-8",
        )

        output = tmp_path / "dataset_order_audit_smoke_latest.json"
        markdown = tmp_path / "dataset_order_audit_smoke_latest.md"
        csv_path = tmp_path / "dataset_order_audit_smoke_latest.csv"
        record_csv = tmp_path / "dataset_order_audit_smoke_records_latest.csv"
        findings_csv = tmp_path / "dataset_order_audit_smoke_latest_findings.csv"
        junit = tmp_path / "dataset_order_audit_smoke_latest_junit.xml"
        command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_order_audit.py"),
            "--input",
            str(pass_input),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_path),
            "--record-csv",
            str(record_csv),
            "--findings-csv",
            str(findings_csv),
            "--junit",
            str(junit),
            "--group-by",
            "overall",
            "--max-longest-answer-run",
            "1",
            "--max-longest-answer-run-pct",
            "25",
            "--max-edge-answer-run",
            "1",
            "--min-answer-switches",
            "3",
            "--fail-on-findings",
        ]
        completed = run_command(command)
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads(output.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_order_status"):
            return rc
        if rc := require(report["record_count"] == 4, "unexpected_order_record_count"):
            return rc
        stats = report["order_stats"][0]
        if rc := require(stats["answer_sequence"] == [0, 1, 2, 3], "unexpected_answer_sequence"):
            return rc
        if rc := require(stats["transition_count"] == 3, "unexpected_transition_count"):
            return rc
        if rc := require(stats["longest_run"]["length"] == 1, "unexpected_longest_run"):
            return rc
        if rc := require("Dataset Order Audit" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(csv_rows) == 1, "unexpected_order_csv_rows"):
            return rc
        if rc := require(csv_rows[0]["transition_count"] == "3", "unexpected_order_csv_transition_count"):
            return rc
        record_rows = list(csv.DictReader(record_csv.open(encoding="utf-8", newline="")))
        if rc := require(len(record_rows) == 4, "unexpected_order_record_csv_rows"):
            return rc
        if rc := require(record_rows[0]["previous_answer_index"] == "", "unexpected_first_previous_answer"):
            return rc
        if rc := require(record_rows[0]["next_answer_index"] == "1", "unexpected_first_next_answer"):
            return rc
        if rc := require(record_rows[1]["changed_from_previous"] == "True", "unexpected_changed_from_previous"):
            return rc
        if rc := require(record_rows[3]["is_trailing_run"] == "True", "unexpected_trailing_run"):
            return rc
        if rc := require(
            "severity,kind,scope,source,detail" in findings_csv.read_text(encoding="utf-8"),
            "missing_findings_csv_header",
        ):
            return rc
        root = ET.parse(junit).getroot()
        if rc := require(root.attrib.get("name") == "holyc_dataset_order_audit", "missing_junit_name"):
            return rc
        if rc := require(root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        sample_output = tmp_path / "sample_order.json"
        sample_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_order_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(sample_output),
            "--group-by",
            "dataset_split",
        ]
        completed = run_command(sample_command)
        if completed.returncode != 0:
            return completed.returncode
        sample_report = json.loads(sample_output.read_text(encoding="utf-8"))
        if rc := require(sample_report["record_count"] == 3, "unexpected_sample_record_count"):
            return rc
        if rc := require(len(sample_report["order_stats"]) == 3, "unexpected_sample_scope_count"):
            return rc

        bad_input = tmp_path / "blocked_order.jsonl"
        bad_input.write_text(
            "\n".join(write_row(f"blocked-{index}", answer) for index, answer in enumerate([0, 0, 0, 0, 1]))
            + "\n",
            encoding="utf-8",
        )
        bad_output = tmp_path / "bad_order_audit.json"
        bad_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_order_audit.py"),
            "--input",
            str(bad_input),
            "--output",
            str(bad_output),
            "--max-longest-answer-run",
            "2",
            "--max-longest-answer-run-pct",
            "50",
            "--max-edge-answer-run",
            "2",
            "--min-answer-switches",
            "2",
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
            print("bad_order_input_not_rejected=true", file=sys.stderr)
            return 1
        bad_report = json.loads(bad_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in bad_report["findings"]}
        for expected in {
            "long_answer_run",
            "long_answer_run_pct",
            "leading_run",
            "too_few_answer_switches",
        }:
            if rc := require(expected in kinds, f"missing_{expected}"):
                return rc

    print("dataset_order_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
