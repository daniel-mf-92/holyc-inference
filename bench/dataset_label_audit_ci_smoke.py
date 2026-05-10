#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset label audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "dataset_label_audit.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_dir / "dataset_label_audit_latest.json"),
        "--markdown",
        str(output_dir / "dataset_label_audit_latest.md"),
        "--csv",
        str(output_dir / "dataset_label_audit_latest.csv"),
        "--record-csv",
        str(output_dir / "dataset_label_audit_records_latest.csv"),
        "--junit",
        str(output_dir / "dataset_label_audit_latest_junit.xml"),
        *extra_args,
    ]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-label-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        good = tmp_path / "good.jsonl"
        bad = tmp_path / "bad.jsonl"
        write_jsonl(
            good,
            [
                {
                    "id": "arc-good",
                    "dataset": "arc-smoke",
                    "split": "validation",
                    "question": "Which letter is first?",
                    "choices": [
                        {"label": "A", "text": "A"},
                        {"label": "B", "text": "B"},
                        {"label": "C", "text": "C"},
                        {"label": "D", "text": "D"},
                    ],
                    "answerKey": "A",
                },
                {
                    "id": "truthfulqa-good",
                    "dataset": "truthfulqa-smoke",
                    "split": "validation",
                    "question": "Pick the true statement.",
                    "mc1_targets": {"choices": ["true", "false"], "labels": [1, 0]},
                },
            ],
        )
        write_jsonl(
            bad,
            [
                {
                    "id": "arc-bad",
                    "question": "Which letter is first?",
                    "choices": [{"label": "A", "text": "A"}, {"label": "A", "text": "duplicate"}],
                    "answerKey": "Z",
                },
                {
                    "id": "truthfulqa-bad",
                    "question": "Pick one.",
                    "mc1_targets": {"choices": ["one", "two"], "labels": [1, 1]},
                },
                {"id": "hellaswag-bad", "ctx": "A person starts", "endings": ["x", "y"], "label": "3"},
            ],
        )

        pass_dir = tmp_path / "pass"
        completed = run_audit(good, pass_dir, "--require-contiguous-arc-labels", "--fail-on-findings")
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        pass_report = json.loads((pass_dir / "dataset_label_audit_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "dataset_label_audit_latest_junit.xml").getroot()
        checks = [
            require(pass_report["status"] == "pass", "dataset_label_audit_pass_status=false"),
            require(pass_report["rows"] == 2, "dataset_label_audit_pass_rows=false"),
            require(pass_junit.attrib.get("failures") == "0", "dataset_label_audit_pass_junit=false"),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_audit(bad, fail_dir, "--fail-on-findings")
        if completed.returncode == 0:
            print("dataset_label_audit_bad_rows_not_rejected=true", file=sys.stderr)
            return 1
        fail_report = json.loads((fail_dir / "dataset_label_audit_latest.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        fail_junit = ET.parse(fail_dir / "dataset_label_audit_latest_junit.xml").getroot()
        checks = [
            require(fail_report["status"] == "fail", "dataset_label_audit_fail_status=false"),
            require("duplicate_choice_labels" in kinds, "dataset_label_audit_missing_duplicate_labels=false"),
            require("answer_label_missing" in kinds, "dataset_label_audit_missing_answer_label=false"),
            require("truthfulqa_correct_label_count" in kinds, "dataset_label_audit_missing_truthfulqa_count=false"),
            require("hellaswag_label_out_of_range" in kinds, "dataset_label_audit_missing_hellaswag_range=false"),
            require(fail_junit.attrib.get("failures") == "1", "dataset_label_audit_fail_junit=false"),
        ]
        if not all(checks):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
