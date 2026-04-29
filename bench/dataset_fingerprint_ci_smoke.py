#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_fingerprint.py."""

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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-fingerprint-") as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "dataset_fingerprint_smoke_latest.json"
        jsonl = tmp_path / "dataset_fingerprint_smoke_latest.jsonl"
        csv_path = tmp_path / "dataset_fingerprint_smoke_latest.csv"
        markdown = tmp_path / "dataset_fingerprint_smoke_latest.md"
        junit = tmp_path / "dataset_fingerprint_smoke_latest_junit.xml"
        command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_fingerprint.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(output),
            "--jsonl",
            str(jsonl),
            "--csv",
            str(csv_path),
            "--markdown",
            str(markdown),
            "--junit",
            str(junit),
            "--fail-on-duplicate-ids",
            "--fail-on-conflicting-input-answers",
            "--fail-on-findings",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads(output.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_fingerprint_status"):
            return rc
        if rc := require(report["record_count"] == 3, "unexpected_fingerprint_record_count"):
            return rc
        if rc := require(report["choice_count_histogram"] == {"4": 3}, "unexpected_fingerprint_choices"):
            return rc
        if rc := require(report["answer_histogram"] == {"0": 3}, "unexpected_fingerprint_answers"):
            return rc
        if rc := require(len(report["fingerprints"]) == 3, "missing_fingerprint_rows"):
            return rc
        first = report["fingerprints"][0]
        if rc := require(len(first["prompt_sha256"]) == 64, "missing_prompt_hash"):
            return rc
        if rc := require(len(first["choices_sha256"]) == 64, "missing_choices_hash"):
            return rc
        if rc := require(len(first["input_sha256"]) == 64, "missing_input_hash"):
            return rc
        if rc := require(len(first["answer_payload_sha256"]) == 64, "missing_answer_payload_hash"):
            return rc
        if rc := require(len(first["full_payload_sha256"]) == 64, "missing_full_payload_hash"):
            return rc
        if rc := require("Eval Dataset Fingerprints" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        if rc := require(len(rows) == 3, "unexpected_csv_rows"):
            return rc
        if rc := require(jsonl.read_text(encoding="utf-8").count("\n") == 3, "unexpected_jsonl_rows"):
            return rc
        root = ET.parse(junit).getroot()
        if rc := require(root.attrib.get("name") == "holyc_dataset_fingerprint", "missing_junit"):
            return rc
        if rc := require(root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        conflict = tmp_path / "conflicting_inputs.jsonl"
        conflict.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "conflict-a",
                            "dataset": "fingerprint-smoke",
                            "split": "validation",
                            "prompt": "Same prompt?",
                            "choices": ["yes", "no"],
                            "answer_index": 0,
                            "provenance": "synthetic fingerprint smoke row",
                        },
                        sort_keys=True,
                    ),
                    json.dumps(
                        {
                            "id": "conflict-b",
                            "dataset": "fingerprint-smoke",
                            "split": "validation",
                            "prompt": " Same   prompt? ",
                            "choices": ["yes", "no"],
                            "answer_index": 1,
                            "provenance": "synthetic fingerprint smoke row",
                        },
                        sort_keys=True,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        conflict_output = tmp_path / "conflict_fingerprint.json"
        conflict_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_fingerprint.py"),
            "--input",
            str(conflict),
            "--output",
            str(conflict_output),
            "--fail-on-conflicting-input-answers",
        ]
        completed = subprocess.run(
            conflict_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("conflicting_inputs_not_rejected=true", file=sys.stderr)
            return 1
        conflict_report = json.loads(conflict_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in conflict_report["findings"]}
        if rc := require("conflicting_input_answers" in kinds, "missing_conflict_finding"):
            return rc

    print("dataset_fingerprint_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
