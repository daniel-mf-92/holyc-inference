#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_fingerprint_diff.py."""

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


def fingerprint(input_path: Path, output_path: Path) -> int:
    command = [
        sys.executable,
        str(ROOT / "bench" / "dataset_fingerprint.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--fail-on-findings",
    ]
    return run_command(command).returncode


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-fingerprint-diff-ci-") as tmp:
        tmp_path = Path(tmp)
        baseline = tmp_path / "baseline.json"
        candidate = tmp_path / "candidate.json"
        if fingerprint(SAMPLE, baseline) != 0:
            return 1
        if fingerprint(SAMPLE, candidate) != 0:
            return 1

        pass_json = tmp_path / "dataset_fingerprint_diff_smoke_latest.json"
        pass_csv = tmp_path / "dataset_fingerprint_diff_smoke_latest.csv"
        pass_md = tmp_path / "dataset_fingerprint_diff_smoke_latest.md"
        pass_junit = tmp_path / "dataset_fingerprint_diff_smoke_latest_junit.xml"
        pass_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_fingerprint_diff.py"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(pass_json),
            "--csv",
            str(pass_csv),
            "--markdown",
            str(pass_md),
            "--junit",
            str(pass_junit),
            "--fail-on-any-change",
            "--fail-on-findings",
        ]
        completed = run_command(pass_command)
        if completed.returncode != 0:
            return completed.returncode

        pass_report = json.loads(pass_json.read_text(encoding="utf-8"))
        if rc := require(pass_report["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(pass_report["change_counts"] == {}, "unexpected_pass_changes"):
            return rc
        if rc := require(pass_report["unchanged_records"] == 3, "unexpected_unchanged_records"):
            return rc
        if rc := require("Eval Dataset Fingerprint Diff" in pass_md.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require(len(list(csv.DictReader(pass_csv.open(encoding="utf-8", newline="")))) == 0, "unexpected_csv_rows"):
            return rc
        junit_root = ET.parse(pass_junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_fingerprint_diff", "missing_junit_name"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failures"):
            return rc

        changed_rows = [json.loads(line) for line in SAMPLE.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed_rows[0]["answer_index"] = 1
        changed_rows[1]["question"] = changed_rows[1]["question"] + " Extra local smoke text."
        changed_rows.append(
            {
                "id": "fingerprint-diff-added",
                "dataset": "smoke-eval",
                "split": "validation",
                "prompt": "Which option is newly added?",
                "choices": ["alpha", "beta", "gamma", "delta"],
                "answer_index": 0,
                "provenance": "synthetic fingerprint diff smoke row",
            }
        )
        changed_jsonl = tmp_path / "changed.jsonl"
        changed_jsonl.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in changed_rows) + "\n",
            encoding="utf-8",
        )
        changed_fingerprint = tmp_path / "changed_fingerprint.json"
        if fingerprint(changed_jsonl, changed_fingerprint) != 0:
            return 1

        fail_json = tmp_path / "dataset_fingerprint_diff_changed.json"
        fail_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_fingerprint_diff.py"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(changed_fingerprint),
            "--output",
            str(fail_json),
            "--fail-on-added",
            "--fail-on-content-changes",
            "--fail-on-answer-changes",
        ]
        completed = subprocess.run(fail_command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode == 0:
            print("fingerprint_diff_change_not_rejected=true", file=sys.stderr)
            return 1
        fail_report = json.loads(fail_json.read_text(encoding="utf-8"))
        counts = fail_report["change_counts"]
        if rc := require(counts.get("added") == 1, "missing_added_change"):
            return rc
        if rc := require(counts.get("content_changed", 0) >= 1, "missing_content_change"):
            return rc
        if rc := require(counts.get("answer_changed") == 1, "missing_answer_change"):
            return rc
        kinds = {finding["kind"] for finding in fail_report["findings"]}
        if rc := require({"added", "content_changed", "answer_changed"}.issubset(kinds), "missing_gate_findings"):
            return rc

    print("dataset_fingerprint_diff_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
