#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for dataset duplicate audits."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run_audit(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "bench" / "dataset_duplicate_audit.py"), *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-duplicate-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        output_dir = tmp_path / "out"
        passing = tmp_path / "passing.jsonl"
        failing = tmp_path / "failing.jsonl"
        write_jsonl(
            passing,
            [
                {
                    "id": "row-1",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": "Which tool measures temperature?",
                    "choices": ["thermometer", "ruler", "scale", "compass"],
                    "answer_index": 0,
                },
                {
                    "id": "row-2",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": "Which tool measures distance?",
                    "choices": ["ruler", "thermometer", "scale", "compass"],
                    "answer_index": 0,
                },
            ],
        )
        write_jsonl(
            failing,
            [
                {
                    "id": "copy-a",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": "Which option is a planet?",
                    "choices": ["Mars", "granite", "steam", "paper"],
                    "answer_index": 0,
                },
                {
                    "id": "copy-b",
                    "dataset": "smoke",
                    "split": "validation",
                    "prompt": "Which   option is a planet?",
                    "choices": ["Mars", "granite", "steam", "paper"],
                    "answer_index": 1,
                },
            ],
        )

        passed = run_audit(
            [
                "--input",
                str(passing),
                "--output",
                str(output_dir / "duplicate_pass.json"),
                "--markdown",
                str(output_dir / "duplicate_pass.md"),
                "--csv",
                str(output_dir / "duplicate_pass_findings.csv"),
                "--record-csv",
                str(output_dir / "duplicate_pass_records.csv"),
                "--junit",
                str(output_dir / "duplicate_pass_junit.xml"),
                "--fail-on-duplicate-prompts",
                "--fail-on-duplicate-payloads",
                "--fail-on-findings",
            ]
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode
        payload = json.loads((output_dir / "duplicate_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        rows = list(csv.DictReader((output_dir / "duplicate_pass_records.csv").open(encoding="utf-8")))
        if rc := require(rows[0]["prompt_duplicate_count"] == "1", "unexpected_prompt_duplicate_count"):
            return rc
        junit = ET.parse(output_dir / "duplicate_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "dataset_duplicate_audit", "missing_junit_name"):
            return rc

        failed = run_audit(
            [
                "--input",
                str(failing),
                "--output",
                str(output_dir / "duplicate_fail.json"),
                "--fail-on-duplicate-payloads",
                "--fail-on-conflicting-answers",
            ]
        )
        if rc := require(failed.returncode == 1, "expected_failure_status"):
            sys.stdout.write(failed.stdout)
            sys.stderr.write(failed.stderr)
            return rc
        failed_payload = json.loads((output_dir / "duplicate_fail.json").read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in failed_payload["findings"]}
        if rc := require({"conflicting_duplicate_prompt", "conflicting_duplicate_payload"} <= kinds, "missing_failure_kinds"):
            return rc
    print("dataset_duplicate_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
