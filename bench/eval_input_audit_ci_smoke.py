#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval input audits."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"
HOLYC = ROOT / "bench" / "eval" / "samples" / "holyc_smoke_predictions.jsonl"
LLAMA = ROOT / "bench" / "eval" / "samples" / "llama_smoke_predictions.jsonl"


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


def audit_command(output_dir: Path, stem: str, *extra_args: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "bench" / "eval_input_audit.py"),
        "--gold",
        str(GOLD),
        "--holyc",
        str(HOLYC),
        "--llama",
        str(LLAMA),
        "--dataset",
        "smoke-eval",
        "--split",
        "validation",
        "--model",
        "synthetic-smoke",
        "--quantization",
        "Q4_0",
        "--output-dir",
        str(output_dir),
        "--output-stem",
        stem,
        *extra_args,
    ]


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-input-audit-ci-") as tmp:
        output_dir = Path(tmp)
        stem = "eval_input_audit_smoke"
        completed = run_command(
            audit_command(
                output_dir,
                stem,
                "--max-majority-gold-answer-pct",
                "100",
                "--min-choices",
                "4",
                "--max-choices",
                "4",
                "--min-top-score-margin",
                "0",
                "--record-csv",
                str(output_dir / f"{stem}_records.csv"),
            )
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_audit_status"):
            return rc
        if rc := require(report["gold_record_count"] == 3, "unexpected_gold_record_count"):
            return rc
        if rc := require(report["gold_distribution"]["choice_count_histogram"] == {"4": 3}, "unexpected_choice_histogram"):
            return rc
        if rc := require(report["gold_choice_gates"] == {"min_choices": 4, "max_choices": 4}, "missing_choice_gates"):
            return rc
        if rc := require(report["summary"]["holyc_coverage"] == 3, "unexpected_holyc_coverage"):
            return rc
        if rc := require(report["summary"]["llama_coverage"] == 3, "unexpected_llama_coverage"):
            return rc

        expected_files = [
            f"{stem}.json",
            f"{stem}.md",
            f"{stem}.csv",
            f"{stem}_records.csv",
            f"{stem}_junit.xml",
        ]
        if rc := require(all((output_dir / name).exists() for name in expected_files), "missing_audit_artifact"):
            return rc
        if rc := require(
            "Eval Input Audit" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"),
            "missing_audit_markdown",
        ):
            return rc
        if rc := require(
            "Score Margins" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"),
            "missing_score_margin_markdown",
        ):
            return rc
        if rc := require(load_csv_rows(output_dir / f"{stem}.csv") == [], "unexpected_pass_issue_rows"):
            return rc
        record_rows = load_csv_rows(output_dir / f"{stem}_records.csv")
        if rc := require(len(record_rows) == 6, "unexpected_record_csv_rows"):
            return rc
        if rc := require({row["source"] for row in record_rows} == {"holyc", "llama.cpp"}, "unexpected_record_csv_sources"):
            return rc
        if rc := require(
            all(row["valid"] == "True" and row["correct"] == "True" for row in record_rows),
            "unexpected_record_csv_validity",
        ):
            return rc
        if rc := require(
            any(row["top_score_margin"] == "8.0" for row in record_rows),
            "missing_record_csv_score_margin",
        ):
            return rc

        junit_root = ET.parse(output_dir / f"{stem}_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_eval_input_audit", "missing_audit_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_audit_junit_failures"):
            return rc

        fail_stem = "eval_input_choice_gate_fail"
        failed = run_command(
            audit_command(output_dir, fail_stem, "--max-choices", "3"),
            expected_failure=True,
        )
        if rc := require(failed.returncode == 2, "expected_choice_gate_failure"):
            return rc
        failed_report = json.loads((output_dir / f"{fail_stem}.json").read_text(encoding="utf-8"))
        if rc := require(failed_report["status"] == "fail", "unexpected_failed_audit_status"):
            return rc
        if rc := require(
            len([issue for issue in failed_report["issues"] if "above --max-choices 3" in issue["message"]]) == 3,
            "unexpected_choice_gate_issues",
        ):
            return rc
        failed_junit = ET.parse(output_dir / f"{fail_stem}_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_audit_junit"):
            return rc

        invalid = run_command(
            audit_command(output_dir, "eval_input_invalid_choice_gate", "--min-choices", "5", "--max-choices", "4"),
            expected_failure=True,
        )
        if rc := require(invalid.returncode == 2, "expected_invalid_choice_gate_args"):
            return rc

        invalid_margin = run_command(
            audit_command(output_dir, "eval_input_invalid_margin_gate", "--min-top-score-margin", "-0.1"),
            expected_failure=True,
        )
        if rc := require(invalid_margin.returncode == 2, "expected_invalid_margin_gate_args"):
            return rc

    print("eval_input_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
