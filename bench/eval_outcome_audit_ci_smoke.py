#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval outcome bucket audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def row(record_id: str, dataset: str, holyc: int, llama: int, answer: int) -> dict[str, object]:
    return {
        "record_id": record_id,
        "dataset": dataset,
        "split": "validation",
        "answer_index": answer,
        "holyc_prediction": holyc,
        "llama_prediction": llama,
        "holyc_correct": holyc == answer,
        "llama_correct": llama == answer,
        "engines_agree": holyc == llama,
    }


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "rows": rows,
        "summary": {"record_count": len(rows)},
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "eval_outcome_audit.py"),
        str(input_path),
        "--output",
        str(output_dir / "eval_outcome_audit_latest.json"),
        "--markdown",
        str(output_dir / "eval_outcome_audit_latest.md"),
        "--csv",
        str(output_dir / "eval_outcome_audit_latest.csv"),
        "--findings-csv",
        str(output_dir / "eval_outcome_audit_findings_latest.csv"),
        "--junit",
        str(output_dir / "eval_outcome_audit_junit_latest.xml"),
        "--min-records",
        "4",
        "--max-llama-only-correct-pct",
        "25",
        "--max-dataset-split-llama-only-correct-pct",
        "50",
        "--max-both-wrong-pct",
        "25",
        "--fail-on-findings",
        *extra_args,
    ]
    return subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def require(condition: bool, message: str) -> bool:
    if not condition:
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-outcome-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "passing.json"
        failing_report = tmp_path / "failing.json"
        write_report(
            passing_report,
            [
                row("a", "arc-smoke", 0, 0, 0),
                row("b", "arc-smoke", 0, 1, 0),
                row("c", "truthfulqa-smoke", 1, 1, 1),
                row("d", "truthfulqa-smoke", 2, 3, 0),
            ],
        )
        write_report(
            failing_report,
            [
                row("a", "arc-smoke", 1, 0, 0),
                row("b", "arc-smoke", 1, 0, 0),
                row("c", "truthfulqa-smoke", 2, 2, 1),
                row("d", "truthfulqa-smoke", 3, 2, 1),
            ],
        )

        pass_dir = tmp_path / "pass"
        completed = run_audit(passing_report, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        pass_payload = json.loads((pass_dir / "eval_outcome_audit_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "eval_outcome_audit_junit_latest.xml").getroot()
        checks = [
            require(pass_payload["status"] == "pass", "eval_outcome_pass_status=false"),
            require(pass_payload["scope_count"] == 3, "eval_outcome_pass_scope_count=false"),
            require(pass_payload["scopes"][0]["holyc_only_correct"] == 1, "eval_outcome_pass_holyc_only=false"),
            require(pass_payload["scopes"][0]["both_wrong_disagree"] == 1, "eval_outcome_pass_both_wrong=false"),
            require(pass_junit.attrib.get("failures") == "0", "eval_outcome_pass_junit=false"),
            require(
                "No eval outcome gate findings"
                in (pass_dir / "eval_outcome_audit_latest.md").read_text(encoding="utf-8"),
                "eval_outcome_pass_markdown=false",
            ),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_audit(failing_report, fail_dir)
        if completed.returncode == 0:
            print("eval_outcome_failing_report_not_rejected=true", file=sys.stderr)
            return 1
        fail_payload = json.loads((fail_dir / "eval_outcome_audit_latest.json").read_text(encoding="utf-8"))
        fail_junit = ET.parse(fail_dir / "eval_outcome_audit_junit_latest.xml").getroot()
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        checks = [
            require(fail_payload["status"] == "fail", "eval_outcome_fail_status=false"),
            require("llama_only_correct_pct_exceeded" in kinds, "eval_outcome_missing_llama_gate=false"),
            require("both_wrong_pct_exceeded" in kinds, "eval_outcome_missing_both_wrong_gate=false"),
            require(fail_junit.attrib.get("failures") == "1", "eval_outcome_fail_junit=false"),
        ]
        if not all(checks):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
