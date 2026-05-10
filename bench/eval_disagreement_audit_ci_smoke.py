#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval disagreement audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_report(path: Path, agreement_count: int, dataset_agreement_count: int) -> None:
    payload = {
        "dataset": "smoke-eval",
        "split": "validation",
        "summary": {
            "record_count": 4,
            "agreement_count": agreement_count,
            "agreement": agreement_count / 4,
            "holyc_accuracy": 0.75,
            "llama_accuracy": 0.5,
            "accuracy_delta_holyc_minus_llama": 0.25,
            "dataset_breakdown": [
                {
                    "dataset": "arc-smoke",
                    "split": "validation",
                    "record_count": 4,
                    "agreement_count": dataset_agreement_count,
                    "agreement": dataset_agreement_count / 4,
                    "holyc_accuracy": 0.75,
                    "llama_accuracy": 0.5,
                    "accuracy_delta_holyc_minus_llama": 0.25,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def run_audit(input_path: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(ROOT / "bench" / "eval_disagreement_audit.py"),
        str(input_path),
        "--output",
        str(output_dir / "eval_disagreement_audit_latest.json"),
        "--markdown",
        str(output_dir / "eval_disagreement_audit_latest.md"),
        "--csv",
        str(output_dir / "eval_disagreement_audit_latest.csv"),
        "--findings-csv",
        str(output_dir / "eval_disagreement_audit_findings_latest.csv"),
        "--junit",
        str(output_dir / "eval_disagreement_audit_junit_latest.xml"),
        "--min-records",
        "4",
        "--max-disagreement-pct",
        "25",
        "--max-dataset-split-disagreement-pct",
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-disagreement-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "passing.json"
        failing_report = tmp_path / "failing.json"
        write_report(passing_report, agreement_count=3, dataset_agreement_count=3)
        write_report(failing_report, agreement_count=2, dataset_agreement_count=2)

        pass_dir = tmp_path / "pass"
        completed = run_audit(passing_report, pass_dir)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        pass_payload = json.loads((pass_dir / "eval_disagreement_audit_latest.json").read_text(encoding="utf-8"))
        pass_junit = ET.parse(pass_dir / "eval_disagreement_audit_junit_latest.xml").getroot()
        checks = [
            require(pass_payload["status"] == "pass", "eval_disagreement_pass_status=false"),
            require(pass_payload["scope_count"] == 2, "eval_disagreement_pass_scope_count=false"),
            require(pass_payload["scopes"][0]["disagreement_pct"] == 25.0, "eval_disagreement_pass_pct=false"),
            require(pass_junit.attrib.get("failures") == "0", "eval_disagreement_pass_junit=false"),
            require(
                "No disagreement gate findings"
                in (pass_dir / "eval_disagreement_audit_latest.md").read_text(encoding="utf-8"),
                "eval_disagreement_pass_markdown=false",
            ),
        ]
        if not all(checks):
            return 1

        fail_dir = tmp_path / "fail"
        completed = run_audit(failing_report, fail_dir)
        if completed.returncode == 0:
            print("eval_disagreement_failing_report_not_rejected=true", file=sys.stderr)
            return 1
        fail_payload = json.loads((fail_dir / "eval_disagreement_audit_latest.json").read_text(encoding="utf-8"))
        fail_junit = ET.parse(fail_dir / "eval_disagreement_audit_junit_latest.xml").getroot()
        kinds = {finding["kind"] for finding in fail_payload["findings"]}
        checks = [
            require(fail_payload["status"] == "fail", "eval_disagreement_fail_status=false"),
            require("disagreement_pct_exceeded" in kinds, "eval_disagreement_missing_gate=false"),
            require(len(fail_payload["findings"]) == 2, "eval_disagreement_fail_count=false"),
            require(fail_junit.attrib.get("failures") == "1", "eval_disagreement_fail_junit=false"),
        ]
        if not all(checks):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
