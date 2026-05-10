#!/usr/bin/env python3
"""Stdlib-only smoke gate for eval significance auditing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_report(path: Path, *, delta: float, holyc_only: int, llama_only: int, p_value: float) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "summary": {
                    "record_count": 12,
                    "holyc_accuracy": 0.5 + delta,
                    "llama_accuracy": 0.5,
                    "accuracy_delta_holyc_minus_llama": delta,
                    "mcnemar_exact": {
                        "holyc_only_correct": holyc_only,
                        "llama_only_correct": llama_only,
                        "discordant_count": holyc_only + llama_only,
                        "p_value": p_value,
                        "method": "exact_binomial_two_sided",
                    },
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "record_count": 12,
                            "holyc_accuracy": 0.5 + delta,
                            "llama_accuracy": 0.5,
                            "accuracy_delta_holyc_minus_llama": delta,
                            "mcnemar_exact": {
                                "holyc_only_correct": holyc_only,
                                "llama_only_correct": llama_only,
                                "discordant_count": holyc_only + llama_only,
                                "p_value": p_value,
                                "method": "exact_binomial_two_sided",
                            },
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def run_audit(input_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_significance_audit.py"),
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "eval_significance_audit_smoke_latest",
            "--max-holyc-loss-p",
            "0.1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-significance-") as tmp:
        tmp_path = Path(tmp)
        pass_report = tmp_path / "eval_compare_pass.json"
        write_report(pass_report, delta=0.0, holyc_only=1, llama_only=1, p_value=1.0)
        pass_out = tmp_path / "pass_out"
        completed = run_audit(pass_report, pass_out)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode
        payload = json.loads((pass_out / "eval_significance_audit_smoke_latest.json").read_text(encoding="utf-8"))
        junit = ET.parse(pass_out / "eval_significance_audit_smoke_latest_junit.xml").getroot()
        if payload["status"] != "pass" or payload["summary"]["scope_count"] != 2 or junit.attrib["failures"] != "0":
            print("passing_significance_audit_failed=true", file=sys.stderr)
            return 1
        if "Eval Significance Audit" not in (pass_out / "eval_significance_audit_smoke_latest.md").read_text(encoding="utf-8"):
            print("significance_markdown_missing=true", file=sys.stderr)
            return 1

        fail_report = tmp_path / "eval_compare_loss.json"
        write_report(fail_report, delta=-0.25, holyc_only=0, llama_only=6, p_value=0.0625)
        fail_out = tmp_path / "fail_out"
        completed = run_audit(fail_report, fail_out)
        if completed.returncode == 0:
            print("significant_holyc_loss_not_rejected=true", file=sys.stderr)
            return 1
        payload = json.loads((fail_out / "eval_significance_audit_smoke_latest.json").read_text(encoding="utf-8"))
        junit = ET.parse(fail_out / "eval_significance_audit_smoke_latest_junit.xml").getroot()
        reasons = {finding["kind"] for finding in payload["findings"]}
        if payload["status"] != "fail" or "significant_holyc_loss" not in reasons:
            print("significant_holyc_loss_missing=true", file=sys.stderr)
            return 1
        if int(junit.attrib["failures"]) < 1:
            print("significance_junit_failure_missing=true", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
