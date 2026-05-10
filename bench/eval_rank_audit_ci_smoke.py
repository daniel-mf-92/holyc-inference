#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval rank audits."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def rank(top1: float, mrr: float, scored: int = 10) -> dict[str, object]:
    return {
        "mean_gold_rank": 1.0 / mrr if mrr else 99.0,
        "mean_reciprocal_rank": mrr,
        "score_coverage": scored / 10,
        "scored_count": scored,
        "top_1_accuracy": top1,
        "top_2_accuracy": max(top1, 0.9),
        "top_3_accuracy": max(top1, 1.0),
        "total_count": 10,
    }


def write_eval_report(path: Path, *, holyc_top1: float, llama_top1: float, holyc_mrr: float = 0.9) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_rank_metrics": rank(holyc_top1, holyc_mrr),
                    "llama_rank_metrics": rank(llama_top1, 0.92),
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "holyc_rank_metrics": rank(holyc_top1, holyc_mrr),
                            "llama_rank_metrics": rank(llama_top1, 0.92),
                        }
                    ],
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_audit(output_dir: Path, report: Path, stem: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_rank_audit.py"),
            str(report),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            stem,
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-rank-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "eval_compare_pass.json"
        write_eval_report(passing_report, holyc_top1=0.9, llama_top1=0.91, holyc_mrr=0.91)
        passed = run_audit(
            tmp_path,
            passing_report,
            "rank_pass",
            "--min-score-coverage",
            "1.0",
            "--min-top-1-accuracy",
            "0.85",
            "--min-mean-reciprocal-rank",
            "0.9",
            "--max-holyc-top-1-loss",
            "0.05",
            "--max-holyc-mrr-loss",
            "0.05",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode

        payload = json.loads((tmp_path / "rank_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["rank_summary_count"] == 4, "unexpected_rank_summary_count"):
            return rc
        if rc := require((tmp_path / "rank_pass.csv").read_text(encoding="utf-8").startswith("severity,"), "missing_csv_header"):
            return rc
        if rc := require((tmp_path / "rank_pass_summaries.csv").read_text(encoding="utf-8").startswith("source,"), "missing_summary_csv_header"):
            return rc
        junit = ET.parse(tmp_path / "rank_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_rank_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        failing_report = tmp_path / "eval_compare_fail.json"
        write_eval_report(failing_report, holyc_top1=0.4, llama_top1=0.9, holyc_mrr=0.5)
        failed = run_audit(
            tmp_path,
            failing_report,
            "rank_fail",
            "--min-top-1-accuracy",
            "0.8",
            "--min-mean-reciprocal-rank",
            "0.8",
            "--max-holyc-top-1-loss",
            "0.2",
            "--max-holyc-mrr-loss",
            "0.2",
            "--include-dataset-breakdown",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "rank_fail.json").read_text(encoding="utf-8"))
        metrics = {finding["metric"] for finding in failed_payload["findings"]}
        if rc := require("top_1_accuracy" in metrics, "missing_top1_finding"):
            return rc
        if rc := require("mean_reciprocal_rank" in metrics, "missing_mrr_finding"):
            return rc
        if rc := require("top_1_accuracy_loss_vs_llama" in metrics, "missing_top1_loss_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "rank_fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_rank_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
