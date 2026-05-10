#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval margin audits."""

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


def margin(mean: float, p10: float, low_rate: float = 0.0, scored: int = 10) -> dict[str, object]:
    return {
        "low_margin_count": int(scored * low_rate),
        "low_margin_rate": low_rate,
        "low_margin_threshold": 0.1,
        "mean_correct_margin": mean,
        "mean_margin": mean,
        "mean_wrong_margin": 0.0,
        "median_margin": mean,
        "min_margin": min(p10, mean),
        "p10_margin": p10,
        "score_coverage": scored / 10,
        "scored_count": scored,
        "total_count": 10,
    }


def write_eval_report(path: Path, *, holyc_mean: float, llama_mean: float, holyc_p10: float = 0.4) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset": "smoke-eval",
                "split": "validation",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "summary": {
                    "holyc_margin_metrics": margin(holyc_mean, holyc_p10),
                    "llama_margin_metrics": margin(llama_mean, 0.5),
                    "dataset_breakdown": [
                        {
                            "dataset": "arc-smoke",
                            "split": "validation",
                            "holyc_margin_metrics": margin(holyc_mean, holyc_p10),
                            "llama_margin_metrics": margin(llama_mean, 0.5),
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
            str(ROOT / "bench" / "eval_margin_audit.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-margin-audit-ci-") as tmp:
        tmp_path = Path(tmp)
        passing_report = tmp_path / "eval_compare_pass.json"
        write_eval_report(passing_report, holyc_mean=0.7, llama_mean=0.72)
        passed = run_audit(
            tmp_path,
            passing_report,
            "margin_pass",
            "--min-score-coverage",
            "1.0",
            "--min-mean-margin",
            "0.5",
            "--max-holyc-mean-margin-loss",
            "0.05",
            "--fail-on-findings",
        )
        if passed.returncode != 0:
            sys.stdout.write(passed.stdout)
            sys.stderr.write(passed.stderr)
            return passed.returncode

        payload = json.loads((tmp_path / "margin_pass.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_pass_status"):
            return rc
        if rc := require(payload["margin_summary_count"] == 4, "unexpected_margin_summary_count"):
            return rc
        if rc := require((tmp_path / "margin_pass.csv").read_text(encoding="utf-8").startswith("severity,"), "missing_csv_header"):
            return rc
        if rc := require((tmp_path / "margin_pass_summaries.csv").read_text(encoding="utf-8").startswith("source,"), "missing_summary_csv_header"):
            return rc
        junit = ET.parse(tmp_path / "margin_pass_junit.xml").getroot()
        if rc := require(junit.attrib.get("name") == "holyc_eval_margin_audit", "missing_junit_suite"):
            return rc
        if rc := require(junit.attrib.get("failures") == "0", "unexpected_pass_junit_failure"):
            return rc

        failing_report = tmp_path / "eval_compare_fail.json"
        write_eval_report(failing_report, holyc_mean=0.15, llama_mean=0.75, holyc_p10=0.02)
        failed = run_audit(
            tmp_path,
            failing_report,
            "margin_fail",
            "--min-mean-margin",
            "0.5",
            "--min-p10-margin",
            "0.1",
            "--max-holyc-mean-margin-loss",
            "0.2",
            "--include-dataset-breakdown",
            "--fail-on-findings",
        )
        if rc := require(failed.returncode == 2, "expected_failure_status"):
            return rc
        failed_payload = json.loads((tmp_path / "margin_fail.json").read_text(encoding="utf-8"))
        messages = [finding["message"] for finding in failed_payload["findings"]]
        if rc := require(any("mean_margin" in message for message in messages), "missing_mean_margin_finding"):
            return rc
        if rc := require(any("p10_margin" in message for message in messages), "missing_p10_margin_finding"):
            return rc
        if rc := require(any("HolyC mean margin loss" in message for message in messages), "missing_loss_finding"):
            return rc
        failed_junit = ET.parse(tmp_path / "margin_fail_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_junit"):
            return rc

    print("eval_margin_audit_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
