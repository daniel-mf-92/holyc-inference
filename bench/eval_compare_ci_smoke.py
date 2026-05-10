#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for HolyC-vs-llama eval comparison."""

from __future__ import annotations

import csv
import json
import math
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


def eval_compare_command(output_dir: Path, stem: str, *extra_args: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "bench" / "eval_compare.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-eval-compare-ci-") as tmp:
        output_dir = Path(tmp)
        stem = "eval_compare_smoke"
        completed = run_command(
            eval_compare_command(
                output_dir,
                stem,
                "--min-holyc-accuracy",
                "1.0",
                "--min-agreement",
                "1.0",
                "--max-accuracy-drop",
                "0.0",
                "--max-mcnemar-loss-p",
                "0.05",
                "--max-holyc-mean-nll",
                "0.01",
                "--max-holyc-nll-delta",
                "0.0",
                "--max-holyc-score-tie-rate",
                "0.0",
                "--fail-on-regression",
            )
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        summary = report["summary"]
        if rc := require(report["status"] == "pass", "unexpected_eval_status"):
            return rc
        if rc := require(summary["record_count"] == 3, "unexpected_eval_record_count"):
            return rc
        if rc := require(summary["holyc_accuracy"] == 1.0, "unexpected_holyc_accuracy"):
            return rc
        if rc := require(summary["llama_accuracy"] == 1.0, "unexpected_llama_accuracy"):
            return rc
        if rc := require(summary["agreement"] == 1.0, "unexpected_eval_agreement"):
            return rc
        if rc := require(summary["paired_correctness"]["both_correct"] == 3, "unexpected_paired_counts"):
            return rc
        if rc := require(summary["mcnemar_exact"]["p_value"] == 1.0, "unexpected_mcnemar_p_value"):
            return rc
        if rc := require(summary["holyc_nll_metrics"]["scored_count"] == 1, "unexpected_holyc_nll_count"):
            return rc
        if rc := require(
            math.isclose(summary["holyc_nll_metrics"]["mean_gold_nll"], 0.000697149257004621),
            "unexpected_holyc_mean_nll",
        ):
            return rc
        if rc := require(
            summary["holyc_tie_metrics"]["tie_rate"] == 0.0,
            "unexpected_holyc_tie_rate",
        ):
            return rc
        if rc := require(
            summary["confidence_intervals"]["holyc_accuracy"]["method"] == "wilson",
            "missing_confidence_interval",
        ):
            return rc

        expected_files = [
            f"{stem}.json",
            f"{stem}.md",
            f"{stem}.csv",
            f"{stem}_breakdown.csv",
            f"{stem}_confusion.csv",
            f"{stem}_calibration_bins.csv",
            f"{stem}_margins.csv",
            f"{stem}_nll.csv",
            f"{stem}_rank.csv",
            f"{stem}_score_ties.csv",
            f"{stem}_disagreements.csv",
            f"{stem}_junit.xml",
        ]
        if rc := require(all((output_dir / name).exists() for name in expected_files), "missing_eval_artifact"):
            return rc
        if rc := require(
            "Eval Compare Report" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"),
            "missing_eval_markdown",
        ):
            return rc

        row_csv = load_csv_rows(output_dir / f"{stem}.csv")
        if rc := require(len(row_csv) == 3, "unexpected_eval_csv_rows"):
            return rc
        if rc := require(
            all(row["holyc_correct"] == "True" and row["llama_correct"] == "True" for row in row_csv),
            "unexpected_eval_csv_correctness",
        ):
            return rc

        breakdown_csv = load_csv_rows(output_dir / f"{stem}_breakdown.csv")
        if rc := require(len(breakdown_csv) == 3, "unexpected_eval_breakdown_rows"):
            return rc
        if rc := require(
            {row["dataset"] for row in breakdown_csv}
            == {"arc-smoke", "hellaswag-smoke", "truthfulqa-smoke"},
            "unexpected_eval_breakdown_datasets",
        ):
            return rc

        nll_csv = load_csv_rows(output_dir / f"{stem}_nll.csv")
        if rc := require(any(row["engine"] == "holyc" for row in nll_csv), "missing_holyc_nll_csv"):
            return rc
        rank_csv = load_csv_rows(output_dir / f"{stem}_rank.csv")
        if rc := require(any(row["engine"] == "holyc" for row in rank_csv), "missing_holyc_rank_csv"):
            return rc
        if rc := require(
            any(row["scope"] == "dataset_split" and row["dataset"] == "arc-smoke" for row in rank_csv),
            "missing_dataset_rank_csv",
        ):
            return rc
        ties_csv = load_csv_rows(output_dir / f"{stem}_score_ties.csv")
        if rc := require(any(row["engine"] == "llama" for row in ties_csv), "missing_llama_tie_csv"):
            return rc
        disagreement_csv = load_csv_rows(output_dir / f"{stem}_disagreements.csv")
        if rc := require(disagreement_csv == [], "unexpected_eval_disagreements"):
            return rc

        junit_root = ET.parse(output_dir / f"{stem}_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_eval_compare", "missing_eval_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_eval_junit_failures"):
            return rc

        fail_stem = "eval_compare_gate_fail"
        failed = run_command(
            eval_compare_command(
                output_dir,
                fail_stem,
                "--min-holyc-nll-coverage",
                "1.0",
                "--fail-on-regression",
            ),
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_eval_gate_failure"):
            return rc
        failed_report = json.loads((output_dir / f"{fail_stem}.json").read_text(encoding="utf-8"))
        if rc := require(failed_report["status"] == "fail", "unexpected_failed_eval_status"):
            return rc
        if rc := require(
            failed_report["regressions"][0]["metric"] == "holyc_nll_score_coverage",
            "unexpected_failed_eval_metric",
        ):
            return rc
        failed_junit = ET.parse(output_dir / f"{fail_stem}_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_eval_junit"):
            return rc

    print("eval_compare_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
