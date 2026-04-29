#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for HolyC-vs-llama perplexity comparison."""

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
HOLYC = ROOT / "bench" / "eval" / "samples" / "holyc_smoke_logprobs.jsonl"
LLAMA = ROOT / "bench" / "eval" / "samples" / "llama_smoke_logprobs.jsonl"


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


def perplexity_command(output_dir: Path, stem: str, *extra_args: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "bench" / "perplexity_compare.py"),
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
    with tempfile.TemporaryDirectory(prefix="holyc-perplexity-compare-ci-") as tmp:
        output_dir = Path(tmp)
        stem = "perplexity_compare_smoke"
        completed = run_command(
            perplexity_command(
                output_dir,
                stem,
                "--min-record-count",
                "3",
                "--min-token-count",
                "11",
                "--max-nll-delta",
                "0.02",
                "--max-perplexity-ratio",
                "1.05",
                "--max-p95-abs-record-nll-delta",
                "0.05",
                "--max-p95-record-nll-delta",
                "0.05",
                "--max-record-nll-delta",
                "0.10",
                "--fail-on-regression",
            )
        )
        if completed.returncode != 0:
            return completed.returncode

        report = json.loads((output_dir / f"{stem}.json").read_text(encoding="utf-8"))
        summary = report["summary"]
        if rc := require(report["status"] == "pass", "unexpected_perplexity_status"):
            return rc
        if rc := require(summary["record_count"] == 3, "unexpected_perplexity_record_count"):
            return rc
        if rc := require(summary["holyc"]["token_count"] == 11, "unexpected_holyc_token_count"):
            return rc
        if rc := require(summary["llama"]["token_count"] == 11, "unexpected_llama_token_count"):
            return rc
        if rc := require(summary["token_count_mismatches"] == 0, "unexpected_token_mismatch_count"):
            return rc
        if rc := require(
            math.isclose(summary["nll_delta_holyc_minus_llama"], 0.005454496131887809),
            "unexpected_perplexity_nll_delta",
        ):
            return rc
        if rc := require(
            math.isclose(summary["p95_record_nll_delta"], 0.010000000000000009),
            "unexpected_perplexity_p95_delta",
        ):
            return rc
        if rc := require(report["min_record_count"] == 3, "missing_min_record_count_gate"):
            return rc
        if rc := require(report["min_token_count"] == 11, "missing_min_token_count_gate"):
            return rc

        expected_files = [
            f"{stem}.json",
            f"{stem}.md",
            f"{stem}.csv",
            f"{stem}_breakdown.csv",
            f"{stem}_junit.xml",
        ]
        if rc := require(all((output_dir / name).exists() for name in expected_files), "missing_perplexity_artifact"):
            return rc
        if rc := require(
            "Perplexity Compare Report" in (output_dir / f"{stem}.md").read_text(encoding="utf-8"),
            "missing_perplexity_markdown",
        ):
            return rc

        rows_csv = load_csv_rows(output_dir / f"{stem}.csv")
        if rc := require(len(rows_csv) == 3, "unexpected_perplexity_csv_rows"):
            return rc
        breakdown_csv = load_csv_rows(output_dir / f"{stem}_breakdown.csv")
        if rc := require(len(breakdown_csv) == 1, "unexpected_perplexity_breakdown_rows"):
            return rc
        if rc := require(breakdown_csv[0]["dataset"] == "smoke-eval", "unexpected_perplexity_breakdown_dataset"):
            return rc

        junit_root = ET.parse(output_dir / f"{stem}_junit.xml").getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_perplexity_compare", "missing_perplexity_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_perplexity_junit_failures"):
            return rc

        fail_stem = "perplexity_compare_gate_fail"
        failed = run_command(
            perplexity_command(
                output_dir,
                fail_stem,
                "--min-record-count",
                "4",
                "--fail-on-regression",
            ),
            expected_failure=True,
        )
        if rc := require(failed.returncode == 1, "expected_perplexity_gate_failure"):
            return rc
        failed_report = json.loads((output_dir / f"{fail_stem}.json").read_text(encoding="utf-8"))
        if rc := require(failed_report["status"] == "fail", "unexpected_failed_perplexity_status"):
            return rc
        if rc := require(
            failed_report["regressions"][0]["metric"] == "record_count",
            "unexpected_failed_perplexity_metric",
        ):
            return rc
        failed_junit = ET.parse(output_dir / f"{fail_stem}_junit.xml").getroot()
        if rc := require(failed_junit.attrib.get("failures") == "1", "unexpected_failed_perplexity_junit"):
            return rc

    print("perplexity_compare_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
