#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for eval_baseline.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_baseline(extra_args: list[str], output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "eval_baseline.py"),
            "--gold",
            str(ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"),
            "--dataset",
            "smoke-eval",
            "--split",
            "validation",
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "eval_baseline_smoke",
            *extra_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def require(condition: bool, message: str) -> int:
    if not condition:
        print(message, file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-eval-baseline-") as tmp:
        tmp_path = Path(tmp)
        passing = run_baseline(["--min-records", "3", "--max-majority-accuracy", "1.0"], tmp_path / "pass")
        if passing.returncode != 0:
            sys.stdout.write(passing.stdout)
            sys.stderr.write(passing.stderr)
            return passing.returncode
        report = json.loads((tmp_path / "pass" / "eval_baseline_smoke.json").read_text(encoding="utf-8"))
        junit = ET.parse(tmp_path / "pass" / "eval_baseline_smoke_junit.xml").getroot()
        if rc := require(report["status"] == "pass", "eval_baseline_smoke_not_pass=true"):
            return rc
        if rc := require(report["summary"]["records"] == 3, "eval_baseline_smoke_record_count=true"):
            return rc
        if rc := require(report["summary"]["majority_accuracy"] == 1.0, "eval_baseline_smoke_majority=true"):
            return rc
        if rc := require(round(report["summary"]["random_expected_accuracy"], 6) == 0.25, "eval_baseline_smoke_random=true"):
            return rc
        if rc := require(junit.attrib["failures"] == "0", "eval_baseline_smoke_junit=true"):
            return rc
        if rc := require(
            "Uniform-random expected accuracy"
            in (tmp_path / "pass" / "eval_baseline_smoke.md").read_text(encoding="utf-8"),
            "eval_baseline_smoke_markdown=true",
        ):
            return rc

        failing = run_baseline(["--max-majority-accuracy", "0.5"], tmp_path / "fail")
        if rc := require(failing.returncode == 1, "eval_baseline_skew_gate_not_failed=true"):
            return rc
        fail_report = json.loads((tmp_path / "fail" / "eval_baseline_smoke.json").read_text(encoding="utf-8"))
        gates = {finding["gate"] for finding in fail_report["findings"]}
        if rc := require("max_majority_accuracy" in gates, "eval_baseline_missing_skew_finding=true"):
            return rc
    print("eval_baseline_ci_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
