#!/usr/bin/env python3
"""Stdlib-only CI smoke gate for QEMU build throughput scorecard."""

from __future__ import annotations

import csv
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-throughput-scorecard-ci-") as tmp:
        tmp_path = Path(tmp)
        artifact = tmp_path / "qemu_prompt_bench_latest.json"
        output_dir = tmp_path / "out"
        artifact.write_text(
            json.dumps(
                {
                    "benchmarks": [
                        {
                            "build": "base",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "phase": "measured",
                            "exit_class": "ok",
                            "timed_out": False,
                            "tokens": 32,
                            "elapsed_us": 200000,
                            "wall_elapsed_us": 400000,
                            "tok_per_s": 160.0,
                            "wall_tok_per_s": 80.0,
                        },
                        {
                            "build": "base",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "phase": "measured",
                            "exit_class": "ok",
                            "timed_out": False,
                            "tokens": 8,
                            "elapsed_us": 100000,
                            "wall_elapsed_us": 200000,
                            "tok_per_s": 80.0,
                            "wall_tok_per_s": 40.0,
                        },
                        {
                            "build": "candidate",
                            "profile": "ci-airgap-smoke",
                            "model": "synthetic-smoke",
                            "quantization": "Q4_0",
                            "phase": "measured",
                            "exit_class": "ok",
                            "timed_out": False,
                            "tokens": 48,
                            "elapsed_us": 240000,
                            "wall_elapsed_us": 480000,
                            "tok_per_s": 200.0,
                            "wall_tok_per_s": 100.0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(ROOT / "bench" / "qemu_build_throughput_scorecard.py"),
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "qemu_build_throughput_scorecard_latest",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        payload = json.loads((output_dir / "qemu_build_throughput_scorecard_latest.json").read_text(encoding="utf-8"))
        if rc := require(payload["status"] == "pass", "unexpected_scorecard_status"):
            return rc
        if rc := require(payload["summary"]["groups"] == 2, "unexpected_scorecard_groups"):
            return rc
        rows = list(csv.DictReader((output_dir / "qemu_build_throughput_scorecard_latest.csv").open(encoding="utf-8")))
        if rc := require(rows[0]["build"] == "base", "unexpected_first_scorecard_build"):
            return rc
        if rc := require(rows[0]["measured_rows"] == "2", "unexpected_base_scorecard_rows"):
            return rc
        if rc := require(rows[0]["mean_tok_per_s"] == "120.0", "unexpected_base_mean_rate"):
            return rc
        if rc := require(rows[0]["stdev_tok_per_s"] == "40.0", "unexpected_base_stdev_rate"):
            return rc
        if rc := require(rows[0]["cv_tok_per_s"] == "0.3333333333333333", "unexpected_base_cv_rate"):
            return rc
        if rc := require(rows[0]["weighted_tok_per_s"] == "133.33333333333334", "unexpected_base_weighted_rate"):
            return rc
        if rc := require(rows[0]["cv_wall_tok_per_s"] == "0.3333333333333333", "unexpected_base_wall_cv_rate"):
            return rc
        if rc := require(rows[0]["weighted_wall_tok_per_s"] == "66.66666666666667", "unexpected_base_weighted_wall_rate"):
            return rc
        if rc := require(rows[1]["mean_tok_per_s"] == "200.0", "unexpected_candidate_scorecard_rate"):
            return rc
        findings = list(csv.DictReader((output_dir / "qemu_build_throughput_scorecard_latest_findings.csv").open(encoding="utf-8")))
        if rc := require(findings == [], "unexpected_scorecard_findings"):
            return rc
        junit = ET.parse(output_dir / "qemu_build_throughput_scorecard_latest_junit.xml").getroot()
        if rc := require(junit.attrib["failures"] == "0", "unexpected_scorecard_junit_failure"):
            return rc

        bad_completed = subprocess.run(command + ["--min-rows", "2"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if rc := require(bad_completed.returncode == 1, "expected_scorecard_min_rows_failure"):
            return rc
        bad_payload = json.loads((output_dir / "qemu_build_throughput_scorecard_latest.json").read_text(encoding="utf-8"))
        bad_kinds = {finding["kind"] for finding in bad_payload["findings"]}
        if rc := require("min_rows" in bad_kinds, "missing_scorecard_min_rows_finding"):
            return rc

        unstable_completed = subprocess.run(command + ["--max-cv", "0.1"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if rc := require(unstable_completed.returncode == 1, "expected_scorecard_cv_failure"):
            return rc
        unstable_payload = json.loads((output_dir / "qemu_build_throughput_scorecard_latest.json").read_text(encoding="utf-8"))
        unstable_kinds = {finding["kind"] for finding in unstable_payload["findings"]}
        if rc := require("max_cv" in unstable_kinds, "missing_scorecard_cv_finding"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
