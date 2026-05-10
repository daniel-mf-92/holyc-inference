#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_index.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
RESULTS = BENCH / "results" / "datasets"


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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-index-") as tmp:
        output_dir = Path(tmp) / "out"
        passing = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_index.py"),
                "--input",
                str(RESULTS),
                "--output-dir",
                str(output_dir),
                "--require-artifact-type",
                "curated_manifest",
                "--require-artifact-type",
                "pack_manifest",
                "--require-artifact-type",
                "inspect_report",
                "--require-dataset-split",
                "smoke-eval:validation",
                "--fail-on-coverage",
                "--fail-on-dataset-split-coverage",
                "--fail-on-findings",
            ]
        )
        if passing.returncode != 0:
            return passing.returncode

        report = json.loads((output_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_dataset_index_status"):
            return rc
        if rc := require(len(report["artifacts"]) >= 3, "missing_index_artifacts"):
            return rc
        if rc := require(not report["artifact_type_coverage_violations"], "unexpected_artifact_type_violation"):
            return rc
        if rc := require(not report["dataset_split_coverage_violations"], "unexpected_dataset_split_violation"):
            return rc
        if rc := require((output_dir / "dataset_index_latest.csv").exists(), "missing_index_csv"):
            return rc
        if rc := require((output_dir / "dataset_index_latest.md").exists(), "missing_index_markdown"):
            return rc
        root = ET.parse(output_dir / "dataset_index_junit_latest.xml").getroot()
        if rc := require(root.attrib.get("name") == "holyc_dataset_index", "missing_index_junit"):
            return rc
        if rc := require(root.attrib.get("failures") == "0", "unexpected_index_junit_failures"):
            return rc

        missing_type_dir = Path(tmp) / "missing-type"
        missing_type = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_index.py"),
                "--input",
                str(RESULTS / "smoke_curated.inspect.json"),
                "--output-dir",
                str(missing_type_dir),
                "--require-artifact-type",
                "curated_manifest",
                "--fail-on-coverage",
            ],
            expected_failure=True,
        )
        if rc := require(missing_type.returncode == 1, "missing_artifact_type_gate_did_not_fail"):
            return rc
        missing_type_report = json.loads((missing_type_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
        gates = {row["artifact_type"] for row in missing_type_report["artifact_type_coverage_violations"]}
        if rc := require(gates == {"curated_manifest"}, "missing_artifact_type_gate_not_reported"):
            return rc

        missing_split_dir = Path(tmp) / "missing-split"
        missing_split = run_command(
            [
                sys.executable,
                str(BENCH / "dataset_index.py"),
                "--input",
                str(RESULTS),
                "--output-dir",
                str(missing_split_dir),
                "--require-dataset-split",
                "missing-eval:test",
                "--fail-on-dataset-split-coverage",
            ],
            expected_failure=True,
        )
        if rc := require(missing_split.returncode == 1, "missing_dataset_split_gate_did_not_fail"):
            return rc
        missing_split_report = json.loads((missing_split_dir / "dataset_index_latest.json").read_text(encoding="utf-8"))
        split_gates = {
            f"{row['dataset']}:{row['split']}" for row in missing_split_report["dataset_split_coverage_violations"]
        }
        if rc := require(split_gates == {"missing-eval:test"}, "missing_dataset_split_gate_not_reported"):
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
