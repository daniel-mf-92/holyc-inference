#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_slice_manifest.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-slice-manifest-") as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "dataset_slice_manifest_smoke_latest.json"
        csv_path = tmp_path / "dataset_slice_manifest_smoke_latest.csv"
        record_csv = tmp_path / "dataset_slice_manifest_smoke_records_latest.csv"
        markdown = tmp_path / "dataset_slice_manifest_smoke_latest.md"
        junit = tmp_path / "dataset_slice_manifest_smoke_latest_junit.xml"
        command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_slice_manifest.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(output),
            "--csv",
            str(csv_path),
            "--record-csv",
            str(record_csv),
            "--markdown",
            str(markdown),
            "--junit",
            str(junit),
            "--require-slice",
            "arc-smoke:validation",
            "--require-slice",
            "hellaswag-smoke:validation",
            "--require-slice",
            "truthfulqa-smoke:validation",
            "--min-total-slices",
            "3",
            "--min-records-per-slice",
            "1",
            "--fail-on-findings",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads(output.read_text(encoding="utf-8"))
        if rc := require(report["status"] == "pass", "unexpected_slice_manifest_status"):
            return rc
        if rc := require(report["record_count"] == 3, "unexpected_slice_manifest_records"):
            return rc
        if rc := require(report["slice_count"] == 3, "unexpected_slice_manifest_slices"):
            return rc
        if rc := require({row["record_count"] for row in report["slices"]} == {1}, "unexpected_slice_counts"):
            return rc
        if rc := require(all(len(row["slice_sha256"]) == 64 for row in report["slices"]), "missing_slice_hash"):
            return rc
        if rc := require("Eval Dataset Slice Manifest" in markdown.read_text(encoding="utf-8"), "missing_markdown"):
            return rc
        if rc := require(len(list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))) == 3, "bad_csv"):
            return rc
        if rc := require(len(list(csv.DictReader(record_csv.open(encoding="utf-8", newline="")))) == 3, "bad_record_csv"):
            return rc
        junit_root = ET.parse(junit).getroot()
        if rc := require(junit_root.attrib.get("name") == "holyc_dataset_slice_manifest", "bad_junit"):
            return rc
        if rc := require(junit_root.attrib.get("failures") == "0", "unexpected_junit_failure"):
            return rc

        missing_output = tmp_path / "missing.json"
        missing_command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_slice_manifest.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(missing_output),
            "--require-slice",
            "missing:validation",
            "--fail-on-findings",
        ]
        completed = subprocess.run(
            missing_command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode == 0:
            print("missing_required_slice_not_rejected=true", file=sys.stderr)
            return 1
        missing_report = json.loads(missing_output.read_text(encoding="utf-8"))
        kinds = {finding["kind"] for finding in missing_report["findings"]}
        if rc := require("missing_required_slice" in kinds, "missing_required_slice_finding"):
            return rc

    print("dataset_slice_manifest_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
