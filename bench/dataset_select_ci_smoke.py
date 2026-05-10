#!/usr/bin/env python3
"""Stdlib-only smoke gate for dataset_select.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "bench" / "datasets" / "samples" / "smoke_eval.jsonl"


def require(condition: bool, message: str) -> int:
    if not condition:
        print(f"{message}=true", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="holyc-dataset-select-") as tmp:
        tmp_path = Path(tmp)
        selected = tmp_path / "selected.jsonl"
        manifest = tmp_path / "manifest.json"
        selected_csv = tmp_path / "selected.csv"
        command = [
            sys.executable,
            str(ROOT / "bench" / "dataset_select.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(selected),
            "--manifest",
            str(manifest),
            "--csv",
            str(selected_csv),
            "--max-records-per-slice",
            "1",
            "--balance-answer",
            "--fail-on-findings",
        ]
        completed = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode

        report = json.loads(manifest.read_text(encoding="utf-8"))
        selected_rows = [json.loads(line) for line in selected.read_text(encoding="utf-8").splitlines()]
        csv_rows = list(csv.DictReader(selected_csv.open(encoding="utf-8", newline="")))
        if rc := require(report["status"] == "pass", "unexpected_dataset_select_status"):
            return rc
        if rc := require(report["candidate_count"] == 3, "unexpected_candidate_count"):
            return rc
        if rc := require(report["selected_count"] == 3, "unexpected_selected_count"):
            return rc
        if rc := require(report["slice_count"] == 3, "unexpected_slice_count"):
            return rc
        if rc := require(len(report["selected_sha256"]) == 64, "missing_selected_hash"):
            return rc
        if rc := require(len(selected_rows) == 3, "bad_selected_jsonl"):
            return rc
        if rc := require(all("record_id" in row and "answer_index" in row for row in selected_rows), "bad_selected_rows"):
            return rc
        if rc := require(len(csv_rows) == 3, "bad_selected_csv"):
            return rc
        if rc := require(
            all(len(row["payload_sha256"]) == 64 and len(row["rank_sha256"]) == 64 for row in csv_rows),
            "bad_selected_csv_hashes",
        ):
            return rc

    print("dataset_select_ci_smoke=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
