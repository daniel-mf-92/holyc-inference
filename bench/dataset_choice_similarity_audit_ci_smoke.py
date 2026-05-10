#!/usr/bin/env python3
"""CI smoke test for dataset_choice_similarity_audit.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
RESULTS = BENCH / "results" / "datasets"
SAMPLE = BENCH / "datasets" / "samples" / "smoke_eval.jsonl"


def run_audit(*extra_args: str) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(BENCH / "dataset_choice_similarity_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_latest.csv"),
        "--pair-csv",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_pairs_latest.csv"),
        "--findings-csv",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_latest_findings.csv"),
        "--junit",
        str(RESULTS / "dataset_choice_similarity_audit_smoke_latest_junit.xml"),
        "--min-unique-choices",
        "4",
        "--max-pair-similarity",
        "0.95",
        "--fail-duplicate-normalized",
        *extra_args,
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_choice_similarity_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["error_count"] == 0, report
    assert not report["findings"], report
    assert {row["record_id"] for row in report["records"]} == {
        "smoke-arc-1",
        "smoke-hellaswag-1",
        "smoke-truthfulqa-1",
    }, report
    assert all(row["unique_normalized_choices"] == 4 for row in report["records"]), report
    assert all(row["duplicate_choice_pairs"] == 0 for row in report["records"]), report
    assert (RESULTS / "dataset_choice_similarity_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_choice_similarity_audit_smoke_latest.csv").exists()
    pair_csv = RESULTS / "dataset_choice_similarity_audit_smoke_pairs_latest.csv"
    assert pair_csv.exists()
    assert "left_normalized" in pair_csv.read_text(encoding="utf-8").splitlines()[0]
    assert (RESULTS / "dataset_choice_similarity_audit_smoke_latest_findings.csv").exists()
    assert (RESULTS / "dataset_choice_similarity_audit_smoke_latest_junit.xml").exists()


def assert_gate_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-choice-similarity-audit-") as tmp:
        source = Path(tmp) / "duplicate.jsonl"
        output = Path(tmp) / "duplicate.json"
        source.write_text(
            json.dumps(
                {
                    "id": "duplicate-choice",
                    "dataset": "arc-smoke",
                    "split": "validation",
                    "question": "Which object measures temperature?",
                    "choices": [
                        {"label": "A", "text": "thermometer"},
                        {"label": "B", "text": "Thermometer!"},
                        {"label": "C", "text": "ruler"},
                        {"label": "D", "text": "compass"},
                    ],
                    "answerKey": "A",
                    "provenance": "synthetic duplicate smoke row",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        cmd = [
            sys.executable,
            str(BENCH / "dataset_choice_similarity_audit.py"),
            "--input",
            str(source),
            "--output",
            str(output),
            "--min-unique-choices",
            "4",
            "--max-pair-similarity",
            "0.95",
            "--fail-duplicate-normalized",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert {"min_unique_choices", "duplicate_normalized_choice", "choice_similarity_exceeded"} <= kinds, report


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    proc = run_audit()
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    assert_smoke_outputs()
    assert_gate_failure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
