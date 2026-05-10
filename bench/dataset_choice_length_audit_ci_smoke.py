#!/usr/bin/env python3
"""CI smoke test for dataset_choice_length_audit.py."""

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


def run_smoke() -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(BENCH / "dataset_choice_length_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_choice_length_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_choice_length_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_choice_length_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_choice_length_audit_smoke_records_latest.csv"),
        "--junit",
        str(RESULTS / "dataset_choice_length_audit_smoke_latest_junit.xml"),
        "--max-choice-byte-span",
        "128",
        "--max-answer-delta-bytes",
        "128",
        "--max-answer-to-mean-other-ratio",
        "8.0",
        "--min-answer-to-mean-other-ratio",
        "0.125",
        "--fail-on-findings",
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_choice_length_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["error_count"] == 0, report
    assert report["max_choice_byte_span"] <= 128, report
    assert not report["findings"], report
    assert (RESULTS / "dataset_choice_length_audit_smoke_latest.csv").exists()
    assert (RESULTS / "dataset_choice_length_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_choice_length_audit_smoke_records_latest.csv").exists()
    assert (RESULTS / "dataset_choice_length_audit_smoke_latest_junit.xml").exists()


def assert_failure_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-choice-length-audit-") as tmp:
        sample = Path(tmp) / "bad.jsonl"
        sample.write_text(
            json.dumps(
                {
                    "id": "length-cue",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "Pick the correct answer.",
                    "choices": ["A", "B", "C", "This answer is deliberately much longer than every distractor."],
                    "answer_index": 3,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output = Path(tmp) / "audit.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_choice_length_audit.py"),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--max-answer-delta-bytes",
            "16",
            "--max-answer-to-mean-other-ratio",
            "4.0",
            "--fail-on-unique-longest-answer",
            "--fail-on-findings",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert {"answer_length_delta_exceeded", "answer_length_ratio_high", "answer_unique_longest"}.issubset(kinds), report


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    proc = run_smoke()
    if proc.returncode:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    assert_smoke_outputs()
    assert_failure_gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
