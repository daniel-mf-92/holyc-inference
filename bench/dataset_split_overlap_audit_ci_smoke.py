#!/usr/bin/env python3
"""CI smoke test for dataset_split_overlap_audit.py."""

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
        str(BENCH / "dataset_split_overlap_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_split_overlap_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_split_overlap_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_split_overlap_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_split_overlap_audit_smoke_records_latest.csv"),
        "--junit",
        str(RESULTS / "dataset_split_overlap_audit_smoke_latest_junit.xml"),
        "--fail-on-prompt-overlap",
        "--fail-on-payload-overlap",
        "--fail-on-findings",
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_split_overlap_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["prompt_overlap_record_count"] == 0, report
    assert report["payload_overlap_record_count"] == 0, report
    assert not report["findings"], report
    assert (RESULTS / "dataset_split_overlap_audit_smoke_latest.csv").exists()
    assert (RESULTS / "dataset_split_overlap_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_split_overlap_audit_smoke_records_latest.csv").exists()
    assert (RESULTS / "dataset_split_overlap_audit_smoke_latest_junit.xml").exists()


def assert_failure_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-split-overlap-audit-") as tmp:
        sample = Path(tmp) / "leaky.jsonl"
        rows = [
            {
                "id": "train-copy",
                "dataset": "unit",
                "split": "train",
                "prompt": "Which object measures temperature?",
                "choices": ["thermometer", "ruler", "scale", "compass"],
                "answer_index": 0,
            },
            {
                "id": "validation-copy",
                "dataset": "unit",
                "split": "validation",
                "prompt": "Which object measures temperature?",
                "choices": ["thermometer", "ruler", "scale", "compass"],
                "answer_index": 0,
            },
        ]
        sample.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
        output = Path(tmp) / "audit.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_split_overlap_audit.py"),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--fail-on-prompt-overlap",
            "--fail-on-payload-overlap",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert {"prompt_split_overlap", "payload_split_overlap"}.issubset(kinds), report


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
