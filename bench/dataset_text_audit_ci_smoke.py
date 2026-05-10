#!/usr/bin/env python3
"""CI smoke test for dataset_text_audit.py."""

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
        str(BENCH / "dataset_text_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_text_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_text_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_text_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_text_audit_smoke_records_latest.csv"),
        "--junit",
        str(RESULTS / "dataset_text_audit_smoke_latest_junit.xml"),
        "--max-prompt-bytes",
        "4096",
        "--max-choice-bytes",
        "1024",
        "--max-line-bytes",
        "4096",
        "--fail-on-control-chars",
        "--fail-on-replacement-chars",
        "--fail-on-blank-text",
        "--fail-on-choice-label-prefixes",
        "--fail-on-findings",
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_text_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["text_field_count"] == 15, report
    assert report["control_char_field_count"] == 0, report
    assert report["replacement_char_field_count"] == 0, report
    assert report["choice_label_prefix_field_count"] == 0, report
    assert not report["findings"], report
    assert (RESULTS / "dataset_text_audit_smoke_latest.csv").exists()
    assert (RESULTS / "dataset_text_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_text_audit_smoke_records_latest.csv").exists()
    assert (RESULTS / "dataset_text_audit_smoke_latest_junit.xml").exists()


def assert_failure_gate() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-text-audit-") as tmp:
        sample = Path(tmp) / "bad.jsonl"
        sample.write_text(
            json.dumps(
                {
                    "id": "bad-control",
                    "dataset": "unit",
                    "split": "validation",
                    "prompt": "bad\x00prompt",
                    "choices": ["ok", "A. bad\ufffdchoice"],
                    "answer_index": 0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        output = Path(tmp) / "audit.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_text_audit.py"),
            "--input",
            str(sample),
            "--output",
            str(output),
            "--fail-on-control-chars",
            "--fail-on-replacement-chars",
            "--fail-on-choice-label-prefixes",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stdout + proc.stderr
        report = json.loads(output.read_text(encoding="utf-8"))
        kinds = {item["kind"] for item in report["findings"]}
        assert {"control_character", "unicode_replacement_character", "choice_label_prefix"}.issubset(kinds), report


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
