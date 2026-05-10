#!/usr/bin/env python3
"""CI smoke test for dataset_mix_audit.py."""

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
        str(BENCH / "dataset_mix_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_mix_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_mix_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_mix_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_mix_audit_smoke_records_latest.csv"),
        "--findings-csv",
        str(RESULTS / "dataset_mix_audit_smoke_latest_findings.csv"),
        "--junit",
        str(RESULTS / "dataset_mix_audit_smoke_latest_junit.xml"),
        "--min-datasets",
        "3",
        "--require-dataset-split",
        "hellaswag-smoke:validation",
        "--require-dataset-split",
        "arc-smoke:validation",
        "--require-dataset-split",
        "truthfulqa-smoke:validation",
        "--max-dataset-pct",
        "34",
        "--max-dataset-split-pct",
        "34",
        *extra_args,
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_mix_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["dataset_count"] == 3, report
    assert report["dataset_counts"] == {
        "arc-smoke": 1,
        "hellaswag-smoke": 1,
        "truthfulqa-smoke": 1,
    }, report
    assert len(report["distribution"]) == 7, report
    assert len(report["record_telemetry"]) == 3, report
    assert {row["dataset_split"] for row in report["record_telemetry"]} == {
        "arc-smoke:validation",
        "hellaswag-smoke:validation",
        "truthfulqa-smoke:validation",
    }, report
    assert not report["findings"], report
    assert (RESULTS / "dataset_mix_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_mix_audit_smoke_latest.csv").exists()
    record_csv = RESULTS / "dataset_mix_audit_smoke_records_latest.csv"
    assert record_csv.exists()
    assert "normalized_payload_sha256" in record_csv.read_text(encoding="utf-8").splitlines()[0]
    assert (RESULTS / "dataset_mix_audit_smoke_latest_findings.csv").exists()
    assert (RESULTS / "dataset_mix_audit_smoke_latest_junit.xml").exists()


def assert_gate_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-mix-audit-") as tmp:
        out = Path(tmp) / "mix.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_mix_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(out),
            "--min-datasets",
            "4",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["status"] == "fail", report
        assert any(item["kind"] == "min_datasets" for item in report["findings"]), report


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
