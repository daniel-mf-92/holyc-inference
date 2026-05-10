#!/usr/bin/env python3
"""CI smoke test for dataset_provenance_balance_audit.py."""

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
        str(BENCH / "dataset_provenance_balance_audit.py"),
        "--input",
        str(SAMPLE),
        "--output",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_latest.json"),
        "--markdown",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_latest.md"),
        "--csv",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_latest.csv"),
        "--record-csv",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_records_latest.csv"),
        "--findings-csv",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_latest_findings.csv"),
        "--junit",
        str(RESULTS / "dataset_provenance_balance_audit_smoke_latest_junit.xml"),
        "--require-provenance",
        "--require-provenance-source",
        "synthetic HellaSwag-shaped smoke row",
        "--require-provenance-source",
        "synthetic ARC-shaped smoke row",
        "--require-provenance-source",
        "synthetic TruthfulQA-shaped smoke row",
        "--min-provenance-sources",
        "3",
        "--min-records-per-provenance",
        "1",
        "--max-provenance-pct",
        "34",
        "--max-dataset-split-provenance-pct",
        "100",
        *extra_args,
    ]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


def assert_smoke_outputs() -> None:
    report = json.loads((RESULTS / "dataset_provenance_balance_audit_smoke_latest.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass", report
    assert report["record_count"] == 3, report
    assert report["provenance_source_count"] == 3, report
    assert report["provenance_counts"] == {
        "synthetic ARC-shaped smoke row": 1,
        "synthetic HellaSwag-shaped smoke row": 1,
        "synthetic TruthfulQA-shaped smoke row": 1,
    }, report
    assert len(report["distribution"]) == 6, report
    assert len(report["record_telemetry"]) == 3, report
    assert not report["findings"], report
    assert (RESULTS / "dataset_provenance_balance_audit_smoke_latest.md").exists()
    assert (RESULTS / "dataset_provenance_balance_audit_smoke_latest.csv").exists()
    assert (RESULTS / "dataset_provenance_balance_audit_smoke_latest_findings.csv").exists()
    assert (RESULTS / "dataset_provenance_balance_audit_smoke_latest_junit.xml").exists()
    record_csv = RESULTS / "dataset_provenance_balance_audit_smoke_records_latest.csv"
    assert record_csv.exists()
    assert "normalized_payload_sha256" in record_csv.read_text(encoding="utf-8").splitlines()[0]


def assert_gate_failure() -> None:
    with tempfile.TemporaryDirectory(prefix="dataset-provenance-balance-audit-") as tmp:
        out = Path(tmp) / "provenance.json"
        cmd = [
            sys.executable,
            str(BENCH / "dataset_provenance_balance_audit.py"),
            "--input",
            str(SAMPLE),
            "--output",
            str(out),
            "--min-provenance-sources",
            "4",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        assert proc.returncode == 1, proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["status"] == "fail", report
        assert any(item["kind"] == "min_provenance_sources" for item in report["findings"]), report


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
