#!/usr/bin/env python3
"""Tests for QEMU benchmark result retention audit tooling."""

from __future__ import annotations

import json
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_result_retention_audit


def write_artifact(path: Path, generated_at: str = "2026-04-29T23:01:49Z", status: str = "pass") -> None:
    path.write_text(
        json.dumps({"generated_at": generated_at, "status": status, "benchmarks": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_latest_with_matching_timestamped_history(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_latest.json"
    history = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(latest)
    history.write_bytes(latest.read_bytes())

    record = qemu_result_retention_audit.audit_latest(latest)
    findings = qemu_result_retention_audit.evaluate([record], min_latest=1, min_history_per_latest=1)

    assert record.status == "pass"
    assert record.expected_history_present is True
    assert record.hashes_match is True
    assert record.history_count == 1
    assert findings == []


def test_evaluate_flags_missing_history_and_mismatch(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "qemu_prompt_bench_latest.json"
    missing.parent.mkdir()
    write_artifact(missing)
    missing_record = qemu_result_retention_audit.audit_latest(missing)

    mismatch = tmp_path / "mismatch" / "qemu_prompt_bench_latest.json"
    mismatch.parent.mkdir()
    write_artifact(mismatch)
    write_artifact(mismatch.parent / "qemu_prompt_bench_20260429T230149Z.json", status="fail")
    mismatch_record = qemu_result_retention_audit.audit_latest(mismatch)

    findings = qemu_result_retention_audit.evaluate([missing_record, mismatch_record], min_latest=3, min_history_per_latest=1)
    kinds = {finding.kind for finding in findings}

    assert "min_latest" in kinds
    assert "history_count" in kinds
    assert "missing_expected_history" in kinds
    assert "latest_history_mismatch" in kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    latest = tmp_path / "qemu_prompt_bench_dry_run_latest.json"
    history = tmp_path / "qemu_prompt_bench_dry_run_20260429T193620Z.json"
    write_artifact(latest, "2026-04-29T19:36:20Z", "planned")
    history.write_bytes(latest.read_bytes())
    output_dir = tmp_path / "out"

    status = qemu_result_retention_audit.main([str(tmp_path), "--output-dir", str(output_dir)])

    assert status == 0
    payload = json.loads((output_dir / "qemu_result_retention_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["latest_artifacts"] == 1
    assert "QEMU Result Retention Audit" in (output_dir / "qemu_result_retention_audit_latest.md").read_text(
        encoding="utf-8"
    )
    assert "history_sha256" in (output_dir / "qemu_result_retention_audit_latest.csv").read_text(encoding="utf-8")
    finding_rows = list(csv.DictReader((output_dir / "qemu_result_retention_audit_latest_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "qemu_result_retention_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_result_retention_audit"
    assert junit_root.attrib["failures"] == "0"
