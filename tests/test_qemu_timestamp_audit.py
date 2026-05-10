#!/usr/bin/env python3
"""Tests for QEMU timestamp audit tooling."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_timestamp_audit


def write_artifact(path: Path, generated_at: str, timestamps: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "status": "pass",
                "benchmarks": [
                    {"launch_index": index + 1, "phase": "measured", "prompt": f"p{index}", "timestamp": timestamp}
                    for index, timestamp in enumerate(timestamps)
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "max_future_seconds": 300.0,
        "max_row_after_generated_at_seconds": 5.0,
        "max_row_before_generated_at_seconds": 3600.0,
        "require_rows": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_audit_accepts_matching_stamp_and_monotonic_rows(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(artifact_path, "2026-04-29T23:01:49Z", ["2026-04-29T23:01:48Z", "2026-04-29T23:01:49Z"])

    artifact, findings = qemu_timestamp_audit.audit_artifact(artifact_path, args())

    assert artifact.status == "pass"
    assert artifact.filename_stamp == "20260429T230149Z"
    assert artifact.parsed_row_timestamps == 2
    assert findings == []


def test_audit_flags_stamp_mismatch_row_regression_and_skew(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(artifact_path, "2026-04-29T23:01:48Z", ["2026-04-29T23:01:49Z", "2026-04-29T23:01:47Z"])

    artifact, findings = qemu_timestamp_audit.audit_artifact(
        artifact_path,
        args(max_row_after_generated_at_seconds=0.0),
    )
    kinds = {finding.kind for finding in findings}

    assert artifact.status == "fail"
    assert "filename_stamp_mismatch" in kinds
    assert "row_timestamp_regressed" in kinds
    assert "row_after_generated_at" in kinds


def test_audit_flags_stale_rows_before_generated_at(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(artifact_path, "2026-04-29T23:01:49Z", ["2026-04-29T21:01:49Z"])

    artifact, findings = qemu_timestamp_audit.audit_artifact(
        artifact_path,
        args(max_row_before_generated_at_seconds=60.0),
    )

    assert artifact.status == "fail"
    assert artifact.min_row_skew_seconds == -7200.0
    assert "row_before_generated_at" in {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, "2026-04-29T23:01:49Z", ["2026-04-29T23:01:49Z"])
    output_dir = tmp_path / "out"

    status = qemu_timestamp_audit.main([str(tmp_path), "--output-dir", str(output_dir), "--require-rows"])

    assert status == 0
    payload = json.loads((output_dir / "qemu_timestamp_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "QEMU Timestamp Audit" in (output_dir / "qemu_timestamp_audit_latest.md").read_text(encoding="utf-8")
    assert "parsed_row_timestamps" in (output_dir / "qemu_timestamp_audit_latest.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "qemu_timestamp_audit_latest_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_timestamp_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_timestamp_audit"
    assert junit_root.attrib["failures"] == "0"
