#!/usr/bin/env python3
"""Tests for QEMU artifact budget audit tooling."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_artifact_budget_audit


def args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "max_file_bytes": 4096,
        "max_serial_output_bytes": 1024,
        "max_stdout_tail_bytes": 64,
        "max_stderr_tail_bytes": 64,
        "max_failure_reason_bytes": 64,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def write_artifact(path: Path, *, serial_bytes: int, stdout_tail: str = "", failure_reason: str = "") -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-29T23:01:49Z",
                "benchmarks": [
                    {
                        "prompt": "smoke",
                        "phase": "measured",
                        "serial_output_bytes": serial_bytes,
                        "stdout_tail": stdout_tail,
                        "stderr_tail": "",
                        "failure_reason": failure_reason,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_artifact_inside_budgets(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(artifact_path, serial_bytes=512, stdout_tail="ok")

    record, findings = qemu_artifact_budget_audit.audit_file(artifact_path, args())

    assert record.status == "pass"
    assert record.row_count == 1
    assert record.max_serial_output_bytes == 512
    assert findings == []


def test_audit_flags_file_serial_tail_and_failure_reason_budgets(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_20260429T230149Z.json"
    write_artifact(artifact_path, serial_bytes=2048, stdout_tail="x" * 80, failure_reason="y" * 80)

    record, findings = qemu_artifact_budget_audit.audit_file(artifact_path, args(max_file_bytes=128))
    kinds = {finding.kind for finding in findings}

    assert record.status == "fail"
    assert "file_size_exceeded" in kinds
    assert "serial_output_budget_exceeded" in kinds
    assert "stdout_tail_budget_exceeded" in kinds
    assert "failure_reason_budget_exceeded" in kinds


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, serial_bytes=512, stdout_tail="ok")
    output_dir = tmp_path / "out"

    status = qemu_artifact_budget_audit.main(
        [
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--max-file-bytes",
            "4096",
            "--max-serial-output-bytes",
            "1024",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "qemu_artifact_budget_audit_latest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "QEMU Artifact Budget Audit" in (
        output_dir / "qemu_artifact_budget_audit_latest.md"
    ).read_text(encoding="utf-8")
    assert "max_serial_output_bytes" in (
        output_dir / "qemu_artifact_budget_audit_latest.csv"
    ).read_text(encoding="utf-8")
    assert "kind" in (output_dir / "qemu_artifact_budget_audit_latest_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "qemu_artifact_budget_audit_latest_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_artifact_budget_audit"
    assert junit_root.attrib["failures"] == "0"
