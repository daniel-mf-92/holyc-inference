#!/usr/bin/env python3
"""Tests for QEMU serial accounting audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_serial_accounting_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "stdout_bytes": 32,
        "stderr_bytes": 8,
        "serial_output_bytes": 40,
        "stdout_lines": 2,
        "stderr_lines": 1,
        "serial_output_lines": 3,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_serial_accounting_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_serial_accounting(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact), "--require-metrics"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_serial_accounting_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks == 2
    assert rows[0].expected_serial_output_bytes == 40


def test_audit_flags_serial_byte_line_and_negative_metric_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(serial_output_bytes=41, serial_output_lines=2, stdout_bytes=-1)])
    args = parse_args([str(artifact), "--require-metrics"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_serial_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    kinds = {finding.kind for finding in findings}
    assert {"metric_drift", "invalid_metric"} <= kinds
    metrics = {finding.metric for finding in findings}
    assert {"serial_output_bytes", "serial_output_lines", "stdout_bytes"} <= metrics


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_serial_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "serial_accounting",
            "--require-metrics",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "serial_accounting.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No serial accounting findings." in (output_dir / "serial_accounting.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "serial_accounting.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["expected_serial_output_bytes"] == "40"
    finding_rows = list(csv.DictReader((output_dir / "serial_accounting_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "serial_accounting_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_serial_accounting_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_serial_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "serial_accounting",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "serial_accounting.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_serial_accounting(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_serial_byte_line_and_negative_metric_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
