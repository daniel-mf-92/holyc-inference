#!/usr/bin/env python3
"""Tests for QEMU prompt echo audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_echo_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "smoke-short",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "prompt_bytes": 16,
        "guest_prompt_bytes": 16,
        "guest_prompt_bytes_match": True,
        "prompt_sha256": "0" * 64,
        "guest_prompt_sha256": "0" * 64,
        "guest_prompt_sha256_match": True,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_prompt_echo_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_matching_guest_echo(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact)])

    rows, findings = qemu_prompt_echo_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks == 4
    assert rows[0].guest_prompt_sha256_match is True


def test_audit_flags_guest_echo_drift_and_bad_match_flags(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(
                guest_prompt_bytes=17,
                guest_prompt_bytes_match=True,
                guest_prompt_sha256="1" * 64,
                guest_prompt_sha256_match=True,
            )
        ],
    )
    args = parse_args([str(artifact)])

    rows, findings = qemu_prompt_echo_audit.audit([artifact], args)

    assert len(rows) == 1
    kinds = {finding.kind for finding in findings}
    assert {"prompt_bytes_drift", "prompt_bytes_match_flag_drift", "prompt_sha256_drift", "prompt_sha256_match_flag_drift"} <= kinds


def test_audit_ignores_failed_and_warmup_rows_by_default(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            artifact_row(phase="warmup", guest_prompt_bytes=99),
            artifact_row(exit_class="timeout", guest_prompt_sha256="2" * 64),
            artifact_row(),
        ],
    )
    args = parse_args([str(artifact)])

    rows, findings = qemu_prompt_echo_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].phase == "measured"


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_prompt_echo_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "echo"])

    assert status == 0
    payload = json.loads((output_dir / "echo.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No prompt echo findings." in (output_dir / "echo.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "echo.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "echo_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "echo_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_prompt_echo_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_prompt_echo_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "echo", "--min-rows", "1"])

    assert status == 1
    payload = json.loads((output_dir / "echo.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_matching_guest_echo(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_guest_echo_drift_and_bad_match_flags(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_ignores_failed_and_warmup_rows_by_default(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
