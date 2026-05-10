#!/usr/bin/env python3
"""Tests for QEMU memory accounting audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_memory_accounting_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 32,
        "memory_bytes": 1024,
        "memory_bytes_per_token": 32.0,
        "host_child_peak_rss_bytes": 4096,
        "host_rss_bytes_per_token": 128.0,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_memory_accounting_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_memory_accounting(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args(
        [
            str(artifact),
            "--require-memory-bytes",
            "--require-host-rss",
            "--require-guest-memory-within-host-rss",
            "--max-host-rss-over-guest-ratio",
            "4",
        ]
    )
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_memory_accounting_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks == 4
    assert rows[0].host_child_peak_rss_bytes == 4096
    assert rows[0].host_rss_over_guest_ratio == 4.0


def test_audit_flags_memory_metric_and_bound_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(memory_bytes=8192, memory_bytes_per_token=1.0, host_rss_bytes_per_token=1.0)])
    args = parse_args([str(artifact), "--require-memory-bytes", "--require-host-rss", "--require-guest-memory-within-host-rss"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_memory_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    kinds = {finding.kind for finding in findings}
    assert {"metric_drift", "memory_bound_violation"} <= kinds
    metrics = {finding.metric for finding in findings}
    assert {"memory_bytes_per_token", "host_rss_bytes_per_token", "memory_bytes"} <= metrics


def test_audit_flags_host_rss_over_guest_ratio(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(memory_bytes=1024, host_child_peak_rss_bytes=8192, host_rss_bytes_per_token=256.0)])
    args = parse_args([str(artifact), "--max-host-rss-over-guest-ratio", "4"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_memory_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    assert rows[0].host_rss_over_guest_ratio == 8.0
    assert any(finding.kind == "host_rss_over_guest_ratio" for finding in findings)


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_memory_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "memory_accounting",
            "--require-memory-bytes",
            "--require-host-rss",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "memory_accounting.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert payload["summary"]["host_rss_over_guest_ratio_max"] == 4.0
    assert "No memory accounting findings." in (output_dir / "memory_accounting.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "memory_accounting.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["host_child_peak_rss_bytes"] == "4096"
    assert rows[0]["host_rss_over_guest_ratio"] == "4.0"
    finding_rows = list(csv.DictReader((output_dir / "memory_accounting_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "memory_accounting_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_memory_accounting_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_memory_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "memory_accounting",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "memory_accounting.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_memory_accounting(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_memory_metric_and_bound_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_host_rss_over_guest_ratio(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
