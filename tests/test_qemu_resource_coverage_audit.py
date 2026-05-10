#!/usr/bin/env python3
"""Tests for QEMU resource telemetry coverage audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_resource_coverage_audit


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
        "memory_bytes": 67_174_400,
        "memory_bytes_per_token": 2_099_200.0,
        "host_child_peak_rss_bytes": 458_752,
        "host_child_cpu_us": 67_610,
        "host_child_cpu_pct": 28.5,
        "host_child_tok_per_cpu_s": 473.3,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_resource_coverage_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_rows_with_resource_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = list(qemu_resource_coverage_audit.DEFAULT_REQUIRED_METRICS)

    records, findings = qemu_resource_coverage_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 1
    assert records[0].metric_count == len(qemu_resource_coverage_audit.DEFAULT_REQUIRED_METRICS)
    assert records[0].missing_metrics == ""


def test_audit_flags_missing_invalid_and_drifted_metrics(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    row = artifact_row(memory_bytes_per_token=1.0, host_child_cpu_pct="nan")
    row.pop("host_child_peak_rss_bytes")
    write_artifact(artifact, [row])
    args = parse_args([str(artifact)])
    args.pattern = ["qemu_prompt_bench*.json"]
    args.require_metric = list(qemu_resource_coverage_audit.DEFAULT_REQUIRED_METRICS)

    records, findings = qemu_resource_coverage_audit.audit([artifact], args)

    assert len(records) == 1
    kinds = {finding.kind for finding in findings}
    assert {"missing_metric", "invalid_metric", "memory_per_token_drift"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_resource_coverage_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "resource",
            "--min-rows",
            "1",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows_with_all_metrics"] == 1
    assert "No resource telemetry coverage findings." in (output_dir / "resource.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "resource.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "resource_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "resource_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_resource_coverage_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_resource_coverage_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "resource",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_rows_with_resource_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_invalid_and_drifted_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
