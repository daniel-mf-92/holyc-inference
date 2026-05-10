#!/usr/bin/env python3
"""Tests for QEMU CPU accounting audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_cpu_accounting_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q8_0",
        "commit": "abc123",
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 40,
        "wall_elapsed_us": 80_000,
        "host_child_user_cpu_us": 20_000,
        "host_child_system_cpu_us": 4_000,
        "host_child_cpu_us": 24_000,
        "host_child_cpu_pct": 30.0,
        "host_child_tok_per_cpu_s": 40 * 1_000_000.0 / 24_000,
        "host_child_peak_rss_bytes": 2048,
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_cpu_accounting_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}), encoding="utf-8")


def test_audit_accepts_consistent_cpu_accounting(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    args = parse_args([str(artifact), "--require-cpu-metrics"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_cpu_accounting_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 1
    assert rows[0].checks == 3
    assert rows[0].host_child_cpu_us == 24_000


def test_audit_flags_cpu_metric_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(host_child_cpu_us=1, host_child_cpu_pct=1.0, host_child_tok_per_cpu_s=1.0)])
    args = parse_args([str(artifact), "--require-cpu-metrics"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_cpu_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    assert {finding.kind for finding in findings} == {"metric_drift"}
    assert {finding.metric for finding in findings} == {
        "host_child_cpu_us",
        "host_child_cpu_pct",
        "host_child_tok_per_cpu_s",
    }


def test_audit_flags_missing_and_negative_cpu_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(host_child_user_cpu_us="", host_child_system_cpu_us=-1)])
    args = parse_args([str(artifact), "--require-cpu-metrics"])
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_cpu_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    assert ("missing_metric", "host_child_user_cpu_us") in {(finding.kind, finding.metric) for finding in findings}
    assert ("invalid_metric", "host_child_system_cpu_us") in {(finding.kind, finding.metric) for finding in findings}


def test_audit_flags_cpu_budget_gate_failures(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(host_child_cpu_pct=95.0, host_child_tok_per_cpu_s=100.0)])
    args = parse_args(
        [
            str(artifact),
            "--require-cpu-metrics",
            "--max-host-child-cpu-pct",
            "80",
            "--min-host-child-tok-per-cpu-s",
            "1000",
        ]
    )
    args.pattern = ["qemu_prompt_bench*.json"]

    rows, findings = qemu_cpu_accounting_audit.audit([artifact], args)

    assert len(rows) == 1
    finding_pairs = {(finding.kind, finding.metric) for finding in findings}
    assert ("max_host_child_cpu_pct", "host_child_cpu_pct") in finding_pairs
    assert ("min_host_child_tok_per_cpu_s", "host_child_tok_per_cpu_s") in finding_pairs


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row()])
    output_dir = tmp_path / "out"

    status = qemu_cpu_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "cpu_accounting",
            "--require-cpu-metrics",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "cpu_accounting.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No CPU accounting findings." in (output_dir / "cpu_accounting.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "cpu_accounting.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    assert rows[0]["host_child_cpu_us"] == "24000.0"
    finding_rows = list(csv.DictReader((output_dir / "cpu_accounting_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "cpu_accounting_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_cpu_accounting_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_min_rows_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_cpu_accounting_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "cpu_accounting",
            "--min-rows",
            "1",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "cpu_accounting.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_cpu_accounting(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_cpu_metric_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_and_negative_cpu_inputs(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_cpu_budget_gate_failures(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_min_rows_missing(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
