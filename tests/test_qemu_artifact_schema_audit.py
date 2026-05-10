#!/usr/bin/env python3
"""Tests for QEMU artifact schema audits."""

from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_artifact_schema_audit


def row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "benchmark": "ci-airgap-smoke/Q4_0/smoke-short/1",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "prompt": "smoke-short",
        "prompt_sha256": "0" * 64,
        "command_sha256": "1" * 64,
        "exit_class": "ok",
        "timestamp": "2026-05-01T00:00:01Z",
        "tokens": 16,
        "elapsed_us": 80_000,
        "wall_elapsed_us": 96_000,
        "tok_per_s": 200.0,
        "wall_tok_per_s": 166.666667,
        "us_per_token": 5_000.0,
        "wall_us_per_token": 6_000.0,
        "returncode": 0,
        "timeout_seconds": 30.0,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]], **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "generated_at": "2026-05-01T00:00:02Z",
        "artifact_schema_version": "qemu-prompt-bench/v1",
        "status": "pass",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "command": ["qemu-system-x86_64", "-nic", "none"],
        "command_sha256": "1" * 64,
        "command_airgap": {"ok": True, "violations": []},
        "prompt_suite": {"path": "bench/prompts/smoke.jsonl", "sha256": "2" * 64},
        "suite_summary": {"runs": len(rows), "ok_runs": len(rows), "failed_runs": 0},
        "warmups": [],
        "benchmarks": rows,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_artifact_schema_audit.build_parser().parse_args(extra)


def test_audit_accepts_complete_prompt_benchmark_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_pass.json"
    write_artifact(artifact, [row()])
    args = parse_args([str(artifact)])

    records, findings = qemu_artifact_schema_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 2
    assert records[0].scope == "artifact"
    assert records[1].present_fields == records[1].required_fields


def test_audit_flags_missing_and_invalid_schema_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_fail.json"
    write_artifact(
        artifact,
        [row(tokens=0, timestamp="not-a-timestamp", prompt_sha256="")],
        status="planned",
        command_airgap="",
        generated_at="not-a-timestamp",
        artifact_schema_version="qemu-prompt-bench/v999",
    )
    args = parse_args([str(artifact)])

    records, findings = qemu_artifact_schema_audit.audit([artifact], args)
    kinds = {finding.kind for finding in findings}
    fields = {finding.field for finding in findings}

    assert records[0].missing_fields == 1
    assert {"missing_field", "invalid_field"} <= kinds
    assert {"status", "generated_at", "artifact_schema_version", "command_airgap", "tokens", "timestamp", "prompt_sha256"} <= fields


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_pass.json"
    write_artifact(artifact, [row()])
    output_dir = tmp_path / "out"

    status = qemu_artifact_schema_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "schema"])

    assert status == 0
    payload = json.loads((output_dir / "schema.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No QEMU artifact schema findings." in (output_dir / "schema.md").read_text(encoding="utf-8")
    assert "required_fields" in (output_dir / "schema.csv").read_text(encoding="utf-8")
    assert "kind" in (output_dir / "schema_findings.csv").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "schema_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_artifact_schema_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_empty.json"
    write_artifact(artifact, [])
    output_dir = tmp_path / "out"

    status = qemu_artifact_schema_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "schema"])

    assert status == 1
    payload = json.loads((output_dir / "schema.json").read_text(encoding="utf-8"))
    assert any(finding["kind"] == "missing_rows" for finding in payload["findings"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-artifact-schema-test-") as tmp:
        test_audit_accepts_complete_prompt_benchmark_artifact(Path(tmp) / "pass")
        test_audit_flags_missing_and_invalid_schema_fields(Path(tmp) / "fail")
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp) / "cli")
        test_cli_fails_without_rows(Path(tmp) / "empty")
    print("test_qemu_artifact_schema_audit=ok")
