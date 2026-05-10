#!/usr/bin/env python3
"""Tests for QEMU output determinism audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_output_determinism_audit


def artifact_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "prompt": "short",
        "commit": "abc123",
        "seed": 0,
        "phase": "measured",
        "exit_class": "ok",
        "tokens": 8,
        "output": "hello world",
    }
    row.update(overrides)
    return row


def write_artifact(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"benchmarks": rows}) + "\n", encoding="utf-8")


def parse_args(extra: list[str]) -> object:
    return qemu_output_determinism_audit.build_parser().parse_args(extra)


def test_audit_accepts_repeated_identical_outputs(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row()])
    args = parse_args([str(artifact), "--require-output-hash", "--require-tokens"])

    rows, findings = qemu_output_determinism_audit.audit([artifact], args)

    assert findings == []
    assert len(rows) == 2
    assert rows[0].output_sha256 == rows[1].output_sha256
    assert rows[0].output_source == "output:derived"


def test_audit_flags_hash_and_token_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(), artifact_row(output="different", tokens=9)])
    args = parse_args([str(artifact), "--require-output-hash", "--require-tokens"])

    rows, findings = qemu_output_determinism_audit.audit([artifact], args)

    assert len(rows) == 2
    kinds = {finding.kind for finding in findings}
    assert {"output_hash_drift", "token_count_drift"} <= kinds


def test_audit_flags_missing_identity_and_repeat_count(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [artifact_row(output="", tokens="")])
    args = parse_args([str(artifact), "--require-output-hash", "--require-tokens", "--min-repeats", "2"])

    rows, findings = qemu_output_determinism_audit.audit([artifact], args)

    assert len(rows) == 1
    kinds = {finding.kind for finding in findings}
    assert {"missing_output_identity", "missing_tokens", "min_repeats"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    output_dir = tmp_path / "out"
    write_artifact(artifact, [artifact_row(), artifact_row()])

    status = qemu_output_determinism_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "determinism", "--require-output-hash", "--require-tokens"]
    )

    assert status == 0
    payload = json.loads((output_dir / "determinism.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["groups"] == 1
    assert "No output determinism findings." in (output_dir / "determinism.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "determinism.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    findings = list(csv.DictReader((output_dir / "determinism_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit = ET.parse(output_dir / "determinism_junit.xml").getroot()
    assert junit.attrib["name"] == "holyc_qemu_output_determinism_audit"
    assert junit.attrib["failures"] == "0"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_repeated_identical_outputs(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_hash_and_token_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_identity_and_repeat_count(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
