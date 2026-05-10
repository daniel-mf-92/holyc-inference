#!/usr/bin/env python3
"""Tests for QEMU benchmark matrix artifact auditing."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_benchmark_matrix
import qemu_benchmark_matrix_audit


def write_matrix_artifact(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "\n".join(
            [
                json.dumps({"id": "alpha", "prompt": "Alpha?", "expected_tokens": 4}),
                json.dumps({"id": "beta", "prompt": "Beta?", "expected_tokens": 5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "prompts": str(prompts),
                "profile": "unit",
                "model": "smoke-model",
                "quantization": "Q8_0",
                "warmup": 1,
                "repeat": 2,
                "builds": [
                    {"build": "base", "image": "base.img", "qemu_args": ["-m", "512M"]},
                    {"build": "cand", "image": "cand.img", "qemu_args": ["-m", "512M"]},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = qemu_benchmark_matrix.build_matrix_report(matrix, "matrix")
    artifact = tmp_path / "qemu_benchmark_matrix_latest.json"
    artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def test_audit_accepts_valid_matrix_artifact(tmp_path: Path) -> None:
    artifact = write_matrix_artifact(tmp_path)

    summary, findings = qemu_benchmark_matrix_audit.audit_artifact(artifact)

    assert findings == []
    assert summary.status == "pass"
    assert summary.builds == 2
    assert summary.launches == 12
    assert summary.airgap_ok_builds == 2


def test_audit_flags_command_and_launch_drift(tmp_path: Path) -> None:
    artifact = write_matrix_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["builds"][0]["command"].remove("none")
    payload["builds"][0]["launch_count"] = 99
    payload["launches"][0]["command_sha256"] = "bad"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary, findings = qemu_benchmark_matrix_audit.audit_artifact(artifact)
    kinds = {finding.kind for finding in findings}

    assert summary.status == "fail"
    assert "command_hash" in kinds
    assert "command_airgap" in kinds
    assert "launch_count_formula" in kinds
    assert "launch_command_hash_drift" in kinds


def test_cli_writes_audit_reports(tmp_path: Path) -> None:
    artifact = write_matrix_artifact(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_benchmark_matrix_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "audit"]
    )

    assert status == 0
    payload = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts"] == 1
    rows = list(csv.DictReader((output_dir / "audit.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    findings = list(csv.DictReader((output_dir / "audit_findings.csv").open(encoding="utf-8")))
    assert findings == []
    assert "QEMU Benchmark Matrix Audit" in (output_dir / "audit.md").read_text(encoding="utf-8")
    junit_root = ET.parse(output_dir / "audit_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_benchmark_matrix_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_required_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_benchmark_matrix_audit.main(
        [str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "audit", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_valid_matrix_artifact(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_command_and_launch_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_audit_reports(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_without_required_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
