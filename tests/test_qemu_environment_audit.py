#!/usr/bin/env python3
"""Tests for QEMU benchmark environment provenance audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_environment_audit
import qemu_prompt_bench


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def artifact_payload(*, environment: dict[str, object] | None = None, row_commit: str = "abc123") -> dict[str, object]:
    return {
        "status": "pass",
        "commit": "abc123",
        "profile": "ci-airgap-smoke",
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "command": COMMAND,
        "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
        "command_airgap": qemu_prompt_bench.command_airgap_metadata(COMMAND),
        "environment": {
            "platform": "macOS-15-test",
            "machine": "arm64",
            "processor": "arm",
            "python": "3.14.0",
            "cpu_count": 10,
            "qemu_bin": "qemu-system-x86_64",
            "qemu_path": "/opt/homebrew/bin/qemu-system-x86_64",
            "qemu_version": "QEMU emulator version 9.2.0",
        }
        if environment is None
        else environment,
        "benchmarks": [
            {
                "commit": row_commit,
                "command": COMMAND,
                "command_sha256": qemu_prompt_bench.command_hash(COMMAND),
            }
        ],
    }


def parse_args(extra: list[str]) -> object:
    return qemu_environment_audit.build_parser().parse_args(extra)


def test_audit_accepts_valid_environment_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps(artifact_payload()), encoding="utf-8")
    args = parse_args([str(artifact), "--require-qemu-path", "--require-qemu-version"])

    record, findings = qemu_environment_audit.audit_artifact(artifact, args)

    assert record is not None
    assert record.commit == "abc123"
    assert record.rows == 1
    assert record.command_airgap_ok is True
    assert findings == []


def test_audit_flags_missing_environment_and_row_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    payload = artifact_payload(environment={}, row_commit="def456")
    payload["command_sha256"] = "bad"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    args = parse_args([str(artifact), "--require-qemu-path"])

    record, findings = qemu_environment_audit.audit_artifact(artifact, args)

    assert record is not None
    kinds = {finding.kind for finding in findings}
    assert {"missing_environment_field", "invalid_cpu_count", "missing_qemu_path", "command_hash", "row_commit_drift"} <= kinds


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps(artifact_payload()), encoding="utf-8")
    output_dir = tmp_path / "out"

    status = qemu_environment_audit.main(
        [
            str(artifact),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "env",
            "--require-qemu-path",
        ]
    )

    assert status == 0
    payload = json.loads((output_dir / "env.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["artifacts_with_airgap_ok"] == 1
    assert "No environment provenance findings." in (output_dir / "env.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "env.csv").open(encoding="utf-8")))
    assert rows[0]["commit"] == "abc123"
    finding_rows = list(csv.DictReader((output_dir / "env_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "env_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_environment_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_on_environment_drift(tmp_path: Path) -> None:
    first = tmp_path / "qemu_prompt_bench_a.json"
    second = tmp_path / "qemu_prompt_bench_b.json"
    first.write_text(json.dumps(artifact_payload()), encoding="utf-8")
    drifted = artifact_payload()
    drifted["environment"]["python"] = "3.13.0"  # type: ignore[index]
    second.write_text(json.dumps(drifted), encoding="utf-8")
    output_dir = tmp_path / "out"

    status = qemu_environment_audit.main(
        [
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--output-stem",
            "env",
            "--fail-on-environment-drift",
            "--min-artifacts",
            "2",
        ]
    )

    assert status == 1
    payload = json.loads((output_dir / "env.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "environment_drift"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_valid_environment_artifact(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_missing_environment_and_row_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_on_environment_drift(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
