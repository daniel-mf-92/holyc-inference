#!/usr/bin/env python3
"""Tests for QEMU identity audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_identity_audit
import qemu_prompt_bench


COMMAND = ["qemu-system-x86_64", "-nic", "none", "-serial", "stdio"]


def row(*, profile: str = "ci-airgap-smoke", command: list[str] | None = None, commit: str = "abc123") -> dict[str, object]:
    row_command = command or COMMAND
    return {
        "benchmark": "qemu_prompt",
        "profile": profile,
        "model": "synthetic-smoke",
        "quantization": "Q4_0",
        "phase": "measured",
        "launch_index": 1,
        "prompt": "alpha",
        "iteration": 1,
        "commit": commit,
        "command": row_command,
        "command_sha256": qemu_prompt_bench.command_hash(row_command),
    }


def write_artifact(path: Path, rows: list[dict[str, object]], *, command: list[str] | None = None, commit: str = "abc123") -> None:
    artifact_command = command or COMMAND
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-01T00:00:00Z",
                "status": "pass",
                "profile": "ci-airgap-smoke",
                "model": "synthetic-smoke",
                "quantization": "Q4_0",
                "commit": commit,
                "command": artifact_command,
                "command_sha256": qemu_prompt_bench.command_hash(artifact_command),
                "warmups": [],
                "benchmarks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_audit_accepts_consistent_identity(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row()])

    artifact, findings = qemu_identity_audit.audit_artifact(artifact_path)

    assert artifact.status == "pass"
    assert artifact.rows == 1
    assert artifact.identity_fields_checked == 3
    assert artifact.command_hashes_checked == 2
    assert findings == []


def test_audit_flags_row_identity_and_commit_drift(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row(profile="other", commit="def456")])

    artifact, findings = qemu_identity_audit.audit_artifact(artifact_path)

    assert artifact.status == "fail"
    assert {"identity_drift", "commit_drift"} <= {finding.kind for finding in findings}


def test_audit_flags_bad_command_hash(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    bad = row()
    bad["command_sha256"] = "bad"
    write_artifact(artifact_path, [bad])

    _, findings = qemu_identity_audit.audit_artifact(artifact_path)

    assert "command_hash" in {finding.kind for finding in findings}


def test_cli_writes_json_markdown_csv_and_junit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact_path, [row()])
    output_dir = tmp_path / "out"

    status = qemu_identity_audit.main(
        [str(artifact_path), "--output-dir", str(output_dir), "--output-stem", "identity", "--min-rows", "1"]
    )

    assert status == 0
    payload = json.loads((output_dir / "identity.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No identity drift findings." in (output_dir / "identity.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "identity.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    findings = list(csv.DictReader((output_dir / "identity_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit_root = ET.parse(output_dir / "identity_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_identity_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_artifacts_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    status = qemu_identity_audit.main(
        [str(input_dir), "--output-dir", str(output_dir), "--output-stem", "identity", "--min-artifacts", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "identity.json").read_text(encoding="utf-8"))
    assert {finding["kind"] for finding in payload["findings"]} == {"min_artifacts", "min_rows"}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_consistent_identity(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_row_identity_and_commit_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_bad_command_hash(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_artifacts_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
