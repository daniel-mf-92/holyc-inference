#!/usr/bin/env python3
"""Tests for QEMU command fingerprint audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_command_fingerprint_audit
import qemu_prompt_bench


def command() -> list[str]:
    return ["qemu-system-x86_64", "-nic", "none", "-m", "512M"]


def artifact_row(cmd: list[str], **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "iteration": 1,
        "launch_index": 1,
        "command": cmd,
        "command_sha256": qemu_prompt_bench.command_hash(cmd),
    }
    row.update(overrides)
    return row


def parse_args(extra: list[str]) -> object:
    return qemu_command_fingerprint_audit.build_parser().parse_args(extra)


def write_artifact(path: Path, rows: list[dict[str, object]], top_command: list[str] | None = None) -> None:
    cmd = top_command or command()
    path.write_text(
        json.dumps(
            {
                "command": cmd,
                "command_sha256": qemu_prompt_bench.command_hash(cmd),
                "warmups": [],
                "benchmarks": rows,
            }
        ),
        encoding="utf-8",
    )


def test_audit_accepts_airgapped_stable_command_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    cmd = command()
    write_artifact(artifact, [artifact_row(cmd)])
    args = parse_args([str(artifact), "--require-top-command"])

    artifact_record, rows, findings = qemu_command_fingerprint_audit.audit_artifact(artifact, args)

    assert findings == []
    assert artifact_record.status == "pass"
    assert rows[0].computed_command_sha256 == qemu_prompt_bench.command_hash(cmd)
    assert rows[0].explicit_nic_none is True


def test_audit_flags_hash_drift_and_networked_command(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    cmd = command()
    bad_cmd = ["qemu-system-x86_64", "-nic", "user"]
    write_artifact(
        artifact,
        [
            artifact_row(cmd, prompt="bad-hash", command_sha256="bad"),
            artifact_row(bad_cmd, prompt="networked", launch_index=2),
        ],
    )
    args = parse_args([str(artifact), "--require-top-command"])

    artifact_record, rows, findings = qemu_command_fingerprint_audit.audit_artifact(artifact, args)

    assert artifact_record.status == "fail"
    assert len(rows) == 2
    kinds = {finding.kind for finding in findings}
    assert {"command_sha256_mismatch", "command_airgap_violation", "row_command_hash_drift"} <= kinds


def test_audit_can_require_single_row_command_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    cmd = command()
    alternate_cmd = ["qemu-system-x86_64", "-nic", "none", "-m", "1G"]
    write_artifact(
        artifact,
        [
            artifact_row(cmd),
            artifact_row(alternate_cmd, prompt="alternate", launch_index=2),
        ],
    )
    args = parse_args([str(artifact), "--allow-row-command-drift", "--require-single-command-hash"])

    artifact_record, rows, findings = qemu_command_fingerprint_audit.audit_artifact(artifact, args)

    assert artifact_record.status == "fail"
    assert len(rows) == 2
    assert {finding.kind for finding in findings} == {"multiple_row_command_hashes"}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    cmd = command()
    write_artifact(artifact, [artifact_row(cmd)])
    output_dir = tmp_path / "out"

    status = qemu_command_fingerprint_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "command_fingerprint", "--require-top-command"]
    )

    assert status == 0
    payload = json.loads((output_dir / "command_fingerprint.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["rows"] == 1
    assert "No command fingerprint findings." in (output_dir / "command_fingerprint.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "command_fingerprint.csv").open(encoding="utf-8")))
    assert rows[0]["prompt"] == "smoke-short"
    finding_rows = list(csv.DictReader((output_dir / "command_fingerprint_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "command_fingerprint_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_command_fingerprint_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_matching_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    status = qemu_command_fingerprint_audit.main([str(tmp_path), "--output-dir", str(output_dir), "--output-stem", "command_fingerprint"])

    assert status == 1
    payload = json.loads((output_dir / "command_fingerprint.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "no_artifacts"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_airgapped_stable_command_hashes(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_hash_drift_and_networked_command(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_can_require_single_row_command_hash(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_matching_artifacts(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
