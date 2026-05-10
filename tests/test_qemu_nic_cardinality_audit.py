#!/usr/bin/env python3
"""Tests for QEMU NIC cardinality audit."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_nic_cardinality_audit


def command() -> list[str]:
    return ["qemu-system-x86_64", "-display", "none", "-nic", "none", "-m", "512M"]


def row(cmd: list[str], **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "prompt": "smoke-short",
        "phase": "measured",
        "launch_index": 1,
        "command": cmd,
    }
    payload.update(overrides)
    return payload


def write_artifact(path: Path, rows: list[dict[str, object]], top_command: list[str] | None = None) -> None:
    path.write_text(
        json.dumps({"command": top_command or command(), "warmups": [], "benchmarks": rows}) + "\n",
        encoding="utf-8",
    )


def parse_args(extra: list[str]) -> object:
    return qemu_nic_cardinality_audit.build_parser().parse_args(extra)


def test_audit_accepts_one_explicit_disabled_nic(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(command())])
    args = parse_args([str(artifact), "--require-top-command"])

    artifact_record, commands, findings = qemu_nic_cardinality_audit.audit_artifact(
        artifact,
        require_top_command=args.require_top_command,
        require_qemu_system_executable=args.require_qemu_system_executable,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 2
    assert all(record.nic_none_count == 1 for record in commands)
    assert all(record.qemu_system_executable for record in commands)


def test_audit_accepts_long_form_disabled_nic(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    spaced = ["qemu-system-x86_64", "--nic", "none", "-m", "512M"]
    equals = ["qemu-system-x86_64", "--nic=none", "-m", "512M"]
    write_artifact(artifact, [row(spaced), row(equals, launch_index=2)], top_command=spaced)

    artifact_record, commands, findings = qemu_nic_cardinality_audit.audit_artifact(
        artifact,
        require_top_command=True,
        require_qemu_system_executable=True,
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 3
    assert all(record.nic_none_count == 1 for record in commands)


def test_audit_rejects_missing_duplicate_networked_and_non_qemu_commands(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(
        artifact,
        [
            row(["qemu-system-x86_64", "-m", "512M"], launch_index=1),
            row(["qemu-system-x86_64", "-nic", "none", "-nic=none"], launch_index=2),
            row(["qemu-system-x86_64", "-nic", "user"], launch_index=3),
            row(["python3", "fake-qemu", "-nic", "none"], launch_index=4),
        ],
    )

    artifact_record, commands, findings = qemu_nic_cardinality_audit.audit_artifact(
        artifact, require_top_command=True, require_qemu_system_executable=True
    )

    assert artifact_record.status == "fail"
    assert len(commands) == 5
    kinds = {finding.kind for finding in findings}
    assert {"nic_none_cardinality", "airgap_violation", "non_qemu_system_executable"} <= kinds


def test_audit_allows_synthetic_harness_when_executable_gate_is_off(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    synthetic = ["bench/fixtures/qemu_synthetic_bench.py", "-nic", "none", "--tokens", "8"]
    write_artifact(artifact, [row(synthetic)], top_command=synthetic)

    artifact_record, commands, findings = qemu_nic_cardinality_audit.audit_artifact(
        artifact, require_top_command=True, require_qemu_system_executable=False
    )

    assert findings == []
    assert artifact_record.status == "pass"
    assert len(commands) == 2
    assert all(record.nic_none_count == 1 for record in commands)


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    write_artifact(artifact, [row(command())])
    output_dir = tmp_path / "out"

    status = qemu_nic_cardinality_audit.main(
        [str(artifact), "--output-dir", str(output_dir), "--output-stem", "nic_cardinality", "--require-top-command"]
    )

    assert status == 0
    payload = json.loads((output_dir / "nic_cardinality.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["command_rows"] == 2
    assert "No NIC cardinality findings." in (output_dir / "nic_cardinality.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "nic_cardinality.csv").open(encoding="utf-8")))
    assert rows[0]["nic_none_count"] == "1"
    finding_rows = list(csv.DictReader((output_dir / "nic_cardinality_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "nic_cardinality_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_nic_cardinality_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_without_command_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    artifact.write_text(json.dumps({"warmups": [], "benchmarks": []}) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    status = qemu_nic_cardinality_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "nic_cardinality"])

    assert status == 1
    payload = json.loads((output_dir / "nic_cardinality.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_command_rows"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_one_explicit_disabled_nic(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_long_form_disabled_nic(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_rejects_missing_duplicate_networked_and_non_qemu_commands(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_allows_synthetic_harness_when_executable_gate_is_off(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_without_command_rows(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
