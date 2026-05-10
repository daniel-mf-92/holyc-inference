#!/usr/bin/env python3
"""Tests for QEMU replay manifest audit tooling."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_replay_manifest
import qemu_replay_manifest_audit
from qemu_replay_manifest_ci_smoke import write_smoke_artifact


def write_manifest(tmp_path: Path, *, mutate: bool = False) -> Path:
    artifact = tmp_path / "qemu_prompt_bench_latest.json"
    manifest_dir = tmp_path / "manifest"
    write_smoke_artifact(artifact)
    status = qemu_replay_manifest.main(
        [str(artifact), "--output-dir", str(manifest_dir), "--output-stem", "qemu_replay_manifest"]
    )
    assert status == 0
    manifest_path = manifest_dir / "qemu_replay_manifest.json"
    if mutate:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["entries"][0]["argv"] = ["qemu-system-x86_64", "-netdev", "user,id=n0"]
        payload["entries"][0]["command_argc"] = 99
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def test_audit_accepts_replay_manifest_and_argv_sidecar(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    args = qemu_replay_manifest_audit.build_parser().parse_args([str(manifest_path)])

    row, findings = qemu_replay_manifest_audit.audit_manifest(manifest_path, args)

    assert findings == []
    assert row.status == "pass"
    assert row.entries == 1
    assert row.argv_sidecar_rows == 1
    assert row.airgap_ok_entries == 1


def test_audit_flags_non_airgapped_manifest_argv(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path, mutate=True)
    args = qemu_replay_manifest_audit.build_parser().parse_args([str(manifest_path)])

    row, findings = qemu_replay_manifest_audit.audit_manifest(manifest_path, args)
    kinds = {finding.kind for finding in findings}

    assert row.status == "fail"
    assert {"command_hash", "command_argc", "command_airgap", "missing_nic_none", "argv_drift"} <= kinds


def test_audit_flags_duplicate_manifest_and_sidecar_keys(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["entries"].append(dict(payload["entries"][0]))
    payload["summary"]["artifacts"] = 2
    payload["summary"]["airgap_ok_entries"] = 2
    payload["summary"]["measured_rows"] *= 2
    payload["summary"]["launch_plan_entries"] *= 2
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sidecar_path = manifest_path.with_name(f"{manifest_path.stem}_argv.jsonl")
    sidecar_row = sidecar_path.read_text(encoding="utf-8").strip()
    sidecar_path.write_text(f"{sidecar_row}\n{sidecar_row}\n", encoding="utf-8")

    args = qemu_replay_manifest_audit.build_parser().parse_args([str(manifest_path)])
    row, findings = qemu_replay_manifest_audit.audit_manifest(manifest_path, args)
    kinds = {finding.kind for finding in findings}

    assert row.status == "fail"
    assert "duplicate_replay_key" in kinds
    assert "duplicate_argv_key" in kinds


def test_audit_flags_sidecar_hash_and_airgap_drift(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    sidecar_path = manifest_path.with_name(f"{manifest_path.stem}_argv.jsonl")
    row = json.loads(sidecar_path.read_text(encoding="utf-8").strip())
    row["argv"] = ["qemu-system-x86_64", "-netdev", "user,id=n0"]
    sidecar_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    args = qemu_replay_manifest_audit.build_parser().parse_args([str(manifest_path)])
    row, findings = qemu_replay_manifest_audit.audit_manifest(manifest_path, args)
    kinds = {finding.kind for finding in findings}

    assert row.status == "fail"
    assert {"argv_command_hash", "argv_command_airgap", "argv_drift"} <= kinds


def test_cli_writes_audit_reports(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_replay_manifest_audit.main(
        [str(manifest_path), "--output-dir", str(output_dir), "--output-stem", "audit"]
    )

    assert status == 0
    payload = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["summary"]["entries"] == 1
    assert "QEMU Replay Manifest Audit" in (output_dir / "audit.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "audit.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    finding_rows = list(csv.DictReader((output_dir / "audit_findings.csv").open(encoding="utf-8")))
    assert finding_rows == []
    junit_root = ET.parse(output_dir / "audit_junit.xml").getroot()
    assert junit_root.attrib["name"] == "holyc_qemu_replay_manifest_audit"
    assert junit_root.attrib["failures"] == "0"


def test_cli_fails_when_no_manifests_match(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    status = qemu_replay_manifest_audit.main(
        [str(input_dir), "--output-dir", str(output_dir), "--output-stem", "audit", "--min-manifests", "1"]
    )

    assert status == 1
    payload = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert payload["findings"][0]["kind"] == "min_manifests"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_accepts_replay_manifest_and_argv_sidecar(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_non_airgapped_manifest_argv(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_duplicate_manifest_and_sidecar_keys(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_audit_flags_sidecar_hash_and_airgap_drift(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_writes_audit_reports(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_cli_fails_when_no_manifests_match(Path(tmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
