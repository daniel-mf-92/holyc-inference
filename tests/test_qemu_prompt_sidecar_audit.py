#!/usr/bin/env python3
"""Tests for QEMU prompt sidecar audits."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import qemu_prompt_sidecar_audit


def write_artifact_bundle(root: Path, stem: str = "qemu_prompt_bench_latest") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / f"{stem}.json"
    base = stem[: -len("_latest")] if stem.endswith("_latest") else stem
    artifact.write_text(json.dumps({"generated_at": "2026-05-01T00:00:00Z", "status": "pass"}) + "\n", encoding="utf-8")
    artifact.with_suffix(".csv").write_text("status\npass\n", encoding="utf-8")
    artifact.with_suffix(".md").write_text("# report\n", encoding="utf-8")
    artifact.with_name(f"{base}_junit_latest.xml").write_text("<testsuite failures=\"0\" />\n", encoding="utf-8")
    return artifact


def test_audit_accepts_complete_sidecar_bundle(tmp_path: Path) -> None:
    artifact = write_artifact_bundle(tmp_path)
    args = qemu_prompt_sidecar_audit.build_parser().parse_args([str(artifact)])

    records, findings = qemu_prompt_sidecar_audit.audit([artifact], args)

    assert findings == []
    assert len(records) == 3
    assert {record.sidecar_kind for record in records} == {"csv", "markdown", "junit"}


def test_audit_rejects_missing_and_empty_sidecars(tmp_path: Path) -> None:
    artifact = write_artifact_bundle(tmp_path)
    artifact.with_suffix(".md").write_text("", encoding="utf-8")
    artifact.with_name("qemu_prompt_bench_junit_latest.xml").unlink()
    args = qemu_prompt_sidecar_audit.build_parser().parse_args([str(artifact)])

    records, findings = qemu_prompt_sidecar_audit.audit([artifact], args)

    assert any(record.status == "fail" for record in records)
    assert {finding.kind for finding in findings} == {"empty_sidecar", "missing_sidecar"}


def test_cli_writes_json_markdown_csv_findings_and_junit(tmp_path: Path) -> None:
    artifact = write_artifact_bundle(tmp_path)
    output_dir = tmp_path / "out"

    status = qemu_prompt_sidecar_audit.main([str(artifact), "--output-dir", str(output_dir), "--output-stem", "sidecar"])

    assert status == 0
    report = json.loads((output_dir / "sidecar.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["summary"]["sidecars_checked"] == 3
    assert "QEMU Prompt Sidecar Audit" in (output_dir / "sidecar.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "sidecar.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "pass"
    findings = list(csv.DictReader((output_dir / "sidecar_findings.csv").open(encoding="utf-8")))
    assert findings == []
    junit = ET.parse(output_dir / "sidecar_junit.xml").getroot()
    assert junit.attrib["name"] == "holyc_qemu_prompt_sidecar_audit"
    assert junit.attrib["failures"] == "0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="holyc-qemu-sidecar-test-") as tmp:
        test_audit_accepts_complete_sidecar_bundle(Path(tmp) / "pass")
        test_audit_rejects_missing_and_empty_sidecars(Path(tmp) / "fail")
        test_cli_writes_json_markdown_csv_findings_and_junit(Path(tmp) / "cli")
    print("test_qemu_prompt_sidecar_audit=ok")
