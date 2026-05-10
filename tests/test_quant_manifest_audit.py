#!/usr/bin/env python3
"""Tests for quant block manifest audits."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "bench" / "quant_manifest_audit.py"
spec = importlib.util.spec_from_file_location("quant_manifest_audit", AUDIT_PATH)
quant_manifest_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules["quant_manifest_audit"] = quant_manifest_audit
spec.loader.exec_module(quant_manifest_audit)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_quant_manifest_passes_for_matching_artifacts(tmp_path: Path) -> None:
    q4 = tmp_path / "weights.q4_0"
    q8 = tmp_path / "weights.q8_0"
    q4.write_bytes(struct.pack("<e", 1.0) + bytes([0x88] * 16))
    q8.write_bytes(struct.pack("<e", 2.0) + bytes([1] * 32))
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "artifacts": [
                {
                    "path": "weights.q4_0",
                    "format": "q4_0",
                    "sha256": quant_manifest_audit.file_sha256(q4),
                    "bytes": 18,
                    "block_count": 1,
                    "element_count": 32,
                },
                {
                    "path": "weights.q8_0",
                    "format": "q8_0",
                    "sha256": quant_manifest_audit.file_sha256(q8),
                    "bytes": 34,
                    "block_count": 1,
                    "element_count": 31,
                },
            ]
        },
    )

    audit = quant_manifest_audit.audit_manifest(manifest, root=tmp_path)

    assert audit.status == "pass"
    assert audit.artifact_count == 2
    assert audit.findings == []
    assert audit.artifacts[0].element_capacity == 32


def test_quant_manifest_reports_metadata_mismatches(tmp_path: Path) -> None:
    q4 = tmp_path / "weights.q4_0"
    q4.write_bytes(struct.pack("<e", 1.0) + bytes([0x88] * 16))
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "path": "weights.q4_0",
            "format": "q4_0",
            "sha256": "bad",
            "bytes": 17,
            "block_count": 2,
            "element_count": 64,
        },
    )

    audit = quant_manifest_audit.audit_manifest(manifest, root=tmp_path)

    kinds = {finding.kind for finding in audit.findings}
    assert audit.status == "fail"
    assert {"sha256_mismatch", "byte_count_mismatch", "block_count_mismatch", "element_count_over_capacity"}.issubset(kinds)


def test_quant_manifest_outputs_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    q8 = tmp_path / "weights.q8_0"
    q8.write_bytes(struct.pack("<e", 1.0) + bytes([0] * 32))
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "audit.json"
    csv = tmp_path / "audit.csv"
    markdown = tmp_path / "audit.md"
    junit = tmp_path / "audit.xml"
    write_json(
        manifest,
        {
            "path": "weights.q8_0",
            "format": "q8_0",
            "sha256": quant_manifest_audit.file_sha256(q8),
            "bytes": 34,
            "block_count": 1,
            "element_count": 32,
        },
    )

    status = quant_manifest_audit.main(
        [
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--csv",
            str(csv),
            "--markdown",
            str(markdown),
            "--junit",
            str(junit),
            "--fail-on-findings",
        ]
    )

    assert status == 0
    assert '"status": "pass"' in output.read_text(encoding="utf-8")
    assert "element_capacity" in csv.read_text(encoding="utf-8")
    assert "Status: PASS" in markdown.read_text(encoding="utf-8")
    suite = ET.parse(junit).getroot()
    assert suite.attrib["failures"] == "0"
