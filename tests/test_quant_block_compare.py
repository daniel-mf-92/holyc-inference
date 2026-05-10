#!/usr/bin/env python3
"""Tests for raw quant block stream comparison."""

from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import quant_block_compare


def half_bits(value: float) -> bytes:
    return struct.pack("<e", value)


def test_q4_compare_passes_identical_streams(tmp_path: Path) -> None:
    reference = tmp_path / "ref.q4"
    candidate = tmp_path / "cand.q4"
    payload = half_bits(1.0) + bytes([0x88] * 16)
    reference.write_bytes(payload)
    candidate.write_bytes(payload)

    result = quant_block_compare.compare_block_streams(reference, candidate, "q4_0")

    assert result.findings == []
    assert result.reference_blocks == 1
    assert result.candidate_blocks == 1
    assert result.scale_mismatch_count == 0
    assert result.quant_mismatch_count == 0
    assert result.quant_mismatch_pct == 0.0


def test_q4_compare_reports_scale_and_quant_mismatches(tmp_path: Path) -> None:
    reference = tmp_path / "ref.q4"
    candidate = tmp_path / "cand.q4"
    reference.write_bytes(half_bits(1.0) + bytes([0x88] * 16))
    candidate.write_bytes(half_bits(2.0) + bytes([0x89] + [0x88] * 15))

    result = quant_block_compare.compare_block_streams(
        reference,
        candidate,
        "q4_0",
        max_mismatches=0,
    )

    assert result.scale_mismatch_count == 1
    assert result.quant_mismatch_count == 1
    assert result.quant_mismatch_pct == 3.125
    assert result.block_mismatch_count == 1
    assert result.first_mismatch_block == 0
    assert "block stream mismatch: scale_mismatches=1 quant_mismatches=1" in result.findings
    assert "total mismatches 2 exceed limit 0" in result.findings
    assert result.mismatches[0].kind == "scale"
    assert result.mismatches[1].detail == "element=0 reference=0 candidate=1"


def test_compare_can_report_mismatches_without_failing_when_allowed(tmp_path: Path) -> None:
    reference = tmp_path / "ref.q4"
    candidate = tmp_path / "cand.q4"
    reference.write_bytes(half_bits(1.0) + bytes([0x88] * 16))
    candidate.write_bytes(half_bits(1.0) + bytes([0x89] + [0x88] * 15))

    result = quant_block_compare.compare_block_streams(reference, candidate, "q4_0", allow_mismatches=True)

    assert result.findings == []
    assert result.quant_mismatch_count == 1
    assert result.block_mismatch_count == 1


def test_q8_compare_reports_size_and_block_count_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "ref.q8"
    candidate = tmp_path / "cand.q8"
    reference.write_bytes(half_bits(1.0) + bytes([0] * 32))
    candidate.write_bytes(half_bits(1.0) + bytes([0] * 31))

    result = quant_block_compare.compare_block_streams(reference, candidate, "q8_0")

    assert result.byte_size_match is False
    assert result.block_count_match is False
    assert result.reference_blocks == 1
    assert result.candidate_blocks == 0
    assert any("not a multiple" in finding for finding in result.findings)
    assert any("block count mismatch" in finding for finding in result.findings)


def test_outputs_json_csv_markdown_and_junit(tmp_path: Path) -> None:
    reference = tmp_path / "ref.q8"
    candidate = tmp_path / "cand.q8"
    payload = half_bits(1.0) + bytes([0] * 32)
    reference.write_bytes(payload)
    candidate.write_bytes(payload)

    result = quant_block_compare.compare_block_streams(reference, candidate, "q8_0")
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"
    markdown_path = tmp_path / "result.md"
    junit_path = tmp_path / "result.xml"

    quant_block_compare.write_json(json_path, result)
    quant_block_compare.write_csv(csv_path, result)
    quant_block_compare.write_markdown(markdown_path, result)
    quant_block_compare.write_junit(junit_path, result)

    assert '"format": "q8_0"' in json_path.read_text(encoding="utf-8")
    assert "quant_mismatch_count" in csv_path.read_text(encoding="utf-8")
    assert "Status: PASS" in markdown_path.read_text(encoding="utf-8")
    suite = ET.parse(junit_path).getroot()
    assert suite.attrib["failures"] == "0"
