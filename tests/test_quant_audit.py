#!/usr/bin/env python3
"""Tests for host-side quantization audit tooling."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import quant_audit


def half_bits(value: float) -> bytes:
    return struct.pack("<e", value)


def test_source_audit_ignores_comments_and_fp16_names(tmp_path: Path) -> None:
    source = tmp_path / "q4_0_fixture.HC"
    source.write_text(
        """
// F64 float double 1.25 should not count in comments.
class Q4_0Block
{
    U16 d_fp16;
};
I64 Q4_0Scale(U16 d_fp16)
{
    return d_fp16 << 16;
}
""",
        encoding="utf-8",
    )

    audit = quant_audit.audit_sources(source)
    assert audit.files_scanned == 1
    assert audit.findings == []


def test_source_audit_flags_float_tokens_and_literals(tmp_path: Path) -> None:
    source = tmp_path / "bad.HC"
    source.write_text(
        """
F64 BadScale()
{
    return 1.5;
}
""",
        encoding="utf-8",
    )

    audit = quant_audit.audit_sources(source)
    kinds = {finding.kind for finding in audit.findings}
    assert kinds == {"float-token", "float-literal"}


def test_q4_0_block_audit_reports_signed_nibble_range(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    packed = bytes([0x0F] + [0x88] * 15)
    block_file.write_bytes(half_bits(1.0) + packed)

    audit = quant_audit.audit_q4_0_blocks(block_file, allow_inf_nan_scale=False)
    assert audit.findings == []
    assert audit.block_count == 1
    assert audit.quant_min == -8
    assert audit.quant_max == 7
    assert audit.scale_normal_count == 1


def test_q8_0_block_audit_checks_size_and_inf_scale(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(bytes.fromhex("007c") + bytes([0] * 32) + b"x")

    audit = quant_audit.audit_q8_0_blocks(block_file, allow_inf_nan_scale=False)
    assert audit.block_count == 1
    assert audit.scale_inf_nan_count == 1
    assert len(audit.findings) == 2
    assert "not a multiple" in audit.findings[0]
    assert "inf/nan" in audit.findings[1]


def test_cli_writes_pass_report(tmp_path: Path) -> None:
    source = tmp_path / "ok.HC"
    output = tmp_path / "report.json"
    source.write_text("I64 Good(U16 d_fp16) { return d_fp16; }\n", encoding="utf-8")

    status = quant_audit.main(["--source-root", str(source), "--output", str(output)])

    assert status == 0
    text = output.read_text(encoding="utf-8")
    assert '"status": "pass"' in text
