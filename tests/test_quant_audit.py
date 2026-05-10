#!/usr/bin/env python3
"""Tests for host-side quantization audit tooling."""

from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
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


def test_source_root_audit_deduplicates_extra_roots(tmp_path: Path) -> None:
    quant_root = tmp_path / "quant"
    math_root = tmp_path / "math"
    quant_root.mkdir()
    math_root.mkdir()
    (quant_root / "Q4.HC").write_text("I64 QuantScale(I64 x) { return x << 16; }\n", encoding="utf-8")
    (math_root / "Bad.HC").write_text("F64 BadScale() { return 1.25; }\n", encoding="utf-8")

    audit = quant_audit.audit_source_roots([quant_root, math_root, quant_root])

    assert audit.files_scanned == 2
    assert {finding.path for finding in audit.findings} == {str(math_root / "Bad.HC")}
    assert {finding.kind for finding in audit.findings} == {"float-token", "float-literal"}


def test_q4_0_block_audit_reports_signed_nibble_range(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    packed = bytes([0x0F] + [0x88] * 15)
    block_file.write_bytes(half_bits(1.0) + packed)

    audit = quant_audit.audit_q4_0_blocks(block_file, allow_inf_nan_scale=False)
    assert audit.findings == []
    assert audit.block_count == 1
    assert audit.element_capacity == 32
    assert audit.quant_min == -8
    assert audit.quant_max == 7
    assert audit.quant_histogram["-8"] == 1
    assert audit.quant_histogram["0"] == 30
    assert audit.quant_histogram["7"] == 1
    assert audit.quant_nonzero_count == 2
    assert audit.quant_zero_pct == 93.75
    assert audit.quant_used_value_count == 3
    assert audit.quant_saturation_count == 2
    assert audit.quant_saturation_pct == 6.25
    assert audit.min_block_used_quant_values == 3
    assert audit.min_block_used_quant_values_index == 0
    assert audit.worst_block_saturation_count == 2
    assert audit.worst_block_saturation_pct == 6.25
    assert audit.worst_block_saturation_index == 0
    assert audit.scale_normal_count == 1
    assert audit.scale_positive_count == 1
    assert audit.scale_negative_count == 0
    assert audit.scale_q16_min == 65536
    assert audit.scale_q16_max == 65536
    assert audit.scale_q16_abs_max == 65536
    assert audit.scale_q16_zero_count == 0
    assert audit.scale_q16_over_limit_count == 0


def test_q8_0_block_audit_checks_size_and_inf_scale(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(bytes.fromhex("007c") + bytes([0] * 32) + b"x")

    audit = quant_audit.audit_q8_0_blocks(block_file, allow_inf_nan_scale=False)
    assert audit.block_count == 1
    assert audit.scale_inf_nan_count == 1
    assert len(audit.findings) == 2
    assert "not a multiple" in audit.findings[0]
    assert "inf/nan" in audit.findings[1]


def test_block_audit_checks_expected_shape(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block_file.write_bytes(half_bits(1.0) + bytes([0x88] * 16))

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        expect_blocks=2,
        expect_elements=33,
    )

    assert audit.element_capacity == 32
    assert "expected 2 blocks, found 1" in audit.findings
    assert "expected 33 elements to occupy 2 blocks, found 1" in audit.findings
    assert "expected 33 elements exceeds block capacity 32" in audit.findings


def test_q4_block_audit_counts_nonzero_tail_padding(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    packed = bytes([0x88] * 15 + [0xF8])
    block_file.write_bytes(half_bits(1.0) + packed)

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        expect_elements=31,
        fail_nonzero_padding_quants=True,
    )

    assert audit.padding_element_count == 1
    assert audit.padding_nonzero_count == 1
    assert (
        "block 0: 1 nonzero padding quant entries after expected element count 31"
        in audit.findings
    )


def test_q8_block_audit_counts_nonzero_tail_padding(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    payload = bytes([0] * 30 + [0, 5])
    block_file.write_bytes(half_bits(1.0) + payload)

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        expect_elements=30,
        fail_nonzero_padding_quants=True,
    )

    assert audit.padding_element_count == 2
    assert audit.padding_nonzero_count == 1
    assert (
        "block 0: 1 nonzero padding quant entries after expected element count 30"
        in audit.findings
    )


def test_block_audit_checks_q16_scale_limit(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(half_bits(2.0) + bytes([0] * 32))

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        max_abs_scale_q16=100000,
    )

    assert audit.scale_q16_min == 131072
    assert audit.scale_q16_max == 131072
    assert audit.scale_q16_abs_max == 131072
    assert audit.scale_q16_zero_count == 0
    assert audit.scale_q16_over_limit_count == 1
    assert "|scale_q16| 131072 exceeds limit 100000" in audit.findings[0]


def test_block_audit_counts_zero_q16_scales(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block_file.write_bytes(half_bits(0.0) + bytes([0x88] * 16))

    audit = quant_audit.audit_q4_0_blocks(block_file, allow_inf_nan_scale=False)

    assert audit.scale_zero_count == 1
    assert audit.scale_q16_min == 0
    assert audit.scale_q16_max == 0
    assert audit.scale_q16_abs_max == 0
    assert audit.scale_q16_zero_count == 1
    assert audit.zero_scale_nonzero_quant_block_count == 0
    assert audit.zero_scale_nonzero_quant_entry_count == 0


def test_block_audit_can_fail_zero_fp16_scales(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block_file.write_bytes(half_bits(0.0) + bytes([0x88] * 16))

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        fail_zero_scales=True,
    )

    assert audit.scale_zero_count == 1
    assert "block 0: fp16 scale is zero bits=0x0000" in audit.findings


def test_block_audit_can_fail_subnormal_fp16_scales(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(bytes.fromhex("0100") + bytes([0] * 32))

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        fail_subnormal_scales=True,
    )

    assert audit.scale_subnormal_count == 1
    assert "block 0: fp16 scale is subnormal bits=0x0001" in audit.findings


def test_block_audit_counts_and_can_fail_negative_fp16_scales(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(half_bits(-0.5) + bytes([0] * 32))

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        fail_negative_scales=True,
    )

    assert audit.scale_positive_count == 0
    assert audit.scale_negative_count == 1
    assert audit.scale_q16_min == -32768
    assert audit.scale_q16_max == -32768
    assert audit.scale_q16_abs_max == 32768
    assert "block 0: fp16 scale sign is negative bits=0xb800" in audit.findings


def test_block_audit_reports_zero_scale_nonzero_quant_payload(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block_file.write_bytes(half_bits(0.0) + bytes([0x89] * 16))

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        fail_zero_scale_nonzero_blocks=True,
    )

    assert audit.zero_scale_nonzero_quant_block_count == 1
    assert audit.zero_scale_nonzero_quant_entry_count == 16
    assert "zero Q16 scale has 16 nonzero quant payload entries" in audit.findings[0]


def test_q8_block_audit_counts_zero_scale_nonzero_quant_payload(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(half_bits(0.0) + bytes([0, 1, -1 & 0xFF] + [0] * 29))

    audit = quant_audit.audit_q8_0_blocks(block_file, allow_inf_nan_scale=False)

    assert audit.findings == []
    assert audit.zero_scale_nonzero_quant_block_count == 1
    assert audit.zero_scale_nonzero_quant_entry_count == 2


def test_block_audit_checks_quant_distribution_gates(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block_file.write_bytes(half_bits(1.0) + bytes([0x0F] * 16))

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        min_used_quant_values=3,
        max_saturation_pct=90.0,
    )

    assert audit.quant_nonzero_count == 32
    assert audit.quant_used_value_count == 2
    assert audit.quant_saturation_count == 32
    assert audit.quant_saturation_pct == 100.0
    assert "used quant values 2 below minimum 3" in audit.findings
    assert "saturated quant values 100.000% exceeds limit 90.000%" in audit.findings


def test_block_audit_can_gate_zero_quant_percentage(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    block_file.write_bytes(half_bits(1.0) + bytes([0] * 24 + list(range(1, 9))))

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        max_zero_quant_pct=50.0,
    )

    assert audit.quant_zero_count == 24
    assert audit.quant_nonzero_count == 8
    assert audit.quant_zero_pct == 75.0
    assert "zero quant values 75.000% exceeds limit 50.000%" in audit.findings


def test_block_audit_checks_per_block_distribution_gates(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    first_block = half_bits(1.0) + bytes([0] * 32)
    second_payload = bytes([128] * 16 + [127] * 16)
    second_block = half_bits(1.0) + second_payload
    block_file.write_bytes(first_block + second_block)

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        min_used_quant_values=3,
        max_saturation_pct=90.0,
        min_block_used_quant_values=2,
        max_block_saturation_pct=50.0,
    )

    assert audit.quant_used_value_count == 3
    assert audit.quant_saturation_count == 32
    assert audit.quant_saturation_pct == 50.0
    assert audit.min_block_used_quant_values == 1
    assert audit.min_block_used_quant_values_index == 0
    assert audit.worst_block_saturation_count == 32
    assert audit.worst_block_saturation_pct == 100.0
    assert audit.worst_block_saturation_index == 1
    assert "block 0: used quant values 1 below per-block minimum 2" in audit.findings
    assert (
        "block 1: saturated quant values 100.000% exceeds per-block limit 50.000%"
        in audit.findings
    )


def test_block_audit_checks_duplicate_complete_blocks(tmp_path: Path) -> None:
    block_file = tmp_path / "q4.bin"
    block = half_bits(1.0) + bytes([0x88] * 16)
    block_file.write_bytes(block * 3)

    audit = quant_audit.audit_q4_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        max_duplicate_block_pct=50.0,
        max_identical_block_run=2,
    )

    assert audit.block_count == 3
    assert audit.duplicate_block_count == 2
    assert audit.duplicate_block_pct == 100.0 * 2 / 3
    assert audit.repeated_block_value_count == 1
    assert audit.max_identical_block_run == 3
    assert audit.max_identical_block_run_start == 0
    assert "duplicate blocks 2/3 (66.667%) exceeds limit 50.000%" in audit.findings
    assert "identical block run 3 exceeds limit 2 starting at block 0" in audit.findings


def test_block_audit_checks_scale_repetition_gates(tmp_path: Path) -> None:
    block_file = tmp_path / "q8.bin"
    first_block = half_bits(1.0) + bytes(range(32))
    second_block = half_bits(1.0) + bytes(range(1, 33))
    third_block = half_bits(2.0) + bytes(range(2, 34))
    block_file.write_bytes(first_block + second_block + third_block)

    audit = quant_audit.audit_q8_0_blocks(
        block_file,
        allow_inf_nan_scale=False,
        min_scale_used_values=3,
        max_duplicate_scale_pct=20.0,
        max_identical_scale_run=1,
    )

    assert audit.scale_used_value_count == 2
    assert audit.duplicate_scale_count == 1
    assert audit.duplicate_scale_pct == 100.0 / 3
    assert audit.max_identical_scale_run == 2
    assert audit.max_identical_scale_run_start == 0
    assert "fp16 scale values 2 below minimum 3" in audit.findings
    assert "duplicate fp16 scales 1/3 (33.333%) exceeds limit 20.000%" in audit.findings
    assert "identical fp16 scale run 2 exceeds limit 1 starting at block 0" in audit.findings


def test_cli_writes_pass_report(tmp_path: Path) -> None:
    source = tmp_path / "ok.HC"
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    source.write_text("I64 Good(U16 d_fp16) { return d_fp16; }\n", encoding="utf-8")

    status = quant_audit.main(
        ["--source-root", str(source), "--output", str(output), "--markdown", str(markdown)]
    )

    assert status == 0
    text = output.read_text(encoding="utf-8")
    assert '"status": "pass"' in text
    assert '"source_roots": [' in text
    assert "Quantization Audit" in markdown.read_text(encoding="utf-8")


def test_cli_fails_on_q16_scale_limit(tmp_path: Path) -> None:
    source = tmp_path / "ok.HC"
    q4_file = tmp_path / "q4.bin"
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    csv_report = tmp_path / "report.csv"
    junit = tmp_path / "report.xml"
    source.write_text("I64 Good(U16 d_fp16) { return d_fp16; }\n", encoding="utf-8")
    q4_file.write_bytes(half_bits(1.0) + bytes([0x88] * 16))

    status = quant_audit.main(
        [
            "--source-root",
            str(source),
            "--q4-block-file",
            str(q4_file),
            "--max-abs-scale-q16",
            "32768",
            "--min-used-quant-values",
            "1",
            "--min-block-used-quant-values",
            "1",
            "--max-block-saturation-pct",
            "100",
            "--max-zero-quant-pct",
            "99",
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--csv",
            str(csv_report),
            "--junit",
            str(junit),
        ]
    )

    assert status == 1
    report = output.read_text(encoding="utf-8")
    assert '"scale_q16_abs_max": 65536' in report
    assert '"scale_q16_over_limit_count": 1' in report
    assert '"scale_positive_count": 1' in report
    assert '"scale_negative_count": 0' in report
    assert '"quant_zero_pct": 100.0' in report
    assert '"min_block_used_quant_values": 1' in report
    assert '"worst_block_saturation_count": 0' in report
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "Scale Q16 min/max/absmax/zero/overlimit" in markdown_text
    assert "Scale values" in markdown_text
    assert "Duplicate scales" in markdown_text
    assert "Scale sign +/-" in markdown_text
    assert "Padding elements/nonzero" in markdown_text
    assert "Zero-scale nonzero blocks/entries" in markdown_text
    assert "Used values" in markdown_text
    assert "100.000% zero" in markdown_text
    assert "Min block used values" in markdown_text
    assert "block," in csv_report.read_text(encoding="utf-8")
    junit_root = ET.parse(junit).getroot()
    assert junit_root.attrib["name"] == "holyc_quant_audit"
    assert junit_root.attrib["tests"] == "3"
    assert junit_root.attrib["failures"] == "2"
    failure = junit_root.find("./testcase/failure")
    assert failure is not None
    assert "|scale_q16| 65536 exceeds limit 32768" in (failure.text or "")


def test_cli_fails_on_nonzero_padding_quants(tmp_path: Path) -> None:
    source = tmp_path / "ok.HC"
    q8_file = tmp_path / "q8.bin"
    output = tmp_path / "report.json"
    source.write_text("I64 Good(U16 d_fp16) { return d_fp16; }\n", encoding="utf-8")
    q8_file.write_bytes(half_bits(1.0) + bytes([0] * 31 + [1]))

    status = quant_audit.main(
        [
            "--source-root",
            str(source),
            "--q8-block-file",
            str(q8_file),
            "--expect-elements",
            "31",
            "--fail-nonzero-padding-quants",
            "--output",
            str(output),
        ]
    )

    report = output.read_text(encoding="utf-8")
    assert status == 1
    assert '"padding_element_count": 1' in report
    assert '"padding_nonzero_count": 1' in report
    assert "nonzero padding quant entries" in report


def test_cli_audits_mixed_q4_and_q8_block_files(tmp_path: Path) -> None:
    source = tmp_path / "ok.HC"
    q4_file = tmp_path / "q4.bin"
    q8_file = tmp_path / "q8.bin"
    output = tmp_path / "report.json"
    source.write_text("I64 Good(U16 d_fp16) { return d_fp16; }\n", encoding="utf-8")
    q4_file.write_bytes(half_bits(1.0) + bytes([0x88] * 16))
    q8_file.write_bytes(half_bits(2.0) + bytes(range(32)))

    status = quant_audit.main(
        [
            "--source-root",
            str(source),
            "--q4-block-file",
            str(q4_file),
            "--q8-block-file",
            str(q8_file),
            "--expect-blocks",
            "1",
            "--output",
            str(output),
        ]
    )

    report = output.read_text(encoding="utf-8")
    assert status == 0
    assert '"format": "q4_0"' in report
    assert '"format": "q8_0"' in report
    assert '"block_count": 1' in report
