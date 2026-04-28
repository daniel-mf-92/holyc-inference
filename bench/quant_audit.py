#!/usr/bin/env python3
"""Host-side quantization audit for HolyC inference sources and raw blocks.

The audit intentionally runs outside TempleOS. It validates two invariants:

* quant HolyC sources do not introduce float runtime types or math helpers
* raw Q4_0/Q8_0 block streams have valid sizes, sane fp16/Q16 scale fields,
  and optional quant-payload distribution gates
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


Q4_0_BLOCK_BYTES = 18
Q4_0_PACKED_BYTES = 16
Q8_0_BLOCK_BYTES = 34
Q8_0_PACKED_BYTES = 32
BLOCK_ELEMENTS = 32

FLOAT_TOKEN_RE = re.compile(
    r"\b(?:F32|F64|Float|Double|float|double|long\s+double|sin|cos|tan|sqrt|exp|log|pow)\b"
)
FP_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class SourceFinding:
    path: str
    line: int
    column: int
    kind: str
    text: str


@dataclass(frozen=True)
class SourceAudit:
    files_scanned: int
    findings: list[SourceFinding]


@dataclass(frozen=True)
class BlockAudit:
    format: str
    path: str
    bytes_read: int
    block_count: int
    element_capacity: int
    scale_zero_count: int
    scale_subnormal_count: int
    scale_normal_count: int
    scale_inf_nan_count: int
    scale_q16_min: int
    scale_q16_max: int
    scale_q16_abs_max: int
    scale_q16_zero_count: int
    scale_q16_over_limit_count: int
    zero_scale_nonzero_quant_block_count: int
    zero_scale_nonzero_quant_entry_count: int
    quant_min: int
    quant_max: int
    quant_zero_count: int
    quant_nonzero_count: int
    quant_used_value_count: int
    quant_saturation_count: int
    quant_saturation_pct: float
    min_block_used_quant_values: int
    min_block_used_quant_values_index: int
    worst_block_saturation_count: int
    worst_block_saturation_pct: float
    worst_block_saturation_index: int
    quant_histogram: dict[str, int]
    findings: list[str]


def strip_holyc_comments(source: str) -> str:
    """Remove comments while preserving line/column positions."""

    chars = list(source)
    index = 0
    in_block = False
    in_line = False
    in_string = False

    while index < len(chars):
        cur = chars[index]
        nxt = chars[index + 1] if index + 1 < len(chars) else ""

        if in_line:
            if cur == "\n":
                in_line = False
            else:
                chars[index] = " "
            index += 1
            continue

        if in_block:
            if cur == "*" and nxt == "/":
                chars[index] = " "
                chars[index + 1] = " "
                in_block = False
                index += 2
            else:
                if cur != "\n":
                    chars[index] = " "
                index += 1
            continue

        if in_string:
            if cur == "\\" and nxt:
                chars[index] = " "
                chars[index + 1] = " "
                index += 2
                continue
            if cur == '"':
                in_string = False
            if cur != "\n":
                chars[index] = " "
            index += 1
            continue

        if cur == "/" and nxt == "/":
            chars[index] = " "
            chars[index + 1] = " "
            in_line = True
            index += 2
            continue

        if cur == "/" and nxt == "*":
            chars[index] = " "
            chars[index + 1] = " "
            in_block = True
            index += 2
            continue

        if cur == '"':
            chars[index] = " "
            in_string = True

        index += 1

    return "".join(chars)


def iter_holyc_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix == ".HC":
            yield root
        return
    yield from sorted(root.rglob("*.HC"))


def line_col(source: str, offset: int) -> tuple[int, int]:
    line = source.count("\n", 0, offset) + 1
    line_start = source.rfind("\n", 0, offset)
    column = offset + 1 if line_start < 0 else offset - line_start
    return line, column


def audit_sources(root: Path) -> SourceAudit:
    findings: list[SourceFinding] = []
    files_scanned = 0

    for path in iter_holyc_files(root):
        files_scanned += 1
        raw = path.read_text(encoding="utf-8")
        stripped = strip_holyc_comments(raw)
        rel_path = str(path)

        for match in FLOAT_TOKEN_RE.finditer(stripped):
            line, column = line_col(stripped, match.start())
            findings.append(SourceFinding(rel_path, line, column, "float-token", match.group(0)))

        for match in FP_LITERAL_RE.finditer(stripped):
            line, column = line_col(stripped, match.start())
            findings.append(SourceFinding(rel_path, line, column, "float-literal", match.group(0)))

    return SourceAudit(files_scanned=files_scanned, findings=findings)


def fp16_class(bits: int) -> str:
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x03FF
    if exponent == 0:
        return "zero" if fraction == 0 else "subnormal"
    if exponent == 0x1F:
        return "inf_nan"
    return "normal"


def fp16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def fp16_to_q16(bits: int) -> int:
    return int(round(fp16_to_float(bits) * (1 << 16)))


def update_scale_q16_stats(
    findings: list[str],
    stats: dict[str, int | None],
    block_index: int,
    scale_bits: int,
    max_abs_scale_q16: int | None,
) -> None:
    scale_q16 = fp16_to_q16(scale_bits)
    abs_scale_q16 = abs(scale_q16)
    stats["min"] = (
        scale_q16 if stats["min"] is None else min(int(stats["min"]), scale_q16)
    )
    stats["max"] = (
        scale_q16 if stats["max"] is None else max(int(stats["max"]), scale_q16)
    )
    stats["abs_max"] = max(int(stats["abs_max"] or 0), abs_scale_q16)
    if scale_q16 == 0:
        stats["zero_count"] = int(stats["zero_count"] or 0) + 1
    if max_abs_scale_q16 is not None and abs_scale_q16 > max_abs_scale_q16:
        stats["over_limit_count"] = int(stats["over_limit_count"] or 0) + 1
        findings.append(
            (
                "block {block}: |scale_q16| {actual} exceeds limit {limit} "
                "(fp16 bits=0x{bits:04x})"
            ).format(
                block=block_index,
                actual=abs_scale_q16,
                limit=max_abs_scale_q16,
                bits=scale_bits,
            )
        )


def check_zero_scale_quant_payload(
    findings: list[str],
    block_index: int,
    scale_q16: int | None,
    block_nonzero_quant_count: int,
    fail_zero_scale_nonzero_blocks: bool,
) -> tuple[int, int]:
    if scale_q16 != 0 or block_nonzero_quant_count == 0:
        return 0, 0

    if fail_zero_scale_nonzero_blocks:
        findings.append(
            (
                "block {block}: zero Q16 scale has {entries} nonzero quant payload entries"
            ).format(block=block_index, entries=block_nonzero_quant_count)
        )

    return 1, block_nonzero_quant_count


def check_scale_class_gates(
    findings: list[str],
    block_index: int,
    scale_bits: int,
    scale_class: str,
    fail_zero_scales: bool,
    fail_subnormal_scales: bool,
) -> None:
    if fail_zero_scales and scale_class == "zero":
        findings.append(f"block {block_index}: fp16 scale is zero bits=0x{scale_bits:04x}")
    if fail_subnormal_scales and scale_class == "subnormal":
        findings.append(f"block {block_index}: fp16 scale is subnormal bits=0x{scale_bits:04x}")


def check_expected_shape(
    findings: list[str],
    block_count: int,
    expect_blocks: int | None,
    expect_elements: int | None,
) -> None:
    if expect_blocks is not None and block_count != expect_blocks:
        findings.append(f"expected {expect_blocks} blocks, found {block_count}")

    if expect_elements is None:
        return

    min_blocks = (expect_elements + BLOCK_ELEMENTS - 1) // BLOCK_ELEMENTS
    capacity = block_count * BLOCK_ELEMENTS
    if block_count != min_blocks:
        findings.append(
            f"expected {expect_elements} elements to occupy {min_blocks} blocks, found {block_count}"
        )
    if expect_elements > capacity:
        findings.append(f"expected {expect_elements} elements exceeds block capacity {capacity}")


def empty_histogram(min_value: int, max_value: int) -> dict[str, int]:
    return {str(value): 0 for value in range(min_value, max_value + 1)}


def check_quant_distribution(
    findings: list[str],
    quant_histogram: dict[str, int],
    saturation_values: tuple[int, int],
    min_used_quant_values: int | None,
    max_saturation_pct: float | None,
) -> tuple[int, int, int, float]:
    used_value_count = sum(1 for count in quant_histogram.values() if count)
    nonzero_count = sum(
        count for value, count in quant_histogram.items() if int(value) != 0
    )
    total_count = sum(quant_histogram.values())
    saturation_count = sum(quant_histogram[str(value)] for value in saturation_values)
    saturation_pct = (saturation_count * 100.0 / total_count) if total_count else 0.0

    if min_used_quant_values is not None and used_value_count < min_used_quant_values:
        findings.append(
            f"used quant values {used_value_count} below minimum {min_used_quant_values}"
        )
    if max_saturation_pct is not None and saturation_pct > max_saturation_pct:
        findings.append(
            "saturated quant values {pct:.3f}% exceeds limit {limit:.3f}%".format(
                pct=saturation_pct,
                limit=max_saturation_pct,
            )
        )

    return nonzero_count, used_value_count, saturation_count, saturation_pct


def check_block_quant_distribution(
    findings: list[str],
    block_index: int,
    block_histogram: dict[int, int],
    saturation_values: tuple[int, int],
    min_block_used_quant_values: int | None,
    max_block_saturation_pct: float | None,
) -> tuple[int, int, float]:
    used_value_count = sum(1 for count in block_histogram.values() if count)
    total_count = sum(block_histogram.values())
    saturation_count = sum(block_histogram.get(value, 0) for value in saturation_values)
    saturation_pct = (saturation_count * 100.0 / total_count) if total_count else 0.0

    if (
        min_block_used_quant_values is not None
        and used_value_count < min_block_used_quant_values
    ):
        findings.append(
            "block {block}: used quant values {actual} below per-block minimum {limit}".format(
                block=block_index,
                actual=used_value_count,
                limit=min_block_used_quant_values,
            )
        )
    if max_block_saturation_pct is not None and saturation_pct > max_block_saturation_pct:
        findings.append(
            "block {block}: saturated quant values {pct:.3f}% exceeds per-block limit {limit:.3f}%".format(
                block=block_index,
                pct=saturation_pct,
                limit=max_block_saturation_pct,
            )
        )

    return used_value_count, saturation_count, saturation_pct


def audit_q4_0_blocks(
    path: Path,
    allow_inf_nan_scale: bool,
    expect_blocks: int | None = None,
    expect_elements: int | None = None,
    max_abs_scale_q16: int | None = None,
    min_used_quant_values: int | None = None,
    max_saturation_pct: float | None = None,
    min_block_used_quant_values: int | None = None,
    max_block_saturation_pct: float | None = None,
    fail_zero_scale_nonzero_blocks: bool = False,
    fail_zero_scales: bool = False,
    fail_subnormal_scales: bool = False,
) -> BlockAudit:
    data = path.read_bytes()
    findings: list[str] = []
    if len(data) % Q4_0_BLOCK_BYTES:
        findings.append(f"size {len(data)} is not a multiple of Q4_0 block size {Q4_0_BLOCK_BYTES}")

    block_count = len(data) // Q4_0_BLOCK_BYTES
    check_expected_shape(findings, block_count, expect_blocks, expect_elements)
    counts = {"zero": 0, "subnormal": 0, "normal": 0, "inf_nan": 0}
    quant_min = math.inf
    quant_max = -math.inf
    quant_zero_count = 0
    zero_scale_nonzero_quant_block_count = 0
    zero_scale_nonzero_quant_entry_count = 0
    quant_histogram = empty_histogram(-8, 7)
    min_block_used_values = math.inf
    min_block_used_values_index = -1
    worst_block_saturation_count = 0
    worst_block_saturation_pct = 0.0
    worst_block_saturation_index = -1
    scale_q16_stats: dict[str, int | None] = {
        "min": None,
        "max": None,
        "abs_max": 0,
        "zero_count": 0,
        "over_limit_count": 0,
    }

    for block_index in range(block_count):
        offset = block_index * Q4_0_BLOCK_BYTES
        scale_bits = int.from_bytes(data[offset : offset + 2], "little")
        scale_class = fp16_class(scale_bits)
        counts[scale_class] += 1
        check_scale_class_gates(
            findings,
            block_index,
            scale_bits,
            scale_class,
            fail_zero_scales,
            fail_subnormal_scales,
        )
        if scale_class == "inf_nan" and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale is inf/nan bits=0x{scale_bits:04x}")

        scale_value = fp16_to_float(scale_bits)
        scale_q16: int | None = None
        if math.isnan(scale_value) and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale decoded to nan")
        if math.isfinite(scale_value):
            scale_q16 = fp16_to_q16(scale_bits)
            update_scale_q16_stats(
                findings,
                scale_q16_stats,
                block_index,
                scale_bits,
                max_abs_scale_q16,
            )

        packed = data[offset + 2 : offset + 2 + Q4_0_PACKED_BYTES]
        block_nonzero_quant_count = 0
        block_histogram = {value: 0 for value in range(-8, 8)}
        for byte in packed:
            for nibble in (byte & 0x0F, (byte >> 4) & 0x0F):
                signed = nibble - 8
                quant_min = min(quant_min, signed)
                quant_max = max(quant_max, signed)
                quant_histogram[str(signed)] += 1
                block_histogram[signed] += 1
                if signed == 0:
                    quant_zero_count += 1
                else:
                    block_nonzero_quant_count += 1

        block_used_values, block_saturation_count, block_saturation_pct = (
            check_block_quant_distribution(
                findings,
                block_index,
                block_histogram,
                (-8, 7),
                min_block_used_quant_values,
                max_block_saturation_pct,
            )
        )
        if block_used_values < min_block_used_values:
            min_block_used_values = block_used_values
            min_block_used_values_index = block_index
        if block_saturation_pct > worst_block_saturation_pct:
            worst_block_saturation_count = block_saturation_count
            worst_block_saturation_pct = block_saturation_pct
            worst_block_saturation_index = block_index

        zero_blocks, zero_entries = check_zero_scale_quant_payload(
            findings,
            block_index,
            scale_q16,
            block_nonzero_quant_count,
            fail_zero_scale_nonzero_blocks,
        )
        zero_scale_nonzero_quant_block_count += zero_blocks
        zero_scale_nonzero_quant_entry_count += zero_entries

    if block_count == 0:
        quant_min = 0
        quant_max = 0
        min_block_used_values = 0

    (
        quant_nonzero_count,
        quant_used_value_count,
        quant_saturation_count,
        quant_saturation_pct,
    ) = check_quant_distribution(
        findings,
        quant_histogram,
        (-8, 7),
        min_used_quant_values,
        max_saturation_pct,
    )

    return BlockAudit(
        format="q4_0",
        path=str(path),
        bytes_read=len(data),
        block_count=block_count,
        element_capacity=block_count * BLOCK_ELEMENTS,
        scale_zero_count=counts["zero"],
        scale_subnormal_count=counts["subnormal"],
        scale_normal_count=counts["normal"],
        scale_inf_nan_count=counts["inf_nan"],
        scale_q16_min=int(scale_q16_stats["min"] or 0),
        scale_q16_max=int(scale_q16_stats["max"] or 0),
        scale_q16_abs_max=int(scale_q16_stats["abs_max"] or 0),
        scale_q16_zero_count=int(scale_q16_stats["zero_count"] or 0),
        scale_q16_over_limit_count=int(scale_q16_stats["over_limit_count"] or 0),
        zero_scale_nonzero_quant_block_count=zero_scale_nonzero_quant_block_count,
        zero_scale_nonzero_quant_entry_count=zero_scale_nonzero_quant_entry_count,
        quant_min=int(quant_min),
        quant_max=int(quant_max),
        quant_zero_count=quant_zero_count,
        quant_nonzero_count=quant_nonzero_count,
        quant_used_value_count=quant_used_value_count,
        quant_saturation_count=quant_saturation_count,
        quant_saturation_pct=quant_saturation_pct,
        min_block_used_quant_values=int(min_block_used_values),
        min_block_used_quant_values_index=min_block_used_values_index,
        worst_block_saturation_count=worst_block_saturation_count,
        worst_block_saturation_pct=worst_block_saturation_pct,
        worst_block_saturation_index=worst_block_saturation_index,
        quant_histogram=quant_histogram,
        findings=findings,
    )


def audit_q8_0_blocks(
    path: Path,
    allow_inf_nan_scale: bool,
    expect_blocks: int | None = None,
    expect_elements: int | None = None,
    max_abs_scale_q16: int | None = None,
    min_used_quant_values: int | None = None,
    max_saturation_pct: float | None = None,
    min_block_used_quant_values: int | None = None,
    max_block_saturation_pct: float | None = None,
    fail_zero_scale_nonzero_blocks: bool = False,
    fail_zero_scales: bool = False,
    fail_subnormal_scales: bool = False,
) -> BlockAudit:
    data = path.read_bytes()
    findings: list[str] = []
    if len(data) % Q8_0_BLOCK_BYTES:
        findings.append(f"size {len(data)} is not a multiple of Q8_0 block size {Q8_0_BLOCK_BYTES}")

    block_count = len(data) // Q8_0_BLOCK_BYTES
    check_expected_shape(findings, block_count, expect_blocks, expect_elements)
    counts = {"zero": 0, "subnormal": 0, "normal": 0, "inf_nan": 0}
    quant_min = math.inf
    quant_max = -math.inf
    quant_zero_count = 0
    zero_scale_nonzero_quant_block_count = 0
    zero_scale_nonzero_quant_entry_count = 0
    quant_histogram = empty_histogram(-128, 127)
    min_block_used_values = math.inf
    min_block_used_values_index = -1
    worst_block_saturation_count = 0
    worst_block_saturation_pct = 0.0
    worst_block_saturation_index = -1
    scale_q16_stats: dict[str, int | None] = {
        "min": None,
        "max": None,
        "abs_max": 0,
        "zero_count": 0,
        "over_limit_count": 0,
    }

    for block_index in range(block_count):
        offset = block_index * Q8_0_BLOCK_BYTES
        scale_bits = int.from_bytes(data[offset : offset + 2], "little")
        scale_class = fp16_class(scale_bits)
        counts[scale_class] += 1
        check_scale_class_gates(
            findings,
            block_index,
            scale_bits,
            scale_class,
            fail_zero_scales,
            fail_subnormal_scales,
        )
        if scale_class == "inf_nan" and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale is inf/nan bits=0x{scale_bits:04x}")

        scale_value = fp16_to_float(scale_bits)
        scale_q16: int | None = None
        if math.isnan(scale_value) and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale decoded to nan")
        if math.isfinite(scale_value):
            scale_q16 = fp16_to_q16(scale_bits)
            update_scale_q16_stats(
                findings,
                scale_q16_stats,
                block_index,
                scale_bits,
                max_abs_scale_q16,
            )

        qs = data[offset + 2 : offset + 2 + Q8_0_PACKED_BYTES]
        block_nonzero_quant_count = 0
        block_histogram = {value: 0 for value in range(-128, 128)}
        for value in struct.unpack("<32b", qs):
            quant_min = min(quant_min, value)
            quant_max = max(quant_max, value)
            quant_histogram[str(value)] += 1
            block_histogram[value] += 1
            if value == 0:
                quant_zero_count += 1
            else:
                block_nonzero_quant_count += 1

        block_used_values, block_saturation_count, block_saturation_pct = (
            check_block_quant_distribution(
                findings,
                block_index,
                block_histogram,
                (-128, 127),
                min_block_used_quant_values,
                max_block_saturation_pct,
            )
        )
        if block_used_values < min_block_used_values:
            min_block_used_values = block_used_values
            min_block_used_values_index = block_index
        if block_saturation_pct > worst_block_saturation_pct:
            worst_block_saturation_count = block_saturation_count
            worst_block_saturation_pct = block_saturation_pct
            worst_block_saturation_index = block_index

        zero_blocks, zero_entries = check_zero_scale_quant_payload(
            findings,
            block_index,
            scale_q16,
            block_nonzero_quant_count,
            fail_zero_scale_nonzero_blocks,
        )
        zero_scale_nonzero_quant_block_count += zero_blocks
        zero_scale_nonzero_quant_entry_count += zero_entries

    if block_count == 0:
        quant_min = 0
        quant_max = 0
        min_block_used_values = 0

    (
        quant_nonzero_count,
        quant_used_value_count,
        quant_saturation_count,
        quant_saturation_pct,
    ) = check_quant_distribution(
        findings,
        quant_histogram,
        (-128, 127),
        min_used_quant_values,
        max_saturation_pct,
    )

    return BlockAudit(
        format="q8_0",
        path=str(path),
        bytes_read=len(data),
        block_count=block_count,
        element_capacity=block_count * BLOCK_ELEMENTS,
        scale_zero_count=counts["zero"],
        scale_subnormal_count=counts["subnormal"],
        scale_normal_count=counts["normal"],
        scale_inf_nan_count=counts["inf_nan"],
        scale_q16_min=int(scale_q16_stats["min"] or 0),
        scale_q16_max=int(scale_q16_stats["max"] or 0),
        scale_q16_abs_max=int(scale_q16_stats["abs_max"] or 0),
        scale_q16_zero_count=int(scale_q16_stats["zero_count"] or 0),
        scale_q16_over_limit_count=int(scale_q16_stats["over_limit_count"] or 0),
        zero_scale_nonzero_quant_block_count=zero_scale_nonzero_quant_block_count,
        zero_scale_nonzero_quant_entry_count=zero_scale_nonzero_quant_entry_count,
        quant_min=int(quant_min),
        quant_max=int(quant_max),
        quant_zero_count=quant_zero_count,
        quant_nonzero_count=quant_nonzero_count,
        quant_used_value_count=quant_used_value_count,
        quant_saturation_count=quant_saturation_count,
        quant_saturation_pct=quant_saturation_pct,
        min_block_used_quant_values=int(min_block_used_values),
        min_block_used_quant_values_index=min_block_used_values_index,
        worst_block_saturation_count=worst_block_saturation_count,
        worst_block_saturation_pct=worst_block_saturation_pct,
        worst_block_saturation_index=worst_block_saturation_index,
        quant_histogram=quant_histogram,
        findings=findings,
    )


def audit_blocks(
    path: Path,
    quant_format: str,
    allow_inf_nan_scale: bool,
    expect_blocks: int | None = None,
    expect_elements: int | None = None,
    max_abs_scale_q16: int | None = None,
    min_used_quant_values: int | None = None,
    max_saturation_pct: float | None = None,
    min_block_used_quant_values: int | None = None,
    max_block_saturation_pct: float | None = None,
    fail_zero_scale_nonzero_blocks: bool = False,
    fail_zero_scales: bool = False,
    fail_subnormal_scales: bool = False,
) -> BlockAudit:
    if quant_format == "q4_0":
        return audit_q4_0_blocks(
            path,
            allow_inf_nan_scale,
            expect_blocks,
            expect_elements,
            max_abs_scale_q16,
            min_used_quant_values,
            max_saturation_pct,
            min_block_used_quant_values,
            max_block_saturation_pct,
            fail_zero_scale_nonzero_blocks,
            fail_zero_scales,
            fail_subnormal_scales,
        )
    if quant_format == "q8_0":
        return audit_q8_0_blocks(
            path,
            allow_inf_nan_scale,
            expect_blocks,
            expect_elements,
            max_abs_scale_q16,
            min_used_quant_values,
            max_saturation_pct,
            min_block_used_quant_values,
            max_block_saturation_pct,
            fail_zero_scale_nonzero_blocks,
            fail_zero_scales,
            fail_subnormal_scales,
        )
    raise ValueError(f"unsupported quant format: {quant_format}")


def encode_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def markdown_report(report: dict) -> str:
    source = report["source_audit"]
    lines = [
        "# Quantization Audit",
        "",
        f"Status: {report['status']}",
        f"Generated: {report['generated_at']}",
        f"Source root: `{report['source_root']}`",
        f"HolyC files scanned: {source['files_scanned']}",
        f"Source findings: {len(source['findings'])}",
        "",
        "## Block Audits",
        "",
    ]

    block_audits = report["block_audits"]
    if not block_audits:
        lines.append("No block files supplied.")
    else:
        lines.extend(
            [
                "| Format | Path | Blocks | Capacity | Scale zero/subnormal/normal/inf_nan "
                "| Scale Q16 min/max/absmax/zero/overlimit | Zero-scale nonzero blocks/entries | Quant min/max | Used values | Zero/nonzero quants "
                "| Saturated quants | Min block used values | Worst block saturation | Findings |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for audit in block_audits:
            scale_counts = "/".join(
                str(audit[name])
                for name in (
                    "scale_zero_count",
                    "scale_subnormal_count",
                    "scale_normal_count",
                    "scale_inf_nan_count",
                )
            )
            scale_q16 = "{}/{}/{}/{}/{}".format(
                audit["scale_q16_min"],
                audit["scale_q16_max"],
                audit["scale_q16_abs_max"],
                audit["scale_q16_zero_count"],
                audit["scale_q16_over_limit_count"],
            )
            lines.append(
                (
                    "| {format} | `{path}` | {block_count} | {element_capacity} | "
                    "{scales} | {scale_q16} | {zero_scale_nonzero_quant_block_count}/{zero_scale_nonzero_quant_entry_count} | "
                    "{quant_min}/{quant_max} | {quant_used_value_count} | "
                    "{quant_zero_count}/{quant_nonzero_count} | {quant_saturation_count} ({quant_saturation_pct:.3f}%) | "
                    "{min_block_used_quant_values} @ {min_block_used_quant_values_index} | "
                    "{worst_block_saturation_count} ({worst_block_saturation_pct:.3f}%) @ {worst_block_saturation_index} | "
                    "{findings} |"
                ).format(
                    format=audit["format"],
                    path=audit["path"],
                    block_count=audit["block_count"],
                    element_capacity=audit["element_capacity"],
                    scales=scale_counts,
                    scale_q16=scale_q16,
                    zero_scale_nonzero_quant_block_count=audit[
                        "zero_scale_nonzero_quant_block_count"
                    ],
                    zero_scale_nonzero_quant_entry_count=audit[
                        "zero_scale_nonzero_quant_entry_count"
                    ],
                    quant_min=audit["quant_min"],
                    quant_max=audit["quant_max"],
                    quant_used_value_count=audit["quant_used_value_count"],
                    quant_zero_count=audit["quant_zero_count"],
                    quant_nonzero_count=audit["quant_nonzero_count"],
                    quant_saturation_count=audit["quant_saturation_count"],
                    quant_saturation_pct=audit["quant_saturation_pct"],
                    min_block_used_quant_values=audit["min_block_used_quant_values"],
                    min_block_used_quant_values_index=audit[
                        "min_block_used_quant_values_index"
                    ],
                    worst_block_saturation_count=audit["worst_block_saturation_count"],
                    worst_block_saturation_pct=audit["worst_block_saturation_pct"],
                    worst_block_saturation_index=audit["worst_block_saturation_index"],
                    findings=len(audit["findings"]),
                )
            )

    if source["findings"]:
        lines.extend(["", "## Source Findings", ""])
        for finding in source["findings"]:
            lines.append(
                f"- `{finding['path']}`:{finding['line']}:{finding['column']} "
                f"{finding['kind']} `{finding['text']}`"
            )

    for audit in block_audits:
        if audit["findings"]:
            lines.extend(["", f"## Findings: `{audit['path']}`", ""])
            lines.extend(f"- {finding}" for finding in audit["findings"])

    return "\n".join(lines) + "\n"


def iter_finding_rows(report: dict[str, Any]) -> Iterable[dict[str, str]]:
    source_audit = report["source_audit"]
    for finding in source_audit["findings"]:
        yield {
            "scope": "source",
            "path": str(finding["path"]),
            "line": str(finding["line"]),
            "column": str(finding["column"]),
            "format": "",
            "kind": str(finding["kind"]),
            "reason": f"{finding['kind']} {finding['text']}",
            "text": str(finding["text"]),
        }

    for audit in report["block_audits"]:
        for finding in audit["findings"]:
            yield {
                "scope": "block",
                "path": str(audit["path"]),
                "line": "",
                "column": "",
                "format": str(audit["format"]),
                "kind": "block-finding",
                "reason": str(finding),
                "text": "",
            }


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["scope", "path", "line", "column", "format", "kind", "reason", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(iter_finding_rows(report))


def write_junit(report: dict[str, Any], path: Path) -> None:
    rows = list(iter_finding_rows(report))
    passing_checks = 0
    if not report["source_audit"]["findings"]:
        passing_checks += 1
    passing_checks += sum(1 for audit in report["block_audits"] if not audit["findings"])

    testcase_count = len(rows) + passing_checks
    if testcase_count == 0:
        testcase_count = 1

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_quant_audit",
            "tests": str(testcase_count),
            "failures": str(len(rows)),
            "errors": "0",
        },
    )

    if not report["source_audit"]["findings"]:
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "quant_audit.source",
                "name": "source_float_runtime_scan",
            },
        )

    for audit in report["block_audits"]:
        if not audit["findings"]:
            ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": "quant_audit.blocks",
                    "name": f"{Path(audit['path']).name}:{audit['format']}",
                },
            )

    for index, row in enumerate(rows, 1):
        classname = f"quant_audit.{row['scope']}"
        name_parts = [Path(row["path"]).name or row["path"]]
        if row["line"]:
            name_parts.append(row["line"])
        if row["format"]:
            name_parts.append(row["format"])
        name_parts.append(str(index))
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": classname,
                "name": ":".join(name_parts),
            },
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "quant_audit_violation",
                "message": row["reason"],
            },
        )
        failure.text = "\n".join(
            [
                f"scope={row['scope']}",
                f"path={row['path']}",
                f"line={row['line']}",
                f"column={row['column']}",
                f"format={row['format']}",
                f"kind={row['kind']}",
                f"reason={row['reason']}",
            ]
        )

    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("src/quant"),
        help="HolyC file or directory to scan",
    )
    parser.add_argument(
        "--block-file",
        type=Path,
        action="append",
        default=[],
        help="Raw block stream to audit with --format",
    )
    parser.add_argument(
        "--q4-block-file",
        type=Path,
        action="append",
        default=[],
        help="Raw Q4_0 block stream to audit; may be mixed with --q8-block-file",
    )
    parser.add_argument(
        "--q8-block-file",
        type=Path,
        action="append",
        default=[],
        help="Raw Q8_0 block stream to audit; may be mixed with --q4-block-file",
    )
    parser.add_argument(
        "--format",
        choices=("q4_0", "q8_0"),
        default="q4_0",
        help="Raw block format",
    )
    parser.add_argument(
        "--expect-blocks",
        type=int,
        help="Require each block file to contain this block count",
    )
    parser.add_argument(
        "--expect-elements",
        type=int,
        help="Require each block file to have capacity for this element count",
    )
    parser.add_argument(
        "--max-abs-scale-q16",
        type=int,
        help="Fail any finite fp16 scale whose rounded Q16 magnitude exceeds this limit",
    )
    parser.add_argument(
        "--min-used-quant-values",
        type=int,
        help="Fail any block file whose quant payload uses fewer distinct values than this",
    )
    parser.add_argument(
        "--max-saturation-pct",
        type=float,
        help="Fail any block file whose min/max quant values exceed this percentage of payload entries",
    )
    parser.add_argument(
        "--min-block-used-quant-values",
        type=int,
        help="Fail any individual block whose payload uses fewer distinct quant values than this",
    )
    parser.add_argument(
        "--max-block-saturation-pct",
        type=float,
        help="Fail any individual block whose min/max quant values exceed this percentage of payload entries",
    )
    parser.add_argument(
        "--fail-zero-scale-nonzero-blocks",
        action="store_true",
        help="Fail block files containing finite scales that round to Q16 zero while retaining nonzero quant payload entries",
    )
    parser.add_argument(
        "--fail-zero-scales",
        action="store_true",
        help="Fail any raw block whose fp16 scale field is exactly zero",
    )
    parser.add_argument(
        "--fail-subnormal-scales",
        action="store_true",
        help="Fail any raw block whose fp16 scale field is subnormal",
    )
    parser.add_argument("--allow-inf-nan-scale", action="store_true", help="Do not fail on fp16 inf/nan scales")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    parser.add_argument("--markdown", type=Path, help="Write Markdown report to this path")
    parser.add_argument("--csv", type=Path, help="Write CSV findings report to this path")
    parser.add_argument("--junit", type=Path, help="Write JUnit XML audit report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source_audit = audit_sources(args.source_root)
    block_specs = (
        [(path, args.format) for path in args.block_file]
        + [(path, "q4_0") for path in args.q4_block_file]
        + [(path, "q8_0") for path in args.q8_block_file]
    )
    block_audits = [
        audit_blocks(
            path,
            quant_format,
            args.allow_inf_nan_scale,
            args.expect_blocks,
            args.expect_elements,
            args.max_abs_scale_q16,
            args.min_used_quant_values,
            args.max_saturation_pct,
            args.min_block_used_quant_values,
            args.max_block_saturation_pct,
            args.fail_zero_scale_nonzero_blocks,
            args.fail_zero_scales,
            args.fail_subnormal_scales,
        )
        for path, quant_format in block_specs
    ]
    failed = bool(source_audit.findings) or any(audit.findings for audit in block_audits)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_root": str(args.source_root),
        "source_audit": {
            "files_scanned": source_audit.files_scanned,
            "findings": [asdict(finding) for finding in source_audit.findings],
        },
        "block_audits": [asdict(audit) for audit in block_audits],
        "status": "fail" if failed else "pass",
    }

    encoded = encode_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, args.csv)

    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        write_junit(report, args.junit)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
