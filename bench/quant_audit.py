#!/usr/bin/env python3
"""Host-side quantization audit for HolyC inference sources and raw blocks.

The audit intentionally runs outside TempleOS. It validates two invariants:

* quant HolyC sources do not introduce float runtime types or math helpers
* raw Q4_0/Q8_0 block streams have valid sizes and sane fp16 scale fields
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


Q4_0_BLOCK_BYTES = 18
Q4_0_PACKED_BYTES = 16
Q8_0_BLOCK_BYTES = 34
Q8_0_PACKED_BYTES = 32

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
    scale_zero_count: int
    scale_subnormal_count: int
    scale_normal_count: int
    scale_inf_nan_count: int
    quant_min: int
    quant_max: int
    quant_zero_count: int
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


def audit_q4_0_blocks(path: Path, allow_inf_nan_scale: bool) -> BlockAudit:
    data = path.read_bytes()
    findings: list[str] = []
    if len(data) % Q4_0_BLOCK_BYTES:
        findings.append(f"size {len(data)} is not a multiple of Q4_0 block size {Q4_0_BLOCK_BYTES}")

    block_count = len(data) // Q4_0_BLOCK_BYTES
    counts = {"zero": 0, "subnormal": 0, "normal": 0, "inf_nan": 0}
    quant_min = math.inf
    quant_max = -math.inf
    quant_zero_count = 0

    for block_index in range(block_count):
        offset = block_index * Q4_0_BLOCK_BYTES
        scale_bits = int.from_bytes(data[offset : offset + 2], "little")
        scale_class = fp16_class(scale_bits)
        counts[scale_class] += 1
        if scale_class == "inf_nan" and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale is inf/nan bits=0x{scale_bits:04x}")

        scale_value = fp16_to_float(scale_bits)
        if math.isnan(scale_value) and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale decoded to nan")

        packed = data[offset + 2 : offset + 2 + Q4_0_PACKED_BYTES]
        for byte in packed:
            for nibble in (byte & 0x0F, (byte >> 4) & 0x0F):
                signed = nibble - 8
                quant_min = min(quant_min, signed)
                quant_max = max(quant_max, signed)
                if signed == 0:
                    quant_zero_count += 1

    if block_count == 0:
        quant_min = 0
        quant_max = 0

    return BlockAudit(
        format="q4_0",
        path=str(path),
        bytes_read=len(data),
        block_count=block_count,
        scale_zero_count=counts["zero"],
        scale_subnormal_count=counts["subnormal"],
        scale_normal_count=counts["normal"],
        scale_inf_nan_count=counts["inf_nan"],
        quant_min=int(quant_min),
        quant_max=int(quant_max),
        quant_zero_count=quant_zero_count,
        findings=findings,
    )


def audit_q8_0_blocks(path: Path, allow_inf_nan_scale: bool) -> BlockAudit:
    data = path.read_bytes()
    findings: list[str] = []
    if len(data) % Q8_0_BLOCK_BYTES:
        findings.append(f"size {len(data)} is not a multiple of Q8_0 block size {Q8_0_BLOCK_BYTES}")

    block_count = len(data) // Q8_0_BLOCK_BYTES
    counts = {"zero": 0, "subnormal": 0, "normal": 0, "inf_nan": 0}
    quant_min = math.inf
    quant_max = -math.inf
    quant_zero_count = 0

    for block_index in range(block_count):
        offset = block_index * Q8_0_BLOCK_BYTES
        scale_bits = int.from_bytes(data[offset : offset + 2], "little")
        scale_class = fp16_class(scale_bits)
        counts[scale_class] += 1
        if scale_class == "inf_nan" and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale is inf/nan bits=0x{scale_bits:04x}")

        scale_value = fp16_to_float(scale_bits)
        if math.isnan(scale_value) and not allow_inf_nan_scale:
            findings.append(f"block {block_index}: fp16 scale decoded to nan")

        qs = data[offset + 2 : offset + 2 + Q8_0_PACKED_BYTES]
        for value in struct.unpack("<32b", qs):
            quant_min = min(quant_min, value)
            quant_max = max(quant_max, value)
            if value == 0:
                quant_zero_count += 1

    if block_count == 0:
        quant_min = 0
        quant_max = 0

    return BlockAudit(
        format="q8_0",
        path=str(path),
        bytes_read=len(data),
        block_count=block_count,
        scale_zero_count=counts["zero"],
        scale_subnormal_count=counts["subnormal"],
        scale_normal_count=counts["normal"],
        scale_inf_nan_count=counts["inf_nan"],
        quant_min=int(quant_min),
        quant_max=int(quant_max),
        quant_zero_count=quant_zero_count,
        findings=findings,
    )


def audit_blocks(path: Path, quant_format: str, allow_inf_nan_scale: bool) -> BlockAudit:
    if quant_format == "q4_0":
        return audit_q4_0_blocks(path, allow_inf_nan_scale)
    if quant_format == "q8_0":
        return audit_q8_0_blocks(path, allow_inf_nan_scale)
    raise ValueError(f"unsupported quant format: {quant_format}")


def encode_report(report: dict) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("src/quant"), help="HolyC file or directory to scan")
    parser.add_argument("--block-file", type=Path, action="append", default=[], help="Raw block stream to audit")
    parser.add_argument("--format", choices=("q4_0", "q8_0"), default="q4_0", help="Raw block format")
    parser.add_argument("--allow-inf-nan-scale", action="store_true", help="Do not fail on fp16 inf/nan scales")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    source_audit = audit_sources(args.source_root)
    block_audits = [audit_blocks(path, args.format, args.allow_inf_nan_scale) for path in args.block_file]
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

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
