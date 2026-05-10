#!/usr/bin/env python3
"""Compare two raw Q4_0/Q8_0 quant block streams.

This host-side tool validates packing determinism across independent packers or
build outputs. It never launches TempleOS or QEMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Q4_0_BLOCK_BYTES = 18
Q4_0_PACKED_BYTES = 16
Q8_0_BLOCK_BYTES = 34
Q8_0_PACKED_BYTES = 32
BLOCK_ELEMENTS = 32


@dataclass(frozen=True)
class BlockMismatch:
    block_index: int
    kind: str
    detail: str


@dataclass(frozen=True)
class CompareResult:
    format: str
    reference_path: str
    candidate_path: str
    timestamp: str
    reference_bytes: int
    candidate_bytes: int
    reference_blocks: int
    candidate_blocks: int
    comparable_blocks: int
    block_count_match: bool
    byte_size_match: bool
    scale_mismatch_count: int
    quant_mismatch_count: int
    quant_mismatch_pct: float
    block_mismatch_count: int
    first_mismatch_block: int | None
    findings: list[str]
    mismatches: list[BlockMismatch]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def block_size(quant_format: str) -> int:
    if quant_format == "q4_0":
        return Q4_0_BLOCK_BYTES
    if quant_format == "q8_0":
        return Q8_0_BLOCK_BYTES
    raise ValueError(f"unsupported quant format: {quant_format}")


def decode_q4_payload(payload: bytes) -> list[int]:
    quants: list[int] = []
    for byte in payload:
        quants.append((byte & 0x0F) - 8)
        quants.append(((byte >> 4) & 0x0F) - 8)
    return quants


def decode_q8_payload(payload: bytes) -> list[int]:
    return [int.from_bytes(bytes([byte]), "little", signed=True) for byte in payload]


def decode_blocks(data: bytes, quant_format: str) -> tuple[list[tuple[int, list[int]]], list[str]]:
    size = block_size(quant_format)
    findings: list[str] = []
    if len(data) % size != 0:
        findings.append(f"{quant_format} byte stream length {len(data)} is not a multiple of {size}")

    blocks: list[tuple[int, list[int]]] = []
    for offset in range(0, len(data) - (len(data) % size), size):
        block = data[offset : offset + size]
        scale_bits = int.from_bytes(block[:2], "little")
        payload = block[2:]
        if quant_format == "q4_0":
            quants = decode_q4_payload(payload)
        else:
            quants = decode_q8_payload(payload)
        blocks.append((scale_bits, quants))
    return blocks, findings


def compare_block_streams(
    reference: Path,
    candidate: Path,
    quant_format: str,
    *,
    allow_mismatches: bool = False,
    max_mismatches: int | None = None,
    max_quant_mismatch_pct: float | None = None,
    max_reported_mismatches: int = 50,
) -> CompareResult:
    reference_data = reference.read_bytes()
    candidate_data = candidate.read_bytes()
    reference_blocks, reference_findings = decode_blocks(reference_data, quant_format)
    candidate_blocks, candidate_findings = decode_blocks(candidate_data, quant_format)

    comparable_blocks = min(len(reference_blocks), len(candidate_blocks))
    findings = [f"reference: {finding}" for finding in reference_findings]
    findings.extend(f"candidate: {finding}" for finding in candidate_findings)

    if len(reference_data) != len(candidate_data):
        findings.append(
            f"byte size mismatch: reference={len(reference_data)} candidate={len(candidate_data)}"
        )
    if len(reference_blocks) != len(candidate_blocks):
        findings.append(
            f"block count mismatch: reference={len(reference_blocks)} candidate={len(candidate_blocks)}"
        )

    scale_mismatch_count = 0
    quant_mismatch_count = 0
    block_mismatch_count = 0
    first_mismatch_block: int | None = None
    mismatches: list[BlockMismatch] = []

    for index in range(comparable_blocks):
        reference_scale, reference_quants = reference_blocks[index]
        candidate_scale, candidate_quants = candidate_blocks[index]
        block_mismatched = False

        if reference_scale != candidate_scale:
            scale_mismatch_count += 1
            block_mismatched = True
            if len(mismatches) < max_reported_mismatches:
                mismatches.append(
                    BlockMismatch(
                        index,
                        "scale",
                        f"reference=0x{reference_scale:04x} candidate=0x{candidate_scale:04x}",
                    )
                )

        for quant_index, (reference_quant, candidate_quant) in enumerate(
            zip(reference_quants, candidate_quants)
        ):
            if reference_quant != candidate_quant:
                quant_mismatch_count += 1
                block_mismatched = True
                if len(mismatches) < max_reported_mismatches:
                    mismatches.append(
                        BlockMismatch(
                            index,
                            "quant",
                            f"element={quant_index} reference={reference_quant} candidate={candidate_quant}",
                        )
                    )

        if block_mismatched:
            block_mismatch_count += 1
            if first_mismatch_block is None:
                first_mismatch_block = index

    comparable_quants = comparable_blocks * BLOCK_ELEMENTS
    quant_mismatch_pct = (
        round((quant_mismatch_count / comparable_quants) * 100.0, 6) if comparable_quants else 0.0
    )
    total_mismatches = scale_mismatch_count + quant_mismatch_count

    if total_mismatches and not allow_mismatches:
        findings.append(
            "block stream mismatch: "
            f"scale_mismatches={scale_mismatch_count} quant_mismatches={quant_mismatch_count}"
        )

    if max_mismatches is not None:
        if total_mismatches > max_mismatches:
            findings.append(f"total mismatches {total_mismatches} exceed limit {max_mismatches}")

    if max_quant_mismatch_pct is not None and quant_mismatch_pct > max_quant_mismatch_pct:
        findings.append(
            f"quant mismatch percentage {quant_mismatch_pct:.6f} exceeds limit {max_quant_mismatch_pct:.6f}"
        )

    return CompareResult(
        format=quant_format,
        reference_path=str(reference),
        candidate_path=str(candidate),
        timestamp=iso_now(),
        reference_bytes=len(reference_data),
        candidate_bytes=len(candidate_data),
        reference_blocks=len(reference_blocks),
        candidate_blocks=len(candidate_blocks),
        comparable_blocks=comparable_blocks,
        block_count_match=len(reference_blocks) == len(candidate_blocks),
        byte_size_match=len(reference_data) == len(candidate_data),
        scale_mismatch_count=scale_mismatch_count,
        quant_mismatch_count=quant_mismatch_count,
        quant_mismatch_pct=quant_mismatch_pct,
        block_mismatch_count=block_mismatch_count,
        first_mismatch_block=first_mismatch_block,
        findings=findings,
        mismatches=mismatches,
    )


def result_to_json(result: CompareResult) -> dict[str, Any]:
    return asdict(result)


def write_json(path: Path, result: CompareResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result_to_json(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, result: CompareResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "format",
                "reference_blocks",
                "candidate_blocks",
                "comparable_blocks",
                "scale_mismatch_count",
                "quant_mismatch_count",
                "quant_mismatch_pct",
                "block_mismatch_count",
                "first_mismatch_block",
                "finding_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "format": result.format,
                "reference_blocks": result.reference_blocks,
                "candidate_blocks": result.candidate_blocks,
                "comparable_blocks": result.comparable_blocks,
                "scale_mismatch_count": result.scale_mismatch_count,
                "quant_mismatch_count": result.quant_mismatch_count,
                "quant_mismatch_pct": f"{result.quant_mismatch_pct:.6f}",
                "block_mismatch_count": result.block_mismatch_count,
                "first_mismatch_block": result.first_mismatch_block,
                "finding_count": len(result.findings),
            }
        )


def write_markdown(path: Path, result: CompareResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = "PASS" if not result.findings else "FAIL"
    lines = [
        "# Quant Block Compare",
        "",
        f"- Status: {status}",
        f"- Format: {result.format}",
        f"- Reference blocks: {result.reference_blocks}",
        f"- Candidate blocks: {result.candidate_blocks}",
        f"- Scale mismatches: {result.scale_mismatch_count}",
        f"- Quant mismatches: {result.quant_mismatch_count} ({result.quant_mismatch_pct:.6f}%)",
        f"- Block mismatches: {result.block_mismatch_count}",
        "",
        "## Findings",
    ]
    lines.extend(f"- {finding}" for finding in result.findings)
    if not result.findings:
        lines.append("- none")
    lines.append("")
    lines.append("## First Mismatches")
    for mismatch in result.mismatches[:10]:
        lines.append(f"- block {mismatch.block_index} {mismatch.kind}: {mismatch.detail}")
    if not result.mismatches:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, result: CompareResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "quant_block_compare",
            "tests": "1",
            "failures": "1" if result.findings else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": f"compare_{result.format}"})
    if result.findings:
        failure = ET.SubElement(case, "failure", {"message": result.findings[0]})
        failure.text = "\n".join(result.findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("q4_0", "q8_0"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--junit", type=Path)
    parser.add_argument(
        "--allow-mismatches",
        action="store_true",
        help="Write mismatch telemetry without failing unless explicit mismatch gates are exceeded",
    )
    parser.add_argument("--max-mismatches", type=int)
    parser.add_argument("--max-quant-mismatch-pct", type=float)
    parser.add_argument("--max-reported-mismatches", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_block_streams(
        args.reference,
        args.candidate,
        args.format,
        allow_mismatches=args.allow_mismatches,
        max_mismatches=args.max_mismatches,
        max_quant_mismatch_pct=args.max_quant_mismatch_pct,
        max_reported_mismatches=args.max_reported_mismatches,
    )

    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result_to_json(result), indent=2, sort_keys=True))
    if args.csv:
        write_csv(args.csv, result)
    if args.markdown:
        write_markdown(args.markdown, result)
    if args.junit:
        write_junit(args.junit, result)

    return 1 if result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
