#!/usr/bin/env python3
"""Audit paired Q4_0/Q8_0 manifest entries for size-equation drift.

This host-side validation reads local quant manifests only. It verifies that
Q4_0/Q8_0 byte counts, block counts, and element capacity stay internally
consistent for each paired tensor/build identity. It never launches QEMU and
never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


BLOCK_BYTES = {"q4_0": 18, "q8_0": 34}
BLOCK_ELEMENTS = 32
DEFAULT_REQUIRED_FORMATS = ("q4_0", "q8_0")
QUANT_SUFFIX_RE = re.compile(r"([._-])q[48]_0$", re.IGNORECASE)


@dataclass(frozen=True)
class SizeEntry:
    key: str
    tensor: str
    model: str
    source: str
    format: str
    blocks: int | None
    expected_blocks: int | None
    bytes: int | None
    expected_bytes: int | None
    elements: int | None
    element_capacity: int | None
    padding_elements: int | None
    status: str


@dataclass(frozen=True)
class PairRow:
    key: str
    tensor: str
    model: str
    source: str
    formats: str
    q4_0_blocks: int | None
    q8_0_blocks: int | None
    q4_0_bytes: int | None
    q8_0_bytes: int | None
    element_counts: str
    pair_byte_ratio: float | None
    status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    key: str
    format: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def int_field(entry: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = int_or_none(entry.get(name))
        if value is not None:
            return value
    return None


def quant_format(entry: dict[str, Any]) -> str:
    return clean(entry.get("format") or entry.get("quant_format")).casefold()


def manifest_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("artifacts", payload.get("blocks"))
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return [payload]


def tensor_name(entry: dict[str, Any]) -> str:
    value = clean(entry.get("tensor") or entry.get("tensor_name") or entry.get("name"))
    if value:
        return QUANT_SUFFIX_RE.sub("", value)
    path = clean(entry.get("path") or entry.get("file"))
    if path:
        return QUANT_SUFFIX_RE.sub("", Path(path).stem)
    return ""


def entry_model(entry: dict[str, Any], manifest: dict[str, Any]) -> str:
    return clean(entry.get("model") or manifest.get("model") or manifest.get("model_name"))


def entry_source(entry: dict[str, Any]) -> str:
    return clean(entry.get("source") or entry.get("build") or entry.get("checkpoint") or entry.get("path") or entry.get("file"))


def entry_identity(entry: dict[str, Any], manifest: dict[str, Any], key_fields: list[str]) -> tuple[str, dict[str, str]]:
    values = {
        "tensor": tensor_name(entry),
        "model": entry_model(entry, manifest),
        "source": entry_source(entry),
    }
    key = "|".join(values.get(field, "") or "-" for field in key_fields)
    return key, values


def sorted_csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in sorted({value for value in values if value not in (None, "")}))


def add_finding(findings: list[Finding], kind: str, key: str, fmt: str, detail: str) -> None:
    findings.append(Finding("error", kind, key, fmt, detail))


def expected_blocks_for_elements(elements: int | None) -> int | None:
    if elements is None:
        return None
    return math.ceil(elements / BLOCK_ELEMENTS)


def audit_entry(
    entry: dict[str, Any],
    manifest: dict[str, Any],
    key_fields: list[str],
    findings: list[Finding],
) -> SizeEntry:
    key, identity = entry_identity(entry, manifest, key_fields)
    fmt = quant_format(entry)
    blocks = int_field(entry, "blocks", "block_count")
    byte_count = int_field(entry, "bytes", "byte_count")
    elements = int_field(entry, "elements", "element_count")
    expected_blocks = expected_blocks_for_elements(elements)
    block_count = blocks if blocks is not None else expected_blocks
    expected_bytes = block_count * BLOCK_BYTES[fmt] if block_count is not None and fmt in BLOCK_BYTES else None
    element_capacity = block_count * BLOCK_ELEMENTS if block_count is not None else None
    padding_elements = element_capacity - elements if element_capacity is not None and elements is not None else None
    start_findings = len(findings)

    if not identity.get("tensor"):
        add_finding(findings, "missing_tensor_identity", key, fmt, "entry has no tensor/name/path identity")
    if fmt not in BLOCK_BYTES:
        add_finding(findings, "unsupported_quant_format", key, fmt, f"unsupported quant format {fmt!r}")
    if blocks is not None and blocks < 0:
        add_finding(findings, "negative_block_count", key, fmt, f"blocks={blocks}")
    if byte_count is not None and byte_count < 0:
        add_finding(findings, "negative_byte_count", key, fmt, f"bytes={byte_count}")
    if elements is not None and elements < 0:
        add_finding(findings, "negative_element_count", key, fmt, f"elements={elements}")
    if blocks is None and elements is None:
        add_finding(findings, "missing_size_basis", key, fmt, "entry needs blocks/block_count or elements/element_count")
    if fmt in BLOCK_BYTES and expected_bytes is not None and byte_count is not None and byte_count != expected_bytes:
        add_finding(findings, "byte_count_mismatch", key, fmt, f"expected {expected_bytes} bytes, got {byte_count}")
    if blocks is not None and expected_blocks is not None and blocks != expected_blocks:
        add_finding(findings, "block_count_mismatch", key, fmt, f"expected {expected_blocks} blocks for {elements} elements, got {blocks}")
    if padding_elements is not None and not (0 <= padding_elements < BLOCK_ELEMENTS):
        add_finding(
            findings,
            "element_capacity_mismatch",
            key,
            fmt,
            f"capacity {element_capacity} does not cover {elements} elements with <{BLOCK_ELEMENTS} padding",
        )

    return SizeEntry(
        key=key,
        tensor=identity.get("tensor", ""),
        model=identity.get("model", ""),
        source=identity.get("source", ""),
        format=fmt,
        blocks=blocks,
        expected_blocks=expected_blocks,
        bytes=byte_count,
        expected_bytes=expected_bytes,
        elements=elements,
        element_capacity=element_capacity,
        padding_elements=padding_elements,
        status="fail" if len(findings) != start_findings else "pass",
    )


def audit_pairs(entries: list[SizeEntry], required_formats: tuple[str, ...], findings: list[Finding]) -> list[PairRow]:
    grouped: dict[str, list[SizeEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.key, []).append(entry)

    pairs: list[PairRow] = []
    for key, group in sorted(grouped.items()):
        by_format: dict[str, list[SizeEntry]] = {}
        for entry in group:
            by_format.setdefault(entry.format, []).append(entry)

        start_findings = len(findings)
        for fmt in required_formats:
            if fmt not in by_format:
                add_finding(findings, "missing_quant_pair", key, fmt, f"missing required format {fmt}")
            elif len(by_format[fmt]) > 1:
                add_finding(findings, "duplicate_quant_format", key, fmt, f"found {len(by_format[fmt])} entries")

        q4 = by_format.get("q4_0", [None])[0]
        q8 = by_format.get("q8_0", [None])[0]
        q4_blocks = q4.blocks if q4 and q4.blocks is not None else (q4.expected_blocks if q4 else None)
        q8_blocks = q8.blocks if q8 and q8.blocks is not None else (q8.expected_blocks if q8 else None)
        q4_bytes = q4.bytes if q4 else None
        q8_bytes = q8.bytes if q8 else None
        q4_expected_bytes = q4_blocks * BLOCK_BYTES["q4_0"] if q4_blocks is not None else None
        q8_expected_bytes = q8_blocks * BLOCK_BYTES["q8_0"] if q8_blocks is not None else None
        element_counts = {entry.elements for entry in group if entry.elements is not None}

        if len(element_counts) > 1:
            add_finding(findings, "pair_element_count_mismatch", key, "", f"paired element counts differ: {sorted_csv(element_counts)}")
        if q4_blocks is not None and q8_blocks is not None and q4_blocks != q8_blocks:
            add_finding(findings, "pair_block_count_mismatch", key, "", f"Q4_0 blocks={q4_blocks}, Q8_0 blocks={q8_blocks}")
        if q4_bytes is not None and q4_expected_bytes is not None and q4_bytes != q4_expected_bytes:
            add_finding(findings, "pair_q4_byte_count_mismatch", key, "q4_0", f"expected {q4_expected_bytes}, got {q4_bytes}")
        if q8_bytes is not None and q8_expected_bytes is not None and q8_bytes != q8_expected_bytes:
            add_finding(findings, "pair_q8_byte_count_mismatch", key, "q8_0", f"expected {q8_expected_bytes}, got {q8_bytes}")

        identity = group[0]
        pairs.append(
            PairRow(
                key=key,
                tensor=identity.tensor,
                model=identity.model,
                source=identity.source,
                formats=sorted_csv(entry.format for entry in group),
                q4_0_blocks=q4_blocks,
                q8_0_blocks=q8_blocks,
                q4_0_bytes=q4_bytes,
                q8_0_bytes=q8_bytes,
                element_counts=sorted_csv(element_counts),
                pair_byte_ratio=round(q8_bytes / q4_bytes, 6) if q4_bytes and q8_bytes else None,
                status="fail" if len(findings) != start_findings else "pass",
            )
        )
    return pairs


def audit_manifest(
    manifest_path: Path,
    *,
    required_formats: tuple[str, ...],
    key_fields: list[str],
    min_pairs: int,
) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{manifest_path}: manifest root must be a JSON object")
    findings: list[Finding] = []
    entries = [audit_entry(entry, payload, key_fields, findings) for entry in manifest_entries(payload)]
    pairs = audit_pairs(entries, required_formats, findings)
    complete_pairs = sum(1 for pair in pairs if pair.status == "pass" and all(fmt in pair.formats.split(",") for fmt in required_formats))
    if not entries:
        add_finding(findings, "empty_manifest", "", "", "manifest has no quant artifacts")
    if complete_pairs < min_pairs:
        add_finding(findings, "min_pairs", "", "", f"found {complete_pairs} complete pair(s), expected at least {min_pairs}")

    q4_total = sum(entry.bytes or 0 for entry in entries if entry.format == "q4_0")
    q8_total = sum(entry.bytes or 0 for entry in entries if entry.format == "q8_0")
    return {
        "tool": "quant_pair_size_audit",
        "generated_at": iso_now(),
        "manifest": str(manifest_path),
        "required_formats": list(required_formats),
        "key_fields": key_fields,
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "summary": {
            "entries": len(entries),
            "pair_keys": len(pairs),
            "complete_pairs": complete_pairs,
            "q4_0_bytes_total": q4_total,
            "q8_0_bytes_total": q8_total,
            "findings": len(findings),
        },
        "entries": [asdict(entry) for entry in entries],
        "pairs": [asdict(pair) for pair in pairs],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / f"{stem}.csv",
        report["pairs"],
        list(PairRow.__dataclass_fields__),
    )
    write_csv(
        args.output_dir / f"{stem}_entries.csv",
        report["entries"],
        list(SizeEntry.__dataclass_fields__),
    )
    write_csv(args.output_dir / f"{stem}_findings.csv", report["findings"], list(Finding.__dataclass_fields__))

    lines = [
        "# Quant Pair Size Audit",
        "",
        f"- Status: {report['status'].upper()}",
        f"- Entries: {report['summary']['entries']}",
        f"- Pair keys: {report['summary']['pair_keys']}",
        f"- Complete pairs: {report['summary']['complete_pairs']}",
        f"- Q4_0 bytes: {report['summary']['q4_0_bytes_total']}",
        f"- Q8_0 bytes: {report['summary']['q8_0_bytes_total']}",
        f"- Findings: {report['summary']['findings']}",
        "",
        "## Findings",
    ]
    if report["findings"]:
        lines.extend(f"- {item['kind']}: {item['key']} {item['format']} {item['detail']}".strip() for item in report["findings"])
    else:
        lines.append("- none")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [item for item in report["findings"] if item["severity"] == "error"]
    suite = ET.Element("testsuite", {"name": "quant_pair_size_audit", "tests": "1", "failures": "1" if failures else "0"})
    case = ET.SubElement(suite, "testcase", {"name": "quant_pair_size_equations"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} quant pair size finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="quant_pair_size_audit_latest")
    parser.add_argument("--required-format", action="append", dest="required_formats", default=[], help="Required quant format; repeatable")
    parser.add_argument("--key-field", action="append", dest="key_fields", default=[], choices=["tensor", "model", "source"], help="Pairing identity field; repeatable")
    parser.add_argument("--min-pairs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required_formats = tuple(fmt.casefold() for fmt in (args.required_formats or DEFAULT_REQUIRED_FORMATS))
    if any(fmt not in BLOCK_BYTES for fmt in required_formats):
        raise SystemExit(f"--required-format must be one of {sorted(BLOCK_BYTES)}")
    if args.min_pairs < 0:
        raise SystemExit("--min-pairs must be >= 0")
    report = audit_manifest(
        args.manifest,
        required_formats=required_formats,
        key_fields=args.key_fields or ["tensor", "model"],
        min_pairs=args.min_pairs,
    )
    write_outputs(report, args)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
