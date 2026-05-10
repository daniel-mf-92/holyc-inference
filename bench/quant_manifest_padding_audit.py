#!/usr/bin/env python3
"""Audit quant manifests for nonzero tail padding payloads.

This host-side validation reads local Q4_0/Q8_0 manifests and raw block files
only. It verifies that manifest element counts match block capacity and that
tail padding quant entries are zeroed, so integer-only kernels do not depend on
garbage payload beyond the logical tensor length.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import quant_audit


SUPPORTED_FORMATS = {"q4_0", "q8_0"}


@dataclass(frozen=True)
class PaddingRecord:
    manifest: str
    entry: str
    path: str
    format: str
    elements: int | None
    blocks: int | None
    actual_blocks: int | None
    element_capacity: int | None
    padding_element_count: int | None
    padding_nonzero_count: int | None
    status: str


@dataclass(frozen=True)
class Finding:
    manifest: str
    entry: str
    path: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def int_field(entry: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = int_or_none(entry.get(name))
        if value is not None:
            return value
    return None


def manifest_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("artifacts", payload.get("blocks"))
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return [payload]


def resolve_path(root: Path, value: Any) -> Path | None:
    text = clean(value)
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else (root / path).resolve()


def entry_name(entry: dict[str, Any], index: int) -> str:
    return clean(entry.get("name") or entry.get("tensor") or entry.get("tensor_name") or entry.get("path")) or f"entry[{index}]"


def audit_entry(
    manifest_path: Path,
    root: Path,
    entry: dict[str, Any],
    index: int,
    args: argparse.Namespace,
) -> tuple[PaddingRecord, list[Finding]]:
    name = entry_name(entry, index)
    fmt = clean(entry.get("format") or entry.get("quant_format")).casefold()
    elements = int_field(entry, "elements", "element_count")
    blocks = int_field(entry, "blocks", "block_count")
    path = resolve_path(root, entry.get("path") or entry.get("file"))
    findings: list[Finding] = []

    if fmt not in SUPPORTED_FORMATS:
        findings.append(Finding(str(manifest_path), name, str(path or ""), "error", "unsupported_format", f"unsupported quant format {fmt!r}"))
    if path is None:
        findings.append(Finding(str(manifest_path), name, "", "error", "missing_path", "entry path/file is required"))
    elif not path.exists():
        findings.append(Finding(str(manifest_path), name, str(path), "error", "missing_artifact", "raw quant block file does not exist"))
    if elements is None:
        findings.append(Finding(str(manifest_path), name, str(path or ""), "error", "missing_elements", "elements/element_count is required for padding validation"))
    elif elements < 0:
        findings.append(Finding(str(manifest_path), name, str(path or ""), "error", "negative_elements", f"elements={elements}"))
    if blocks is not None and blocks < 0:
        findings.append(Finding(str(manifest_path), name, str(path or ""), "error", "negative_blocks", f"blocks={blocks}"))

    if findings or path is None or fmt not in SUPPORTED_FORMATS:
        return (
            PaddingRecord(str(manifest_path), name, str(path or ""), fmt, elements, blocks, None, None, None, None, "fail"),
            findings,
        )

    try:
        block_audit = quant_audit.audit_blocks(
            path,
            fmt,
            allow_inf_nan_scale=args.allow_inf_nan_scale,
            expect_blocks=blocks,
            expect_elements=elements,
            fail_nonzero_padding_quants=True,
        )
    except OSError as exc:
        finding = Finding(str(manifest_path), name, str(path), "error", "read_error", str(exc))
        return (
            PaddingRecord(str(manifest_path), name, str(path), fmt, elements, blocks, None, None, None, None, "fail"),
            [finding],
        )

    for detail in block_audit.findings:
        kind = "nonzero_padding_quant" if "nonzero padding quant" in detail else "block_audit_finding"
        severity = "warning" if args.allow_nonzero_padding and kind == "nonzero_padding_quant" else "error"
        findings.append(Finding(str(manifest_path), name, str(path), severity, kind, detail))

    return (
        PaddingRecord(
            manifest=str(manifest_path),
            entry=name,
            path=str(path),
            format=fmt,
            elements=elements,
            blocks=blocks,
            actual_blocks=block_audit.block_count,
            element_capacity=block_audit.element_capacity,
            padding_element_count=block_audit.padding_element_count,
            padding_nonzero_count=block_audit.padding_nonzero_count,
            status="fail" if any(finding.severity == "error" for finding in findings) else "pass",
        ),
        findings,
    )


def audit_manifest(manifest_path: Path, args: argparse.Namespace) -> tuple[list[PaddingRecord], list[Finding]]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [Finding(str(manifest_path), "", "", "error", "manifest_read_error", str(exc))]
    if not isinstance(payload, dict):
        return [], [Finding(str(manifest_path), "", "", "error", "manifest_type", "manifest root must be a JSON object")]

    root = args.root or manifest_path.parent
    records: list[PaddingRecord] = []
    findings: list[Finding] = []
    entries = manifest_entries(payload)
    if not entries:
        findings.append(Finding(str(manifest_path), "", "", "error", "empty_manifest", "manifest has no entries"))
    for index, entry in enumerate(entries):
        record, entry_findings = audit_entry(manifest_path, root, entry, index, args)
        records.append(record)
        findings.extend(entry_findings)
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(records: list[PaddingRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    summary = {
        "entries": len(records),
        "padding_elements": sum(record.padding_element_count or 0 for record in records),
        "nonzero_padding_elements": sum(record.padding_nonzero_count or 0 for record in records),
        "findings": len(findings),
        "status": status,
    }
    payload = {
        "tool": "quant_manifest_padding_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary,
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(record) for record in records], list(PaddingRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    lines = [
        "# Quant Manifest Padding Audit",
        "",
        f"Status: {status}",
        f"Entries: {summary['entries']}",
        f"Padding elements: {summary['padding_elements']}",
        f"Nonzero padding elements: {summary['nonzero_padding_elements']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding.severity}: {finding.kind} `{finding.entry}` {finding.detail}")
    else:
        lines.append("No quant manifest padding findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite = ET.Element("testsuite", name="quant_manifest_padding_audit", tests=str(max(1, len(records))), failures=str(sum(1 for finding in findings if finding.severity == "error")))
    if not records:
        case = ET.SubElement(suite, "testcase", name="inputs")
        failure = ET.SubElement(case, "failure", message="no records")
        failure.text = "no manifest records were audited"
    for record in records:
        case = ET.SubElement(suite, "testcase", name=record.entry, classname="quant_manifest_padding_audit")
        record_errors = [finding for finding in findings if finding.entry == record.entry and finding.severity == "error"]
        if record_errors:
            failure = ET.SubElement(case, "failure", message=f"{len(record_errors)} padding finding(s)")
            failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in record_errors)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, help="root for relative artifact paths; defaults to each manifest directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="quant_manifest_padding_audit_latest")
    parser.add_argument("--allow-inf-nan-scale", action="store_true")
    parser.add_argument("--allow-nonzero-padding", action="store_true", help="downgrade nonzero padding quant entries to warnings")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[PaddingRecord] = []
    findings: list[Finding] = []
    for manifest in args.manifests:
        manifest_records, manifest_findings = audit_manifest(manifest, args)
        records.extend(manifest_records)
        findings.extend(manifest_findings)
    write_outputs(records, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
