#!/usr/bin/env python3
"""Audit quant manifests for paired Q4_0/Q8_0 tensor coverage.

This host-side validation reads local quant manifests only. It checks that each
tensor/build identity has the required quant formats, and that paired entries
agree on element counts when that metadata is present. It never launches QEMU or
touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED_FORMATS = ("q4_0", "q8_0")
QUANT_SUFFIX_RE = re.compile(r"([._-])q[48]_0$", re.IGNORECASE)


@dataclass(frozen=True)
class PairRow:
    key: str
    tensor: str
    model: str
    source: str
    formats: str
    artifact_count: int
    q4_0_count: int
    q8_0_count: int
    element_counts: str
    block_counts: str
    byte_counts: str
    status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    key: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def quant_format(entry: dict[str, Any]) -> str:
    return clean(entry.get("format") or entry.get("quant_format")).casefold()


def manifest_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("artifacts", payload.get("blocks"))
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, dict)]
    return [payload]


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def int_field(entry: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = int_or_none(entry.get(name))
        if value is not None:
            return value
    return None


def tensor_name(entry: dict[str, Any]) -> str:
    value = clean(entry.get("tensor") or entry.get("tensor_name") or entry.get("name"))
    if value:
        return QUANT_SUFFIX_RE.sub("", value)
    path = clean(entry.get("path") or entry.get("file"))
    if path:
        return QUANT_SUFFIX_RE.sub("", Path(path).stem)
    return ""


def entry_source(entry: dict[str, Any]) -> str:
    return clean(entry.get("source") or entry.get("build") or entry.get("checkpoint") or entry.get("path") or entry.get("file"))


def entry_model(entry: dict[str, Any], manifest: dict[str, Any]) -> str:
    return clean(entry.get("model") or manifest.get("model") or manifest.get("model_name"))


def pair_key(entry: dict[str, Any], manifest: dict[str, Any], key_fields: list[str]) -> tuple[str, dict[str, str]]:
    values = {
        "tensor": tensor_name(entry),
        "model": entry_model(entry, manifest),
        "source": entry_source(entry),
    }
    key = "|".join(values.get(field, "") or "-" for field in key_fields)
    return key, values


def sorted_csv(values: Iterable[Any]) -> str:
    return ",".join(str(value) for value in sorted({value for value in values if value not in (None, "")}))


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
    grouped: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, dict[str, str]] = {}
    for index, entry in enumerate(manifest_entries(payload), 1):
        fmt = quant_format(entry)
        key, identity = pair_key(entry, payload, key_fields)
        identities[key] = identity
        grouped.setdefault(key, []).append(entry)
        if not identity.get("tensor"):
            findings.append(Finding("error", "missing_tensor_identity", key, f"artifact[{index}] has no tensor/name/path identity"))
        if fmt not in set(required_formats):
            findings.append(Finding("error", "unsupported_quant_format", key, f"artifact[{index}] uses format {fmt!r}"))

    pair_rows: list[PairRow] = []
    complete_pairs = 0
    for key, entries in sorted(grouped.items()):
        formats = [quant_format(entry) for entry in entries]
        missing = [fmt for fmt in required_formats if fmt not in formats]
        duplicate_formats = [fmt for fmt in required_formats if formats.count(fmt) > 1]
        element_counts = [int_field(entry, "elements", "element_count") for entry in entries]
        comparable_element_counts = {value for value in element_counts if value is not None}
        status = "pass"

        if missing:
            status = "fail"
            findings.append(Finding("error", "missing_quant_pair", key, f"missing required format(s): {', '.join(missing)}"))
        if duplicate_formats:
            status = "fail"
            findings.append(Finding("error", "duplicate_quant_format", key, f"duplicate required format(s): {', '.join(duplicate_formats)}"))
        if len(comparable_element_counts) > 1:
            status = "fail"
            findings.append(
                Finding(
                    "error",
                    "element_count_mismatch",
                    key,
                    f"paired element counts differ: {sorted_csv(comparable_element_counts)}",
                )
            )

        if status == "pass" and all(fmt in formats for fmt in required_formats):
            complete_pairs += 1

        identity = identities.get(key, {})
        pair_rows.append(
            PairRow(
                key=key,
                tensor=identity.get("tensor", ""),
                model=identity.get("model", ""),
                source=identity.get("source", ""),
                formats=sorted_csv(formats),
                artifact_count=len(entries),
                q4_0_count=formats.count("q4_0"),
                q8_0_count=formats.count("q8_0"),
                element_counts=sorted_csv(comparable_element_counts),
                block_counts=sorted_csv(int_field(entry, "blocks", "block_count") for entry in entries),
                byte_counts=sorted_csv(int_field(entry, "bytes", "byte_count") for entry in entries),
                status=status,
            )
        )

    if not pair_rows:
        findings.append(Finding("error", "empty_manifest", "", "manifest has no quant artifacts"))
    if complete_pairs < min_pairs:
        findings.append(Finding("error", "min_pairs", "", f"found {complete_pairs} complete pair(s), expected at least {min_pairs}"))

    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    return {
        "tool": "quant_pair_manifest_audit",
        "generated_at": iso_now(),
        "manifest": str(manifest_path),
        "required_formats": list(required_formats),
        "key_fields": key_fields,
        "status": status,
        "summary": {
            "pair_keys": len(pair_rows),
            "complete_pairs": complete_pairs,
            "artifact_count": sum(row.artifact_count for row in pair_rows),
            "findings": len(findings),
        },
        "pairs": [asdict(row) for row in pair_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / f"{stem}.csv",
        report["pairs"],
        ["key", "tensor", "model", "source", "formats", "artifact_count", "q4_0_count", "q8_0_count", "element_counts", "block_counts", "byte_counts", "status"],
    )
    write_csv(args.output_dir / f"{stem}_findings.csv", report["findings"], ["severity", "kind", "key", "detail"])

    lines = [
        "# Quant Pair Manifest Audit",
        "",
        f"- Status: {report['status'].upper()}",
        f"- Pair keys: {report['summary']['pair_keys']}",
        f"- Complete pairs: {report['summary']['complete_pairs']}",
        f"- Findings: {report['summary']['findings']}",
        "",
        "## Findings",
    ]
    if report["findings"]:
        lines.extend(f"- {item['kind']}: {item['detail']}" for item in report["findings"])
    else:
        lines.append("- none")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = [item for item in report["findings"] if item["severity"] == "error"]
    suite = ET.Element("testsuite", {"name": "quant_pair_manifest_audit", "tests": "1", "failures": "1" if failures else "0"})
    case = ET.SubElement(suite, "testcase", {"name": "quant_pairing"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} quant pairing finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="quant_pair_manifest_audit_latest")
    parser.add_argument("--required-format", action="append", dest="required_formats", default=[], help="Required quant format; repeatable")
    parser.add_argument("--key-field", action="append", dest="key_fields", default=[], choices=["tensor", "model", "source"], help="Pairing identity field; repeatable")
    parser.add_argument("--min-pairs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required_formats = tuple(fmt.casefold() for fmt in (args.required_formats or DEFAULT_REQUIRED_FORMATS))
    key_fields = args.key_fields or ["tensor", "model"]
    if args.min_pairs < 0:
        raise SystemExit("--min-pairs must be >= 0")
    report = audit_manifest(args.manifest, required_formats=required_formats, key_fields=key_fields, min_pairs=args.min_pairs)
    write_outputs(report, args)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
