#!/usr/bin/env python3
"""Audit `.hceval` export/repack stability for offline eval artifacts.

The audit parses a packed HolyC-loadable eval dataset, exports records through
the normalized JSONL exporter path, repacks those rows in memory, and compares
source digests, binary digests, and record fingerprints. It is host-side only
and performs no network or QEMU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

import dataset_pack
import hceval_export
import hceval_inspect


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_export_rows(rows: list[dict[str, Any]], dataset: str, split: str) -> list[dataset_pack.EvalRecord]:
    return [dataset_pack.normalize_row(row, index, dataset, split) for index, row in enumerate(rows)]


def append_mismatch(
    findings: list[Finding],
    *,
    kind: str,
    expected: Any,
    actual: Any,
    detail: str,
) -> None:
    if expected != actual:
        findings.append(
            Finding(
                "error",
                kind,
                f"{detail}: expected {json.dumps(expected, sort_keys=True)}, got {json.dumps(actual, sort_keys=True)}",
            )
        )


def build_report(input_path: Path, pack_manifest: Path | None) -> dict[str, Any]:
    findings: list[Finding] = []
    dataset = hceval_inspect.parse_hceval(input_path)
    inspector_findings = hceval_inspect.validate_dataset(dataset, pack_manifest)
    for detail in inspector_findings:
        findings.append(Finding("error", "inspection_finding", detail))

    manifest_metadata = hceval_export.manifest_metadata_by_id(pack_manifest) if pack_manifest else {}
    rows = hceval_export.export_records(dataset, include_hashes=True, manifest_metadata=manifest_metadata)
    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    records = normalize_export_rows(rows, dataset_name, split)
    repacked = dataset_pack.pack_records(records, dataset_name, split)
    repacked_sha256 = hashlib.sha256(repacked).hexdigest()
    repacked_source_sha256 = hashlib.sha256(dataset_pack.canonical_rows(records)).hexdigest()
    original_fingerprints = hceval_inspect.record_fingerprints(dataset)
    repacked_fingerprints = dataset_pack.record_fingerprints(records)
    original_layout = hceval_inspect.binary_layout(dataset)
    repacked_layout = dataset_pack.binary_layout(records, dataset_name, split)

    append_mismatch(
        findings,
        kind="source_digest_mismatch",
        expected=dataset.source_digest,
        actual=repacked_source_sha256,
        detail="exported rows repack to a different source digest",
    )
    append_mismatch(
        findings,
        kind="binary_sha256_mismatch",
        expected=dataset.payload_sha256,
        actual=repacked_sha256,
        detail="exported rows repack to a different binary sha256",
    )
    append_mismatch(
        findings,
        kind="record_fingerprint_mismatch",
        expected=original_fingerprints,
        actual=repacked_fingerprints,
        detail="exported rows repack to different record fingerprints",
    )
    append_mismatch(
        findings,
        kind="binary_layout_mismatch",
        expected=original_layout,
        actual=repacked_layout,
        detail="exported rows repack to a different binary layout",
    )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "binary_sha256": dataset.payload_sha256,
        "dataset": dataset_name,
        "error_count": error_count,
        "exported_record_count": len(rows),
        "findings": [asdict(finding) for finding in findings],
        "format": "hceval-export-roundtrip-audit",
        "generated_at": iso_now(),
        "input": str(input_path),
        "original_binary_layout": original_layout,
        "pack_manifest": str(pack_manifest) if pack_manifest else "",
        "record_count": len(dataset.records),
        "repacked_binary_layout": repacked_layout,
        "repacked_binary_sha256": repacked_sha256,
        "repacked_source_sha256": repacked_source_sha256,
        "source_sha256": dataset.source_digest,
        "split": split,
        "status": "fail" if error_count else "pass",
        "warning_count": warning_count,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Export Roundtrip Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Input: `{report['input']}`",
        f"Pack manifest: `{report['pack_manifest'] or '-'}`",
        f"Dataset: {report['dataset'] or '-'}",
        f"Split: {report['split'] or '-'}",
        f"Records: {report['record_count']}",
        "",
        "## Digest Parity",
        "",
        f"- Source sha256: `{report['source_sha256']}`",
        f"- Repacked source sha256: `{report['repacked_source_sha256']}`",
        f"- Binary sha256: `{report['binary_sha256']}`",
        f"- Repacked binary sha256: `{report['repacked_binary_sha256']}`",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(f"- {finding['severity']} {finding['kind']}: {finding['detail']}" for finding in report["findings"])
    else:
        lines.append("No findings.")
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    rows = [
        ("source_sha256", report["source_sha256"], report["repacked_source_sha256"]),
        ("binary_sha256", report["binary_sha256"], report["repacked_binary_sha256"]),
        ("binary_layout", report["original_binary_layout"], report["repacked_binary_layout"]),
        ("record_count", report["record_count"], report["exported_record_count"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "original", "repacked", "status"])
        writer.writeheader()
        for metric, original, repacked in rows:
            writer.writerow(
                {
                    "metric": metric,
                    "original": json.dumps(original, sort_keys=True) if isinstance(original, dict) else original,
                    "repacked": json.dumps(repacked, sort_keys=True) if isinstance(repacked, dict) else repacked,
                    "status": "pass" if original == repacked else "fail",
                }
            )


def junit_report(report: dict[str, Any]) -> str:
    findings = report.get("findings", [])
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_export_roundtrip_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "hceval_export_roundtrip_audit",
            "name": str(report.get("input", "hceval")),
        },
    )
    if findings:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "hceval_export_roundtrip_failure",
                "message": f"{len(findings)} export roundtrip finding(s)",
            },
        )
        failure.text = "\n".join(f"{row['kind']}: {row['detail']}" for row in findings)
    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .hceval binary")
    parser.add_argument("--pack-manifest", type=Path, help="Optional pack manifest to restore per-record metadata")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report")
    parser.add_argument("--csv", type=Path, help="Optional CSV parity report")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML report")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when findings are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(args.input, args.pack_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_json={args.output}")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        write_csv(report, args.csv)
        print(f"wrote_csv={args.csv}")
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        args.junit.write_text(junit_report(report), encoding="utf-8")
        print(f"wrote_junit={args.junit}")

    if args.fail_on_findings and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
