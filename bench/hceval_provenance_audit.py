#!/usr/bin/env python3
"""Audit provenance coverage in packed HCEval dataset bundles.

This host-side tool scans `.hceval` binaries produced by `dataset_pack.py`,
verifies packed records keep non-empty provenance strings, and optionally
requires companion manifests. It never launches QEMU, touches guest code, or
uses networking.
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

import hceval_inspect


@dataclass(frozen=True)
class ProvenanceRow:
    input: str
    manifest: str
    dataset: str
    split: str
    record_count: int
    records_with_provenance: int
    missing_provenance_count: int
    distinct_provenance_count: int
    min_provenance_bytes: int
    mean_provenance_bytes: float
    max_provenance_bytes: int
    payload_sha256: str
    source_sha256: str
    status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    input: str
    record_id: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def discover_inputs(inputs: list[Path], glob_pattern: str) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for input_path in inputs:
        candidates = sorted(input_path.glob(glob_pattern)) if input_path.is_dir() else [input_path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(candidate)
                seen.add(resolved)
    return discovered


def manifest_for(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


def pct(count: int, total: int) -> float | None:
    return count / total * 100.0 if total else None


def inspect_one(path: Path, args: argparse.Namespace) -> tuple[ProvenanceRow, list[Finding]]:
    findings: list[Finding] = []
    manifest_path = manifest_for(path, args.manifest_suffix)
    manifest = manifest_path if manifest_path.exists() else None
    if args.require_manifest and manifest is None:
        findings.append(
            Finding("error", "missing_manifest", str(path), "", f"missing companion manifest {manifest_path}")
        )

    try:
        dataset = hceval_inspect.parse_hceval(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "read_error", str(path), "", str(exc)))
        return (
            ProvenanceRow(
                input=str(path),
                manifest=str(manifest) if manifest else "",
                dataset="",
                split="",
                record_count=0,
                records_with_provenance=0,
                missing_provenance_count=0,
                distinct_provenance_count=0,
                min_provenance_bytes=0,
                mean_provenance_bytes=0.0,
                max_provenance_bytes=0,
                payload_sha256="",
                source_sha256="",
                status="fail",
            ),
            findings,
        )

    dataset_name = str(dataset.metadata.get("dataset") or "")
    split = str(dataset.metadata.get("split") or "")
    if not dataset_name:
        findings.append(Finding("error", "missing_dataset", str(path), "", "metadata.dataset is empty"))
    if not split:
        findings.append(Finding("error", "missing_split", str(path), "", "metadata.split is empty"))

    provenance_values = [record.provenance.strip() for record in dataset.records]
    provenance_bytes = [len(value.encode("utf-8")) for value in provenance_values]
    for record, value, size in zip(dataset.records, provenance_values, provenance_bytes, strict=True):
        if not value:
            findings.append(Finding("error", "missing_provenance", str(path), record.record_id, "record provenance is empty"))
        if args.max_provenance_bytes is not None and size > args.max_provenance_bytes:
            findings.append(
                Finding(
                    "error",
                    "provenance_too_large",
                    str(path),
                    record.record_id,
                    f"provenance is {size} bytes, limit is {args.max_provenance_bytes}",
                )
            )

    non_empty = [value for value in provenance_values if value]
    coverage = pct(len(non_empty), len(dataset.records))
    if coverage is None or coverage < args.min_provenance_coverage_pct:
        findings.append(
            Finding(
                "error",
                "provenance_coverage",
                str(path),
                "",
                f"coverage_pct={coverage} below threshold {args.min_provenance_coverage_pct}",
            )
        )
    distinct_count = len(set(non_empty))
    if distinct_count < args.min_distinct_provenance:
        findings.append(
            Finding(
                "error",
                "distinct_provenance",
                str(path),
                "",
                f"distinct provenance count {distinct_count} below threshold {args.min_distinct_provenance}",
            )
        )

    return (
        ProvenanceRow(
            input=str(path),
            manifest=str(manifest) if manifest else "",
            dataset=dataset_name,
            split=split,
            record_count=len(dataset.records),
            records_with_provenance=len(non_empty),
            missing_provenance_count=len(dataset.records) - len(non_empty),
            distinct_provenance_count=distinct_count,
            min_provenance_bytes=min(provenance_bytes) if provenance_bytes else 0,
            mean_provenance_bytes=sum(provenance_bytes) / len(provenance_bytes) if provenance_bytes else 0.0,
            max_provenance_bytes=max(provenance_bytes) if provenance_bytes else 0,
            payload_sha256=dataset.payload_sha256,
            source_sha256=dataset.source_digest,
            status="fail" if findings else "pass",
        ),
        findings,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = discover_inputs(args.input, args.glob)
    rows: list[ProvenanceRow] = []
    findings: list[Finding] = []
    for path in inputs:
        row, row_findings = inspect_one(path, args)
        rows.append(row)
        findings.extend(row_findings)

    if not inputs:
        findings.append(Finding("error", "no_inputs", ",".join(str(path) for path in args.input), "", "no .hceval inputs were discovered"))

    total_records = sum(row.record_count for row in rows)
    total_with_provenance = sum(row.records_with_provenance for row in rows)
    dataset_counts: dict[str, int] = {}
    for row in rows:
        if row.dataset:
            dataset_counts[row.dataset] = dataset_counts.get(row.dataset, 0) + row.record_count

    return {
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "findings": [asdict(finding) for finding in findings],
        "format": "hceval-provenance-audit",
        "generated_at": iso_now(),
        "input_count": len(inputs),
        "inputs": [str(path) for path in inputs],
        "provenance_coverage_pct": pct(total_with_provenance, total_records),
        "record_count": total_records,
        "records_with_provenance": total_with_provenance,
        "rows": [asdict(row) for row in rows],
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Provenance Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {report['input_count']}",
        f"Records: {report['record_count']}",
        f"Provenance coverage: {report['provenance_coverage_pct'] if report['provenance_coverage_pct'] is not None else '-'}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(
            f"- {finding['kind']} `{finding['input']}` {finding['record_id'] or '-'}: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No findings.")

    lines.extend(["", "## Inputs", "", "| Input | Dataset | Split | Records | With provenance | Distinct | Status |", "| --- | --- | --- | ---: | ---: | ---: | --- |"])
    for row in report["rows"]:
        lines.append(
            f"| {row['input']} | {row['dataset'] or '-'} | {row['split'] or '-'} | {row['record_count']} | "
            f"{row['records_with_provenance']} | {row['distinct_provenance_count']} | {row['status']} |"
        )
    if not report["rows"]:
        lines.append("| - | - | - | 0 | 0 | 0 | fail |")
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = list(ProvenanceRow.__dataclass_fields__.keys()) + ["finding_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            output = {field: row.get(field, "") for field in fieldnames}
            output["finding_count"] = sum(1 for finding in report["findings"] if finding["input"] == row["input"])
            writer.writerow(output)


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = list(Finding.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow({field: finding.get(field, "") for field in fieldnames})


def junit_report(report: dict[str, Any]) -> str:
    rows = report.get("rows", [])
    failures = [row for row in rows if row.get("status") != "pass"]
    if not rows and report.get("findings"):
        failures = [{"input": "input", "findings": report["findings"]}]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_provenance_audit",
            "tests": str(max(len(rows), 1)),
            "failures": str(len(failures)),
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    for row in rows or [{"input": "input", "status": report["status"]}]:
        case = ET.SubElement(suite, "testcase", {"classname": "hceval_provenance_audit", "name": str(row.get("input", "input"))})
        if row.get("status") != "pass":
            row_findings = [finding for finding in report["findings"] if finding["input"] == row.get("input")]
            failure = ET.SubElement(
                case,
                "failure",
                {"type": "hceval_provenance_failure", "message": f"{len(row_findings)} finding(s)"},
            )
            failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in row_findings)
    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input .hceval file or directory")
    parser.add_argument("--glob", default="**/*.hceval", help="Directory glob used for --input directories")
    parser.add_argument("--manifest-suffix", default=".manifest.json", help="Companion manifest suffix appended to .hceval path")
    parser.add_argument("--require-manifest", action="store_true", help="Fail inputs without companion manifests")
    parser.add_argument("--min-provenance-coverage-pct", type=float, default=100.0)
    parser.add_argument("--min-distinct-provenance", type=int, default=1)
    parser.add_argument("--max-provenance-bytes", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional CSV row report path")
    parser.add_argument("--findings-csv", type=Path, help="Optional CSV findings report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML report path")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when findings are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        write_csv(report, args.csv)
        print(f"wrote_csv={args.csv}")
    if args.findings_csv:
        write_findings_csv(report, args.findings_csv)
        print(f"wrote_findings_csv={args.findings_csv}")
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        args.junit.write_text(junit_report(report), encoding="utf-8")
        print(f"wrote_junit={args.junit}")

    print(f"status={report['status']}")
    return 1 if args.fail_on_findings and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
