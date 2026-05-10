#!/usr/bin/env python3
"""Audit packed HCEval metadata canonicalization and header consistency.

This host-side tool reads existing `.hceval` binaries only. It never launches
QEMU, never downloads data, and never touches the TempleOS guest.
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

import dataset_pack
import hceval_inspect


EXPECTED_METADATA_KEYS = ("dataset", "format", "record_count", "split", "version")


@dataclass(frozen=True)
class MetadataRow:
    input: str
    status: str
    dataset: str
    split: str
    format: str
    version: str
    header_records: int
    parsed_records: int
    metadata_record_count: str
    metadata_bytes: int
    canonical_metadata_bytes: int
    source_sha256: str
    payload_sha256: str
    finding_count: int


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    input: str
    field: str
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
            if resolved in seen:
                continue
            discovered.append(candidate)
            seen.add(resolved)
    return discovered


def header_record_count(path: Path) -> int:
    payload = path.read_bytes()
    hceval_inspect.require_span(payload, 0, dataset_pack.HEADER.size, "header")
    _magic, _version, _flags, record_count, _metadata_len, _source_digest = dataset_pack.HEADER.unpack_from(payload, 0)
    return int(record_count)


def metadata_text(path: Path, metadata_bytes: int) -> str:
    payload = path.read_bytes()
    cursor = dataset_pack.HEADER.size
    hceval_inspect.require_span(payload, cursor, metadata_bytes, "metadata")
    return payload[cursor : cursor + metadata_bytes].decode("utf-8")


def add_metadata_findings(path: Path, dataset: hceval_inspect.HCEvalDataset, findings: list[Finding]) -> None:
    metadata = dataset.metadata
    keys = tuple(sorted(metadata.keys()))
    if keys != EXPECTED_METADATA_KEYS:
        findings.append(
            Finding(
                "error",
                "metadata_key_drift",
                str(path),
                "metadata",
                f"metadata keys {list(keys)!r} differ from expected {list(EXPECTED_METADATA_KEYS)!r}",
            )
        )

    if metadata.get("format") != "hceval-mc":
        findings.append(Finding("error", "metadata_format", str(path), "format", "format must be 'hceval-mc'"))
    if metadata.get("version") != dataset_pack.VERSION:
        findings.append(
            Finding("error", "metadata_version", str(path), "version", f"version must be {dataset_pack.VERSION}")
        )
    if not isinstance(metadata.get("dataset"), str) or not metadata.get("dataset"):
        findings.append(Finding("error", "metadata_dataset", str(path), "dataset", "dataset must be a non-empty string"))
    if not isinstance(metadata.get("split"), str) or not metadata.get("split"):
        findings.append(Finding("error", "metadata_split", str(path), "split", "split must be a non-empty string"))
    if metadata.get("record_count") != len(dataset.records):
        findings.append(
            Finding(
                "error",
                "metadata_record_count",
                str(path),
                "record_count",
                f"metadata record_count {metadata.get('record_count')!r} != parsed {len(dataset.records)}",
            )
        )

    canonical = dataset_pack.metadata_bytes(
        str(metadata.get("dataset", "")),
        str(metadata.get("split", "")),
        len(dataset.records),
    )
    actual = metadata_text(path, dataset.metadata_bytes).encode("utf-8")
    if actual != canonical:
        findings.append(
            Finding(
                "error",
                "metadata_not_canonical",
                str(path),
                "metadata_bytes",
                "metadata bytes differ from dataset_pack canonical compact JSON",
            )
        )

    if header_record_count(path) != len(dataset.records):
        findings.append(
            Finding(
                "error",
                "header_record_count",
                str(path),
                "record_count",
                f"header record count {header_record_count(path)} != parsed {len(dataset.records)}",
            )
        )


def audit_input(path: Path) -> tuple[MetadataRow, list[Finding]]:
    findings: list[Finding] = []
    try:
        dataset = hceval_inspect.parse_hceval(path)
        add_metadata_findings(path, dataset, findings)
        metadata = dataset.metadata
        header_records = header_record_count(path)
        canonical_len = len(
            dataset_pack.metadata_bytes(str(metadata.get("dataset", "")), str(metadata.get("split", "")), len(dataset.records))
        )
        return (
            MetadataRow(
                input=str(path),
                status="fail" if findings else "pass",
                dataset=str(metadata.get("dataset", "")),
                split=str(metadata.get("split", "")),
                format=str(metadata.get("format", "")),
                version=str(metadata.get("version", "")),
                header_records=header_records,
                parsed_records=len(dataset.records),
                metadata_record_count=str(metadata.get("record_count", "")),
                metadata_bytes=dataset.metadata_bytes,
                canonical_metadata_bytes=canonical_len,
                source_sha256=dataset.source_digest,
                payload_sha256=dataset.payload_sha256,
                finding_count=len(findings),
            ),
            findings,
        )
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        findings.append(Finding("error", "read_error", str(path), "input", str(exc)))
        return (
            MetadataRow(str(path), "fail", "", "", "", "", 0, 0, "", 0, 0, "", "", len(findings)),
            findings,
        )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = discover_inputs(args.input, args.glob)
    rows: list[MetadataRow] = []
    findings: list[Finding] = []
    if not inputs:
        findings.append(
            Finding(
                "error",
                "no_inputs",
                ",".join(str(path) for path in args.input) or "input",
                "input",
                "no .hceval inputs were discovered",
            )
        )
    for path in inputs:
        row, input_findings = audit_input(path)
        rows.append(row)
        findings.extend(input_findings)

    return {
        "format": "hceval-metadata-audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "input_count": len(inputs),
        "record_count": sum(row.parsed_records for row in rows),
        "finding_count": len(findings),
        "inputs": [str(path) for path in inputs],
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
        "settings": {"glob": args.glob},
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Metadata Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {report['input_count']}",
        f"Records: {report['record_count']}",
        f"Findings: {report['finding_count']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(f"- {finding['kind']} `{finding['input']}`: {finding['detail']}" for finding in report["findings"])
    else:
        lines.append("No findings.")

    lines.extend(["", "## Inputs", "", "| Input | Status | Dataset | Split | Records | Metadata bytes |"])
    lines.append("| --- | --- | --- | --- | ---: | ---: |")
    for row in report["rows"]:
        lines.append(
            f"| {row['input']} | {row['status']} | {row['dataset'] or '-'} | {row['split'] or '-'} | "
            f"{row['parsed_records']} | {row['metadata_bytes']} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = list(MetadataRow.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = list(Finding.__dataclass_fields__.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["findings"])


def junit_report(report: dict[str, Any]) -> str:
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_metadata_audit",
            "tests": str(max(report["input_count"], 1)),
            "failures": str(len(failures)),
            "errors": "0",
            "timestamp": str(report["generated_at"]),
        },
    )
    findings_by_input: dict[str, list[dict[str, Any]]] = {}
    for finding in report["findings"]:
        findings_by_input.setdefault(str(finding["input"]), []).append(finding)
    rows = report["rows"] or [{"input": ",".join(report["inputs"]) or "input"}]
    for row in rows:
        input_name = str(row["input"])
        case = ET.SubElement(suite, "testcase", {"classname": "hceval_metadata_audit", "name": input_name})
        input_findings = findings_by_input.get(input_name, [])
        if input_findings:
            failure = ET.SubElement(
                case,
                "failure",
                {"type": "hceval_metadata_failure", "message": f"{len(input_findings)} finding(s)"},
            )
            failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in input_findings)
    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input .hceval file or directory")
    parser.add_argument("--glob", default="**/*.hceval", help="Directory glob used for --input directories")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional per-input CSV report path")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV report path")
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
