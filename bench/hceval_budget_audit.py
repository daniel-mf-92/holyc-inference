#!/usr/bin/env python3
"""Audit HolyC eval binary artifacts against suite-level byte budgets.

This host-side tool reads existing `.hceval` files only. It never launches QEMU,
never downloads data, and never touches the TempleOS guest.
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
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import hceval_inspect


DEFAULT_PATTERNS = ("*.hceval",)


@dataclass(frozen=True)
class BudgetRecord:
    source: str
    status: str
    dataset: str
    split: str
    record_count: int
    binary_bytes: int
    metadata_bytes: int
    body_bytes: int
    record_header_bytes: int
    choice_length_prefix_bytes: int
    total_prompt_bytes: int
    total_choice_bytes: int
    total_provenance_bytes: int
    max_prompt_bytes: int
    max_choice_bytes: int
    max_record_payload_bytes: int
    manifest: str
    finding_count: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    field: str
    value: str
    limit: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_input_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            seen: set[Path] = set()
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file():
            yield path


def companion_manifest_candidates(path: Path, suffix: str) -> list[Path]:
    if suffix.startswith("."):
        candidates = [path.with_name(f"{path.name}{suffix}"), path.with_suffix(suffix)]
    else:
        candidates = [path.with_name(f"{path.name}{suffix}")]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def companion_manifest(path: Path, suffix: str) -> Path:
    candidates = companion_manifest_candidates(path, suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def add_limit_finding(
    findings: list[Finding],
    path: Path,
    kind: str,
    field: str,
    value: int,
    limit: int | None,
    relation: str,
) -> None:
    if limit is None:
        return
    failed = value < limit if relation == "min" else value > limit
    if failed:
        comparator = "below" if relation == "min" else "above"
        findings.append(
            Finding(
                str(path),
                "error",
                kind,
                field,
                str(value),
                str(limit),
                f"{field} {value} is {comparator} budget {limit}",
            )
        )


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[BudgetRecord, list[Finding]]:
    findings: list[Finding] = []
    manifest_path = companion_manifest(path, args.manifest_suffix)
    manifest_arg = manifest_path if manifest_path.exists() else None
    if args.require_manifest and manifest_arg is None:
        findings.append(
            Finding(
                str(path),
                "error",
                "missing_manifest",
                "manifest",
                str(manifest_path),
                "existing file",
                "companion manifest is required for budgeted HCEval artifacts",
            )
        )

    try:
        dataset = hceval_inspect.parse_hceval(path)
        inspect_findings = hceval_inspect.validate_dataset(
            dataset,
            manifest_arg,
            max_prompt_bytes=args.max_prompt_bytes,
            max_choice_bytes=args.max_choice_bytes,
            max_record_payload_bytes=args.max_record_payload_bytes,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        finding = Finding(str(path), "error", "load_error", "input", "", "", str(exc))
        return (
            BudgetRecord(
                str(path),
                "fail",
                "",
                "",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                str(manifest_arg or ""),
                1,
                str(exc),
            ),
            [finding],
        )

    for detail in inspect_findings:
        findings.append(Finding(str(path), "error", "inspection_finding", "dataset", "", "", detail))

    layout = hceval_inspect.binary_layout(dataset)
    stats = hceval_inspect.byte_stats(dataset.records)
    add_limit_finding(findings, path, "min_records", "record_count", len(dataset.records), args.min_records, "min")
    add_limit_finding(findings, path, "max_records", "record_count", len(dataset.records), args.max_records, "max")
    add_limit_finding(findings, path, "max_binary_bytes", "binary_bytes", layout["binary_bytes"], args.max_binary_bytes, "max")
    add_limit_finding(findings, path, "max_metadata_bytes", "metadata_bytes", layout["metadata_bytes"], args.max_metadata_bytes, "max")
    add_limit_finding(findings, path, "max_body_bytes", "body_bytes", layout["body_bytes"], args.max_body_bytes, "max")
    add_limit_finding(
        findings,
        path,
        "max_record_header_bytes",
        "record_header_bytes",
        layout["record_header_bytes"],
        args.max_record_header_bytes,
        "max",
    )
    add_limit_finding(
        findings,
        path,
        "max_choice_length_prefix_bytes",
        "choice_length_prefix_bytes",
        layout["choice_length_prefix_bytes"],
        args.max_choice_length_prefix_bytes,
        "max",
    )
    add_limit_finding(
        findings,
        path,
        "max_total_prompt_bytes",
        "total_prompt_bytes",
        stats["total_prompt_bytes"],
        args.max_total_prompt_bytes,
        "max",
    )
    add_limit_finding(
        findings,
        path,
        "max_total_choice_bytes",
        "total_choice_bytes",
        stats["total_choice_bytes"],
        args.max_total_choice_bytes,
        "max",
    )
    total_provenance_bytes = sum(len(record.provenance.encode("utf-8")) for record in dataset.records)
    add_limit_finding(
        findings,
        path,
        "max_total_provenance_bytes",
        "total_provenance_bytes",
        total_provenance_bytes,
        args.max_total_provenance_bytes,
        "max",
    )

    return (
        BudgetRecord(
            source=str(path),
            status="fail" if findings else "pass",
            dataset=str(dataset.metadata.get("dataset", "")),
            split=str(dataset.metadata.get("split", "")),
            record_count=len(dataset.records),
            binary_bytes=int(layout["binary_bytes"]),
            metadata_bytes=int(layout["metadata_bytes"]),
            body_bytes=int(layout["body_bytes"]),
            record_header_bytes=int(layout["record_header_bytes"]),
            choice_length_prefix_bytes=int(layout["choice_length_prefix_bytes"]),
            total_prompt_bytes=int(stats["total_prompt_bytes"]),
            total_choice_bytes=int(stats["total_choice_bytes"]),
            total_provenance_bytes=total_provenance_bytes,
            max_prompt_bytes=int(stats["max_prompt_bytes"]),
            max_choice_bytes=int(stats["max_choice_bytes"]),
            max_record_payload_bytes=int(stats["max_record_payload_bytes"]),
            manifest=str(manifest_arg or ""),
            finding_count=len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[BudgetRecord], list[Finding]]:
    sources = list(iter_input_files(paths, args.pattern))
    records: list[BudgetRecord] = []
    findings: list[Finding] = []
    if len(sources) < args.min_artifacts:
        findings.append(
            Finding(
                "",
                "error",
                "min_artifacts",
                "artifacts",
                str(len(sources)),
                str(args.min_artifacts),
                f"found {len(sources)} HCEval artifact(s), below minimum {args.min_artifacts}",
            )
        )
    for source in sources:
        record, source_findings = audit_artifact(source, args)
        records.append(record)
        findings.extend(source_findings)
    return records, findings


def build_report(records: list[BudgetRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "passing_artifacts": sum(1 for record in records if record.status == "pass"),
            "failing_artifacts": sum(1 for record in records if record.status == "fail"),
            "total_records": sum(record.record_count for record in records),
            "total_binary_bytes": sum(record.binary_bytes for record in records),
            "total_prompt_bytes": sum(record.total_prompt_bytes for record in records),
            "total_choice_bytes": sum(record.total_choice_bytes for record in records),
            "total_provenance_bytes": sum(record.total_provenance_bytes for record in records),
            "max_binary_bytes": max((record.binary_bytes for record in records), default=0),
            "max_total_prompt_bytes": max((record.total_prompt_bytes for record in records), default=0),
            "max_total_choice_bytes": max((record.total_choice_bytes for record in records), default=0),
            "max_total_provenance_bytes": max((record.total_provenance_bytes for record in records), default=0),
            "max_record_payload_bytes": max((record.max_record_payload_bytes for record in records), default=0),
            "findings": len(findings),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# HCEval Budget Audit",
        "",
        f"Status: {report['status']}",
        f"Artifacts: {summary['artifacts']}",
        f"Total records: {summary['total_records']}",
        f"Total binary bytes: {summary['total_binary_bytes']}",
        f"Total prompt bytes: {summary['total_prompt_bytes']}",
        f"Total choice bytes: {summary['total_choice_bytes']}",
        f"Total provenance bytes: {summary['total_provenance_bytes']}",
        f"Findings: {summary['findings']}",
        "",
    ]
    findings = report["findings"]
    if findings:
        lines.extend(["| Artifact | Kind | Field | Value | Limit | Detail |", "| --- | --- | --- | ---: | ---: | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {kind} | {field} | {value} | {limit} | {detail} |".format(
                    source=finding["source"] or "-",
                    kind=finding["kind"],
                    field=finding["field"],
                    value=finding["value"] or "-",
                    limit=finding["limit"] or "-",
                    detail=finding["detail"].replace("|", "\\|"),
                )
            )
    else:
        lines.append("All audited HCEval artifacts are within budget.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[BudgetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BudgetRecord.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    failures_by_source: dict[str, list[Finding]] = {}
    for finding in findings:
        failures_by_source.setdefault(finding.source or "coverage", []).append(finding)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_budget_audit",
            "tests": str(max(1, len(failures_by_source) or 1)),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "hceval_budget_audit", "name": "all_artifacts"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "hceval_budget_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "hceval_budget_violation",
                "message": "; ".join(finding.kind for finding in source_findings),
            },
        )
        failure.text = "\n".join(finding.detail for finding in source_findings)
    ET.indent(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="HCEval files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="hceval_budget_audit_latest")
    parser.add_argument("--manifest-suffix", default=".manifest.json", help="companion manifest suffix")
    parser.add_argument("--require-manifest", action="store_true", help="fail if a companion manifest is missing")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-binary-bytes", type=int)
    parser.add_argument("--max-metadata-bytes", type=int)
    parser.add_argument("--max-body-bytes", type=int)
    parser.add_argument("--max-record-header-bytes", type=int)
    parser.add_argument("--max-choice-length-prefix-bytes", type=int)
    parser.add_argument("--max-total-prompt-bytes", type=int)
    parser.add_argument("--max-total-choice-bytes", type=int)
    parser.add_argument("--max-total-provenance-bytes", type=int)
    parser.add_argument("--max-prompt-bytes", type=int)
    parser.add_argument("--max-choice-bytes", type=int)
    parser.add_argument("--max-record-payload-bytes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, findings = audit(args.inputs, args)
    report = build_report(records, findings)

    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", report)
    write_markdown(output_dir / f"{stem}.md", report)
    write_records_csv(output_dir / f"{stem}.csv", records)
    write_findings_csv(output_dir / f"{stem}_findings.csv", findings)
    write_junit(output_dir / f"{stem}_junit.xml", findings)

    print(f"wrote_json={output_dir / f'{stem}.json'}")
    print(f"artifacts={len(records)}")
    print(f"findings={len(findings)}")
    print(f"status={report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
