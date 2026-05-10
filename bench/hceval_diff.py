#!/usr/bin/env python3
"""Diff two HolyC-loadable offline eval datasets.

This host-side tool reads `.hceval` files produced by `dataset_pack.py` and
reports record, order, metadata, and payload drift. It never launches QEMU and
never touches the TempleOS guest.
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
class DiffRecord:
    record_id: str
    status: str
    reference_index: int | None
    candidate_index: int | None
    changed_fields: str
    reference_input_sha256: str
    candidate_input_sha256: str
    reference_answer_payload_sha256: str
    candidate_answer_payload_sha256: str


@dataclass(frozen=True)
class DiffFinding:
    kind: str
    record_id: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def keyed_fingerprints(dataset: hceval_inspect.HCEvalDataset) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in hceval_inspect.record_fingerprints(dataset)}


def record_maps(
    reference: hceval_inspect.HCEvalDataset,
    candidate: hceval_inspect.HCEvalDataset,
) -> tuple[dict[str, hceval_inspect.InspectRecord], dict[str, hceval_inspect.InspectRecord]]:
    return (
        {record.record_id: record for record in reference.records},
        {record.record_id: record for record in candidate.records},
    )


def changed_fields(
    reference: hceval_inspect.InspectRecord,
    candidate: hceval_inspect.InspectRecord,
) -> list[str]:
    fields: list[str] = []
    for field in ("prompt", "choices", "answer_index", "provenance", "flags"):
        if getattr(reference, field) != getattr(candidate, field):
            fields.append(field)
    return fields


def build_diff_records(
    reference: hceval_inspect.HCEvalDataset,
    candidate: hceval_inspect.HCEvalDataset,
) -> list[DiffRecord]:
    reference_rows, candidate_rows = record_maps(reference, candidate)
    reference_fingerprints = keyed_fingerprints(reference)
    candidate_fingerprints = keyed_fingerprints(candidate)
    reference_order = {record.record_id: index for index, record in enumerate(reference.records)}
    candidate_order = {record.record_id: index for index, record in enumerate(candidate.records)}

    records: list[DiffRecord] = []
    for record_id in sorted(set(reference_rows) | set(candidate_rows)):
        ref = reference_rows.get(record_id)
        cand = candidate_rows.get(record_id)
        ref_fp = reference_fingerprints.get(record_id, {})
        cand_fp = candidate_fingerprints.get(record_id, {})
        if ref is None:
            status = "added"
            fields: list[str] = []
        elif cand is None:
            status = "removed"
            fields = []
        else:
            fields = changed_fields(ref, cand)
            order_changed = reference_order[record_id] != candidate_order[record_id]
            status = "changed" if fields else "reordered" if order_changed else "unchanged"
            if order_changed:
                fields = [*fields, "record_index"]

        records.append(
            DiffRecord(
                record_id=record_id,
                status=status,
                reference_index=reference_order.get(record_id),
                candidate_index=candidate_order.get(record_id),
                changed_fields=",".join(fields),
                reference_input_sha256=str(ref_fp.get("input_sha256", "")),
                candidate_input_sha256=str(cand_fp.get("input_sha256", "")),
                reference_answer_payload_sha256=str(ref_fp.get("answer_payload_sha256", "")),
                candidate_answer_payload_sha256=str(cand_fp.get("answer_payload_sha256", "")),
            )
        )
    return records


def metadata_findings(
    reference: hceval_inspect.HCEvalDataset,
    candidate: hceval_inspect.HCEvalDataset,
) -> list[DiffFinding]:
    findings: list[DiffFinding] = []
    for key in ("format", "version", "dataset", "split", "record_count"):
        if reference.metadata.get(key) != candidate.metadata.get(key):
            findings.append(
                DiffFinding(
                    "metadata_changed",
                    "",
                    f"{key}: reference={reference.metadata.get(key)!r} candidate={candidate.metadata.get(key)!r}",
                )
            )
    if reference.source_digest != candidate.source_digest:
        findings.append(DiffFinding("source_digest_changed", "", "header source digest differs"))
    if reference.payload_sha256 != candidate.payload_sha256:
        findings.append(DiffFinding("payload_digest_changed", "", "binary payload digest differs"))
    return findings


def evaluate(records: list[DiffRecord], metadata_drift: list[DiffFinding], *, allow_order_changes: bool) -> list[DiffFinding]:
    findings = list(metadata_drift)
    for record in records:
        if record.status == "added":
            findings.append(DiffFinding("record_added", record.record_id, "candidate contains an extra record"))
        elif record.status == "removed":
            findings.append(DiffFinding("record_removed", record.record_id, "candidate is missing the reference record"))
        elif record.status == "changed":
            findings.append(
                DiffFinding("record_changed", record.record_id, f"changed fields: {record.changed_fields}")
            )
        elif record.status == "reordered" and not allow_order_changes:
            findings.append(
                DiffFinding(
                    "record_reordered",
                    record.record_id,
                    f"reference index {record.reference_index} moved to candidate index {record.candidate_index}",
                )
            )
    return findings


def build_report(
    reference_path: Path,
    candidate_path: Path,
    reference: hceval_inspect.HCEvalDataset,
    candidate: hceval_inspect.HCEvalDataset,
    records: list[DiffRecord],
    findings: list[DiffFinding],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "reference": {
            "path": str(reference_path),
            "dataset": reference.metadata.get("dataset", ""),
            "split": reference.metadata.get("split", ""),
            "record_count": len(reference.records),
            "payload_sha256": reference.payload_sha256,
            "source_sha256": reference.source_digest,
        },
        "candidate": {
            "path": str(candidate_path),
            "dataset": candidate.metadata.get("dataset", ""),
            "split": candidate.metadata.get("split", ""),
            "record_count": len(candidate.records),
            "payload_sha256": candidate.payload_sha256,
            "source_sha256": candidate.source_digest,
        },
        "config": {"allow_order_changes": args.allow_order_changes},
        "summary": {
            "records": len(records),
            "unchanged": sum(1 for record in records if record.status == "unchanged"),
            "changed": sum(1 for record in records if record.status == "changed"),
            "added": sum(1 for record in records if record.status == "added"),
            "removed": sum(1 for record in records if record.status == "removed"),
            "reordered": sum(1 for record in records if record.status == "reordered"),
            "findings": len(findings),
        },
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[DiffRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DiffRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[DiffFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DiffFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# HCEval Dataset Diff",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Reference: {report['reference']['path']}",
        f"Candidate: {report['candidate']['path']}",
        "",
        "## Summary",
        "",
        f"- Records: {summary['records']}",
        f"- Unchanged: {summary['unchanged']}",
        f"- Changed: {summary['changed']}",
        f"- Added: {summary['added']}",
        f"- Removed: {summary['removed']}",
        f"- Reordered: {summary['reordered']}",
        f"- Findings: {summary['findings']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(f"- {item['kind']}: {item['record_id'] or '-'}: {item['detail']}" for item in report["findings"])
    else:
        lines.append("No findings.")
    return "\n".join(lines) + "\n"


def junit_report(report: dict[str, Any]) -> str:
    findings = report.get("findings", [])
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_diff",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "hceval_diff", "name": "dataset_diff"})
    if findings:
        failure = ET.SubElement(
            case,
            "failure",
            {"type": "hceval_diff_failure", "message": f"{len(findings)} dataset diff finding(s)"},
        )
        failure.text = "\n".join(f"{item['kind']}: {item['record_id'] or '-'}: {item['detail']}" for item in findings)
    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Reference .hceval binary")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate .hceval binary")
    parser.add_argument("--output", type=Path, help="Optional JSON diff report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional per-record diff CSV path")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path")
    parser.add_argument("--allow-order-changes", action="store_true", help="Do not fail on record order-only drift")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reference = hceval_inspect.parse_hceval(args.reference)
        candidate = hceval_inspect.parse_hceval(args.candidate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    records = build_diff_records(reference, candidate)
    findings = evaluate(
        records,
        metadata_findings(reference, candidate),
        allow_order_changes=args.allow_order_changes,
    )
    report = build_report(args.reference, args.candidate, reference, candidate, records, findings, args)

    if args.output:
        write_json(args.output, report)
    if args.markdown:
        write_text(args.markdown, markdown_report(report))
    if args.csv:
        write_records_csv(args.csv, records)
    if args.findings_csv:
        write_findings_csv(args.findings_csv, findings)
    if args.junit:
        write_text(args.junit, junit_report(report))

    print(f"status={report['status']}")
    print(f"records={report['summary']['records']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
