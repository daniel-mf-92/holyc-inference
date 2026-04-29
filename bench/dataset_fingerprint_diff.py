#!/usr/bin/env python3
"""Compare eval dataset fingerprint reports for row-level drift.

The diff is offline-only and consumes JSON reports emitted by
dataset_fingerprint.py. It is intended for CI gates that should reject
unreviewed changes to curated eval prompts, choices, answers, or record
membership before packed HCEval artifacts are promoted.
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


CONTENT_FIELDS = ("prompt_sha256", "choices_sha256", "input_sha256", "choice_count")
ANSWER_FIELDS = ("answer_index",)
METADATA_FIELDS = ("dataset", "split")
PAYLOAD_FIELDS = ("answer_payload_sha256", "full_payload_sha256")


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    key: str
    detail: str


@dataclass(frozen=True)
class DiffRow:
    change_type: str
    key: str
    field: str
    baseline_value: str
    candidate_value: str
    baseline_source: str
    candidate_source: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def source_ref(row: dict[str, Any]) -> str:
    source = stringify(row.get("source"))
    row_number = stringify(row.get("row_number"))
    return f"{source}:{row_number}" if source or row_number else ""


def read_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: fingerprint report must be a JSON object")
    fingerprints = data.get("fingerprints")
    if not isinstance(fingerprints, list):
        raise ValueError(f"{path}: missing fingerprints list")
    return data


def index_fingerprints(
    report: dict[str, Any],
    *,
    key_field: str,
    source_name: str,
    findings: list[Finding],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(report.get("fingerprints", []), start=1):
        if not isinstance(row, dict):
            findings.append(
                Finding("error", "invalid_fingerprint_row", f"{source_name}:{row_index}", "row is not an object")
            )
            continue
        key = stringify(row.get(key_field))
        if not key:
            findings.append(
                Finding("error", "missing_key_field", f"{source_name}:{row_index}", f"missing {key_field}")
            )
            continue
        if key in indexed:
            findings.append(
                Finding("error", "duplicate_key", key, f"{source_name} has duplicate {key_field}")
            )
            continue
        indexed[key] = row
    return indexed


def append_field_change(
    rows: list[DiffRow],
    change_type: str,
    key: str,
    field: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    rows.append(
        DiffRow(
            change_type=change_type,
            key=key,
            field=field,
            baseline_value=stringify(baseline.get(field)),
            candidate_value=stringify(candidate.get(field)),
            baseline_source=source_ref(baseline),
            candidate_source=source_ref(candidate),
        )
    )


def diff_indexes(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[DiffRow]:
    rows: list[DiffRow] = []
    for key in sorted(set(baseline) | set(candidate)):
        left = baseline.get(key)
        right = candidate.get(key)
        if left is None and right is not None:
            rows.append(DiffRow("added", key, "", "", "", "", source_ref(right)))
            continue
        if left is not None and right is None:
            rows.append(DiffRow("removed", key, "", "", "", source_ref(left), ""))
            continue
        if left is None or right is None:
            continue

        for field in CONTENT_FIELDS:
            if left.get(field) != right.get(field):
                append_field_change(rows, "content_changed", key, field, left, right)
        for field in ANSWER_FIELDS:
            if left.get(field) != right.get(field):
                append_field_change(rows, "answer_changed", key, field, left, right)
        for field in METADATA_FIELDS:
            if left.get(field) != right.get(field):
                append_field_change(rows, "metadata_changed", key, field, left, right)
        for field in PAYLOAD_FIELDS:
            if left.get(field) != right.get(field):
                append_field_change(rows, "payload_changed", key, field, left, right)
    return rows


def change_counts(rows: list[DiffRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.change_type] = counts.get(row.change_type, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def add_gate_findings(
    findings: list[Finding],
    rows: list[DiffRow],
    *,
    fail_on_added: bool,
    fail_on_removed: bool,
    fail_on_content_changes: bool,
    fail_on_answer_changes: bool,
    fail_on_metadata_changes: bool,
    fail_on_payload_changes: bool,
    fail_on_any_change: bool,
) -> None:
    gated_types: set[str] = set()
    if fail_on_added:
        gated_types.add("added")
    if fail_on_removed:
        gated_types.add("removed")
    if fail_on_content_changes:
        gated_types.add("content_changed")
    if fail_on_answer_changes:
        gated_types.add("answer_changed")
    if fail_on_metadata_changes:
        gated_types.add("metadata_changed")
    if fail_on_payload_changes:
        gated_types.add("payload_changed")
    if fail_on_any_change:
        gated_types.update(row.change_type for row in rows)

    for row in rows:
        if row.change_type in gated_types:
            field = f" field={row.field}" if row.field else ""
            findings.append(
                Finding("error", row.change_type, row.key, f"{row.change_type}{field} is gated")
            )


def build_report(
    baseline_path: Path,
    candidate_path: Path,
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    rows: list[DiffRow],
    findings: list[Finding],
    *,
    key_field: str,
    fail_on_findings: bool,
) -> dict[str, Any]:
    gated = fail_on_findings or any(finding.severity == "error" for finding in findings)
    status = "fail" if gated and any(finding.severity == "error" for finding in findings) else "pass"
    baseline_keys = {
        stringify(row.get(key_field))
        for row in baseline_report.get("fingerprints", [])
        if isinstance(row, dict) and stringify(row.get(key_field))
    }
    candidate_keys = {
        stringify(row.get(key_field))
        for row in candidate_report.get("fingerprints", [])
        if isinstance(row, dict) and stringify(row.get(key_field))
    }
    return {
        "generated_at": iso_now(),
        "status": status,
        "key_field": key_field,
        "baseline": {
            "path": str(baseline_path),
            "report_status": baseline_report.get("status", ""),
            "record_count": len(baseline_keys),
        },
        "candidate": {
            "path": str(candidate_path),
            "report_status": candidate_report.get("status", ""),
            "record_count": len(candidate_keys),
        },
        "unchanged_records": len(baseline_keys & candidate_keys)
        - len({row.key for row in rows if row.change_type not in {"added", "removed"}}),
        "change_counts": change_counts(rows),
        "changes": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[DiffRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DiffRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval Dataset Fingerprint Diff",
        "",
        f"- Status: {report['status']}",
        f"- Key field: {report['key_field']}",
        f"- Baseline records: {report['baseline']['record_count']}",
        f"- Candidate records: {report['candidate']['record_count']}",
        f"- Unchanged records: {report['unchanged_records']}",
        f"- Changes: {len(report['changes'])}",
        f"- Findings: {len(report['findings'])}",
        "",
        "## Change Counts",
        "",
    ]
    if report["change_counts"]:
        for kind, count in report["change_counts"].items():
            lines.append(f"- {kind}: {count}")
    else:
        lines.append("- none: 0")
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['severity']} {finding['kind']} {finding['key']}: {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_fingerprint_diff",
            "tests": "1",
            "failures": str(len(failures)),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "dataset_fingerprint_diff"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} fingerprint diff finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['key']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline dataset_fingerprint.py JSON report")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate dataset_fingerprint.py JSON report")
    parser.add_argument("--output", required=True, type=Path, help="Summary JSON output")
    parser.add_argument("--csv", type=Path, help="Optional row-level diff CSV")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--key-field", default="record_id", help="Fingerprint row field used as the stable key")
    parser.add_argument("--fail-on-added", action="store_true")
    parser.add_argument("--fail-on-removed", action="store_true")
    parser.add_argument("--fail-on-content-changes", action="store_true")
    parser.add_argument("--fail-on-answer-changes", action="store_true")
    parser.add_argument("--fail-on-metadata-changes", action="store_true")
    parser.add_argument("--fail-on-payload-changes", action="store_true")
    parser.add_argument("--fail-on-any-change", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings: list[Finding] = []
    try:
        baseline_report = read_report(args.baseline)
        candidate_report = read_report(args.candidate)
        baseline_index = index_fingerprints(
            baseline_report,
            key_field=args.key_field,
            source_name=str(args.baseline),
            findings=findings,
        )
        candidate_index = index_fingerprints(
            candidate_report,
            key_field=args.key_field,
            source_name=str(args.candidate),
            findings=findings,
        )
        rows = diff_indexes(baseline_index, candidate_index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    add_gate_findings(
        findings,
        rows,
        fail_on_added=args.fail_on_added,
        fail_on_removed=args.fail_on_removed,
        fail_on_content_changes=args.fail_on_content_changes,
        fail_on_answer_changes=args.fail_on_answer_changes,
        fail_on_metadata_changes=args.fail_on_metadata_changes,
        fail_on_payload_changes=args.fail_on_payload_changes,
        fail_on_any_change=args.fail_on_any_change,
    )
    report = build_report(
        args.baseline,
        args.candidate,
        baseline_report,
        candidate_report,
        rows,
        findings,
        key_field=args.key_field,
        fail_on_findings=args.fail_on_findings,
    )
    write_json(args.output, report)
    if args.csv:
        write_csv(args.csv, rows)
    if args.markdown:
        write_markdown(args.markdown, report)
    if args.junit:
        write_junit(args.junit, report)
    print(args.output)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
