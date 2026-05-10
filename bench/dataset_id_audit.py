#!/usr/bin/env python3
"""Audit local eval JSONL record IDs before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then checks that record identifiers are explicit, bounded, and
unique enough for stable HolyC eval artifacts.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack

ID_KEYS = ("id", "ind", "question_id")


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    explicit_id: bool
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class IdFinding:
    severity: str
    kind: str
    scope: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def row_has_explicit_id(row: dict[str, Any]) -> bool:
    return any(dataset_pack.clean_text(row.get(key)) for key in ID_KEYS)


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[IdFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[IdFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(IdFinding("error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(IdFinding("error", "schema_error", f"{path}:{index + 1}", str(exc)))
                continue
            records.append(LoadedRecord(str(path), index + 1, row_has_explicit_id(row), record))

    return records, inputs, findings


def id_class(record_id: str, pattern: re.Pattern[str] | None) -> str:
    if not record_id:
        return "empty"
    if pattern is not None and not pattern.fullmatch(record_id):
        return "pattern_mismatch"
    if record_id.strip() != record_id:
        return "surrounding_space"
    return "ok"


def record_telemetry(loaded: LoadedRecord, pattern: re.Pattern[str] | None) -> dict[str, Any]:
    record = loaded.record
    record_id_bytes = len(record.record_id.encode("utf-8"))
    return {
        "source": source_ref(loaded),
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "record_id_bytes": record_id_bytes,
        "explicit_id": loaded.explicit_id,
        "id_class": id_class(record.record_id, pattern),
        "dataset_split_record_key": f"{record.dataset}:{record.split}:{record.record_id}",
    }


def sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def duplicate_groups(rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(str(row["source"]))
    return {value: sources for value, sources in sorted(grouped.items()) if len(sources) > 1}


def add_gate_findings(
    findings: list[IdFinding],
    rows: list[dict[str, Any]],
    *,
    require_explicit_id: bool,
    max_record_id_bytes: int | None,
    id_pattern: str | None,
    fail_duplicate_record_ids: bool,
    fail_duplicate_dataset_split_record_ids: bool,
) -> None:
    for row in rows:
        scope = f"{row['dataset']}:{row['split']}:{row['record_id']}"
        if require_explicit_id and not row["explicit_id"]:
            findings.append(IdFinding("error", "implicit_record_id", scope, f"{row['source']} used generated id"))
        if max_record_id_bytes is not None and int(row["record_id_bytes"]) > max_record_id_bytes:
            findings.append(
                IdFinding(
                    "error",
                    "record_id_too_long",
                    scope,
                    f"{row['source']} record id is {row['record_id_bytes']} bytes, above {max_record_id_bytes}",
                )
            )
        if id_pattern is not None and row["id_class"] == "pattern_mismatch":
            findings.append(
                IdFinding(
                    "error",
                    "record_id_pattern_mismatch",
                    scope,
                    f"{row['source']} record id does not match {id_pattern!r}",
                )
            )

    duplicate_ids = duplicate_groups(rows, "record_id")
    if fail_duplicate_record_ids:
        for record_id, sources in duplicate_ids.items():
            findings.append(
                IdFinding(
                    "error",
                    "duplicate_record_id",
                    record_id,
                    f"record id appears in {len(sources)} rows: {', '.join(sources)}",
                )
            )

    duplicate_dataset_split_ids = duplicate_groups(rows, "dataset_split_record_key")
    if fail_duplicate_dataset_split_record_ids:
        for key, sources in duplicate_dataset_split_ids.items():
            findings.append(
                IdFinding(
                    "error",
                    "duplicate_dataset_split_record_id",
                    key,
                    f"dataset/split record id appears in {len(sources)} rows: {', '.join(sources)}",
                )
            )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = [Path(path) for path in args.input]
    pattern = re.compile(args.id_pattern) if args.id_pattern else None
    records, input_info, findings = load_records(inputs, args.default_dataset, args.default_split)
    rows = [record_telemetry(record, pattern) for record in records]

    add_gate_findings(
        findings,
        rows,
        require_explicit_id=args.require_explicit_id,
        max_record_id_bytes=args.max_record_id_bytes,
        id_pattern=args.id_pattern,
        fail_duplicate_record_ids=args.fail_duplicate_record_ids,
        fail_duplicate_dataset_split_record_ids=args.fail_duplicate_dataset_split_record_ids,
    )

    id_bytes = [int(row["record_id_bytes"]) for row in rows]
    duplicate_ids = duplicate_groups(rows, "record_id")
    duplicate_dataset_split_ids = duplicate_groups(rows, "dataset_split_record_key")
    error_count = sum(1 for finding in findings if finding.severity == "error")

    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": input_info,
        "record_count": len(rows),
        "explicit_id_count": sum(1 for row in rows if row["explicit_id"]),
        "implicit_id_count": sum(1 for row in rows if not row["explicit_id"]),
        "unique_record_id_count": len({row["record_id"] for row in rows}),
        "duplicate_record_id_count": len(duplicate_ids),
        "duplicate_dataset_split_record_id_count": len(duplicate_dataset_split_ids),
        "max_record_id_bytes": max(id_bytes, default=0),
        "id_class_histogram": sorted_counts(str(row["id_class"]) for row in rows),
        "record_telemetry": rows,
        "duplicate_record_ids": duplicate_ids,
        "duplicate_dataset_split_record_ids": duplicate_dataset_split_ids,
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "scope", "detail"])
        writer.writeheader()
        writer.writerows(findings)


def write_record_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "record_id_bytes",
        "explicit_id",
        "id_class",
        "dataset_split_record_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset ID Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Explicit IDs: {report['explicit_id_count']}",
        f"- Implicit IDs: {report['implicit_id_count']}",
        f"- Unique record IDs: {report['unique_record_id_count']}",
        f"- Duplicate record IDs: {report['duplicate_record_id_count']}",
        f"- Duplicate dataset/split record IDs: {report['duplicate_dataset_split_record_id_count']}",
        f"- Max record ID bytes: {report['max_record_id_bytes']}",
        "",
        "## ID Classes",
        "",
    ]
    for key, count in report["id_class_histogram"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['severity']} {finding['kind']} {finding['scope']}: {finding['detail']}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = report["findings"]
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_id_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(sum(1 for finding in findings if finding["severity"] == "error")),
        },
    )
    if not findings:
        ET.SubElement(testsuite, "testcase", {"classname": "dataset_id_audit", "name": "id_audit_pass"})
    for finding in findings:
        testcase = ET.SubElement(
            testsuite,
            {"classname": "dataset_id_audit", "name": f"{finding['kind']}:{finding['scope']}"},
        )
        if finding["severity"] == "error":
            failure = ET.SubElement(testcase, "failure", {"message": finding["detail"]})
            failure.text = finding["detail"]
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Local eval JSONL input path")
    parser.add_argument("--output", required=True, help="JSON report output path")
    parser.add_argument("--markdown", help="Optional Markdown summary output path")
    parser.add_argument("--csv", help="Optional findings CSV output path")
    parser.add_argument("--record-csv", help="Optional per-record ID telemetry CSV output path")
    parser.add_argument("--junit", help="Optional JUnit XML output path")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--require-explicit-id", action="store_true")
    parser.add_argument("--max-record-id-bytes", type=int)
    parser.add_argument("--id-pattern", help="Regex that normalized record IDs must fully match")
    parser.add_argument("--fail-duplicate-record-ids", action="store_true")
    parser.add_argument("--fail-duplicate-dataset-split-record-ids", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args)
    write_json(Path(args.output), report)
    if args.markdown:
        write_markdown(Path(args.markdown), report)
    if args.csv:
        write_csv(Path(args.csv), report["findings"])
    if args.record_csv:
        write_record_csv(Path(args.record_csv), report["record_telemetry"])
    if args.junit:
        write_junit(Path(args.junit), report)
    return 1 if args.fail_on_findings and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
