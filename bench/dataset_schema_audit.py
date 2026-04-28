#!/usr/bin/env python3
"""Audit local eval JSONL schema before curation or HCEval packing.

The audit is offline-only. It normalizes the same row shapes accepted by
dataset_pack.py, records dataset/split/choice/answer/byte telemetry, and can
fail on malformed rows, missing provenance, or loader-size gate violations.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
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

import dataset_pack


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    source: str
    detail: str


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def append_finding(findings: list[Finding], severity: str, kind: str, source: str, detail: str) -> None:
    findings.append(Finding(severity=severity, kind=kind, source=source, detail=detail))


def read_rows(path: Path, findings: list[Finding]) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "error", "read_error", str(path), str(exc))
        return []


def source_ref(path: Path, row_number: int) -> str:
    return f"{path}:{row_number}"


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
    require_provenance: bool,
    findings: list[Finding],
) -> tuple[list[LoadedRecord], list[dict[str, Any]]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []

    for path in paths:
        rows = read_rows(path, findings)
        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            row_number = index + 1
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                append_finding(findings, "error", "schema_error", source_ref(path, row_number), str(exc))
                continue

            if require_provenance and not clean(row.get("provenance") or row.get("source")):
                append_finding(
                    findings,
                    "error",
                    "missing_provenance",
                    source_ref(path, row_number),
                    "row should include provenance or source before curation",
                )
            records.append(LoadedRecord(str(path), row_number, record))

    return records, inputs


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def nested_counts(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        dataset_counts = counts.setdefault(loaded.record.dataset, {})
        split = loaded.record.split
        dataset_counts[split] = dataset_counts.get(split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def duplicate_ids(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[LoadedRecord]] = {}
    for loaded in records:
        groups.setdefault((loaded.record.dataset, loaded.record.record_id), []).append(loaded)
    duplicates: list[dict[str, Any]] = []
    for (dataset, record_id), group in sorted(groups.items()):
        if len(group) > 1:
            duplicates.append(
                {
                    "dataset": dataset,
                    "record_id": record_id,
                    "sources": [source_ref(Path(loaded.source), loaded.row_number) for loaded in group],
                }
            )
    return duplicates


def apply_record_gates(
    records: list[LoadedRecord],
    min_choices: int | None,
    max_choices: int | None,
    max_prompt_bytes: int | None,
    max_choice_bytes: int | None,
    max_record_payload_bytes: int | None,
    fail_on_duplicate_ids: bool,
    findings: list[Finding],
) -> None:
    for loaded in records:
        record = loaded.record
        ref = source_ref(Path(loaded.source), loaded.row_number)
        if min_choices is not None and len(record.choices) < min_choices:
            append_finding(
                findings,
                "error",
                "too_few_choices",
                ref,
                f"{len(record.choices)} choices, minimum is {min_choices}",
            )
        if max_choices is not None and len(record.choices) > max_choices:
            append_finding(
                findings,
                "error",
                "too_many_choices",
                ref,
                f"{len(record.choices)} choices, maximum is {max_choices}",
            )

    for duplicate in duplicate_ids(records):
        append_finding(
            findings,
            "error" if fail_on_duplicate_ids else "warning",
            "duplicate_record_id",
            ",".join(duplicate["sources"]),
            f"{duplicate['dataset']} record id {duplicate['record_id']!r} appears more than once",
        )

    for message in dataset_pack.size_limit_findings(
        [loaded.record for loaded in records],
        max_prompt_bytes=max_prompt_bytes,
        max_choice_bytes=max_choice_bytes,
        max_record_payload_bytes=max_record_payload_bytes,
    ):
        append_finding(findings, "error", "byte_limit", "records", message)


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    findings: list[Finding],
) -> dict[str, Any]:
    normalized = [loaded.record for loaded in records]
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-schema-audit",
        "status": "fail" if error_count else "pass",
        "inputs": inputs,
        "input_row_count": sum(int(input_info["rows"]) for input_info in inputs),
        "normalized_record_count": len(records),
        "dataset_split_counts": nested_counts(records),
        "choice_count_histogram": sorted_counts(len(record.choices) for record in normalized),
        "answer_histogram": dataset_pack.answer_histogram(normalized),
        "byte_stats": dataset_pack.byte_stats(normalized),
        "duplicate_record_ids": duplicate_ids(records),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Dataset Schema Audit",
        "",
        f"- Status: {report['status']}",
        f"- Input rows: {report['input_row_count']}",
        f"- Normalized records: {report['normalized_record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Dataset/Split Counts",
        "",
        "| dataset | split | records |",
        "| --- | --- | ---: |",
    ]
    for dataset, split_counts in report["dataset_split_counts"].items():
        for split, count in split_counts.items():
            lines.append(f"| {dataset} | {split} | {count} |")

    lines.extend(
        [
            "",
            "## Histograms",
            "",
            f"- Choice counts: `{json.dumps(report['choice_count_histogram'], sort_keys=True)}`",
            f"- Answer indexes: `{json.dumps(report['answer_histogram'], sort_keys=True)}`",
            "",
            "## Findings",
            "",
        ]
    )
    if not report["findings"]:
        lines.append("No schema findings.")
    else:
        lines.extend(["| severity | kind | source | detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['kind']} | {finding['source']} | {finding['detail']} |"
            )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "source", "detail"])
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_junit(report: dict[str, Any], path: Path) -> None:
    error_findings = [finding for finding in report["findings"] if finding["severity"] == "error"]
    testcase_count = len(error_findings) + (1 if not error_findings else 0)
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_schema_audit",
            "tests": str(testcase_count),
            "failures": str(len(error_findings)),
            "errors": "0",
        },
    )
    if error_findings:
        for finding in error_findings:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "classname": "dataset_schema_audit",
                    "name": f"{finding['kind']}:{finding['source']}",
                },
            )
            ET.SubElement(
                testcase,
                "failure",
                {"type": finding["kind"], "message": finding["detail"]},
            )
    else:
        ET.SubElement(testsuite, "testcase", {"classname": "dataset_schema_audit", "name": "schema_pass"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(
    report: dict[str, Any],
    output: Path,
    markdown: Path | None,
    csv_path: Path | None,
    junit: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(report), encoding="utf-8")
    if csv_path:
        write_csv(report, csv_path)
    if junit:
        write_junit(report, junit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input eval JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report")
    parser.add_argument("--csv", type=Path, help="Optional CSV findings output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument("--require-provenance", action="store_true", help="Fail if a row lacks provenance/source")
    parser.add_argument("--min-choices", type=int, help="Fail if any normalized row has fewer choices")
    parser.add_argument("--max-choices", type=int, help="Fail if any normalized row has more choices")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail if any cleaned prompt exceeds this UTF-8 limit")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail if any cleaned choice exceeds this UTF-8 limit")
    parser.add_argument(
        "--max-record-payload-bytes",
        type=int,
        help="Fail if any record payload excluding the fixed record header exceeds this byte limit",
    )
    parser.add_argument("--fail-on-duplicate-ids", action="store_true", help="Treat duplicate record IDs as errors")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit nonzero when errors are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings: list[Finding] = []
    records, inputs = load_records(
        args.input,
        args.default_dataset,
        args.default_split,
        args.require_provenance,
        findings,
    )
    apply_record_gates(
        records,
        args.min_choices,
        args.max_choices,
        args.max_prompt_bytes,
        args.max_choice_bytes,
        args.max_record_payload_bytes,
        args.fail_on_duplicate_ids,
        findings,
    )
    report = build_report(inputs, records, findings)
    write_outputs(report, args.output, args.markdown, args.csv, args.junit)

    print(f"wrote_report={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")

    if args.fail_on_findings and report["error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
