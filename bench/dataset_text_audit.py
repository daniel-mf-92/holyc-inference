#!/usr/bin/env python3
"""Audit normalized eval text for loader-hostile characters and byte budgets.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then checks prompt and choice strings for blank text, NUL/C0
control characters, Unicode replacement markers, and optional UTF-8 byte gates
before rows are curated or packed into HCEval binaries.
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


ALLOWED_CONTROL_CHARS = {"\n", "\t"}
CHOICE_LABEL_PREFIX_RE = re.compile(r"^\s*(?:[A-Za-z]|\d{1,3})[\).:\]-]\s+")


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class TextFinding:
    severity: str
    kind: str
    source: str
    dataset: str
    split: str
    record_id: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(record: LoadedRecord) -> str:
    return f"{record.source}:{record.row_number}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[TextFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[TextFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(
                TextFinding("error", "read_error", str(path), "", "", "", "", str(exc))
            )
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(
                    TextFinding(
                        "error",
                        "schema_error",
                        f"{path}:{index + 1}",
                        "",
                        "",
                        "",
                        "",
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def text_metrics(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    lines = text.split("\n") if text else [""]
    control_chars = sorted({f"U+{ord(char):04X}" for char in text if ord(char) < 32 and char not in ALLOWED_CONTROL_CHARS})
    return {
        "bytes": len(encoded),
        "chars": len(text),
        "line_count": len(lines),
        "max_line_bytes": max((len(line.encode("utf-8")) for line in lines), default=0),
        "control_chars": ",".join(control_chars),
        "nul_count": text.count("\x00"),
        "replacement_char_count": text.count("\ufffd"),
    }


def choice_label_prefix(text: str) -> str:
    match = CHOICE_LABEL_PREFIX_RE.match(text)
    return match.group(0).strip() if match else ""


def add_finding(
    findings: list[TextFinding],
    loaded: LoadedRecord,
    severity: str,
    kind: str,
    field: str,
    detail: str,
) -> None:
    record = loaded.record
    findings.append(
        TextFinding(
            severity=severity,
            kind=kind,
            source=source_ref(loaded),
            dataset=record.dataset,
            split=record.split,
            record_id=record.record_id,
            field=field,
            detail=detail,
        )
    )


def iter_text_fields(record: dataset_pack.EvalRecord) -> Iterable[tuple[str, str]]:
    yield "prompt", record.prompt
    for index, choice in enumerate(record.choices):
        yield f"choice[{index}]", choice


def build_record_rows(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loaded in records:
        record = loaded.record
        for field, text in iter_text_fields(record):
            metrics = text_metrics(text)
            rows.append(
                {
                    "source": source_ref(loaded),
                    "dataset": record.dataset,
                    "split": record.split,
                    "record_id": record.record_id,
                    "field": field,
                    "bytes": metrics["bytes"],
                    "chars": metrics["chars"],
                    "line_count": metrics["line_count"],
                    "max_line_bytes": metrics["max_line_bytes"],
                    "control_chars": metrics["control_chars"],
                    "nul_count": metrics["nul_count"],
                    "replacement_char_count": metrics["replacement_char_count"],
                    "choice_label_prefix": choice_label_prefix(text) if field.startswith("choice[") else "",
                }
            )
    return rows


def audit_records(
    records: list[LoadedRecord],
    findings: list[TextFinding],
    *,
    max_prompt_bytes: int | None,
    max_choice_bytes: int | None,
    max_line_bytes: int | None,
    fail_on_control_chars: bool,
    fail_on_replacement_chars: bool,
    fail_on_blank_text: bool,
    fail_on_choice_label_prefixes: bool,
) -> None:
    for loaded in records:
        for field, text in iter_text_fields(loaded.record):
            metrics = text_metrics(text)
            field_max_bytes = max_prompt_bytes if field == "prompt" else max_choice_bytes

            if fail_on_blank_text and not text.strip():
                add_finding(findings, loaded, "error", "blank_text", field, "field is empty after normalization")

            if metrics["control_chars"]:
                add_finding(
                    findings,
                    loaded,
                    "error" if fail_on_control_chars else "warning",
                    "control_character",
                    field,
                    f"disallowed control characters: {metrics['control_chars']}",
                )

            if metrics["replacement_char_count"]:
                add_finding(
                    findings,
                    loaded,
                    "error" if fail_on_replacement_chars else "warning",
                    "unicode_replacement_character",
                    field,
                    f"{metrics['replacement_char_count']} replacement characters",
                )

            if field_max_bytes is not None and metrics["bytes"] > field_max_bytes:
                add_finding(
                    findings,
                    loaded,
                    "error",
                    "field_byte_budget_exceeded",
                    field,
                    f"{metrics['bytes']} UTF-8 bytes exceeds limit {field_max_bytes}",
                )

            if max_line_bytes is not None and metrics["max_line_bytes"] > max_line_bytes:
                add_finding(
                    findings,
                    loaded,
                    "error",
                    "line_byte_budget_exceeded",
                    field,
                    f"longest line is {metrics['max_line_bytes']} UTF-8 bytes; limit {max_line_bytes}",
                )

            prefix = choice_label_prefix(text) if field.startswith("choice[") else ""
            if prefix and fail_on_choice_label_prefixes:
                add_finding(
                    findings,
                    loaded,
                    "error",
                    "choice_label_prefix",
                    field,
                    f"choice text starts with raw label prefix {prefix!r}",
                )


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def counts_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    record_rows: list[dict[str, Any]],
    findings: list[TextFinding],
) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    prompt_rows = [row for row in record_rows if row["field"] == "prompt"]
    choice_rows = [row for row in record_rows if row["field"].startswith("choice[")]
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-text-audit",
        "status": "fail" if error_count else "pass",
        "inputs": inputs,
        "record_count": len(records),
        "text_field_count": len(record_rows),
        "counts_by_dataset_split": counts_by_dataset_split(records),
        "prompt_byte_max": max((row["bytes"] for row in prompt_rows), default=0),
        "choice_byte_max": max((row["bytes"] for row in choice_rows), default=0),
        "line_byte_max": max((row["max_line_bytes"] for row in record_rows), default=0),
        "control_char_field_count": sum(1 for row in record_rows if row["control_chars"]),
        "replacement_char_field_count": sum(1 for row in record_rows if row["replacement_char_count"]),
        "choice_label_prefix_field_count": sum(1 for row in choice_rows if row["choice_label_prefix"]),
        "prompt_byte_histogram": sorted_counts(row["bytes"] for row in prompt_rows),
        "choice_byte_histogram": sorted_counts(row["bytes"] for row in choice_rows),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Text Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Text fields: {report['text_field_count']}",
        f"- Max prompt bytes: {report['prompt_byte_max']}",
        f"- Max choice bytes: {report['choice_byte_max']}",
        f"- Max line bytes: {report['line_byte_max']}",
        f"- Choice label prefix fields: {report['choice_label_prefix_field_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Dataset/Split Counts",
        "",
        "| dataset | split | records |",
        "| --- | --- | ---: |",
    ]
    for dataset, split_counts in report["counts_by_dataset_split"].items():
        for split, count in split_counts.items():
            lines.append(f"| {md_cell(dataset)} | {md_cell(split)} | {count} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No text quality findings.")
    else:
        lines.extend(
            [
                "| severity | kind | source | dataset | split | record_id | field | detail |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {source} | {dataset} | {split} | {record_id} | {field} | {detail} |".format(
                    **{key: md_cell(value) for key, value in finding.items()}
                )
            )
    return "\n".join(lines) + "\n"


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "source", "dataset", "split", "record_id", "field", "detail"],
        )
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_record_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "field",
        "bytes",
        "chars",
        "line_count",
        "max_line_bytes",
        "control_chars",
        "nul_count",
        "replacement_char_count",
        "choice_label_prefix",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_junit(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_text_audit",
            "tests": "1",
            "failures": str(len(failures)),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "text_quality"})
    for finding in failures:
        failure = ET.SubElement(case, "failure", {"message": f"{finding['kind']}: {finding['field']}"})
        failure.text = json.dumps(finding, ensure_ascii=False, sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local eval JSONL input; repeatable")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary output")
    parser.add_argument("--csv", type=Path, help="Optional findings CSV output")
    parser.add_argument("--record-csv", type=Path, help="Optional per-field telemetry CSV output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail prompts larger than this UTF-8 byte count")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail choices larger than this UTF-8 byte count")
    parser.add_argument("--max-line-bytes", type=int, help="Fail any prompt/choice line larger than this UTF-8 byte count")
    parser.add_argument("--fail-on-control-chars", action="store_true", help="Treat disallowed C0 controls as errors")
    parser.add_argument("--fail-on-replacement-chars", action="store_true", help="Treat U+FFFD replacement chars as errors")
    parser.add_argument("--fail-on-blank-text", action="store_true", help="Fail blank prompts or choices")
    parser.add_argument("--fail-on-choice-label-prefixes", action="store_true", help="Fail choices that retain raw labels such as 'A.' or '1)'")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit nonzero if any warning or error findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    audit_records(
        records,
        findings,
        max_prompt_bytes=args.max_prompt_bytes,
        max_choice_bytes=args.max_choice_bytes,
        max_line_bytes=args.max_line_bytes,
        fail_on_control_chars=args.fail_on_control_chars,
        fail_on_replacement_chars=args.fail_on_replacement_chars,
        fail_on_blank_text=args.fail_on_blank_text,
        fail_on_choice_label_prefixes=args.fail_on_choice_label_prefixes,
    )
    record_rows = build_record_rows(records)
    report = build_report(inputs, records, record_rows, findings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_findings_csv(report, args.csv)
    if args.record_csv:
        write_record_csv(record_rows, args.record_csv)
    if args.junit:
        write_junit(report, args.junit)

    if report["error_count"]:
        return 1
    if args.fail_on_findings and report["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
