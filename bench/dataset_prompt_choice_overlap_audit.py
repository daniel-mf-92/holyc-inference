#!/usr/bin/env python3
"""Audit local eval JSONL for prompt/choice text overlap before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then flags prompts that already contain normalized choice text.
This catches accidental prompt templating leaks before HCEval binaries are
loaded by the air-gapped TempleOS guest.
"""

from __future__ import annotations

import argparse
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


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class OverlapFinding:
    severity: str
    kind: str
    source: str
    dataset: str
    split: str
    record_id: str
    choice_index: int
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_for_match(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[OverlapFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[OverlapFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(OverlapFinding("error", "read_error", str(path), "", "", "", -1, str(exc)))
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
                    OverlapFinding(
                        "error",
                        "schema_error",
                        f"{path}:{index + 1}",
                        "",
                        "",
                        "",
                        -1,
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def choice_appears_in_prompt(prompt: str, choice: str, min_choice_chars: int) -> bool:
    normalized_choice = normalize_for_match(choice)
    if len(normalized_choice) < min_choice_chars:
        return False
    return normalized_choice in normalize_for_match(prompt)


def audit_records(
    records: list[LoadedRecord],
    findings: list[OverlapFinding],
    *,
    min_choice_chars: int,
    fail_on_any_overlap: bool,
    fail_on_answer_overlap: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loaded in records:
        record = loaded.record
        overlapped_indexes: list[int] = []
        for index, choice in enumerate(record.choices):
            if not choice_appears_in_prompt(record.prompt, choice, min_choice_chars):
                continue
            overlapped_indexes.append(index)
            is_answer = index == record.answer_index
            severity = "error" if fail_on_any_overlap or (is_answer and fail_on_answer_overlap) else "warning"
            kind = "answer_choice_overlap" if is_answer else "distractor_choice_overlap"
            findings.append(
                OverlapFinding(
                    severity,
                    kind,
                    source_ref(loaded),
                    record.dataset,
                    record.split,
                    record.record_id,
                    index,
                    "choice text appears in prompt after case/whitespace normalization; "
                    f"choice_chars={len(normalize_for_match(choice))}",
                )
            )

        rows.append(
            {
                "source": source_ref(loaded),
                "dataset": record.dataset,
                "split": record.split,
                "record_id": record.record_id,
                "choice_count": len(record.choices),
                "answer_index": record.answer_index,
                "overlap_count": len(overlapped_indexes),
                "answer_overlaps_prompt": record.answer_index in overlapped_indexes,
                "overlap_choice_indexes": ",".join(str(index) for index in overlapped_indexes),
            }
        )
    return rows


def dataset_split_counts(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(splits.items())) for dataset, splits in sorted(counts.items())}


def build_report(
    records: list[LoadedRecord],
    inputs: list[dict[str, Any]],
    findings: list[OverlapFinding],
    record_rows: list[dict[str, Any]],
    min_choice_chars: int,
) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    overlap_records = sum(1 for row in record_rows if row["overlap_count"])
    answer_overlap_records = sum(1 for row in record_rows if row["answer_overlaps_prompt"])
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "input_files": inputs,
        "record_count": len(records),
        "counts_by_dataset_split": dataset_split_counts(records),
        "min_choice_chars": min_choice_chars,
        "overlap_record_count": overlap_records,
        "overlap_record_pct": (overlap_records / len(records) * 100.0) if records else None,
        "answer_overlap_record_count": answer_overlap_records,
        "answer_overlap_record_pct": (answer_overlap_records / len(records) * 100.0) if records else None,
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Prompt/Choice Overlap Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Records with any overlap: {report['overlap_record_count']}",
        f"- Records with answer overlap: {report['answer_overlap_record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Findings",
    ]
    if not report["findings"]:
        lines.append("")
        lines.append("No prompt/choice overlap findings.")
    else:
        for finding in report["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['source']} "
                f"{finding['record_id']} choice={finding['choice_index']}: {finding['detail']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_findings_csv(path: Path, findings: list[OverlapFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(OverlapFinding("", "", "", "", "", "", 0, "")).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_record_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "choice_count",
        "answer_index",
        "overlap_count",
        "answer_overlaps_prompt",
        "overlap_choice_indexes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_prompt_choice_overlap_audit",
            "tests": "1",
            "failures": "1" if report["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "prompt_choice_overlap"})
    if report["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": "prompt/choice overlap audit failed"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Local eval JSONL input; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--markdown", type=Path, help="Markdown report path.")
    parser.add_argument("--csv", type=Path, help="CSV findings path.")
    parser.add_argument("--record-csv", type=Path, help="Per-record overlap telemetry CSV path.")
    parser.add_argument("--junit", type=Path, help="JUnit XML path.")
    parser.add_argument("--dataset", default="eval", help="Default dataset name for normalized rows without one.")
    parser.add_argument("--split", default="validation", help="Default split name for normalized rows without one.")
    parser.add_argument(
        "--min-choice-chars",
        type=int,
        default=4,
        help="Ignore choices shorter than this normalized length.",
    )
    parser.add_argument(
        "--fail-on-any-overlap",
        action="store_true",
        help="Fail when any choice appears in its prompt.",
    )
    parser.add_argument(
        "--fail-on-answer-overlap",
        action="store_true",
        help="Fail when the correct choice appears in its prompt.",
    )
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Fail when any warning or error finding is present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    record_rows = audit_records(
        records,
        findings,
        min_choice_chars=args.min_choice_chars,
        fail_on_any_overlap=args.fail_on_any_overlap,
        fail_on_answer_overlap=args.fail_on_answer_overlap,
    )
    if args.fail_on_findings:
        findings = [
            OverlapFinding(
                "error" if item.severity == "warning" else item.severity,
                item.kind,
                item.source,
                item.dataset,
                item.split,
                item.record_id,
                item.choice_index,
                item.detail,
            )
            for item in findings
        ]
    report = build_report(records, inputs, findings, record_rows, args.min_choice_chars)
    write_json(args.output, report)
    if args.markdown:
        write_markdown(args.markdown, report)
    if args.csv:
        write_findings_csv(args.csv, findings)
    if args.record_csv:
        write_record_csv(args.record_csv, record_rows)
    if args.junit:
        write_junit(args.junit, report)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
