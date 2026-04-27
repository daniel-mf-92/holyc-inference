#!/usr/bin/env python3
"""Audit local eval JSONL for split leakage before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then checks for record-id collisions, prompt reuse across
splits, and repeated prompt+choice payloads across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
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
class LeakFinding:
    kind: str
    severity: str
    dataset: str
    splits: list[str]
    key_sha256: str
    record_ids: list[str]
    sources: list[str]
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_text_key(text: str) -> str:
    normalized = dataset_pack.clean_text(text).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def key_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_key(record: dataset_pack.EvalRecord) -> str:
    return key_digest(stable_text_key(record.prompt))


def payload_key(record: dataset_pack.EvalRecord) -> str:
    return key_digest(
        {
            "choices": [stable_text_key(choice) for choice in record.choices],
            "prompt": stable_text_key(record.prompt),
        }
    )


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def load_jsonl_records(paths: Iterable[Path], default_dataset: str, default_split: str) -> list[LoadedRecord]:
    loaded: list[LoadedRecord] = []
    for path in paths:
        rows = dataset_pack.read_jsonl(path)
        for index, row in enumerate(rows):
            record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            loaded.append(LoadedRecord(source=str(path), row_number=index + 1, record=record))
    return loaded


def collect_by(records: Iterable[LoadedRecord], key_fn) -> dict[tuple[str, str], list[LoadedRecord]]:
    groups: dict[tuple[str, str], list[LoadedRecord]] = {}
    for loaded in records:
        groups.setdefault((loaded.record.dataset, key_fn(loaded.record)), []).append(loaded)
    return groups


def make_finding(kind: str, severity: str, dataset: str, key: str, group: list[LoadedRecord], detail: str) -> LeakFinding:
    return LeakFinding(
        kind=kind,
        severity=severity,
        dataset=dataset,
        splits=sorted({loaded.record.split for loaded in group}),
        key_sha256=key,
        record_ids=sorted({loaded.record.record_id for loaded in group}),
        sources=sorted(source_ref(loaded) for loaded in group),
        detail=detail,
    )


def audit_records(records: list[LoadedRecord]) -> list[LeakFinding]:
    findings: list[LeakFinding] = []

    for (dataset, record_id), group in sorted(collect_by(records, lambda record: record.record_id).items()):
        if len(group) > 1:
            splits = {loaded.record.split for loaded in group}
            severity = "error" if len(splits) > 1 else "warning"
            findings.append(
                make_finding(
                    "duplicate_record_id",
                    severity,
                    dataset,
                    key_digest(record_id),
                    group,
                    "record id appears more than once",
                )
            )

    for (dataset, key), group in sorted(collect_by(records, prompt_key).items()):
        splits = {loaded.record.split for loaded in group}
        if len(splits) > 1:
            findings.append(
                make_finding(
                    "prompt_split_leak",
                    "error",
                    dataset,
                    key,
                    group,
                    "normalized prompt appears in multiple splits",
                )
            )

    for (dataset, key), group in sorted(collect_by(records, payload_key).items()):
        splits = {loaded.record.split for loaded in group}
        if len(splits) > 1:
            findings.append(
                make_finding(
                    "payload_split_leak",
                    "error",
                    dataset,
                    key,
                    group,
                    "normalized prompt and choices appear in multiple splits",
                )
            )
        answers = {loaded.record.answer_index for loaded in group}
        if len(group) > 1 and len(answers) > 1:
            findings.append(
                make_finding(
                    "answer_conflict",
                    "error",
                    dataset,
                    key,
                    group,
                    "same normalized prompt and choices use conflicting answer indexes",
                )
            )

    return findings


def counts_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        dataset_counts = counts.setdefault(loaded.record.dataset, {})
        split = loaded.record.split
        dataset_counts[split] = dataset_counts.get(split, 0) + 1
    return {dataset: dict(sorted(splits.items())) for dataset, splits in sorted(counts.items())}


def build_report(inputs: list[Path], records: list[LoadedRecord], findings: list[LeakFinding]) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-leak-audit",
        "inputs": [str(path) for path in inputs],
        "record_count": len(records),
        "counts_by_dataset_split": counts_by_dataset_split(records),
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Leak Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
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
            lines.append(f"| {dataset} | {split} | {count} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No split leakage findings.")
    else:
        lines.extend(["| severity | kind | dataset | splits | records | detail |", "| --- | --- | --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {dataset} | {splits} | {records} | {detail} |".format(
                    severity=finding["severity"],
                    kind=finding["kind"],
                    dataset=finding["dataset"],
                    splits=", ".join(finding["splits"]),
                    records=", ".join(finding["record_ids"]),
                    detail=finding["detail"],
                )
            )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "dataset", "splits", "key_sha256", "record_ids", "sources", "detail"],
        )
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(
                {
                    "severity": finding["severity"],
                    "kind": finding["kind"],
                    "dataset": finding["dataset"],
                    "splits": ",".join(finding["splits"]),
                    "key_sha256": finding["key_sha256"],
                    "record_ids": ",".join(finding["record_ids"]),
                    "sources": ",".join(finding["sources"]),
                    "detail": finding["detail"],
                }
            )


def write_outputs(report: dict[str, Any], output: Path, markdown: Path | None, csv_path: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(report), encoding="utf-8")
    if csv_path:
        write_csv(report, csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Local eval JSONL input; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report output")
    parser.add_argument("--csv", type=Path, help="Optional CSV findings output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument("--fail-on-leaks", action="store_true", help="Exit non-zero when error-level leaks are found")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_jsonl_records(args.input, args.default_dataset, args.default_split)
        if not records:
            raise ValueError("no records loaded")
        findings = audit_records(records)
        report = build_report(args.input, records, findings)
        write_outputs(report, args.output, args.markdown, args.csv)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_report={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    print(f"status={report['status']}")
    print(f"records={report['record_count']}")
    print(f"findings={len(report['findings'])}")
    if args.fail_on_leaks and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
