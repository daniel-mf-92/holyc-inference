#!/usr/bin/env python3
"""Audit local eval JSONL dataset/split mix before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py and reports whether a curated eval suite is dominated by one
dataset, split, or dataset/split bucket.
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
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class MixFinding:
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


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[MixFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[MixFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(MixFinding("error", "read_error", str(path), str(exc)))
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
                    MixFinding(
                        "error",
                        "schema_error",
                        f"{path}:{index + 1}",
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def dataset_split_counts(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(splits.items())) for dataset, splits in sorted(counts.items())}


def pct(count: int, total: int) -> float | None:
    return (count / total * 100.0) if total else None


def distribution_rows(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    total = len(records)
    dataset_counts = sorted_counts(loaded.record.dataset for loaded in records)
    split_counts = sorted_counts(loaded.record.split for loaded in records)
    rows: list[dict[str, Any]] = []

    for dataset, count in dataset_counts.items():
        rows.append(
            {
                "scope": "dataset",
                "dataset": dataset,
                "split": "",
                "records": count,
                "pct_of_total": pct(count, total),
            }
        )
    for split, count in split_counts.items():
        rows.append(
            {
                "scope": "split",
                "dataset": "",
                "split": split,
                "records": count,
                "pct_of_total": pct(count, total),
            }
        )
    for dataset, splits in dataset_split_counts(records).items():
        for split, count in splits.items():
            rows.append(
                {
                    "scope": "dataset_split",
                    "dataset": dataset,
                    "split": split,
                    "records": count,
                    "pct_of_total": pct(count, total),
                }
            )
    return rows


def record_rows(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loaded in records:
        record = loaded.record
        choice_bytes = [len(choice.encode("utf-8")) for choice in record.choices]
        payload = {
            "dataset": record.dataset,
            "split": record.split,
            "prompt": record.prompt,
            "choices": record.choices,
            "answer_index": record.answer_index,
        }
        payload_sha256 = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "source": loaded.source,
                "row_number": loaded.row_number,
                "record_id": record.record_id,
                "dataset": record.dataset,
                "split": record.split,
                "dataset_split": f"{record.dataset}:{record.split}",
                "choice_count": len(record.choices),
                "answer_index": record.answer_index,
                "prompt_bytes": len(record.prompt.encode("utf-8")),
                "choice_bytes_total": sum(choice_bytes),
                "choice_bytes_max": max(choice_bytes) if choice_bytes else 0,
                "provenance": record.provenance,
                "normalized_payload_sha256": payload_sha256,
            }
        )
    return rows


def add_min_count_findings(
    findings: list[MixFinding],
    rows: list[dict[str, Any]],
    scope: str,
    minimum: int | None,
) -> None:
    if minimum is None:
        return
    for row in rows:
        if row["scope"] != scope:
            continue
        if row["records"] < minimum:
            label = scope_label(row)
            findings.append(
                MixFinding(
                    "error",
                    f"min_{scope}_records",
                    label,
                    f"{label} has {row['records']} records, below {minimum}",
                )
            )


def add_max_pct_findings(
    findings: list[MixFinding],
    rows: list[dict[str, Any]],
    scope: str,
    maximum_pct: float | None,
) -> None:
    if maximum_pct is None:
        return
    for row in rows:
        if row["scope"] != scope:
            continue
        share = row["pct_of_total"]
        if share is not None and share > maximum_pct:
            label = scope_label(row)
            findings.append(
                MixFinding(
                    "error",
                    f"max_{scope}_pct",
                    label,
                    f"{label} is {share:.2f}% of records, above {maximum_pct:.2f}%",
                )
            )


def scope_label(row: dict[str, Any]) -> str:
    if row["scope"] == "dataset":
        return row["dataset"]
    if row["scope"] == "split":
        return row["split"]
    return f"{row['dataset']}:{row['split']}"


def add_required_findings(
    findings: list[MixFinding],
    records: list[LoadedRecord],
    required_datasets: list[str],
    required_splits: list[str],
    required_dataset_splits: list[str],
) -> None:
    datasets = {loaded.record.dataset for loaded in records}
    splits = {loaded.record.split for loaded in records}
    pairs = {f"{loaded.record.dataset}:{loaded.record.split}" for loaded in records}

    for dataset in sorted(set(required_datasets)):
        if dataset not in datasets:
            findings.append(MixFinding("error", "missing_dataset", dataset, "required dataset is absent"))
    for split in sorted(set(required_splits)):
        if split not in splits:
            findings.append(MixFinding("error", "missing_split", split, "required split is absent"))
    for pair in sorted(set(required_dataset_splits)):
        if pair not in pairs:
            findings.append(MixFinding("error", "missing_dataset_split", pair, "required dataset/split is absent"))


def add_cardinality_findings(
    findings: list[MixFinding],
    records: list[LoadedRecord],
    min_datasets: int | None,
    min_splits: int | None,
) -> None:
    dataset_count = len({loaded.record.dataset for loaded in records})
    split_count = len({loaded.record.split for loaded in records})
    if min_datasets is not None and dataset_count < min_datasets:
        findings.append(
            MixFinding("error", "min_datasets", "overall", f"{dataset_count} datasets found, below {min_datasets}")
        )
    if min_splits is not None and split_count < min_splits:
        findings.append(MixFinding("error", "min_splits", "overall", f"{split_count} splits found, below {min_splits}"))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    rows = distribution_rows(records)

    add_cardinality_findings(findings, records, args.min_datasets, args.min_splits)
    add_required_findings(findings, records, args.require_dataset, args.require_split, args.require_dataset_split)
    add_min_count_findings(findings, rows, "dataset", args.min_records_per_dataset)
    add_min_count_findings(findings, rows, "split", args.min_records_per_split)
    add_min_count_findings(findings, rows, "dataset_split", args.min_records_per_dataset_split)
    add_max_pct_findings(findings, rows, "dataset", args.max_dataset_pct)
    add_max_pct_findings(findings, rows, "split", args.max_split_pct)
    add_max_pct_findings(findings, rows, "dataset_split", args.max_dataset_split_pct)

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-mix-audit",
        "inputs": inputs,
        "record_count": len(records),
        "dataset_count": len({loaded.record.dataset for loaded in records}),
        "split_count": len({loaded.record.split for loaded in records}),
        "dataset_counts": sorted_counts(loaded.record.dataset for loaded in records),
        "split_counts": sorted_counts(loaded.record.split for loaded in records),
        "dataset_split_counts": dataset_split_counts(records),
        "distribution": rows,
        "record_telemetry": record_rows(records),
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Mix Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Datasets: {report['dataset_count']}",
        f"- Splits: {report['split_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Distribution",
        "",
        "| scope | dataset | split | records | pct of total |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in report["distribution"]:
        share = row["pct_of_total"]
        share_text = "" if share is None else f"{share:.2f}"
        lines.append(f"| {row['scope']} | {row['dataset']} | {row['split']} | {row['records']} | {share_text} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No dataset mix findings.")
    else:
        lines.extend(["| severity | kind | scope | detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['scope']} | {finding['detail']} |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "dataset", "split", "records", "pct_of_total"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["severity", "kind", "scope", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(findings)


def write_record_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "row_number",
        "record_id",
        "dataset",
        "split",
        "dataset_split",
        "choice_count",
        "answer_index",
        "prompt_bytes",
        "choice_bytes_total",
        "choice_bytes_max",
        "provenance",
        "normalized_payload_sha256",
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
            "name": "dataset_mix_audit",
            "tests": "1",
            "failures": str(report["error_count"]),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "bench.dataset_mix_audit", "name": "dataset_mix"})
    if report["error_count"]:
        failure = ET.SubElement(case, "failure", {"message": f"{report['error_count']} dataset mix errors"})
        failure.text = "\n".join(f"{item['kind']} {item['scope']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local JSONL eval input.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--csv", type=Path, help="Optional distribution CSV path.")
    parser.add_argument("--record-csv", type=Path, help="Optional per-record dataset mix telemetry CSV path.")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path.")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path.")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata.")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata.")
    parser.add_argument("--min-datasets", type=int, help="Fail unless at least this many datasets are present.")
    parser.add_argument("--min-splits", type=int, help="Fail unless at least this many splits are present.")
    parser.add_argument("--require-dataset", action="append", default=[], help="Dataset name that must be present.")
    parser.add_argument("--require-split", action="append", default=[], help="Split name that must be present.")
    parser.add_argument(
        "--require-dataset-split",
        action="append",
        default=[],
        help="Dataset/split pair that must be present, formatted DATASET:SPLIT.",
    )
    parser.add_argument("--min-records-per-dataset", type=int, help="Minimum records for every present dataset.")
    parser.add_argument("--min-records-per-split", type=int, help="Minimum records for every present split.")
    parser.add_argument("--min-records-per-dataset-split", type=int, help="Minimum records for every present pair.")
    parser.add_argument("--max-dataset-pct", type=float, help="Maximum percent any dataset may contribute.")
    parser.add_argument("--max-split-pct", type=float, help="Maximum percent any split may contribute.")
    parser.add_argument("--max-dataset-split-pct", type=float, help="Maximum percent any dataset/split may contribute.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_csv(args.csv, report["distribution"])
    if args.record_csv:
        write_record_csv(args.record_csv, report["record_telemetry"])
    if args.findings_csv:
        write_findings_csv(args.findings_csv, report["findings"])
    if args.junit:
        write_junit(args.junit, report)

    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
