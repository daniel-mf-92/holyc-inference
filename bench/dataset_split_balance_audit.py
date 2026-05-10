#!/usr/bin/env python3
"""Audit eval dataset split coverage and balance before packing.

This offline host-side tool reads the JSONL row shapes accepted by
dataset_pack.py, normalizes dataset/split metadata, and emits sidecars for CI
gates that catch missing splits or one split dominating a curated subset.
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
class SplitSummary:
    dataset: str
    split: str
    records: int
    pct_of_dataset: float
    pct_of_total: float


@dataclass(frozen=True)
class DatasetSummary:
    dataset: str
    records: int
    split_count: int
    splits: str
    largest_split_pct: float
    smallest_split_pct: float


@dataclass(frozen=True)
class Finding:
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
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[Finding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[Finding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(Finding("error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(Finding("error", "schema_error", f"{path}:{index + 1}", str(exc)))
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def count_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for loaded in records:
        counts[loaded.record.dataset][loaded.record.split] += 1
    return {
        dataset: dict(sorted(split_counts.items()))
        for dataset, split_counts in sorted(counts.items())
    }


def split_summaries(records: list[LoadedRecord]) -> tuple[list[SplitSummary], list[DatasetSummary]]:
    counts = count_by_dataset_split(records)
    total = sum(sum(split_counts.values()) for split_counts in counts.values())
    split_rows: list[SplitSummary] = []
    dataset_rows: list[DatasetSummary] = []

    for dataset, split_counts in counts.items():
        dataset_total = sum(split_counts.values())
        split_pcts: list[float] = []
        for split, count in split_counts.items():
            pct_of_dataset = round(count / dataset_total * 100.0, 6) if dataset_total else 0.0
            pct_of_total = round(count / total * 100.0, 6) if total else 0.0
            split_pcts.append(pct_of_dataset)
            split_rows.append(SplitSummary(dataset, split, count, pct_of_dataset, pct_of_total))
        dataset_rows.append(
            DatasetSummary(
                dataset=dataset,
                records=dataset_total,
                split_count=len(split_counts),
                splits=",".join(split_counts),
                largest_split_pct=max(split_pcts) if split_pcts else 0.0,
                smallest_split_pct=min(split_pcts) if split_pcts else 0.0,
            )
        )

    return split_rows, dataset_rows


def parse_dataset_split(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("dataset/split requirements must use DATASET:SPLIT")
    dataset, split = value.split(":", 1)
    if not dataset or not split:
        raise argparse.ArgumentTypeError("dataset and split must be non-empty")
    return dataset, split


def add_gate_findings(
    findings: list[Finding],
    split_rows: list[SplitSummary],
    dataset_rows: list[DatasetSummary],
    *,
    min_records: int,
    min_records_per_dataset_split: int,
    min_splits_per_dataset: int,
    max_largest_split_pct: float | None,
    required_splits: list[str],
    required_dataset_splits: list[tuple[str, str]],
) -> None:
    total_records = sum(row.records for row in split_rows)
    if total_records < min_records:
        findings.append(Finding("error", "min_records", "all", f"{total_records} records, expected at least {min_records}"))

    observed_splits = {row.split for row in split_rows}
    for split in required_splits:
        if split not in observed_splits:
            findings.append(Finding("error", "required_split_missing", split, "split has no records"))

    observed_dataset_splits = {(row.dataset, row.split) for row in split_rows}
    for dataset, split in required_dataset_splits:
        if (dataset, split) not in observed_dataset_splits:
            findings.append(
                Finding("error", "required_dataset_split_missing", f"{dataset}:{split}", "dataset/split has no records")
            )

    for row in split_rows:
        if row.records < min_records_per_dataset_split:
            findings.append(
                Finding(
                    "error",
                    "min_records_per_dataset_split",
                    f"{row.dataset}:{row.split}",
                    f"{row.records} records, expected at least {min_records_per_dataset_split}",
                )
            )

    for row in dataset_rows:
        if row.split_count < min_splits_per_dataset:
            findings.append(
                Finding(
                    "error",
                    "min_splits_per_dataset",
                    row.dataset,
                    f"{row.split_count} splits, expected at least {min_splits_per_dataset}",
                )
            )
        if max_largest_split_pct is not None and row.largest_split_pct > max_largest_split_pct:
            findings.append(
                Finding(
                    "error",
                    "largest_split_pct",
                    row.dataset,
                    f"largest split contains {row.largest_split_pct:.2f}% of records, above {max_largest_split_pct:.2f}%",
                )
            )


def build_report(
    inputs: list[dict[str, Any]],
    split_rows: list[SplitSummary],
    dataset_rows: list[DatasetSummary],
    findings: list[Finding],
) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-split-balance-audit",
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "inputs": inputs,
        "summary": {
            "records": sum(row.records for row in split_rows),
            "datasets": len(dataset_rows),
            "dataset_splits": len(split_rows),
            "findings": len(findings),
        },
        "datasets": [asdict(row) for row in dataset_rows],
        "splits": [asdict(row) for row in split_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    write_csv(path, findings, list(Finding.__dataclass_fields__))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Split Balance Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {report['summary']['records']}",
        f"Datasets: {report['summary']['datasets']}",
        f"Dataset/splits: {report['summary']['dataset_splits']}",
        f"Findings: {report['summary']['findings']}",
        "",
        "## Datasets",
        "",
        "| Dataset | Records | Splits | Largest split pct | Smallest split pct |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for row in report["datasets"]:
        lines.append(
            "| {dataset} | {records} | {splits} | {largest_split_pct} | {smallest_split_pct} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Dataset Splits",
            "",
            "| Dataset | Split | Records | Pct of dataset | Pct of total |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["splits"]:
        lines.append(
            "| {dataset} | {split} | {records} | {pct_of_dataset} | {pct_of_total} |".format(**row)
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(["| Severity | Kind | Scope | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append("| {severity} | {kind} | {scope} | {detail} |".format(**finding))
    else:
        lines.append("No dataset split-balance findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = [finding for finding in findings if finding.severity == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_split_balance_audit",
            "tests": "1",
            "failures": "1" if errors else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "split_balance"})
    if errors:
        failure = ET.SubElement(case, "failure", {"message": f"{len(errors)} split-balance finding(s)"})
        failure.text = "\n".join(f"{finding.scope}: {finding.kind}: {finding.detail}" for finding in errors)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Input JSONL file; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="dataset_split_balance_audit_latest")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-records-per-dataset-split", type=int, default=1)
    parser.add_argument("--min-splits-per-dataset", type=int, default=1)
    parser.add_argument("--max-largest-split-pct", type=float)
    parser.add_argument("--require-split", action="append", default=[], help="Require at least one row with this split")
    parser.add_argument(
        "--require-dataset-split",
        action="append",
        default=[],
        type=parse_dataset_split,
        help="Require a DATASET:SPLIT bucket",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded, inputs, findings = load_records(args.input, args.default_dataset, args.default_split)
    splits, datasets = split_summaries(loaded)
    add_gate_findings(
        findings,
        splits,
        datasets,
        min_records=args.min_records,
        min_records_per_dataset_split=args.min_records_per_dataset_split,
        min_splits_per_dataset=args.min_splits_per_dataset,
        max_largest_split_pct=args.max_largest_split_pct,
        required_splits=args.require_split,
        required_dataset_splits=args.require_dataset_split,
    )
    report = build_report(inputs, splits, datasets, findings)

    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", splits, list(SplitSummary.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_datasets.csv", datasets, list(DatasetSummary.__dataclass_fields__))
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    (args.output_dir / f"{stem}.md").write_text(markdown_report(report), encoding="utf-8")
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
