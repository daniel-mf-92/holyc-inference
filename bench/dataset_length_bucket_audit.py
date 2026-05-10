#!/usr/bin/env python3
"""Audit normalized eval dataset length-bucket coverage before packing.

This offline host-side curation tool reads the same JSONL row shapes accepted
by dataset_pack.py, buckets records by prompt+choice bytes, and writes coverage
sidecars. It does not launch QEMU, open sockets, or touch the TempleOS guest.
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


DEFAULT_BUCKET_EDGES = "128,256,512,1024,2048"


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class RecordBucket:
    source: str
    dataset: str
    split: str
    record_id: str
    bucket: str
    min_total_bytes: int
    max_total_bytes: int
    prompt_bytes: int
    choice_bytes: int
    total_bytes: int
    choice_count: int
    answer_index: int


@dataclass(frozen=True)
class BucketSummary:
    bucket: str
    min_total_bytes: int
    max_total_bytes: int
    record_count: int
    dataset_split_count: int
    datasets: str
    choice_count_histogram: str
    answer_index_histogram: str


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


def parse_bucket_edges(text: str) -> list[int]:
    edges: list[int] = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            edge = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"bucket edge {stripped!r} is not an integer") from exc
        if edge <= 0:
            raise argparse.ArgumentTypeError("bucket edges must be positive")
        edges.append(edge)
    if sorted(set(edges)) != edges:
        raise argparse.ArgumentTypeError("bucket edges must be strictly increasing")
    return edges


def bucket_for_size(size: int, edges: list[int]) -> tuple[str, int, int]:
    lower = 0
    for edge in edges:
        if size <= edge:
            return f"{lower}-{edge}", lower, edge
        lower = edge + 1
    return f"{lower}+", lower, -1


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


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def record_bucket(loaded: LoadedRecord, edges: list[int]) -> RecordBucket:
    record = loaded.record
    prompt_bytes = len(record.prompt.encode("utf-8"))
    choice_bytes = sum(len(choice.encode("utf-8")) for choice in record.choices)
    total_bytes = prompt_bytes + choice_bytes
    bucket, min_total, max_total = bucket_for_size(total_bytes, edges)
    return RecordBucket(
        source=source_ref(loaded),
        dataset=record.dataset,
        split=record.split,
        record_id=record.record_id,
        bucket=bucket,
        min_total_bytes=min_total,
        max_total_bytes=max_total,
        prompt_bytes=prompt_bytes,
        choice_bytes=choice_bytes,
        total_bytes=total_bytes,
        choice_count=len(record.choices),
        answer_index=record.answer_index,
    )


def histogram_text(values: Iterable[int]) -> str:
    counts = collections.Counter(values)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def summarize_bucket(bucket: str, rows: list[RecordBucket]) -> BucketSummary:
    datasets = sorted({f"{row.dataset}:{row.split}" for row in rows})
    return BucketSummary(
        bucket=bucket,
        min_total_bytes=rows[0].min_total_bytes,
        max_total_bytes=rows[0].max_total_bytes,
        record_count=len(rows),
        dataset_split_count=len(datasets),
        datasets=",".join(datasets),
        choice_count_histogram=histogram_text(row.choice_count for row in rows),
        answer_index_histogram=histogram_text(row.answer_index for row in rows),
    )


def bucket_summaries(rows: list[RecordBucket]) -> list[BucketSummary]:
    grouped: dict[str, list[RecordBucket]] = collections.defaultdict(list)
    for row in rows:
        grouped[row.bucket].append(row)
    return [
        summarize_bucket(bucket, grouped[bucket])
        for bucket in sorted(grouped, key=lambda key: (grouped[key][0].min_total_bytes, key))
    ]


def add_gate_findings(
    findings: list[Finding],
    rows: list[RecordBucket],
    summaries: list[BucketSummary],
    *,
    min_records: int,
    min_covered_buckets: int,
    max_largest_bucket_pct: float | None,
    required_buckets: list[str],
) -> None:
    if len(rows) < min_records:
        findings.append(Finding("error", "min_records", "all", f"{len(rows)} records, expected at least {min_records}"))

    covered = {summary.bucket for summary in summaries}
    if len(covered) < min_covered_buckets:
        findings.append(
            Finding(
                "error",
                "min_covered_buckets",
                "all",
                f"{len(covered)} buckets covered, expected at least {min_covered_buckets}",
            )
        )

    for bucket in required_buckets:
        if bucket not in covered:
            findings.append(Finding("error", "required_bucket_missing", bucket, "bucket has no records"))

    if max_largest_bucket_pct is not None and rows:
        largest = max(summary.record_count for summary in summaries)
        largest_pct = largest / len(rows) * 100.0
        if largest_pct > max_largest_bucket_pct:
            findings.append(
                Finding(
                    "error",
                    "largest_bucket_pct",
                    "all",
                    f"largest bucket contains {largest_pct:.2f}% of records, above {max_largest_bucket_pct:.2f}%",
                )
            )


def build_report(
    inputs: list[dict[str, Any]],
    rows: list[RecordBucket],
    summaries: list[BucketSummary],
    findings: list[Finding],
    edges: list[int],
) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-length-bucket-audit",
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "bucket_edges": edges,
        "inputs": inputs,
        "summary": {
            "records": len(rows),
            "buckets": len(summaries),
            "findings": len(findings),
            "largest_bucket_pct": round(max((row.record_count for row in summaries), default=0) / len(rows) * 100.0, 6)
            if rows
            else None,
        },
        "buckets": [asdict(summary) for summary in summaries],
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
        "# Dataset Length Bucket Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {report['summary']['records']}",
        f"Buckets: {report['summary']['buckets']}",
        f"Largest bucket pct: {report['summary']['largest_bucket_pct']}",
        "",
        "## Buckets",
        "",
        "| Bucket | Records | Dataset splits | Choice counts | Answer indexes |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in report["buckets"]:
        lines.append(
            "| {bucket} | {record_count} | {dataset_split_count} | {choice_count_histogram} | {answer_index_histogram} |".format(
                **row
            )
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(["| Severity | Kind | Scope | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append("| {severity} | {kind} | {scope} | {detail} |".format(**finding))
    else:
        lines.append("No dataset length-bucket findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = [finding for finding in findings if finding.severity == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_length_bucket_audit",
            "tests": "1",
            "failures": "1" if errors else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "length_bucket_coverage"})
    if errors:
        failure = ET.SubElement(case, "failure", {"message": f"{len(errors)} length-bucket finding(s)"})
        failure.text = "\n".join(f"{finding.scope}: {finding.kind}: {finding.detail}" for finding in errors)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Input JSONL file; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="dataset_length_bucket_audit_latest")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--bucket-edges", type=parse_bucket_edges, default=parse_bucket_edges(DEFAULT_BUCKET_EDGES))
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-covered-buckets", type=int, default=1)
    parser.add_argument("--max-largest-bucket-pct", type=float)
    parser.add_argument("--require-bucket", action="append", default=[], help="Require a named bucket such as 0-128 or 2049+")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded, inputs, findings = load_records(args.input, args.default_dataset, args.default_split)
    rows = [record_bucket(record, args.bucket_edges) for record in loaded]
    summaries = bucket_summaries(rows)
    add_gate_findings(
        findings,
        rows,
        summaries,
        min_records=args.min_records,
        min_covered_buckets=args.min_covered_buckets,
        max_largest_bucket_pct=args.max_largest_bucket_pct,
        required_buckets=args.require_bucket,
    )
    report = build_report(inputs, rows, summaries, findings, args.bucket_edges)

    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", summaries, list(BucketSummary.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_records.csv", rows, list(RecordBucket.__dataclass_fields__))
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    (args.output_dir / f"{stem}.md").write_text(markdown_report(report), encoding="utf-8")
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
