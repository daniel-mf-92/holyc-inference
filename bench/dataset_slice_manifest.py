#!/usr/bin/env python3
"""Emit deterministic dataset/split manifests for local eval JSONL files.

This host-side, offline-only helper normalizes the same row shapes accepted by
dataset_pack.py and records per-slice coverage before curation, packing, or
HolyC-vs-llama.cpp comparisons. It never fetches datasets and never launches
QEMU.
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
class SliceRecord:
    source: str
    row_number: int
    record_id: str
    dataset: str
    split: str
    choice_count: int
    answer_index: int
    prompt_bytes: int
    choices_bytes: int
    payload_sha256: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_ref(path: Path, row_number: int) -> str:
    return f"{path}:{row_number}"


def append_finding(findings: list[Finding], severity: str, kind: str, source: str, detail: str) -> None:
    findings.append(Finding(severity=severity, kind=kind, source=source, detail=detail))


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def record_payload_sha256(record: dataset_pack.EvalRecord) -> str:
    return sha256_json(asdict(record))


def slice_record(path: Path, row_number: int, record: dataset_pack.EvalRecord) -> SliceRecord:
    return SliceRecord(
        source=str(path),
        row_number=row_number,
        record_id=record.record_id,
        dataset=record.dataset,
        split=record.split,
        choice_count=len(record.choices),
        answer_index=record.answer_index,
        prompt_bytes=len(record.prompt.encode("utf-8")),
        choices_bytes=sum(len(choice.encode("utf-8")) for choice in record.choices),
        payload_sha256=record_payload_sha256(record),
    )


def load_records(
    inputs: Iterable[Path],
    default_dataset: str,
    default_split: str,
    findings: list[Finding],
) -> tuple[list[SliceRecord], list[dict[str, Any]]]:
    records: list[SliceRecord] = []
    sources: list[dict[str, Any]] = []
    for path in inputs:
        try:
            rows = dataset_pack.read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            append_finding(findings, "error", "read_error", str(path), str(exc))
            sources.append({"path": str(path), "rows": 0})
            continue

        source_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            source_info["sha256"] = file_sha256(path)
        sources.append(source_info)

        for index, row in enumerate(rows):
            row_number = index + 1
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                append_finding(findings, "error", "schema_error", source_ref(path, row_number), str(exc))
                continue
            records.append(slice_record(path, row_number, record))
    return records, sources


def grouped_records(records: list[SliceRecord]) -> dict[tuple[str, str], list[SliceRecord]]:
    groups: dict[tuple[str, str], list[SliceRecord]] = collections.defaultdict(list)
    for record in records:
        groups[(record.dataset, record.split)].append(record)
    return dict(sorted(groups.items()))


def slice_manifest(groups: dict[tuple[str, str], list[SliceRecord]]) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    for (dataset, split), records in groups.items():
        ordered = sorted(records, key=lambda item: (item.record_id, item.source, item.row_number))
        record_refs = [
            {
                "record_id": item.record_id,
                "source": item.source,
                "row_number": item.row_number,
                "payload_sha256": item.payload_sha256,
            }
            for item in ordered
        ]
        slices.append(
            {
                "dataset": dataset,
                "split": split,
                "record_count": len(ordered),
                "choice_count_histogram": sorted_counts(item.choice_count for item in ordered),
                "answer_histogram": sorted_counts(item.answer_index for item in ordered),
                "prompt_bytes": sum(item.prompt_bytes for item in ordered),
                "choices_bytes": sum(item.choices_bytes for item in ordered),
                "slice_sha256": sha256_json(
                    {
                        "dataset": dataset,
                        "records": record_refs,
                        "split": split,
                    }
                ),
                "records": record_refs,
            }
        )
    return slices


def parse_required_slices(values: Iterable[str]) -> list[tuple[str, str]]:
    required: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"required slice {value!r} must use DATASET:SPLIT")
        dataset, split = value.split(":", 1)
        dataset = dataset.strip()
        split = split.strip()
        if not dataset or not split:
            raise ValueError(f"required slice {value!r} must use DATASET:SPLIT")
        required.append((dataset, split))
    return required


def apply_gates(
    findings: list[Finding],
    groups: dict[tuple[str, str], list[SliceRecord]],
    required_slices: list[tuple[str, str]],
    min_records_per_slice: int | None,
    min_total_slices: int | None,
) -> None:
    if min_total_slices is not None and len(groups) < min_total_slices:
        append_finding(
            findings,
            "error",
            "too_few_slices",
            "dataset_slice_manifest",
            f"{len(groups)} slices found; expected at least {min_total_slices}",
        )

    for dataset, split in required_slices:
        if (dataset, split) not in groups:
            append_finding(
                findings,
                "error",
                "missing_required_slice",
                f"{dataset}:{split}",
                "required dataset/split slice is absent",
            )

    if min_records_per_slice is None:
        return
    for (dataset, split), records in groups.items():
        if len(records) < min_records_per_slice:
            append_finding(
                findings,
                "error",
                "slice_too_small",
                f"{dataset}:{split}",
                f"{len(records)} records found; expected at least {min_records_per_slice}",
            )


def build_report(
    records: list[SliceRecord],
    sources: list[dict[str, Any]],
    findings: list[Finding],
    required_slices: list[tuple[str, str]],
    min_records_per_slice: int | None,
    min_total_slices: int | None,
) -> dict[str, Any]:
    groups = grouped_records(records)
    apply_gates(findings, groups, required_slices, min_records_per_slice, min_total_slices)
    slices = slice_manifest(groups)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    return {
        "tool": "dataset_slice_manifest",
        "timestamp": iso_now(),
        "status": status,
        "source_count": len(sources),
        "record_count": len(records),
        "slice_count": len(slices),
        "required_slices": [f"{dataset}:{split}" for dataset, split in required_slices],
        "min_records_per_slice": min_records_per_slice,
        "min_total_slices": min_total_slices,
        "sources": sources,
        "slices": slices,
        "records": [asdict(record) for record in sorted(records, key=lambda item: (item.dataset, item.split, item.record_id))],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "record_count",
        "choice_count_histogram",
        "answer_histogram",
        "prompt_bytes",
        "choices_bytes",
        "slice_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["slices"]:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "record_count": row["record_count"],
                    "choice_count_histogram": json.dumps(row["choice_count_histogram"], sort_keys=True),
                    "answer_histogram": json.dumps(row["answer_histogram"], sort_keys=True),
                    "prompt_bytes": row["prompt_bytes"],
                    "choices_bytes": row["choices_bytes"],
                    "slice_sha256": row["slice_sha256"],
                }
            )


def write_record_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "record_id",
        "source",
        "row_number",
        "choice_count",
        "answer_index",
        "prompt_bytes",
        "choices_bytes",
        "payload_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["records"]:
            writer.writerow({key: row[key] for key in fieldnames})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval Dataset Slice Manifest",
        "",
        f"Status: {report['status'].upper()}",
        f"Records: {report['record_count']}",
        f"Slices: {report['slice_count']}",
        "",
        "| Dataset | Split | Records | Answers | SHA256 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in report["slices"]:
        lines.append(
            "| {dataset} | {split} | {record_count} | `{answers}` | `{sha}` |".format(
                dataset=row["dataset"],
                split=row["split"],
                record_count=row["record_count"],
                answers=json.dumps(row["answer_histogram"], sort_keys=True),
                sha=row["slice_sha256"],
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['source']}: {finding['detail']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    error_findings = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_slice_manifest",
            "tests": "1",
            "failures": str(1 if error_findings else 0),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "dataset_slice_manifest", "name": "slice_manifest"})
    if error_findings:
        failure = ET.SubElement(case, "failure", {"type": error_findings[0]["kind"]})
        failure.text = "\n".join(
            f"{finding['kind']} {finding['source']}: {finding['detail']}" for finding in error_findings
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local eval JSONL input")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--csv", type=Path, help="Slice summary CSV path")
    parser.add_argument("--record-csv", type=Path, help="Per-record CSV path")
    parser.add_argument("--markdown", type=Path, help="Markdown report path")
    parser.add_argument("--junit", type=Path, help="JUnit XML report path")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows without one")
    parser.add_argument("--default-split", default="validation", help="Split name for rows without one")
    parser.add_argument("--require-slice", action="append", default=[], help="Require DATASET:SPLIT coverage")
    parser.add_argument("--min-records-per-slice", type=int, help="Fail if any present slice has fewer rows")
    parser.add_argument("--min-total-slices", type=int, help="Fail if fewer total dataset/split slices are present")
    parser.add_argument("--fail-on-findings", action="store_true", help="Return nonzero when findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    findings: list[Finding] = []
    try:
        required_slices = parse_required_slices(args.require_slice)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    records, sources = load_records(args.input, args.default_dataset, args.default_split, findings)
    report = build_report(
        records,
        sources,
        findings,
        required_slices,
        args.min_records_per_slice,
        args.min_total_slices,
    )
    write_json(args.output, report)
    if args.csv:
        write_csv(args.csv, report)
    if args.record_csv:
        write_record_csv(args.record_csv, report)
    if args.markdown:
        write_markdown(args.markdown, report)
    if args.junit:
        write_junit(args.junit, report)
    if args.fail_on_findings and report["findings"]:
        return 1
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
