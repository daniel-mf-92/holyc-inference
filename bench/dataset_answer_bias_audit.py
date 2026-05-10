#!/usr/bin/env python3
"""Audit local eval JSONL for answer-length bias before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then reports whether correct choices are systematically longer
or shorter than distractors across a curated multiple-choice subset.
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
class BiasFinding:
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


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[BiasFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[BiasFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(BiasFinding("error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(BiasFinding("error", "schema_error", f"{path}:{index + 1}", str(exc)))
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def pct(count: int, total: int) -> float | None:
    return count / total * 100.0 if total else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def answer_position(choice_lengths: list[int], answer_index: int) -> str:
    answer_len = choice_lengths[answer_index]
    min_len = min(choice_lengths)
    max_len = max(choice_lengths)
    if min_len == max_len:
        return "all_tied"
    tied_shortest = answer_len == min_len and choice_lengths.count(min_len) > 1
    tied_longest = answer_len == max_len and choice_lengths.count(max_len) > 1
    if tied_shortest:
        return "tied_shortest"
    if tied_longest:
        return "tied_longest"
    if answer_len == min_len:
        return "shortest"
    if answer_len == max_len:
        return "longest"
    return "middle"


def record_telemetry(loaded: LoadedRecord) -> dict[str, Any]:
    record = loaded.record
    choice_lengths = [len(choice.encode("utf-8")) for choice in record.choices]
    answer_bytes = choice_lengths[record.answer_index]
    distractor_lengths = [length for index, length in enumerate(choice_lengths) if index != record.answer_index]
    distractor_mean = mean([float(length) for length in distractor_lengths])
    return {
        "source": source_ref(loaded),
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "choice_count": len(record.choices),
        "answer_index": record.answer_index,
        "answer_bytes": answer_bytes,
        "min_choice_bytes": min(choice_lengths),
        "max_choice_bytes": max(choice_lengths),
        "mean_distractor_bytes": distractor_mean,
        "answer_to_distractor_mean_ratio": answer_bytes / distractor_mean if distractor_mean else None,
        "answer_length_position": answer_position(choice_lengths, record.answer_index),
    }


def sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def dataset_split_counts(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(splits.items())) for dataset, splits in sorted(counts.items())}


def summary_from_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    positions = sorted_counts(row["answer_length_position"] for row in rows)
    longest_count = positions.get("longest", 0) + positions.get("tied_longest", 0) + positions.get("all_tied", 0)
    shortest_count = positions.get("shortest", 0) + positions.get("tied_shortest", 0) + positions.get("all_tied", 0)
    ratios = [
        float(row["answer_to_distractor_mean_ratio"])
        for row in rows
        if row["answer_to_distractor_mean_ratio"] is not None
    ]
    return {
        "record_count": total,
        "position_histogram": positions,
        "answer_longest_or_tied_pct": pct(longest_count, total),
        "answer_shortest_or_tied_pct": pct(shortest_count, total),
        "mean_answer_to_distractor_ratio": mean(ratios),
    }


def dataset_split_scope(row: dict[str, Any]) -> str:
    return f"{row['dataset']}:{row['split']}"


def dataset_split_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(dataset_split_scope(row), []).append(row)
    return {scope: summary_from_telemetry(group) for scope, group in sorted(grouped.items())}


def add_threshold_findings(
    findings: list[BiasFinding],
    summary: dict[str, Any],
    scope: str,
    max_answer_longest_pct: float | None,
    max_answer_shortest_pct: float | None,
    min_mean_answer_distractor_ratio: float | None,
    max_mean_answer_distractor_ratio: float | None,
) -> None:
    longest_pct = summary["answer_longest_or_tied_pct"]
    if max_answer_longest_pct is not None and longest_pct is not None and longest_pct > max_answer_longest_pct:
        findings.append(
            BiasFinding(
                "error",
                "answer_longest_bias",
                scope,
                f"correct choice is longest/tied-longest for {longest_pct:.2f}% of records, above {max_answer_longest_pct:.2f}%",
            )
        )

    shortest_pct = summary["answer_shortest_or_tied_pct"]
    if max_answer_shortest_pct is not None and shortest_pct is not None and shortest_pct > max_answer_shortest_pct:
        findings.append(
            BiasFinding(
                "error",
                "answer_shortest_bias",
                scope,
                f"correct choice is shortest/tied-shortest for {shortest_pct:.2f}% of records, above {max_answer_shortest_pct:.2f}%",
            )
        )

    ratio = summary["mean_answer_to_distractor_ratio"]
    if min_mean_answer_distractor_ratio is not None and ratio is not None and ratio < min_mean_answer_distractor_ratio:
        findings.append(
            BiasFinding(
                "error",
                "mean_answer_too_short",
                scope,
                f"mean correct-choice bytes are {ratio:.3f}x distractors, below {min_mean_answer_distractor_ratio:.3f}x",
            )
        )
    if max_mean_answer_distractor_ratio is not None and ratio is not None and ratio > max_mean_answer_distractor_ratio:
        findings.append(
            BiasFinding(
                "error",
                "mean_answer_too_long",
                scope,
                f"mean correct-choice bytes are {ratio:.3f}x distractors, above {max_mean_answer_distractor_ratio:.3f}x",
            )
        )


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    telemetry: list[dict[str, Any]],
    findings: list[BiasFinding],
) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-answer-bias-audit",
        "inputs": inputs,
        "record_count": len(records),
        "counts_by_dataset_split": dataset_split_counts(records),
        "summary": summary_from_telemetry(telemetry),
        "dataset_split_summaries": dataset_split_summaries(telemetry),
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
        "record_telemetry": telemetry,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Dataset Answer Bias Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        f"- Answer longest/tied-longest pct: {summary['answer_longest_or_tied_pct']}",
        f"- Answer shortest/tied-shortest pct: {summary['answer_shortest_or_tied_pct']}",
        f"- Mean answer/distractor byte ratio: {summary['mean_answer_to_distractor_ratio']}",
        "",
        "## Dataset/Split Counts",
        "",
        "| dataset | split | records |",
        "| --- | --- | ---: |",
    ]
    for dataset, split_counts in report["counts_by_dataset_split"].items():
        for split, count in split_counts.items():
            lines.append(f"| {dataset} | {split} | {count} |")

    lines.extend(["", "## Answer Length Positions", "", "| position | records |", "| --- | ---: |"])
    for position, count in summary["position_histogram"].items():
        lines.append(f"| {position} | {count} |")

    lines.extend(
        [
            "",
            "## Dataset/Split Answer Length",
            "",
            "| scope | records | longest/tied-longest pct | shortest/tied-shortest pct | mean answer/distractor ratio |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, scoped_summary in report["dataset_split_summaries"].items():
        lines.append(
            "| {scope} | {record_count} | {longest} | {shortest} | {ratio} |".format(
                scope=scope,
                record_count=scoped_summary["record_count"],
                longest=scoped_summary["answer_longest_or_tied_pct"],
                shortest=scoped_summary["answer_shortest_or_tied_pct"],
                ratio=scoped_summary["mean_answer_to_distractor_ratio"],
            )
        )

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No answer-length bias findings.")
    else:
        lines.extend(["| severity | kind | scope | detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['scope']} | {finding['detail']} |")
    return "\n".join(lines) + "\n"


def write_findings_csv(path: Path, findings: list[BiasFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "scope", "detail"])
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_record_csv(path: Path, telemetry: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "choice_count",
        "answer_index",
        "answer_bytes",
        "min_choice_bytes",
        "max_choice_bytes",
        "mean_distractor_bytes",
        "answer_to_distractor_mean_ratio",
        "answer_length_position",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in telemetry:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_answer_bias_audit",
            "tests": "1",
            "failures": str(1 if report["status"] == "fail" else 0),
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "answer_bias_audit"})
    if report["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": "answer bias audit failed"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Input JSONL file; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary path")
    parser.add_argument("--csv", type=Path, help="Optional findings CSV path")
    parser.add_argument("--record-csv", type=Path, help="Optional per-record answer-length telemetry CSV path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path")
    parser.add_argument("--dataset", default="eval", help="Default dataset name for rows missing dataset")
    parser.add_argument("--split", default="validation", help="Default split name for rows missing split")
    parser.add_argument("--max-answer-longest-pct", type=float, help="Fail if correct answer is longest/tied-longest above this pct")
    parser.add_argument("--max-answer-shortest-pct", type=float, help="Fail if correct answer is shortest/tied-shortest above this pct")
    parser.add_argument("--min-mean-answer-distractor-ratio", type=float, help="Fail if mean answer bytes vs distractors is below this ratio")
    parser.add_argument("--max-mean-answer-distractor-ratio", type=float, help="Fail if mean answer bytes vs distractors is above this ratio")
    parser.add_argument(
        "--check-dataset-splits",
        action="store_true",
        help="Apply answer-length thresholds to each dataset:split as well as the overall mix",
    )
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when findings are present")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    telemetry = [record_telemetry(record) for record in records]
    add_threshold_findings(
        findings,
        summary_from_telemetry(telemetry),
        "overall",
        args.max_answer_longest_pct,
        args.max_answer_shortest_pct,
        args.min_mean_answer_distractor_ratio,
        args.max_mean_answer_distractor_ratio,
    )
    if args.check_dataset_splits:
        for scope, summary in dataset_split_summaries(telemetry).items():
            add_threshold_findings(
                findings,
                summary,
                scope,
                args.max_answer_longest_pct,
                args.max_answer_shortest_pct,
                args.min_mean_answer_distractor_ratio,
                args.max_mean_answer_distractor_ratio,
            )
    report = build_report(inputs, records, telemetry, findings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_findings_csv(args.csv, findings)
    if args.record_csv:
        write_record_csv(args.record_csv, telemetry)
    if args.junit:
        write_junit(args.junit, report)

    if args.fail_on_findings and report["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
