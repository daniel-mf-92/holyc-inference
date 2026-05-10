#!/usr/bin/env python3
"""Audit multiple-choice eval rows for answer-length cue bias.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then reports choice byte spans and flags rows where the correct
choice is an outlier by absolute delta, ratio, or percentile rank. This catches
curation artifacts that can make HolyC-vs-llama evals answerable from option
length alone.
"""

from __future__ import annotations

import argparse
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
class LengthFinding:
    severity: str
    kind: str
    source: str
    dataset: str
    split: str
    record_id: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[LengthFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[LengthFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(LengthFinding("error", "read_error", str(path), "", "", "", str(exc)))
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
                    LengthFinding(
                        "error",
                        "schema_error",
                        f"{path}:{index + 1}",
                        "",
                        "",
                        "",
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def choice_bytes(record: dataset_pack.EvalRecord) -> list[int]:
    return [len(choice.encode("utf-8")) for choice in record.choices]


def answer_length_rank(lengths: list[int], answer_index: int) -> str:
    answer_len = lengths[answer_index]
    if answer_len == min(lengths) and lengths.count(answer_len) == 1:
        return "shortest"
    if answer_len == max(lengths) and lengths.count(answer_len) == 1:
        return "longest"
    return "middle_or_tied"


def record_telemetry(loaded: LoadedRecord) -> dict[str, Any]:
    record = loaded.record
    lengths = choice_bytes(record)
    answer_len = lengths[record.answer_index]
    other_lengths = [length for index, length in enumerate(lengths) if index != record.answer_index]
    min_len = min(lengths)
    max_len = max(lengths)
    mean_other = sum(other_lengths) / len(other_lengths) if other_lengths else 0.0
    nearest_other_delta = min(abs(answer_len - length) for length in other_lengths) if other_lengths else 0
    answer_to_mean_other_ratio = round(answer_len / mean_other, 6) if mean_other else 0.0
    max_to_min_ratio = round(max_len / min_len, 6) if min_len else 0.0
    return {
        "source": source_ref(loaded),
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "choice_count": len(record.choices),
        "answer_index": record.answer_index,
        "answer_choice_bytes": answer_len,
        "min_choice_bytes": min_len,
        "max_choice_bytes": max_len,
        "choice_byte_span": max_len - min_len,
        "nearest_other_delta_bytes": nearest_other_delta,
        "answer_to_mean_other_ratio": answer_to_mean_other_ratio,
        "max_to_min_ratio": max_to_min_ratio,
        "answer_length_rank": answer_length_rank(lengths, record.answer_index),
        "choice_bytes": ",".join(str(length) for length in lengths),
    }


def add_record_finding(findings: list[LengthFinding], row: dict[str, Any], kind: str, detail: str) -> None:
    findings.append(
        LengthFinding(
            "error",
            kind,
            str(row["source"]),
            str(row["dataset"]),
            str(row["split"]),
            str(row["record_id"]),
            detail,
        )
    )


def add_gate_findings(
    findings: list[LengthFinding],
    rows: list[dict[str, Any]],
    *,
    max_choice_byte_span: int | None,
    max_answer_delta_bytes: int | None,
    max_answer_to_mean_other_ratio: float | None,
    min_answer_to_mean_other_ratio: float | None,
    fail_on_unique_longest_answer: bool,
    fail_on_unique_shortest_answer: bool,
) -> None:
    for row in rows:
        if max_choice_byte_span is not None and int(row["choice_byte_span"]) > max_choice_byte_span:
            add_record_finding(
                findings,
                row,
                "choice_byte_span_exceeded",
                f"choice byte span {row['choice_byte_span']} exceeds {max_choice_byte_span}",
            )
        if max_answer_delta_bytes is not None and int(row["nearest_other_delta_bytes"]) > max_answer_delta_bytes:
            add_record_finding(
                findings,
                row,
                "answer_length_delta_exceeded",
                f"answer is {row['nearest_other_delta_bytes']} bytes from nearest distractor, above {max_answer_delta_bytes}",
            )
        ratio = float(row["answer_to_mean_other_ratio"])
        if max_answer_to_mean_other_ratio is not None and ratio > max_answer_to_mean_other_ratio:
            add_record_finding(
                findings,
                row,
                "answer_length_ratio_high",
                f"answer/mean-distractor byte ratio {ratio:.6f} exceeds {max_answer_to_mean_other_ratio:.6f}",
            )
        if min_answer_to_mean_other_ratio is not None and ratio < min_answer_to_mean_other_ratio:
            add_record_finding(
                findings,
                row,
                "answer_length_ratio_low",
                f"answer/mean-distractor byte ratio {ratio:.6f} is below {min_answer_to_mean_other_ratio:.6f}",
            )
        if fail_on_unique_longest_answer and row["answer_length_rank"] == "longest":
            add_record_finding(findings, row, "answer_unique_longest", "correct answer is uniquely longest choice")
        if fail_on_unique_shortest_answer and row["answer_length_rank"] == "shortest":
            add_record_finding(findings, row, "answer_unique_shortest", "correct answer is uniquely shortest choice")


def sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records, inputs, findings = load_records(
        [Path(path) for path in args.input],
        args.default_dataset,
        args.default_split,
    )
    rows = [record_telemetry(record) for record in records]
    add_gate_findings(
        findings,
        rows,
        max_choice_byte_span=args.max_choice_byte_span,
        max_answer_delta_bytes=args.max_answer_delta_bytes,
        max_answer_to_mean_other_ratio=args.max_answer_to_mean_other_ratio,
        min_answer_to_mean_other_ratio=args.min_answer_to_mean_other_ratio,
        fail_on_unique_longest_answer=args.fail_on_unique_longest_answer,
        fail_on_unique_shortest_answer=args.fail_on_unique_shortest_answer,
    )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "format": "hceval-choice-length-audit",
        "status": "fail" if error_count else "pass",
        "inputs": inputs,
        "record_count": len(rows),
        "error_count": error_count,
        "answer_length_rank_histogram": sorted_counts(str(row["answer_length_rank"]) for row in rows),
        "max_choice_byte_span": max((int(row["choice_byte_span"]) for row in rows), default=0),
        "max_answer_delta_bytes": max((int(row["nearest_other_delta_bytes"]) for row in rows), default=0),
        "max_answer_to_mean_other_ratio": max((float(row["answer_to_mean_other_ratio"]) for row in rows), default=0.0),
        "min_answer_to_mean_other_ratio": min((float(row["answer_to_mean_other_ratio"]) for row in rows), default=0.0),
        "record_telemetry": rows,
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "source", "dataset", "split", "record_id", "detail"],
        )
        writer.writeheader()
        writer.writerows(findings)


def write_record_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "choice_count",
        "answer_index",
        "answer_choice_bytes",
        "min_choice_bytes",
        "max_choice_bytes",
        "choice_byte_span",
        "nearest_other_delta_bytes",
        "answer_to_mean_other_ratio",
        "max_to_min_ratio",
        "answer_length_rank",
        "choice_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Choice Length Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Max choice byte span: {report['max_choice_byte_span']}",
        f"- Max answer delta bytes: {report['max_answer_delta_bytes']}",
        f"- Answer/mean-distractor ratio range: {report['min_answer_to_mean_other_ratio']} - {report['max_answer_to_mean_other_ratio']}",
        "",
        "## Answer Length Ranks",
        "",
    ]
    for key, count in report["answer_length_rank_histogram"].items():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['dataset']}:{finding['split']}:{finding['record_id']}: {finding['detail']}"
            )
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = report["findings"]
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_choice_length_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(sum(1 for finding in findings if finding["severity"] == "error")),
        },
    )
    if not findings:
        ET.SubElement(testsuite, "testcase", {"classname": "dataset_choice_length_audit", "name": "choice_length_audit_pass"})
    for finding in findings:
        testcase = ET.SubElement(
            testsuite,
            {
                "classname": "dataset_choice_length_audit",
                "name": f"{finding['kind']}:{finding['dataset']}:{finding['record_id']}",
            },
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
    parser.add_argument("--record-csv", help="Optional per-record length telemetry CSV output path")
    parser.add_argument("--junit", help="Optional JUnit XML output path")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--max-choice-byte-span", type=int)
    parser.add_argument("--max-answer-delta-bytes", type=int)
    parser.add_argument("--max-answer-to-mean-other-ratio", type=float)
    parser.add_argument("--min-answer-to-mean-other-ratio", type=float)
    parser.add_argument("--fail-on-unique-longest-answer", action="store_true")
    parser.add_argument("--fail-on-unique-shortest-answer", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args)
    write_json(Path(args.output), report)
    if args.markdown:
        write_markdown(Path(args.markdown), report)
    if args.csv:
        write_findings_csv(Path(args.csv), report["findings"])
    if args.record_csv:
        write_record_csv(Path(args.record_csv), report["record_telemetry"])
    if args.junit:
        write_junit(Path(args.junit), report)
    return 1 if args.fail_on_findings and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
