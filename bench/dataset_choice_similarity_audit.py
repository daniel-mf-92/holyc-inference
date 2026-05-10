#!/usr/bin/env python3
"""Audit multiple-choice eval rows for duplicate or near-duplicate choices.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py and flags rows where options collapse to duplicates or exceed a
similarity threshold, which catches ambiguous curated examples before packing.
"""

from __future__ import annotations

import argparse
import csv
import difflib
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
class SimilarityFinding:
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
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[SimilarityFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[SimilarityFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(SimilarityFinding("error", "read_error", str(path), "", "", "", str(exc)))
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
                    SimilarityFinding("error", "schema_error", f"{path}:{index + 1}", "", "", "", str(exc))
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def normalize_choice(text: str) -> str:
    lowered = text.casefold()
    stripped = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def choice_pairs(record: dataset_pack.EvalRecord) -> list[dict[str, Any]]:
    normalized = [normalize_choice(choice) for choice in record.choices]
    rows: list[dict[str, Any]] = []
    for left in range(len(record.choices)):
        for right in range(left + 1, len(record.choices)):
            ratio = difflib.SequenceMatcher(None, normalized[left], normalized[right]).ratio()
            rows.append(
                {
                    "left_index": left,
                    "right_index": right,
                    "left_normalized": normalized[left],
                    "right_normalized": normalized[right],
                    "similarity": round(ratio, 6),
                    "duplicate_normalized": normalized[left] == normalized[right],
                }
            )
    return rows


def record_telemetry(loaded: LoadedRecord) -> dict[str, Any]:
    record = loaded.record
    normalized = [normalize_choice(choice) for choice in record.choices]
    pairs = choice_pairs(record)
    max_pair = max((pair["similarity"] for pair in pairs), default=0.0)
    duplicate_pairs = sum(1 for pair in pairs if pair["duplicate_normalized"])
    return {
        "source": source_ref(loaded),
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "choice_count": len(record.choices),
        "unique_normalized_choices": len(set(normalized)),
        "duplicate_choice_pairs": duplicate_pairs,
        "max_choice_similarity": max_pair,
        "normalized_choice_sha256": hashlib.sha256(
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "pair_telemetry": pairs,
    }


def add_record_finding(findings: list[SimilarityFinding], row: dict[str, Any], kind: str, detail: str) -> None:
    findings.append(
        SimilarityFinding(
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
    findings: list[SimilarityFinding],
    rows: list[dict[str, Any]],
    *,
    min_unique_choices: int | None,
    max_pair_similarity: float | None,
    fail_duplicate_normalized: bool,
) -> None:
    for row in rows:
        if min_unique_choices is not None and int(row["unique_normalized_choices"]) < min_unique_choices:
            add_record_finding(
                findings,
                row,
                "min_unique_choices",
                f"{row['unique_normalized_choices']} unique normalized choices, below {min_unique_choices}",
            )
        if fail_duplicate_normalized and int(row["duplicate_choice_pairs"]) > 0:
            add_record_finding(
                findings,
                row,
                "duplicate_normalized_choice",
                f"{row['duplicate_choice_pairs']} duplicate normalized choice pairs",
            )
        if max_pair_similarity is not None:
            for pair in row["pair_telemetry"]:
                if float(pair["similarity"]) > max_pair_similarity:
                    add_record_finding(
                        findings,
                        row,
                        "choice_similarity_exceeded",
                        (
                            f"choices {pair['left_index']} and {pair['right_index']} have "
                            f"similarity {pair['similarity']:.6f}, above {max_pair_similarity:.6f}"
                        ),
                    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    rows = [record_telemetry(loaded) for loaded in records]
    add_gate_findings(
        findings,
        rows,
        min_unique_choices=args.min_unique_choices,
        max_pair_similarity=args.max_pair_similarity,
        fail_duplicate_normalized=args.fail_duplicate_normalized,
    )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-choice-similarity-audit",
        "inputs": inputs,
        "record_count": len(records),
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "warning_count": warning_count,
        "records": rows,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Choice Similarity Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Records",
        "",
        "| source | record id | unique choices | duplicate pairs | max similarity |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report["records"]:
        lines.append(
            "| "
            f"{row['source']} | {row['record_id']} | {row['unique_normalized_choices']} | "
            f"{row['duplicate_choice_pairs']} | {float(row['max_choice_similarity']):.6f} |"
        )
    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No dataset choice similarity findings.")
    else:
        lines.extend(["| severity | kind | source | record id | detail |", "| --- | --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['kind']} | {finding['source']} | "
                f"{finding['record_id']} | {finding['detail']} |"
            )
    lines.append("")
    return "\n".join(lines)


def flat_record_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row["source"],
            "dataset": row["dataset"],
            "split": row["split"],
            "record_id": row["record_id"],
            "choice_count": row["choice_count"],
            "unique_normalized_choices": row["unique_normalized_choices"],
            "duplicate_choice_pairs": row["duplicate_choice_pairs"],
            "max_choice_similarity": row["max_choice_similarity"],
            "normalized_choice_sha256": row["normalized_choice_sha256"],
        }
        for row in rows
    ]


def flat_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for pair in row["pair_telemetry"]:
            output.append(
                {
                    "source": row["source"],
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "record_id": row["record_id"],
                    **pair,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_choice_similarity_audit",
            "tests": "1",
            "failures": str(report["error_count"]),
            "errors": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": "bench.dataset_choice_similarity_audit", "name": "dataset_choice_similarity"},
    )
    if report["error_count"]:
        failure = ET.SubElement(case, "failure", {"message": f"{report['error_count']} choice similarity errors"})
        failure.text = "\n".join(
            f"{item['kind']} {item['source']} {item['record_id']}: {item['detail']}"
            for item in report["findings"]
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local JSONL eval input.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--csv", type=Path, help="Optional per-record CSV path.")
    parser.add_argument("--pair-csv", type=Path, help="Optional per-choice-pair CSV path.")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path.")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path.")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata.")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata.")
    parser.add_argument("--min-unique-choices", type=int, help="Minimum normalized unique choices per record.")
    parser.add_argument("--max-pair-similarity", type=float, help="Maximum allowed normalized choice-pair similarity.")
    parser.add_argument(
        "--fail-duplicate-normalized",
        action="store_true",
        help="Fail rows with duplicate choices after normalization.",
    )
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
        write_csv(
            args.csv,
            flat_record_rows(report["records"]),
            [
                "source",
                "dataset",
                "split",
                "record_id",
                "choice_count",
                "unique_normalized_choices",
                "duplicate_choice_pairs",
                "max_choice_similarity",
                "normalized_choice_sha256",
            ],
        )
    if args.pair_csv:
        write_csv(
            args.pair_csv,
            flat_pair_rows(report["records"]),
            [
                "source",
                "dataset",
                "split",
                "record_id",
                "left_index",
                "right_index",
                "left_normalized",
                "right_normalized",
                "similarity",
                "duplicate_normalized",
            ],
        )
    if args.findings_csv:
        write_csv(
            args.findings_csv,
            report["findings"],
            ["severity", "kind", "source", "dataset", "split", "record_id", "detail"],
        )
    if args.junit:
        write_junit(args.junit, report)

    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
