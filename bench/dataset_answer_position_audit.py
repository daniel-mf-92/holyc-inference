#!/usr/bin/env python3
"""Audit local eval JSONL for correct-answer position concentration.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then reports answer-index histograms overall and per
dataset/split so curated subsets do not accidentally encode a position shortcut.
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
class ScopeSummary:
    scope: str
    dataset: str
    split: str
    record_count: int
    choice_count_min: int
    choice_count_max: int
    distinct_answer_positions: int
    dominant_answer_index: int
    dominant_answer_count: int
    dominant_answer_pct: float
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


def histogram_text(values: Iterable[int]) -> str:
    counts = collections.Counter(values)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def pct(count: int, total: int) -> float:
    return count / total * 100.0 if total else 0.0


def summarize_scope(scope: str, dataset: str, split: str, records: list[LoadedRecord]) -> ScopeSummary:
    answer_counts = collections.Counter(record.record.answer_index for record in records)
    dominant_answer_index, dominant_answer_count = max(answer_counts.items(), key=lambda item: (item[1], -item[0]))
    choice_counts = [len(record.record.choices) for record in records]
    return ScopeSummary(
        scope=scope,
        dataset=dataset,
        split=split,
        record_count=len(records),
        choice_count_min=min(choice_counts),
        choice_count_max=max(choice_counts),
        distinct_answer_positions=len(answer_counts),
        dominant_answer_index=dominant_answer_index,
        dominant_answer_count=dominant_answer_count,
        dominant_answer_pct=pct(dominant_answer_count, len(records)),
        answer_index_histogram=histogram_text(record.record.answer_index for record in records),
    )


def build_summaries(records: list[LoadedRecord]) -> list[ScopeSummary]:
    if not records:
        return []
    summaries = [summarize_scope("overall", "", "", records)]
    grouped: dict[tuple[str, str], list[LoadedRecord]] = collections.defaultdict(list)
    for record in records:
        grouped[(record.record.dataset, record.record.split)].append(record)
    for (dataset, split), scope_records in sorted(grouped.items()):
        summaries.append(summarize_scope("dataset_split", dataset, split, scope_records))
    return summaries


def add_gate_findings(
    findings: list[Finding],
    summaries: list[ScopeSummary],
    *,
    min_records: int,
    min_scope_records: int,
    min_distinct_answer_positions: int | None,
    max_dominant_answer_pct: float | None,
) -> None:
    for summary in summaries:
        scope_name = "overall" if summary.scope == "overall" else f"{summary.dataset}:{summary.split}"
        min_required = min_records if summary.scope == "overall" else min_scope_records
        if summary.record_count < min_required:
            findings.append(
                Finding(
                    "error",
                    "insufficient_records",
                    scope_name,
                    f"record_count={summary.record_count} minimum={min_required}",
                )
            )
        if (
            min_distinct_answer_positions is not None
            and summary.record_count >= min_required
            and summary.distinct_answer_positions < min_distinct_answer_positions
        ):
            findings.append(
                Finding(
                    "error",
                    "answer_position_coverage",
                    scope_name,
                    f"distinct_answer_positions={summary.distinct_answer_positions} minimum={min_distinct_answer_positions}",
                )
            )
        if max_dominant_answer_pct is not None and summary.dominant_answer_pct > max_dominant_answer_pct:
            findings.append(
                Finding(
                    "error",
                    "dominant_answer_position",
                    scope_name,
                    f"answer_index={summary.dominant_answer_index} pct={summary.dominant_answer_pct:.2f} max={max_dominant_answer_pct:.2f}",
                )
            )


def build_report(inputs: list[dict[str, Any]], summaries: list[ScopeSummary], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "inputs": inputs,
        "summary": {
            "inputs": len(inputs),
            "scopes": len(summaries),
            "records": summaries[0].record_count if summaries else 0,
            "findings": len(findings),
        },
        "scopes": [asdict(summary) for summary in summaries],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Answer Position Audit",
        "",
        f"- status: {report['status']}",
        f"- records: {report['summary']['records']}",
        f"- scopes: {report['summary']['scopes']}",
        f"- findings: {report['summary']['findings']}",
        "",
        "## Scopes",
        "",
        "| scope | records | distinct positions | dominant index | dominant pct | histogram |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["scopes"]:
        scope = row["scope"] if row["scope"] == "overall" else f"{row['dataset']}:{row['split']}"
        lines.append(
            f"| {scope} | {row['record_count']} | {row['distinct_answer_positions']} | "
            f"{row['dominant_answer_index']} | {row['dominant_answer_pct']:.2f} | {row['answer_index_histogram']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['severity']} {finding['kind']} {finding['scope']}: {finding['detail']}")
    else:
        lines.append("No dataset answer-position findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_answer_position_audit",
            "tests": "1",
            "failures": "1" if errors else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "answer_position_distribution"})
    if errors:
        failure = ET.SubElement(case, "failure", {"message": f"{len(errors)} answer-position findings"})
        failure.text = "\n".join(f"{item['kind']} {item['scope']}: {item['detail']}" for item in errors)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-scope-records", type=int, default=1)
    parser.add_argument("--min-distinct-answer-positions", type=int)
    parser.add_argument("--max-dominant-answer-pct", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="dataset_answer_position_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, inputs, findings = load_records(args.inputs, args.default_dataset, args.default_split)
    summaries = build_summaries(records)
    add_gate_findings(
        findings,
        summaries,
        min_records=args.min_records,
        min_scope_records=args.min_scope_records,
        min_distinct_answer_positions=args.min_distinct_answer_positions,
        max_dominant_answer_pct=args.max_dominant_answer_pct,
    )
    report = build_report(inputs, summaries, findings)

    base = args.output_dir / args.output_stem
    write_json(base.with_suffix(".json"), report)
    write_markdown(base.with_suffix(".md"), report)
    write_csv(
        base.with_suffix(".csv"),
        report["scopes"],
        [
            "scope",
            "dataset",
            "split",
            "record_count",
            "choice_count_min",
            "choice_count_max",
            "distinct_answer_positions",
            "dominant_answer_index",
            "dominant_answer_count",
            "dominant_answer_pct",
            "answer_index_histogram",
        ],
    )
    write_csv(
        args.output_dir / f"{args.output_stem}_findings.csv",
        report["findings"],
        ["severity", "kind", "scope", "detail"],
    )
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
