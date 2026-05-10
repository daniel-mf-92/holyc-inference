#!/usr/bin/env python3
"""Report normalized eval dataset size, byte, and answer-position statistics.

This offline host-side tool reads the same JSONL row shapes accepted by
dataset_pack.py. It never downloads datasets and never touches the TempleOS
guest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
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
class ScopeStats:
    scope: str
    dataset: str
    split: str
    records: int
    prompt_bytes_min: int | None
    prompt_bytes_p50: float | None
    prompt_bytes_p95: float | None
    prompt_bytes_max: int | None
    choice_bytes_min: int | None
    choice_bytes_p50: float | None
    choice_bytes_p95: float | None
    choice_bytes_max: int | None
    choices_per_record_min: int | None
    choices_per_record_p50: float | None
    choices_per_record_max: int | None
    answer_index_histogram: str
    choice_count_histogram: str


@dataclass(frozen=True)
class RecordStats:
    source: str
    dataset: str
    split: str
    record_id: str
    prompt_bytes: int
    choice_count: int
    choice_bytes_min: int
    choice_bytes_max: int
    answer_index: int
    answer_choice_bytes: int


@dataclass(frozen=True)
class Finding:
    source: str
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


def source_ref(record: LoadedRecord) -> str:
    return f"{record.source}:{record.row_number}"


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
            findings.append(Finding(str(path), "error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(Finding(str(path), "error", "schema_error", f"{path}:{index + 1}", str(exc)))
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def histogram_text(values: Iterable[int]) -> str:
    counts = collections.Counter(values)
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def record_stats(loaded: LoadedRecord) -> RecordStats:
    record = loaded.record
    choice_bytes = [len(choice.encode("utf-8")) for choice in record.choices]
    return RecordStats(
        source=source_ref(loaded),
        dataset=record.dataset,
        split=record.split,
        record_id=record.record_id,
        prompt_bytes=len(record.prompt.encode("utf-8")),
        choice_count=len(record.choices),
        choice_bytes_min=min(choice_bytes),
        choice_bytes_max=max(choice_bytes),
        answer_index=record.answer_index,
        answer_choice_bytes=choice_bytes[record.answer_index],
    )


def scope_key(row: RecordStats) -> str:
    return f"{row.dataset}:{row.split}"


def group_scope_rows(rows: list[RecordStats]) -> dict[str, list[RecordStats]]:
    grouped: dict[str, list[RecordStats]] = {}
    if rows:
        grouped["all:all"] = list(rows)
    for row in rows:
        grouped.setdefault(scope_key(row), []).append(row)
    return grouped


def build_scope_stats(scope: str, rows: list[RecordStats]) -> ScopeStats:
    prompt_bytes = [row.prompt_bytes for row in rows]
    choice_bytes = [size for row in rows for size in (row.choice_bytes_min, row.choice_bytes_max)]
    choice_counts = [row.choice_count for row in rows]
    dataset, split = scope.split(":", 1)
    return ScopeStats(
        scope=scope,
        dataset=dataset,
        split=split,
        records=len(rows),
        prompt_bytes_min=min(prompt_bytes) if prompt_bytes else None,
        prompt_bytes_p50=percentile(prompt_bytes, 50),
        prompt_bytes_p95=percentile(prompt_bytes, 95),
        prompt_bytes_max=max(prompt_bytes) if prompt_bytes else None,
        choice_bytes_min=min(choice_bytes) if choice_bytes else None,
        choice_bytes_p50=percentile(choice_bytes, 50),
        choice_bytes_p95=percentile(choice_bytes, 95),
        choice_bytes_max=max(choice_bytes) if choice_bytes else None,
        choices_per_record_min=min(choice_counts) if choice_counts else None,
        choices_per_record_p50=percentile(choice_counts, 50),
        choices_per_record_max=max(choice_counts) if choice_counts else None,
        answer_index_histogram=histogram_text(row.answer_index for row in rows),
        choice_count_histogram=histogram_text(choice_counts),
    )


def add_gate_findings(
    findings: list[Finding],
    stats: ScopeStats,
    *,
    min_records_per_scope: int,
    max_prompt_p95_bytes: int | None,
    max_choice_p95_bytes: int | None,
) -> None:
    if stats.records < min_records_per_scope:
        findings.append(
            Finding(
                "-",
                "error",
                "min_records_per_scope",
                stats.scope,
                f"{stats.records} records, expected at least {min_records_per_scope}",
            )
        )
    if max_prompt_p95_bytes is not None and stats.prompt_bytes_p95 is not None and stats.prompt_bytes_p95 > max_prompt_p95_bytes:
        findings.append(
            Finding(
                "-",
                "error",
                "prompt_p95_bytes",
                stats.scope,
                f"prompt p95 {stats.prompt_bytes_p95:.2f} bytes exceeds {max_prompt_p95_bytes}",
            )
        )
    if max_choice_p95_bytes is not None and stats.choice_bytes_p95 is not None and stats.choice_bytes_p95 > max_choice_p95_bytes:
        findings.append(
            Finding(
                "-",
                "error",
                "choice_p95_bytes",
                stats.scope,
                f"choice p95 {stats.choice_bytes_p95:.2f} bytes exceeds {max_choice_p95_bytes}",
            )
        )


def build_report(
    inputs: list[dict[str, Any]],
    record_rows: list[RecordStats],
    scope_rows: list[ScopeStats],
    findings: list[Finding],
) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "inputs": inputs,
        "summary": {
            "inputs": len(inputs),
            "records": len(record_rows),
            "scopes": len(scope_rows),
            "findings": len(findings),
        },
        "scopes": [asdict(row) for row in scope_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    write_csv(path, findings, list(Finding.__dataclass_fields__))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Stats Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {report['summary']['records']}",
        f"Scopes: {report['summary']['scopes']}",
        f"Findings: {report['summary']['findings']}",
        "",
        "## Scopes",
        "",
        "| Scope | Records | Prompt p50 bytes | Prompt p95 bytes | Choice p50 bytes | Choice p95 bytes | Answer indexes | Choice counts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["scopes"]:
        lines.append(
            "| {scope} | {records} | {prompt_bytes_p50} | {prompt_bytes_p95} | {choice_bytes_p50} | {choice_bytes_p95} | {answer_index_histogram} | {choice_count_histogram} |".format(
                **row
            )
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(["| Severity | Kind | Scope | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append("| {severity} | {kind} | {scope} | {detail} |".format(**finding))
    else:
        lines.append("No dataset stats findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, findings: list[Finding]) -> None:
    errors = [finding for finding in findings if finding.severity == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_stats_report",
            "tests": "1",
            "failures": "1" if errors else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "dataset_stats"})
    if errors:
        failure = ET.SubElement(case, "failure", {"message": f"{len(errors)} dataset stats finding(s)"})
        failure.text = "\n".join(f"{finding.scope}: {finding.kind}: {finding.detail}" for finding in errors)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Eval JSONL files")
    parser.add_argument("--default-dataset", default="local")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-records-per-scope", type=int, default=1)
    parser.add_argument("--max-prompt-p95-bytes", type=int, default=None)
    parser.add_argument("--max-choice-p95-bytes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results/datasets"))
    parser.add_argument("--output-stem", default="dataset_stats_report_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded, inputs, findings = load_records(args.inputs, args.default_dataset, args.default_split)
    rows = [record_stats(record) for record in loaded]
    grouped = group_scope_rows(rows)
    scopes = [build_scope_stats(scope, grouped[scope]) for scope in sorted(grouped)]

    if len(rows) < args.min_records:
        findings.append(Finding("-", "error", "min_records", "all", f"{len(rows)} records, expected at least {args.min_records}"))
    for stats in scopes:
        add_gate_findings(
            findings,
            stats,
            min_records_per_scope=args.min_records_per_scope,
            max_prompt_p95_bytes=args.max_prompt_p95_bytes,
            max_choice_p95_bytes=args.max_choice_p95_bytes,
        )

    report = build_report(inputs, rows, scopes, findings)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", scopes, list(ScopeStats.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_records.csv", rows, list(RecordStats.__dataclass_fields__))
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    (args.output_dir / f"{stem}.md").write_text(markdown_report(report), encoding="utf-8")
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
