#!/usr/bin/env python3
"""Audit local eval JSONL provenance balance before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py and reports whether a curated eval subset has missing,
under-covered, or dominant provenance/source buckets.
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

MISSING_PROVENANCE = "(missing)"


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class ProvenanceFinding:
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
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[ProvenanceFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[ProvenanceFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(ProvenanceFinding("error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(ProvenanceFinding("error", "schema_error", f"{path}:{index + 1}", str(exc)))
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def provenance_label(value: str) -> str:
    cleaned = dataset_pack.clean_text(value)
    return cleaned or MISSING_PROVENANCE


def pct(count: int, total: int) -> float | None:
    return (count / total * 100.0) if total else None


def sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def dataset_split_counts(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(splits.items())) for dataset, splits in sorted(counts.items())}


def distribution_rows(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    total = len(records)
    rows: list[dict[str, Any]] = []

    for provenance, count in sorted_counts(provenance_label(loaded.record.provenance) for loaded in records).items():
        rows.append(
            {
                "scope": "provenance",
                "dataset": "",
                "split": "",
                "provenance": provenance,
                "records": count,
                "pct_of_scope": pct(count, total),
            }
        )

    grouped: dict[tuple[str, str], collections.Counter[str]] = {}
    for loaded in records:
        key = (loaded.record.dataset, loaded.record.split)
        counter = grouped.setdefault(key, collections.Counter())
        counter[provenance_label(loaded.record.provenance)] += 1

    for (dataset, split), counter in sorted(grouped.items()):
        scope_total = sum(counter.values())
        for provenance in sorted(counter):
            rows.append(
                {
                    "scope": "dataset_split_provenance",
                    "dataset": dataset,
                    "split": split,
                    "provenance": provenance,
                    "records": counter[provenance],
                    "pct_of_scope": pct(counter[provenance], scope_total),
                }
            )

    return rows


def record_rows(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for loaded in records:
        record = loaded.record
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
                "provenance": record.provenance,
                "provenance_bucket": provenance_label(record.provenance),
                "answer_index": record.answer_index,
                "choice_count": len(record.choices),
                "normalized_payload_sha256": payload_sha256,
            }
        )
    return rows


def add_missing_provenance_findings(
    findings: list[ProvenanceFinding],
    records: list[LoadedRecord],
    require_provenance: bool,
) -> None:
    if not require_provenance:
        return
    for loaded in records:
        if provenance_label(loaded.record.provenance) == MISSING_PROVENANCE:
            findings.append(
                ProvenanceFinding(
                    "error",
                    "missing_provenance",
                    f"{loaded.source}:{loaded.row_number}",
                    f"record {loaded.record.record_id} has empty provenance",
                )
            )


def add_required_source_findings(
    findings: list[ProvenanceFinding],
    records: list[LoadedRecord],
    required_sources: list[str],
) -> None:
    present = {provenance_label(loaded.record.provenance) for loaded in records}
    for source in sorted(set(required_sources)):
        if source not in present:
            findings.append(ProvenanceFinding("error", "missing_provenance_source", source, "required provenance is absent"))


def add_cardinality_findings(
    findings: list[ProvenanceFinding],
    records: list[LoadedRecord],
    min_sources: int | None,
) -> None:
    if min_sources is None:
        return
    source_count = len(
        {
            provenance_label(loaded.record.provenance)
            for loaded in records
            if provenance_label(loaded.record.provenance) != MISSING_PROVENANCE
        }
    )
    if source_count < min_sources:
        findings.append(
            ProvenanceFinding(
                "error",
                "min_provenance_sources",
                "overall",
                f"{source_count} non-empty provenance sources found, below {min_sources}",
            )
        )


def add_min_count_findings(
    findings: list[ProvenanceFinding],
    rows: list[dict[str, Any]],
    minimum: int | None,
) -> None:
    if minimum is None:
        return
    for row in rows:
        if row["scope"] != "provenance" or row["provenance"] == MISSING_PROVENANCE:
            continue
        if row["records"] < minimum:
            findings.append(
                ProvenanceFinding(
                    "error",
                    "min_records_per_provenance",
                    row["provenance"],
                    f"{row['provenance']} has {row['records']} records, below {minimum}",
                )
            )


def add_max_pct_findings(
    findings: list[ProvenanceFinding],
    rows: list[dict[str, Any]],
    scope: str,
    maximum_pct: float | None,
) -> None:
    if maximum_pct is None:
        return
    for row in rows:
        if row["scope"] != scope:
            continue
        share = row["pct_of_scope"]
        if share is not None and share > maximum_pct:
            if scope == "provenance":
                label = row["provenance"]
                kind = "max_provenance_pct"
            else:
                label = f"{row['dataset']}:{row['split']}:{row['provenance']}"
                kind = "max_dataset_split_provenance_pct"
            findings.append(
                ProvenanceFinding(
                    "error",
                    kind,
                    label,
                    f"{label} is {share:.2f}% of scope records, above {maximum_pct:.2f}%",
                )
            )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    rows = distribution_rows(records)
    provenance_counts = sorted_counts(provenance_label(loaded.record.provenance) for loaded in records)

    add_missing_provenance_findings(findings, records, args.require_provenance)
    add_required_source_findings(findings, records, args.require_provenance_source)
    add_cardinality_findings(findings, records, args.min_provenance_sources)
    add_min_count_findings(findings, rows, args.min_records_per_provenance)
    add_max_pct_findings(findings, rows, "provenance", args.max_provenance_pct)
    add_max_pct_findings(
        findings,
        rows,
        "dataset_split_provenance",
        args.max_dataset_split_provenance_pct,
    )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-provenance-balance-audit",
        "inputs": inputs,
        "record_count": len(records),
        "dataset_split_counts": dataset_split_counts(records),
        "provenance_source_count": len(
            {source for source in provenance_counts if source != MISSING_PROVENANCE}
        ),
        "provenance_counts": provenance_counts,
        "distribution": rows,
        "record_telemetry": record_rows(records),
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Provenance Balance Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Provenance sources: {report['provenance_source_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Distribution",
        "",
        "| scope | dataset | split | provenance | records | pct of scope |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in report["distribution"]:
        share = row["pct_of_scope"]
        share_text = "" if share is None else f"{share:.2f}"
        lines.append(
            f"| {row['scope']} | {row['dataset']} | {row['split']} | {row['provenance']} | "
            f"{row['records']} | {share_text} |"
        )

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No provenance balance findings.")
    else:
        lines.extend(["| severity | kind | scope | detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['scope']} | {finding['detail']} |")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scope", "dataset", "split", "provenance", "records", "pct_of_scope"]
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
        "provenance",
        "provenance_bucket",
        "answer_index",
        "choice_count",
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
            "name": "dataset_provenance_balance_audit",
            "tests": "1",
            "failures": str(report["error_count"]),
            "errors": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {"classname": "bench.dataset_provenance_balance_audit", "name": "provenance_balance"},
    )
    if report["error_count"]:
        failure = ET.SubElement(case, "failure", {"message": f"{report['error_count']} provenance balance errors"})
        failure.text = "\n".join(f"{item['kind']} {item['scope']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local JSONL eval input.")
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--csv", type=Path, help="Optional provenance distribution CSV path.")
    parser.add_argument("--record-csv", type=Path, help="Optional per-record provenance telemetry CSV path.")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path.")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path.")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata.")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata.")
    parser.add_argument("--require-provenance", action="store_true", help="Fail records with empty provenance.")
    parser.add_argument(
        "--require-provenance-source",
        action="append",
        default=[],
        help="Provenance/source string that must be present.",
    )
    parser.add_argument(
        "--min-provenance-sources",
        type=int,
        help="Fail unless at least this many non-empty provenance sources are present.",
    )
    parser.add_argument(
        "--min-records-per-provenance",
        type=int,
        help="Minimum records for every present non-empty provenance source.",
    )
    parser.add_argument("--max-provenance-pct", type=float, help="Maximum percent any provenance may contribute.")
    parser.add_argument(
        "--max-dataset-split-provenance-pct",
        type=float,
        help="Maximum percent any provenance may contribute within a dataset/split bucket.",
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
