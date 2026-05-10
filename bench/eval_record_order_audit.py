#!/usr/bin/env python3
"""Audit HolyC/llama prediction record order against a local gold eval set.

This host-side tool reads local gold and prediction artifacts only. It catches
record-order drift before apples-to-apples eval comparison, while also reporting
duplicate, missing, and extra IDs that make an order comparison ambiguous.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


@dataclass(frozen=True)
class OrderRow:
    record_id: str
    gold_position: int
    holyc_position: int | None
    llama_position: int | None
    holyc_matches_gold: bool
    llama_matches_gold: bool
    engines_match: bool


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def add_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


def load_prediction_order(path: Path, source: str, findings: list[Finding]) -> list[str]:
    try:
        rows = eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, source, "", "load_error", f"cannot read predictions: {exc}")
        return []

    order: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        try:
            record_id = eval_compare.case_id(row, f"{path}:{index}")
        except ValueError as exc:
            add_finding(findings, source, "", "missing_id", str(exc))
            continue
        if record_id in seen:
            add_finding(findings, source, record_id, "duplicate_id", "duplicate prediction row")
            continue
        seen.add(record_id)
        order.append(record_id)
    return order


def position_map(order: list[str]) -> dict[str, int]:
    return {record_id: index + 1 for index, record_id in enumerate(order)}


def compare_source_order(
    source: str,
    source_order: list[str],
    source_positions: dict[str, int],
    gold_order: list[str],
    gold_positions: dict[str, int],
    findings: list[Finding],
) -> None:
    source_ids = set(source_positions)
    gold_ids = set(gold_positions)
    for record_id in sorted(source_ids - gold_ids):
        add_finding(findings, source, record_id, "extra_id", "prediction id is not present in gold")
    for record_id in sorted(gold_ids - source_ids):
        add_finding(findings, source, record_id, "missing_id", "gold id is missing from predictions")

    paired_gold_order = [record_id for record_id in gold_order if record_id in source_ids]
    paired_source_order = [record_id for record_id in source_order if record_id in gold_ids]
    if paired_gold_order != paired_source_order:
        for expected_position, (expected_id, observed_id) in enumerate(
            zip(paired_gold_order, paired_source_order, strict=False), 1
        ):
            if expected_id != observed_id:
                add_finding(
                    findings,
                    source,
                    observed_id,
                    "order_mismatch",
                    f"position {expected_position} expected {expected_id!r} from gold order",
                )
                break
        if len(paired_gold_order) != len(paired_source_order):
            add_finding(
                findings,
                source,
                "",
                "paired_order_length_mismatch",
                f"paired gold/source lengths differ: {len(paired_gold_order)} vs {len(paired_source_order)}",
            )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, "gold", "", "load_error", f"cannot read gold: {exc}")
        gold = {}

    gold_order = list(gold)
    holyc_order = load_prediction_order(args.holyc, "holyc", findings)
    llama_order = load_prediction_order(args.llama, "llama.cpp", findings)
    gold_positions = position_map(gold_order)
    holyc_positions = position_map(holyc_order)
    llama_positions = position_map(llama_order)

    compare_source_order("holyc", holyc_order, holyc_positions, gold_order, gold_positions, findings)
    compare_source_order("llama.cpp", llama_order, llama_positions, gold_order, gold_positions, findings)

    paired_ids = set(gold_positions) & set(holyc_positions) & set(llama_positions)
    for record_id in sorted(paired_ids, key=lambda item: gold_positions[item]):
        if holyc_positions[record_id] != llama_positions[record_id]:
            add_finding(
                findings,
                "paired",
                record_id,
                "engine_order_mismatch",
                f"HolyC position {holyc_positions[record_id]} differs from llama.cpp position {llama_positions[record_id]}",
            )

    rows = [
        OrderRow(
            record_id=record_id,
            gold_position=gold_positions[record_id],
            holyc_position=holyc_positions.get(record_id),
            llama_position=llama_positions.get(record_id),
            holyc_matches_gold=holyc_positions.get(record_id) == gold_positions[record_id],
            llama_matches_gold=llama_positions.get(record_id) == gold_positions[record_id],
            engines_match=(
                record_id in holyc_positions
                and record_id in llama_positions
                and holyc_positions[record_id] == llama_positions[record_id]
            ),
        )
        for record_id in gold_order
    ]

    return {
        "format": "eval-record-order-audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {"gold": str(args.gold), "holyc": str(args.holyc), "llama": str(args.llama)},
        "summary": {
            "gold_records": len(gold_order),
            "holyc_records": len(holyc_order),
            "llama_records": len(llama_order),
            "paired_records": len(paired_ids),
            "order_rows": len(rows),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Record Order Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {summary['gold_records']} |",
        f"| HolyC records | {summary['holyc_records']} |",
        f"| llama.cpp records | {summary['llama_records']} |",
        f"| Paired records | {summary['paired_records']} |",
        f"| Findings | {summary['findings']} |",
        "",
        "## Record Order",
        "",
        "| Gold Pos | Record ID | HolyC Pos | llama.cpp Pos | Engines Match |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['gold_position']} | {row['record_id']} | {row['holyc_position'] or '-'} | "
            f"{row['llama_position'] or '-'} | {row['engines_match']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(
            f"- {finding['source']} {finding['kind']} {finding['record_id'] or 'global'}: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No eval record order findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(OrderRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["rows"])


def write_findings_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Finding.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["findings"])


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_eval_record_order_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    testcase = ET.SubElement(suite, "testcase", {"name": "eval_record_order"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_record_order_failure"})
        failure.text = "\n".join(
            f"{item['source']} {item['kind']} {item['record_id']}: {item['detail']}" for item in findings
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--holyc", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_record_order_audit_latest")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    stem = args.output_dir / args.output_stem
    write_json(stem.with_suffix(".json"), report)
    write_markdown(stem.with_suffix(".md"), report)
    write_csv(stem.with_suffix(".csv"), report)
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", report)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", report)
    if args.fail_on_findings and report["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
