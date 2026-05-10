#!/usr/bin/env python3
"""Audit raw HolyC/llama prediction coverage against a local gold eval set.

This host-side tool reads local JSON/JSONL/CSV prediction artifacts only. It
checks global and per dataset/split coverage before eval comparison, so missing
or extra rows are caught before quality metrics are computed.
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
class CoverageRow:
    dataset: str
    split: str
    gold_records: int
    holyc_records: int
    llama_records: int
    holyc_coverage_pct: float
    llama_coverage_pct: float
    paired_records: int
    paired_coverage_pct: float
    missing_holyc_records: int
    missing_llama_records: int


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    dataset: str
    split: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


def add_finding(
    findings: list[Finding],
    source: str,
    dataset: str,
    split: str,
    record_id: str,
    kind: str,
    detail: str,
) -> None:
    findings.append(Finding("error", source, dataset, split, record_id, kind, detail))


def row_identity(gold: eval_compare.GoldCase | None) -> tuple[str, str]:
    if gold is None:
        return "", ""
    return gold.dataset, gold.split


def load_prediction_ids(path: Path, source: str, findings: list[Finding]) -> dict[str, int]:
    ids: dict[str, int] = {}
    try:
        rows = eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, source, "", "", "", "load_error", f"cannot read predictions: {exc}")
        return ids

    for index, row in enumerate(rows, 1):
        try:
            record_id = eval_compare.case_id(row, f"{path}:{index}")
        except ValueError as exc:
            add_finding(findings, source, "", "", "", "missing_id", str(exc))
            continue
        if record_id in ids:
            add_finding(findings, source, "", "", record_id, "duplicate_id", "duplicate prediction row")
            continue
        ids[record_id] = index
    return ids


def build_coverage_rows(
    gold: dict[str, eval_compare.GoldCase],
    holyc_ids: set[str],
    llama_ids: set[str],
) -> list[CoverageRow]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for record_id, gold_case in gold.items():
        grouped.setdefault((gold_case.dataset, gold_case.split), set()).add(record_id)

    rows: list[CoverageRow] = []
    for (dataset, split), record_ids in sorted(grouped.items()):
        holyc_present = record_ids & holyc_ids
        llama_present = record_ids & llama_ids
        paired = record_ids & holyc_ids & llama_ids
        gold_records = len(record_ids)
        rows.append(
            CoverageRow(
                dataset=dataset,
                split=split,
                gold_records=gold_records,
                holyc_records=len(holyc_present),
                llama_records=len(llama_present),
                holyc_coverage_pct=pct(len(holyc_present), gold_records),
                llama_coverage_pct=pct(len(llama_present), gold_records),
                paired_records=len(paired),
                paired_coverage_pct=pct(len(paired), gold_records),
                missing_holyc_records=gold_records - len(holyc_present),
                missing_llama_records=gold_records - len(llama_present),
            )
        )
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gold = {}
        add_finding(findings, "gold", "", "", "", "load_error", f"cannot read gold: {exc}")

    holyc = load_prediction_ids(args.holyc, "holyc", findings)
    llama = load_prediction_ids(args.llama, "llama.cpp", findings)
    gold_ids = set(gold)
    holyc_ids = set(holyc)
    llama_ids = set(llama)

    for source, ids in (("holyc", holyc_ids), ("llama.cpp", llama_ids)):
        for record_id in sorted(ids - gold_ids):
            add_finding(findings, source, "", "", record_id, "extra_id", "prediction id is not present in gold")

    for record_id in sorted(gold_ids - holyc_ids):
        dataset, split = row_identity(gold.get(record_id))
        add_finding(findings, "holyc", dataset, split, record_id, "missing_id", "gold id is missing from HolyC predictions")
    for record_id in sorted(gold_ids - llama_ids):
        dataset, split = row_identity(gold.get(record_id))
        add_finding(findings, "llama.cpp", dataset, split, record_id, "missing_id", "gold id is missing from llama.cpp predictions")

    coverage_rows = build_coverage_rows(gold, holyc_ids, llama_ids)
    paired_ids = gold_ids & holyc_ids & llama_ids
    holyc_coverage = pct(len(gold_ids & holyc_ids), len(gold_ids))
    llama_coverage = pct(len(gold_ids & llama_ids), len(gold_ids))
    paired_coverage = pct(len(paired_ids), len(gold_ids))

    if args.min_gold_records is not None and len(gold_ids) < args.min_gold_records:
        add_finding(
            findings,
            "gold",
            "",
            "",
            "",
            "min_gold_records",
            f"gold record count {len(gold_ids)} is below --min-gold-records {args.min_gold_records}",
        )
    if args.min_coverage_pct is not None:
        for source, value in (("holyc", holyc_coverage), ("llama.cpp", llama_coverage), ("paired", paired_coverage)):
            if value < args.min_coverage_pct:
                add_finding(
                    findings,
                    source,
                    "",
                    "",
                    "",
                    "min_coverage_pct",
                    f"{source} coverage {value:.2f}% is below --min-coverage-pct {args.min_coverage_pct:.2f}%",
                )
    if args.min_slice_coverage_pct is not None:
        for row in coverage_rows:
            for source, value in (
                ("holyc", row.holyc_coverage_pct),
                ("llama.cpp", row.llama_coverage_pct),
                ("paired", row.paired_coverage_pct),
            ):
                if value < args.min_slice_coverage_pct:
                    add_finding(
                        findings,
                        source,
                        row.dataset,
                        row.split,
                        "",
                        "min_slice_coverage_pct",
                        f"{source} slice coverage {value:.2f}% is below --min-slice-coverage-pct {args.min_slice_coverage_pct:.2f}%",
                    )

    status = "fail" if findings else "pass"
    return {
        "generated_at": iso_now(),
        "status": status,
        "inputs": {"gold": str(args.gold), "holyc": str(args.holyc), "llama": str(args.llama)},
        "gates": {
            "min_gold_records": args.min_gold_records,
            "min_coverage_pct": args.min_coverage_pct,
            "min_slice_coverage_pct": args.min_slice_coverage_pct,
        },
        "summary": {
            "gold_records": len(gold_ids),
            "holyc_records": len(holyc_ids & gold_ids),
            "llama_records": len(llama_ids & gold_ids),
            "paired_records": len(paired_ids),
            "holyc_coverage_pct": holyc_coverage,
            "llama_coverage_pct": llama_coverage,
            "paired_coverage_pct": paired_coverage,
            "slice_count": len(coverage_rows),
            "findings": len(findings),
        },
        "coverage": [asdict(row) for row in coverage_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Prediction Coverage Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {summary['gold_records']} |",
        f"| HolyC coverage | {summary['holyc_records']} ({summary['holyc_coverage_pct']:.2f}%) |",
        f"| llama.cpp coverage | {summary['llama_records']} ({summary['llama_coverage_pct']:.2f}%) |",
        f"| Paired coverage | {summary['paired_records']} ({summary['paired_coverage_pct']:.2f}%) |",
        f"| Dataset/split slices | {summary['slice_count']} |",
        f"| Findings | {summary['findings']} |",
        "",
        "## Slice Coverage",
        "",
        "| Dataset | Split | Gold | HolyC | llama.cpp | Paired |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["coverage"]:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['gold_records']} | "
            f"{row['holyc_records']} ({row['holyc_coverage_pct']:.2f}%) | "
            f"{row['llama_records']} ({row['llama_coverage_pct']:.2f}%) | "
            f"{row['paired_records']} ({row['paired_coverage_pct']:.2f}%) |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(
            f"- {finding['source']} {finding['kind']} {finding['record_id'] or finding['dataset'] or 'global'}: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No eval prediction coverage findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CoverageRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["coverage"])


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
        {"name": "holyc_eval_prediction_coverage_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    testcase = ET.SubElement(suite, "testcase", {"name": "eval_prediction_coverage"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_prediction_coverage_failure"})
        failure.text = "\n".join(f"{item['source']} {item['kind']} {item['record_id']}: {item['detail']}" for item in findings)
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
    parser.add_argument("--output-stem", default="eval_prediction_coverage_audit_latest")
    parser.add_argument("--min-gold-records", type=int)
    parser.add_argument("--min-coverage-pct", type=float)
    parser.add_argument("--min-slice-coverage-pct", type=float)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> int:
    if args.min_gold_records is not None and args.min_gold_records < 1:
        print("error: --min-gold-records must be at least 1", file=sys.stderr)
        return 2
    for option in ("min_coverage_pct", "min_slice_coverage_pct"):
        value = getattr(args, option)
        if value is not None and not 0.0 <= value <= 100.0:
            print(f"error: --{option.replace('_', '-')} must be between 0 and 100", file=sys.stderr)
            return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if code := validate_args(args):
        return code
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
