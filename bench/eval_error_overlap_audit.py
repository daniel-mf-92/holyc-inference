#!/usr/bin/env python3
"""Audit HolyC-vs-llama error-set overlap on a local gold eval set.

This host-side tool reads local gold and prediction artifacts only. It reports
which records both engines miss, which misses are unique to one engine, and the
Jaccard overlap of the two error sets before quality deltas are interpreted.
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
class ErrorOverlapRow:
    record_id: str
    dataset: str
    split: str
    answer_index: int
    holyc_prediction: int | None
    llama_prediction: int | None
    holyc_correct: bool | None
    llama_correct: bool | None
    error_class: str


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


def load_predictions(
    path: Path,
    source: str,
    gold: dict[str, eval_compare.GoldCase],
    findings: list[Finding],
) -> dict[str, eval_compare.Prediction]:
    predictions: dict[str, eval_compare.Prediction] = {}
    try:
        rows = eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, source, "", "load_error", f"cannot read predictions: {exc}")
        return predictions

    for index, row in enumerate(rows):
        label = f"{path}:{index + 1}"
        try:
            record_id = eval_compare.case_id(row, label)
        except ValueError as exc:
            add_finding(findings, source, "", "missing_id", str(exc))
            continue
        if record_id in predictions:
            add_finding(findings, source, record_id, "duplicate_id", "duplicate prediction row")
            continue
        gold_case = gold.get(record_id)
        if gold_case is None:
            add_finding(findings, source, record_id, "extra_id", "prediction id is not present in gold")
            continue
        try:
            predictions[record_id] = eval_compare.normalize_prediction(row, gold_case, path, index)
        except ValueError as exc:
            add_finding(findings, source, record_id, "invalid_prediction", str(exc))

    for record_id in sorted(set(gold) - set(predictions)):
        add_finding(findings, source, record_id, "missing_id", "gold id is missing from predictions")
    return predictions


def classify_error(holyc_correct: bool | None, llama_correct: bool | None) -> str:
    if holyc_correct is True and llama_correct is True:
        return "both_correct"
    if holyc_correct is False and llama_correct is False:
        return "shared_error"
    if holyc_correct is False and llama_correct is True:
        return "holyc_unique_error"
    if holyc_correct is True and llama_correct is False:
        return "llama_unique_error"
    if holyc_correct is None and llama_correct is False:
        return "llama_error_unpaired"
    if holyc_correct is False and llama_correct is None:
        return "holyc_error_unpaired"
    return "unpaired"


def pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100.0 if denominator else 0.0


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, "gold", "", "load_error", f"cannot read gold: {exc}")
        gold = {}

    holyc = load_predictions(args.holyc, "holyc", gold, findings)
    llama = load_predictions(args.llama, "llama.cpp", gold, findings)

    rows: list[ErrorOverlapRow] = []
    holyc_errors: set[str] = set()
    llama_errors: set[str] = set()
    paired_ids = set(gold) & set(holyc) & set(llama)

    for record_id, gold_case in gold.items():
        holyc_prediction = holyc.get(record_id)
        llama_prediction = llama.get(record_id)
        holyc_correct = (
            holyc_prediction.predicted_index == gold_case.answer_index if holyc_prediction is not None else None
        )
        llama_correct = (
            llama_prediction.predicted_index == gold_case.answer_index if llama_prediction is not None else None
        )
        if holyc_correct is False:
            holyc_errors.add(record_id)
        if llama_correct is False:
            llama_errors.add(record_id)
        rows.append(
            ErrorOverlapRow(
                record_id=record_id,
                dataset=gold_case.dataset,
                split=gold_case.split,
                answer_index=gold_case.answer_index,
                holyc_prediction=holyc_prediction.predicted_index if holyc_prediction is not None else None,
                llama_prediction=llama_prediction.predicted_index if llama_prediction is not None else None,
                holyc_correct=holyc_correct,
                llama_correct=llama_correct,
                error_class=classify_error(holyc_correct, llama_correct),
            )
        )

    shared_errors = holyc_errors & llama_errors
    holyc_unique_errors = holyc_errors - llama_errors
    llama_unique_errors = llama_errors - holyc_errors
    error_union = holyc_errors | llama_errors
    error_jaccard = len(shared_errors) / len(error_union) if error_union else 1.0
    unique_error_delta = len(holyc_unique_errors) - len(llama_unique_errors)

    if args.min_paired_records is not None and len(paired_ids) < args.min_paired_records:
        add_finding(
            findings,
            "paired",
            "",
            "min_paired_records",
            f"paired record count {len(paired_ids)} is below {args.min_paired_records}",
        )
    if args.min_error_jaccard is not None and error_jaccard < args.min_error_jaccard:
        add_finding(
            findings,
            "paired",
            "",
            "min_error_jaccard",
            f"error-set Jaccard {error_jaccard:.6f} is below {args.min_error_jaccard:.6f}",
        )
    if args.max_holyc_unique_error_excess is not None and unique_error_delta > args.max_holyc_unique_error_excess:
        add_finding(
            findings,
            "paired",
            "",
            "max_holyc_unique_error_excess",
            f"HolyC has {unique_error_delta} more unique errors than llama.cpp, above {args.max_holyc_unique_error_excess}",
        )

    summary = {
        "gold_records": len(gold),
        "paired_records": len(paired_ids),
        "holyc_errors": len(holyc_errors),
        "llama_errors": len(llama_errors),
        "shared_errors": len(shared_errors),
        "holyc_unique_errors": len(holyc_unique_errors),
        "llama_unique_errors": len(llama_unique_errors),
        "error_union": len(error_union),
        "error_jaccard": error_jaccard,
        "shared_error_pct_of_holyc_errors": pct(len(shared_errors), len(holyc_errors)),
        "shared_error_pct_of_llama_errors": pct(len(shared_errors), len(llama_errors)),
        "unique_error_delta_holyc_minus_llama": unique_error_delta,
        "findings": len(findings),
    }
    return {
        "format": "eval-error-overlap-audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {"gold": str(args.gold), "holyc": str(args.holyc), "llama": str(args.llama)},
        "gates": {
            "min_paired_records": args.min_paired_records,
            "min_error_jaccard": args.min_error_jaccard,
            "max_holyc_unique_error_excess": args.max_holyc_unique_error_excess,
        },
        "summary": summary,
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Error Overlap Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {summary['gold_records']} |",
        f"| Paired records | {summary['paired_records']} |",
        f"| HolyC errors | {summary['holyc_errors']} |",
        f"| llama.cpp errors | {summary['llama_errors']} |",
        f"| Shared errors | {summary['shared_errors']} |",
        f"| HolyC unique errors | {summary['holyc_unique_errors']} |",
        f"| llama.cpp unique errors | {summary['llama_unique_errors']} |",
        f"| Error Jaccard | {summary['error_jaccard']:.6f} |",
        f"| Findings | {summary['findings']} |",
        "",
        "## Error Rows",
        "",
        "| Record ID | Dataset | Answer | HolyC | llama.cpp | Class |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        if row["error_class"] == "both_correct":
            continue
        lines.append(
            f"| {row['record_id']} | {row['dataset']}/{row['split']} | {row['answer_index']} | "
            f"{row['holyc_prediction'] if row['holyc_prediction'] is not None else '-'} | "
            f"{row['llama_prediction'] if row['llama_prediction'] is not None else '-'} | {row['error_class']} |"
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(
            f"- {finding['source']} {finding['kind']} {finding['record_id'] or 'global'}: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No eval error-overlap findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ErrorOverlapRow.__dataclass_fields__)
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
    root = ET.Element(
        "testsuite",
        {"name": "holyc_eval_error_overlap_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    case = ET.SubElement(root, "testcase", {"name": "error_overlap"})
    if findings:
        failure = ET.SubElement(case, "failure", {"type": "eval_error_overlap_failure"})
        failure.text = "\n".join(
            f"{finding['source']} {finding['kind']} {finding['record_id']}: {finding['detail']}"
            for finding in findings
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--holyc", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--min-paired-records", type=int)
    parser.add_argument("--min-error-jaccard", type=float)
    parser.add_argument("--max-holyc-unique-error-excess", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_error_overlap_audit_latest")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_paired_records is not None and args.min_paired_records < 0:
        print("error: --min-paired-records must be non-negative", file=sys.stderr)
        return 2
    if args.min_error_jaccard is not None and not 0.0 <= args.min_error_jaccard <= 1.0:
        print("error: --min-error-jaccard must be between 0 and 1", file=sys.stderr)
        return 2
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
