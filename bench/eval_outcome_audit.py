#!/usr/bin/env python3
"""Audit HolyC-vs-llama eval outcome buckets from eval_compare reports.

This host-side tool consumes local eval_compare JSON artifacts, groups rows by
paired correctness outcome, and emits dashboard-friendly JSON, Markdown, CSV,
findings CSV, and JUnit outputs. It never launches QEMU and never fetches data.
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


@dataclass(frozen=True)
class ScopeRow:
    source: str
    scope: str
    dataset: str
    split: str
    record_count: int
    both_correct: int
    holyc_only_correct: int
    llama_only_correct: int
    both_wrong_same: int
    both_wrong_disagree: int
    holyc_accuracy: float
    llama_accuracy: float
    llama_only_correct_pct: float
    both_wrong_pct: float


@dataclass(frozen=True)
class Finding:
    source: str
    scope: str
    dataset: str
    split: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def pct(count: int, total: int) -> float:
    return (count / total * 100.0) if total else 0.0


def outcome(row: dict[str, Any]) -> str:
    holyc_correct = as_bool(row.get("holyc_correct"))
    llama_correct = as_bool(row.get("llama_correct"))
    if holyc_correct and llama_correct:
        return "both_correct"
    if holyc_correct and not llama_correct:
        return "holyc_only_correct"
    if llama_correct and not holyc_correct:
        return "llama_only_correct"
    holyc_prediction = as_int(row.get("holyc_prediction"))
    llama_prediction = as_int(row.get("llama_prediction"))
    return "both_wrong_same" if holyc_prediction == llama_prediction else "both_wrong_disagree"


def scope_key(row: dict[str, Any], fallback_dataset: str, fallback_split: str) -> tuple[str, str]:
    dataset = str(row.get("dataset") or fallback_dataset or "")
    split = str(row.get("split") or fallback_split or "")
    return dataset, split


def summarize_scope(source: Path, scope: str, dataset: str, split: str, rows: list[dict[str, Any]]) -> ScopeRow:
    counts = {
        "both_correct": 0,
        "holyc_only_correct": 0,
        "llama_only_correct": 0,
        "both_wrong_same": 0,
        "both_wrong_disagree": 0,
    }
    for row in rows:
        counts[outcome(row)] += 1
    total = len(rows)
    holyc_correct = counts["both_correct"] + counts["holyc_only_correct"]
    llama_correct = counts["both_correct"] + counts["llama_only_correct"]
    both_wrong = counts["both_wrong_same"] + counts["both_wrong_disagree"]
    return ScopeRow(
        source=str(source),
        scope=scope,
        dataset=dataset,
        split=split,
        record_count=total,
        both_correct=counts["both_correct"],
        holyc_only_correct=counts["holyc_only_correct"],
        llama_only_correct=counts["llama_only_correct"],
        both_wrong_same=counts["both_wrong_same"],
        both_wrong_disagree=counts["both_wrong_disagree"],
        holyc_accuracy=(holyc_correct / total) if total else 0.0,
        llama_accuracy=(llama_correct / total) if total else 0.0,
        llama_only_correct_pct=pct(counts["llama_only_correct"], total),
        both_wrong_pct=pct(both_wrong, total),
    )


def rows_from_report(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows array")
    return [row for row in rows if isinstance(row, dict)]


def report_scopes(path: Path, report: dict[str, Any]) -> list[ScopeRow]:
    rows = rows_from_report(path, report)
    fallback_dataset = str(report.get("dataset") or "")
    fallback_split = str(report.get("split") or "")
    scopes = [summarize_scope(path, "overall", fallback_dataset, fallback_split, rows)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(scope_key(row, fallback_dataset, fallback_split), []).append(row)
    for (dataset, split), group_rows in sorted(grouped.items()):
        scopes.append(summarize_scope(path, "dataset_split", dataset, split, group_rows))
    return scopes


def append_gate_findings(
    findings: list[Finding],
    rows: list[ScopeRow],
    *,
    min_records: int,
    max_llama_only_correct_pct: float | None,
    max_dataset_split_llama_only_correct_pct: float | None,
    max_both_wrong_pct: float | None,
) -> None:
    for row in rows:
        if row.scope == "overall" and row.record_count < min_records:
            findings.append(
                Finding(
                    row.source,
                    row.scope,
                    row.dataset,
                    row.split,
                    "error",
                    "insufficient_records",
                    f"record_count={row.record_count} min_records={min_records}",
                )
            )
        llama_threshold = (
            max_llama_only_correct_pct
            if row.scope == "overall"
            else max_dataset_split_llama_only_correct_pct
        )
        if llama_threshold is not None and row.llama_only_correct_pct > llama_threshold:
            findings.append(
                Finding(
                    row.source,
                    row.scope,
                    row.dataset,
                    row.split,
                    "error",
                    "llama_only_correct_pct_exceeded",
                    f"llama_only_correct_pct={row.llama_only_correct_pct:.6g} threshold={llama_threshold:.6g}",
                )
            )
        if row.scope == "overall" and max_both_wrong_pct is not None and row.both_wrong_pct > max_both_wrong_pct:
            findings.append(
                Finding(
                    row.source,
                    row.scope,
                    row.dataset,
                    row.split,
                    "error",
                    "both_wrong_pct_exceeded",
                    f"both_wrong_pct={row.both_wrong_pct:.6g} threshold={max_both_wrong_pct:.6g}",
                )
            )


def audit_reports(
    paths: list[Path],
    *,
    min_records: int,
    max_llama_only_correct_pct: float | None,
    max_dataset_split_llama_only_correct_pct: float | None,
    max_both_wrong_pct: float | None,
) -> dict[str, Any]:
    rows: list[ScopeRow] = []
    findings: list[Finding] = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError("root must be an object")
            rows.extend(report_scopes(path, report))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), "file", "", "", "error", "unreadable_report", str(exc)))

    append_gate_findings(
        findings,
        rows,
        min_records=min_records,
        max_llama_only_correct_pct=max_llama_only_correct_pct,
        max_dataset_split_llama_only_correct_pct=max_dataset_split_llama_only_correct_pct,
        max_both_wrong_pct=max_both_wrong_pct,
    )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": [str(path) for path in paths],
        "min_records": min_records,
        "max_llama_only_correct_pct": max_llama_only_correct_pct,
        "max_dataset_split_llama_only_correct_pct": max_dataset_split_llama_only_correct_pct,
        "max_both_wrong_pct": max_both_wrong_pct,
        "scope_count": len(rows),
        "error_count": error_count,
        "scopes": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "scope",
        "dataset",
        "split",
        "record_count",
        "both_correct",
        "holyc_only_correct",
        "llama_only_correct",
        "both_wrong_same",
        "both_wrong_disagree",
        "holyc_accuracy",
        "llama_accuracy",
        "llama_only_correct_pct",
        "both_wrong_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "scope", "dataset", "split", "severity", "kind", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow({name: finding.get(name, "") for name in fieldnames})


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Outcome Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {len(report['inputs'])}",
        f"Scopes: {report['scope_count']}",
        "",
        "| Scope | Dataset | Split | Records | Both correct | HolyC-only correct | llama-only correct | Both wrong % |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scopes"]:
        lines.append(
            "| {scope} | {dataset} | {split} | {records} | {both} | {holyc_only} | {llama_only} | {both_wrong:.4f} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                split=row["split"],
                records=row["record_count"],
                both=row["both_correct"],
                holyc_only=row["holyc_only_correct"],
                llama_only=row["llama_only_correct"],
                both_wrong=row["both_wrong_pct"],
            )
        )
    lines.append("")
    if report["findings"]:
        lines.extend(["| Severity | Kind | Scope | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            scope = ":".join(part for part in (finding["scope"], finding["dataset"], finding["split"]) if part)
            lines.append(f"| {finding['severity']} | {finding['kind']} | {scope} | {finding['detail']} |")
    else:
        lines.append("No eval outcome gate findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 1 if report["status"] == "fail" else 0
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_outcome_audit",
            "tests": "1",
            "failures": str(failures),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "outcome_bucket_gates"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": "eval outcome audit failed"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON report(s)")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--markdown", type=Path, help="Markdown output path")
    parser.add_argument("--csv", type=Path, help="scope CSV output path")
    parser.add_argument("--findings-csv", type=Path, help="findings CSV output path")
    parser.add_argument("--junit", type=Path, help="JUnit XML output path")
    parser.add_argument("--min-records", type=int, default=1, help="minimum overall records per report")
    parser.add_argument("--max-llama-only-correct-pct", type=float, help="maximum overall llama-only-correct pct")
    parser.add_argument(
        "--max-dataset-split-llama-only-correct-pct",
        type=float,
        help="maximum dataset/split llama-only-correct pct",
    )
    parser.add_argument("--max-both-wrong-pct", type=float, help="maximum overall both-wrong percentage")
    parser.add_argument("--fail-on-findings", action="store_true", help="return nonzero when findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_reports(
        args.inputs,
        min_records=args.min_records,
        max_llama_only_correct_pct=args.max_llama_only_correct_pct,
        max_dataset_split_llama_only_correct_pct=args.max_dataset_split_llama_only_correct_pct,
        max_both_wrong_pct=args.max_both_wrong_pct,
    )
    write_json(args.output, report)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_csv(args.csv, report["scopes"])
    if args.findings_csv:
        write_findings_csv(args.findings_csv, report["findings"])
    if args.junit:
        write_junit(args.junit, report)
    return 1 if args.fail_on_findings and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
