#!/usr/bin/env python3
"""Audit HolyC vs llama.cpp disagreement rates from eval_compare reports.

This host-side tool consumes local eval_compare JSON artifacts, summarizes
overall and dataset/split disagreement rates, and emits dashboard-friendly JSON,
Markdown, CSV, findings CSV, and JUnit outputs. It never launches QEMU and never
fetches remote data.
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
    agreement_count: int
    disagreement_count: int
    disagreement_pct: float
    holyc_accuracy: float
    llama_accuracy: float
    accuracy_delta_holyc_minus_llama: float


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


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def disagreement_pct(record_count: int, agreement_count: int) -> float:
    if record_count <= 0:
        return 0.0
    return (record_count - agreement_count) / record_count * 100.0


def scope_from_summary(path: Path, report: dict[str, Any]) -> ScopeRow:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    record_count = as_int(summary.get("record_count"))
    agreement_count = as_int(summary.get("agreement_count"), round(as_float(summary.get("agreement")) * record_count))
    return ScopeRow(
        source=str(path),
        scope="overall",
        dataset=str(report.get("dataset") or ""),
        split=str(report.get("split") or ""),
        record_count=record_count,
        agreement_count=agreement_count,
        disagreement_count=max(0, record_count - agreement_count),
        disagreement_pct=disagreement_pct(record_count, agreement_count),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        accuracy_delta_holyc_minus_llama=as_float(summary.get("accuracy_delta_holyc_minus_llama")),
    )


def scopes_from_breakdown(path: Path, report: dict[str, Any]) -> list[ScopeRow]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    rows = summary.get("dataset_breakdown")
    if not isinstance(rows, list):
        return []
    scopes: list[ScopeRow] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        record_count = as_int(row.get("record_count"))
        agreement_count = as_int(row.get("agreement_count"), round(as_float(row.get("agreement")) * record_count))
        scopes.append(
            ScopeRow(
                source=str(path),
                scope="dataset_split",
                dataset=str(row.get("dataset") or ""),
                split=str(row.get("split") or ""),
                record_count=record_count,
                agreement_count=agreement_count,
                disagreement_count=max(0, record_count - agreement_count),
                disagreement_pct=disagreement_pct(record_count, agreement_count),
                holyc_accuracy=as_float(row.get("holyc_accuracy")),
                llama_accuracy=as_float(row.get("llama_accuracy")),
                accuracy_delta_holyc_minus_llama=as_float(row.get("accuracy_delta_holyc_minus_llama")),
            )
        )
    return scopes


def append_gate_findings(
    findings: list[Finding],
    rows: list[ScopeRow],
    *,
    min_records: int,
    max_disagreement_pct: float | None,
    max_dataset_split_disagreement_pct: float | None,
) -> None:
    for row in rows:
        if row.scope == "overall" and row.record_count < min_records:
            findings.append(
                Finding(
                    source=row.source,
                    scope=row.scope,
                    dataset=row.dataset,
                    split=row.split,
                    severity="error",
                    kind="insufficient_records",
                    detail=f"record_count={row.record_count} min_records={min_records}",
                )
            )
        threshold = max_disagreement_pct if row.scope == "overall" else max_dataset_split_disagreement_pct
        if threshold is not None and row.disagreement_pct > threshold:
            findings.append(
                Finding(
                    source=row.source,
                    scope=row.scope,
                    dataset=row.dataset,
                    split=row.split,
                    severity="error",
                    kind="disagreement_pct_exceeded",
                    detail=f"disagreement_pct={row.disagreement_pct:.6g} threshold={threshold:.6g}",
                )
            )


def audit_reports(
    paths: list[Path],
    *,
    min_records: int,
    max_disagreement_pct: float | None,
    max_dataset_split_disagreement_pct: float | None,
) -> dict[str, Any]:
    rows: list[ScopeRow] = []
    findings: list[Finding] = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(
                Finding(str(path), "file", "", "", "error", "unreadable_report", f"cannot read eval report: {exc}")
            )
            continue
        rows.append(scope_from_summary(path, report))
        rows.extend(scopes_from_breakdown(path, report))

    append_gate_findings(
        findings,
        rows,
        min_records=min_records,
        max_disagreement_pct=max_disagreement_pct,
        max_dataset_split_disagreement_pct=max_dataset_split_disagreement_pct,
    )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": [str(path) for path in paths],
        "min_records": min_records,
        "max_disagreement_pct": max_disagreement_pct,
        "max_dataset_split_disagreement_pct": max_dataset_split_disagreement_pct,
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
        "agreement_count",
        "disagreement_count",
        "disagreement_pct",
        "holyc_accuracy",
        "llama_accuracy",
        "accuracy_delta_holyc_minus_llama",
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
        "# Eval Disagreement Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {len(report['inputs'])}",
        f"Scopes: {report['scope_count']}",
        "",
        "| Scope | Dataset | Split | Records | Disagreement % | HolyC Acc | llama Acc |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scopes"]:
        lines.append(
            "| {scope} | {dataset} | {split} | {records} | {disagreement:.4f} | {holyc:.4f} | {llama:.4f} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                split=row["split"],
                records=row["record_count"],
                disagreement=row["disagreement_pct"],
                holyc=row["holyc_accuracy"],
                llama=row["llama_accuracy"],
            )
        )
    lines.append("")
    if report["findings"]:
        lines.extend(["| Severity | Kind | Scope | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            scope = ":".join(part for part in (finding["scope"], finding["dataset"], finding["split"]) if part)
            lines.append(f"| {finding['severity']} | {finding['kind']} | {scope} | {finding['detail']} |")
    else:
        lines.append("No disagreement gate findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 1 if report["status"] == "fail" else 0
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_disagreement_audit",
            "tests": "1",
            "failures": str(failures),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "disagreement_gates"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": "eval disagreement audit failed"})
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
    parser.add_argument("--max-disagreement-pct", type=float, help="maximum overall disagreement percentage")
    parser.add_argument(
        "--max-dataset-split-disagreement-pct",
        type=float,
        help="maximum dataset/split disagreement percentage",
    )
    parser.add_argument("--fail-on-findings", action="store_true", help="return nonzero when findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_reports(
        args.inputs,
        min_records=args.min_records,
        max_disagreement_pct=args.max_disagreement_pct,
        max_dataset_split_disagreement_pct=args.max_dataset_split_disagreement_pct,
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
