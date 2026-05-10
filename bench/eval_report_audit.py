#!/usr/bin/env python3
"""Audit eval_compare JSON artifacts for internal consistency.

This host-side tool reads existing HolyC-vs-llama eval reports, recomputes core
summary counters from the row payloads, checks metric bounds, and emits JSON,
Markdown, CSV findings, and JUnit outputs. It never launches QEMU and never
touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportRecord:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    row_count: int
    summary_record_count: int | None
    holyc_accuracy: float | None
    recomputed_holyc_accuracy: float | None
    llama_accuracy: float | None
    recomputed_llama_accuracy: float | None
    agreement: float | None
    recomputed_agreement: float | None
    regressions: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    metric: str
    expected: str
    actual: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return None


def iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def close_enough(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance


def row_correct(row: dict[str, Any], engine: str) -> bool | None:
    explicit = as_bool(row.get(f"{engine}_correct"))
    if explicit is not None:
        return explicit
    prediction = as_int(row.get(f"{engine}_prediction"))
    answer = as_int(row.get("answer_index"))
    if prediction is None or answer is None:
        return None
    return prediction == answer


def row_agrees(row: dict[str, Any]) -> bool | None:
    holyc_prediction = as_int(row.get("holyc_prediction"))
    llama_prediction = as_int(row.get("llama_prediction"))
    if holyc_prediction is not None and llama_prediction is not None:
        return holyc_prediction == llama_prediction
    return as_bool(row.get("engines_agree"))


def fraction(count: int, total: int) -> float | None:
    return (count / total) if total else None


def metric(summary: dict[str, Any], key: str) -> float | None:
    return as_float(summary.get(key))


def load_report(path: Path, tolerance: float) -> tuple[ReportRecord, list[Finding]]:
    findings: list[Finding] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("root must be an object")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("missing rows array")
        row_dicts = [row for row in rows if isinstance(row, dict)]
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("missing summary object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        record = ReportRecord(str(path), "invalid", "", "", "", "", 0, None, None, None, None, None, None, None, 0, str(exc))
        findings.append(Finding(str(path), "error", "invalid_report", "report", "valid eval_compare JSON", "", str(exc)))
        return record, findings

    row_count = len(row_dicts)
    holyc_correct = sum(1 for row in row_dicts if row_correct(row, "holyc") is True)
    llama_correct = sum(1 for row in row_dicts if row_correct(row, "llama") is True)
    agreements = sum(1 for row in row_dicts if row_agrees(row) is True)

    recomputed_holyc_accuracy = fraction(holyc_correct, row_count)
    recomputed_llama_accuracy = fraction(llama_correct, row_count)
    recomputed_agreement = fraction(agreements, row_count)
    summary_record_count = as_int(summary.get("record_count"))
    holyc_accuracy = metric(summary, "holyc_accuracy")
    llama_accuracy = metric(summary, "llama_accuracy")
    agreement = metric(summary, "agreement")

    record = ReportRecord(
        source=str(path),
        status=str(payload.get("status") or "").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        row_count=row_count,
        summary_record_count=summary_record_count,
        holyc_accuracy=holyc_accuracy,
        recomputed_holyc_accuracy=recomputed_holyc_accuracy,
        llama_accuracy=llama_accuracy,
        recomputed_llama_accuracy=recomputed_llama_accuracy,
        agreement=agreement,
        recomputed_agreement=recomputed_agreement,
        regressions=len(payload.get("regressions") or []),
    )

    if len(rows) != row_count:
        findings.append(
            Finding(str(path), "error", "invalid_row", "rows", "all rows are objects", str(len(rows) - row_count), "non-object row(s) present")
        )
    if summary_record_count != row_count:
        findings.append(
            Finding(str(path), "error", "summary_mismatch", "record_count", str(row_count), str(summary_record_count), "summary record_count differs from rows length")
        )
    expected_metrics = {
        "holyc_accuracy": (holyc_accuracy, recomputed_holyc_accuracy),
        "llama_accuracy": (llama_accuracy, recomputed_llama_accuracy),
        "agreement": (agreement, recomputed_agreement),
    }
    for name, (actual, expected) in expected_metrics.items():
        if actual is None:
            findings.append(Finding(str(path), "error", "missing_metric", name, "present finite float", "", f"summary.{name} is missing or non-finite"))
        elif actual < 0.0 or actual > 1.0:
            findings.append(Finding(str(path), "error", "metric_out_of_bounds", name, "[0,1]", str(actual), f"summary.{name} is outside [0,1]"))
        elif not close_enough(actual, expected, tolerance):
            findings.append(Finding(str(path), "error", "summary_mismatch", name, str(expected), str(actual), f"summary.{name} differs from row recomputation"))

    if record.status == "pass" and record.regressions:
        findings.append(
            Finding(str(path), "error", "status_regression_mismatch", "status", "non-pass when regressions exist", "pass", "report status is pass despite regression entries")
        )

    return record, findings


def build_report(paths: list[Path], tolerance: float, min_reports: int) -> dict[str, Any]:
    records: list[ReportRecord] = []
    findings: list[Finding] = []
    files = iter_input_files(paths)
    for path in files:
        record, path_findings = load_report(path, tolerance)
        records.append(record)
        findings.extend(path_findings)
    if len(records) < min_reports:
        findings.append(
            Finding("", "error", "min_reports", "reports", str(min_reports), str(len(records)), f"found {len(records)} report(s), below minimum {min_reports}")
        )
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "reports": len(records),
            "findings": len(findings),
            "rows": sum(record.row_count for record in records),
            "invalid_reports": sum(1 for record in records if record.status == "invalid"),
        },
        "reports": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        "# Eval Report Audit",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {summary['reports']}",
        f"- Rows: {summary['rows']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Source | Status | Rows | HolyC Acc | llama Acc | Agreement | Findings |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_source: dict[str, int] = {}
    for finding in report["findings"]:
        by_source[finding["source"]] = by_source.get(finding["source"], 0) + 1
    for record in report["reports"]:
        lines.append(
            f"| {record['source']} | {record['status']} | {record['row_count']} | "
            f"{record['holyc_accuracy']} | {record['llama_accuracy']} | {record['agreement']} | "
            f"{by_source.get(record['source'], 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report[key]
    if key == "reports":
        fieldnames = list(ReportRecord.__dataclass_fields__)
    else:
        fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_eval_report_audit", "tests": "1", "failures": str(int(bool(findings))), "errors": "0"},
    )
    case = ET.SubElement(suite, "testcase", {"name": "eval_report_consistency"})
    if findings:
        failure = ET.SubElement(case, "failure", {"type": "eval_report_audit", "message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding['source']}: {finding['kind']}: {finding['detail']}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--output", type=Path, default=Path("bench/results/eval_report_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("bench/results/eval_report_audit_latest.md"))
    parser.add_argument("--csv", type=Path, default=Path("bench/results/eval_report_audit_latest.csv"))
    parser.add_argument("--findings-csv", type=Path, default=Path("bench/results/eval_report_audit_findings_latest.csv"))
    parser.add_argument("--junit", type=Path, default=Path("bench/results/eval_report_audit_latest_junit.xml"))
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args.inputs, args.tolerance, args.min_reports)
    write_json(args.output, report)
    write_markdown(args.markdown, report)
    write_csv(args.csv, report, "reports")
    write_csv(args.findings_csv, report, "findings")
    write_junit(args.junit, report)
    print(f"status={report['status']}")
    print(f"reports={report['summary']['reports']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if args.fail_on_findings and report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
