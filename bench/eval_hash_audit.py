#!/usr/bin/env python3
"""Audit eval_compare artifact hashes for strict local reproducibility.

This host-side tool reads existing HolyC-vs-llama eval_compare JSON reports and
checks that dataset/prediction fingerprints are valid SHA-256 values. It can
optionally compare the recorded gold hash against a local gold dataset file.
It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_FIELDS = ("gold_sha256", "holyc_predictions_sha256", "llama_predictions_sha256")


@dataclass(frozen=True)
class HashRecord:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    record_count: int
    gold_sha256: str
    holyc_predictions_sha256: str
    llama_predictions_sha256: str
    error: str = ""


@dataclass(frozen=True)
class Finding:
    severity: str
    gate: str
    source: str
    field: str
    value: str
    expected: str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def load_record(path: Path) -> HashRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return HashRecord(str(path), "missing", "", "", "", "", 0, "", "", "", str(exc))
    except json.JSONDecodeError as exc:
        return HashRecord(str(path), "invalid", "", "", "", "", 0, "", "", "", f"invalid json: {exc}")
    if not isinstance(payload, dict):
        return HashRecord(str(path), "invalid", "", "", "", "", 0, "", "", "", "root must be an object")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return HashRecord(
        source=str(path),
        status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        record_count=as_int(summary.get("record_count")),
        gold_sha256=str(payload.get("gold_sha256") or ""),
        holyc_predictions_sha256=str(payload.get("holyc_predictions_sha256") or ""),
        llama_predictions_sha256=str(payload.get("llama_predictions_sha256") or ""),
    )


def validate_hash_field(record: HashRecord, field: str) -> Finding | None:
    value = getattr(record, field)
    if not value:
        return Finding("error", "missing_hash", record.source, field, "", "64 lowercase hex chars", f"{field} is missing")
    if not SHA256_RE.fullmatch(value):
        return Finding(
            "error",
            "invalid_hash_format",
            record.source,
            field,
            value,
            "64 lowercase hex chars",
            f"{field} is not a canonical SHA-256 hex digest",
        )
    return None


def evaluate(records: list[HashRecord], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    if len(records) < args.min_reports:
        findings.append(
            Finding("error", "min_reports", "", "reports", str(len(records)), str(args.min_reports), "too few eval reports")
        )
    expected_gold_sha256 = file_sha256(args.gold_path) if args.gold_path else ""
    for record in records:
        if record.status in {"missing", "invalid"}:
            findings.append(Finding("error", record.status, record.source, "report", record.status, "valid", record.error))
            continue
        if args.fail_on_failed_reports and record.status != "pass":
            findings.append(
                Finding("error", "report_status", record.source, "status", record.status, "pass", "eval report did not pass")
            )
        for field in HASH_FIELDS:
            finding = validate_hash_field(record, field)
            if finding:
                findings.append(finding)
        if expected_gold_sha256 and record.gold_sha256 and record.gold_sha256 != expected_gold_sha256:
            findings.append(
                Finding(
                    "error",
                    "gold_hash_mismatch",
                    record.source,
                    "gold_sha256",
                    record.gold_sha256,
                    expected_gold_sha256,
                    "recorded gold hash does not match local gold file",
                )
            )
        if not args.allow_identical_engine_hashes and record.holyc_predictions_sha256 == record.llama_predictions_sha256:
            findings.append(
                Finding(
                    "warning",
                    "identical_engine_hashes",
                    record.source,
                    "holyc_predictions_sha256,llama_predictions_sha256",
                    record.holyc_predictions_sha256,
                    "distinct prediction artifact hashes",
                    "HolyC and llama prediction artifacts have identical hashes",
                )
            )
    return findings


def blocking_findings(findings: list[Finding], *, fail_on_warnings: bool) -> list[Finding]:
    if fail_on_warnings:
        return findings
    return [finding for finding in findings if finding.severity == "error"]


def build_report(records: list[HashRecord], findings: list[Finding], *, fail_on_warnings: bool = False) -> dict[str, Any]:
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    blocking = blocking_findings(findings, fail_on_warnings=fail_on_warnings)
    return {
        "generated_at": iso_now(),
        "status": "fail" if blocking else "pass",
        "summary": {
            "reports": len(records),
            "rows": sum(record.record_count for record in records),
            "findings": len(findings),
            "errors": len(errors),
            "warnings": len(warnings),
            "blocking_findings": len(blocking),
            "dataset_splits": len({(record.dataset, record.split) for record in records if record.dataset or record.split}),
        },
        "reports": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval Hash Audit",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {report['summary']['reports']}",
        f"- Rows: {report['summary']['rows']}",
        f"- Findings: {report['summary']['findings']}",
        "",
        "| Source | Status | Dataset | Split | Model | Quantization | Rows |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for record in report["reports"]:
        lines.append(
            f"| {record['source']} | {record['status']} | {record['dataset']} | {record['split']} | "
            f"{record['model']} | {record['quantization']} | {record['record_count']} |"
        )
    if report["findings"]:
        lines.extend(["", "## Findings", "", "| Severity | Gate | Source | Field | Expected | Value |", "| --- | --- | --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['gate']} | {finding['source']} | {finding['field']} | "
                f"{finding['expected']} | {finding['value']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    findings = report["findings"]
    blocking_count = int(report["summary"].get("blocking_findings", len(findings)))
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_eval_hash_audit", "tests": "1", "failures": str(int(bool(blocking_count))), "errors": "0"},
    )
    case = ET.SubElement(suite, "testcase", {"name": "eval_hash_fingerprints"})
    if blocking_count:
        failure = ET.SubElement(case, "failure", {"type": "eval_hash_audit", "message": f"{blocking_count} blocking finding(s)"})
        failure.text = "\n".join(f"{finding['source']}: {finding['gate']}: {finding['message']}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--gold-path", type=Path, help="optional local gold JSONL/file to compare against gold_sha256")
    parser.add_argument("--output", type=Path, default=Path("bench/results/eval_hash_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("bench/results/eval_hash_audit_latest.md"))
    parser.add_argument("--csv", type=Path, default=Path("bench/results/eval_hash_audit_latest.csv"))
    parser.add_argument("--findings-csv", type=Path, default=Path("bench/results/eval_hash_audit_findings_latest.csv"))
    parser.add_argument("--junit", type=Path, default=Path("bench/results/eval_hash_audit_latest_junit.xml"))
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--fail-on-failed-reports", action="store_true")
    parser.add_argument("--allow-identical-engine-hashes", action="store_true")
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records = [load_record(path) for path in iter_input_files(args.inputs)]
    findings = evaluate(records, args)
    report = build_report(records, findings, fail_on_warnings=args.fail_on_warnings)
    write_json(args.output, report)
    write_markdown(args.markdown, report)
    write_csv(args.csv, [asdict(record) for record in records], list(HashRecord.__dataclass_fields__))
    write_csv(args.findings_csv, [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    write_junit(args.junit, report)
    print(f"status={report['status']}")
    print(f"reports={report['summary']['reports']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if args.fail_on_findings and blocking_findings(findings, fail_on_warnings=args.fail_on_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
