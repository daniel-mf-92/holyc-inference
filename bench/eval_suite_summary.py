#!/usr/bin/env python3
"""Summarize HolyC-vs-llama eval comparison artifacts for CI.

This host-side tool consumes existing eval_compare JSON reports and writes a
compact suite-level JSON/CSV/Markdown/JUnit summary. It never launches QEMU and
does not touch the TempleOS guest.
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
class SuiteRecord:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    record_count: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    accuracy_delta_holyc_minus_llama: float | None
    agreement: float | None
    regressions: int
    gold_sha256: str
    holyc_predictions_sha256: str
    llama_predictions_sha256: str
    error: str = ""


@dataclass(frozen=True)
class SuiteFinding:
    gate: str
    source: str
    value: float | int | str | None
    threshold: float | int | str | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def load_record(path: Path) -> SuiteRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return SuiteRecord(str(path), "missing", "", "", "", "", 0, None, None, None, None, 0, "", "", "", str(exc))
    except json.JSONDecodeError as exc:
        return SuiteRecord(
            str(path),
            "invalid",
            "",
            "",
            "",
            "",
            0,
            None,
            None,
            None,
            None,
            0,
            "",
            "",
            "",
            f"invalid json: {exc}",
        )
    if not isinstance(payload, dict):
        return SuiteRecord(str(path), "invalid", "", "", "", "", 0, None, None, None, None, 0, "", "", "", "root must be an object")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return SuiteRecord(
            str(path),
            "invalid",
            str(payload.get("dataset") or ""),
            str(payload.get("split") or ""),
            str(payload.get("model") or ""),
            str(payload.get("quantization") or ""),
            0,
            None,
            None,
            None,
            None,
            0,
            str(payload.get("gold_sha256") or ""),
            str(payload.get("holyc_predictions_sha256") or ""),
            str(payload.get("llama_predictions_sha256") or ""),
            "missing summary object",
        )
    return SuiteRecord(
        source=str(path),
        status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        record_count=as_int(summary.get("record_count")),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        accuracy_delta_holyc_minus_llama=as_float(summary.get("accuracy_delta_holyc_minus_llama")),
        agreement=as_float(summary.get("agreement")),
        regressions=len(payload.get("regressions") or []),
        gold_sha256=str(payload.get("gold_sha256") or ""),
        holyc_predictions_sha256=str(payload.get("holyc_predictions_sha256") or ""),
        llama_predictions_sha256=str(payload.get("llama_predictions_sha256") or ""),
    )


def evaluate(records: list[SuiteRecord], args: argparse.Namespace) -> list[SuiteFinding]:
    findings: list[SuiteFinding] = []
    if len(records) < args.min_reports:
        findings.append(
            SuiteFinding(
                "min_reports",
                "",
                len(records),
                args.min_reports,
                f"found {len(records)} report(s), below minimum {args.min_reports}",
            )
        )
    if sum(record.record_count for record in records) < args.min_records:
        total = sum(record.record_count for record in records)
        findings.append(
            SuiteFinding(
                "min_records",
                "",
                total,
                args.min_records,
                f"found {total} eval row(s), below minimum {args.min_records}",
            )
        )
    coverage_sets = {
        "dataset": {record.dataset for record in records if record.dataset},
        "split": {record.split for record in records if record.split},
        "model": {record.model for record in records if record.model},
        "quantization": {record.quantization for record in records if record.quantization},
    }
    coverage_requirements = {
        "dataset": args.require_dataset,
        "split": args.require_split,
        "model": args.require_model,
        "quantization": args.require_quantization,
    }
    for field, required_values in coverage_requirements.items():
        present = coverage_sets[field]
        for required in required_values:
            if required not in present:
                findings.append(
                    SuiteFinding(
                        f"required_{field}",
                        "",
                        ",".join(sorted(present)),
                        required,
                        f"required {field} {required!r} is missing from eval suite reports",
                    )
                )
    for record in records:
        if record.status in {"missing", "invalid"}:
            findings.append(SuiteFinding(record.status, record.source, record.status, "valid", record.error))
            continue
        if args.fail_on_failed_reports and record.status != "pass":
            findings.append(
                SuiteFinding(
                    "report_status",
                    record.source,
                    record.status,
                    "pass",
                    f"{record.source} status is {record.status}",
                )
            )
        if args.fail_on_regressions and record.regressions:
            findings.append(
                SuiteFinding(
                    "regressions",
                    record.source,
                    record.regressions,
                    0,
                    f"{record.source} has {record.regressions} quality regression(s)",
                )
            )
        if args.min_holyc_accuracy is not None and (
            record.holyc_accuracy is None or record.holyc_accuracy < args.min_holyc_accuracy
        ):
            findings.append(
                SuiteFinding(
                    "min_holyc_accuracy",
                    record.source,
                    record.holyc_accuracy,
                    args.min_holyc_accuracy,
                    f"{record.source} HolyC accuracy is below threshold",
                )
            )
        if args.min_agreement is not None and (record.agreement is None or record.agreement < args.min_agreement):
            findings.append(
                SuiteFinding(
                    "min_agreement",
                    record.source,
                    record.agreement,
                    args.min_agreement,
                    f"{record.source} HolyC-vs-llama agreement is below threshold",
                )
            )
    return findings


def build_report(records: list[SuiteRecord], findings: list[SuiteFinding]) -> dict[str, Any]:
    total_records = sum(record.record_count for record in records)
    weighted_accuracy = None
    weighted_agreement = None
    if total_records:
        weighted_accuracy = sum((record.holyc_accuracy or 0.0) * record.record_count for record in records) / total_records
        weighted_agreement = sum((record.agreement or 0.0) * record.record_count for record in records) / total_records
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "reports": len(records),
            "records": total_records,
            "findings": len(findings),
            "regressions": sum(record.regressions for record in records),
            "weighted_holyc_accuracy": weighted_accuracy,
            "weighted_agreement": weighted_agreement,
            "status_counts": status_counts,
        },
        "reports": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[SuiteRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SuiteRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[SuiteFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SuiteFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Suite Summary",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {summary['reports']}",
        f"- Records: {summary['records']}",
        f"- Regressions: {summary['regressions']}",
        f"- Weighted HolyC accuracy: {summary['weighted_holyc_accuracy']}",
        f"- Weighted agreement: {summary['weighted_agreement']}",
        "",
        "## Reports",
        "",
        "| Source | Status | Dataset | Split | Model | Quant | Records | HolyC acc | Llama acc | Agreement | Regressions |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in report["reports"]:
        lines.append(
            "| {source} | {status} | {dataset} | {split} | {model} | {quantization} | {record_count} | "
            "{holyc_accuracy} | {llama_accuracy} | {agreement} | {regressions} |".format(**record)
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    else:
        lines.append("No suite gate findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[SuiteFinding]) -> None:
    root = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_suite_summary",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(root, "testcase", {"name": "eval_suite_gates"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_suite_summary_failure"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_suite_summary_latest")
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-holyc-accuracy", type=float)
    parser.add_argument("--min-agreement", type=float)
    parser.add_argument("--require-dataset", action="append", default=[], help="Require at least one report for this dataset")
    parser.add_argument("--require-split", action="append", default=[], help="Require at least one report for this split")
    parser.add_argument("--require-model", action="append", default=[], help="Require at least one report for this model")
    parser.add_argument("--require-quantization", action="append", default=[], help="Require at least one report for this quantization")
    parser.add_argument("--fail-on-failed-reports", action="store_true")
    parser.add_argument("--fail-on-regressions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    records = [load_record(path) for path in iter_input_files(args.inputs)]
    findings = evaluate(records, args)
    report = build_report(records, findings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
