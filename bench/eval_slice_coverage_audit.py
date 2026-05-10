#!/usr/bin/env python3
"""Audit dataset/split coverage in HolyC-vs-llama eval comparison reports.

This host-side tool consumes existing eval_compare JSON artifacts and checks
that required eval slices are present with enough rows. It never launches QEMU,
never touches the TempleOS guest, and uses only local files.
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
from typing import Any, Iterable


@dataclass(frozen=True)
class SliceRecord:
    source: str
    report_status: str
    dataset: str
    split: str
    model: str
    quantization: str
    record_count: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    agreement: float | None
    regressions: int


@dataclass(frozen=True)
class Finding:
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


def iter_input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def parse_required_slices(values: Iterable[str]) -> list[tuple[str, str]]:
    required: list[tuple[str, str]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"required slice {value!r} must use DATASET:SPLIT")
        dataset, split = value.split(":", 1)
        dataset = dataset.strip()
        split = split.strip()
        if not dataset or not split:
            raise ValueError(f"required slice {value!r} must use DATASET:SPLIT")
        required.append((dataset, split))
    return required


def row_records(payload: dict[str, Any], source: Path) -> list[SliceRecord]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return []
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset") or payload.get("dataset") or "")
        split = str(row.get("split") or payload.get("split") or "")
        key = (dataset, split)
        bucket = counts.setdefault(key, {"records": 0, "holyc_correct": 0, "llama_correct": 0, "agree": 0})
        bucket["records"] += 1
        bucket["holyc_correct"] += int(bool(row.get("holyc_correct")))
        bucket["llama_correct"] += int(bool(row.get("llama_correct")))
        bucket["agree"] += int(bool(row.get("engines_agree")))
    records: list[SliceRecord] = []
    for (dataset, split), count in sorted(counts.items()):
        total = count["records"]
        records.append(
            SliceRecord(
                source=str(source),
                report_status=str(payload.get("status") or "pass").lower(),
                dataset=dataset,
                split=split,
                model=str(payload.get("model") or ""),
                quantization=str(payload.get("quantization") or ""),
                record_count=total,
                holyc_accuracy=count["holyc_correct"] / total if total else None,
                llama_accuracy=count["llama_correct"] / total if total else None,
                agreement=count["agree"] / total if total else None,
                regressions=len(payload.get("regressions") or []),
            )
        )
    return records


def summary_records(payload: dict[str, Any], source: Path) -> list[SliceRecord]:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []
    breakdown = summary.get("dataset_breakdown")
    if not isinstance(breakdown, list) or not breakdown:
        return []
    records: list[SliceRecord] = []
    for item in breakdown:
        if not isinstance(item, dict):
            continue
        records.append(
            SliceRecord(
                source=str(source),
                report_status=str(payload.get("status") or "pass").lower(),
                dataset=str(item.get("dataset") or payload.get("dataset") or ""),
                split=str(item.get("split") or payload.get("split") or ""),
                model=str(payload.get("model") or ""),
                quantization=str(payload.get("quantization") or ""),
                record_count=as_int(item.get("record_count")),
                holyc_accuracy=as_float(item.get("holyc_accuracy")),
                llama_accuracy=as_float(item.get("llama_accuracy")),
                agreement=as_float(item.get("agreement")),
                regressions=len(payload.get("regressions") or []),
            )
        )
    return records


def top_level_record(payload: dict[str, Any], source: Path) -> SliceRecord:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return SliceRecord(
        source=str(source),
        report_status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        record_count=as_int(summary.get("record_count")),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        agreement=as_float(summary.get("agreement")),
        regressions=len(payload.get("regressions") or []),
    )


def load_records(path: Path) -> tuple[list[SliceRecord], list[Finding]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], [Finding("read_error", str(path), "missing", "readable", str(exc))]
    except json.JSONDecodeError as exc:
        return [], [Finding("invalid_json", str(path), "invalid", "valid", f"invalid json: {exc}")]
    if not isinstance(payload, dict):
        return [], [Finding("invalid_report", str(path), "non-object", "object", "report root must be an object")]
    records = summary_records(payload, path) or row_records(payload, path) or [top_level_record(payload, path)]
    return records, []


def evaluate(records: list[SliceRecord], load_findings: list[Finding], args: argparse.Namespace) -> list[Finding]:
    findings = list(load_findings)
    if len({record.source for record in records}) < args.min_reports:
        findings.append(
            Finding(
                "min_reports",
                "",
                len({record.source for record in records}),
                args.min_reports,
                f"found {len({record.source for record in records})} report(s), below minimum {args.min_reports}",
            )
        )
    slice_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.dataset, record.split)
        slice_counts[key] = slice_counts.get(key, 0) + record.record_count
    if len(slice_counts) < args.min_slices:
        findings.append(
            Finding(
                "min_slices",
                "",
                len(slice_counts),
                args.min_slices,
                f"found {len(slice_counts)} dataset/split slice(s), below minimum {args.min_slices}",
            )
        )
    for dataset, split in parse_required_slices(args.require_slice):
        count = slice_counts.get((dataset, split), 0)
        if count == 0:
            findings.append(
                Finding(
                    "required_slice",
                    f"{dataset}:{split}",
                    0,
                    "present",
                    f"required eval slice {dataset}:{split} is absent",
                )
            )
    for (dataset, split), count in sorted(slice_counts.items()):
        if count < args.min_records_per_slice:
            findings.append(
                Finding(
                    "min_records_per_slice",
                    f"{dataset}:{split}",
                    count,
                    args.min_records_per_slice,
                    f"{dataset}:{split} has {count} record(s), below minimum {args.min_records_per_slice}",
                )
            )
    for record in records:
        if args.fail_on_failed_reports and record.report_status != "pass":
            findings.append(
                Finding(
                    "report_status",
                    record.source,
                    record.report_status,
                    "pass",
                    f"{record.source} status is {record.report_status}",
                )
            )
        if args.fail_on_regressions and record.regressions:
            findings.append(
                Finding(
                    "regressions",
                    record.source,
                    record.regressions,
                    0,
                    f"{record.source} has {record.regressions} regression(s)",
                )
            )
        if args.min_slice_holyc_accuracy is not None and (
            record.holyc_accuracy is None or record.holyc_accuracy < args.min_slice_holyc_accuracy
        ):
            findings.append(
                Finding(
                    "min_slice_holyc_accuracy",
                    f"{record.dataset}:{record.split}",
                    record.holyc_accuracy,
                    args.min_slice_holyc_accuracy,
                    f"{record.dataset}:{record.split} HolyC accuracy is below threshold",
                )
            )
        if args.min_slice_agreement is not None and (
            record.agreement is None or record.agreement < args.min_slice_agreement
        ):
            findings.append(
                Finding(
                    "min_slice_agreement",
                    f"{record.dataset}:{record.split}",
                    record.agreement,
                    args.min_slice_agreement,
                    f"{record.dataset}:{record.split} HolyC-vs-llama agreement is below threshold",
                )
            )
    return findings


def build_report(records: list[SliceRecord], findings: list[Finding]) -> dict[str, Any]:
    slice_counts: dict[str, int] = {}
    for record in records:
        key = f"{record.dataset}:{record.split}"
        slice_counts[key] = slice_counts.get(key, 0) + record.record_count
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "reports": len({record.source for record in records}),
            "slices": len(slice_counts),
            "records": sum(record.record_count for record in records),
            "findings": len(findings),
            "slice_counts": dict(sorted(slice_counts.items())),
        },
        "slices": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[SliceRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SliceRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Slice Coverage Audit",
        "",
        f"- Status: {report['status']}",
        f"- Reports: {summary['reports']}",
        f"- Slices: {summary['slices']}",
        f"- Records: {summary['records']}",
        f"- Findings: {summary['findings']}",
        "",
        "## Slices",
        "",
        "| Source | Status | Dataset | Split | Model | Quant | Records | HolyC acc | Llama acc | Agreement |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in report["slices"]:
        lines.append(
            "| {source} | {report_status} | {dataset} | {split} | {model} | {quantization} | {record_count} | "
            "{holyc_accuracy} | {llama_accuracy} | {agreement} |".format(**record)
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        for finding in report["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    else:
        lines.append("No eval slice coverage findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    root = ET.Element(
        "testsuite",
        {"name": "holyc_eval_slice_coverage_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    testcase = ET.SubElement(root, "testcase", {"name": "eval_slice_coverage_gates"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"type": "eval_slice_coverage_failure"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="eval_compare JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_slice_coverage_audit_latest")
    parser.add_argument("--min-reports", type=int, default=1)
    parser.add_argument("--min-slices", type=int, default=1)
    parser.add_argument("--min-records-per-slice", type=int, default=1)
    parser.add_argument("--require-slice", action="append", default=[], help="Required DATASET:SPLIT slice")
    parser.add_argument("--min-slice-holyc-accuracy", type=float)
    parser.add_argument("--min-slice-agreement", type=float)
    parser.add_argument("--fail-on-failed-reports", action="store_true")
    parser.add_argument("--fail-on-regressions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    records: list[SliceRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(args.inputs):
        loaded_records, load_findings = load_records(path)
        records.extend(loaded_records)
        findings.extend(load_findings)
    findings = evaluate(records, findings, args)
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
