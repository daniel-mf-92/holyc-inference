#!/usr/bin/env python3
"""Audit raw multiple-choice labels in local eval JSONL sources.

This host-side tool catches label-shape mistakes before dataset_pack.py
normalizes rows into HolyC-loadable HCEval binaries. It never fetches remote
datasets and only reads local JSONL files.
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

import dataset_pack


@dataclass(frozen=True)
class LabelFinding:
    row: int
    record_id: str
    severity: str
    kind: str
    detail: str


@dataclass(frozen=True)
class LabelRecord:
    row: int
    record_id: str
    dataset: str
    split: str
    shape: str
    choice_count: int
    raw_labels: str
    answer_label: str
    correct_label_count: int | None
    status: str
    findings: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def clean(value: Any) -> str:
    return dataset_pack.clean_text(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(payload)
    return rows


def record_id(row: dict[str, Any], index: int) -> str:
    dataset = clean(row.get("dataset") or row.get("source_dataset") or "eval")
    return clean(row.get("id") or row.get("ind") or row.get("question_id") or f"{dataset}-{index + 1}")


def row_shape(row: dict[str, Any]) -> str:
    if isinstance(row.get("mc1_targets"), dict) or isinstance(row.get("mc2_targets"), dict):
        return "truthfulqa"
    if isinstance(row.get("endings"), list):
        return "hellaswag"
    if isinstance(row.get("choices"), list) and any(isinstance(item, dict) for item in row["choices"]):
        return "arc"
    if isinstance(row.get("choices"), (list, dict)):
        return "normalized"
    return "unknown"


def raw_choice_labels(row: dict[str, Any]) -> list[str]:
    raw_choices = row.get("choices") if "choices" in row else row.get("endings")
    if isinstance(raw_choices, list):
        labels: list[str] = []
        for index, item in enumerate(raw_choices):
            if isinstance(item, dict):
                labels.append(clean(item.get("label") or chr(ord("A") + index)))
            else:
                labels.append(chr(ord("A") + index))
        return labels
    if isinstance(raw_choices, dict):
        raw_labels = raw_choices.get("label") or raw_choices.get("labels")
        raw_text = raw_choices.get("text") or raw_choices.get("choices")
        if isinstance(raw_text, list):
            if isinstance(raw_labels, list):
                return [
                    clean(raw_labels[index]) if index < len(raw_labels) and clean(raw_labels[index]) else chr(ord("A") + index)
                    for index in range(len(raw_text))
                ]
            return [chr(ord("A") + index) for index in range(len(raw_text))]
    targets = row.get("mc1_targets") or row.get("mc2_targets")
    if isinstance(targets, dict) and isinstance(targets.get("choices"), list):
        return [str(index) for index, _ in enumerate(targets["choices"])]
    return []


def answer_label(row: dict[str, Any]) -> str:
    for key in ("answerKey", "answer_index", "answer", "label"):
        if key in row:
            return clean(row.get(key))
    return ""


def expected_labels(count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(count)]


def append_finding(
    findings: list[LabelFinding],
    row: int,
    record: str,
    severity: str,
    kind: str,
    detail: str,
) -> None:
    findings.append(LabelFinding(row=row, record_id=record, severity=severity, kind=kind, detail=detail))


def audit_row(
    row: dict[str, Any],
    index: int,
    *,
    require_contiguous_arc_labels: bool,
) -> tuple[LabelRecord, list[LabelFinding]]:
    row_number = index + 1
    rid = record_id(row, index)
    dataset = clean(row.get("dataset") or row.get("source_dataset"))
    split = clean(row.get("split"))
    shape = row_shape(row)
    labels = raw_choice_labels(row)
    answer = answer_label(row)
    findings: list[LabelFinding] = []
    correct_label_count: int | None = None

    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        append_finding(findings, row_number, rid, "error", "duplicate_choice_labels", ",".join(duplicate_labels))

    if shape == "arc":
        if answer and answer not in labels and not answer.isdigit():
            append_finding(findings, row_number, rid, "error", "answer_label_missing", f"{answer!r} not in choice labels")
        if require_contiguous_arc_labels and labels != expected_labels(len(labels)):
            append_finding(
                findings,
                row_number,
                rid,
                "error",
                "non_contiguous_arc_labels",
                f"labels={labels} expected={expected_labels(len(labels))}",
            )

    if shape == "hellaswag":
        if answer == "":
            append_finding(findings, row_number, rid, "error", "missing_hellaswag_label", "missing label")
        elif not answer.isdigit():
            append_finding(findings, row_number, rid, "error", "non_integer_hellaswag_label", answer)
        elif int(answer) >= len(labels):
            append_finding(findings, row_number, rid, "error", "hellaswag_label_out_of_range", answer)

    if shape == "truthfulqa":
        targets = row.get("mc1_targets") or row.get("mc2_targets")
        raw_correct = targets.get("labels") if isinstance(targets, dict) else None
        if not isinstance(raw_correct, list):
            append_finding(findings, row_number, rid, "error", "missing_truthfulqa_labels", "missing target labels")
        else:
            invalid = [value for value in raw_correct if value not in (0, 1, "0", "1", False, True)]
            correct_label_count = sum(1 for value in raw_correct if int(value) == 1) if not invalid else None
            if invalid:
                append_finding(findings, row_number, rid, "error", "non_binary_truthfulqa_labels", repr(invalid))
            elif correct_label_count != 1:
                append_finding(
                    findings,
                    row_number,
                    rid,
                    "error",
                    "truthfulqa_correct_label_count",
                    f"correct_labels={correct_label_count}",
                )

    if not labels:
        append_finding(findings, row_number, rid, "error", "missing_choice_labels", "no choices found")

    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    record = LabelRecord(
        row=row_number,
        record_id=rid,
        dataset=dataset,
        split=split,
        shape=shape,
        choice_count=len(labels),
        raw_labels=",".join(labels),
        answer_label=answer,
        correct_label_count=correct_label_count,
        status=status,
        findings=";".join(finding.kind for finding in findings),
    )
    return record, findings


def audit_dataset(path: Path, *, require_contiguous_arc_labels: bool = False) -> dict[str, Any]:
    rows = read_jsonl(path)
    records: list[LabelRecord] = []
    findings: list[LabelFinding] = []
    for index, row in enumerate(rows):
        record, row_findings = audit_row(row, index, require_contiguous_arc_labels=require_contiguous_arc_labels)
        records.append(record)
        findings.extend(row_findings)

    shape_counts: dict[str, int] = {}
    for record in records:
        shape_counts[record.shape] = shape_counts.get(record.shape, 0) + 1

    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "input": str(path),
        "rows": len(records),
        "shape_counts": dict(sorted(shape_counts.items())),
        "error_count": error_count,
        "findings": [asdict(finding) for finding in findings],
        "records": [asdict(record) for record in records],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Label Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Rows: {report['rows']}",
        f"Findings: {len(report['findings'])}",
        "",
    ]
    if report["findings"]:
        lines.extend(["| Row | Record ID | Kind | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {row} | {record_id} | {kind} | {detail} |".format(
                    row=finding["row"],
                    record_id=finding["record_id"],
                    kind=finding["kind"],
                    detail=str(finding["detail"]).replace("|", "\\|"),
                )
            )
    else:
        lines.append("All raw multiple-choice labels passed enabled checks.")
    return "\n".join(lines) + "\n"


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row", "record_id", "severity", "kind", "detail"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_record_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row",
        "record_id",
        "dataset",
        "split",
        "shape",
        "choice_count",
        "raw_labels",
        "answer_label",
        "correct_label_count",
        "status",
        "findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in report["records"]:
            writer.writerow(record)


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_label_audit",
            "tests": "1",
            "failures": "1" if report["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "dataset_label_integrity"})
    if report["status"] == "fail":
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "dataset_label_findings",
                "message": f"{report['error_count']} label audit finding(s)",
            },
        )
        failure.text = "\n".join(
            f"row {finding['row']} {finding['record_id']}: {finding['kind']} {finding['detail']}"
            for finding in report["findings"]
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local eval JSONL source")
    parser.add_argument("--output", type=Path, default=Path("bench/results/datasets/dataset_label_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("bench/results/datasets/dataset_label_audit_latest.md"))
    parser.add_argument("--csv", type=Path, default=Path("bench/results/datasets/dataset_label_audit_latest.csv"))
    parser.add_argument("--record-csv", type=Path, default=None)
    parser.add_argument("--junit", type=Path, default=Path("bench/results/datasets/dataset_label_audit_latest_junit.xml"))
    parser.add_argument("--require-contiguous-arc-labels", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_dataset(args.input, require_contiguous_arc_labels=args.require_contiguous_arc_labels)
    except (OSError, ValueError) as exc:
        print(f"dataset_label_audit: {exc}", file=sys.stderr)
        return 2

    write_json(args.output, report)
    write_markdown(args.markdown, report)
    write_csv(args.csv, report)
    if args.record_csv:
        write_record_csv(args.record_csv, report)
    write_junit(args.junit, report)
    return 1 if args.fail_on_findings and report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
