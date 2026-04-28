#!/usr/bin/env python3
"""Offline eval input audit for HolyC vs llama.cpp comparisons.

The audit validates local gold and prediction files before an apples-to-apples
comparison run. It checks record-id coverage, duplicate rows, prediction ranges,
optional dataset/split/model/quantization metadata, and writes JSON plus Markdown
reports, CSV issue rows, and JUnit XML under bench/results. It is host-side only
and never launches QEMU.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


METADATA_KEYS = ("model", "quantization", "dataset", "split")


@dataclass(frozen=True)
class Issue:
    severity: str
    source: str
    message: str


@dataclass(frozen=True)
class PredictionAudit:
    source: str
    rows: int
    valid_predictions: int
    scored_predictions: int
    score_coverage_pct: float | None
    score_length_histogram: dict[str, int]
    prediction_histogram: dict[str, int]
    majority_prediction: str
    majority_prediction_count: int
    majority_prediction_pct: float | None
    duplicate_ids: list[str]
    missing_ids: list[str]
    extra_ids: list[str]
    metadata: dict[str, list[str]]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sorted_counts(values: Iterable[int]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {str(key): counter[key] for key in sorted(counter)}


def sorted_label_counts(values: Iterable[str]) -> dict[str, int]:
    counter = collections.Counter(values)
    return {key: counter[key] for key in sorted(counter)}


def majority_label(histogram: dict[str, int]) -> tuple[str, int, float | None]:
    if not histogram:
        return "", 0, None
    label, count = max(histogram.items(), key=lambda item: (item[1], item[0]))
    total = sum(histogram.values())
    pct = (count / total * 100.0) if total else None
    return label, count, pct


def append_issue(issues: list[Issue], severity: str, source: str, message: str) -> None:
    issues.append(Issue(severity=severity, source=source, message=message))


def metadata_value(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metadata"), dict):
        value = row["metadata"].get(key)
    if value is None or str(value).strip() == "":
        return None
    return str(value).strip()


def collect_metadata(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {key: set() for key in METADATA_KEYS}
    for row in rows:
        for key in METADATA_KEYS:
            value = metadata_value(row, key)
            if value is not None:
                values[key].add(value)
    return {key: sorted(found) for key, found in values.items() if found}


def read_rows_with_issues(path: Path, source_name: str, issues: list[Issue]) -> list[dict[str, Any]]:
    try:
        return eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_issue(issues, "error", source_name, f"cannot read prediction rows: {exc}")
        return []


def audit_predictions(
    path: Path,
    source_name: str,
    gold: dict[str, eval_compare.GoldCase],
    expected_model: str,
    expected_quantization: str,
    expected_dataset: str,
    expected_split: str,
    max_majority_prediction_pct: float | None,
    min_score_coverage_pct: float | None,
    issues: list[Issue],
) -> PredictionAudit:
    rows = read_rows_with_issues(path, source_name, issues)
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    extra_ids: set[str] = set()
    valid_predictions = 0
    scored_predictions = 0
    score_lengths: list[int] = []
    prediction_labels: list[str] = []
    metadata = collect_metadata(rows)

    for index, row in enumerate(rows):
        row_label = f"{path}:{index + 1}"
        try:
            record_id = eval_compare.case_id(row, row_label)
        except ValueError as exc:
            append_issue(issues, "error", source_name, str(exc))
            continue

        if record_id in seen:
            duplicate_ids.add(record_id)
            append_issue(issues, "error", source_name, f"duplicate prediction id {record_id!r}")
            continue
        seen.add(record_id)

        if record_id not in gold:
            extra_ids.add(record_id)
            append_issue(issues, "error", source_name, f"prediction id {record_id!r} is not in gold")
            continue

        try:
            prediction = eval_compare.normalize_prediction(row, gold[record_id], path, index)
        except ValueError as exc:
            append_issue(issues, "error", source_name, str(exc))
            continue
        valid_predictions += 1
        if prediction.scores is not None:
            scored_predictions += 1
            score_lengths.append(len(prediction.scores))
        prediction_labels.append(str(prediction.predicted_index))

        row_dataset = metadata_value(row, "dataset")
        row_split = metadata_value(row, "split")
        if row_dataset is not None and row_dataset != expected_dataset:
            append_issue(
                issues,
                "error",
                source_name,
                f"{row_label}: dataset metadata {row_dataset!r} does not match expected {expected_dataset!r}",
            )
        if row_split is not None and row_split != expected_split:
            append_issue(
                issues,
                "error",
                source_name,
                f"{row_label}: split metadata {row_split!r} does not match expected {expected_split!r}",
            )

    missing_ids = sorted(set(gold) - seen)
    for record_id in missing_ids:
        append_issue(issues, "error", source_name, f"missing prediction id {record_id!r}")

    check_metadata_values(metadata, source_name, "model", expected_model, issues)
    check_metadata_values(metadata, source_name, "quantization", expected_quantization, issues)

    prediction_histogram = sorted_label_counts(prediction_labels)
    majority_prediction, majority_prediction_count, majority_prediction_pct = majority_label(prediction_histogram)
    if (
        max_majority_prediction_pct is not None
        and majority_prediction_pct is not None
        and majority_prediction_pct > max_majority_prediction_pct
    ):
        append_issue(
            issues,
            "error",
            source_name,
            (
                f"majority prediction index {majority_prediction!r} covers "
                f"{majority_prediction_pct:.2f}% of valid predictions, above "
                f"{max_majority_prediction_pct:.2f}% gate"
            ),
        )
    score_coverage_pct = (scored_predictions / valid_predictions * 100.0) if valid_predictions else None
    if (
        min_score_coverage_pct is not None
        and score_coverage_pct is not None
        and score_coverage_pct < min_score_coverage_pct
    ):
        append_issue(
            issues,
            "error",
            source_name,
            (
                f"score vector coverage is {score_coverage_pct:.2f}% of valid predictions, below "
                f"{min_score_coverage_pct:.2f}% gate"
            ),
        )

    return PredictionAudit(
        source=source_name,
        rows=len(rows),
        valid_predictions=valid_predictions,
        scored_predictions=scored_predictions,
        score_coverage_pct=score_coverage_pct,
        score_length_histogram=sorted_counts(score_lengths),
        prediction_histogram=prediction_histogram,
        majority_prediction=majority_prediction,
        majority_prediction_count=majority_prediction_count,
        majority_prediction_pct=majority_prediction_pct,
        duplicate_ids=sorted(duplicate_ids),
        missing_ids=missing_ids,
        extra_ids=sorted(extra_ids),
        metadata=metadata,
    )


def check_metadata_values(
    metadata: dict[str, list[str]],
    source_name: str,
    key: str,
    expected: str,
    issues: list[Issue],
) -> None:
    values = metadata.get(key, [])
    if len(values) > 1:
        append_issue(issues, "error", source_name, f"multiple {key} metadata values found: {', '.join(values)}")
    if expected and values and values != [expected]:
        append_issue(
            issues,
            "error",
            source_name,
            f"{key} metadata {values} does not match expected {expected!r}",
        )


def cross_check_metadata(
    holyc: PredictionAudit,
    llama: PredictionAudit,
    key: str,
    issues: list[Issue],
) -> None:
    holyc_values = holyc.metadata.get(key, [])
    llama_values = llama.metadata.get(key, [])
    if holyc_values and llama_values and holyc_values != llama_values:
        append_issue(
            issues,
            "error",
            "metadata",
            f"HolyC {key} metadata {holyc_values} differs from llama.cpp {key} metadata {llama_values}",
        )


def audit_gold(path: Path, dataset: str, split: str, issues: list[Issue]) -> dict[str, eval_compare.GoldCase]:
    try:
        gold = eval_compare.load_gold(path, dataset, split)
    except (OSError, ValueError) as exc:
        append_issue(issues, "error", "gold", f"cannot load gold dataset: {exc}")
        return {}

    if not gold:
        append_issue(issues, "error", "gold", "gold dataset contains no records")

    splits = sorted({case.split for case in gold.values()})
    if len(splits) > 1:
        append_issue(issues, "warning", "gold", f"gold rows contain multiple split values: {', '.join(splits)}")
    return gold


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[Issue] = []
    gold = audit_gold(args.gold, args.dataset, args.split, issues)
    answer_histogram = sorted_counts(case.answer_index for case in gold.values())
    choice_count_histogram = sorted_counts(len(case.choices) for case in gold.values())

    holyc = audit_predictions(
        args.holyc,
        "holyc",
        gold,
        args.model,
        args.quantization,
        args.dataset,
        args.split,
        args.max_majority_prediction_pct,
        args.min_score_coverage_pct,
        issues,
    )
    llama = audit_predictions(
        args.llama,
        "llama.cpp",
        gold,
        args.model,
        args.quantization,
        args.dataset,
        args.split,
        args.max_majority_prediction_pct,
        args.min_score_coverage_pct,
        issues,
    )
    for key in ("model", "quantization"):
        cross_check_metadata(holyc, llama, key, issues)

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    status = "fail" if error_count else "pass"

    return {
        "answer_histogram": answer_histogram,
        "choice_count_histogram": choice_count_histogram,
        "dataset": args.dataset,
        "files": {
            "gold": {"path": str(args.gold), "sha256": file_sha256(args.gold) if args.gold.exists() else ""},
            "holyc": {"path": str(args.holyc), "sha256": file_sha256(args.holyc) if args.holyc.exists() else ""},
            "llama": {"path": str(args.llama), "sha256": file_sha256(args.llama) if args.llama.exists() else ""},
        },
        "generated_at": iso_now(),
        "gold_record_count": len(gold),
        "issues": [asdict(issue) for issue in issues],
        "model": args.model,
        "prediction_audits": {
            "holyc": asdict(holyc),
            "llama": asdict(llama),
        },
        "quantization": args.quantization,
        "split": args.split,
        "status": status,
        "summary": {
            "errors": error_count,
            "warnings": warning_count,
            "holyc_coverage": holyc.valid_predictions,
            "llama_coverage": llama.valid_predictions,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Eval Input Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Dataset: {report['dataset']}",
        f"Split: {report['split']}",
        f"Model: {report['model'] or '-'}",
        f"Quantization: {report['quantization'] or '-'}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {report['gold_record_count']} |",
        f"| HolyC valid predictions | {summary['holyc_coverage']} |",
        f"| llama.cpp valid predictions | {summary['llama_coverage']} |",
        f"| Errors | {summary['errors']} |",
        f"| Warnings | {summary['warnings']} |",
        "",
        "## Prediction Distribution",
        "",
        "| Engine | Histogram | Majority | Majority % |",
        "| --- | --- | --- | ---: |",
    ]
    for name, audit in report["prediction_audits"].items():
        majority_pct = audit["majority_prediction_pct"]
        majority_pct_text = "-" if majority_pct is None else f"{majority_pct:.2f}"
        lines.append(
            f"| {name} | {json.dumps(audit['prediction_histogram'], sort_keys=True)} | "
            f"{audit['majority_prediction'] or '-'} | {majority_pct_text} |"
        )
    lines.extend(
        [
            "",
            "## Score Coverage",
            "",
            "| Engine | Scored predictions | Coverage % | Score lengths |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, audit in report["prediction_audits"].items():
        coverage_pct = audit["score_coverage_pct"]
        coverage_pct_text = "-" if coverage_pct is None else f"{coverage_pct:.2f}"
        lines.append(
            f"| {name} | {audit['scored_predictions']}/{audit['valid_predictions']} | "
            f"{coverage_pct_text} | {json.dumps(audit['score_length_histogram'], sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Issues",
            "",
        ]
    )
    if report["issues"]:
        lines.append("| Severity | Source | Message |")
        lines.append("| --- | --- | --- |")
        for issue in report["issues"]:
            lines.append(f"| {issue['severity']} | {issue['source']} | {issue['message']} |")
    else:
        lines.append("No input issues found.")
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["severity", "source", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for issue in report["issues"]:
            writer.writerow({field: issue[field] for field in fields})


def write_junit(report: dict[str, Any], path: Path) -> None:
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    warnings = [issue for issue in report["issues"] if issue["severity"] == "warning"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_input_audit",
            "tests": "1",
            "failures": "1" if errors else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "eval_input_audit", "name": "input_gate"})
    if errors:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "eval_input_audit_error",
                "message": f"{len(errors)} eval input error(s), {len(warnings)} warning(s)",
            },
        )
        failure.text = "\n".join(
            f"{issue['source']}: {issue['message']}" for issue in errors
        )
    elif warnings:
        system_out = ET.SubElement(case, "system-out")
        system_out.text = "\n".join(f"{issue['source']}: {issue['message']}" for issue in warnings)
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"
    csv_path = output_dir / f"{output_stem}.csv"
    junit_path = output_dir / f"{output_stem}_junit.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(report, csv_path)
    write_junit(report, junit_path)
    return json_path, md_path, csv_path, junit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Local gold JSONL dataset")
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC prediction JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp prediction JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--model", default="")
    parser.add_argument("--quantization", default="")
    parser.add_argument(
        "--max-majority-prediction-pct",
        type=float,
        help="Fail when either engine predicts one answer index for more than this percentage of valid rows",
    )
    parser.add_argument(
        "--min-score-coverage-pct",
        type=float,
        help="Fail when either engine has score vectors for less than this percentage of valid rows",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_input_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    json_path, md_path, csv_path, junit_path = write_report(report, args.output_dir, args.output_stem)
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_junit={junit_path}")
    print(f"status={report['status']}")
    print(f"errors={report['summary']['errors']}")
    print(f"warnings={report['summary']['warnings']}")
    return 2 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
