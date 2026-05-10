#!/usr/bin/env python3
"""Audit multiple-choice label mapping before HolyC/llama eval comparison.

This host-side tool reads local gold and prediction artifacts only. It reports
which raw answer formats each engine emits (numeric index, alpha label, choice
text, or scores-only), checks those raw answers against the gold choice map, and
can gate mixed formats or HolyC/llama format drift before eval_compare.py
computes quality metrics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
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


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


@dataclass(frozen=True)
class ChoiceMapRow:
    source: str
    record_id: str
    dataset: str
    split: str
    raw_format: str
    raw_prediction: str
    normalized_index: int | None
    answer_index: int
    choice_count: int
    valid: bool
    correct: bool | None
    has_scores: bool


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def first_present(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def parse_scores(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("scores must be a non-empty list")
    scores = [float(item) for item in value]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("scores must contain only finite numbers")
    return scores


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], 1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def classify_prediction(value: Any, scores: list[float] | None, choices: list[str]) -> tuple[str, int | None, str]:
    if value is None or value == "":
        if scores is not None:
            return "scores_only", argmax(scores), ""
        return "missing", None, ""
    if isinstance(value, int):
        return "index", value, str(value)
    if isinstance(value, float) and value.is_integer():
        return "index", int(value), str(int(value))

    text = str(value).strip()
    if text.isdigit():
        return "index", int(text), text
    if len(text) == 1 and "A" <= text.upper() <= "Z":
        return "alpha", ord(text.upper()) - ord("A"), text

    normalized_choices = [choice.strip().lower() for choice in choices]
    if text.lower() in normalized_choices:
        return "choice_text", normalized_choices.index(text.lower()), text
    return "unmapped", None, text


def add_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


def load_prediction_rows(path: Path, source: str, findings: list[Finding]) -> list[dict[str, Any]]:
    try:
        return eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, source, "", "load_error", f"cannot read predictions: {exc}")
        return []


def audit_gold(gold: dict[str, eval_compare.GoldCase], findings: list[Finding]) -> dict[str, Any]:
    choice_count_histogram: collections.Counter[str] = collections.Counter()
    duplicate_choice_rows: list[str] = []
    answer_position_histogram: collections.Counter[str] = collections.Counter()

    for record_id, case in gold.items():
        choice_count_histogram[str(len(case.choices))] += 1
        answer_position_histogram[str(case.answer_index)] += 1
        normalized = [choice.strip().lower() for choice in case.choices]
        if len(set(normalized)) != len(normalized):
            duplicate_choice_rows.append(record_id)
            add_finding(findings, "gold", record_id, "duplicate_choice_text", "choice text is not unique after normalization")
        if case.answer_index < 0 or case.answer_index >= len(case.choices):
            add_finding(findings, "gold", record_id, "answer_out_of_range", "gold answer index is outside choice range")

    return {
        "records": len(gold),
        "choice_count_histogram": dict(sorted(choice_count_histogram.items())),
        "answer_position_histogram": dict(sorted(answer_position_histogram.items())),
        "duplicate_choice_rows": duplicate_choice_rows,
    }


def audit_engine(
    path: Path,
    source: str,
    gold: dict[str, eval_compare.GoldCase],
    findings: list[Finding],
) -> list[ChoiceMapRow]:
    rows: list[ChoiceMapRow] = []
    seen: set[str] = set()
    for index, row in enumerate(load_prediction_rows(path, source, findings), 1):
        row_label = f"{path}:{index}"
        try:
            record_id = eval_compare.case_id(row, row_label)
        except ValueError as exc:
            add_finding(findings, source, "", "missing_id", str(exc))
            continue
        if record_id in seen:
            add_finding(findings, source, record_id, "duplicate_prediction", "duplicate prediction row")
            continue
        seen.add(record_id)
        case = gold.get(record_id)
        if case is None:
            add_finding(findings, source, record_id, "extra_prediction", "prediction id is not present in gold")
            continue

        try:
            scores = parse_scores(first_present(row, eval_compare.SCORE_KEYS))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            add_finding(findings, source, record_id, "invalid_scores", str(exc))
            scores = None
        if scores is not None and len(scores) != len(case.choices):
            add_finding(
                findings,
                source,
                record_id,
                "score_count_mismatch",
                f"score count {len(scores)} does not match choice count {len(case.choices)}",
            )

        raw_prediction = first_present(row, eval_compare.PREDICTION_KEYS)
        raw_format, normalized_index, raw_text = classify_prediction(raw_prediction, scores, case.choices)
        valid = normalized_index is not None and 0 <= normalized_index < len(case.choices)
        if raw_format in {"missing", "unmapped"}:
            add_finding(findings, source, record_id, raw_format, "raw prediction cannot be mapped to a gold choice")
        elif not valid:
            add_finding(
                findings,
                source,
                record_id,
                "prediction_out_of_range",
                f"normalized prediction {normalized_index} is outside choice range 0..{len(case.choices) - 1}",
            )

        rows.append(
            ChoiceMapRow(
                source=source,
                record_id=record_id,
                dataset=case.dataset,
                split=case.split,
                raw_format=raw_format,
                raw_prediction=raw_text,
                normalized_index=normalized_index if valid else None,
                answer_index=case.answer_index,
                choice_count=len(case.choices),
                valid=valid,
                correct=(normalized_index == case.answer_index) if valid else None,
                has_scores=scores is not None,
            )
        )

    for record_id in sorted(set(gold) - seen):
        add_finding(findings, source, record_id, "missing_prediction", "gold id is missing from predictions")
    return rows


def summarize_engine(rows: list[ChoiceMapRow]) -> dict[str, Any]:
    formats = collections.Counter(row.raw_format for row in rows)
    datasets = collections.Counter(f"{row.dataset}/{row.split}" for row in rows)
    valid_rows = sum(1 for row in rows if row.valid)
    scored_rows = sum(1 for row in rows if row.has_scores)
    correct_rows = sum(1 for row in rows if row.correct is True)
    return {
        "rows": len(rows),
        "valid_rows": valid_rows,
        "valid_pct": (valid_rows / len(rows) * 100.0) if rows else 0.0,
        "scored_rows": scored_rows,
        "scored_pct": (scored_rows / len(rows) * 100.0) if rows else 0.0,
        "correct_rows": correct_rows,
        "accuracy": (correct_rows / valid_rows) if valid_rows else None,
        "format_histogram": dict(sorted(formats.items())),
        "dataset_split_histogram": dict(sorted(datasets.items())),
    }


def evaluate_gates(
    findings: list[Finding],
    summaries: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.min_valid_pct is not None:
        for source, summary in summaries.items():
            if summary["valid_pct"] < args.min_valid_pct:
                add_finding(
                    findings,
                    source,
                    "",
                    "min_valid_pct",
                    f"valid prediction pct {summary['valid_pct']:.3f} is below {args.min_valid_pct:.3f}",
                )
    if args.fail_mixed_formats:
        for source, summary in summaries.items():
            meaningful = {
                key
                for key, value in summary["format_histogram"].items()
                if value and key not in {"missing", "unmapped"}
            }
            if len(meaningful) > 1:
                add_finding(findings, source, "", "mixed_formats", f"prediction formats differ: {sorted(meaningful)}")
    if args.require_engine_format_parity:
        holyc_formats = set(summaries["holyc"]["format_histogram"])
        llama_formats = set(summaries["llama.cpp"]["format_histogram"])
        if holyc_formats != llama_formats:
            add_finding(
                findings,
                "paired",
                "",
                "engine_format_parity",
                f"HolyC formats {sorted(holyc_formats)} differ from llama.cpp formats {sorted(llama_formats)}",
            )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        gold = {}
        add_finding(findings, "gold", "", "load_error", f"cannot read gold: {exc}")

    gold_summary = audit_gold(gold, findings)
    holyc_rows = audit_engine(args.holyc, "holyc", gold, findings)
    llama_rows = audit_engine(args.llama, "llama.cpp", gold, findings)
    summaries = {"holyc": summarize_engine(holyc_rows), "llama.cpp": summarize_engine(llama_rows)}
    evaluate_gates(findings, summaries, args)

    return {
        "schema_version": 1,
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {
            "gold": str(args.gold),
            "holyc": str(args.holyc),
            "llama": str(args.llama),
            "dataset": args.dataset,
            "split": args.split,
        },
        "gold": gold_summary,
        "summary": summaries,
        "rows": [asdict(row) for row in holyc_rows + llama_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [field.name for field in ChoiceMapRow.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    fieldnames = [field.name for field in Finding.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow({key: finding.get(key, "") for key in fieldnames})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eval Choice Map Audit",
        "",
        f"- Status: {report['status']}",
        f"- Gold records: {report['gold']['records']}",
        f"- HolyC formats: {report['summary']['holyc']['format_histogram']}",
        f"- llama.cpp formats: {report['summary']['llama.cpp']['format_histogram']}",
        f"- Findings: {len(report['findings'])}",
    ]
    if report["findings"]:
        lines.extend(["", "## Findings"])
        for finding in report["findings"]:
            lines.append(f"- {finding['source']} {finding['record_id']} {finding['kind']}: {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_choice_map_audit",
            "tests": "1",
            "failures": "1" if report["findings"] else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "choice_map"})
    if report["findings"]:
        failure = ET.SubElement(case, "failure", {"message": f"{len(report['findings'])} choice-map finding(s)"})
        failure.text = "\n".join(f"{item['source']} {item['record_id']} {item['kind']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--holyc", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_choice_map_audit_latest")
    parser.add_argument("--min-valid-pct", type=float, default=None)
    parser.add_argument("--fail-mixed-formats", action="store_true")
    parser.add_argument("--require-engine-format-parity", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    (output_base.with_suffix(".json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_base.with_suffix(".csv"), report["rows"])
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", report["findings"])
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", report)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
