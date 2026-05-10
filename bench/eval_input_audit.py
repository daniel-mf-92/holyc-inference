#!/usr/bin/env python3
"""Offline eval input audit for HolyC vs llama.cpp comparisons.

The audit validates local gold and prediction files before an apples-to-apples
comparison run. It checks record-id coverage, duplicate rows, prediction ranges,
gold choice-count bounds, optional dataset/split/model/quantization metadata,
optional prompt/choice/input hash parity against gold rows, and writes JSON plus
Markdown reports, CSV issue rows, and JUnit XML under bench/results. It is
host-side only and never launches QEMU.
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
PROMPT_HASH_KEYS = ("prompt_sha256", "prompt_hash", "gold_prompt_sha256")
CHOICES_HASH_KEYS = ("choices_sha256", "choices_hash", "gold_choices_sha256")
INPUT_HASH_KEYS = ("input_sha256", "gold_input_sha256", "prompt_choices_sha256")


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
    top_score_ties: int
    top_score_tie_pct: float | None
    low_top_score_margins: int
    low_top_score_margin_pct: float | None
    min_top_score_margin: float | None
    score_length_histogram: dict[str, int]
    prediction_histogram: dict[str, int]
    majority_prediction: str
    majority_prediction_count: int
    majority_prediction_pct: float | None
    prompt_hash_matches: int
    prompt_hash_mismatches: int
    prompt_hash_missing: int
    choices_hash_matches: int
    choices_hash_mismatches: int
    choices_hash_missing: int
    input_hash_matches: int
    input_hash_mismatches: int
    input_hash_missing: int
    duplicate_ids: list[str]
    missing_ids: list[str]
    extra_ids: list[str]
    metadata: dict[str, list[str]]
    records: list[dict[str, Any]]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choices_sha256(choices: list[str]) -> str:
    encoded = json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_sha256(gold: eval_compare.GoldCase) -> str:
    encoded = json.dumps(
        {
            "prompt_sha256": text_sha256(gold.prompt),
            "choices_sha256": choices_sha256(gold.choices),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_metadata_hash(row: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, str | None]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            value = metadata.get(key)
        if value is not None and str(value).strip() != "":
            return key, str(value).strip()
    return None, None


def check_hash_value(
    issues: list[Issue],
    source_name: str,
    row_label: str,
    label: str,
    found_key: str | None,
    found_value: str | None,
    expected_value: str,
    require_hash: bool,
) -> tuple[int, int, int]:
    if found_value is None:
        if require_hash:
            append_issue(issues, "error", source_name, f"{row_label}: missing {label} hash metadata")
        return 0, 0, 1
    if found_value.lower() != expected_value.lower():
        key_text = found_key or label
        append_issue(
            issues,
            "error",
            source_name,
            f"{row_label}: {key_text} {found_value!r} does not match gold {label} hash {expected_value!r}",
        )
        return 0, 1, 0
    return 1, 0, 0


def hash_status(found_value: str | None, expected_value: str) -> str:
    if found_value is None:
        return "missing"
    if found_value.lower() != expected_value.lower():
        return "mismatch"
    return "match"


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
    max_top_score_tie_pct: float | None,
    min_top_score_margin: float | None,
    require_input_hashes: bool,
    issues: list[Issue],
) -> PredictionAudit:
    rows = read_rows_with_issues(path, source_name, issues)
    seen: set[str] = set()
    duplicate_ids: set[str] = set()
    extra_ids: set[str] = set()
    valid_predictions = 0
    scored_predictions = 0
    top_score_ties = 0
    low_top_score_margins = 0
    score_lengths: list[int] = []
    top_score_margins: list[float] = []
    prediction_labels: list[str] = []
    metadata = collect_metadata(rows)
    records: list[dict[str, Any]] = []
    prompt_hash_matches = 0
    prompt_hash_mismatches = 0
    prompt_hash_missing = 0
    choices_hash_matches = 0
    choices_hash_mismatches = 0
    choices_hash_missing = 0
    input_hash_matches = 0
    input_hash_mismatches = 0
    input_hash_missing = 0

    for index, row in enumerate(rows):
        row_label = f"{path}:{index + 1}"
        record_telemetry: dict[str, Any] = {
            "source": source_name,
            "row_number": index + 1,
            "record_id": "",
            "valid": False,
            "answer_index": "",
            "predicted_index": "",
            "correct": "",
            "has_scores": False,
            "score_count": 0,
            "top_score_tie_count": "",
            "top_score_margin": "",
            "prompt_hash_status": "not_checked",
            "choices_hash_status": "not_checked",
            "input_hash_status": "not_checked",
            "issue": "",
        }
        try:
            record_id = eval_compare.case_id(row, row_label)
        except ValueError as exc:
            append_issue(issues, "error", source_name, str(exc))
            record_telemetry["issue"] = str(exc)
            records.append(record_telemetry)
            continue
        record_telemetry["record_id"] = record_id

        if record_id in seen:
            duplicate_ids.add(record_id)
            message = f"duplicate prediction id {record_id!r}"
            append_issue(issues, "error", source_name, message)
            record_telemetry["issue"] = message
            records.append(record_telemetry)
            continue
        seen.add(record_id)

        if record_id not in gold:
            extra_ids.add(record_id)
            message = f"prediction id {record_id!r} is not in gold"
            append_issue(issues, "error", source_name, message)
            record_telemetry["issue"] = message
            records.append(record_telemetry)
            continue
        gold_case = gold[record_id]
        record_telemetry["answer_index"] = gold_case.answer_index

        try:
            prediction = eval_compare.normalize_prediction(row, gold_case, path, index)
        except ValueError as exc:
            append_issue(issues, "error", source_name, str(exc))
            record_telemetry["issue"] = str(exc)
            records.append(record_telemetry)
            continue
        valid_predictions += 1
        record_telemetry["valid"] = True
        record_telemetry["predicted_index"] = prediction.predicted_index
        record_telemetry["correct"] = prediction.predicted_index == gold_case.answer_index
        if prediction.scores is not None:
            scored_predictions += 1
            score_lengths.append(len(prediction.scores))
            record_telemetry["has_scores"] = True
            record_telemetry["score_count"] = len(prediction.scores)
            top_score = max(prediction.scores)
            top_score_tie_count = sum(1 for score in prediction.scores if score == top_score)
            record_telemetry["top_score_tie_count"] = top_score_tie_count
            if top_score_tie_count > 1:
                top_score_ties += 1
            sorted_scores = sorted(prediction.scores, reverse=True)
            top_score_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            top_score_margins.append(top_score_margin)
            record_telemetry["top_score_margin"] = top_score_margin
            if min_top_score_margin is not None and top_score_margin < min_top_score_margin:
                low_top_score_margins += 1
                append_issue(
                    issues,
                    "error",
                    source_name,
                    (
                        f"{row_label}: top score margin {top_score_margin:.6g} is below "
                        f"--min-top-score-margin {min_top_score_margin:.6g}"
                    ),
                )
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
        found_key, found_value = first_metadata_hash(row, PROMPT_HASH_KEYS)
        record_telemetry["prompt_hash_status"] = hash_status(found_value, text_sha256(gold_case.prompt))
        matches, mismatches, missing = check_hash_value(
            issues,
            source_name,
            row_label,
            "prompt",
            found_key,
            found_value,
            text_sha256(gold_case.prompt),
            require_input_hashes,
        )
        prompt_hash_matches += matches
        prompt_hash_mismatches += mismatches
        prompt_hash_missing += missing
        found_key, found_value = first_metadata_hash(row, CHOICES_HASH_KEYS)
        record_telemetry["choices_hash_status"] = hash_status(found_value, choices_sha256(gold_case.choices))
        matches, mismatches, missing = check_hash_value(
            issues,
            source_name,
            row_label,
            "choices",
            found_key,
            found_value,
            choices_sha256(gold_case.choices),
            require_input_hashes,
        )
        choices_hash_matches += matches
        choices_hash_mismatches += mismatches
        choices_hash_missing += missing
        found_key, found_value = first_metadata_hash(row, INPUT_HASH_KEYS)
        record_telemetry["input_hash_status"] = hash_status(found_value, input_sha256(gold_case))
        matches, mismatches, missing = check_hash_value(
            issues,
            source_name,
            row_label,
            "input",
            found_key,
            found_value,
            input_sha256(gold_case),
            require_input_hashes,
        )
        input_hash_matches += matches
        input_hash_mismatches += mismatches
        input_hash_missing += missing
        records.append(record_telemetry)

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
    top_score_tie_pct = (top_score_ties / scored_predictions * 100.0) if scored_predictions else None
    low_top_score_margin_pct = (
        (low_top_score_margins / scored_predictions * 100.0) if scored_predictions else None
    )
    observed_min_top_score_margin = min(top_score_margins) if top_score_margins else None
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
    if (
        max_top_score_tie_pct is not None
        and top_score_tie_pct is not None
        and top_score_tie_pct > max_top_score_tie_pct
    ):
        append_issue(
            issues,
            "error",
            source_name,
            (
                f"top score ties cover {top_score_tie_pct:.2f}% of scored predictions, above "
                f"{max_top_score_tie_pct:.2f}% gate"
            ),
        )

    return PredictionAudit(
        source=source_name,
        rows=len(rows),
        valid_predictions=valid_predictions,
        scored_predictions=scored_predictions,
        score_coverage_pct=score_coverage_pct,
        top_score_ties=top_score_ties,
        top_score_tie_pct=top_score_tie_pct,
        low_top_score_margins=low_top_score_margins,
        low_top_score_margin_pct=low_top_score_margin_pct,
        min_top_score_margin=observed_min_top_score_margin,
        score_length_histogram=sorted_counts(score_lengths),
        prediction_histogram=prediction_histogram,
        majority_prediction=majority_prediction,
        majority_prediction_count=majority_prediction_count,
        majority_prediction_pct=majority_prediction_pct,
        prompt_hash_matches=prompt_hash_matches,
        prompt_hash_mismatches=prompt_hash_mismatches,
        prompt_hash_missing=prompt_hash_missing,
        choices_hash_matches=choices_hash_matches,
        choices_hash_mismatches=choices_hash_mismatches,
        choices_hash_missing=choices_hash_missing,
        input_hash_matches=input_hash_matches,
        input_hash_mismatches=input_hash_mismatches,
        input_hash_missing=input_hash_missing,
        duplicate_ids=sorted(duplicate_ids),
        missing_ids=missing_ids,
        extra_ids=sorted(extra_ids),
        metadata=metadata,
        records=records,
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


def audit_gold(
    path: Path,
    dataset: str,
    split: str,
    min_choices: int | None,
    max_choices: int | None,
    issues: list[Issue],
) -> dict[str, eval_compare.GoldCase]:
    try:
        gold = eval_compare.load_gold(path, dataset, split)
    except (OSError, ValueError) as exc:
        append_issue(issues, "error", "gold", f"cannot load gold dataset: {exc}")
        return {}

    if not gold:
        append_issue(issues, "error", "gold", "gold dataset contains no records")

    for record_id, case in sorted(gold.items()):
        choice_count = len(case.choices)
        if min_choices is not None and choice_count < min_choices:
            append_issue(
                issues,
                "error",
                "gold",
                f"gold id {record_id!r} has {choice_count} choices, below --min-choices {min_choices}",
            )
        if max_choices is not None and choice_count > max_choices:
            append_issue(
                issues,
                "error",
                "gold",
                f"gold id {record_id!r} has {choice_count} choices, above --max-choices {max_choices}",
            )

    splits = sorted({case.split for case in gold.values()})
    if len(splits) > 1:
        append_issue(issues, "warning", "gold", f"gold rows contain multiple split values: {', '.join(splits)}")
    return gold


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    issues: list[Issue] = []
    gold = audit_gold(args.gold, args.dataset, args.split, args.min_choices, args.max_choices, issues)
    answer_histogram = sorted_counts(case.answer_index for case in gold.values())
    choice_count_histogram = sorted_counts(len(case.choices) for case in gold.values())
    majority_gold_answer, majority_gold_answer_count, majority_gold_answer_pct = majority_label(answer_histogram)
    if (
        args.max_majority_gold_answer_pct is not None
        and majority_gold_answer_pct is not None
        and majority_gold_answer_pct > args.max_majority_gold_answer_pct
    ):
        append_issue(
            issues,
            "error",
            "gold",
            (
                f"majority gold answer index {majority_gold_answer!r} covers "
                f"{majority_gold_answer_pct:.2f}% of gold rows, above "
                f"{args.max_majority_gold_answer_pct:.2f}% gate"
            ),
        )

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
        args.max_top_score_tie_pct,
        args.min_top_score_margin,
        args.require_input_hashes,
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
        args.max_top_score_tie_pct,
        args.min_top_score_margin,
        args.require_input_hashes,
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
        "gold_distribution": {
            "answer_histogram": answer_histogram,
            "choice_count_histogram": choice_count_histogram,
            "majority_answer": majority_gold_answer,
            "majority_answer_count": majority_gold_answer_count,
            "majority_answer_pct": majority_gold_answer_pct,
        },
        "gold_choice_gates": {
            "min_choices": args.min_choices,
            "max_choices": args.max_choices,
        },
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
        "## Gold Distribution",
        "",
        "| Answer histogram | Choice counts | Majority | Majority % |",
        "| --- | --- | --- | ---: |",
    ]
    gold_distribution = report["gold_distribution"]
    majority_pct = gold_distribution["majority_answer_pct"]
    majority_pct_text = "-" if majority_pct is None else f"{majority_pct:.2f}"
    lines.append(
        f"| {json.dumps(gold_distribution['answer_histogram'], sort_keys=True)} | "
        f"{json.dumps(gold_distribution['choice_count_histogram'], sort_keys=True)} | "
        f"{gold_distribution['majority_answer'] or '-'} | {majority_pct_text} |"
    )
    choice_gates = report["gold_choice_gates"]
    lines.extend(
        [
            "",
            f"Choice gates: min={choice_gates['min_choices'] or '-'}, max={choice_gates['max_choices'] or '-'}",
        ]
    )
    lines.extend(
        [
            "",
            "## Prediction Distribution",
            "",
            "| Engine | Histogram | Majority | Majority % |",
            "| --- | --- | --- | ---: |",
        ]
    )
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
            "| Engine | Scored predictions | Coverage % | Top-score ties | Tie % | Score lengths |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, audit in report["prediction_audits"].items():
        coverage_pct = audit["score_coverage_pct"]
        coverage_pct_text = "-" if coverage_pct is None else f"{coverage_pct:.2f}"
        tie_pct = audit["top_score_tie_pct"]
        tie_pct_text = "-" if tie_pct is None else f"{tie_pct:.2f}"
        lines.append(
            f"| {name} | {audit['scored_predictions']}/{audit['valid_predictions']} | "
            f"{coverage_pct_text} | {audit['top_score_ties']} | {tie_pct_text} | "
            f"{json.dumps(audit['score_length_histogram'], sort_keys=True)} |"
        )
    lines.extend(
        [
            "",
            "## Score Margins",
            "",
            "| Engine | Min margin | Low margins | Low margin % |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, audit in report["prediction_audits"].items():
        min_margin = audit["min_top_score_margin"]
        min_margin_text = "-" if min_margin is None else f"{min_margin:.6g}"
        low_pct = audit["low_top_score_margin_pct"]
        low_pct_text = "-" if low_pct is None else f"{low_pct:.2f}"
        lines.append(
            f"| {name} | {min_margin_text} | {audit['low_top_score_margins']} | {low_pct_text} |"
        )
    lines.extend(
        [
            "",
            "## Input Hash Parity",
            "",
            "| Engine | Prompt matches | Prompt missing | Prompt mismatches | Choices matches | Choices missing | Choices mismatches | Input matches | Input missing | Input mismatches |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, audit in report["prediction_audits"].items():
        lines.append(
            f"| {name} | {audit['prompt_hash_matches']} | {audit['prompt_hash_missing']} | "
            f"{audit['prompt_hash_mismatches']} | {audit['choices_hash_matches']} | "
            f"{audit['choices_hash_missing']} | {audit['choices_hash_mismatches']} | "
            f"{audit['input_hash_matches']} | {audit['input_hash_missing']} | "
            f"{audit['input_hash_mismatches']} |"
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


def write_record_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "source",
        "row_number",
        "record_id",
        "valid",
        "answer_index",
        "predicted_index",
        "correct",
        "has_scores",
        "score_count",
        "top_score_tie_count",
        "top_score_margin",
        "prompt_hash_status",
        "choices_hash_status",
        "input_hash_status",
        "issue",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for audit in report["prediction_audits"].values():
            for row in audit["records"]:
                writer.writerow({field: row.get(field, "") for field in fields})


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


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    output_stem: str,
    record_csv: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{output_stem}.json"
    md_path = output_dir / f"{output_stem}.md"
    csv_path = output_dir / f"{output_stem}.csv"
    junit_path = output_dir / f"{output_stem}_junit.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(report, csv_path)
    write_junit(report, junit_path)
    if record_csv is not None:
        record_csv.parent.mkdir(parents=True, exist_ok=True)
        write_record_csv(report, record_csv)
    return json_path, md_path, csv_path, junit_path, record_csv


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
        "--max-majority-gold-answer-pct",
        type=float,
        help="Fail when one gold answer index covers more than this percentage of gold rows",
    )
    parser.add_argument(
        "--min-choices",
        type=int,
        help="Fail when any normalized gold row has fewer than this many choices",
    )
    parser.add_argument(
        "--max-choices",
        type=int,
        help="Fail when any normalized gold row has more than this many choices",
    )
    parser.add_argument(
        "--min-score-coverage-pct",
        type=float,
        help="Fail when either engine has score vectors for less than this percentage of valid rows",
    )
    parser.add_argument(
        "--max-top-score-tie-pct",
        type=float,
        help="Fail when tied maximum scores exceed this percentage of scored prediction rows for either engine",
    )
    parser.add_argument(
        "--min-top-score-margin",
        type=float,
        help="Fail when any scored prediction row has top-score minus second-score margin below this value",
    )
    parser.add_argument(
        "--require-input-hashes",
        action="store_true",
        help="Fail unless each valid prediction row carries matching prompt, choices, and combined input SHA256 metadata",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_input_audit_latest")
    parser.add_argument("--record-csv", type=Path, help="Optional per-engine prediction telemetry CSV path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_majority_gold_answer_pct is not None and not 0.0 <= args.max_majority_gold_answer_pct <= 100.0:
        print("error: --max-majority-gold-answer-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if args.max_majority_prediction_pct is not None and not 0.0 <= args.max_majority_prediction_pct <= 100.0:
        print("error: --max-majority-prediction-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if args.min_choices is not None and args.min_choices < 1:
        print("error: --min-choices must be at least 1", file=sys.stderr)
        return 2
    if args.max_choices is not None and args.max_choices < 1:
        print("error: --max-choices must be at least 1", file=sys.stderr)
        return 2
    if args.min_choices is not None and args.max_choices is not None and args.min_choices > args.max_choices:
        print("error: --min-choices cannot be greater than --max-choices", file=sys.stderr)
        return 2
    if args.min_score_coverage_pct is not None and not 0.0 <= args.min_score_coverage_pct <= 100.0:
        print("error: --min-score-coverage-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if args.max_top_score_tie_pct is not None and not 0.0 <= args.max_top_score_tie_pct <= 100.0:
        print("error: --max-top-score-tie-pct must be between 0 and 100", file=sys.stderr)
        return 2
    if args.min_top_score_margin is not None and args.min_top_score_margin < 0.0:
        print("error: --min-top-score-margin must be non-negative", file=sys.stderr)
        return 2
    report = build_report(args)
    json_path, md_path, csv_path, junit_path, record_csv_path = write_report(
        report,
        args.output_dir,
        args.output_stem,
        args.record_csv,
    )
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_junit={junit_path}")
    if record_csv_path is not None:
        print(f"wrote_record_csv={record_csv_path}")
    print(f"status={report['status']}")
    print(f"errors={report['summary']['errors']}")
    print(f"warnings={report['summary']['warnings']}")
    return 2 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
