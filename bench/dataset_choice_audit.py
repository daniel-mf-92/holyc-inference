#!/usr/bin/env python3
"""Audit local multiple-choice eval rows for choice-text quality issues.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then flags duplicate options, overlapping options, answer text
leaked in prompts, choice label prefixes, and large within-record choice length
skew before rows are curated or packed into HCEval binaries.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack


LABEL_PREFIX_RE = re.compile(r"^\s*(?:\(?[A-Ha-h]\)?[\.\):]|[0-9]{1,2}[\.\):])\s+")


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class ChoiceFinding:
    severity: str
    kind: str
    source: str
    dataset: str
    split: str
    record_id: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text_key(text: str) -> str:
    normalized = dataset_pack.clean_text(text).casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def source_ref(record: LoadedRecord) -> str:
    return f"{record.source}:{record.row_number}"


def add_finding(
    findings: list[ChoiceFinding],
    loaded: LoadedRecord,
    severity: str,
    kind: str,
    detail: str,
) -> None:
    record = loaded.record
    findings.append(
        ChoiceFinding(
            severity=severity,
            kind=kind,
            source=source_ref(loaded),
            dataset=record.dataset,
            split=record.split,
            record_id=record.record_id,
            detail=detail,
        )
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(paths: Iterable[Path], default_dataset: str, default_split: str) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[ChoiceFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[ChoiceFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(
                ChoiceFinding(
                    severity="error",
                    kind="read_error",
                    source=str(path),
                    dataset="",
                    split="",
                    record_id="",
                    detail=str(exc),
                )
            )
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(
                    ChoiceFinding(
                        severity="error",
                        kind="schema_error",
                        source=f"{path}:{index + 1}",
                        dataset="",
                        split="",
                        record_id="",
                        detail=str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def duplicate_choice_groups(choices: list[str]) -> dict[str, list[int]]:
    positions: dict[str, list[int]] = {}
    for index, choice in enumerate(choices):
        positions.setdefault(stable_text_key(choice), []).append(index)
    return {key: indexes for key, indexes in positions.items() if key and len(indexes) > 1}


def choice_length_ratio(choices: list[str]) -> float | None:
    byte_lengths = [len(choice.encode("utf-8")) for choice in choices if choice]
    if not byte_lengths:
        return None
    shortest = min(byte_lengths)
    if shortest == 0:
        return None
    return max(byte_lengths) / shortest


def choice_overlap_pairs(choices: list[str], min_overlap_chars: int) -> list[dict[str, Any]]:
    keyed = [(index, stable_text_key(choice)) for index, choice in enumerate(choices)]
    pairs: list[dict[str, Any]] = []
    for left_pos, (left_index, left_key) in enumerate(keyed):
        if len(left_key) < min_overlap_chars:
            continue
        for right_index, right_key in keyed[left_pos + 1 :]:
            if len(right_key) < min_overlap_chars or left_key == right_key:
                continue
            shorter, longer = (left_key, right_key) if len(left_key) <= len(right_key) else (right_key, left_key)
            if shorter in longer:
                pairs.append(
                    {
                        "choice_indexes": [left_index, right_index],
                        "shorter_chars": len(shorter),
                        "longer_chars": len(longer),
                    }
                )
    return pairs


def record_telemetry(
    loaded: LoadedRecord,
    min_choice_overlap_chars: int,
    min_answer_leak_chars: int,
    min_choice_leak_chars: int,
) -> dict[str, Any]:
    record = loaded.record
    choice_lengths = [len(choice.encode("utf-8")) for choice in record.choices]
    stable_choices = [stable_text_key(choice) for choice in record.choices]
    prompt = stable_text_key(record.prompt)
    duplicate_groups = duplicate_choice_groups(record.choices)
    overlap_pairs = choice_overlap_pairs(record.choices, min_choice_overlap_chars)
    answer_choice = stable_choices[record.answer_index]
    leaked_choice_indexes = [
        index
        for index, choice_key in enumerate(stable_choices)
        if choice_key and len(choice_key) >= min_choice_leak_chars and choice_key in prompt
    ]
    label_prefix_count = sum(1 for choice in record.choices if LABEL_PREFIX_RE.match(choice))
    min_choice_bytes = min(choice_lengths, default=0)
    max_choice_bytes = max(choice_lengths, default=0)
    return {
        "source": source_ref(loaded),
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "choice_count": len(record.choices),
        "answer_index": record.answer_index,
        "min_choice_bytes": min_choice_bytes,
        "max_choice_bytes": max_choice_bytes,
        "total_choice_bytes": sum(choice_lengths),
        "choice_length_ratio": choice_length_ratio(record.choices),
        "duplicate_choice_group_count": len(duplicate_groups),
        "duplicate_choice_text_count": sum(len(indexes) for indexes in duplicate_groups.values()),
        "choice_overlap_pair_count": len(overlap_pairs),
        "choice_label_prefix_count": label_prefix_count,
        "prompt_contains_correct_choice": (
            bool(answer_choice and len(answer_choice) >= min_answer_leak_chars and answer_choice in prompt)
        ),
        "prompt_choice_leak_count": len(leaked_choice_indexes),
        "prompt_choice_leak_indexes": ",".join(str(index) for index in leaked_choice_indexes),
    }


def build_record_telemetry(
    records: list[LoadedRecord],
    min_choice_overlap_chars: int,
    min_answer_leak_chars: int,
    min_choice_leak_chars: int,
) -> list[dict[str, Any]]:
    return [
        record_telemetry(
            loaded,
            min_choice_overlap_chars,
            min_answer_leak_chars,
            min_choice_leak_chars,
        )
        for loaded in records
    ]


def audit_records(
    records: list[LoadedRecord],
    fail_on_duplicate_choices: bool,
    fail_on_choice_overlap: bool,
    fail_on_label_prefixes: bool,
    fail_on_prompt_answer_leak: bool,
    fail_on_prompt_choice_leak: bool,
    fail_on_length_skew: bool,
    max_choice_length_ratio: float | None,
    min_choice_overlap_chars: int,
    min_answer_leak_chars: int,
    min_choice_leak_chars: int,
    findings: list[ChoiceFinding],
) -> None:
    for loaded in records:
        record = loaded.record

        duplicates = duplicate_choice_groups(record.choices)
        if duplicates:
            details = []
            for key, indexes in sorted(duplicates.items()):
                one_based = [index + 1 for index in indexes]
                details.append(f"{key!r} at choices {one_based}")
            add_finding(
                findings,
                loaded,
                "error" if fail_on_duplicate_choices else "warning",
                "duplicate_choice_text",
                "; ".join(details),
            )

        overlaps = choice_overlap_pairs(record.choices, min_choice_overlap_chars)
        if overlaps:
            add_finding(
                findings,
                loaded,
                "error" if fail_on_choice_overlap else "warning",
                "choice_text_overlap",
                json.dumps(overlaps, ensure_ascii=False, sort_keys=True),
            )

        prefixed = [
            {"choice_index": index, "choice": choice}
            for index, choice in enumerate(record.choices)
            if LABEL_PREFIX_RE.match(choice)
        ]
        if prefixed:
            add_finding(
                findings,
                loaded,
                "error" if fail_on_label_prefixes else "warning",
                "choice_label_prefix",
                json.dumps(prefixed, ensure_ascii=False, sort_keys=True),
            )

        correct_choice = stable_text_key(record.choices[record.answer_index])
        prompt = stable_text_key(record.prompt)
        if len(correct_choice) >= min_answer_leak_chars and correct_choice in prompt:
            add_finding(
                findings,
                loaded,
                "error" if fail_on_prompt_answer_leak else "warning",
                "prompt_contains_correct_choice",
                f"correct choice text appears in normalized prompt ({len(correct_choice)} chars)",
            )

        leaked_choices = [
            {"choice_index": index, "is_answer": index == record.answer_index, "chars": len(choice_key)}
            for index, choice in enumerate(record.choices)
            if (choice_key := stable_text_key(choice))
            and len(choice_key) >= min_choice_leak_chars
            and choice_key in prompt
        ]
        if leaked_choices:
            add_finding(
                findings,
                loaded,
                "error" if fail_on_prompt_choice_leak else "warning",
                "prompt_contains_choice_text",
                json.dumps(leaked_choices, ensure_ascii=False, sort_keys=True),
            )

        ratio = choice_length_ratio(record.choices)
        if max_choice_length_ratio is not None and ratio is not None and ratio > max_choice_length_ratio:
            add_finding(
                findings,
                loaded,
                "error" if fail_on_length_skew else "warning",
                "choice_length_skew",
                f"max/min UTF-8 choice byte ratio {ratio:.2f} exceeds {max_choice_length_ratio:.2f}",
            )


def counts_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        dataset_counts = counts.setdefault(loaded.record.dataset, {})
        split = loaded.record.split
        dataset_counts[split] = dataset_counts.get(split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def build_report(inputs: list[dict[str, Any]], records: list[LoadedRecord], findings: list[ChoiceFinding]) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-choice-audit",
        "status": "fail" if error_count else "pass",
        "inputs": inputs,
        "record_count": len(records),
        "counts_by_dataset_split": counts_by_dataset_split(records),
        "choice_count_histogram": sorted_counts(len(loaded.record.choices) for loaded in records),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Choice Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Dataset/Split Counts",
        "",
        "| dataset | split | records |",
        "| --- | --- | ---: |",
    ]
    for dataset, split_counts in report["counts_by_dataset_split"].items():
        for split, count in split_counts.items():
            lines.append(f"| {dataset} | {split} | {count} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No choice quality findings.")
    else:
        lines.extend(
            [
                "| severity | kind | source | dataset | split | record_id | detail |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {source} | {dataset} | {split} | {record_id} | {detail} |".format(
                    **finding
                )
            )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "source", "dataset", "split", "record_id", "detail"],
        )
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_record_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "choice_count",
        "answer_index",
        "min_choice_bytes",
        "max_choice_bytes",
        "total_choice_bytes",
        "choice_length_ratio",
        "duplicate_choice_group_count",
        "duplicate_choice_text_count",
        "choice_overlap_pair_count",
        "choice_label_prefix_count",
        "prompt_contains_correct_choice",
        "prompt_choice_leak_count",
        "prompt_choice_leak_indexes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def write_junit(report: dict[str, Any], path: Path) -> None:
    error_findings = [finding for finding in report["findings"] if finding["severity"] == "error"]
    testcase_count = len(error_findings) + (1 if not error_findings else 0)
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_choice_audit",
            "tests": str(testcase_count),
            "failures": str(len(error_findings)),
            "errors": "0",
        },
    )
    if error_findings:
        for finding in error_findings:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "classname": "dataset_choice_audit",
                    "name": f"{finding['kind']}:{finding['source']}",
                },
            )
            ET.SubElement(testcase, "failure", {"type": finding["kind"], "message": finding["detail"]})
    else:
        ET.SubElement(testsuite, "testcase", {"classname": "dataset_choice_audit", "name": "choice_audit_pass"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(
    report: dict[str, Any],
    output: Path,
    markdown: Path | None,
    csv_path: Path | None,
    record_csv: Path | None,
    junit: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(report), encoding="utf-8")
    if csv_path:
        write_csv(report, csv_path)
    if record_csv:
        write_record_csv(report["record_telemetry"], record_csv)
    if junit:
        write_junit(report, junit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input eval JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report")
    parser.add_argument("--csv", type=Path, help="Optional CSV findings output")
    parser.add_argument("--record-csv", type=Path, help="Optional per-record choice telemetry CSV output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument("--fail-on-duplicate-choices", action="store_true", help="Fail on duplicate options in a row")
    parser.add_argument(
        "--fail-on-choice-overlap",
        action="store_true",
        help="Fail when one normalized choice text contains another choice text in the same row",
    )
    parser.add_argument("--fail-on-label-prefixes", action="store_true", help="Fail when choices include A./B./1) prefixes")
    parser.add_argument(
        "--fail-on-prompt-answer-leak",
        action="store_true",
        help="Fail when the normalized prompt contains the correct choice text",
    )
    parser.add_argument(
        "--fail-on-prompt-choice-leak",
        action="store_true",
        help="Fail when the normalized prompt contains any candidate choice text",
    )
    parser.add_argument("--fail-on-length-skew", action="store_true", help="Fail when --max-choice-length-ratio is exceeded")
    parser.add_argument(
        "--max-choice-length-ratio",
        type=float,
        help="Warn/fail when longest choice bytes divided by shortest choice bytes exceeds this ratio",
    )
    parser.add_argument(
        "--min-choice-overlap-chars",
        type=int,
        default=8,
        help="Minimum normalized choice chars before same-row choice overlap detection runs",
    )
    parser.add_argument(
        "--min-answer-leak-chars",
        type=int,
        default=12,
        help="Minimum normalized correct-choice chars before prompt leak detection runs",
    )
    parser.add_argument(
        "--min-choice-leak-chars",
        type=int,
        default=12,
        help="Minimum normalized choice chars before any-choice prompt leak detection runs",
    )
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit nonzero when errors are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_choice_length_ratio is not None and args.max_choice_length_ratio < 1.0:
        print("error: --max-choice-length-ratio must be at least 1.0", file=sys.stderr)
        return 2
    if args.min_choice_overlap_chars < 1:
        print("error: --min-choice-overlap-chars must be at least 1", file=sys.stderr)
        return 2
    if args.min_answer_leak_chars < 1:
        print("error: --min-answer-leak-chars must be at least 1", file=sys.stderr)
        return 2
    if args.min_choice_leak_chars < 1:
        print("error: --min-choice-leak-chars must be at least 1", file=sys.stderr)
        return 2

    records, inputs, findings = load_records(args.input, args.default_dataset, args.default_split)
    audit_records(
        records,
        args.fail_on_duplicate_choices,
        args.fail_on_choice_overlap,
        args.fail_on_label_prefixes,
        args.fail_on_prompt_answer_leak,
        args.fail_on_prompt_choice_leak,
        args.fail_on_length_skew,
        args.max_choice_length_ratio,
        args.min_choice_overlap_chars,
        args.min_answer_leak_chars,
        args.min_choice_leak_chars,
        findings,
    )
    report = build_report(inputs, records, findings)
    report["record_telemetry"] = build_record_telemetry(
        records,
        args.min_choice_overlap_chars,
        args.min_answer_leak_chars,
        args.min_choice_leak_chars,
    )
    write_outputs(report, args.output, args.markdown, args.csv, args.record_csv, args.junit)

    print(f"wrote_report={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.record_csv:
        print(f"wrote_record_csv={args.record_csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")

    if args.fail_on_findings and report["error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
