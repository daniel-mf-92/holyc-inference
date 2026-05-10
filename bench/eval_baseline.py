#!/usr/bin/env python3
"""Compute offline multiple-choice eval baselines from a local gold set.

This host-side tool reads a gold JSONL dataset, normalizes it through the same
loader used by eval_compare, and reports deterministic majority-class and
uniform-random baselines. It performs no network I/O and never launches QEMU.
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

import eval_compare


@dataclass(frozen=True)
class BaselineRow:
    scope: str
    dataset: str
    split: str
    records: int
    choice_count_min: int
    choice_count_max: int
    answer_index_histogram: str
    majority_answer_index: int
    majority_correct: int
    majority_accuracy: float
    random_expected_correct: float
    random_expected_accuracy: float


@dataclass(frozen=True)
class Finding:
    gate: str
    scope: str
    value: float | int | str
    threshold: float | int | str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def histogram(cases: list[eval_compare.GoldCase]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for case in cases:
        counts[case.answer_index] = counts.get(case.answer_index, 0) + 1
    return dict(sorted(counts.items()))


def summarize(scope: str, dataset: str, split: str, cases: list[eval_compare.GoldCase]) -> BaselineRow:
    counts = histogram(cases)
    choice_counts = [len(case.choices) for case in cases]
    majority_answer = min(counts, key=lambda answer: (-counts[answer], answer)) if counts else -1
    majority_correct = counts.get(majority_answer, 0)
    random_expected = sum(1.0 / len(case.choices) for case in cases if case.choices)
    return BaselineRow(
        scope=scope,
        dataset=dataset,
        split=split,
        records=len(cases),
        choice_count_min=min(choice_counts) if choice_counts else 0,
        choice_count_max=max(choice_counts) if choice_counts else 0,
        answer_index_histogram=json.dumps(counts, sort_keys=True, separators=(",", ":")),
        majority_answer_index=majority_answer,
        majority_correct=majority_correct,
        majority_accuracy=safe_div(majority_correct, len(cases)),
        random_expected_correct=random_expected,
        random_expected_accuracy=safe_div(random_expected, len(cases)),
    )


def build_rows(gold: dict[str, eval_compare.GoldCase]) -> list[BaselineRow]:
    cases = list(gold.values())
    rows = [summarize("overall", "", "", cases)]
    groups: dict[tuple[str, str], list[eval_compare.GoldCase]] = {}
    for case in cases:
        groups.setdefault((case.dataset, case.split), []).append(case)
    for (dataset, split), group_cases in sorted(groups.items()):
        rows.append(summarize("dataset_split", dataset, split, group_cases))
    return rows


def evaluate(rows: list[BaselineRow], *, min_records: int, max_majority_accuracy: float | None) -> list[Finding]:
    findings: list[Finding] = []
    overall = rows[0] if rows else None
    if overall is None or overall.records < min_records:
        findings.append(
            Finding(
                "min_records",
                "overall",
                overall.records if overall else 0,
                min_records,
                "gold set has too few normalized records",
            )
        )
    if max_majority_accuracy is not None:
        for row in rows:
            if row.majority_accuracy > max_majority_accuracy:
                findings.append(
                    Finding(
                        "max_majority_accuracy",
                        row.scope if row.scope == "overall" else f"{row.dataset}/{row.split}",
                        row.majority_accuracy,
                        max_majority_accuracy,
                        "majority baseline is above the configured skew threshold",
                    )
                )
    return findings


def write_json(path: Path, rows: list[BaselineRow], findings: list[Finding], args: argparse.Namespace) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": iso_now(),
                "status": "fail" if findings else "pass",
                "gold_sha256": eval_compare.file_sha256(args.gold),
                "dataset": args.dataset,
                "split": args.split,
                "summary": asdict(rows[0]) if rows else {},
                "baselines": [asdict(row) for row in rows],
                "findings": [asdict(finding) for finding in findings],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[BaselineRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BaselineRow.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, rows: list[BaselineRow], findings: list[Finding]) -> None:
    lines = ["# Eval Baselines", ""]
    if rows:
        overall = rows[0]
        lines.extend(
            [
                f"Records: {overall.records}",
                f"Majority accuracy: {overall.majority_accuracy:.6f}",
                f"Uniform-random expected accuracy: {overall.random_expected_accuracy:.6f}",
                "",
                "| Scope | Dataset | Split | Records | Majority | Random Expected |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.scope} | {row.dataset} | {row.split} | {row.records} | "
                f"{row.majority_accuracy:.6f} | {row.random_expected_accuracy:.6f} |"
            )
    if findings:
        lines.extend(["", "## Findings", "", "| Gate | Scope | Value | Threshold | Message |", "| --- | --- | ---: | ---: | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.gate} | {finding.scope} | {finding.value} | {finding.threshold} | {finding.message} |"
            )
    else:
        lines.extend(["", "No baseline gate findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_baseline",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "eval_baseline", "name": "baseline_gates"})
    for finding in findings:
        case = ET.SubElement(suite, "testcase", {"classname": "eval_baseline", "name": finding.gate})
        failure = ET.SubElement(case, "failure", {"type": finding.gate, "message": finding.message})
        failure.text = f"{finding.scope}: {finding.value} > {finding.threshold}"
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="Local gold JSONL dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name passed to the shared eval normalizer")
    parser.add_argument("--split", required=True, help="Split name passed to the shared eval normalizer")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_baseline_latest")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--max-majority-accuracy", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_records < 0:
        parser.error("--min-records must be >= 0")
    if args.max_majority_accuracy is not None and not 0.0 <= args.max_majority_accuracy <= 1.0:
        parser.error("--max-majority-accuracy must be between 0 and 1")

    gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    rows = build_rows(gold)
    findings = evaluate(rows, min_records=args.min_records, max_majority_accuracy=args.max_majority_accuracy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings, args)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
