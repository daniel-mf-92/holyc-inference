#!/usr/bin/env python3
"""Audit eval prediction prompt/input hashes against a local gold dataset.

This host-side tool reads local gold and prediction artifacts only. It verifies
optional or required `prompt_sha256`, `choices_sha256`, and `input_sha256`
metadata so HolyC-vs-llama eval runs cannot silently compare predictions
generated from stale prompts, reordered choices, or different input payloads.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack
import eval_compare


PROMPT_HASH_KEYS = ("prompt_sha256", "prompt_hash", "prompt_digest")
CHOICES_HASH_KEYS = ("choices_sha256", "choices_hash", "choices_digest")
INPUT_HASH_KEYS = ("input_sha256", "input_hash", "input_digest")


@dataclass(frozen=True)
class ExpectedHashes:
    record_id: str
    dataset: str
    split: str
    prompt_sha256: str
    choices_sha256: str
    input_sha256: str


@dataclass(frozen=True)
class HashRow:
    source: str
    record_id: str
    dataset: str
    split: str
    observed_prompt_sha256: str
    expected_prompt_sha256: str
    observed_choices_sha256: str
    expected_choices_sha256: str
    observed_input_sha256: str
    expected_input_sha256: str
    prompt_hash_ok: bool | None
    choices_hash_ok: bool | None
    input_hash_ok: bool | None


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def first_present(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def add_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


def expected_hashes(case: eval_compare.GoldCase) -> ExpectedHashes:
    prompt_hash = dataset_pack.sha256_text(case.prompt)
    choices_hash = dataset_pack.sha256_json(case.choices)
    input_hash = dataset_pack.sha256_json({"choices_sha256": choices_hash, "prompt_sha256": prompt_hash})
    return ExpectedHashes(
        record_id=case.record_id,
        dataset=case.dataset,
        split=case.split,
        prompt_sha256=prompt_hash,
        choices_sha256=choices_hash,
        input_sha256=input_hash,
    )


def compare_hash(
    findings: list[Finding],
    source: str,
    record_id: str,
    name: str,
    observed: str,
    expected: str,
    require_hashes: bool,
) -> bool | None:
    if not observed:
        if require_hashes:
            add_finding(findings, source, record_id, f"missing_{name}", f"{name} is required but absent")
        return None
    if observed != expected:
        add_finding(findings, source, record_id, f"{name}_mismatch", f"observed {observed} != expected {expected}")
        return False
    return True


def load_prediction_rows(path: Path, source: str, findings: list[Finding]) -> list[dict[str, Any]]:
    try:
        return eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        add_finding(findings, source, "", "load_error", f"cannot read predictions: {exc}")
        return []


def audit_source(
    path: Path,
    source: str,
    gold: dict[str, ExpectedHashes],
    require_hashes: bool,
    findings: list[Finding],
) -> list[HashRow]:
    rows: list[HashRow] = []
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

        expected = gold.get(record_id)
        if expected is None:
            add_finding(findings, source, record_id, "extra_prediction", "prediction id is not present in gold")
            continue

        observed_prompt = first_present(row, PROMPT_HASH_KEYS)
        observed_choices = first_present(row, CHOICES_HASH_KEYS)
        observed_input = first_present(row, INPUT_HASH_KEYS)
        prompt_ok = compare_hash(findings, source, record_id, "prompt_sha256", observed_prompt, expected.prompt_sha256, require_hashes)
        choices_ok = compare_hash(findings, source, record_id, "choices_sha256", observed_choices, expected.choices_sha256, require_hashes)
        input_ok = compare_hash(findings, source, record_id, "input_sha256", observed_input, expected.input_sha256, require_hashes)

        rows.append(
            HashRow(
                source=source,
                record_id=record_id,
                dataset=expected.dataset,
                split=expected.split,
                observed_prompt_sha256=observed_prompt,
                expected_prompt_sha256=expected.prompt_sha256,
                observed_choices_sha256=observed_choices,
                expected_choices_sha256=expected.choices_sha256,
                observed_input_sha256=observed_input,
                expected_input_sha256=expected.input_sha256,
                prompt_hash_ok=prompt_ok,
                choices_hash_ok=choices_ok,
                input_hash_ok=input_ok,
            )
        )

    for record_id in sorted(set(gold) - seen):
        add_finding(findings, source, record_id, "missing_prediction", "gold id is missing from predictions")
    return rows


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        cases = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        cases = {}
        add_finding(findings, "gold", "", "load_error", f"cannot read gold: {exc}")

    gold_hashes = {record_id: expected_hashes(case) for record_id, case in cases.items()}
    rows = [
        *audit_source(args.holyc, "holyc", gold_hashes, args.require_hashes, findings),
        *audit_source(args.llama, "llama.cpp", gold_hashes, args.require_hashes, findings),
    ]
    observed_hash_rows = [
        row
        for row in rows
        if row.observed_prompt_sha256 or row.observed_choices_sha256 or row.observed_input_sha256
    ]
    matched_hash_fields = sum(
        1
        for row in rows
        for value in (row.prompt_hash_ok, row.choices_hash_ok, row.input_hash_ok)
        if value is True
    )
    mismatched_hash_fields = sum(
        1
        for row in rows
        for value in (row.prompt_hash_ok, row.choices_hash_ok, row.input_hash_ok)
        if value is False
    )
    missing_hash_fields = sum(
        1
        for row in rows
        for value in (row.prompt_hash_ok, row.choices_hash_ok, row.input_hash_ok)
        if value is None
    )

    if args.min_hashed_rows is not None and len(observed_hash_rows) < args.min_hashed_rows:
        add_finding(
            findings,
            "eval",
            "",
            "min_hashed_rows",
            f"hashed prediction rows {len(observed_hash_rows)} is below --min-hashed-rows {args.min_hashed_rows}",
        )

    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {"gold": str(args.gold), "holyc": str(args.holyc), "llama": str(args.llama)},
        "gates": {"require_hashes": args.require_hashes, "min_hashed_rows": args.min_hashed_rows},
        "summary": {
            "gold_records": len(gold_hashes),
            "audited_prediction_rows": len(rows),
            "hashed_prediction_rows": len(observed_hash_rows),
            "matched_hash_fields": matched_hash_fields,
            "mismatched_hash_fields": mismatched_hash_fields,
            "missing_hash_fields": missing_hash_fields,
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Eval Prompt Hash Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold records | {summary['gold_records']} |",
        f"| Audited prediction rows | {summary['audited_prediction_rows']} |",
        f"| Hashed prediction rows | {summary['hashed_prediction_rows']} |",
        f"| Matched hash fields | {summary['matched_hash_fields']} |",
        f"| Mismatched hash fields | {summary['mismatched_hash_fields']} |",
        f"| Missing hash fields | {summary['missing_hash_fields']} |",
        f"| Findings | {summary['findings']} |",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", "", "| Source | Record | Kind | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                f"| {finding['source']} | {finding['record_id']} | {finding['kind']} | {finding['detail']} |"
            )
    else:
        lines.append("No prompt hash findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_prompt_hash_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "prompt_hashes"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} prompt hash finding(s)"})
        failure.text = "\n".join(f"{item['source']} {item['record_id']} {item['kind']}: {item['detail']}" for item in findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Gold eval JSONL dataset")
    parser.add_argument("--holyc", type=Path, required=True, help="HolyC prediction JSON/JSONL/CSV")
    parser.add_argument("--llama", type=Path, required=True, help="llama.cpp prediction JSON/JSONL/CSV")
    parser.add_argument("--dataset", default="eval", help="Fallback dataset name for normalized gold rows")
    parser.add_argument("--split", default="validation", help="Fallback split name for normalized gold rows")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_prompt_hash_audit_latest")
    parser.add_argument("--require-hashes", action="store_true", help="Fail when prediction rows omit hash metadata")
    parser.add_argument("--min-hashed-rows", type=int, help="Require at least this many prediction rows with any hash metadata")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)
    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", report)
    write_markdown(output_dir / f"{stem}.md", report)
    write_csv(
        output_dir / f"{stem}.csv",
        report["rows"],
        [
            "source",
            "record_id",
            "dataset",
            "split",
            "observed_prompt_sha256",
            "expected_prompt_sha256",
            "observed_choices_sha256",
            "expected_choices_sha256",
            "observed_input_sha256",
            "expected_input_sha256",
            "prompt_hash_ok",
            "choices_hash_ok",
            "input_hash_ok",
        ],
    )
    write_csv(
        output_dir / f"{stem}_findings.csv",
        report["findings"],
        ["severity", "source", "record_id", "kind", "detail"],
    )
    write_junit(output_dir / f"{stem}_junit.xml", report)
    if args.fail_on_findings and report["findings"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
