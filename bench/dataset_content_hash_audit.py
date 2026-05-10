#!/usr/bin/env python3
"""Audit local eval JSONL row-level prompt/choice/input hashes.

This curation gate is offline-only. It normalizes rows with dataset_pack.py,
then checks embedded prompt, choices, and input SHA-256 metadata against the
normalized record content so packed eval slices remain reproducible.
"""

from __future__ import annotations

import argparse
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

import dataset_pack


PROMPT_HASH_KEYS = ("prompt_sha256", "prompt_hash")
CHOICES_HASH_KEYS = ("choices_sha256", "choices_hash")
INPUT_HASH_KEYS = ("input_sha256", "prompt_choices_sha256")


@dataclass(frozen=True)
class HashRecord:
    source: str
    row_number: int
    record_id: str
    dataset: str
    split: str
    prompt_hash: str
    choices_hash: str
    input_hash: str
    prompt_hash_status: str
    choices_hash_status: str
    input_hash_status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    row_number: int
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choices_sha256(choices: list[str]) -> str:
    payload = json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def input_sha256(prompt_hash: str, choices_hash: str) -> str:
    payload = json.dumps(
        {"choices_sha256": choices_hash, "prompt_sha256": prompt_hash},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def metadata_value(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            value = metadata.get(key)
        if value not in (None, ""):
            return key, str(value).strip()
    return "", ""


def hash_status(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    record_id: str,
    label: str,
    found_key: str,
    found_value: str,
    expected_value: str,
    required: bool,
) -> str:
    if not found_value:
        if required:
            findings.append(
                Finding(
                    "error",
                    str(source),
                    row_number,
                    record_id,
                    f"missing_{label}_hash",
                    f"missing {label} hash metadata",
                )
            )
        return "missing"
    if found_value.lower() != expected_value.lower():
        findings.append(
            Finding(
                "error",
                str(source),
                row_number,
                record_id,
                f"{label}_hash_mismatch",
                f"{found_key}={found_value!r} expected {expected_value!r}",
            )
        )
        return "mismatch"
    return "match"


def load_path(path: Path, args: argparse.Namespace) -> tuple[list[HashRecord], list[Finding]]:
    records: list[HashRecord] = []
    findings: list[Finding] = []
    try:
        rows = dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return records, [Finding("error", str(path), 0, "", "read_error", str(exc))]

    for index, row in enumerate(rows):
        row_number = index + 1
        try:
            record = dataset_pack.normalize_row(row, index, args.default_dataset, args.default_split)
        except ValueError as exc:
            findings.append(Finding("error", str(path), row_number, "", "schema_error", str(exc)))
            continue

        prompt_hash = sha256_text(record.prompt)
        choices_hash = choices_sha256(record.choices)
        combined_hash = input_sha256(prompt_hash, choices_hash)
        prompt_key, prompt_found = metadata_value(row, PROMPT_HASH_KEYS)
        choices_key, choices_found = metadata_value(row, CHOICES_HASH_KEYS)
        input_key, input_found = metadata_value(row, INPUT_HASH_KEYS)
        prompt_status = hash_status(
            findings,
            source=path,
            row_number=row_number,
            record_id=record.record_id,
            label="prompt",
            found_key=prompt_key,
            found_value=prompt_found,
            expected_value=prompt_hash,
            required=args.require_prompt_hash,
        )
        choices_status = hash_status(
            findings,
            source=path,
            row_number=row_number,
            record_id=record.record_id,
            label="choices",
            found_key=choices_key,
            found_value=choices_found,
            expected_value=choices_hash,
            required=args.require_choices_hash,
        )
        input_status = hash_status(
            findings,
            source=path,
            row_number=row_number,
            record_id=record.record_id,
            label="input",
            found_key=input_key,
            found_value=input_found,
            expected_value=combined_hash,
            required=args.require_input_hash,
        )
        records.append(
            HashRecord(
                source=str(path),
                row_number=row_number,
                record_id=record.record_id,
                dataset=record.dataset,
                split=record.split,
                prompt_hash=prompt_hash,
                choices_hash=choices_hash,
                input_hash=combined_hash,
                prompt_hash_status=prompt_status,
                choices_hash_status=choices_status,
                input_hash_status=input_status,
            )
        )
    return records, findings


def status_counts(records: list[HashRecord], field: str) -> dict[str, int]:
    counts = {"match": 0, "missing": 0, "mismatch": 0}
    for record in records:
        counts[getattr(record, field)] += 1
    return counts


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records: list[HashRecord] = []
    findings: list[Finding] = []
    for path in args.input:
        path_records, path_findings = load_path(path, args)
        records.extend(path_records)
        findings.extend(path_findings)

    if len(records) < args.min_records:
        findings.append(
            Finding(
                "error",
                "",
                0,
                "",
                "insufficient_records",
                f"record_count={len(records)} minimum={args.min_records}",
            )
        )

    summary = {
        "record_count": len(records),
        "input_count": len(args.input),
        "prompt_hash": status_counts(records, "prompt_hash_status"),
        "choices_hash": status_counts(records, "choices_hash_status"),
        "input_hash": status_counts(records, "input_hash_status"),
    }
    return {
        "created_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": summary,
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Content Hash Audit",
        "",
        f"- status: {report['status']}",
        f"- records: {report['summary']['record_count']}",
        f"- findings: {len(report['findings'])}",
        f"- prompt hashes: {report['summary']['prompt_hash']}",
        f"- choices hashes: {report['summary']['choices_hash']}",
        f"- input hashes: {report['summary']['input_hash']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        for item in report["findings"]:
            row = f":{item['row_number']}" if item["row_number"] else ""
            record_id = f" `{item['record_id']}`" if item["record_id"] else ""
            lines.append(f"- {item['severity']} {item['kind']} `{item['source']}{row}`{record_id}: {item['detail']}")
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(HashRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["records"])


def write_findings_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["findings"])


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_content_hash_audit",
            "tests": str(max(1, report["summary"]["record_count"])),
            "failures": str(len(report["findings"]) if report["status"] == "fail" else 0),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "dataset_content_hash_audit"})
    if report["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{len(report['findings'])} content hash finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Eval JSONL input; repeatable")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--require-prompt-hash", action="store_true")
    parser.add_argument("--require-choices-hash", action="store_true")
    parser.add_argument("--require-input-hash", action="store_true")
    parser.add_argument("--require-all-hashes", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="dataset_content_hash_audit_latest")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.require_all_hashes:
        args.require_prompt_hash = True
        args.require_choices_hash = True
        args.require_input_hash = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", report)
    write_markdown(output_dir / f"{stem}.md", report)
    write_records_csv(output_dir / f"{stem}.csv", report)
    write_findings_csv(output_dir / f"{stem}_findings.csv", report)
    write_junit(output_dir / f"{stem}_junit.xml", report)
    print(f"dataset_content_hash_audit_status={report['status']} findings={len(report['findings'])}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
