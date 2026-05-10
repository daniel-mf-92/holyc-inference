#!/usr/bin/env python3
"""Audit local eval JSONL files for duplicate normalized dataset rows.

This curation gate is offline-only. It normalizes rows with dataset_pack.py,
then hashes prompt text and prompt+choice payloads so repeated examples can be
reviewed before they become HolyC-loadable HCEval binaries.
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


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class DuplicateFinding:
    severity: str
    kind: str
    scope: str
    key_sha256: str
    record_count: int
    answer_indexes: str
    records: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def stable_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_sha256(record: dataset_pack.EvalRecord) -> str:
    return stable_json_sha256({"prompt": stable_text(record.prompt)})


def payload_sha256(record: dataset_pack.EvalRecord) -> str:
    return stable_json_sha256(
        {
            "prompt": stable_text(record.prompt),
            "choices": [stable_text(choice) for choice in record.choices],
        }
    )


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[DuplicateFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[DuplicateFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(DuplicateFinding("error", "read_error", "", "", 0, "", "", str(exc)))
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
                    DuplicateFinding(
                        "error",
                        "schema_error",
                        "",
                        "",
                        0,
                        "",
                        f"{path}:{index + 1}",
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def scope_label(record: dataset_pack.EvalRecord, scope: str) -> str:
    if scope == "global":
        return "global"
    if scope == "dataset":
        return record.dataset
    return f"{record.dataset}/{record.split}"


def group_records(records: list[LoadedRecord], *, key_name: str, scope: str) -> dict[tuple[str, str], list[LoadedRecord]]:
    groups: dict[tuple[str, str], list[LoadedRecord]] = collections.defaultdict(list)
    for loaded in records:
        key = prompt_sha256(loaded.record) if key_name == "prompt" else payload_sha256(loaded.record)
        groups[(scope_label(loaded.record, scope), key)].append(loaded)
    return dict(groups)


def record_refs(records: list[LoadedRecord]) -> list[str]:
    return [
        f"{source_ref(loaded)}#{loaded.record.dataset}/{loaded.record.split}/{loaded.record.record_id}"
        for loaded in records
    ]


def answer_indexes(records: list[LoadedRecord]) -> list[int]:
    return sorted({loaded.record.answer_index for loaded in records})


def append_duplicate_findings(
    records: list[LoadedRecord],
    findings: list[DuplicateFinding],
    *,
    key_name: str,
    scope: str,
    fail_on_duplicate: bool,
    fail_on_conflicting_answers: bool,
) -> None:
    for (label, key), grouped in sorted(group_records(records, key_name=key_name, scope=scope).items()):
        if len(grouped) < 2:
            continue
        answers = answer_indexes(grouped)
        conflict = len(answers) > 1
        if conflict and fail_on_conflicting_answers:
            severity = "error"
        else:
            severity = "error" if fail_on_duplicate else "warning"
        refs = record_refs(grouped)
        kind = f"duplicate_{key_name}"
        if conflict:
            kind = f"conflicting_duplicate_{key_name}"
        findings.append(
            DuplicateFinding(
                severity=severity,
                kind=kind,
                scope=label,
                key_sha256=key,
                record_count=len(grouped),
                answer_indexes=",".join(str(answer) for answer in answers),
                records=";".join(refs),
                detail=f"{len(grouped)} records share normalized {key_name} in {label}",
            )
        )


def build_record_rows(records: list[LoadedRecord], *, scope: str) -> list[dict[str, Any]]:
    prompt_groups = group_records(records, key_name="prompt", scope=scope)
    payload_groups = group_records(records, key_name="payload", scope=scope)
    rows: list[dict[str, Any]] = []
    for loaded in records:
        record = loaded.record
        label = scope_label(record, scope)
        prompt_key = prompt_sha256(record)
        payload_key = payload_sha256(record)
        rows.append(
            {
                "source": source_ref(loaded),
                "dataset": record.dataset,
                "split": record.split,
                "record_id": record.record_id,
                "scope": label,
                "answer_index": record.answer_index,
                "prompt_sha256": prompt_key,
                "payload_sha256": payload_key,
                "prompt_duplicate_count": len(prompt_groups[(label, prompt_key)]),
                "payload_duplicate_count": len(payload_groups[(label, payload_key)]),
            }
        )
    return rows


def counts_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    record_rows: list[dict[str, Any]],
    findings: list[DuplicateFinding],
    *,
    scope: str,
) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-dataset-duplicate-audit",
        "status": "fail" if error_count else "pass",
        "scope": scope,
        "inputs": inputs,
        "record_count": len(records),
        "counts_by_dataset_split": counts_by_dataset_split(records),
        "duplicate_prompt_record_count": sum(1 for row in record_rows if row["prompt_duplicate_count"] > 1),
        "duplicate_payload_record_count": sum(1 for row in record_rows if row["payload_duplicate_count"] > 1),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [asdict(finding) for finding in findings],
    }


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Duplicate Audit",
        "",
        f"- Status: {report['status']}",
        f"- Scope: {report['scope']}",
        f"- Records: {report['record_count']}",
        f"- Duplicate-prompt records: {report['duplicate_prompt_record_count']}",
        f"- Duplicate-payload records: {report['duplicate_payload_record_count']}",
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
            lines.append(f"| {md_cell(dataset)} | {md_cell(split)} | {count} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No duplicate findings.")
    else:
        lines.extend(
            [
                "| severity | kind | scope | record_count | answer_indexes | detail |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {scope} | {record_count} | {answer_indexes} | {detail} |".format(
                    **{key: md_cell(value) for key, value in finding.items()}
                )
            )
    return "\n".join(lines) + "\n"


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "scope", "key_sha256", "record_count", "answer_indexes", "records", "detail"],
        )
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_record_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "dataset",
        "split",
        "record_id",
        "scope",
        "answer_index",
        "prompt_sha256",
        "payload_sha256",
        "prompt_duplicate_count",
        "payload_duplicate_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_junit(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "dataset_duplicate_audit",
            "tests": "1",
            "failures": str(len(failures)),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "duplicate_rows"})
    for finding in failures:
        failure = ET.SubElement(case, "failure", {"message": f"{finding['kind']}: {finding['scope']}"})
        failure.text = json.dumps(finding, ensure_ascii=False, sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local eval JSONL input; repeatable")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata")
    parser.add_argument("--scope", choices=("global", "dataset", "dataset_split"), default="dataset_split")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary output")
    parser.add_argument("--csv", type=Path, help="Optional findings CSV output")
    parser.add_argument("--record-csv", type=Path, help="Optional per-record duplicate telemetry CSV output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--fail-on-duplicate-prompts", action="store_true", help="Treat duplicate normalized prompts as errors")
    parser.add_argument("--fail-on-duplicate-payloads", action="store_true", help="Treat duplicate normalized prompt+choice payloads as errors")
    parser.add_argument("--fail-on-conflicting-answers", action="store_true", help="Treat duplicate groups with conflicting answer indexes as errors")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit nonzero if any warning or error findings exist")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    records, inputs, findings = load_records(args.input, args.dataset, args.split)
    append_duplicate_findings(
        records,
        findings,
        key_name="prompt",
        scope=args.scope,
        fail_on_duplicate=args.fail_on_duplicate_prompts,
        fail_on_conflicting_answers=args.fail_on_conflicting_answers,
    )
    append_duplicate_findings(
        records,
        findings,
        key_name="payload",
        scope=args.scope,
        fail_on_duplicate=args.fail_on_duplicate_payloads,
        fail_on_conflicting_answers=args.fail_on_conflicting_answers,
    )
    record_rows = build_record_rows(records, scope=args.scope)
    report = build_report(inputs, records, record_rows, findings, scope=args.scope)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_findings_csv(report, args.csv)
    if args.record_csv:
        write_record_csv(record_rows, args.record_csv)
    if args.junit:
        write_junit(report, args.junit)

    if report["error_count"]:
        return 1
    if args.fail_on_findings and report["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
