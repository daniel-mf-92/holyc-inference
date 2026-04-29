#!/usr/bin/env python3
"""Emit stable row fingerprints for local eval JSONL datasets.

This host-side, offline-only helper normalizes the same row shapes accepted by
dataset_pack.py and writes prompt/choice/input hashes that can be carried into
HolyC and llama.cpp prediction files for apples-to-apples eval audits.
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

import dataset_pack


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    source: str
    detail: str


@dataclass(frozen=True)
class Fingerprint:
    source: str
    row_number: int
    record_id: str
    dataset: str
    split: str
    choice_count: int
    answer_index: int
    prompt_bytes: int
    choices_bytes: int
    record_payload_bytes: int
    prompt_sha256: str
    choices_sha256: str
    input_sha256: str
    answer_payload_sha256: str
    full_payload_sha256: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path, row_number: int) -> str:
    return f"{path}:{row_number}"


def append_finding(findings: list[Finding], severity: str, kind: str, source: str, detail: str) -> None:
    findings.append(Finding(severity=severity, kind=kind, source=source, detail=detail))


def fingerprint_record(source: Path, row_number: int, record: dataset_pack.EvalRecord) -> Fingerprint:
    prompt_hash = sha256_text(record.prompt)
    choices_hash = sha256_json(record.choices)
    input_hash = sha256_json({"choices_sha256": choices_hash, "prompt_sha256": prompt_hash})
    answer_payload_hash = sha256_json(
        {
            "answer_index": record.answer_index,
            "choices_sha256": choices_hash,
            "input_sha256": input_hash,
            "prompt_sha256": prompt_hash,
        }
    )
    full_payload_hash = sha256_json(asdict(record))
    return Fingerprint(
        source=str(source),
        row_number=row_number,
        record_id=record.record_id,
        dataset=record.dataset,
        split=record.split,
        choice_count=len(record.choices),
        answer_index=record.answer_index,
        prompt_bytes=len(record.prompt.encode("utf-8")),
        choices_bytes=sum(len(choice.encode("utf-8")) for choice in record.choices),
        record_payload_bytes=dataset_pack.record_payload_bytes(record),
        prompt_sha256=prompt_hash,
        choices_sha256=choices_hash,
        input_sha256=input_hash,
        answer_payload_sha256=answer_payload_hash,
        full_payload_sha256=full_payload_hash,
    )


def load_fingerprints(
    inputs: Iterable[Path],
    default_dataset: str,
    default_split: str,
    findings: list[Finding],
) -> tuple[list[Fingerprint], list[dict[str, Any]]]:
    fingerprints: list[Fingerprint] = []
    sources: list[dict[str, Any]] = []
    for path in inputs:
        try:
            rows = dataset_pack.read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            append_finding(findings, "error", "read_error", str(path), str(exc))
            sources.append({"path": str(path), "rows": 0})
            continue

        source_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            source_info["sha256"] = file_sha256(path)
        sources.append(source_info)

        for index, row in enumerate(rows):
            row_number = index + 1
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                append_finding(findings, "error", "schema_error", source_ref(path, row_number), str(exc))
                continue
            fingerprints.append(fingerprint_record(path, row_number, record))
    return fingerprints, sources


def grouped_duplicates(fingerprints: list[Fingerprint], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Fingerprint]] = collections.defaultdict(list)
    for fingerprint in fingerprints:
        groups[str(getattr(fingerprint, field))].append(fingerprint)

    duplicates: list[dict[str, Any]] = []
    for value, group in sorted(groups.items()):
        if len(group) <= 1:
            continue
        answers = sorted({item.answer_index for item in group})
        duplicates.append(
            {
                field: value,
                "records": len(group),
                "record_ids": sorted({item.record_id for item in group}),
                "answers": answers,
                "conflicting_answers": len(answers) > 1,
                "sources": [source_ref(Path(item.source), item.row_number) for item in group],
            }
        )
    return duplicates


def sorted_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = collections.Counter(str(value) for value in values)
    return {key: counter[key] for key in sorted(counter)}


def nested_counts(fingerprints: list[Fingerprint]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for fingerprint in fingerprints:
        split_counts = counts.setdefault(fingerprint.dataset, {})
        split_counts[fingerprint.split] = split_counts.get(fingerprint.split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def build_report(
    fingerprints: list[Fingerprint],
    sources: list[dict[str, Any]],
    findings: list[Finding],
    fail_on_duplicate_ids: bool,
    fail_on_duplicate_inputs: bool,
    fail_on_conflicting_input_answers: bool,
    fail_on_findings: bool,
) -> dict[str, Any]:
    duplicate_ids = grouped_duplicates(fingerprints, "record_id")
    duplicate_inputs = grouped_duplicates(fingerprints, "input_sha256")
    conflicting_inputs = [item for item in duplicate_inputs if item["conflicting_answers"]]

    if fail_on_duplicate_ids:
        for item in duplicate_ids:
            append_finding(
                findings,
                "error",
                "duplicate_record_id",
                item["record_id"],
                f"{item['records']} rows share record_id",
            )
    if fail_on_duplicate_inputs:
        for item in duplicate_inputs:
            append_finding(
                findings,
                "error",
                "duplicate_input_sha256",
                item["input_sha256"],
                f"{item['records']} rows share prompt+choices input",
            )
    if fail_on_conflicting_input_answers:
        for item in conflicting_inputs:
            append_finding(
                findings,
                "error",
                "conflicting_input_answers",
                item["input_sha256"],
                f"answers={item['answers']} record_ids={item['record_ids']}",
            )

    enabled_gate = (
        fail_on_duplicate_ids
        or fail_on_duplicate_inputs
        or fail_on_conflicting_input_answers
        or fail_on_findings
    )
    status = "fail" if enabled_gate and any(finding.severity == "error" for finding in findings) else "pass"
    return {
        "generated_at": iso_now(),
        "status": status,
        "sources": sources,
        "record_count": len(fingerprints),
        "dataset_split_counts": nested_counts(fingerprints),
        "choice_count_histogram": sorted_counts(item.choice_count for item in fingerprints),
        "answer_histogram": sorted_counts(item.answer_index for item in fingerprints),
        "duplicate_record_ids": duplicate_ids,
        "duplicate_inputs": duplicate_inputs,
        "conflicting_input_answers": conflicting_inputs,
        "findings": [asdict(finding) for finding in findings],
        "fingerprints": [asdict(fingerprint) for fingerprint in fingerprints],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, fingerprints: list[Fingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for fingerprint in fingerprints:
            handle.write(json.dumps(asdict(fingerprint), sort_keys=True) + "\n")


def write_csv(path: Path, fingerprints: list[Fingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Fingerprint.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fingerprint in fingerprints:
            writer.writerow(asdict(fingerprint))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Eval Dataset Fingerprints",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Findings: {len(report['findings'])}",
        f"- Duplicate record IDs: {len(report['duplicate_record_ids'])}",
        f"- Duplicate inputs: {len(report['duplicate_inputs'])}",
        f"- Conflicting input answers: {len(report['conflicting_input_answers'])}",
        "",
        "## Dataset/Split Counts",
        "",
    ]
    for dataset, split_counts in report["dataset_split_counts"].items():
        for split, count in split_counts.items():
            lines.append(f"- {dataset}/{split}: {count}")
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['source']}: {finding['detail']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_fingerprint",
            "tests": "1",
            "failures": str(len(failures)),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "dataset_fingerprint"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(failures)} dataset fingerprint findings"})
        failure.text = "\n".join(f"{item['kind']}: {item['source']}: {item['detail']}" for item in failures)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path, help="Local eval JSONL input")
    parser.add_argument("--output", required=True, type=Path, help="Summary JSON output")
    parser.add_argument("--jsonl", type=Path, help="Optional row fingerprint JSONL output")
    parser.add_argument("--csv", type=Path, help="Optional row fingerprint CSV output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown summary output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--dataset", default="eval", help="Default dataset for rows without dataset metadata")
    parser.add_argument("--split", default="validation", help="Default split for rows without split metadata")
    parser.add_argument("--fail-on-duplicate-ids", action="store_true")
    parser.add_argument("--fail-on-duplicate-inputs", action="store_true")
    parser.add_argument("--fail-on-conflicting-input-answers", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings: list[Finding] = []
    fingerprints, sources = load_fingerprints(args.input, args.dataset, args.split, findings)
    report = build_report(
        fingerprints,
        sources,
        findings,
        fail_on_duplicate_ids=args.fail_on_duplicate_ids,
        fail_on_duplicate_inputs=args.fail_on_duplicate_inputs,
        fail_on_conflicting_input_answers=args.fail_on_conflicting_input_answers,
        fail_on_findings=args.fail_on_findings,
    )
    write_json(args.output, report)
    if args.jsonl:
        write_jsonl(args.jsonl, fingerprints)
    if args.csv:
        write_csv(args.csv, fingerprints)
    if args.markdown:
        write_markdown(args.markdown, report)
    if args.junit:
        write_junit(args.junit, report)
    print(args.output)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
