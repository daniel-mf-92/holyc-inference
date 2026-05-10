#!/usr/bin/env python3
"""Audit local eval JSONL for cross-dataset contamination before packing.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, then checks whether prompts or prompt+choice payloads are
reused across dataset families in a mixed eval suite.
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
from typing import Any, Callable, Iterable

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
class ContaminationFinding:
    severity: str
    kind: str
    datasets: list[str]
    splits: list[str]
    key_sha256: str
    record_ids: list[str]
    sources: list[str]
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


def key_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_key(record: dataset_pack.EvalRecord) -> str:
    return key_digest(stable_text_key(record.prompt))


def payload_key(record: dataset_pack.EvalRecord) -> str:
    return key_digest(
        {
            "choices": [stable_text_key(choice) for choice in record.choices],
            "prompt": stable_text_key(record.prompt),
        }
    )


def answer_payload_key(record: dataset_pack.EvalRecord) -> str:
    return key_digest(
        {
            "answer_index": record.answer_index,
            "choices": [stable_text_key(choice) for choice in record.choices],
            "prompt": stable_text_key(record.prompt),
        }
    )


def source_ref(loaded: LoadedRecord) -> str:
    return f"{loaded.source}:{loaded.row_number}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[ContaminationFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[ContaminationFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(
                ContaminationFinding(
                    severity="error",
                    kind="read_error",
                    datasets=[],
                    splits=[],
                    key_sha256="",
                    record_ids=[],
                    sources=[str(path)],
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
                    ContaminationFinding(
                        severity="error",
                        kind="schema_error",
                        datasets=[],
                        splits=[],
                        key_sha256="",
                        record_ids=[],
                        sources=[f"{path}:{index + 1}"],
                        detail=str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def collect_by(records: Iterable[LoadedRecord], key_fn: Callable[[dataset_pack.EvalRecord], str]) -> dict[str, list[LoadedRecord]]:
    groups: dict[str, list[LoadedRecord]] = {}
    for loaded in records:
        groups.setdefault(key_fn(loaded.record), []).append(loaded)
    return groups


def finding_for(kind: str, key: str, group: list[LoadedRecord], detail: str) -> ContaminationFinding:
    return ContaminationFinding(
        severity="error",
        kind=kind,
        datasets=sorted({loaded.record.dataset for loaded in group}),
        splits=sorted({loaded.record.split for loaded in group}),
        key_sha256=key,
        record_ids=sorted({loaded.record.record_id for loaded in group}),
        sources=sorted(source_ref(loaded) for loaded in group),
        detail=detail,
    )


def answer_histogram(group: list[LoadedRecord]) -> dict[str, int]:
    counter = collections.Counter(str(loaded.record.answer_index) for loaded in group)
    return {key: counter[key] for key in sorted(counter, key=int)}


def audit_records(
    records: list[LoadedRecord],
    check_prompt_reuse: bool,
    check_payload_reuse: bool,
    check_answer_conflicts: bool,
) -> list[ContaminationFinding]:
    findings: list[ContaminationFinding] = []

    if check_prompt_reuse:
        for key, group in sorted(collect_by(records, prompt_key).items()):
            datasets = {loaded.record.dataset for loaded in group}
            if len(datasets) > 1:
                findings.append(
                    finding_for(
                        "cross_dataset_prompt_reuse",
                        key,
                        group,
                        "normalized prompt appears in multiple datasets",
                    )
                )

    if check_payload_reuse or check_answer_conflicts:
        for key, group in sorted(collect_by(records, payload_key).items()):
            datasets = {loaded.record.dataset for loaded in group}
            if len(datasets) <= 1:
                continue
            if check_payload_reuse:
                findings.append(
                    finding_for(
                        "cross_dataset_payload_reuse",
                        key,
                        group,
                        "normalized prompt and choices appear in multiple datasets",
                    )
                )
            if check_answer_conflicts and len(answer_histogram(group)) > 1:
                findings.append(
                    finding_for(
                        "cross_dataset_answer_conflict",
                        key,
                        group,
                        f"cross-dataset payload has conflicting answer indexes: {answer_histogram(group)}",
                    )
                )

    return findings


def counts_by_dataset_split(records: list[LoadedRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for loaded in records:
        split_counts = counts.setdefault(loaded.record.dataset, {})
        split_counts[loaded.record.split] = split_counts.get(loaded.record.split, 0) + 1
    return {dataset: dict(sorted(split_counts.items())) for dataset, split_counts in sorted(counts.items())}


def unique_count(records: list[LoadedRecord], key_fn: Callable[[dataset_pack.EvalRecord], str]) -> int:
    return len({key_fn(loaded.record) for loaded in records})


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    findings: list[ContaminationFinding],
    args: argparse.Namespace,
) -> dict[str, Any]:
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "format": "hceval-contamination-audit",
        "inputs": inputs,
        "record_count": len(records),
        "dataset_count": len({loaded.record.dataset for loaded in records}),
        "split_count": len({loaded.record.split for loaded in records}),
        "unique_prompt_count": unique_count(records, prompt_key),
        "unique_payload_count": unique_count(records, payload_key),
        "unique_answer_payload_count": unique_count(records, answer_payload_key),
        "counts_by_dataset_split": counts_by_dataset_split(records),
        "checks": {
            "prompt_reuse": args.check_prompt_reuse,
            "payload_reuse": args.check_payload_reuse,
            "answer_conflicts": args.check_answer_conflicts,
        },
        "status": "fail" if error_count else "pass",
        "error_count": error_count,
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Contamination Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Datasets: {report['dataset_count']}",
        f"- Unique prompts: {report['unique_prompt_count']}",
        f"- Unique prompt+choice payloads: {report['unique_payload_count']}",
        f"- Findings: {report['finding_count']}",
        "",
        "## Dataset/Split Counts",
        "",
        "| dataset | split | records |",
        "| --- | --- | ---: |",
    ]
    for dataset, splits in report["counts_by_dataset_split"].items():
        for split, count in splits.items():
            lines.append(f"| {dataset} | {split} | {count} |")

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No cross-dataset contamination findings.")
    else:
        lines.extend(
            [
                "| severity | kind | datasets | splits | records | detail |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {datasets} | {splits} | {records} | {detail} |".format(
                    severity=finding["severity"],
                    kind=finding["kind"],
                    datasets=", ".join(finding["datasets"]),
                    splits=", ".join(finding["splits"]),
                    records=", ".join(finding["record_ids"]),
                    detail=finding["detail"],
                )
            )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "kind", "datasets", "splits", "key_sha256", "record_ids", "sources", "detail"],
        )
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(
                {
                    "severity": finding["severity"],
                    "kind": finding["kind"],
                    "datasets": ",".join(finding["datasets"]),
                    "splits": ",".join(finding["splits"]),
                    "key_sha256": finding["key_sha256"],
                    "record_ids": ",".join(finding["record_ids"]),
                    "sources": ",".join(finding["sources"]),
                    "detail": finding["detail"],
                }
            )


def write_junit(report: dict[str, Any], path: Path) -> None:
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_contamination_audit",
            "tests": str(len(failures) if failures else 1),
            "failures": str(len(failures)),
            "errors": "0",
        },
    )
    if not failures:
        ET.SubElement(suite, "testcase", {"classname": "dataset_contamination_audit", "name": "cross_dataset"})
    else:
        for index, finding in enumerate(failures, 1):
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": f"dataset_contamination_audit.{finding['kind']}",
                    "name": f"{finding['key_sha256'][:12]}:{index}",
                },
            )
            failure = ET.SubElement(case, "failure", {"type": finding["kind"], "message": finding["detail"]})
            failure.text = "\n".join(
                [
                    f"datasets={','.join(finding['datasets'])}",
                    f"splits={','.join(finding['splits'])}",
                    f"record_ids={','.join(finding['record_ids'])}",
                    f"sources={','.join(finding['sources'])}",
                    f"key_sha256={finding['key_sha256']}",
                    finding["detail"],
                ]
            )

    ET.indent(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_csv(report, args.csv)
    if args.junit:
        write_junit(report, args.junit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Local eval JSONL input; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="JSON report output")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report output")
    parser.add_argument("--csv", type=Path, help="Optional CSV findings output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML findings output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument("--skip-prompt-reuse", dest="check_prompt_reuse", action="store_false", help="Do not flag reused prompts across datasets")
    parser.add_argument("--skip-payload-reuse", dest="check_payload_reuse", action="store_false", help="Do not flag reused prompt+choice payloads across datasets")
    parser.add_argument("--skip-answer-conflicts", dest="check_answer_conflicts", action="store_false", help="Do not flag answer conflicts for reused cross-dataset payloads")
    parser.add_argument("--fail-on-contamination", action="store_true", help="Exit non-zero when error-level contamination is found")
    parser.set_defaults(check_prompt_reuse=True, check_payload_reuse=True, check_answer_conflicts=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, inputs, findings = load_records(args.input, args.default_dataset, args.default_split)
    if records:
        findings.extend(
            audit_records(
                records,
                check_prompt_reuse=args.check_prompt_reuse,
                check_payload_reuse=args.check_payload_reuse,
                check_answer_conflicts=args.check_answer_conflicts,
            )
        )
    elif not findings:
        findings.append(
            ContaminationFinding(
                severity="error",
                kind="empty_input",
                datasets=[],
                splits=[],
                key_sha256="",
                record_ids=[],
                sources=[str(path) for path in args.input],
                detail="no records loaded",
            )
        )

    report = build_report(inputs, records, findings, args)
    write_outputs(report, args)

    print(f"wrote_report={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")
    print(f"records={report['record_count']}")
    print(f"findings={report['finding_count']}")
    if args.fail_on_contamination and report["status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
