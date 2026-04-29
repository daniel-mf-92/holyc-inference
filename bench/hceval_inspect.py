#!/usr/bin/env python3
"""Inspect and validate HolyC-loadable offline eval datasets.

The inspector reads `.hceval` files produced by `dataset_pack.py`, validates the
binary structure, verifies embedded/source hashes when possible, and optionally
checks the companion manifest. It is host-side only and performs no network or
QEMU operations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack


@dataclass(frozen=True)
class InspectRecord:
    record_id: str
    prompt: str
    choices: list[str]
    answer_index: int
    provenance: str
    flags: int
    offset: int
    length: int


@dataclass(frozen=True)
class HCEvalDataset:
    metadata: dict[str, Any]
    metadata_bytes: int
    records: list[InspectRecord]
    source_digest: str
    payload_sha256: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def require_span(payload: bytes, cursor: int, length: int, label: str) -> None:
    if length < 0:
        raise ValueError(f"{label}: negative length {length}")
    if cursor + length > len(payload):
        raise ValueError(f"{label}: truncated at offset {cursor}, need {length} bytes")


def read_u32_prefixed(payload: bytes, cursor: int, label: str) -> tuple[str, int]:
    require_span(payload, cursor, 4, f"{label} length")
    (length,) = struct.unpack_from("<I", payload, cursor)
    cursor += 4
    require_span(payload, cursor, length, label)
    try:
        text = payload[cursor : cursor + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: invalid UTF-8: {exc}") from exc
    return text, cursor + length


def read_text(payload: bytes, cursor: int, length: int, label: str) -> tuple[str, int]:
    require_span(payload, cursor, length, label)
    try:
        text = payload[cursor : cursor + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: invalid UTF-8: {exc}") from exc
    return text, cursor + length


def parse_hceval(path: Path) -> HCEvalDataset:
    payload = path.read_bytes()
    require_span(payload, 0, dataset_pack.HEADER.size, "header")
    magic, version, flags, record_count, metadata_len, source_digest = dataset_pack.HEADER.unpack_from(payload, 0)
    if magic != dataset_pack.MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version != dataset_pack.VERSION:
        raise ValueError(f"{path}: unsupported version {version}")
    if flags != 0:
        raise ValueError(f"{path}: unsupported header flags {flags}")

    cursor = dataset_pack.HEADER.size
    metadata_text, cursor = read_text(payload, cursor, metadata_len, "metadata")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: metadata is not JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata JSON must be an object")

    records: list[InspectRecord] = []
    for record_index in range(record_count):
        label = f"record {record_index + 1}"
        record_offset = cursor
        require_span(payload, cursor, dataset_pack.RECORD_HEADER.size, f"{label} header")
        id_len, prompt_len, choice_count, answer_index, provenance_len, record_flags = (
            dataset_pack.RECORD_HEADER.unpack_from(payload, cursor)
        )
        cursor += dataset_pack.RECORD_HEADER.size

        record_id, cursor = read_text(payload, cursor, id_len, f"{label} id")
        prompt, cursor = read_text(payload, cursor, prompt_len, f"{label} prompt")
        provenance, cursor = read_text(payload, cursor, provenance_len, f"{label} provenance")
        choices = []
        for choice_index in range(choice_count):
            choice, cursor = read_u32_prefixed(payload, cursor, f"{label} choice {choice_index + 1}")
            choices.append(choice)

        records.append(
            InspectRecord(
                record_id=record_id,
                prompt=prompt,
                choices=choices,
                answer_index=answer_index,
                provenance=provenance,
                flags=record_flags,
                offset=record_offset,
                length=cursor - record_offset,
            )
        )

    if cursor != len(payload):
        raise ValueError(f"{path}: {len(payload) - cursor} trailing bytes after final record")

    return HCEvalDataset(
        metadata=metadata,
        metadata_bytes=metadata_len,
        records=records,
        source_digest=source_digest.hex(),
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def as_pack_records(dataset: HCEvalDataset) -> list[dataset_pack.EvalRecord]:
    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    return [
        dataset_pack.EvalRecord(
            record_id=record.record_id,
            dataset=dataset_name,
            split=split,
            prompt=record.prompt,
            choices=record.choices,
            answer_index=record.answer_index,
            provenance=record.provenance,
        )
        for record in dataset.records
    ]


def answer_histogram(records: list[InspectRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = str(record.answer_index)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def record_spans(records: list[InspectRecord]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        choice_bytes = [len(choice.encode("utf-8")) for choice in record.choices]
        spans.append(
            {
                "record_index": index,
                "record_id": record.record_id,
                "offset": record.offset,
                "length": record.length,
                "payload_bytes": record.length - dataset_pack.RECORD_HEADER.size,
                "prompt_bytes": len(record.prompt.encode("utf-8")),
                "provenance_bytes": len(record.provenance.encode("utf-8")),
                "choice_count": len(record.choices),
                "choice_bytes": choice_bytes,
                "choice_bytes_total": sum(choice_bytes),
                "answer_index": record.answer_index,
            }
        )
    return spans


def record_fingerprints(dataset: HCEvalDataset) -> list[dict[str, Any]]:
    return dataset_pack.record_fingerprints(as_pack_records(dataset))


def byte_stats(records: list[InspectRecord]) -> dict[str, int]:
    pack_records = [
        dataset_pack.EvalRecord(
            record_id=record.record_id,
            dataset="",
            split="",
            prompt=record.prompt,
            choices=record.choices,
            answer_index=record.answer_index,
            provenance=record.provenance,
        )
        for record in records
    ]
    return dataset_pack.byte_stats(pack_records)


def choice_count_histogram(records: list[InspectRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = str(len(record.choices))
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: int(item[0])))


def choice_count_stats(records: list[InspectRecord]) -> dict[str, float | int]:
    choice_counts = [len(record.choices) for record in records]
    total_choices = sum(choice_counts)
    return {
        "avg_choices_per_record": total_choices / len(choice_counts) if choice_counts else 0.0,
        "max_choices_per_record": max(choice_counts, default=0),
        "min_choices_per_record": min(choice_counts, default=0),
        "total_choices": total_choices,
    }


def binary_layout(dataset: HCEvalDataset) -> dict[str, int]:
    body_bytes = sum(record.length for record in dataset.records)
    choice_count = sum(len(record.choices) for record in dataset.records)
    return {
        "binary_bytes": dataset_pack.HEADER.size + dataset.metadata_bytes + body_bytes,
        "body_bytes": body_bytes,
        "choice_length_prefix_bytes": choice_count * 4,
        "fixed_header_bytes": dataset_pack.HEADER.size,
        "metadata_bytes": dataset.metadata_bytes,
        "record_count": len(dataset.records),
        "record_header_bytes": len(dataset.records) * dataset_pack.RECORD_HEADER.size,
        "record_payload_bytes": sum(record.length - dataset_pack.RECORD_HEADER.size for record in dataset.records),
    }


def validate_dataset(
    dataset: HCEvalDataset,
    manifest_path: Path | None = None,
    max_prompt_bytes: int | None = None,
    max_choice_bytes: int | None = None,
    max_record_payload_bytes: int | None = None,
) -> list[str]:
    findings: list[str] = []
    metadata = dataset.metadata
    records = dataset.records

    if metadata.get("format") != "hceval-mc":
        findings.append(f"metadata format is {metadata.get('format')!r}, expected 'hceval-mc'")
    if metadata.get("version") != dataset_pack.VERSION:
        findings.append(f"metadata version is {metadata.get('version')!r}, expected {dataset_pack.VERSION}")
    if metadata.get("record_count") != len(records):
        findings.append(f"metadata record_count {metadata.get('record_count')!r} != parsed {len(records)}")

    seen_ids: set[str] = set()
    for index, record in enumerate(records, 1):
        if not record.record_id:
            findings.append(f"record {index}: empty id")
        if record.record_id in seen_ids:
            findings.append(f"record {index}: duplicate id {record.record_id!r}")
        seen_ids.add(record.record_id)
        if not record.prompt:
            findings.append(f"record {index}: empty prompt")
        if len(record.choices) < 2:
            findings.append(f"record {index}: expected at least two choices")
        if len(record.choices) > dataset_pack.MAX_CHOICES:
            findings.append(f"record {index}: expected no more than {dataset_pack.MAX_CHOICES} choices")
        if any(not choice for choice in record.choices):
            findings.append(f"record {index}: empty choice")
        if record.answer_index >= len(record.choices):
            findings.append(f"record {index}: answer index {record.answer_index} is outside choice range")
        if record.flags != 0:
            findings.append(f"record {index}: unsupported flags {record.flags}")

    reconstructed_digest = hashlib.sha256(dataset_pack.canonical_rows(as_pack_records(dataset))).hexdigest()
    source_verified: bool | None = reconstructed_digest == dataset.source_digest
    findings.extend(
        dataset_pack.size_limit_findings(
            as_pack_records(dataset),
            max_prompt_bytes=max_prompt_bytes,
            max_choice_bytes=max_choice_bytes,
            max_record_payload_bytes=max_record_payload_bytes,
        )
    )

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("binary_sha256") != dataset.payload_sha256:
            findings.append("manifest binary_sha256 does not match input")
        if manifest.get("source_sha256") != dataset.source_digest:
            findings.append("manifest source_sha256 does not match header digest")
        if manifest.get("record_count") != len(records):
            findings.append("manifest record_count does not match parsed records")
        if manifest.get("dataset") != metadata.get("dataset"):
            findings.append("manifest dataset does not match metadata")
        if manifest.get("split") != metadata.get("split"):
            findings.append("manifest split does not match metadata")
        if "byte_stats" in manifest and manifest.get("byte_stats") != byte_stats(records):
            findings.append("manifest byte_stats does not match parsed records")
        if (
            "choice_count_histogram" in manifest
            and manifest.get("choice_count_histogram") != choice_count_histogram(records)
        ):
            findings.append("manifest choice_count_histogram does not match parsed records")
        if "choice_count_stats" in manifest and manifest.get("choice_count_stats") != choice_count_stats(records):
            findings.append("manifest choice_count_stats does not match parsed records")
        if "binary_layout" in manifest and manifest.get("binary_layout") != binary_layout(dataset):
            findings.append("manifest binary_layout does not match parsed binary")
        if "record_spans" in manifest and manifest.get("record_spans") != record_spans(records):
            findings.append("manifest record_spans does not match parsed binary")
        if "record_fingerprints" in manifest and manifest.get("record_fingerprints") != record_fingerprints(dataset):
            findings.append("manifest record_fingerprints does not match parsed binary")
        manifest_records = manifest.get("records")
        if isinstance(manifest_records, list):
            source_verified = hashlib.sha256(
                json.dumps(
                    manifest_records,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest() == dataset.source_digest

    if source_verified is False:
        findings.append("header source digest does not match reconstructed records")

    return findings


def build_report(path: Path, dataset: HCEvalDataset, findings: list[str]) -> dict[str, Any]:
    return {
        "answer_histogram": answer_histogram(dataset.records),
        "binary_layout": binary_layout(dataset),
        "byte_stats": byte_stats(dataset.records),
        "choice_count_histogram": choice_count_histogram(dataset.records),
        "choice_count_stats": choice_count_stats(dataset.records),
        "dataset": dataset.metadata.get("dataset", ""),
        "findings": findings,
        "format": dataset.metadata.get("format", ""),
        "generated_at": iso_now(),
        "input": str(path),
        "payload_sha256": dataset.payload_sha256,
        "record_count": len(dataset.records),
        "record_fingerprints": record_fingerprints(dataset),
        "record_spans": record_spans(dataset.records),
        "records": [asdict(record) for record in dataset.records],
        "source_sha256": dataset.source_digest,
        "split": dataset.metadata.get("split", ""),
        "status": "pass" if not findings else "fail",
        "version": dataset.metadata.get("version"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Dataset Inspection",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Dataset: {report['dataset'] or '-'}",
        f"Split: {report['split'] or '-'}",
        f"Records: {report['record_count']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(f"- {finding}" for finding in report["findings"])
    else:
        lines.append("No findings.")

    lines.extend(
        [
            "",
            "## Byte Stats",
            "",
            f"- Max prompt bytes: {report['byte_stats']['max_prompt_bytes']}",
            f"- Max choice bytes: {report['byte_stats']['max_choice_bytes']}",
            f"- Max record payload bytes: {report['byte_stats']['max_record_payload_bytes']}",
            f"- Total prompt bytes: {report['byte_stats']['total_prompt_bytes']}",
            f"- Total choice bytes: {report['byte_stats']['total_choice_bytes']}",
            "",
            "## Choice Counts",
            "",
            f"- Histogram: `{json.dumps(report['choice_count_histogram'], sort_keys=True)}`",
            f"- Min choices per record: {report['choice_count_stats']['min_choices_per_record']}",
            f"- Max choices per record: {report['choice_count_stats']['max_choices_per_record']}",
            f"- Average choices per record: {report['choice_count_stats']['avg_choices_per_record']:.2f}",
            f"- Total choices: {report['choice_count_stats']['total_choices']}",
            "",
            "## Binary Layout",
            "",
            f"- Fixed header bytes: {report['binary_layout']['fixed_header_bytes']}",
            f"- Metadata bytes: {report['binary_layout']['metadata_bytes']}",
            f"- Record header bytes: {report['binary_layout']['record_header_bytes']}",
            f"- Record payload bytes: {report['binary_layout']['record_payload_bytes']}",
            f"- Choice length-prefix bytes: {report['binary_layout']['choice_length_prefix_bytes']}",
            f"- Body bytes: {report['binary_layout']['body_bytes']}",
            f"- Binary bytes: {report['binary_layout']['binary_bytes']}",
        ]
    )

    lines.extend(["", "## Records", ""])
    if report["records"]:
        lines.append("| ID | Choices | Answer | Provenance |")
        lines.append("| --- | ---: | ---: | --- |")
        for record in report["records"]:
            lines.append(
                f"| {record['record_id']} | {len(record['choices'])} | "
                f"{record['answer_index']} | {record['provenance']} |"
            )
    else:
        lines.append("No records.")
    return "\n".join(lines) + "\n"


def junit_report(report: dict[str, Any]) -> str:
    findings = [str(finding) for finding in report.get("findings", [])]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_inspect",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "hceval_inspect",
            "name": str(report.get("input", "dataset")),
        },
    )
    if findings:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "hceval_inspection_failure",
                "message": f"{len(findings)} dataset inspection finding(s)",
            },
        )
        failure.text = "\n".join(findings)

    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "record_index",
        "record_id",
        "offset",
        "length",
        "payload_bytes",
        "prompt_bytes",
        "provenance_bytes",
        "choice_count",
        "choice_bytes_total",
        "answer_index",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for span in report.get("record_spans", []):
            writer.writerow({field: span.get(field, "") for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .hceval file")
    parser.add_argument("--manifest", type=Path, help="Optional companion manifest JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON inspection report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown inspection report path")
    parser.add_argument("--csv", type=Path, help="Optional CSV record-span report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML inspection report path")
    parser.add_argument("--no-records", action="store_true", help="Omit full record text from JSON output")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail if any prompt exceeds this UTF-8 byte limit")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail if any choice exceeds this UTF-8 byte limit")
    parser.add_argument(
        "--max-record-payload-bytes",
        type=int,
        help="Fail if any record payload excluding the fixed record header exceeds this byte limit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = parse_hceval(args.input)
        findings = validate_dataset(
            dataset,
            args.manifest,
            max_prompt_bytes=args.max_prompt_bytes,
            max_choice_bytes=args.max_choice_bytes,
            max_record_payload_bytes=args.max_record_payload_bytes,
        )
        report = build_report(args.input, dataset, findings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.no_records:
        report = dict(report)
        report["records"] = []

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote_markdown={args.markdown}")

    if args.csv:
        write_csv(report, args.csv)
        print(f"wrote_csv={args.csv}")

    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        args.junit.write_text(junit_report(report), encoding="utf-8")
        print(f"wrote_junit={args.junit}")

    print(f"status={report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
