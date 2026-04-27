#!/usr/bin/env python3
"""Inspect and validate HolyC-loadable offline eval datasets.

The inspector reads `.hceval` files produced by `dataset_pack.py`, validates the
binary structure, verifies embedded/source hashes when possible, and optionally
checks the companion manifest. It is host-side only and performs no network or
QEMU operations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
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


@dataclass(frozen=True)
class HCEvalDataset:
    metadata: dict[str, Any]
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
            )
        )

    if cursor != len(payload):
        raise ValueError(f"{path}: {len(payload) - cursor} trailing bytes after final record")

    return HCEvalDataset(
        metadata=metadata,
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


def validate_dataset(dataset: HCEvalDataset, manifest_path: Path | None = None) -> list[str]:
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
        "dataset": dataset.metadata.get("dataset", ""),
        "findings": findings,
        "format": dataset.metadata.get("format", ""),
        "generated_at": iso_now(),
        "input": str(path),
        "payload_sha256": dataset.payload_sha256,
        "record_count": len(dataset.records),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .hceval file")
    parser.add_argument("--manifest", type=Path, help="Optional companion manifest JSON")
    parser.add_argument("--output", type=Path, help="Optional JSON inspection report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown inspection report path")
    parser.add_argument("--no-records", action="store_true", help="Omit full record text from JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = parse_hceval(args.input)
        findings = validate_dataset(dataset, args.manifest)
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

    print(f"status={report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
