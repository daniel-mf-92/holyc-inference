#!/usr/bin/env python3
"""Offline eval dataset packer for HolyC-loadable multiple-choice records.

The packer accepts JSONL rows from normalized records plus common HellaSwag,
ARC, and TruthfulQA multiple-choice shapes. It writes a deterministic binary
stream and a JSON manifest; it never downloads data or uses network services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MAGIC = b"HCEVAL1\0"
VERSION = 1
HEADER = struct.Struct("<8sHHII32s")
RECORD_HEADER = struct.Struct("<IIIIII")
MAX_CHOICES = 16


@dataclass(frozen=True)
class EvalRecord:
    record_id: str
    dataset: str
    split: str
    prompt: str
    choices: list[str]
    answer_index: int
    provenance: str


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def answer_from_labels(answer: Any, labels: list[str]) -> int | None:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.isdigit():
            return int(stripped)
        upper_labels = [label.upper() for label in labels]
        if stripped.upper() in upper_labels:
            return upper_labels.index(stripped.upper())
    return None


def normalize_choices(raw_choices: Any) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    choices: list[str] = []

    if isinstance(raw_choices, list):
        for index, item in enumerate(raw_choices):
            if isinstance(item, dict):
                labels.append(clean_text(item.get("label") or chr(ord("A") + index)))
                choices.append(clean_text(item.get("text") or item.get("choice") or item.get("answer")))
            else:
                labels.append(chr(ord("A") + index))
                choices.append(clean_text(item))
    elif isinstance(raw_choices, dict):
        text_values = raw_choices.get("text") or raw_choices.get("choices")
        label_values = raw_choices.get("label") or raw_choices.get("labels")
        if isinstance(text_values, list):
            raw_labels = [clean_text(label) for label in label_values] if isinstance(label_values, list) else []
            for index, item in enumerate(text_values):
                labels.append(
                    raw_labels[index] if index < len(raw_labels) and raw_labels[index] else chr(ord("A") + index)
                )
                choices.append(clean_text(item))

    return choices, labels


def normalize_truthfulqa(row: dict[str, Any]) -> tuple[list[str], int | None]:
    targets = row.get("mc1_targets") or row.get("mc2_targets")
    if not isinstance(targets, dict):
        return [], None
    raw_choices = targets.get("choices")
    raw_labels = targets.get("labels")
    if not isinstance(raw_choices, list) or not isinstance(raw_labels, list):
        return [], None
    choices = [clean_text(choice) for choice in raw_choices]
    answer_index = None
    for index, label in enumerate(raw_labels):
        if int(label) == 1:
            answer_index = index
            break
    return choices, answer_index


def normalize_prompt(row: dict[str, Any]) -> str:
    prompt = clean_text(
        row.get("prompt")
        or row.get("query")
        or row.get("question")
        or row.get("ctx")
        or row.get("input")
    )
    if prompt:
        return prompt

    ctx_parts = [clean_text(row.get("ctx_a")), clean_text(row.get("ctx_b"))]
    hellaswag_context = clean_text(" ".join(part for part in ctx_parts if part))
    if hellaswag_context:
        return hellaswag_context

    return clean_text(row.get("activity_label"))


def normalize_row(row: dict[str, Any], index: int, default_dataset: str, default_split: str) -> EvalRecord:
    dataset = clean_text(row.get("dataset") or row.get("source_dataset") or default_dataset)
    split = clean_text(row.get("split") or default_split)
    record_id = clean_text(row.get("id") or row.get("ind") or row.get("question_id") or f"{dataset}-{index + 1}")
    provenance = clean_text(row.get("provenance") or row.get("source") or dataset)

    prompt = normalize_prompt(row)

    choices, labels = normalize_choices(row.get("choices") or row.get("endings"))
    answer_index = answer_from_labels(
        row.get("answer_index")
        if "answer_index" in row
        else row.get("answer")
        if "answer" in row
        else row.get("label")
        if "label" in row
        else row.get("answerKey"),
        labels,
    )

    if not choices:
        choices, answer_index = normalize_truthfulqa(row)

    validate_record(record_id, prompt, choices, answer_index, index)
    return EvalRecord(
        record_id=record_id,
        dataset=dataset,
        split=split,
        prompt=prompt,
        choices=choices,
        answer_index=int(answer_index),
        provenance=provenance,
    )


def validate_record(
    record_id: str, prompt: str, choices: list[str], answer_index: int | None, index: int
) -> None:
    row_label = f"row {index + 1}"
    if not record_id:
        raise ValueError(f"{row_label}: missing id")
    if not prompt:
        raise ValueError(f"{row_label}: missing prompt/question text")
    if len(choices) < 2:
        raise ValueError(f"{row_label}: expected at least two choices")
    if len(choices) > MAX_CHOICES:
        raise ValueError(f"{row_label}: expected no more than {MAX_CHOICES} choices")
    if any(not choice for choice in choices):
        raise ValueError(f"{row_label}: choices must be non-empty after cleaning")
    if answer_index is None or answer_index < 0 or answer_index >= len(choices):
        raise ValueError(f"{row_label}: answer index {answer_index!r} is outside choice range")


def normalize_records(rows: Iterable[dict[str, Any]], dataset: str, split: str) -> list[EvalRecord]:
    return [normalize_row(row, index, dataset, split) for index, row in enumerate(rows)]


def encode_u32_prefixed(text: str) -> bytes:
    payload = text.encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


def record_bytes(record: EvalRecord) -> bytes:
    record_id = record.record_id.encode("utf-8")
    prompt = record.prompt.encode("utf-8")
    provenance = record.provenance.encode("utf-8")
    choice_payload = b"".join(encode_u32_prefixed(choice) for choice in record.choices)
    return (
        RECORD_HEADER.pack(
            len(record_id),
            len(prompt),
            len(record.choices),
            record.answer_index,
            len(provenance),
            0,
        )
        + record_id
        + prompt
        + provenance
        + choice_payload
    )


def record_payload_bytes(record: EvalRecord) -> int:
    return len(record_bytes(record)) - RECORD_HEADER.size


def byte_stats(records: list[EvalRecord]) -> dict[str, int]:
    prompt_lengths = [len(record.prompt.encode("utf-8")) for record in records]
    choice_lengths = [
        len(choice.encode("utf-8"))
        for record in records
        for choice in record.choices
    ]
    record_lengths = [record_payload_bytes(record) for record in records]
    return {
        "max_choice_bytes": max(choice_lengths, default=0),
        "max_prompt_bytes": max(prompt_lengths, default=0),
        "max_record_payload_bytes": max(record_lengths, default=0),
        "total_choice_bytes": sum(choice_lengths),
        "total_prompt_bytes": sum(prompt_lengths),
    }


def size_limit_findings(
    records: list[EvalRecord],
    max_prompt_bytes: int | None = None,
    max_choice_bytes: int | None = None,
    max_record_payload_bytes: int | None = None,
) -> list[str]:
    findings: list[str] = []
    for index, record in enumerate(records, 1):
        prompt_bytes = len(record.prompt.encode("utf-8"))
        if max_prompt_bytes is not None and prompt_bytes > max_prompt_bytes:
            findings.append(
                f"record {index} ({record.record_id}): prompt is {prompt_bytes} bytes, "
                f"limit is {max_prompt_bytes}"
            )
        for choice_index, choice in enumerate(record.choices, 1):
            choice_bytes = len(choice.encode("utf-8"))
            if max_choice_bytes is not None and choice_bytes > max_choice_bytes:
                findings.append(
                    f"record {index} ({record.record_id}) choice {choice_index}: "
                    f"choice is {choice_bytes} bytes, limit is {max_choice_bytes}"
                )
        payload_bytes = record_payload_bytes(record)
        if max_record_payload_bytes is not None and payload_bytes > max_record_payload_bytes:
            findings.append(
                f"record {index} ({record.record_id}): payload is {payload_bytes} bytes, "
                f"limit is {max_record_payload_bytes}"
            )
    return findings


def canonical_rows(records: list[EvalRecord]) -> bytes:
    rows = [asdict(record) for record in records]
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def metadata_bytes(dataset: str, split: str, record_count: int) -> bytes:
    metadata = {
        "dataset": dataset,
        "format": "hceval-mc",
        "record_count": record_count,
        "split": split,
        "version": VERSION,
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def record_span(record: EvalRecord, index: int, offset: int) -> dict[str, Any]:
    payload = record_bytes(record)
    choice_bytes = [len(choice.encode("utf-8")) for choice in record.choices]
    return {
        "record_index": index,
        "record_id": record.record_id,
        "offset": offset,
        "length": len(payload),
        "payload_bytes": len(payload) - RECORD_HEADER.size,
        "prompt_bytes": len(record.prompt.encode("utf-8")),
        "provenance_bytes": len(record.provenance.encode("utf-8")),
        "choice_count": len(record.choices),
        "choice_bytes": choice_bytes,
        "choice_bytes_total": sum(choice_bytes),
        "answer_index": record.answer_index,
    }


def record_spans(records: list[EvalRecord], dataset: str, split: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cursor = HEADER.size + len(metadata_bytes(dataset, split, len(records)))
    for index, record in enumerate(records):
        span = record_span(record, index, cursor)
        spans.append(span)
        cursor += int(span["length"])
    return spans


def pack_records(records: list[EvalRecord], dataset: str, split: str) -> bytes:
    packed_metadata = metadata_bytes(dataset, split, len(records))
    body = b"".join(record_bytes(record) for record in records)
    source_digest = hashlib.sha256(canonical_rows(records)).digest()
    return HEADER.pack(MAGIC, VERSION, 0, len(records), len(packed_metadata), source_digest) + packed_metadata + body


def write_outputs(records: list[EvalRecord], output: Path, manifest_path: Path, dataset: str, split: str) -> None:
    payload = pack_records(records, dataset, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)

    manifest = {
        "answer_histogram": answer_histogram(records),
        "binary_sha256": hashlib.sha256(payload).hexdigest(),
        "byte_stats": byte_stats(records),
        "dataset": dataset,
        "format": "hceval-mc",
        "magic": MAGIC.decode("ascii"),
        "output": str(output),
        "record_count": len(records),
        "record_spans": record_spans(records, dataset, split),
        "records": [asdict(record) for record in records],
        "split": split,
        "source_sha256": hashlib.sha256(canonical_rows(records)).hexdigest(),
        "version": VERSION,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def answer_histogram(records: list[EvalRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = str(record.answer_index)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output .hceval binary path")
    parser.add_argument("--manifest", type=Path, help="Output manifest JSON path")
    parser.add_argument("--dataset", default="eval", help="Default dataset name for rows without dataset")
    parser.add_argument("--split", default="validation", help="Default split for rows without split")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail if any cleaned prompt exceeds this UTF-8 byte limit")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail if any cleaned choice exceeds this UTF-8 byte limit")
    parser.add_argument(
        "--max-record-payload-bytes",
        type=int,
        help="Fail if any record payload excluding the fixed record header exceeds this byte limit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = read_jsonl(args.input)
        records = normalize_records(rows, args.dataset, args.split)
        findings = size_limit_findings(
            records,
            max_prompt_bytes=args.max_prompt_bytes,
            max_choice_bytes=args.max_choice_bytes,
            max_record_payload_bytes=args.max_record_payload_bytes,
        )
        if findings:
            raise ValueError("; ".join(findings))
        manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
        write_outputs(records, args.output, manifest, args.dataset, args.split)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_binary={args.output}")
    print(f"wrote_manifest={manifest}")
    print(f"records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
