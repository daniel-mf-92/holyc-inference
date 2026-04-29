#!/usr/bin/env python3
"""Export HolyC-loadable eval datasets back to normalized JSONL.

The exporter reads `.hceval` files produced by `dataset_pack.py` and writes
offline JSONL rows suitable for `eval_compare.py` gold inputs and
`eval_input_audit.py` hash-parity checks. It performs no network or QEMU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack
import hceval_inspect


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choices_sha256(choices: list[str]) -> str:
    encoded = json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_sha256(prompt: str, choices: list[str]) -> str:
    encoded = json.dumps(
        {
            "choices_sha256": choices_sha256(choices),
            "prompt_sha256": text_sha256(prompt),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_metadata_by_id(manifest_path: Path | None) -> dict[str, dict[str, str]]:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path}: manifest records are missing")
    metadata: dict[str, dict[str, str]] = {}
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{manifest_path}: record {index} is not an object")
        record_id = str(row.get("record_id") or row.get("id") or "").strip()
        if not record_id:
            raise ValueError(f"{manifest_path}: record {index} has no record_id")
        metadata[record_id] = {
            "dataset": dataset_pack.clean_text(row.get("dataset")),
            "split": dataset_pack.clean_text(row.get("split")),
            "provenance": dataset_pack.clean_text(row.get("provenance")),
        }
    return metadata


def export_row(
    record: hceval_inspect.InspectRecord,
    default_dataset: str,
    default_split: str,
    metadata: dict[str, str],
    include_hashes: bool,
) -> dict[str, Any]:
    dataset = metadata.get("dataset") or default_dataset
    split = metadata.get("split") or default_split
    provenance = metadata.get("provenance") or record.provenance
    row: dict[str, Any] = {
        "answer_index": record.answer_index,
        "choices": record.choices,
        "dataset": dataset,
        "id": record.record_id,
        "prompt": record.prompt,
        "provenance": provenance,
        "record_id": record.record_id,
        "split": split,
    }
    if include_hashes:
        row["prompt_sha256"] = text_sha256(record.prompt)
        row["choices_sha256"] = choices_sha256(record.choices)
        row["input_sha256"] = input_sha256(record.prompt, record.choices)
    return row


def export_records(
    dataset: hceval_inspect.HCEvalDataset,
    include_hashes: bool,
    manifest_metadata: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    manifest_metadata = manifest_metadata or {}
    return [
        export_row(record, dataset_name, split, manifest_metadata.get(record.record_id, {}), include_hashes)
        for record in dataset.records
    ]


def canonical_export_digest(rows: list[dict[str, Any]]) -> str:
    records = [
        dataset_pack.EvalRecord(
            record_id=str(row["record_id"]),
            dataset=str(row["dataset"]),
            split=str(row["split"]),
            prompt=str(row["prompt"]),
            choices=[str(choice) for choice in row["choices"]],
            answer_index=int(row["answer_index"]),
            provenance=str(row["provenance"]),
        )
        for row in rows
    ]
    return hashlib.sha256(dataset_pack.canonical_rows(records)).hexdigest()


def build_manifest(
    input_path: Path,
    output_path: Path,
    dataset: hceval_inspect.HCEvalDataset,
    rows: list[dict[str, Any]],
    include_hashes: bool,
    pack_manifest_path: Path | None,
) -> dict[str, Any]:
    written_source_sha256 = canonical_export_digest(rows)
    return {
        "binary_sha256": dataset.payload_sha256,
        "dataset": dataset.metadata.get("dataset", ""),
        "format": "hceval-export-jsonl",
        "include_hashes": include_hashes,
        "input": str(input_path),
        "output": str(output_path),
        "pack_manifest": str(pack_manifest_path) if pack_manifest_path is not None else "",
        "record_count": len(rows),
        "source_sha256": dataset.source_digest,
        "source_sha256_matches": written_source_sha256 == dataset.source_digest,
        "split": dataset.metadata.get("split", ""),
        "version": 1,
        "written_source_sha256": written_source_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .hceval file")
    parser.add_argument("--output", type=Path, required=True, help="Output normalized JSONL path")
    parser.add_argument("--manifest", type=Path, help="Optional export manifest JSON path")
    parser.add_argument(
        "--pack-manifest",
        type=Path,
        help="Optional input pack manifest used to restore per-record dataset/split metadata",
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Omit prompt_sha256, choices_sha256, and input_sha256 fields",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = hceval_inspect.parse_hceval(args.input)
        if args.pack_manifest:
            findings = hceval_inspect.validate_dataset(dataset, args.pack_manifest)
            if findings:
                raise ValueError("; ".join(findings))
        rows = export_records(
            dataset,
            include_hashes=not args.no_hashes,
            manifest_metadata=manifest_metadata_by_id(args.pack_manifest),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    print(f"wrote_jsonl={args.output}")
    print(f"records={len(rows)}")

    if args.manifest:
        manifest = build_manifest(
            args.input,
            args.output,
            dataset,
            rows,
            include_hashes=not args.no_hashes,
            pack_manifest_path=args.pack_manifest,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_manifest={args.manifest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
