#!/usr/bin/env python3
"""Curate local eval JSONL into normalized HolyC eval rows.

This is an offline-only helper for preparing small, deterministic evaluation
subsets from locally staged HellaSwag-, ARC-, TruthfulQA-, or normalized JSONL
files. It never downloads data. Use `dataset_pack.py` after this step, or pass
`--pack-output` to write the `.hceval` binary in the same run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import dataset_pack


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_key(record: dataset_pack.EvalRecord, seed: str) -> str:
    payload = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}\0{payload}".encode("utf-8")).hexdigest()


def answer_histogram(records: list[dataset_pack.EvalRecord]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for record in records:
        key = str(record.answer_index)
        histogram[key] = histogram.get(key, 0) + 1
    return histogram


def count_by(records: list[dataset_pack.EvalRecord], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, field))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_by_dataset_split(records: list[dataset_pack.EvalRecord]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        dataset_counts = counts.setdefault(record.dataset, {})
        dataset_counts[record.split] = dataset_counts.get(record.split, 0) + 1
    return {
        dataset: dict(sorted(split_counts.items()))
        for dataset, split_counts in sorted(counts.items())
    }


def apply_filters(
    records: list[dataset_pack.EvalRecord],
    include_dataset: set[str],
    include_split: set[str],
    require_provenance: bool,
) -> list[dataset_pack.EvalRecord]:
    filtered = []
    for record in records:
        if include_dataset and record.dataset not in include_dataset:
            continue
        if include_split and record.split not in include_split:
            continue
        if require_provenance and not record.provenance:
            continue
        filtered.append(record)
    return filtered


def reject_duplicate_ids(records: list[dataset_pack.EvalRecord]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.record_id in seen:
            duplicates.append(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"duplicate record ids after filtering: {preview}")


def cap_records_by_field(
    records: list[dataset_pack.EvalRecord],
    field: str,
    max_per_group: int | None,
    seed: str,
) -> list[dataset_pack.EvalRecord]:
    if max_per_group is None:
        return list(records)

    groups: dict[str, list[dataset_pack.EvalRecord]] = {}
    for record in records:
        groups.setdefault(str(getattr(record, field)), []).append(record)

    selected: list[dataset_pack.EvalRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: stable_sample_key(record, seed))
        selected.extend(group[:max_per_group])
    return selected


def cap_records_by_dataset_split(
    records: list[dataset_pack.EvalRecord],
    max_per_group: int | None,
    seed: str,
) -> list[dataset_pack.EvalRecord]:
    if max_per_group is None:
        return list(records)

    groups: dict[tuple[str, str], list[dataset_pack.EvalRecord]] = {}
    for record in records:
        groups.setdefault((record.dataset, record.split), []).append(record)

    selected: list[dataset_pack.EvalRecord] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: stable_sample_key(record, seed))
        selected.extend(group[:max_per_group])
    return selected


def select_records(
    records: list[dataset_pack.EvalRecord],
    max_records: int | None,
    seed: str,
    balance_answer_index: bool,
) -> list[dataset_pack.EvalRecord]:
    selected = list(records)
    if max_records is not None and max_records < len(selected):
        if balance_answer_index:
            selected = select_balanced_by_answer_index(selected, max_records, seed)
        else:
            selected = sorted(selected, key=lambda record: stable_sample_key(record, seed))[:max_records]
    return sorted(selected, key=lambda record: (record.dataset, record.split, record.record_id))


def select_balanced_by_answer_index(
    records: list[dataset_pack.EvalRecord],
    max_records: int,
    seed: str,
) -> list[dataset_pack.EvalRecord]:
    groups: dict[int, list[dataset_pack.EvalRecord]] = {}
    for record in records:
        groups.setdefault(record.answer_index, []).append(record)
    for answer_index, group in groups.items():
        groups[answer_index] = sorted(group, key=lambda record: stable_sample_key(record, seed))

    selected: list[dataset_pack.EvalRecord] = []
    while len(selected) < max_records:
        added = False
        for answer_index in sorted(groups):
            group = groups[answer_index]
            if not group:
                continue
            selected.append(group.pop(0))
            added = True
            if len(selected) == max_records:
                break
        if not added:
            break
    return selected


def write_jsonl(records: list[dataset_pack.EvalRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(
    args: argparse.Namespace,
    source_count: int,
    filtered_count: int,
    capped_count: int,
    selected: list[dataset_pack.EvalRecord],
) -> dict[str, Any]:
    normalized_digest = hashlib.sha256(dataset_pack.canonical_rows(selected)).hexdigest()
    return {
        "answer_histogram": answer_histogram(selected),
        "created_at": iso_now(),
        "dataset_counts": count_by(selected, "dataset"),
        "dataset_split_counts": count_by_dataset_split(selected),
        "filters": {
            "include_dataset": sorted(args.include_dataset),
            "include_split": sorted(args.include_split),
            "balance_answer_index": args.balance_answer_index,
            "max_records_per_dataset": args.max_records_per_dataset,
            "max_records_per_dataset_split": args.max_records_per_dataset_split,
            "max_records_per_split": args.max_records_per_split,
            "max_records": args.max_records,
            "require_provenance": args.require_provenance,
            "seed": args.seed,
        },
        "format": "hceval-curated-jsonl",
        "license": args.source_license,
        "normalized_sha256": normalized_digest,
        "output": str(args.output),
        "pack_output": str(args.pack_output) if args.pack_output else None,
        "record_count": len(selected),
        "selected_record_ids": [record.record_id for record in selected],
        "source": {
            "path": str(args.input),
            "record_count": source_count,
            "sha256": file_sha256(args.input),
        },
        "source_name": args.source_name,
        "source_url": args.source_url,
        "source_version": args.source_version,
        "split_counts": count_by(selected, "split"),
        "total_after_filters": filtered_count,
        "total_after_group_caps": capped_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local source JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Curated normalized JSONL output")
    parser.add_argument("--manifest", type=Path, required=True, help="Curation manifest JSON output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument("--include-dataset", action="append", default=[], help="Keep only this dataset; repeatable")
    parser.add_argument("--include-split", action="append", default=[], help="Keep only this split; repeatable")
    parser.add_argument("--max-records", type=int, help="Deterministically sample at most this many records")
    parser.add_argument(
        "--max-records-per-dataset",
        type=int,
        help="Deterministically keep at most this many records from each dataset before global sampling",
    )
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        help="Deterministically keep at most this many records from each split before global sampling",
    )
    parser.add_argument(
        "--max-records-per-dataset-split",
        type=int,
        help="Deterministically keep at most this many records from each dataset/split pair before global sampling",
    )
    parser.add_argument("--seed", default="holyc-eval-v1", help="Stable sampling seed")
    parser.add_argument(
        "--balance-answer-index",
        action="store_true",
        help="When sampling, round-robin by answer index to reduce label skew",
    )
    parser.add_argument("--require-provenance", action="store_true", help="Drop rows without provenance")
    parser.add_argument("--source-name", required=True, help="Original dataset or collection name")
    parser.add_argument("--source-version", default="", help="Original dataset version or release")
    parser.add_argument("--source-license", default="", help="License or usage note")
    parser.add_argument("--source-url", default="", help="Reference URL recorded only; never fetched")
    parser.add_argument("--pack-output", type=Path, help="Optional .hceval binary output")
    parser.add_argument("--pack-manifest", type=Path, help="Optional .hceval manifest output")
    parser.add_argument("--pack-dataset", default="", help="Binary metadata dataset name")
    parser.add_argument("--pack-split", default="", help="Binary metadata split name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_records is not None and args.max_records < 1:
        print("error: --max-records must be >= 1", file=sys.stderr)
        return 2
    if args.max_records_per_dataset is not None and args.max_records_per_dataset < 1:
        print("error: --max-records-per-dataset must be >= 1", file=sys.stderr)
        return 2
    if args.max_records_per_split is not None and args.max_records_per_split < 1:
        print("error: --max-records-per-split must be >= 1", file=sys.stderr)
        return 2
    if args.max_records_per_dataset_split is not None and args.max_records_per_dataset_split < 1:
        print("error: --max-records-per-dataset-split must be >= 1", file=sys.stderr)
        return 2
    if args.pack_manifest and not args.pack_output:
        print("error: --pack-manifest requires --pack-output", file=sys.stderr)
        return 2

    try:
        rows = dataset_pack.read_jsonl(args.input)
        records = dataset_pack.normalize_records(rows, args.default_dataset, args.default_split)
        filtered = apply_filters(
            records,
            set(args.include_dataset),
            set(args.include_split),
            args.require_provenance,
        )
        reject_duplicate_ids(filtered)
        capped = cap_records_by_field(filtered, "dataset", args.max_records_per_dataset, args.seed)
        capped = cap_records_by_field(capped, "split", args.max_records_per_split, args.seed)
        capped = cap_records_by_dataset_split(capped, args.max_records_per_dataset_split, args.seed)
        selected = select_records(capped, args.max_records, args.seed, args.balance_answer_index)
        if not selected:
            raise ValueError("no records selected after filters")

        write_jsonl(selected, args.output)
        manifest = build_manifest(args, len(records), len(filtered), len(capped), selected)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        pack_manifest = None
        if args.pack_output:
            pack_dataset = args.pack_dataset or args.source_name
            pack_split = args.pack_split or args.default_split
            pack_manifest = args.pack_manifest or args.pack_output.with_suffix(args.pack_output.suffix + ".manifest.json")
            dataset_pack.write_outputs(selected, args.pack_output, pack_manifest, pack_dataset, pack_split)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote_jsonl={args.output}")
    print(f"wrote_manifest={args.manifest}")
    if args.pack_output:
        print(f"wrote_binary={args.pack_output}")
        print(f"wrote_pack_manifest={pack_manifest}")
    print(f"records={len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
