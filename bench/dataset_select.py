#!/usr/bin/env python3
"""Deterministically select offline eval dataset subsets before packing.

This host-side curation helper normalizes the same JSONL shapes accepted by
dataset_pack.py, selects a stable subset per dataset/split slice, and writes a
normalized JSONL plus manifest. It never downloads data or contacts services.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
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
class CandidateRecord:
    record: dataset_pack.EvalRecord
    source: str
    row_number: int
    payload_sha256: str
    rank_sha256: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(path: Path, row_number: int) -> str:
    return f"{path}:{row_number}"


def append_finding(findings: list[Finding], severity: str, kind: str, source: str, detail: str) -> None:
    findings.append(Finding(severity, kind, source, detail))


def record_key(record: dataset_pack.EvalRecord) -> tuple[str, str, str]:
    return (record.dataset, record.split, record.record_id)


def answer_histogram(records: Iterable[CandidateRecord]) -> dict[str, int]:
    counter = collections.Counter(str(candidate.record.answer_index) for candidate in records)
    return {key: counter[key] for key in sorted(counter)}


def load_candidates(
    inputs: Iterable[Path],
    default_dataset: str,
    default_split: str,
    seed: str,
    findings: list[Finding],
) -> tuple[list[CandidateRecord], list[dict[str, Any]]]:
    candidates: list[CandidateRecord] = []
    sources: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], CandidateRecord] = {}
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
            payload_sha256 = sha256_json(asdict(record))
            rank_sha256 = hashlib.sha256(f"{seed}\0{payload_sha256}".encode("utf-8")).hexdigest()
            candidate = CandidateRecord(record, str(path), row_number, payload_sha256, rank_sha256)
            key = record_key(record)
            if key in seen:
                prior = seen[key]
                append_finding(
                    findings,
                    "error",
                    "duplicate_record_id",
                    source_ref(path, row_number),
                    f"{'/'.join(key)} already appeared at {prior.source}:{prior.row_number}",
                )
            else:
                seen[key] = candidate
            candidates.append(candidate)
    return candidates, sources


def grouped_candidates(candidates: Iterable[CandidateRecord]) -> dict[tuple[str, str], list[CandidateRecord]]:
    groups: dict[tuple[str, str], list[CandidateRecord]] = collections.defaultdict(list)
    for candidate in candidates:
        groups[(candidate.record.dataset, candidate.record.split)].append(candidate)
    return dict(sorted(groups.items()))


def stable_order(candidates: Iterable[CandidateRecord]) -> list[CandidateRecord]:
    return sorted(candidates, key=lambda item: (item.rank_sha256, item.record.record_id, item.source, item.row_number))


def balanced_select(candidates: list[CandidateRecord], limit: int | None) -> list[CandidateRecord]:
    by_answer: dict[int, list[CandidateRecord]] = collections.defaultdict(list)
    for candidate in stable_order(candidates):
        by_answer[candidate.record.answer_index].append(candidate)
    selected: list[CandidateRecord] = []
    answers = sorted(by_answer)
    while answers and (limit is None or len(selected) < limit):
        progressed = False
        for answer in answers:
            bucket = by_answer[answer]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if limit is not None and len(selected) >= limit:
                break
        if not progressed:
            break
    return stable_order(selected)


def select_candidates(
    candidates: list[CandidateRecord],
    *,
    max_records_per_slice: int | None,
    max_records_total: int | None,
    balance_answer: bool,
) -> list[CandidateRecord]:
    selected: list[CandidateRecord] = []
    for _, records in grouped_candidates(candidates).items():
        if balance_answer:
            selected.extend(balanced_select(records, max_records_per_slice))
        else:
            ordered = stable_order(records)
            selected.extend(ordered[:max_records_per_slice] if max_records_per_slice is not None else ordered)
    selected = stable_order(selected)
    if max_records_total is not None:
        selected = selected[:max_records_total]
    return selected


def build_manifest(
    candidates: list[CandidateRecord],
    selected: list[CandidateRecord],
    sources: list[dict[str, Any]],
    findings: list[Finding],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected_by_slice = grouped_candidates(selected)
    slices: list[dict[str, Any]] = []
    for (dataset, split), records in selected_by_slice.items():
        ordered = stable_order(records)
        refs = [
            {
                "record_id": item.record.record_id,
                "source": item.source,
                "row_number": item.row_number,
                "payload_sha256": item.payload_sha256,
                "rank_sha256": item.rank_sha256,
            }
            for item in ordered
        ]
        slices.append(
            {
                "dataset": dataset,
                "split": split,
                "selected_records": len(ordered),
                "candidate_records": len(grouped_candidates(candidates).get((dataset, split), [])),
                "answer_histogram": answer_histogram(ordered),
                "slice_sha256": sha256_json({"dataset": dataset, "split": split, "records": refs}),
                "records": refs,
            }
        )

    return {
        "generated_at": iso_now(),
        "status": "fail" if any(finding.severity == "error" for finding in findings) else "pass",
        "seed": args.seed,
        "balance_answer": args.balance_answer,
        "max_records_per_slice": args.max_records_per_slice,
        "max_records_total": args.max_records_total,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "slice_count": len(slices),
        "selected_sha256": sha256_json([asdict(item.record) for item in selected]),
        "sources": sources,
        "slices": slices,
        "records": [
            {
                **asdict(item.record),
                "source": item.source,
                "row_number": item.row_number,
                "payload_sha256": item.payload_sha256,
                "rank_sha256": item.rank_sha256,
            }
            for item in selected
        ],
        "findings": [asdict(finding) for finding in findings],
    }


def write_selected_jsonl(path: Path, selected: list[CandidateRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in selected:
            handle.write(json.dumps(asdict(candidate.record), ensure_ascii=False, sort_keys=True) + "\n")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Input JSONL file; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="Normalized selected JSONL output")
    parser.add_argument("--manifest", type=Path, required=True, help="Selection manifest JSON output")
    parser.add_argument("--default-dataset", default="eval")
    parser.add_argument("--default-split", default="validation")
    parser.add_argument("--seed", default="holyc-eval-v1", help="Stable selection seed")
    parser.add_argument("--max-records-per-slice", type=int)
    parser.add_argument("--max-records-total", type=int)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--balance-answer", action="store_true", help="Round-robin answer_index buckets within each slice")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    for field in ("max_records_per_slice", "max_records_total", "min_records"):
        value = getattr(args, field)
        if value is not None and value < 0:
            append_finding(findings, "error", "invalid_argument", field, "value must be >= 0")
    return findings


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings = validate_args(args)
    candidates, sources = load_candidates(args.input, args.default_dataset, args.default_split, args.seed, findings)
    selected = select_candidates(
        candidates,
        max_records_per_slice=args.max_records_per_slice,
        max_records_total=args.max_records_total,
        balance_answer=args.balance_answer,
    )
    if len(selected) < args.min_records:
        append_finding(findings, "error", "too_few_records", "dataset_select", f"{len(selected)} selected; expected at least {args.min_records}")

    manifest = build_manifest(candidates, selected, sources, findings, args)
    write_selected_jsonl(args.output, selected)
    write_manifest(args.manifest, manifest)
    print(f"wrote_selected={args.output}")
    print(f"wrote_manifest={args.manifest}")
    return 1 if args.fail_on_findings and manifest["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
