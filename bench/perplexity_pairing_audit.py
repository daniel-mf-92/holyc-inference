#!/usr/bin/env python3
"""Audit HolyC/llama.cpp perplexity inputs for apples-to-apples pairing.

This host-side tool reads local JSON, JSONL, or CSV logprob/perplexity records
only. It never launches QEMU, touches the TempleOS guest, or uses network
services.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import perplexity_compare


@dataclass(frozen=True)
class PairRecord:
    record_id: str
    dataset: str
    split: str
    holyc_token_count: int
    llama_token_count: int
    token_count_match: bool
    dataset_match: bool
    split_match: bool


@dataclass(frozen=True)
class SourceSummary:
    source: str
    engine: str
    rows: int
    unique_records: int
    duplicate_ids: int
    datasets: str
    splits: str
    token_count: int


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    record_id: str
    field: str
    holyc_value: str
    llama_value: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_records(source: Path) -> tuple[list[perplexity_compare.PerplexityRecord], list[Finding]]:
    findings: list[Finding] = []
    records: list[perplexity_compare.PerplexityRecord] = []
    try:
        rows = perplexity_compare.read_rows(source)
        for index, row in enumerate(rows):
            try:
                records.append(perplexity_compare.normalize_record(row, source, index))
            except ValueError as exc:
                findings.append(
                    Finding(
                        "error",
                        "invalid_record",
                        f"row-{index + 1}",
                        "record",
                        "",
                        "",
                        str(exc),
                    )
                )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "load_error", "", "source", str(source), "", str(exc)))
    return records, findings


def source_summary(source: Path, engine: str, records: list[perplexity_compare.PerplexityRecord]) -> SourceSummary:
    seen: set[str] = set()
    duplicate_ids = 0
    for record in records:
        if record.record_id in seen:
            duplicate_ids += 1
        seen.add(record.record_id)
    return SourceSummary(
        source=str(source),
        engine=engine,
        rows=len(records),
        unique_records=len(seen),
        duplicate_ids=duplicate_ids,
        datasets=",".join(sorted({record.dataset for record in records if record.dataset})),
        splits=",".join(sorted({record.split for record in records if record.split})),
        token_count=sum(record.token_count for record in records),
    )


def index_records(
    records: list[perplexity_compare.PerplexityRecord], engine: str, findings: list[Finding]
) -> dict[str, perplexity_compare.PerplexityRecord]:
    indexed: dict[str, perplexity_compare.PerplexityRecord] = {}
    for record in records:
        if record.record_id in indexed:
            findings.append(
                Finding(
                    "error",
                    "duplicate_record_id",
                    record.record_id,
                    engine,
                    record.record_id if engine == "holyc" else "",
                    record.record_id if engine == "llama" else "",
                    f"{engine} input contains duplicate record id",
                )
            )
            continue
        indexed[record.record_id] = record
    return indexed


def text(value: object) -> str:
    return "" if value is None else str(value)


def audit_pairing(
    holyc_records: list[perplexity_compare.PerplexityRecord],
    llama_records: list[perplexity_compare.PerplexityRecord],
    *,
    min_pairs: int,
) -> tuple[list[PairRecord], list[Finding]]:
    findings: list[Finding] = []
    holyc_by_id = index_records(holyc_records, "holyc", findings)
    llama_by_id = index_records(llama_records, "llama", findings)
    pairs: list[PairRecord] = []

    for record_id in sorted(set(holyc_by_id) | set(llama_by_id)):
        holyc = holyc_by_id.get(record_id)
        llama = llama_by_id.get(record_id)
        if holyc is None:
            findings.append(Finding("error", "missing_holyc_record", record_id, "record_id", "", record_id, "llama record has no HolyC pair"))
            continue
        if llama is None:
            findings.append(Finding("error", "missing_llama_record", record_id, "record_id", record_id, "", "HolyC record has no llama.cpp pair"))
            continue

        pair = PairRecord(
            record_id=record_id,
            dataset=holyc.dataset or llama.dataset,
            split=holyc.split or llama.split,
            holyc_token_count=holyc.token_count,
            llama_token_count=llama.token_count,
            token_count_match=holyc.token_count == llama.token_count,
            dataset_match=holyc.dataset == llama.dataset,
            split_match=holyc.split == llama.split,
        )
        pairs.append(pair)
        if not pair.token_count_match:
            findings.append(
                Finding(
                    "error",
                    "token_count_mismatch",
                    record_id,
                    "token_count",
                    text(holyc.token_count),
                    text(llama.token_count),
                    "HolyC and llama.cpp token counts must match for perplexity pairing",
                )
            )
        if not pair.dataset_match:
            findings.append(
                Finding("error", "dataset_mismatch", record_id, "dataset", holyc.dataset, llama.dataset, "dataset metadata must match")
            )
        if not pair.split_match:
            findings.append(Finding("error", "split_mismatch", record_id, "split", holyc.split, llama.split, "split metadata must match"))

    if len(pairs) < min_pairs:
        findings.append(
            Finding("error", "min_pairs", "", "pairs", text(len(pairs)), text(min_pairs), "paired record count is below the required minimum")
        )
    return pairs, findings


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Perplexity Pairing Audit",
        "",
        f"- Status: {payload['status']}",
        f"- HolyC rows: {summary['holyc_rows']}",
        f"- llama.cpp rows: {summary['llama_rows']}",
        f"- Paired rows: {summary['paired_rows']}",
        f"- Findings: {summary['findings']}",
        "",
    ]
    findings = payload["findings"]
    assert isinstance(findings, list)
    if findings:
        lines.append("## Findings")
        for finding in findings[:20]:
            assert isinstance(finding, dict)
            lines.append(f"- {finding['kind']}: {finding['record_id'] or '-'} {finding['detail']}")
    else:
        lines.append("No perplexity pairing findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_perplexity_pairing_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"name": "perplexity_pairing"})
    for index, finding in enumerate(findings, 1):
        case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}_{index}"})
        failure = ET.SubElement(case, "failure", {"message": finding.detail, "type": finding.kind})
        failure.text = json.dumps(asdict(finding), sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path, help="HolyC logprob/perplexity JSON, JSONL, or CSV")
    parser.add_argument("--llama", required=True, type=Path, help="llama.cpp logprob/perplexity JSON, JSONL, or CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="perplexity_pairing_audit_latest")
    parser.add_argument("--min-pairs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    holyc_records, load_findings = load_records(args.holyc)
    llama_records, llama_load_findings = load_records(args.llama)
    load_findings.extend(llama_load_findings)
    pairs, pair_findings = audit_pairing(holyc_records, llama_records, min_pairs=args.min_pairs)
    findings = load_findings + pair_findings
    sources = [
        source_summary(args.holyc, "holyc", holyc_records),
        source_summary(args.llama, "llama", llama_records),
    ]
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "holyc_rows": len(holyc_records),
            "llama_rows": len(llama_records),
            "paired_rows": len(pairs),
            "token_count_mismatches": sum(1 for finding in findings if finding.kind == "token_count_mismatch"),
            "metadata_mismatches": sum(1 for finding in findings if finding.kind in {"dataset_mismatch", "split_mismatch"}),
            "findings": len(findings),
        },
        "sources": [asdict(source) for source in sources],
        "pairs": [asdict(pair) for pair in pairs],
        "findings": [asdict(finding) for finding in findings],
    }

    stem = args.output_dir / args.output_stem
    (stem.with_suffix(".json")).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), pairs)
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(stem.with_suffix(".md"), payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
