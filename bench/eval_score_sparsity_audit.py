#!/usr/bin/env python3
"""Audit scored eval predictions for sparse or degenerate score vectors.

This host-side audit reads HolyC and llama.cpp scored-prediction JSONL files
only. It never launches QEMU, touches the TempleOS guest, or uses networking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordSparsity:
    source: str
    engine: str
    record_id: str
    score_count: int
    finite_score_count: int
    nonzero_score_count: int
    zero_score_count: int
    zero_score_pct: float
    unique_score_count: int
    finite: bool


@dataclass(frozen=True)
class EngineSummary:
    source: str
    engine: str
    record_count: int
    scored_record_count: int
    invalid_record_count: int
    total_score_count: int
    total_nonzero_score_count: int
    total_zero_score_count: int
    zero_score_pct: float
    min_nonzero_scores_per_record: int
    mean_nonzero_scores_per_record: float
    min_unique_scores_per_record: int
    mean_unique_scores_per_record: float


@dataclass(frozen=True)
class Finding:
    severity: str
    engine: str
    record_id: str
    metric: str
    value: float | int | str
    limit: float | int | str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scores(raw: Any) -> list[float] | None:
    if not isinstance(raw, list):
        return None
    scores: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        scores.append(float(value))
    return scores


def stable_score(value: float) -> str:
    return f"{value:.12g}"


def load_records(
    path: Path,
    *,
    engine: str,
    zero_epsilon: float,
) -> tuple[list[RecordSparsity], list[Finding]]:
    records: list[RecordSparsity] = []
    findings: list[Finding] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                findings.append(
                    Finding("error", engine, f"line:{line_no}", "json", "invalid", "valid", f"{path}:{line_no}: invalid JSON: {exc}")
                )
                continue
            if not isinstance(payload, dict):
                findings.append(
                    Finding("error", engine, f"line:{line_no}", "record", type(payload).__name__, "object", f"{path}:{line_no}: row must be an object")
                )
                continue
            record_id = str(payload.get("id") or payload.get("record_id") or f"line:{line_no}")
            if record_id in seen:
                findings.append(Finding("error", engine, record_id, "duplicate_id", 1, 0, f"{engine} duplicate scored prediction id {record_id}"))
            seen.add(record_id)
            scores = parse_scores(payload.get("scores"))
            if scores is None:
                findings.append(
                    Finding("error", engine, record_id, "scores", "missing_or_invalid", "numeric_list", f"{engine} {record_id} scores must be a numeric list")
                )
                continue
            finite_scores = [score for score in scores if math.isfinite(score)]
            finite = len(finite_scores) == len(scores)
            nonzero_count = sum(1 for score in finite_scores if abs(score) > zero_epsilon)
            zero_count = len(finite_scores) - nonzero_count
            records.append(
                RecordSparsity(
                    source=str(path),
                    engine=engine,
                    record_id=record_id,
                    score_count=len(scores),
                    finite_score_count=len(finite_scores),
                    nonzero_score_count=nonzero_count,
                    zero_score_count=zero_count,
                    zero_score_pct=zero_count / len(finite_scores) * 100.0 if finite_scores else 0.0,
                    unique_score_count=len({stable_score(score) for score in finite_scores}),
                    finite=finite,
                )
            )
            if not finite:
                findings.append(Finding("error", engine, record_id, "finite", "false", "true", f"{engine} {record_id} has non-finite scores"))
    return records, findings


def summarize(source: Path, engine: str, records: list[RecordSparsity]) -> EngineSummary:
    valid = [record for record in records if record.finite and record.score_count > 0]
    total_scores = sum(record.finite_score_count for record in valid)
    total_nonzero = sum(record.nonzero_score_count for record in valid)
    total_zero = sum(record.zero_score_count for record in valid)
    nonzero_counts = [record.nonzero_score_count for record in valid]
    unique_counts = [record.unique_score_count for record in valid]
    return EngineSummary(
        source=str(source),
        engine=engine,
        record_count=len(records),
        scored_record_count=len(valid),
        invalid_record_count=len(records) - len(valid),
        total_score_count=total_scores,
        total_nonzero_score_count=total_nonzero,
        total_zero_score_count=total_zero,
        zero_score_pct=total_zero / total_scores * 100.0 if total_scores else 0.0,
        min_nonzero_scores_per_record=min(nonzero_counts) if nonzero_counts else 0,
        mean_nonzero_scores_per_record=sum(nonzero_counts) / len(nonzero_counts) if nonzero_counts else 0.0,
        min_unique_scores_per_record=min(unique_counts) if unique_counts else 0,
        mean_unique_scores_per_record=sum(unique_counts) / len(unique_counts) if unique_counts else 0.0,
    )


def audit(
    holyc_path: Path,
    llama_path: Path,
    *,
    zero_epsilon: float,
    min_records: int,
    min_nonzero_scores_per_record: int,
    min_unique_scores_per_record: int,
    max_zero_score_pct: float,
) -> tuple[list[RecordSparsity], list[EngineSummary], list[Finding]]:
    holyc_records, findings = load_records(holyc_path, engine="holyc", zero_epsilon=zero_epsilon)
    llama_records, llama_findings = load_records(llama_path, engine="llama", zero_epsilon=zero_epsilon)
    findings.extend(llama_findings)
    summaries = [summarize(holyc_path, "holyc", holyc_records), summarize(llama_path, "llama", llama_records)]
    all_records = holyc_records + llama_records

    for summary in summaries:
        if summary.scored_record_count < min_records:
            findings.append(
                Finding("error", summary.engine, "", "scored_record_count", summary.scored_record_count, min_records, f"{summary.engine} has too few scored records")
            )
        if summary.zero_score_pct > max_zero_score_pct:
            findings.append(
                Finding("error", summary.engine, "", "zero_score_pct", summary.zero_score_pct, max_zero_score_pct, f"{summary.engine} zero-score percentage is above limit")
            )
        if summary.min_nonzero_scores_per_record < min_nonzero_scores_per_record:
            findings.append(
                Finding(
                    "error",
                    summary.engine,
                    "",
                    "min_nonzero_scores_per_record",
                    summary.min_nonzero_scores_per_record,
                    min_nonzero_scores_per_record,
                    f"{summary.engine} minimum nonzero scores per record is below limit",
                )
            )
        if summary.min_unique_scores_per_record < min_unique_scores_per_record:
            findings.append(
                Finding(
                    "error",
                    summary.engine,
                    "",
                    "min_unique_scores_per_record",
                    summary.min_unique_scores_per_record,
                    min_unique_scores_per_record,
                    f"{summary.engine} minimum unique scores per record is below limit",
                )
            )

    for record in all_records:
        if record.finite and record.score_count > 0 and record.nonzero_score_count < min_nonzero_scores_per_record:
            findings.append(
                Finding(
                    "error",
                    record.engine,
                    record.record_id,
                    "nonzero_score_count",
                    record.nonzero_score_count,
                    min_nonzero_scores_per_record,
                    f"{record.engine} {record.record_id} has too few nonzero scores",
                )
            )
        if record.finite and record.score_count > 0 and record.unique_score_count < min_unique_scores_per_record:
            findings.append(
                Finding(
                    "error",
                    record.engine,
                    record.record_id,
                    "unique_score_count",
                    record.unique_score_count,
                    min_unique_scores_per_record,
                    f"{record.engine} {record.record_id} has too few unique scores",
                )
            )
    return all_records, summaries, findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[RecordSparsity]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()) if records else ["source", "engine", "record_id"])
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_summary_csv(path: Path, summaries: list[EngineSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summaries[0]).keys()) if summaries else ["source", "engine"])
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(findings[0]).keys()) if findings else ["severity", "engine", "record_id", "metric", "value", "limit", "message"])
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Score Sparsity Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Records: {payload['summary']['records']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
        "## Engine Summary",
        "",
        "| Engine | Records | Zero Score % | Min Nonzero | Min Unique |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in payload["summaries"]:
        lines.append(
            f"| {summary['engine']} | {summary['scored_record_count']} | "
            f"{summary['zero_score_pct']:.2f} | {summary['min_nonzero_scores_per_record']} | "
            f"{summary['min_unique_scores_per_record']} |"
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        lines.append("| Engine | Record | Metric | Value | Limit | Message |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for finding in payload["findings"]:
            lines.append(
                f"| {finding['engine']} | {finding['record_id'] or '-'} | {finding['metric']} | "
                f"{finding['value']} | {finding['limit']} | {finding['message']} |"
            )
    else:
        lines.append("No score sparsity findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_score_sparsity_audit",
            "tests": "1",
            "failures": "1" if payload["status"] == "fail" else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "score_sparsity"})
    if payload["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{len(payload['findings'])} score sparsity findings"})
        failure.text = "\n".join(finding["message"] for finding in payload["findings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path, help="HolyC scored prediction JSONL")
    parser.add_argument("--llama", required=True, type=Path, help="llama.cpp scored prediction JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_score_sparsity_audit_latest")
    parser.add_argument("--zero-epsilon", type=float, default=0.0, help="Treat absolute scores <= this value as zero")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-nonzero-scores-per-record", type=int, default=1)
    parser.add_argument("--min-unique-scores-per-record", type=int, default=2)
    parser.add_argument("--max-zero-score-pct", type=float, default=95.0)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, summaries, findings = audit(
        args.holyc,
        args.llama,
        zero_epsilon=args.zero_epsilon,
        min_records=args.min_records,
        min_nonzero_scores_per_record=args.min_nonzero_scores_per_record,
        min_unique_scores_per_record=args.min_unique_scores_per_record,
        max_zero_score_pct=args.max_zero_score_pct,
    )
    payload: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {
            "holyc": str(args.holyc),
            "holyc_sha256": file_sha256(args.holyc),
            "llama": str(args.llama),
            "llama_sha256": file_sha256(args.llama),
        },
        "config": {
            "zero_epsilon": args.zero_epsilon,
            "min_records": args.min_records,
            "min_nonzero_scores_per_record": args.min_nonzero_scores_per_record,
            "min_unique_scores_per_record": args.min_unique_scores_per_record,
            "max_zero_score_pct": args.max_zero_score_pct,
        },
        "summary": {"records": len(records), "summaries": len(summaries), "findings": len(findings)},
        "summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }

    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", payload)
    write_markdown(output_dir / f"{stem}.md", payload)
    write_summary_csv(output_dir / f"{stem}.csv", summaries)
    write_records_csv(output_dir / f"{stem}_records.csv", records)
    write_findings_csv(output_dir / f"{stem}_findings.csv", findings)
    write_junit(output_dir / f"{stem}_junit.xml", payload)
    print(f"eval_score_sparsity_audit={payload['status']} findings={len(findings)}")
    return 2 if args.fail_on_findings and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
