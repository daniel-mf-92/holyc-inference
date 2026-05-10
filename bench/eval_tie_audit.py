#!/usr/bin/env python3
"""Audit scored eval predictions for ambiguous top-choice ties.

This host-side tool reads HolyC and llama.cpp scored-prediction JSONL files,
checks that the top choice is unique, measures top-vs-runner-up margin, and
reports HolyC-vs-llama top-index disagreement. It does not launch QEMU or touch
the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecordTie:
    source: str
    engine: str
    record_id: str
    choice_count: int
    top_index: int
    top_score: float
    runner_up_score: float | None
    top_margin: float | None
    top_tie_count: int
    has_top_tie: bool


@dataclass(frozen=True)
class EngineSummary:
    source: str
    engine: str
    record_count: int
    scored_count: int
    invalid_count: int
    top_tie_count: int
    top_tie_rate: float
    mean_top_margin: float
    min_top_margin: float


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


def parse_scores(raw: Any) -> list[float] | None:
    if not isinstance(raw, list) or not raw:
        return None
    scores: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        score = float(item)
        if not math.isfinite(score):
            return None
        scores.append(score)
    return scores


def top_stats(scores: list[float], epsilon: float) -> tuple[int, float, float | None, float | None, int]:
    ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
    top_index, top_score = ranked[0]
    top_tie_count = sum(1 for _, score in ranked if abs(score - top_score) <= epsilon)
    if len(ranked) == 1:
        return top_index, top_score, None, None, top_tie_count
    runner_up_score = ranked[1][1]
    return top_index, top_score, runner_up_score, top_score - runner_up_score, top_tie_count


def load_records(path: Path, *, engine: str, epsilon: float) -> tuple[list[RecordTie], list[Finding], int]:
    records: list[RecordTie] = []
    findings: list[Finding] = []
    invalid_count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_count += 1
                findings.append(Finding("error", engine, f"line:{line_no}", "json", "invalid", "valid", f"{path}:{line_no}: invalid JSON: {exc}"))
                continue
            if not isinstance(payload, dict):
                invalid_count += 1
                findings.append(Finding("error", engine, f"line:{line_no}", "record", type(payload).__name__, "object", f"{path}:{line_no}: record must be an object"))
                continue
            record_id = str(payload.get("id") or payload.get("record_id") or f"line:{line_no}")
            if record_id in seen:
                findings.append(Finding("error", engine, record_id, "duplicate_id", 1, 0, f"{engine} duplicate prediction id {record_id}"))
            seen.add(record_id)
            scores = parse_scores(payload.get("scores"))
            if scores is None:
                invalid_count += 1
                findings.append(Finding("error", engine, record_id, "scores", "missing_or_invalid", "finite_numeric_list", f"{engine} {record_id} scores must be a non-empty finite numeric list"))
                continue
            top_index, top_score, runner_up_score, top_margin, tie_count = top_stats(scores, epsilon)
            records.append(
                RecordTie(
                    source=str(path),
                    engine=engine,
                    record_id=record_id,
                    choice_count=len(scores),
                    top_index=top_index,
                    top_score=top_score,
                    runner_up_score=runner_up_score,
                    top_margin=top_margin,
                    top_tie_count=tie_count,
                    has_top_tie=tie_count > 1,
                )
            )
    return records, findings, invalid_count


def summarize(source: Path, engine: str, records: list[RecordTie], invalid_count: int) -> EngineSummary:
    margins = [record.top_margin for record in records if record.top_margin is not None]
    tie_count = sum(1 for record in records if record.has_top_tie)
    return EngineSummary(
        source=str(source),
        engine=engine,
        record_count=len(records) + invalid_count,
        scored_count=len(records),
        invalid_count=invalid_count,
        top_tie_count=tie_count,
        top_tie_rate=tie_count / len(records) if records else 0.0,
        mean_top_margin=sum(margins) / len(margins) if margins else 0.0,
        min_top_margin=min(margins) if margins else 0.0,
    )


def audit(
    holyc_path: Path,
    llama_path: Path,
    *,
    epsilon: float,
    min_records: int,
    max_top_tie_rate: float,
    min_mean_top_margin: float,
    max_top_index_disagreement_rate: float | None,
) -> tuple[list[RecordTie], list[EngineSummary], list[Finding], dict[str, Any]]:
    holyc_records, findings, holyc_invalid = load_records(holyc_path, engine="holyc", epsilon=epsilon)
    llama_records, llama_findings, llama_invalid = load_records(llama_path, engine="llama", epsilon=epsilon)
    findings.extend(llama_findings)
    summaries = [
        summarize(holyc_path, "holyc", holyc_records, holyc_invalid),
        summarize(llama_path, "llama", llama_records, llama_invalid),
    ]

    for summary in summaries:
        if summary.record_count < min_records:
            findings.append(Finding("error", summary.engine, "", "record_count", summary.record_count, min_records, f"{summary.engine} has {summary.record_count} records, below {min_records}"))
        if summary.top_tie_rate > max_top_tie_rate:
            findings.append(Finding("error", summary.engine, "", "top_tie_rate", summary.top_tie_rate, max_top_tie_rate, f"{summary.engine} top tie rate {summary.top_tie_rate:.6g} is above {max_top_tie_rate:.6g}"))
        if summary.mean_top_margin < min_mean_top_margin:
            findings.append(Finding("error", summary.engine, "", "mean_top_margin", summary.mean_top_margin, min_mean_top_margin, f"{summary.engine} mean top margin {summary.mean_top_margin:.6g} is below {min_mean_top_margin:.6g}"))

    holyc_by_id = {record.record_id: record for record in holyc_records}
    llama_by_id = {record.record_id: record for record in llama_records}
    shared_ids = sorted(set(holyc_by_id) & set(llama_by_id))
    for record_id in sorted(set(holyc_by_id) - set(llama_by_id)):
        findings.append(Finding("error", "llama", record_id, "missing_pair", "missing", "present", f"llama is missing scored prediction id {record_id}"))
    for record_id in sorted(set(llama_by_id) - set(holyc_by_id)):
        findings.append(Finding("error", "holyc", record_id, "missing_pair", "missing", "present", f"HolyC is missing scored prediction id {record_id}"))
    disagreements = [
        record_id
        for record_id in shared_ids
        if holyc_by_id[record_id].top_index != llama_by_id[record_id].top_index
    ]
    disagreement_rate = len(disagreements) / len(shared_ids) if shared_ids else 0.0
    if max_top_index_disagreement_rate is not None and disagreement_rate > max_top_index_disagreement_rate:
        findings.append(Finding("error", "paired", "", "top_index_disagreement_rate", disagreement_rate, max_top_index_disagreement_rate, f"paired top-index disagreement rate {disagreement_rate:.6g} is above {max_top_index_disagreement_rate:.6g}"))

    paired_summary = {
        "shared_records": len(shared_ids),
        "top_index_disagreements": len(disagreements),
        "top_index_disagreement_rate": disagreement_rate,
    }
    return holyc_records + llama_records, summaries, findings, paired_summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    output_stem: str,
    records: list[RecordTie],
    summaries: list[EngineSummary],
    findings: list[Finding],
    paired_summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if findings else "pass"
    payload = {
        "tool": "eval_tie_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "records": len(records),
            "findings": len(findings),
            **paired_summary,
        },
        "engine_summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    (output_dir / f"{output_stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / f"{output_stem}.csv", [asdict(summary) for summary in summaries], list(asdict(summaries[0]).keys()) if summaries else ["source", "engine"])
    write_csv(output_dir / f"{output_stem}_records.csv", [asdict(record) for record in records], list(asdict(records[0]).keys()) if records else ["source", "engine", "record_id"])
    write_csv(output_dir / f"{output_stem}_findings.csv", [asdict(finding) for finding in findings], ["severity", "engine", "record_id", "metric", "value", "limit", "message"])

    lines = [
        "# Eval Tie Audit",
        "",
        f"- Status: {status}",
        f"- Records: {len(records)}",
        f"- Shared records: {paired_summary['shared_records']}",
        f"- Top-index disagreement rate: {paired_summary['top_index_disagreement_rate']:.6g}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            record = f" `{finding.record_id}`" if finding.record_id else ""
            lines.append(f"- {finding.severity}: {finding.engine}{record} {finding.metric}: {finding.message}")
    else:
        lines.append("No eval tie findings.")
    (output_dir / f"{output_stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", {"name": "holyc_eval_tie_audit", "tests": "1", "failures": "1" if status == "fail" else "0"})
    case = ET.SubElement(suite, "testcase", {"name": "eval_top_choice_ties"})
    if status == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} eval tie finding(s)"})
        failure.text = "\n".join(f"{finding.engine} {finding.metric}: {finding.message}" for finding in findings)
    ET.ElementTree(suite).write(output_dir / f"{output_stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_tie_audit_latest")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--max-top-tie-rate", type=float, default=0.0)
    parser.add_argument("--min-mean-top-margin", type=float, default=0.0)
    parser.add_argument("--max-top-index-disagreement-rate", type=float)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epsilon < 0 or not math.isfinite(args.epsilon):
        raise ValueError("--epsilon must be a finite non-negative number")
    if args.min_records < 0:
        raise ValueError("--min-records must be non-negative")
    for name in ("max_top_tie_rate", "min_mean_top_margin"):
        value = getattr(args, name)
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if not 0.0 <= args.max_top_tie_rate <= 1.0:
        raise ValueError("--max-top-tie-rate must be between 0 and 1")
    if args.max_top_index_disagreement_rate is not None and not 0.0 <= args.max_top_index_disagreement_rate <= 1.0:
        raise ValueError("--max-top-index-disagreement-rate must be between 0 and 1")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        records, summaries, findings, paired_summary = audit(
            args.holyc,
            args.llama,
            epsilon=args.epsilon,
            min_records=args.min_records,
            max_top_tie_rate=args.max_top_tie_rate,
            min_mean_top_margin=args.min_mean_top_margin,
            max_top_index_disagreement_rate=args.max_top_index_disagreement_rate,
        )
        write_outputs(args.output_dir, args.output_stem, records, summaries, findings, paired_summary)
    except (OSError, ValueError) as exc:
        print(f"eval_tie_audit: {exc}", file=sys.stderr)
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
