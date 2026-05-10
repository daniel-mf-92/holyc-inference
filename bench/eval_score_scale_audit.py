#!/usr/bin/env python3
"""Audit scored eval prediction vectors for scale and shape hygiene.

This host-side audit reads HolyC and llama.cpp scored-prediction JSONL files
and checks that score vectors are finite, non-constant, structurally paired,
and on comparable scales. It does not launch QEMU or touch guest code.
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
class RecordScore:
    source: str
    engine: str
    record_id: str
    choice_count: int
    min_score: float
    max_score: float
    span: float
    mean_abs_score: float
    finite: bool
    constant: bool


@dataclass(frozen=True)
class EngineSummary:
    source: str
    engine: str
    record_count: int
    scored_count: int
    invalid_count: int
    constant_count: int
    min_choice_count: int
    max_choice_count: int
    min_span: float
    mean_span: float
    max_span: float
    mean_abs_score: float
    max_abs_score: float


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
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        scores.append(float(item))
    return scores


def load_scores(path: Path, *, engine: str, min_score_span: float) -> tuple[list[RecordScore], list[Finding]]:
    records: list[RecordScore] = []
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
                    Finding("error", engine, f"line:{line_no}", "record", type(payload).__name__, "object", f"{path}:{line_no}: record must be an object")
                )
                continue
            record_id = str(payload.get("id") or payload.get("record_id") or f"line:{line_no}")
            if record_id in seen:
                findings.append(
                    Finding("error", engine, record_id, "duplicate_id", 1, 0, f"{engine} duplicate scored prediction id {record_id}")
                )
            seen.add(record_id)
            scores = parse_scores(payload.get("scores"))
            if scores is None:
                findings.append(
                    Finding("error", engine, record_id, "scores", "missing_or_invalid", "numeric_list", f"{engine} {record_id} scores must be a numeric list")
                )
                continue
            finite = all(math.isfinite(score) for score in scores)
            if finite and scores:
                min_score = min(scores)
                max_score = max(scores)
                span = max_score - min_score
                mean_abs = sum(abs(score) for score in scores) / len(scores)
            else:
                min_score = 0.0
                max_score = 0.0
                span = 0.0
                mean_abs = 0.0
            constant = finite and len(scores) > 0 and span < min_score_span
            record = RecordScore(
                source=str(path),
                engine=engine,
                record_id=record_id,
                choice_count=len(scores),
                min_score=min_score,
                max_score=max_score,
                span=span,
                mean_abs_score=mean_abs,
                finite=finite,
                constant=constant,
            )
            records.append(record)
            if not finite:
                findings.append(
                    Finding("error", engine, record_id, "finite", "false", "true", f"{engine} {record_id} has non-finite scores")
                )
            if constant:
                findings.append(
                    Finding("error", engine, record_id, "score_span", span, min_score_span, f"{engine} {record_id} score span {span:.6g} is below {min_score_span:.6g}")
                )
    return records, findings


def summarize(source: Path, engine: str, records: list[RecordScore]) -> EngineSummary:
    valid = [record for record in records if record.finite and record.choice_count > 0]
    spans = [record.span for record in valid]
    mean_abs = [record.mean_abs_score for record in valid]
    choices = [record.choice_count for record in records]
    return EngineSummary(
        source=str(source),
        engine=engine,
        record_count=len(records),
        scored_count=len(valid),
        invalid_count=len(records) - len(valid),
        constant_count=sum(1 for record in records if record.constant),
        min_choice_count=min(choices) if choices else 0,
        max_choice_count=max(choices) if choices else 0,
        min_span=min(spans) if spans else 0.0,
        mean_span=sum(spans) / len(spans) if spans else 0.0,
        max_span=max(spans) if spans else 0.0,
        mean_abs_score=sum(mean_abs) / len(mean_abs) if mean_abs else 0.0,
        max_abs_score=max(mean_abs) if mean_abs else 0.0,
    )


def ratio(a: float, b: float) -> float:
    if a == 0.0 and b == 0.0:
        return 1.0
    if a == 0.0 or b == 0.0:
        return math.inf
    high = max(abs(a), abs(b))
    low = min(abs(a), abs(b))
    return high / low


def audit(
    holyc_path: Path,
    llama_path: Path,
    *,
    min_records: int,
    min_choices: int,
    min_score_span: float,
    max_mean_span_ratio: float,
    max_mean_abs_score_ratio: float,
) -> tuple[list[RecordScore], list[EngineSummary], list[Finding]]:
    holyc_records, findings = load_scores(holyc_path, engine="holyc", min_score_span=min_score_span)
    llama_records, llama_findings = load_scores(llama_path, engine="llama", min_score_span=min_score_span)
    findings.extend(llama_findings)
    all_records = holyc_records + llama_records
    summaries = [summarize(holyc_path, "holyc", holyc_records), summarize(llama_path, "llama", llama_records)]

    for summary in summaries:
        if summary.record_count < min_records:
            findings.append(
                Finding("error", summary.engine, "", "record_count", summary.record_count, min_records, f"{summary.engine} has {summary.record_count} records, below {min_records}")
            )
        if summary.min_choice_count < min_choices:
            findings.append(
                Finding("error", summary.engine, "", "min_choice_count", summary.min_choice_count, min_choices, f"{summary.engine} minimum choice count {summary.min_choice_count} is below {min_choices}")
            )

    holyc_by_id = {record.record_id: record for record in holyc_records}
    llama_by_id = {record.record_id: record for record in llama_records}
    missing_llama = sorted(set(holyc_by_id) - set(llama_by_id))
    missing_holyc = sorted(set(llama_by_id) - set(holyc_by_id))
    for record_id in missing_llama:
        findings.append(Finding("error", "pair", record_id, "missing_llama", 1, 0, f"{record_id} exists for HolyC but not llama"))
    for record_id in missing_holyc:
        findings.append(Finding("error", "pair", record_id, "missing_holyc", 1, 0, f"{record_id} exists for llama but not HolyC"))
    for record_id in sorted(set(holyc_by_id) & set(llama_by_id)):
        holyc = holyc_by_id[record_id]
        llama = llama_by_id[record_id]
        if holyc.choice_count != llama.choice_count:
            findings.append(
                Finding("error", "pair", record_id, "choice_count", f"{holyc.choice_count}:{llama.choice_count}", "equal", f"{record_id} HolyC/llama choice counts differ")
            )

    span_ratio = ratio(summaries[0].mean_span, summaries[1].mean_span)
    if span_ratio > max_mean_span_ratio:
        findings.append(
            Finding("error", "pair", "", "mean_span_ratio", span_ratio, max_mean_span_ratio, f"HolyC/llama mean score span ratio {span_ratio:.6g} exceeds {max_mean_span_ratio:.6g}")
        )
    abs_ratio = ratio(summaries[0].mean_abs_score, summaries[1].mean_abs_score)
    if abs_ratio > max_mean_abs_score_ratio:
        findings.append(
            Finding("error", "pair", "", "mean_abs_score_ratio", abs_ratio, max_mean_abs_score_ratio, f"HolyC/llama mean absolute score ratio {abs_ratio:.6g} exceeds {max_mean_abs_score_ratio:.6g}")
        )

    return all_records, summaries, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Score Scale Audit",
        "",
        f"- Status: {payload['status']}",
        f"- HolyC predictions: `{payload['inputs']['holyc']['path']}`",
        f"- llama predictions: `{payload['inputs']['llama']['path']}`",
        f"- Findings: {len(payload['findings'])}",
        "",
        "## Summaries",
        "",
        "| engine | records | scored | min choices | mean span | mean abs score | constants |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in payload["summaries"]:
        lines.append(
            f"| {summary['engine']} | {summary['record_count']} | {summary['scored_count']} | "
            f"{summary['min_choice_count']} | {summary['mean_span']:.6g} | {summary['mean_abs_score']:.6g} | {summary['constant_count']} |"
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            lines.append(f"- {finding['severity']}: {finding['message']}")
    else:
        lines.append("- No score-scale findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, *, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_score_scale_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "score_scale_hygiene"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} score-scale finding(s)"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holyc", required=True, type=Path, help="HolyC scored prediction JSONL")
    parser.add_argument("--llama", required=True, type=Path, help="llama.cpp scored prediction JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_score_scale_audit_latest")
    parser.add_argument("--min-records", type=nonnegative_int, default=1)
    parser.add_argument("--min-choices", type=nonnegative_int, default=2)
    parser.add_argument("--min-score-span", type=positive_float, default=1e-9)
    parser.add_argument("--max-mean-span-ratio", type=positive_float, default=10.0)
    parser.add_argument("--max-mean-abs-score-ratio", type=positive_float, default=10.0)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in (args.holyc, args.llama):
        if not path.exists():
            parser.error(f"input not found: {path}")

    records, summaries, findings = audit(
        args.holyc,
        args.llama,
        min_records=args.min_records,
        min_choices=args.min_choices,
        min_score_span=args.min_score_span,
        max_mean_span_ratio=args.max_mean_span_ratio,
        max_mean_abs_score_ratio=args.max_mean_abs_score_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": {
            "holyc": {"path": str(args.holyc), "sha256": file_sha256(args.holyc)},
            "llama": {"path": str(args.llama), "sha256": file_sha256(args.llama)},
        },
        "thresholds": {
            "min_records": args.min_records,
            "min_choices": args.min_choices,
            "min_score_span": args.min_score_span,
            "max_mean_span_ratio": args.max_mean_span_ratio,
            "max_mean_abs_score_ratio": args.max_mean_abs_score_ratio,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_summaries.csv", [asdict(summary) for summary in summaries], list(EngineSummary.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_records.csv", [asdict(record) for record in records], list(RecordScore.__dataclass_fields__))
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings=findings)

    print(f"eval_score_scale_audit_status={payload['status']}")
    print(f"eval_score_scale_audit_findings={len(findings)}")
    print(f"eval_score_scale_audit_json={json_path}")
    return 2 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
