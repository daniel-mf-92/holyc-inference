#!/usr/bin/env python3
"""Audit scored eval predictions for entropy and confidence collapse.

This host-side audit reads paired HolyC and llama.cpp scored-prediction JSONL
files, computes softmax entropy per record, and flags degenerate confidence
profiles or HolyC-vs-llama entropy drift. It does not launch QEMU, use network
services, or touch guest code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EntropyRecord:
    source: str
    engine: str
    record_id: str
    choice_count: int
    top_probability: float
    entropy: float
    normalized_entropy: float
    effective_choices: float


@dataclass(frozen=True)
class EngineSummary:
    source: str
    engine: str
    record_count: int
    scored_count: int
    invalid_count: int
    min_normalized_entropy: float
    mean_normalized_entropy: float
    max_normalized_entropy: float
    mean_top_probability: float
    mean_effective_choices: float


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
    if not isinstance(raw, list) or not raw:
        return None
    scores: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            return None
        value = float(item)
        if not math.isfinite(value):
            return None
        scores.append(value)
    return scores


def softmax_probabilities(scores: list[float]) -> list[float]:
    offset = max(scores)
    exps = [math.exp(score - offset) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


def entropy_metrics(scores: list[float]) -> tuple[float, float, float, float]:
    probabilities = softmax_probabilities(scores)
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    return max(probabilities), entropy, normalized_entropy, math.exp(entropy)


def record_id(payload: dict[str, Any], line_no: int) -> str:
    return str(payload.get("id") or payload.get("record_id") or payload.get("question_id") or f"line:{line_no}")


def load_entropy_records(
    path: Path,
    *,
    engine: str,
    min_normalized_entropy: float,
    max_normalized_entropy: float,
) -> tuple[list[EntropyRecord], list[Finding]]:
    records: list[EntropyRecord] = []
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
            rid = record_id(payload, line_no)
            if rid in seen:
                findings.append(Finding("error", engine, rid, "duplicate_id", 1, 0, f"{engine} duplicate prediction id {rid}"))
            seen.add(rid)
            scores = parse_scores(payload.get("scores"))
            if scores is None:
                findings.append(
                    Finding("error", engine, rid, "scores", "missing_or_invalid", "finite_numeric_list", f"{engine} {rid} scores must be a finite numeric list")
                )
                continue
            top_probability, entropy, normalized_entropy, effective_choices = entropy_metrics(scores)
            records.append(
                EntropyRecord(
                    source=str(path),
                    engine=engine,
                    record_id=rid,
                    choice_count=len(scores),
                    top_probability=top_probability,
                    entropy=entropy,
                    normalized_entropy=normalized_entropy,
                    effective_choices=effective_choices,
                )
            )
            if normalized_entropy < min_normalized_entropy:
                findings.append(
                    Finding(
                        "error",
                        engine,
                        rid,
                        "normalized_entropy_low",
                        normalized_entropy,
                        min_normalized_entropy,
                        f"{engine} {rid} normalized entropy {normalized_entropy:.6g} is below {min_normalized_entropy:.6g}",
                    )
                )
            if normalized_entropy > max_normalized_entropy:
                findings.append(
                    Finding(
                        "error",
                        engine,
                        rid,
                        "normalized_entropy_high",
                        normalized_entropy,
                        max_normalized_entropy,
                        f"{engine} {rid} normalized entropy {normalized_entropy:.6g} is above {max_normalized_entropy:.6g}",
                    )
                )
    return records, findings


def summarize(source: Path, engine: str, records: list[EntropyRecord], invalid_count: int) -> EngineSummary:
    entropies = [record.normalized_entropy for record in records]
    top_probabilities = [record.top_probability for record in records]
    effective_choices = [record.effective_choices for record in records]
    return EngineSummary(
        source=str(source),
        engine=engine,
        record_count=len(records) + invalid_count,
        scored_count=len(records),
        invalid_count=invalid_count,
        min_normalized_entropy=min(entropies) if entropies else 0.0,
        mean_normalized_entropy=sum(entropies) / len(entropies) if entropies else 0.0,
        max_normalized_entropy=max(entropies) if entropies else 0.0,
        mean_top_probability=sum(top_probabilities) / len(top_probabilities) if top_probabilities else 0.0,
        mean_effective_choices=sum(effective_choices) / len(effective_choices) if effective_choices else 0.0,
    )


def audit(
    holyc_path: Path,
    llama_path: Path,
    *,
    min_records: int,
    min_choices: int,
    min_normalized_entropy: float,
    max_normalized_entropy: float,
    max_mean_entropy_delta: float,
    max_record_entropy_delta: float,
) -> tuple[list[EntropyRecord], list[EngineSummary], list[Finding]]:
    holyc_records, findings = load_entropy_records(
        holyc_path,
        engine="holyc",
        min_normalized_entropy=min_normalized_entropy,
        max_normalized_entropy=max_normalized_entropy,
    )
    llama_records, llama_findings = load_entropy_records(
        llama_path,
        engine="llama",
        min_normalized_entropy=min_normalized_entropy,
        max_normalized_entropy=max_normalized_entropy,
    )
    findings.extend(llama_findings)

    holyc_invalid = sum(1 for finding in findings if finding.engine == "holyc" and finding.metric in {"json", "record", "scores"})
    llama_invalid = sum(1 for finding in findings if finding.engine == "llama" and finding.metric in {"json", "record", "scores"})
    summaries = [
        summarize(holyc_path, "holyc", holyc_records, holyc_invalid),
        summarize(llama_path, "llama", llama_records, llama_invalid),
    ]

    for summary in summaries:
        if summary.scored_count < min_records:
            findings.append(
                Finding("error", summary.engine, "", "scored_count", summary.scored_count, min_records, f"{summary.engine} has {summary.scored_count} scored records, below {min_records}")
            )

    all_records = holyc_records + llama_records
    for record in all_records:
        if record.choice_count < min_choices:
            findings.append(
                Finding("error", record.engine, record.record_id, "choice_count", record.choice_count, min_choices, f"{record.engine} {record.record_id} choice count is below {min_choices}")
            )

    holyc_by_id = {record.record_id: record for record in holyc_records}
    llama_by_id = {record.record_id: record for record in llama_records}
    for rid in sorted(set(holyc_by_id) - set(llama_by_id)):
        findings.append(Finding("error", "pair", rid, "missing_llama", 1, 0, f"{rid} exists for HolyC but not llama"))
    for rid in sorted(set(llama_by_id) - set(holyc_by_id)):
        findings.append(Finding("error", "pair", rid, "missing_holyc", 1, 0, f"{rid} exists for llama but not HolyC"))
    for rid in sorted(set(holyc_by_id) & set(llama_by_id)):
        holyc = holyc_by_id[rid]
        llama = llama_by_id[rid]
        if holyc.choice_count != llama.choice_count:
            findings.append(
                Finding("error", "pair", rid, "choice_count", f"{holyc.choice_count}:{llama.choice_count}", "equal", f"{rid} HolyC/llama choice counts differ")
            )
        delta = abs(holyc.normalized_entropy - llama.normalized_entropy)
        if delta > max_record_entropy_delta:
            findings.append(
                Finding("error", "pair", rid, "record_entropy_delta", delta, max_record_entropy_delta, f"{rid} normalized entropy delta {delta:.6g} exceeds {max_record_entropy_delta:.6g}")
            )

    mean_delta = abs(summaries[0].mean_normalized_entropy - summaries[1].mean_normalized_entropy)
    if mean_delta > max_mean_entropy_delta:
        findings.append(
            Finding("error", "pair", "", "mean_entropy_delta", mean_delta, max_mean_entropy_delta, f"HolyC/llama mean normalized entropy delta {mean_delta:.6g} exceeds {max_mean_entropy_delta:.6g}")
        )

    return all_records, summaries, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Entropy Audit",
        "",
        f"- Status: {payload['status']}",
        f"- HolyC predictions: `{payload['inputs']['holyc']['path']}`",
        f"- llama predictions: `{payload['inputs']['llama']['path']}`",
        f"- Findings: {len(payload['findings'])}",
        "",
        "## Summaries",
        "",
        "| engine | scored | mean entropy | min entropy | max entropy | mean top p | effective choices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in payload["summaries"]:
        lines.append(
            f"| {summary['engine']} | {summary['scored_count']} | {summary['mean_normalized_entropy']:.6g} | "
            f"{summary['min_normalized_entropy']:.6g} | {summary['max_normalized_entropy']:.6g} | "
            f"{summary['mean_top_probability']:.6g} | {summary['mean_effective_choices']:.6g} |"
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            lines.append(f"- {finding['severity']}: {finding['message']}")
    else:
        lines.append("- No entropy findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_entropy_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "skipped": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "entropy_hygiene"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} entropy finding(s)"})
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
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
    parser.add_argument("--output-stem", default="eval_entropy_audit_latest")
    parser.add_argument("--min-records", type=nonnegative_int, default=1)
    parser.add_argument("--min-choices", type=nonnegative_int, default=2)
    parser.add_argument("--min-normalized-entropy", type=probability, default=0.0)
    parser.add_argument("--max-normalized-entropy", type=probability, default=1.0)
    parser.add_argument("--max-mean-entropy-delta", type=probability, default=0.25)
    parser.add_argument("--max-record-entropy-delta", type=probability, default=0.50)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in (args.holyc, args.llama):
        if not path.exists():
            parser.error(f"input not found: {path}")
    if args.min_normalized_entropy > args.max_normalized_entropy:
        parser.error("--min-normalized-entropy cannot exceed --max-normalized-entropy")

    records, summaries, findings = audit(
        args.holyc,
        args.llama,
        min_records=args.min_records,
        min_choices=args.min_choices,
        min_normalized_entropy=args.min_normalized_entropy,
        max_normalized_entropy=args.max_normalized_entropy,
        max_mean_entropy_delta=args.max_mean_entropy_delta,
        max_record_entropy_delta=args.max_record_entropy_delta,
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
            "min_normalized_entropy": args.min_normalized_entropy,
            "max_normalized_entropy": args.max_normalized_entropy,
            "max_mean_entropy_delta": args.max_mean_entropy_delta,
            "max_record_entropy_delta": args.max_record_entropy_delta,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_summaries.csv", [asdict(summary) for summary in summaries], list(EngineSummary.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_records.csv", [asdict(record) for record in records], list(EntropyRecord.__dataclass_fields__))
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)

    print(f"eval_entropy_audit_status={payload['status']}")
    print(f"eval_entropy_audit_findings={len(findings)}")
    print(f"eval_entropy_audit_json={json_path}")
    return 2 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
