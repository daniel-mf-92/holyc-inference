#!/usr/bin/env python3
"""Audit paired HolyC-vs-llama scored prediction deltas.

This offline host-side tool reads a local gold JSONL dataset plus local HolyC
and llama.cpp scored prediction files. It checks that paired score vectors stay
within configurable absolute-delta gates and never launches QEMU or touches the
TempleOS guest.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import eval_compare


@dataclass(frozen=True)
class DeltaPair:
    record_id: str
    dataset: str
    split: str
    choice_count: int
    holyc_top_index: int
    llama_top_index: int
    top_index_match: bool
    max_abs_delta: float
    mean_abs_delta: float
    top_score_abs_delta: float
    gold_score_abs_delta: float


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


def finite_scores(scores: list[float] | None) -> list[float] | None:
    if scores is None:
        return None
    return scores if all(math.isfinite(score) for score in scores) else None


def top_index(scores: list[float]) -> int:
    return max(range(len(scores)), key=lambda index: (scores[index], -index))


def load_predictions(
    path: Path,
    source: str,
    gold: dict[str, eval_compare.GoldCase],
    findings: list[Finding],
) -> dict[str, eval_compare.Prediction]:
    predictions: dict[str, eval_compare.Prediction] = {}
    try:
        rows = eval_compare.read_prediction_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, source, "-", "load_error", f"cannot read predictions: {exc}")
        return predictions

    for index, row in enumerate(rows):
        row_label = f"{path}:{index + 1}"
        try:
            record_id = eval_compare.case_id(row, row_label)
        except ValueError as exc:
            append_finding(findings, source, "-", "missing_id", str(exc))
            continue
        if record_id in predictions:
            append_finding(findings, source, record_id, "duplicate_id", "duplicate prediction row")
            continue
        gold_case = gold.get(record_id)
        if gold_case is None:
            append_finding(findings, source, record_id, "extra_id", "prediction id is not present in gold")
            continue
        try:
            predictions[record_id] = eval_compare.normalize_prediction(row, gold_case, path, index)
        except ValueError as exc:
            append_finding(findings, source, record_id, "invalid_prediction", str(exc))
    return predictions


def build_pairs(
    gold: dict[str, eval_compare.GoldCase],
    holyc: dict[str, eval_compare.Prediction],
    llama: dict[str, eval_compare.Prediction],
    findings: list[Finding],
) -> list[DeltaPair]:
    pairs: list[DeltaPair] = []
    for record_id, gold_case in sorted(gold.items()):
        holyc_prediction = holyc.get(record_id)
        llama_prediction = llama.get(record_id)
        if holyc_prediction is None:
            append_finding(findings, "holyc", record_id, "missing_id", "gold id is missing from HolyC predictions")
        if llama_prediction is None:
            append_finding(findings, "llama.cpp", record_id, "missing_id", "gold id is missing from llama.cpp predictions")
        if holyc_prediction is None or llama_prediction is None:
            continue

        holyc_scores = finite_scores(holyc_prediction.scores)
        llama_scores = finite_scores(llama_prediction.scores)
        if holyc_scores is None:
            append_finding(findings, "holyc", record_id, "missing_scores", "HolyC row needs finite score vector")
        if llama_scores is None:
            append_finding(findings, "llama.cpp", record_id, "missing_scores", "llama.cpp row needs finite score vector")
        if holyc_scores is None or llama_scores is None:
            continue
        if len(holyc_scores) != len(gold_case.choices) or len(llama_scores) != len(gold_case.choices):
            append_finding(
                findings,
                "pair",
                record_id,
                "score_shape_mismatch",
                f"score counts holyc={len(holyc_scores)} llama.cpp={len(llama_scores)} choices={len(gold_case.choices)}",
            )
            continue

        deltas = [abs(holyc_score - llama_score) for holyc_score, llama_score in zip(holyc_scores, llama_scores)]
        holyc_top = top_index(holyc_scores)
        llama_top = top_index(llama_scores)
        pairs.append(
            DeltaPair(
                record_id=record_id,
                dataset=gold_case.dataset,
                split=gold_case.split,
                choice_count=len(gold_case.choices),
                holyc_top_index=holyc_top,
                llama_top_index=llama_top,
                top_index_match=holyc_top == llama_top,
                max_abs_delta=max(deltas),
                mean_abs_delta=sum(deltas) / len(deltas),
                top_score_abs_delta=abs(holyc_scores[holyc_top] - llama_scores[llama_top]),
                gold_score_abs_delta=deltas[gold_case.answer_index],
            )
        )
    return pairs


def pct(count: int, total: int) -> float | None:
    return count / total * 100.0 if total else None


def summarize(pairs: list[DeltaPair], gold_count: int) -> dict[str, Any]:
    paired = len(pairs)
    top_matches = sum(1 for pair in pairs if pair.top_index_match)
    return {
        "gold_records": gold_count,
        "paired_scored_records": paired,
        "pair_coverage_pct": pct(paired, gold_count),
        "top_index_match_pct": pct(top_matches, paired),
        "max_abs_delta": max((pair.max_abs_delta for pair in pairs), default=None),
        "mean_abs_delta": (sum(pair.mean_abs_delta for pair in pairs) / paired) if paired else None,
        "max_top_score_abs_delta": max((pair.top_score_abs_delta for pair in pairs), default=None),
        "max_gold_score_abs_delta": max((pair.gold_score_abs_delta for pair in pairs), default=None),
    }


def add_gate_findings(summary: dict[str, Any], pairs: list[DeltaPair], args: argparse.Namespace, findings: list[Finding]) -> None:
    coverage = summary["pair_coverage_pct"]
    if coverage is None or coverage < args.min_pair_coverage_pct:
        append_finding(findings, "summary", "-", "pair_coverage", f"pair_coverage_pct={coverage} below threshold {args.min_pair_coverage_pct}")
    top_match = summary["top_index_match_pct"]
    if top_match is None or top_match < args.min_top_index_match_pct:
        append_finding(findings, "summary", "-", "top_index_match", f"top_index_match_pct={top_match} below threshold {args.min_top_index_match_pct}")

    for pair in pairs:
        if pair.max_abs_delta > args.max_abs_delta:
            append_finding(
                findings,
                "pair",
                pair.record_id,
                "max_abs_delta",
                f"max_abs_delta={pair.max_abs_delta:.6g} above threshold {args.max_abs_delta:.6g}",
            )
        if pair.mean_abs_delta > args.max_mean_abs_delta:
            append_finding(
                findings,
                "pair",
                pair.record_id,
                "mean_abs_delta",
                f"mean_abs_delta={pair.mean_abs_delta:.6g} above threshold {args.max_mean_abs_delta:.6g}",
            )
        if pair.top_score_abs_delta > args.max_top_score_abs_delta:
            append_finding(
                findings,
                "pair",
                pair.record_id,
                "top_score_abs_delta",
                f"top_score_abs_delta={pair.top_score_abs_delta:.6g} above threshold {args.max_top_score_abs_delta:.6g}",
            )


def write_csv(path: Path, rows: list[DeltaPair]) -> None:
    fieldnames = list(DeltaPair.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__.keys()))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Eval Score Delta Audit",
        "",
        f"- status: {payload['status']}",
        f"- gold_records: {summary['gold_records']}",
        f"- paired_scored_records: {summary['paired_scored_records']}",
        f"- pair_coverage_pct: {summary['pair_coverage_pct']}",
        f"- top_index_match_pct: {summary['top_index_match_pct']}",
        f"- max_abs_delta: {summary['max_abs_delta']}",
        f"- mean_abs_delta: {summary['mean_abs_delta']}",
        f"- max_top_score_abs_delta: {summary['max_top_score_abs_delta']}",
        "",
    ]
    if payload["findings"]:
        lines.append("## Findings")
        for finding in payload["findings"]:
            lines.append(f"- {finding['kind']}: {finding['detail']}")
    else:
        lines.append("No score delta findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    root = ET.Element(
        "testsuite",
        {"name": "holyc_eval_score_delta_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    case = ET.SubElement(root, "testcase", {"name": "score_delta"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--holyc", required=True, type=Path)
    parser.add_argument("--llama", required=True, type=Path)
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--min-pair-coverage-pct", type=float, default=100.0)
    parser.add_argument("--min-top-index-match-pct", type=float, default=0.0)
    parser.add_argument("--max-abs-delta", type=float, default=1.0)
    parser.add_argument("--max-mean-abs-delta", type=float, default=0.5)
    parser.add_argument("--max-top-score-abs-delta", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_score_delta_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "gold", "-", "load_error", str(exc))
        gold = {}
    holyc = load_predictions(args.holyc, "holyc", gold, findings)
    llama = load_predictions(args.llama, "llama.cpp", gold, findings)
    pairs = build_pairs(gold, holyc, llama, findings)
    summary = summarize(pairs, len(gold))
    add_gate_findings(summary, pairs, args, findings)

    status = "fail" if findings else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "inputs": {"gold": str(args.gold), "holyc": str(args.holyc), "llama": str(args.llama)},
        "thresholds": {
            "min_pair_coverage_pct": args.min_pair_coverage_pct,
            "min_top_index_match_pct": args.min_top_index_match_pct,
            "max_abs_delta": args.max_abs_delta,
            "max_mean_abs_delta": args.max_mean_abs_delta,
            "max_top_score_abs_delta": args.max_top_score_abs_delta,
        },
        "summary": summary,
        "pairs": [asdict(pair) for pair in pairs],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_csv(args.output_dir / f"{stem}.csv", pairs)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
