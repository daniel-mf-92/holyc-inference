#!/usr/bin/env python3
"""Audit HolyC-vs-llama top-k score ranking overlap.

This offline host-side tool consumes a local gold JSONL dataset and local
HolyC/llama.cpp scored prediction files. It never launches QEMU, downloads
datasets, or touches the TempleOS guest.
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
class TopKPair:
    record_id: str
    dataset: str
    split: str
    choice_count: int
    answer_index: int
    k: int
    holyc_topk: str
    llama_topk: str
    overlap_count: int
    jaccard: float
    topk_exact_match: bool
    top1_match: bool
    holyc_gold_in_topk: bool
    llama_gold_in_topk: bool


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    record_id: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_scores(scores: list[float] | None) -> list[float] | None:
    if scores is None:
        return None
    if not all(math.isfinite(score) for score in scores):
        return None
    return scores


def topk_indices(scores: list[float], k: int) -> list[int]:
    return [index for index, _score in sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))[:k]]


def append_finding(findings: list[Finding], source: str, record_id: str, kind: str, detail: str) -> None:
    findings.append(Finding("error", source, record_id, kind, detail))


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
            prediction = eval_compare.normalize_prediction(row, gold_case, path, index)
        except ValueError as exc:
            append_finding(findings, source, record_id, "invalid_prediction", str(exc))
            continue
        predictions[record_id] = prediction
    return predictions


def build_pairs(
    gold: dict[str, eval_compare.GoldCase],
    holyc: dict[str, eval_compare.Prediction],
    llama: dict[str, eval_compare.Prediction],
    top_k: int,
    findings: list[Finding],
) -> list[TopKPair]:
    pairs: list[TopKPair] = []
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

        k = min(top_k, len(gold_case.choices))
        holyc_topk = topk_indices(holyc_scores, k)
        llama_topk = topk_indices(llama_scores, k)
        holyc_set = set(holyc_topk)
        llama_set = set(llama_topk)
        overlap_count = len(holyc_set & llama_set)
        union_count = len(holyc_set | llama_set)
        pairs.append(
            TopKPair(
                record_id=record_id,
                dataset=gold_case.dataset,
                split=gold_case.split,
                choice_count=len(gold_case.choices),
                answer_index=gold_case.answer_index,
                k=k,
                holyc_topk=",".join(str(index) for index in holyc_topk),
                llama_topk=",".join(str(index) for index in llama_topk),
                overlap_count=overlap_count,
                jaccard=overlap_count / union_count if union_count else 0.0,
                topk_exact_match=holyc_topk == llama_topk,
                top1_match=holyc_topk[:1] == llama_topk[:1],
                holyc_gold_in_topk=gold_case.answer_index in holyc_set,
                llama_gold_in_topk=gold_case.answer_index in llama_set,
            )
        )
    return pairs


def pct(count: int, total: int) -> float | None:
    return count / total * 100.0 if total else None


def summarize(pairs: list[TopKPair], gold_count: int) -> dict[str, Any]:
    total = len(pairs)
    topk_exact = sum(1 for pair in pairs if pair.topk_exact_match)
    top1_match = sum(1 for pair in pairs if pair.top1_match)
    holyc_gold = sum(1 for pair in pairs if pair.holyc_gold_in_topk)
    llama_gold = sum(1 for pair in pairs if pair.llama_gold_in_topk)
    return {
        "gold_records": gold_count,
        "paired_scored_records": total,
        "pair_coverage_pct": pct(total, gold_count),
        "topk_exact_match_pct": pct(topk_exact, total),
        "top1_match_pct": pct(top1_match, total),
        "top1_disagree_pct": pct(total - top1_match, total),
        "avg_jaccard": sum(pair.jaccard for pair in pairs) / total if total else None,
        "holyc_gold_in_topk_pct": pct(holyc_gold, total),
        "llama_gold_in_topk_pct": pct(llama_gold, total),
    }


def add_gate_findings(summary: dict[str, Any], args: argparse.Namespace, findings: list[Finding]) -> None:
    gates = (
        ("pair_coverage_pct", args.min_pair_coverage_pct, "coverage"),
        ("topk_exact_match_pct", args.min_topk_exact_match_pct, "topk_exact_match"),
        ("avg_jaccard", args.min_avg_jaccard, "avg_jaccard"),
    )
    for metric, threshold, kind in gates:
        value = summary.get(metric)
        if value is None or float(value) < float(threshold):
            append_finding(findings, "summary", "-", kind, f"{metric}={value} below threshold {threshold}")
    top1_disagree = summary.get("top1_disagree_pct")
    if top1_disagree is None or float(top1_disagree) > args.max_top1_disagree_pct:
        append_finding(
            findings,
            "summary",
            "-",
            "top1_disagree",
            f"top1_disagree_pct={top1_disagree} above threshold {args.max_top1_disagree_pct}",
        )


def write_csv(path: Path, rows: list[Any]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(TopKPair.__dataclass_fields__.keys())
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
        "# Eval Top-k Overlap Audit",
        "",
        f"- status: {payload['status']}",
        f"- top_k: {payload['top_k']}",
        f"- gold_records: {summary['gold_records']}",
        f"- paired_scored_records: {summary['paired_scored_records']}",
        f"- pair_coverage_pct: {summary['pair_coverage_pct']}",
        f"- topk_exact_match_pct: {summary['topk_exact_match_pct']}",
        f"- top1_disagree_pct: {summary['top1_disagree_pct']}",
        f"- avg_jaccard: {summary['avg_jaccard']}",
        "",
    ]
    if payload["findings"]:
        lines.append("## Findings")
        for finding in payload["findings"]:
            lines.append(f"- {finding['kind']}: {finding['detail']}")
    else:
        lines.append("No top-k overlap findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    root = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_topk_overlap_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    case = ET.SubElement(root, "testcase", {"name": "topk_overlap"})
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
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-pair-coverage-pct", type=float, default=100.0)
    parser.add_argument("--min-topk-exact-match-pct", type=float, default=0.0)
    parser.add_argument("--min-avg-jaccard", type=float, default=0.0)
    parser.add_argument("--max-top1-disagree-pct", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_topk_overlap_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    findings: list[Finding] = []
    try:
        gold = eval_compare.load_gold(args.gold, args.dataset, args.split)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "gold", "-", "load_error", str(exc))
        gold = {}
    holyc = load_predictions(args.holyc, "holyc", gold, findings)
    llama = load_predictions(args.llama, "llama.cpp", gold, findings)
    pairs = build_pairs(gold, holyc, llama, args.top_k, findings)
    summary = summarize(pairs, len(gold))
    add_gate_findings(summary, args, findings)

    status = "fail" if findings else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "top_k": args.top_k,
        "inputs": {
            "gold": str(args.gold),
            "holyc": str(args.holyc),
            "llama": str(args.llama),
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
