#!/usr/bin/env python3
"""Audit scored eval rank metrics for HolyC vs llama.cpp reports.

This host-side tool consumes existing `eval_compare.py` JSON artifacts and
checks top-k accuracy, mean reciprocal rank, scored-row coverage, and HolyC
rank loss against llama.cpp. It never launches QEMU and never touches the
TempleOS guest.
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
from typing import Any


@dataclass(frozen=True)
class RankSummary:
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    scored_count: int
    total_count: int
    score_coverage: float
    top_1_accuracy: float
    top_2_accuracy: float
    top_3_accuracy: float
    mean_gold_rank: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    metric: str
    value: float | int | str
    limit: float | int | str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: eval report must be a JSON object")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError(f"{path}: missing summary object")
    return payload


def parse_rank_summary(
    source: Path,
    metrics: dict[str, Any],
    *,
    scope: str,
    dataset: str,
    split: str,
    engine: str,
) -> RankSummary:
    return RankSummary(
        source=str(source),
        scope=scope,
        dataset=dataset,
        split=split,
        engine=engine,
        scored_count=as_int(metrics.get("scored_count")),
        total_count=as_int(metrics.get("total_count")),
        score_coverage=as_float(metrics.get("score_coverage")),
        top_1_accuracy=as_float(metrics.get("top_1_accuracy")),
        top_2_accuracy=as_float(metrics.get("top_2_accuracy")),
        top_3_accuracy=as_float(metrics.get("top_3_accuracy")),
        mean_gold_rank=as_float(metrics.get("mean_gold_rank")),
        mean_reciprocal_rank=as_float(metrics.get("mean_reciprocal_rank")),
    )


def collect_summaries(path: Path, payload: dict[str, Any]) -> list[RankSummary]:
    summaries: list[RankSummary] = []
    summary = payload["summary"]
    for engine in ("holyc", "llama"):
        key = f"{engine}_rank_metrics"
        metrics = summary.get(key)
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: missing summary.{key}")
        summaries.append(
            parse_rank_summary(
                path,
                metrics,
                scope="overall",
                dataset=str(payload.get("dataset") or ""),
                split=str(payload.get("split") or ""),
                engine=engine,
            )
        )

    for item in summary.get("dataset_breakdown") or []:
        if not isinstance(item, dict):
            continue
        for engine in ("holyc", "llama"):
            metrics = item.get(f"{engine}_rank_metrics")
            if isinstance(metrics, dict):
                summaries.append(
                    parse_rank_summary(
                        path,
                        metrics,
                        scope="dataset",
                        dataset=str(item.get("dataset") or ""),
                        split=str(item.get("split") or ""),
                        engine=engine,
                    )
                )
    return summaries


def add_min_finding(
    findings: list[Finding],
    summary: RankSummary,
    metric: str,
    value: float | int,
    limit: float | int | None,
) -> None:
    if limit is None or value >= limit:
        return
    findings.append(
        Finding(
            "error",
            summary.source,
            summary.scope,
            summary.dataset,
            summary.split,
            summary.engine,
            metric,
            value,
            limit,
            f"{summary.engine} {summary.scope} {metric} {value} is below limit {limit}",
        )
    )


def audit_summaries(
    summaries: list[RankSummary],
    *,
    min_score_coverage: float | None,
    min_scored_count: int | None,
    min_top_1_accuracy: float | None,
    min_top_2_accuracy: float | None,
    min_top_3_accuracy: float | None,
    min_mean_reciprocal_rank: float | None,
    max_holyc_top_1_loss: float | None,
    max_holyc_mrr_loss: float | None,
    include_dataset_breakdown: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    filtered = [item for item in summaries if include_dataset_breakdown or item.scope == "overall"]
    by_scope: dict[tuple[str, str, str, str], dict[str, RankSummary]] = {}
    for summary in filtered:
        by_scope.setdefault((summary.source, summary.scope, summary.dataset, summary.split), {})[summary.engine] = summary
        add_min_finding(findings, summary, "score_coverage", summary.score_coverage, min_score_coverage)
        add_min_finding(findings, summary, "scored_count", summary.scored_count, min_scored_count)
        add_min_finding(findings, summary, "top_1_accuracy", summary.top_1_accuracy, min_top_1_accuracy)
        add_min_finding(findings, summary, "top_2_accuracy", summary.top_2_accuracy, min_top_2_accuracy)
        add_min_finding(findings, summary, "top_3_accuracy", summary.top_3_accuracy, min_top_3_accuracy)
        add_min_finding(
            findings,
            summary,
            "mean_reciprocal_rank",
            summary.mean_reciprocal_rank,
            min_mean_reciprocal_rank,
        )

    for engines in by_scope.values():
        holyc = engines.get("holyc")
        llama = engines.get("llama")
        if holyc is None or llama is None:
            continue
        if max_holyc_top_1_loss is not None:
            loss = llama.top_1_accuracy - holyc.top_1_accuracy
            if loss > max_holyc_top_1_loss:
                findings.append(
                    Finding(
                        "error",
                        holyc.source,
                        holyc.scope,
                        holyc.dataset,
                        holyc.split,
                        "holyc",
                        "top_1_accuracy_loss_vs_llama",
                        loss,
                        max_holyc_top_1_loss,
                        f"HolyC top-1 accuracy loss {loss} is above llama.cpp limit {max_holyc_top_1_loss}",
                    )
                )
        if max_holyc_mrr_loss is not None:
            loss = llama.mean_reciprocal_rank - holyc.mean_reciprocal_rank
            if loss > max_holyc_mrr_loss:
                findings.append(
                    Finding(
                        "error",
                        holyc.source,
                        holyc.scope,
                        holyc.dataset,
                        holyc.split,
                        "holyc",
                        "mean_reciprocal_rank_loss_vs_llama",
                        loss,
                        max_holyc_mrr_loss,
                        f"HolyC MRR loss {loss} is above llama.cpp limit {max_holyc_mrr_loss}",
                    )
                )
    return findings


def audit_reports(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[RankSummary] = []
    findings: list[Finding] = []
    for path in paths:
        try:
            summaries.extend(collect_summaries(path, load_report(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding("error", str(path), "file", "", "", "", "unreadable_report", "", "", str(exc)))

    findings.extend(
        audit_summaries(
            summaries,
            min_score_coverage=args.min_score_coverage,
            min_scored_count=args.min_scored_count,
            min_top_1_accuracy=args.min_top_1_accuracy,
            min_top_2_accuracy=args.min_top_2_accuracy,
            min_top_3_accuracy=args.min_top_3_accuracy,
            min_mean_reciprocal_rank=args.min_mean_reciprocal_rank,
            max_holyc_top_1_loss=args.max_holyc_top_1_loss,
            max_holyc_mrr_loss=args.max_holyc_mrr_loss,
            include_dataset_breakdown=args.include_dataset_breakdown,
        )
    )
    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "inputs": [str(path) for path in paths],
        "include_dataset_breakdown": args.include_dataset_breakdown,
        "rank_summary_count": len(summaries),
        "error_count": error_count,
        "rank_summaries": [asdict(summary) for summary in summaries],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Rank Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Inputs: {len(payload['inputs'])}",
        f"- Rank summaries: {payload['rank_summary_count']}",
        f"- Findings: {len(payload['findings'])}",
        "",
    ]
    if payload["findings"]:
        lines.extend(["## Findings", ""])
        for finding in payload["findings"]:
            lines.append(
                f"- `{finding['metric']}` {finding['source']} {finding['scope']} "
                f"{finding['dataset']} {finding['split']} {finding['engine']}: {finding['message']}"
            )
    else:
        lines.append("No rank gate findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = 1 if payload["status"] == "fail" else 0
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_rank_audit",
            "tests": "1",
            "failures": str(failures),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "rank_gates"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{len(payload['findings'])} rank finding(s)"})
        failure.text = "\n".join(finding["message"] for finding in payload["findings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def non_negative(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="eval_compare JSON report(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_rank_audit_latest")
    parser.add_argument("--min-score-coverage", type=non_negative)
    parser.add_argument("--min-scored-count", type=int)
    parser.add_argument("--min-top-1-accuracy", type=non_negative)
    parser.add_argument("--min-top-2-accuracy", type=non_negative)
    parser.add_argument("--min-top-3-accuracy", type=non_negative)
    parser.add_argument("--min-mean-reciprocal-rank", type=non_negative)
    parser.add_argument("--max-holyc-top-1-loss", type=non_negative)
    parser.add_argument("--max-holyc-mrr-loss", type=non_negative)
    parser.add_argument("--include-dataset-breakdown", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    if args.min_scored_count is not None and args.min_scored_count < 0:
        print("--min-scored-count must be non-negative", file=sys.stderr)
        return 2

    payload = audit_reports(args.reports, args)
    output_dir = args.output_dir
    stem = args.output_stem
    write_json(output_dir / f"{stem}.json", payload)
    write_markdown(output_dir / f"{stem}.md", payload)
    write_csv(
        output_dir / f"{stem}.csv",
        payload["findings"],
        ["severity", "source", "scope", "dataset", "split", "engine", "metric", "value", "limit", "message"],
    )
    write_csv(
        output_dir / f"{stem}_summaries.csv",
        payload["rank_summaries"],
        [
            "source",
            "scope",
            "dataset",
            "split",
            "engine",
            "scored_count",
            "total_count",
            "score_coverage",
            "top_1_accuracy",
            "top_2_accuracy",
            "top_3_accuracy",
            "mean_gold_rank",
            "mean_reciprocal_rank",
        ],
    )
    write_junit(output_dir / f"{stem}_junit.xml", payload)
    print(json.dumps({"status": payload["status"], "findings": len(payload["findings"])}, sort_keys=True))
    return 2 if args.fail_on_findings and payload["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
