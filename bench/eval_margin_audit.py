#!/usr/bin/env python3
"""Audit eval_compare score-margin metrics for HolyC vs llama.cpp reports.

The audit reads existing `eval_compare.py` JSON artifacts and checks scored
multiple-choice margin coverage, weak top-1 margins, and HolyC-vs-llama margin
loss. It is host-side only and never launches QEMU or touches the TempleOS
guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarginSummary:
    source: str
    scope: str
    dataset: str
    split: str
    engine: str
    scored_count: int
    total_count: int
    score_coverage: float
    mean_margin: float
    median_margin: float
    p10_margin: float
    min_margin: float
    low_margin_count: int
    low_margin_rate: float
    low_margin_threshold: float


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
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
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: missing summary object")
    return payload


def margin_summary(
    source: Path,
    metrics: dict[str, Any],
    *,
    engine: str,
    scope: str,
    dataset: str,
    split: str,
) -> MarginSummary:
    return MarginSummary(
        source=str(source),
        scope=scope,
        dataset=dataset,
        split=split,
        engine=engine,
        scored_count=parse_int(metrics.get("scored_count")),
        total_count=parse_int(metrics.get("total_count")),
        score_coverage=parse_float(metrics.get("score_coverage")),
        mean_margin=parse_float(metrics.get("mean_margin")),
        median_margin=parse_float(metrics.get("median_margin")),
        p10_margin=parse_float(metrics.get("p10_margin")),
        min_margin=parse_float(metrics.get("min_margin")),
        low_margin_count=parse_int(metrics.get("low_margin_count")),
        low_margin_rate=parse_float(metrics.get("low_margin_rate")),
        low_margin_threshold=parse_float(metrics.get("low_margin_threshold")),
    )


def collect_summaries(path: Path, payload: dict[str, Any]) -> list[MarginSummary]:
    summaries: list[MarginSummary] = []
    root_summary = payload["summary"]
    for engine in ("holyc", "llama"):
        key = f"{engine}_margin_metrics"
        metrics = root_summary.get(key)
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: missing summary.{key}")
        summaries.append(
            margin_summary(
                path,
                metrics,
                engine=engine,
                scope="overall",
                dataset=str(payload.get("dataset") or ""),
                split=str(payload.get("split") or ""),
            )
        )

    for item in root_summary.get("dataset_breakdown") or []:
        if not isinstance(item, dict):
            continue
        for engine in ("holyc", "llama"):
            key = f"{engine}_margin_metrics"
            metrics = item.get(key)
            if isinstance(metrics, dict):
                summaries.append(
                    margin_summary(
                        path,
                        metrics,
                        engine=engine,
                        scope="dataset",
                        dataset=str(item.get("dataset") or ""),
                        split=str(item.get("split") or ""),
                    )
                )
    return summaries


def append_threshold_finding(
    findings: list[Finding],
    summary: MarginSummary,
    metric: str,
    value: float | int,
    limit: float | int | None,
    *,
    comparison: str,
) -> None:
    if limit is None:
        return
    failed = value < limit if comparison == "min" else value > limit
    if not failed:
        return
    comparator = "below" if comparison == "min" else "above"
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
            f"{summary.engine} {summary.scope} {metric} {value} is {comparator} limit {limit}",
        )
    )


def audit_summaries(
    summaries: list[MarginSummary],
    *,
    min_score_coverage: float | None,
    min_scored_count: int | None,
    min_mean_margin: float | None,
    min_p10_margin: float | None,
    min_min_margin: float | None,
    max_low_margin_rate: float | None,
    max_holyc_mean_margin_loss: float | None,
    max_holyc_p10_margin_loss: float | None,
    include_dataset_breakdown: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    filtered = [item for item in summaries if include_dataset_breakdown or item.scope == "overall"]
    by_scope: dict[tuple[str, str, str, str], dict[str, MarginSummary]] = {}
    for summary in filtered:
        by_scope.setdefault((summary.source, summary.scope, summary.dataset, summary.split), {})[summary.engine] = summary
        append_threshold_finding(
            findings, summary, "score_coverage", summary.score_coverage, min_score_coverage, comparison="min"
        )
        append_threshold_finding(
            findings, summary, "scored_count", summary.scored_count, min_scored_count, comparison="min"
        )
        append_threshold_finding(
            findings, summary, "mean_margin", summary.mean_margin, min_mean_margin, comparison="min"
        )
        append_threshold_finding(findings, summary, "p10_margin", summary.p10_margin, min_p10_margin, comparison="min")
        append_threshold_finding(findings, summary, "min_margin", summary.min_margin, min_min_margin, comparison="min")
        append_threshold_finding(
            findings, summary, "low_margin_rate", summary.low_margin_rate, max_low_margin_rate, comparison="max"
        )

    for (_, _, _, _), engines in by_scope.items():
        holyc = engines.get("holyc")
        llama = engines.get("llama")
        if holyc is None or llama is None:
            continue
        if max_holyc_mean_margin_loss is not None:
            loss = llama.mean_margin - holyc.mean_margin
            if loss > max_holyc_mean_margin_loss:
                findings.append(
                    Finding(
                        "error",
                        holyc.source,
                        holyc.scope,
                        holyc.dataset,
                        holyc.split,
                        "holyc",
                        "mean_margin_loss_vs_llama",
                        loss,
                        max_holyc_mean_margin_loss,
                        f"HolyC mean margin loss {loss} is above llama.cpp limit {max_holyc_mean_margin_loss}",
                    )
                )
        if max_holyc_p10_margin_loss is not None:
            loss = llama.p10_margin - holyc.p10_margin
            if loss > max_holyc_p10_margin_loss:
                findings.append(
                    Finding(
                        "error",
                        holyc.source,
                        holyc.scope,
                        holyc.dataset,
                        holyc.split,
                        "holyc",
                        "p10_margin_loss_vs_llama",
                        loss,
                        max_holyc_p10_margin_loss,
                        f"HolyC p10 margin loss {loss} is above llama.cpp limit {max_holyc_p10_margin_loss}",
                    )
                )
    return findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Margin Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Reports: {payload['report_count']}",
        f"- Margin summaries: {payload['margin_summary_count']}",
        f"- Findings: {len(payload['findings'])}",
        "",
        "## Findings",
        "",
    ]
    if payload["findings"]:
        lines.extend(
            f"- {item['severity']}: {item['source']} {item['scope']} {item['engine']} {item['metric']} - {item['message']}"
            for item in payload["findings"]
        )
    else:
        lines.append("- No margin gate findings.")
    lines.extend(["", "## Summaries", ""])
    for item in payload["margin_summaries"]:
        lines.append(
            f"- {item['source']} {item['scope']} {item['engine']}: "
            f"coverage={item['score_coverage']:.4f} mean={item['mean_margin']:.4f} "
            f"p10={item['p10_margin']:.4f} low_rate={item['low_margin_rate']:.4f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_summaries_csv(path: Path, summaries: list[MarginSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MarginSummary.__dataclass_fields__))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def write_junit(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element("testsuite", name="holyc_eval_margin_audit", tests="1", failures="1" if findings else "0")
    case = ET.SubElement(suite, "testcase", name="margin_gates")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} margin finding(s)")
        failure.text = "\n".join(finding.message for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def bounded_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(f"{name} must be between 0 and 1")
    return parsed


def nonnegative_float(value: str, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError(f"{name} must be non-negative")
    return parsed


def positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"{name} must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="eval_compare JSON report(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_margin_audit_latest")
    parser.add_argument("--min-score-coverage", type=lambda value: bounded_float(value, "--min-score-coverage"))
    parser.add_argument("--min-scored-count", type=lambda value: positive_int(value, "--min-scored-count"))
    parser.add_argument("--min-mean-margin", type=lambda value: bounded_float(value, "--min-mean-margin"))
    parser.add_argument("--min-p10-margin", type=lambda value: bounded_float(value, "--min-p10-margin"))
    parser.add_argument("--min-min-margin", type=lambda value: bounded_float(value, "--min-min-margin"))
    parser.add_argument("--max-low-margin-rate", type=lambda value: bounded_float(value, "--max-low-margin-rate"))
    parser.add_argument(
        "--max-holyc-mean-margin-loss",
        type=lambda value: nonnegative_float(value, "--max-holyc-mean-margin-loss"),
    )
    parser.add_argument(
        "--max-holyc-p10-margin-loss",
        type=lambda value: nonnegative_float(value, "--max-holyc-p10-margin-loss"),
    )
    parser.add_argument("--include-dataset-breakdown", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reports = [(path, load_report(path)) for path in args.reports]
        summaries: list[MarginSummary] = []
        for path, payload in reports:
            summaries.extend(collect_summaries(path, payload))
        findings = audit_summaries(
            summaries,
            min_score_coverage=args.min_score_coverage,
            min_scored_count=args.min_scored_count,
            min_mean_margin=args.min_mean_margin,
            min_p10_margin=args.min_p10_margin,
            min_min_margin=args.min_min_margin,
            max_low_margin_rate=args.max_low_margin_rate,
            max_holyc_mean_margin_loss=args.max_holyc_mean_margin_loss,
            max_holyc_p10_margin_loss=args.max_holyc_p10_margin_loss,
            include_dataset_breakdown=args.include_dataset_breakdown,
        )
        payload = {
            "generated_at": iso_now(),
            "status": "fail" if findings else "pass",
            "report_count": len(reports),
            "reports": [{"path": str(path), "sha256": file_sha256(path)} for path, _ in reports],
            "margin_summary_count": len(summaries),
            "margin_summaries": [asdict(summary) for summary in summaries],
            "findings": [asdict(finding) for finding in findings],
        }
        output_dir = args.output_dir
        stem = args.output_stem
        write_json(output_dir / f"{stem}.json", payload)
        write_markdown(output_dir / f"{stem}.md", payload)
        write_csv(output_dir / f"{stem}.csv", findings)
        write_summaries_csv(output_dir / f"{stem}_summaries.csv", summaries)
        write_junit(output_dir / f"{stem}_junit.xml", findings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"eval_margin_audit: {exc}", file=sys.stderr)
        return 2
    return 2 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
