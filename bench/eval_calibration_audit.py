#!/usr/bin/env python3
"""Audit eval_compare calibration metrics for HolyC vs llama.cpp reports.

The audit reads existing `eval_compare.py` JSON artifacts, checks score
coverage, calibration error, Brier score, and HolyC-vs-llama deltas, then writes
JSON, Markdown, CSV, and JUnit outputs. It is host-side only and never launches
QEMU or touches the TempleOS guest.
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
class EngineSummary:
    source: str
    engine: str
    dataset: str
    split: str
    model: str
    quantization: str
    scored_count: int
    total_count: int
    score_coverage: float
    accuracy_when_scored: float
    mean_confidence: float
    ece: float
    brier_score: float


@dataclass(frozen=True)
class Finding:
    severity: str
    source: str
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


def engine_summary(path: Path, payload: dict[str, Any], engine: str) -> EngineSummary:
    key = f"{engine}_calibration"
    metrics = payload.get("summary", {}).get(key)
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: missing summary.{key}")
    return EngineSummary(
        source=str(path),
        engine=engine,
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        scored_count=parse_int(metrics.get("scored_count")),
        total_count=parse_int(metrics.get("total_count")),
        score_coverage=parse_float(metrics.get("score_coverage")),
        accuracy_when_scored=parse_float(metrics.get("accuracy_when_scored")),
        mean_confidence=parse_float(metrics.get("mean_confidence")),
        ece=parse_float(metrics.get("ece")),
        brier_score=parse_float(metrics.get("brier_score")),
    )


def append_threshold_finding(
    findings: list[Finding],
    summary: EngineSummary,
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
            severity="error",
            source=summary.source,
            engine=summary.engine,
            metric=metric,
            value=value,
            limit=limit,
            message=f"{summary.engine} {metric} {value} is {comparator} limit {limit}",
        )
    )


def audit_summaries(
    summaries: list[EngineSummary],
    *,
    min_score_coverage: float | None,
    min_scored_count: int | None,
    min_accuracy_when_scored: float | None,
    max_ece: float | None,
    max_brier_score: float | None,
    max_holyc_ece_delta: float | None,
    max_holyc_brier_delta: float | None,
) -> list[Finding]:
    findings: list[Finding] = []
    by_source: dict[str, dict[str, EngineSummary]] = {}
    for summary in summaries:
        by_source.setdefault(summary.source, {})[summary.engine] = summary
        append_threshold_finding(
            findings, summary, "score_coverage", summary.score_coverage, min_score_coverage, comparison="min"
        )
        append_threshold_finding(
            findings, summary, "scored_count", summary.scored_count, min_scored_count, comparison="min"
        )
        append_threshold_finding(
            findings,
            summary,
            "accuracy_when_scored",
            summary.accuracy_when_scored,
            min_accuracy_when_scored,
            comparison="min",
        )
        append_threshold_finding(findings, summary, "ece", summary.ece, max_ece, comparison="max")
        append_threshold_finding(
            findings, summary, "brier_score", summary.brier_score, max_brier_score, comparison="max"
        )

    for source, engines in by_source.items():
        holyc = engines.get("holyc")
        llama = engines.get("llama")
        if holyc is None or llama is None:
            continue
        if max_holyc_ece_delta is not None:
            delta = holyc.ece - llama.ece
            if delta > max_holyc_ece_delta:
                findings.append(
                    Finding(
                        "error",
                        source,
                        "holyc",
                        "ece_delta_holyc_minus_llama",
                        delta,
                        max_holyc_ece_delta,
                        f"HolyC ECE delta {delta} is above llama.cpp limit {max_holyc_ece_delta}",
                    )
                )
        if max_holyc_brier_delta is not None:
            delta = holyc.brier_score - llama.brier_score
            if delta > max_holyc_brier_delta:
                findings.append(
                    Finding(
                        "error",
                        source,
                        "holyc",
                        "brier_delta_holyc_minus_llama",
                        delta,
                        max_holyc_brier_delta,
                        f"HolyC Brier delta {delta} is above llama.cpp limit {max_holyc_brier_delta}",
                    )
                )
    return findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Calibration Audit",
        "",
        f"Status: **{payload['status']}**",
        f"Reports: {payload['report_count']}",
        f"Engine summaries: {payload['engine_summary_count']}",
        f"Findings: {len(payload['findings'])}",
        "",
        "## Engine Metrics",
        "",
        "| Source | Engine | Scored | Coverage | Accuracy | Mean confidence | ECE | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in payload["engine_summaries"]:
        lines.append(
            "| {source} | {engine} | {scored_count}/{total_count} | {score_coverage:.4f} | "
            "{accuracy_when_scored:.4f} | {mean_confidence:.4f} | {ece:.4f} | {brier_score:.4f} |".format(
                **summary
            )
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            lines.append(f"- {finding['severity']}: {finding['message']} ({finding['source']})")
    else:
        lines.append("No calibration gate findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "source", "engine", "metric", "value", "limit", "message"])
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_calibration_audit",
            "tests": "1",
            "failures": "1" if payload["status"] == "fail" else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "eval_calibration_audit", "name": "calibration_gates"})
    if payload["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{len(payload['findings'])} calibration finding(s)"})
        failure.text = "\n".join(finding["message"] for finding in payload["findings"])
    ET.indent(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="eval_compare JSON report(s)")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_calibration_audit_latest")
    parser.add_argument("--min-score-coverage", type=float)
    parser.add_argument("--min-scored-count", type=int)
    parser.add_argument("--min-accuracy-when-scored", type=float)
    parser.add_argument("--max-ece", type=float)
    parser.add_argument("--max-brier-score", type=float)
    parser.add_argument("--max-holyc-ece-delta", type=float)
    parser.add_argument("--max-holyc-brier-delta", type=float)
    parser.add_argument("--fail-on-findings", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    percentage_gates = [
        ("--min-score-coverage", args.min_score_coverage),
        ("--min-accuracy-when-scored", args.min_accuracy_when_scored),
        ("--max-ece", args.max_ece),
        ("--max-brier-score", args.max_brier_score),
        ("--max-holyc-ece-delta", args.max_holyc_ece_delta),
        ("--max-holyc-brier-delta", args.max_holyc_brier_delta),
    ]
    for label, value in percentage_gates:
        if value is not None and value < 0:
            raise ValueError(f"{label} must be non-negative")
    if args.min_scored_count is not None and args.min_scored_count < 0:
        raise ValueError("--min-scored-count must be non-negative")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        payloads = [(path, load_report(path)) for path in args.reports]
        summaries = [
            engine_summary(path, payload, engine)
            for path, payload in payloads
            for engine in ("holyc", "llama")
        ]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"eval_calibration_audit: {exc}", file=sys.stderr)
        return 2

    findings = audit_summaries(
        summaries,
        min_score_coverage=args.min_score_coverage,
        min_scored_count=args.min_scored_count,
        min_accuracy_when_scored=args.min_accuracy_when_scored,
        max_ece=args.max_ece,
        max_brier_score=args.max_brier_score,
        max_holyc_ece_delta=args.max_holyc_ece_delta,
        max_holyc_brier_delta=args.max_holyc_brier_delta,
    )
    status = "fail" if findings else "pass"
    report = {
        "generated_at": iso_now(),
        "status": status,
        "report_count": len(payloads),
        "engine_summary_count": len(summaries),
        "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path, _payload in payloads],
        "gates": {
            "min_score_coverage": args.min_score_coverage,
            "min_scored_count": args.min_scored_count,
            "min_accuracy_when_scored": args.min_accuracy_when_scored,
            "max_ece": args.max_ece,
            "max_brier_score": args.max_brier_score,
            "max_holyc_ece_delta": args.max_holyc_ece_delta,
            "max_holyc_brier_delta": args.max_holyc_brier_delta,
        },
        "engine_summaries": [asdict(summary) for summary in summaries],
        "findings": [asdict(finding) for finding in findings],
    }

    output_dir = args.output_dir
    stem = args.output_stem
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}.csv"
    junit_path = output_dir / f"{stem}_junit.xml"
    write_json(json_path, report)
    write_markdown(markdown_path, report)
    write_csv(csv_path, findings)
    write_junit(junit_path, report)

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={markdown_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_junit={junit_path}")
    return 2 if findings and args.fail_on_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
