#!/usr/bin/env python3
"""Compute quality/throughput frontiers from eval perf scorecards.

This host-side tool consumes existing scorecard JSON artifacts only. It does
not launch QEMU, alter TempleOS images, or require network access.
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
from typing import Any, Iterable


@dataclass(frozen=True)
class FrontierRow:
    source: str
    cohort: str
    model: str
    quantization: str
    dataset: str
    split: str
    status: str
    records: int
    quality_metric: str
    quality_value: float | None
    speed_metric: str
    speed_value: float | None
    max_memory_bytes: int | None
    frontier: bool
    dominated_by: str


@dataclass(frozen=True)
class Finding:
    gate: str
    cohort: str
    model: str
    quantization: str
    value: float | int | str | None
    threshold: float | int | str | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int:
    number = as_float(value)
    return int(number) if number is not None else 0


def load_json(path: Path) -> tuple[Any | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"


def iter_json_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def iter_scorecard_rows(path: Path) -> Iterable[dict[str, Any]]:
    payload, error = load_json(path)
    if error or not isinstance(payload, dict):
        yield {
            "source": str(path),
            "status": "invalid",
            "model": "",
            "quantization": "",
            "dataset": "",
            "split": "",
            "error": error or "root must be object",
        }
        return
    rows = payload.get("scorecard")
    if not isinstance(rows, list):
        yield {
            "source": str(path),
            "status": "invalid",
            "model": str(payload.get("model") or ""),
            "quantization": str(payload.get("quantization") or ""),
            "dataset": str(payload.get("dataset") or ""),
            "split": str(payload.get("split") or ""),
            "error": "missing scorecard list",
        }
        return
    for row in rows:
        if isinstance(row, dict):
            merged = {"source": str(path)}
            merged.update(row)
            yield merged


def load_rows(paths: Iterable[Path], quality_metric: str, speed_metric: str) -> list[FrontierRow]:
    rows: list[FrontierRow] = []
    for path in iter_json_files(paths):
        for row in iter_scorecard_rows(path):
            dataset = str(row.get("dataset") or "")
            split = str(row.get("split") or "")
            model = str(row.get("model") or "")
            quantization = str(row.get("quantization") or "")
            rows.append(
                FrontierRow(
                    source=str(row.get("source") or path),
                    cohort=f"{dataset}/{split}",
                    model=model,
                    quantization=quantization,
                    dataset=dataset,
                    split=split,
                    status=str(row.get("status") or "pass").lower(),
                    records=as_int(row.get("records")),
                    quality_metric=quality_metric,
                    quality_value=as_float(row.get(quality_metric)),
                    speed_metric=speed_metric,
                    speed_value=as_float(row.get(speed_metric)),
                    max_memory_bytes=as_int(row.get("max_memory_bytes")) or None,
                    frontier=False,
                    dominated_by="",
                )
            )
    return rows


def row_key(row: FrontierRow) -> str:
    return f"{row.model}:{row.quantization}:{row.dataset}/{row.split}"


def dominates(candidate: FrontierRow, other: FrontierRow, *, memory_aware: bool = False) -> bool:
    if candidate.quality_value is None or candidate.speed_value is None:
        return False
    if other.quality_value is None or other.speed_value is None:
        return False
    if memory_aware and (candidate.max_memory_bytes is None or other.max_memory_bytes is None):
        return False
    memory_ok = not memory_aware or candidate.max_memory_bytes <= other.max_memory_bytes
    memory_better = memory_aware and candidate.max_memory_bytes < other.max_memory_bytes
    return (
        candidate.quality_value >= other.quality_value
        and candidate.speed_value >= other.speed_value
        and memory_ok
        and (candidate.quality_value > other.quality_value or candidate.speed_value > other.speed_value or memory_better)
    )


def mark_frontier(rows: list[FrontierRow], *, memory_aware: bool = False) -> list[FrontierRow]:
    by_cohort: dict[str, list[FrontierRow]] = {}
    for row in rows:
        by_cohort.setdefault(row.cohort, []).append(row)

    marked: list[FrontierRow] = []
    for cohort_rows in by_cohort.values():
        for row in cohort_rows:
            dominator = next(
                (candidate for candidate in cohort_rows if candidate is not row and dominates(candidate, row, memory_aware=memory_aware)),
                None,
            )
            marked.append(
                FrontierRow(
                    **{
                        **asdict(row),
                        "frontier": dominator is None and row.quality_value is not None and row.speed_value is not None,
                        "dominated_by": row_key(dominator) if dominator else "",
                    }
                )
            )
    return sorted(marked, key=lambda row: (row.cohort, not row.frontier, row.model, row.quantization))


def evaluate(rows: list[FrontierRow], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    frontier_rows = [row for row in rows if row.frontier]
    if len(rows) < args.min_rows:
        findings.append(Finding("min_rows", "", "", "", len(rows), args.min_rows, "too few scorecard rows"))
    if len(frontier_rows) < args.min_frontier_rows:
        findings.append(
            Finding("min_frontier_rows", "", "", "", len(frontier_rows), args.min_frontier_rows, "too few frontier rows")
        )
    for row in rows:
        if row.status == "invalid":
            findings.append(Finding("invalid_scorecard", row.cohort, row.model, row.quantization, row.status, "valid", "invalid scorecard input"))
        if args.fail_on_failed_scorecard and row.status != "pass":
            findings.append(Finding("scorecard_status", row.cohort, row.model, row.quantization, row.status, "pass", "scorecard row did not pass"))
        if args.fail_on_missing_metrics and (row.quality_value is None or row.speed_value is None):
            findings.append(
                Finding(
                    "missing_metric",
                    row.cohort,
                    row.model,
                    row.quantization,
                    "missing",
                    f"{row.quality_metric}+{row.speed_metric}",
                    "scorecard row is missing quality or speed metric",
                )
            )
        if args.memory_aware and args.fail_on_missing_metrics and row.max_memory_bytes is None:
            findings.append(
                Finding(
                    "missing_memory_metric",
                    row.cohort,
                    row.model,
                    row.quantization,
                    "missing",
                    "max_memory_bytes",
                    "memory-aware frontier row is missing max_memory_bytes",
                )
            )
    for required in args.require_frontier_quantization:
        if not any(row.quantization == required and row.frontier for row in frontier_rows):
            findings.append(
                Finding("require_frontier_quantization", "", "", required, "missing", required, f"{required} is absent from the frontier")
            )
    return findings


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    fieldnames = list(dictionaries[0]) if dictionaries else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dictionaries)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Efficiency Frontier",
        "",
        f"- Status: {payload['status']}",
        f"- Scorecard rows: {payload['summary']['rows']}",
        f"- Frontier rows: {payload['summary']['frontier_rows']}",
        f"- Dominated rows: {payload['summary']['dominated_rows']}",
        f"- Memory-aware: {payload['summary']['memory_aware']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
        "| cohort | model | quantization | quality | speed | max memory bytes | frontier | dominated by |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {cohort} | {model} | {quantization} | {quality_value} | {speed_value} | {max_memory_bytes} | {frontier} | {dominated_by} |".format(
                **{key: ("" if value is None else value) for key, value in row.items()}
            )
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in payload["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_efficiency_frontier",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "frontier_gates"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} frontier finding(s)"})
        failure.text = "\n".join(f"{finding.gate}: {finding.message}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scorecards", nargs="+", type=Path, help="eval_perf_scorecard JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_efficiency_frontier_latest")
    parser.add_argument("--quality-metric", default="holyc_accuracy")
    parser.add_argument("--speed-metric", default="median_wall_tok_per_s")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-frontier-rows", type=int, default=1)
    parser.add_argument("--require-frontier-quantization", action="append", default=[])
    parser.add_argument("--fail-on-failed-scorecard", action="store_true")
    parser.add_argument("--fail-on-missing-metrics", action="store_true")
    parser.add_argument(
        "--memory-aware",
        action="store_true",
        help="include max_memory_bytes as a lower-is-better Pareto dimension when marking dominated rows",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = mark_frontier(load_rows(args.scorecards, args.quality_metric, args.speed_metric), memory_aware=args.memory_aware)
    findings = evaluate(rows, args)
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "frontier_rows": sum(1 for row in rows if row.frontier),
            "dominated_rows": sum(1 for row in rows if row.dominated_by),
            "findings": len(findings),
            "quality_metric": args.quality_metric,
            "speed_metric": args.speed_metric,
            "memory_aware": args.memory_aware,
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(stem.with_suffix(".md"), payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
