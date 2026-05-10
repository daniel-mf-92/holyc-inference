#!/usr/bin/env python3
"""Audit benchmark artifacts against absolute performance SLO gates.

This host-side tool reads existing qemu_prompt_bench JSON/JSONL/CSV artifacts.
It does not launch QEMU or touch the TempleOS guest.
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


RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
LOWER_IS_BETTER = {
    "us_per_token",
    "wall_us_per_token",
    "ttft_us",
    "memory_bytes",
    "memory_bytes_per_token",
    "host_child_peak_rss_bytes",
}
HIGHER_IS_BETTER = {"tok_per_s", "wall_tok_per_s"}


@dataclass(frozen=True)
class BenchRow:
    source: str
    row: int
    prompt: str
    profile: str
    model: str
    quantization: str
    commit: str
    phase: str
    exit_class: str
    timed_out: bool
    returncode: int | None
    failure_reason: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    prompt: str
    metric: str
    value: float | None
    threshold: float | None
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return

    if not isinstance(payload, dict):
        return

    yielded = False
    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS and key != "warmups"}
    for key in RESULT_KEYS:
        nested = payload.get(key)
        if isinstance(nested, list):
            yielded = True
            for item in nested:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    yield merged

    if not yielded:
        yield payload


def load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from flatten_json_payload(payload)


def load_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            yield from flatten_json_payload(payload)


def load_csv_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".csv"}
            )
        elif path.suffix.lower() in {".json", ".jsonl", ".csv"}:
            yield path


def row_identity(raw: dict[str, Any], fallback_source: str, fallback_row: int) -> BenchRow:
    metrics = {
        metric: value
        for metric in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)
        if (value := as_float(raw.get(metric))) is not None
    }
    return BenchRow(
        source=str(raw.get("source") or fallback_source),
        row=as_int(raw.get("row")) or fallback_row,
        prompt=str(raw.get("prompt") or raw.get("prompt_id") or "-"),
        profile=str(raw.get("profile") or "-"),
        model=str(raw.get("model") or "-"),
        quantization=str(raw.get("quantization") or "-"),
        commit=str(raw.get("commit") or "-"),
        phase=str(raw.get("phase") or "measured"),
        exit_class=str(raw.get("exit_class") or "-"),
        timed_out=as_bool(raw.get("timed_out")),
        returncode=as_int(raw.get("returncode")),
        failure_reason=str(raw.get("failure_reason") or ""),
        metrics=metrics,
    )


def load_rows(paths: Iterable[Path]) -> list[BenchRow]:
    rows: list[BenchRow] = []
    for path in sorted(iter_input_files(paths)):
        suffix = path.suffix.lower()
        if suffix == ".json":
            loader = load_json_records
        elif suffix == ".jsonl":
            loader = load_jsonl_records
        else:
            loader = load_csv_records
        for row_number, raw in enumerate(loader(path), 1):
            if not isinstance(raw, dict):
                continue
            bench_row = row_identity(raw, str(path), row_number)
            if bench_row.phase == "warmup":
                continue
            rows.append(bench_row)
    return rows


def metric_finding(row: BenchRow, metric: str, value: float, threshold: float, detail: str) -> Finding:
    return Finding(
        source=row.source,
        row=row.row,
        severity="error",
        kind="slo_violation",
        prompt=row.prompt,
        metric=metric,
        value=value,
        threshold=threshold,
        detail=detail,
    )


def evaluate(rows: list[BenchRow], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    thresholds: dict[str, float] = {
        "tok_per_s": args.min_tok_per_s,
        "wall_tok_per_s": args.min_wall_tok_per_s,
        "us_per_token": args.max_us_per_token,
        "wall_us_per_token": args.max_wall_us_per_token,
        "ttft_us": args.max_ttft_us,
        "memory_bytes": args.max_memory_bytes,
        "memory_bytes_per_token": args.max_memory_bytes_per_token,
        "host_child_peak_rss_bytes": args.max_host_child_peak_rss_bytes,
    }

    for row in rows:
        if args.require_success:
            failed = row.timed_out or row.exit_class not in {"-", "ok"} or (
                row.returncode is not None and row.returncode != 0
            )
            if failed:
                findings.append(
                    Finding(
                        source=row.source,
                        row=row.row,
                        severity="error",
                        kind="run_failure",
                        prompt=row.prompt,
                        metric="success",
                        value=None,
                        threshold=None,
                        detail=(
                            f"exit_class={row.exit_class} returncode={row.returncode} "
                            f"timed_out={row.timed_out} failure_reason={row.failure_reason or '-'}"
                        ),
                    )
                )

        for metric, threshold in thresholds.items():
            if threshold is None:
                continue
            value = row.metrics.get(metric)
            if value is None:
                findings.append(
                    Finding(
                        source=row.source,
                        row=row.row,
                        severity="error",
                        kind="missing_metric",
                        prompt=row.prompt,
                        metric=metric,
                        value=None,
                        threshold=threshold,
                        detail=f"missing required metric {metric}",
                    )
                )
                continue
            if metric in HIGHER_IS_BETTER and value < threshold:
                findings.append(metric_finding(row, metric, value, threshold, f"{metric} {value} < {threshold}"))
            if metric in LOWER_IS_BETTER and value > threshold:
                findings.append(metric_finding(row, metric, value, threshold, f"{metric} {value} > {threshold}"))

    if args.max_failure_pct is not None:
        failed_rows = [
            row
            for row in rows
            if row.timed_out or row.exit_class not in {"-", "ok"} or (row.returncode is not None and row.returncode != 0)
        ]
        failure_pct = round((len(failed_rows) / len(rows) * 100.0), 6) if rows else 0.0
        if failure_pct > args.max_failure_pct:
            findings.append(
                Finding(
                    source="suite",
                    row=0,
                    severity="error",
                    kind="failure_rate",
                    prompt="-",
                    metric="failure_pct",
                    value=failure_pct,
                    threshold=args.max_failure_pct,
                    detail=f"failure_pct {failure_pct} > {args.max_failure_pct}",
                )
            )

    if args.min_rows is not None and len(rows) < args.min_rows:
        findings.append(
            Finding(
                source="suite",
                row=0,
                severity="error",
                kind="coverage",
                prompt="-",
                metric="rows",
                value=float(len(rows)),
                threshold=float(args.min_rows),
                detail=f"measured rows {len(rows)} < {args.min_rows}",
            )
        )

    return findings


def metric_stats(rows: list[BenchRow], metric: str) -> dict[str, float | int | None]:
    values = sorted(row.metrics[metric] for row in rows if metric in row.metrics)
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": round(sum(values) / len(values), 6),
    }


def summarize(rows: list[BenchRow], findings: list[Finding]) -> dict[str, Any]:
    failed_rows = [
        row
        for row in rows
        if row.timed_out or row.exit_class not in {"-", "ok"} or (row.returncode is not None and row.returncode != 0)
    ]
    return {
        "rows": len(rows),
        "failed_rows": len(failed_rows),
        "failure_pct": round((len(failed_rows) / len(rows) * 100.0), 6) if rows else 0.0,
        "findings": len(findings),
        "profiles": sorted({row.profile for row in rows}),
        "models": sorted({row.model for row in rows}),
        "quantizations": sorted({row.quantization for row in rows}),
        "commits": sorted({row.commit for row in rows}),
        "metrics": {metric: metric_stats(rows, metric) for metric in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)},
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Perf SLO Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Rows checked: {summary['rows']}",
        f"Failed rows: {summary['failed_rows']} ({summary['failure_pct']}%)",
        f"Findings: {summary['findings']}",
        "",
        "## Metrics",
        "",
        "| Metric | Count | Min | Mean | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, stats in summary["metrics"].items():
        lines.append(
            f"| {metric} | {stats['count']} | {stats['min']} | {stats['mean']} | {stats['max']} |"
        )

    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No findings.")
    else:
        lines.extend(["| Source | Row | Kind | Prompt | Metric | Value | Threshold | Detail |", "| --- | ---: | --- | --- | --- | ---: | ---: | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {source} | {row} | {kind} | {prompt} | {metric} | {value} | {threshold} | {detail} |".format(
                    **finding
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "row", "severity", "kind", "prompt", "metric", "value", "threshold", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_perf_slo_audit",
            "tests": "1",
            "failures": str(len(findings)),
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "perf_slo_audit"})
    for finding in findings:
        failure = ET.SubElement(
            case,
            "failure",
            {"type": finding.kind, "message": f"{finding.metric}: {finding.detail}"},
        )
        failure.text = json.dumps(asdict(finding), sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    junit_path = args.output_dir / f"{args.output_stem}_junit.xml"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(csv_path, [Finding(**finding) for finding in report["findings"]])
    write_junit(junit_path, [Finding(**finding) for finding in report["findings"]])
    return json_path, md_path, csv_path, junit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-stem", default="perf_slo_audit_latest")
    parser.add_argument("--min-rows", type=int)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--max-failure-pct", type=float)
    parser.add_argument("--min-tok-per-s", type=float)
    parser.add_argument("--min-wall-tok-per-s", type=float)
    parser.add_argument("--max-us-per-token", type=float)
    parser.add_argument("--max-wall-us-per-token", type=float)
    parser.add_argument("--max-ttft-us", type=float)
    parser.add_argument("--max-memory-bytes", type=float)
    parser.add_argument("--max-memory-bytes-per-token", type=float)
    parser.add_argument("--max-host-child-peak-rss-bytes", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(args.inputs)
    findings = evaluate(rows, args)
    report = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "inputs": [str(path) for path in args.inputs],
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if key.startswith(("min_", "max_")) or key == "require_success"
        },
        "summary": summarize(rows, findings),
        "findings": [asdict(finding) for finding in findings],
    }
    json_path, md_path, csv_path, junit_path = write_outputs(report, args)
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_junit={junit_path}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
