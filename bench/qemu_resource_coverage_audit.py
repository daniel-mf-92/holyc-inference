#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for resource telemetry coverage.

This host-side tool reads saved benchmark artifacts only. It never launches QEMU
and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
DEFAULT_REQUIRED_METRICS = (
    "memory_bytes",
    "memory_bytes_per_token",
    "host_child_peak_rss_bytes",
    "host_child_cpu_us",
    "host_child_cpu_pct",
    "host_child_tok_per_cpu_s",
)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class ResourceRecord:
    source: str
    row: int
    prompt: str
    profile: str
    model: str
    quantization: str
    commit: str
    phase: str
    exit_class: str
    tokens: int | None
    metric_count: int
    missing_metrics: str
    invalid_metrics: str
    memory_bytes: float | None
    memory_bytes_per_token: float | None
    host_child_peak_rss_bytes: float | None
    host_child_cpu_us: float | None
    host_child_cpu_pct: float | None
    host_child_tok_per_cpu_s: float | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    prompt: str
    metric: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def iter_input_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            seen: set[Path] = set()
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file():
            yield path


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS and key != "warmups"}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    yield merged
    if not yielded:
        yield payload


def load_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield from flatten_json_payload(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def resource_record(source: Path, row_number: int, row: dict[str, Any], required_metrics: tuple[str, ...]) -> tuple[ResourceRecord, list[Finding]]:
    prompt = row_text(row, "prompt", "prompt_id")
    missing: list[str] = []
    invalid: list[str] = []
    findings: list[Finding] = []
    values: dict[str, float | None] = {}
    for metric in required_metrics:
        value = finite_float(row.get(metric))
        values[metric] = value
        if metric not in row or row.get(metric) in (None, ""):
            missing.append(metric)
            findings.append(Finding(str(source), row_number, "error", "missing_metric", prompt, metric, "required resource metric is absent"))
        elif value is None:
            invalid.append(metric)
            findings.append(Finding(str(source), row_number, "error", "invalid_metric", prompt, metric, "metric must be finite numeric telemetry"))
        elif value < 0:
            invalid.append(metric)
            findings.append(Finding(str(source), row_number, "error", "negative_metric", prompt, metric, "metric must be non-negative"))

    tokens = finite_int(row.get("tokens"))
    memory = values.get("memory_bytes")
    memory_per_token = values.get("memory_bytes_per_token")
    if tokens is not None and tokens > 0 and memory is not None and memory_per_token is not None:
        expected = memory / tokens
        tolerance = max(1.0, abs(expected) * 0.001)
        if abs(memory_per_token - expected) > tolerance:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "memory_per_token_drift",
                    prompt,
                    "memory_bytes_per_token",
                    f"expected about {expected:.6f} from memory_bytes/tokens",
                )
            )

    return (
        ResourceRecord(
            source=str(source),
            row=row_number,
            prompt=prompt,
            profile=row_text(row, "profile"),
            model=row_text(row, "model"),
            quantization=row_text(row, "quantization"),
            commit=row_text(row, "commit"),
            phase=row_text(row, "phase", default="measured"),
            exit_class=row_text(row, "exit_class"),
            tokens=tokens,
            metric_count=sum(1 for metric in required_metrics if values.get(metric) is not None),
            missing_metrics=";".join(missing),
            invalid_metrics=";".join(invalid),
            memory_bytes=values.get("memory_bytes"),
            memory_bytes_per_token=values.get("memory_bytes_per_token"),
            host_child_peak_rss_bytes=values.get("host_child_peak_rss_bytes"),
            host_child_cpu_us=values.get("host_child_cpu_us"),
            host_child_cpu_pct=values.get("host_child_cpu_pct"),
            host_child_tok_per_cpu_s=values.get("host_child_tok_per_cpu_s"),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ResourceRecord], list[Finding]]:
    records: list[ResourceRecord] = []
    findings: list[Finding] = []
    required_metrics = tuple(args.require_metric)
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "-", "-", str(exc)))
            continue
        for row_number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            phase = row_text(row, "phase", default="measured")
            exit_class = row_text(row, "exit_class")
            if args.measured_only and phase == "warmup":
                continue
            if args.ok_only and exit_class not in {"ok", "-"}:
                continue
            record, row_findings = resource_record(path, row_number, row, required_metrics)
            records.append(record)
            findings.extend(row_findings)

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "-", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(records) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "-", "rows", f"found {len(records)}, expected at least {args.min_rows}"))
    return records, findings


def summary(records: list[ResourceRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "findings": len(findings),
        "rows_with_all_metrics": sum(1 for record in records if not record.missing_metrics and not record.invalid_metrics),
        "missing_metric_findings": sum(1 for finding in findings if finding.kind == "missing_metric"),
        "invalid_metric_findings": sum(1 for finding in findings if finding.kind in {"invalid_metric", "negative_metric"}),
        "memory_per_token_drift_findings": sum(1 for finding in findings if finding.kind == "memory_per_token_drift"),
        "profiles": sorted({record.profile for record in records if record.profile != "-"}),
        "models": sorted({record.model for record in records if record.model != "-"}),
        "quantizations": sorted({record.quantization for record in records if record.quantization != "-"}),
    }


def write_json(path: Path, records: list[ResourceRecord], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(records, findings),
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ResourceRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Resource Coverage Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(records)}",
        f"Findings: {len(findings)}",
        "",
        "## Coverage",
        "",
        "| Source | Row | Prompt | Profile | Model | Quantization | Metrics | Missing | Invalid |",
        "| --- | ---: | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.source} | {record.row} | {record.prompt} | {record.profile} | {record.model} | "
            f"{record.quantization} | {record.metric_count} | {record.missing_metrics or '-'} | {record.invalid_metrics or '-'} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.detail} |")
    else:
        lines.append("No resource telemetry coverage findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[ResourceRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ResourceRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_resource_coverage_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "resource_telemetry_coverage"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} resource telemetry finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_resource_coverage_audit_latest")
    parser.add_argument("--require-metric", action="append", default=[], help="Required resource metric; repeatable")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--all-phases", dest="measured_only", action="store_false", help="Include warmup rows")
    parser.add_argument("--all-exit-classes", dest="ok_only", action="store_false", help="Include non-ok rows")
    parser.set_defaults(measured_only=True, ok_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if not args.require_metric:
        args.require_metric = list(DEFAULT_REQUIRED_METRICS)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")

    records, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", records, findings)
    write_markdown(args.output_dir / f"{stem}.md", records, findings)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
