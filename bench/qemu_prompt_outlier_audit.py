#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for per-prompt run outliers.

This host-side tool reads existing qemu_prompt_bench JSON/JSONL/CSV artifacts.
It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
AUDIT_METRICS = ("wall_elapsed_us", "wall_tok_per_s", "host_overhead_pct")


@dataclass(frozen=True)
class Sample:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    iteration: int | None
    wall_elapsed_us: float | None
    wall_tok_per_s: float | None
    host_overhead_pct: float | None


@dataclass(frozen=True)
class Group:
    profile: str
    model: str
    quantization: str
    prompt: str
    samples: int
    ok_samples: int
    wall_elapsed_us_median: float | None
    wall_elapsed_us_mad: float | None
    wall_tok_per_s_median: float | None
    wall_tok_per_s_mad: float | None
    host_overhead_pct_median: float | None
    host_overhead_pct_mad: float | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    group: str
    metric: str
    value: float | None
    median: float | None
    limit: float | None
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finite_int(value: Any) -> int | None:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


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
                if stripped:
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


def group_key(sample: Sample) -> tuple[str, str, str, str]:
    return (sample.profile, sample.model, sample.quantization, sample.prompt)


def group_label(key: tuple[str, str, str, str]) -> str:
    return "/".join(key)


def metric_values(samples: list[Sample], metric: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = getattr(sample, metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def median_absolute_deviation(values: list[float]) -> float | None:
    if not values:
        return None
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def relative_delta_pct(value: float, median: float) -> float:
    if median == 0:
        return 0.0 if value == 0 else math.inf
    return abs(value - median) * 100.0 / abs(median)


def sample_from_row(source: Path, row_number: int, row: dict[str, Any], require_metrics: set[str]) -> tuple[Sample, list[Finding]]:
    profile = row_text(row, "profile")
    model = row_text(row, "model")
    quantization = row_text(row, "quantization")
    prompt = row_text(row, "prompt", "prompt_id")
    key = (profile, model, quantization, prompt)
    group = group_label(key)
    findings: list[Finding] = []
    metrics = {metric: finite_float(row.get(metric)) for metric in AUDIT_METRICS}
    for metric in sorted(require_metrics):
        if metric not in row or row.get(metric) in (None, ""):
            findings.append(Finding(str(source), row_number, "error", "missing_metric", group, metric, None, None, None, "required outlier metric is absent"))
        elif metrics[metric] is None:
            findings.append(Finding(str(source), row_number, "error", "invalid_metric", group, metric, None, None, None, "required outlier metric must be finite numeric telemetry"))

    return (
        Sample(
            source=str(source),
            row=row_number,
            profile=profile,
            model=model,
            quantization=quantization,
            prompt=prompt,
            phase=row_text(row, "phase", default="measured"),
            exit_class=row_text(row, "exit_class"),
            iteration=finite_int(row.get("iteration")),
            wall_elapsed_us=metrics["wall_elapsed_us"],
            wall_tok_per_s=metrics["wall_tok_per_s"],
            host_overhead_pct=metrics["host_overhead_pct"],
        ),
        findings,
    )


def build_group(key: tuple[str, str, str, str], samples: list[Sample]) -> Group:
    values = {metric: metric_values(samples, metric) for metric in AUDIT_METRICS}
    return Group(
        profile=key[0],
        model=key[1],
        quantization=key[2],
        prompt=key[3],
        samples=len(samples),
        ok_samples=sum(1 for sample in samples if sample.exit_class == "ok"),
        wall_elapsed_us_median=statistics.median(values["wall_elapsed_us"]) if values["wall_elapsed_us"] else None,
        wall_elapsed_us_mad=median_absolute_deviation(values["wall_elapsed_us"]),
        wall_tok_per_s_median=statistics.median(values["wall_tok_per_s"]) if values["wall_tok_per_s"] else None,
        wall_tok_per_s_mad=median_absolute_deviation(values["wall_tok_per_s"]),
        host_overhead_pct_median=statistics.median(values["host_overhead_pct"]) if values["host_overhead_pct"] else None,
        host_overhead_pct_mad=median_absolute_deviation(values["host_overhead_pct"]),
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[Sample], list[Group], list[Finding]]:
    samples: list[Sample] = []
    findings: list[Finding] = []
    groups_by_key: dict[tuple[str, str, str, str], list[Sample]] = {}
    require_metrics = set(args.require_metric)

    for source in iter_input_files(paths, args.pattern):
        for row_number, row in enumerate(load_rows(source), 1):
            sample, sample_findings = sample_from_row(source, row_number, row, require_metrics)
            findings.extend(sample_findings)
            if sample.phase != "measured" or sample.exit_class != "ok":
                continue
            samples.append(sample)
            groups_by_key.setdefault(group_key(sample), []).append(sample)

    if len(samples) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "-", "rows", float(len(samples)), None, float(args.min_rows), "not enough measured OK rows"))

    groups = [build_group(key, group_samples) for key, group_samples in sorted(groups_by_key.items())]
    audit_metrics = set(args.audit_metric)
    for key, group_samples in sorted(groups_by_key.items()):
        label = group_label(key)
        if len(group_samples) < args.min_samples_per_group:
            findings.append(Finding("-", 0, "error", "min_samples_per_group", label, "samples", float(len(group_samples)), None, float(args.min_samples_per_group), "not enough measured OK rows for outlier detection"))
            continue
        for metric in sorted(audit_metrics):
            values = metric_values(group_samples, metric)
            if len(values) < args.min_samples_per_group:
                findings.append(Finding("-", 0, "error", "min_metric_samples", label, metric, float(len(values)), None, float(args.min_samples_per_group), "not enough finite metric samples"))
                continue
            median = statistics.median(values)
            for sample in group_samples:
                value = getattr(sample, metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                delta = relative_delta_pct(float(value), median)
                if delta > args.max_relative_delta_pct:
                    findings.append(
                        Finding(
                            sample.source,
                            sample.row,
                            "error",
                            "relative_outlier",
                            label,
                            metric,
                            float(value),
                            median,
                            args.max_relative_delta_pct,
                            f"relative delta {delta:.3f}% exceeds per-prompt limit",
                        )
                    )

    return samples, groups, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_report(samples: list[Sample], groups: list[Group], findings: list[Finding], args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {"samples": len(samples), "groups": len(groups), "findings": len(findings)},
        "thresholds": {
            "min_rows": args.min_rows,
            "min_samples_per_group": args.min_samples_per_group,
            "max_relative_delta_pct": args.max_relative_delta_pct,
            "audit_metric": args.audit_metric,
            "require_metric": args.require_metric,
        },
        "groups": [asdict(group) for group in groups],
        "samples": [asdict(sample) for sample in samples],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    json_path = args.output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = [f"# QEMU Prompt Outlier Audit", "", f"Status: {status}", f"Samples: {len(samples)}", f"Groups: {len(groups)}", f"Findings: {len(findings)}", ""]
    if findings:
        markdown.extend(["## Findings", "", "| Kind | Group | Metric | Value | Median | Limit | Detail |", "| --- | --- | --- | ---: | ---: | ---: | --- |"])
        for finding in findings:
            markdown.append(f"| {finding.kind} | {finding.group} | {finding.metric} | {finding.value} | {finding.median} | {finding.limit} | {finding.detail} |")
    else:
        markdown.append("No prompt outlier findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(group) for group in groups], list(asdict(groups[0]).keys()) if groups else list(Group.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_samples.csv", [asdict(sample) for sample in samples], list(Sample.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    testsuite = ET.Element("testsuite", {"name": "holyc_qemu_prompt_outlier_audit", "tests": "1", "failures": "1" if status == "fail" else "0"})
    case = ET.SubElement(testsuite, "testcase", {"name": "prompt_outliers"})
    if status == "fail":
        failure = ET.SubElement(case, "failure", {"type": "qemu_prompt_outlier"})
        failure.text = "\n".join(f"{finding.kind}: {finding.group} {finding.metric} {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_outlier_audit_latest")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-samples-per-group", type=int, default=3)
    parser.add_argument("--max-relative-delta-pct", type=float, default=25.0)
    parser.add_argument("--audit-metric", action="append", choices=AUDIT_METRICS, default=["wall_elapsed_us", "wall_tok_per_s"])
    parser.add_argument("--require-metric", action="append", choices=AUDIT_METRICS, default=["wall_elapsed_us", "wall_tok_per_s"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples, groups, findings = audit(args.inputs, args)
    report_path = write_report(samples, groups, findings, args)
    print(report_path)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
