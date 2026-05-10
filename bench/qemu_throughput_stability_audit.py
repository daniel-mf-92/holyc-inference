#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for throughput stability.

This host-side tool reads benchmark JSON/JSONL/CSV artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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


@dataclass(frozen=True)
class ThroughputSample:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    phase: str
    exit_class: str
    tokens: int | None
    elapsed_us: float | None
    wall_elapsed_us: float | None
    tok_per_s: float | None
    wall_tok_per_s: float | None


@dataclass(frozen=True)
class ThroughputGroup:
    profile: str
    model: str
    quantization: str
    prompt: str
    samples: int
    tokens_total: int
    wall_tok_per_s_min: float | None
    wall_tok_per_s_mean: float | None
    wall_tok_per_s_median: float | None
    wall_tok_per_s_max: float | None
    wall_tok_per_s_cv: float | None
    tok_per_s_median: float | None
    sources: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    group: str
    metric: str
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


def group_key(sample: ThroughputSample) -> tuple[str, str, str, str]:
    return (sample.profile, sample.model, sample.quantization, sample.prompt)


def group_label(key: tuple[str, str, str, str]) -> str:
    return "/".join(key)


def metric_values(samples: list[ThroughputSample], metric: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = getattr(sample, metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = statistics.mean(values)
    if mean <= 0:
        return None
    return statistics.pstdev(values) / mean


def throughput_sample(source: Path, row_number: int, row: dict[str, Any], required_metrics: tuple[str, ...]) -> tuple[ThroughputSample, list[Finding]]:
    profile = row_text(row, "profile")
    model = row_text(row, "model")
    quantization = row_text(row, "quantization")
    prompt = row_text(row, "prompt", "prompt_id")
    group = group_label((profile, model, quantization, prompt))
    findings: list[Finding] = []
    metrics: dict[str, float | None] = {}
    for metric in ("elapsed_us", "wall_elapsed_us", "tok_per_s", "wall_tok_per_s"):
        value = finite_float(row.get(metric))
        metrics[metric] = value
        if metric in required_metrics:
            if metric not in row or row.get(metric) in (None, ""):
                findings.append(Finding(str(source), row_number, "error", "missing_metric", group, metric, "required throughput metric is absent"))
            elif value is None:
                findings.append(Finding(str(source), row_number, "error", "invalid_metric", group, metric, "metric must be finite numeric telemetry"))
            elif value <= 0:
                findings.append(Finding(str(source), row_number, "error", "nonpositive_metric", group, metric, "metric must be positive"))

    tokens = finite_int(row.get("tokens"))
    wall_elapsed_us = metrics["wall_elapsed_us"]
    wall_tok_per_s = metrics["wall_tok_per_s"]
    if tokens is not None and tokens > 0 and wall_elapsed_us is not None and wall_elapsed_us > 0 and wall_tok_per_s is not None:
        expected = tokens * 1_000_000.0 / wall_elapsed_us
        tolerance = max(0.01, abs(expected) * 0.001)
        if abs(wall_tok_per_s - expected) > tolerance:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "wall_tok_per_s_drift",
                    group,
                    "wall_tok_per_s",
                    f"expected about {expected:.6f} from tokens/wall_elapsed_us",
                )
            )

    return (
        ThroughputSample(
            source=str(source),
            row=row_number,
            profile=profile,
            model=model,
            quantization=quantization,
            prompt=prompt,
            commit=row_text(row, "commit"),
            phase=row_text(row, "phase", default="measured"),
            exit_class=row_text(row, "exit_class"),
            tokens=tokens,
            elapsed_us=metrics["elapsed_us"],
            wall_elapsed_us=metrics["wall_elapsed_us"],
            tok_per_s=metrics["tok_per_s"],
            wall_tok_per_s=metrics["wall_tok_per_s"],
        ),
        findings,
    )


def throughput_group(key: tuple[str, str, str, str], samples: list[ThroughputSample]) -> ThroughputGroup:
    wall_values = metric_values(samples, "wall_tok_per_s")
    tok_values = metric_values(samples, "tok_per_s")
    return ThroughputGroup(
        profile=key[0],
        model=key[1],
        quantization=key[2],
        prompt=key[3],
        samples=len(samples),
        tokens_total=sum(sample.tokens or 0 for sample in samples),
        wall_tok_per_s_min=min(wall_values) if wall_values else None,
        wall_tok_per_s_mean=statistics.mean(wall_values) if wall_values else None,
        wall_tok_per_s_median=statistics.median(wall_values) if wall_values else None,
        wall_tok_per_s_max=max(wall_values) if wall_values else None,
        wall_tok_per_s_cv=coefficient_of_variation(wall_values),
        tok_per_s_median=statistics.median(tok_values) if tok_values else None,
        sources=";".join(sorted({sample.source for sample in samples})),
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ThroughputSample], list[ThroughputGroup], list[Finding]]:
    samples: list[ThroughputSample] = []
    findings: list[Finding] = []
    seen_files = 0
    required_metrics = tuple(args.require_metric)
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "-", "-", str(exc)))
            continue
        for row_number, row in enumerate(rows, 1):
            phase = row_text(row, "phase", default="measured")
            exit_class = row_text(row, "exit_class")
            if args.measured_only and phase == "warmup":
                continue
            if args.ok_only and exit_class not in {"ok", "-"}:
                continue
            sample, sample_findings = throughput_sample(path, row_number, row, required_metrics)
            samples.append(sample)
            findings.extend(sample_findings)

    grouped: dict[tuple[str, str, str, str], list[ThroughputSample]] = {}
    for sample in samples:
        grouped.setdefault(group_key(sample), []).append(sample)
    groups = [throughput_group(key, grouped[key]) for key in sorted(grouped)]

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "-", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(samples) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "-", "rows", f"found {len(samples)}, expected at least {args.min_rows}"))

    for group in groups:
        label = group_label((group.profile, group.model, group.quantization, group.prompt))
        if group.samples < args.min_samples_per_group:
            findings.append(
                Finding("-", 0, "error", "min_samples_per_group", label, "samples", f"found {group.samples}, expected at least {args.min_samples_per_group}")
            )
        if args.min_wall_tok_per_s is not None and group.wall_tok_per_s_min is not None and group.wall_tok_per_s_min < args.min_wall_tok_per_s:
            findings.append(
                Finding("-", 0, "error", "min_wall_tok_per_s", label, "wall_tok_per_s_min", f"{group.wall_tok_per_s_min:.6f} < {args.min_wall_tok_per_s:.6f}")
            )
        if args.max_wall_tok_per_s_cv is not None and group.wall_tok_per_s_cv is not None and group.wall_tok_per_s_cv > args.max_wall_tok_per_s_cv:
            findings.append(
                Finding("-", 0, "error", "max_wall_tok_per_s_cv", label, "wall_tok_per_s_cv", f"{group.wall_tok_per_s_cv:.6f} > {args.max_wall_tok_per_s_cv:.6f}")
            )

    return samples, groups, findings


def summary(samples: list[ThroughputSample], groups: list[ThroughputGroup], findings: list[Finding]) -> dict[str, Any]:
    wall_values = metric_values(samples, "wall_tok_per_s")
    return {
        "rows": len(samples),
        "groups": len(groups),
        "findings": len(findings),
        "profiles": sorted({sample.profile for sample in samples if sample.profile != "-"}),
        "models": sorted({sample.model for sample in samples if sample.model != "-"}),
        "quantizations": sorted({sample.quantization for sample in samples if sample.quantization != "-"}),
        "tokens_total": sum(sample.tokens or 0 for sample in samples),
        "wall_tok_per_s_min": min(wall_values) if wall_values else None,
        "wall_tok_per_s_median": statistics.median(wall_values) if wall_values else None,
        "wall_tok_per_s_cv": coefficient_of_variation(wall_values),
    }


def write_json(path: Path, samples: list[ThroughputSample], groups: list[ThroughputGroup], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(samples, groups, findings),
        "groups": [asdict(group) for group in groups],
        "samples": [asdict(sample) for sample in samples],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.6f}"


def write_markdown(path: Path, samples: list[ThroughputSample], groups: list[ThroughputGroup], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Throughput Stability Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(samples)}",
        f"Groups: {len(groups)}",
        f"Findings: {len(findings)}",
        "",
        "## Groups",
        "",
        "| Profile | Model | Quantization | Prompt | Samples | Wall tok/s min | Wall tok/s median | Wall tok/s CV |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group.profile} | {group.model} | {group.quantization} | {group.prompt} | {group.samples} | "
            f"{format_number(group.wall_tok_per_s_min)} | {format_number(group.wall_tok_per_s_median)} | {format_number(group.wall_tok_per_s_cv)} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Group | Metric | Detail |", "| --- | ---: | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.group} | {finding.metric} | {finding.detail} |")
    else:
        lines.append("No throughput stability findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, groups: list[ThroughputGroup]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThroughputGroup.__dataclass_fields__))
        writer.writeheader()
        for group in groups:
            writer.writerow(asdict(group))


def write_samples_csv(path: Path, samples: list[ThroughputSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ThroughputSample.__dataclass_fields__))
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))


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
            "name": "holyc_qemu_throughput_stability_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "throughput_stability"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} throughput stability finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_throughput_stability_audit_latest")
    parser.add_argument("--require-metric", action="append", default=[], help="Required throughput metric; repeatable")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-samples-per-group", type=int, default=1)
    parser.add_argument("--min-wall-tok-per-s", type=float)
    parser.add_argument("--max-wall-tok-per-s-cv", type=float)
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
        args.require_metric = ["wall_elapsed_us", "wall_tok_per_s"]
    if args.min_artifacts < 0 or args.min_rows < 0 or args.min_samples_per_group < 0:
        parser.error("--min-artifacts, --min-rows, and --min-samples-per-group must be >= 0")

    samples, groups, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", samples, groups, findings)
    write_markdown(args.output_dir / f"{stem}.md", samples, groups, findings)
    write_csv(args.output_dir / f"{stem}.csv", groups)
    write_samples_csv(args.output_dir / f"{stem}_samples.csv", samples)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
