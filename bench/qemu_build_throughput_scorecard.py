#!/usr/bin/env python3
"""Summarize saved QEMU prompt benchmark throughput by build.

This host-side tool reads benchmark artifacts only. It never launches QEMU and
therefore cannot affect the TempleOS guest air gap.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.csv", "qemu_prompt_bench*.jsonl")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class ThroughputRow:
    build: str
    profile: str
    model: str
    quantization: str
    source_count: int
    measured_rows: int
    total_tokens: int
    mean_tok_per_s: float
    median_tok_per_s: float
    stdev_tok_per_s: float
    cv_tok_per_s: float
    weighted_tok_per_s: float | None
    min_tok_per_s: float
    max_tok_per_s: float
    mean_wall_tok_per_s: float | None
    stdev_wall_tok_per_s: float | None
    cv_wall_tok_per_s: float | None
    weighted_wall_tok_per_s: float | None
    total_elapsed_us: int
    total_wall_elapsed_us: int


@dataclass(frozen=True)
class ThroughputFinding:
    severity: str
    kind: str
    build: str
    profile: str
    model: str
    quantization: str
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


def text_value(row: dict[str, Any], key: str, default: str = "-") -> str:
    value = row.get(key)
    if value in (None, ""):
        return default
    return str(value)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def measured_ok(row: dict[str, Any]) -> bool:
    return text_value(row, "phase", "measured") == "measured" and text_value(row, "exit_class", "") == "ok" and not truthy(row.get("timed_out"))


def infer_build(row: dict[str, Any], source: Path) -> str:
    for key in ("build", "build_id", "build_name", "commit"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    parent = source.parent.name
    if "_" in parent and parent.startswith(("ci-", "bench-", "qemu-")):
        return parent
    return source.stem


def summarize_group(rows: list[dict[str, Any]], key: tuple[str, str, str, str], sources: set[str]) -> tuple[ThroughputRow | None, list[ThroughputFinding]]:
    build, profile, model, quantization = key
    findings: list[ThroughputFinding] = []
    tok_rates: list[float] = []
    wall_rates: list[float] = []
    total_tokens = 0
    total_elapsed_us = 0
    total_wall_elapsed_us = 0

    for row in rows:
        tok_per_s = finite_float(row.get("tok_per_s"))
        wall_tok_per_s = finite_float(row.get("wall_tok_per_s"))
        tokens = finite_int(row.get("tokens")) or 0
        elapsed_us = finite_int(row.get("elapsed_us")) or 0
        wall_elapsed_us = finite_int(row.get("wall_elapsed_us")) or 0
        if tok_per_s is None or tok_per_s <= 0:
            findings.append(
                ThroughputFinding("error", "missing_tok_per_s", build, profile, model, quantization, "measured row lacks positive tok_per_s")
            )
            continue
        tok_rates.append(tok_per_s)
        if wall_tok_per_s is not None and wall_tok_per_s > 0:
            wall_rates.append(wall_tok_per_s)
        total_tokens += tokens
        total_elapsed_us += elapsed_us
        total_wall_elapsed_us += wall_elapsed_us

    if not tok_rates:
        return None, findings
    mean_tok_per_s = statistics.fmean(tok_rates)
    stdev_tok_per_s = statistics.pstdev(tok_rates) if len(tok_rates) > 1 else 0.0
    cv_tok_per_s = stdev_tok_per_s / mean_tok_per_s if mean_tok_per_s > 0 else 0.0
    mean_wall_tok_per_s = statistics.fmean(wall_rates) if wall_rates else None
    stdev_wall_tok_per_s = statistics.pstdev(wall_rates) if len(wall_rates) > 1 else (0.0 if wall_rates else None)
    cv_wall_tok_per_s = (
        stdev_wall_tok_per_s / mean_wall_tok_per_s
        if stdev_wall_tok_per_s is not None and mean_wall_tok_per_s is not None and mean_wall_tok_per_s > 0
        else (0.0 if wall_rates else None)
    )
    weighted_tok_per_s = None
    if total_tokens > 0 and total_elapsed_us > 0:
        weighted_tok_per_s = total_tokens * 1_000_000.0 / total_elapsed_us
    weighted_wall_tok_per_s = None
    if total_tokens > 0 and total_wall_elapsed_us > 0:
        weighted_wall_tok_per_s = total_tokens * 1_000_000.0 / total_wall_elapsed_us
    return (
        ThroughputRow(
            build=build,
            profile=profile,
            model=model,
            quantization=quantization,
            source_count=len(sources),
            measured_rows=len(tok_rates),
            total_tokens=total_tokens,
            mean_tok_per_s=mean_tok_per_s,
            median_tok_per_s=statistics.median(tok_rates),
            stdev_tok_per_s=stdev_tok_per_s,
            cv_tok_per_s=cv_tok_per_s,
            weighted_tok_per_s=weighted_tok_per_s,
            min_tok_per_s=min(tok_rates),
            max_tok_per_s=max(tok_rates),
            mean_wall_tok_per_s=mean_wall_tok_per_s,
            stdev_wall_tok_per_s=stdev_wall_tok_per_s,
            cv_wall_tok_per_s=cv_wall_tok_per_s,
            weighted_wall_tok_per_s=weighted_wall_tok_per_s,
            total_elapsed_us=total_elapsed_us,
            total_wall_elapsed_us=total_wall_elapsed_us,
        ),
        findings,
    )


def build_scorecard(
    paths: list[Path],
    patterns: list[str],
    min_rows: int,
    *,
    max_cv: float | None = None,
    max_wall_cv: float | None = None,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    group_sources: dict[tuple[str, str, str, str], set[str]] = {}
    input_files = list(iter_input_files(paths, patterns))
    for source in input_files:
        for raw in load_rows(source):
            if not measured_ok(raw):
                continue
            key = (
                infer_build(raw, source),
                text_value(raw, "profile"),
                text_value(raw, "model"),
                text_value(raw, "quantization"),
            )
            groups.setdefault(key, []).append(raw)
            group_sources.setdefault(key, set()).add(str(source))

    rows: list[ThroughputRow] = []
    findings: list[ThroughputFinding] = []
    for key in sorted(groups):
        summary, group_findings = summarize_group(groups[key], key, group_sources[key])
        findings.extend(group_findings)
        if summary is None:
            continue
        if summary.measured_rows < min_rows:
            findings.append(
                ThroughputFinding(
                    "error",
                    "min_rows",
                    summary.build,
                    summary.profile,
                    summary.model,
                    summary.quantization,
                    f"{summary.measured_rows} measured rows below required {min_rows}",
                )
            )
        if max_cv is not None and summary.cv_tok_per_s > max_cv:
            findings.append(
                ThroughputFinding(
                    "error",
                    "max_cv",
                    summary.build,
                    summary.profile,
                    summary.model,
                    summary.quantization,
                    f"tok/s coefficient of variation {summary.cv_tok_per_s:.6g} exceeds limit {max_cv:.6g}",
                )
            )
        if max_wall_cv is not None and summary.cv_wall_tok_per_s is not None and summary.cv_wall_tok_per_s > max_wall_cv:
            findings.append(
                ThroughputFinding(
                    "error",
                    "max_wall_cv",
                    summary.build,
                    summary.profile,
                    summary.model,
                    summary.quantization,
                    f"wall tok/s coefficient of variation {summary.cv_wall_tok_per_s:.6g} exceeds limit {max_wall_cv:.6g}",
                )
            )
        rows.append(summary)

    if not rows:
        findings.append(ThroughputFinding("error", "no_measured_rows", "-", "-", "-", "-", "no measured ok QEMU prompt rows found"))

    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    return {
        "generated_at": iso_now(),
        "status": status,
        "inputs": [str(path) for path in input_files],
        "summary": {
            "groups": len(rows),
            "input_files": len(input_files),
            "measured_rows": sum(row.measured_rows for row in rows),
            "total_tokens": sum(row.total_tokens for row in rows),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# QEMU Build Throughput Scorecard",
        "",
        f"- Status: {payload['status']}",
        f"- Groups: {payload['summary']['groups']}",
        f"- Measured rows: {payload['summary']['measured_rows']}",
        f"- Total tokens: {payload['summary']['total_tokens']}",
        "",
        "| Build | Profile | Model | Quantization | Rows | Mean tok/s | CV | Weighted tok/s | Median tok/s | Mean wall tok/s | Wall CV | Weighted wall tok/s |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        wall = "-" if row["mean_wall_tok_per_s"] is None else f"{row['mean_wall_tok_per_s']:.6f}"
        wall_cv = "-" if row["cv_wall_tok_per_s"] is None else f"{row['cv_wall_tok_per_s']:.6f}"
        weighted = "-" if row["weighted_tok_per_s"] is None else f"{row['weighted_tok_per_s']:.6f}"
        weighted_wall = "-" if row["weighted_wall_tok_per_s"] is None else f"{row['weighted_wall_tok_per_s']:.6f}"
        lines.append(
            f"| {row['build']} | {row['profile']} | {row['model']} | {row['quantization']} | "
            f"{row['measured_rows']} | {row['mean_tok_per_s']:.6f} | {row['cv_tok_per_s']:.6f} | {weighted} | "
            f"{row['median_tok_per_s']:.6f} | {wall} | {wall_cv} | {weighted_wall} |"
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in payload["findings"]:
            lines.append(f"- {finding['severity']}: {finding['kind']} {finding['build']} {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    failures = [finding for finding in payload["findings"] if finding["severity"] == "error"]
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_build_throughput_scorecard",
            "tests": str(max(1, len(payload["rows"]))),
            "failures": str(len(failures)),
        },
    )
    if not payload["rows"]:
        testcase = ET.SubElement(testsuite, "testcase", {"name": "no_rows"})
        ET.SubElement(testcase, "failure", {"message": "no measured rows"}).text = "no measured ok QEMU prompt rows found"
    for row in payload["rows"]:
        name = f"{row['build']}:{row['profile']}:{row['model']}:{row['quantization']}"
        testcase = ET.SubElement(testsuite, "testcase", {"name": name})
        for finding in failures:
            if (
                finding["build"] == row["build"]
                and finding["profile"] == row["profile"]
                and finding["model"] == row["model"]
                and finding["quantization"] == row["quantization"]
            ):
                ET.SubElement(testcase, "failure", {"message": finding["kind"]}).text = finding["detail"]
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(payload: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{output_stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        output_dir / f"{output_stem}.csv",
        payload["rows"],
        [
            "build",
            "profile",
            "model",
            "quantization",
            "source_count",
            "measured_rows",
            "total_tokens",
            "mean_tok_per_s",
            "median_tok_per_s",
            "stdev_tok_per_s",
            "cv_tok_per_s",
            "weighted_tok_per_s",
            "min_tok_per_s",
            "max_tok_per_s",
            "mean_wall_tok_per_s",
            "stdev_wall_tok_per_s",
            "cv_wall_tok_per_s",
            "weighted_wall_tok_per_s",
            "total_elapsed_us",
            "total_wall_elapsed_us",
        ],
    )
    write_csv(
        output_dir / f"{output_stem}_findings.csv",
        payload["findings"],
        ["severity", "kind", "build", "profile", "model", "quantization", "detail"],
    )
    write_markdown(output_dir / f"{output_stem}.md", payload)
    write_junit(output_dir / f"{output_stem}_junit.xml", payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/CSV/JSONL files or directories")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Glob to use when an input is a directory")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum measured rows required per build group")
    parser.add_argument("--max-cv", type=float, help="Maximum allowed tok/s coefficient of variation per build group")
    parser.add_argument("--max-wall-cv", type=float, help="Maximum allowed wall tok/s coefficient of variation per build group")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_build_throughput_scorecard_latest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.min_rows < 1:
        raise SystemExit("--min-rows must be at least 1")
    if args.max_cv is not None and args.max_cv < 0:
        raise SystemExit("--max-cv must be non-negative")
    if args.max_wall_cv is not None and args.max_wall_cv < 0:
        raise SystemExit("--max-wall-cv must be non-negative")
    payload = build_scorecard(
        args.inputs,
        args.patterns or list(DEFAULT_PATTERNS),
        args.min_rows,
        max_cv=args.max_cv,
        max_wall_cv=args.max_wall_cv,
    )
    write_outputs(payload, args.output_dir, args.output_stem)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
