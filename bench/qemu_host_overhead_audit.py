#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for host-overhead accounting.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
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
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class OverheadRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    iteration: int | None
    exit_class: str
    elapsed_us: float | None
    wall_elapsed_us: float | None
    host_overhead_us: float | None
    host_overhead_pct: float | None
    expected_host_overhead_us: float | None
    expected_host_overhead_pct: float | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    metric: str
    detail: str


@dataclass(frozen=True)
class OverheadGroup:
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    rows: int
    ok_rows: int
    negative_host_overhead_rows: int
    min_host_overhead_pct: float | None
    max_host_overhead_pct: float | None
    median_host_overhead_pct: float | None
    p95_host_overhead_pct: float | None
    avg_host_overhead_pct: float | None


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

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    if key == "warmups" and "phase" not in merged:
                        merged["phase"] = "warmup"
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


def overhead_row(source: Path, row_number: int, row: dict[str, Any]) -> OverheadRow:
    elapsed_us = finite_float(row.get("elapsed_us"))
    wall_elapsed_us = finite_float(row.get("wall_elapsed_us"))
    host_overhead_us = finite_float(row.get("host_overhead_us"))
    host_overhead_pct = finite_float(row.get("host_overhead_pct"))
    expected_us = None
    expected_pct = None
    if elapsed_us is not None and wall_elapsed_us is not None:
        expected_us = wall_elapsed_us - elapsed_us
        if elapsed_us > 0:
            expected_pct = expected_us * 100.0 / elapsed_us
    return OverheadRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        phase=row_text(row, "phase", default="measured").lower(),
        iteration=finite_int(row.get("iteration")),
        exit_class=row_text(row, "exit_class").lower(),
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        host_overhead_us=host_overhead_us,
        host_overhead_pct=host_overhead_pct,
        expected_host_overhead_us=expected_us,
        expected_host_overhead_pct=expected_pct,
    )


def row_label(row: OverheadRow) -> str:
    return f"{row.profile}/{row.model}/{row.quantization}/{row.prompt}/{row.phase}/{row.iteration or '-'}"


def group_key(row: OverheadRow) -> tuple[str, str, str, str, str]:
    return (row.profile, row.model, row.quantization, row.prompt, row.phase)


def build_groups(rows: list[OverheadRow]) -> list[OverheadGroup]:
    grouped: dict[tuple[str, str, str, str, str], list[OverheadRow]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    summaries: list[OverheadGroup] = []
    for key, group_rows in sorted(grouped.items()):
        values = sorted(row.host_overhead_pct for row in group_rows if row.host_overhead_pct is not None)
        summaries.append(
            OverheadGroup(
                profile=key[0],
                model=key[1],
                quantization=key[2],
                prompt=key[3],
                phase=key[4],
                rows=len(group_rows),
                ok_rows=sum(1 for row in group_rows if row.exit_class == "ok"),
                negative_host_overhead_rows=sum(1 for row in group_rows if row.host_overhead_us is not None and row.host_overhead_us < 0),
                min_host_overhead_pct=min(values, default=None),
                max_host_overhead_pct=max(values, default=None),
                median_host_overhead_pct=statistics.median(values) if values else None,
                p95_host_overhead_pct=percentile(values, 95),
                avg_host_overhead_pct=sum(values) / len(values) if values else None,
            )
        )
    return summaries


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[OverheadRow], list[Finding]]:
    rows: list[OverheadRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", str(exc)))
            continue
        for row_number, raw_row in enumerate(loaded_rows, 1):
            row = overhead_row(path, row_number, raw_row)
            rows.append(row)
            label = row_label(row)
            if row.elapsed_us is None or row.elapsed_us <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_elapsed_us", "elapsed_us", f"{label}: elapsed_us must be positive"))
            if row.wall_elapsed_us is None or row.wall_elapsed_us <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_wall_elapsed_us", "wall_elapsed_us", f"{label}: wall_elapsed_us must be positive"))
            if args.fail_negative_host_overhead and row.elapsed_us is not None and row.wall_elapsed_us is not None and row.wall_elapsed_us < row.elapsed_us:
                findings.append(Finding(row.source, row.row, "error", "wall_elapsed_before_guest_elapsed", "wall_elapsed_us", f"{label}: wall_elapsed_us is smaller than elapsed_us"))
            if row.expected_host_overhead_us is not None:
                if row.host_overhead_us is None:
                    findings.append(Finding(row.source, row.row, "error", "missing_host_overhead_us", "host_overhead_us", f"{label}: host_overhead_us is required"))
                elif abs(row.host_overhead_us - row.expected_host_overhead_us) > args.us_tolerance:
                    findings.append(Finding(row.source, row.row, "error", "host_overhead_us_drift", "host_overhead_us", f"{label}: stored {row.host_overhead_us:.6g}, expected {row.expected_host_overhead_us:.6g}"))
            if row.expected_host_overhead_pct is not None:
                if row.host_overhead_pct is None:
                    findings.append(Finding(row.source, row.row, "error", "missing_host_overhead_pct", "host_overhead_pct", f"{label}: host_overhead_pct is required"))
                elif abs(row.host_overhead_pct - row.expected_host_overhead_pct) > args.pct_tolerance:
                    findings.append(Finding(row.source, row.row, "error", "host_overhead_pct_drift", "host_overhead_pct", f"{label}: stored {row.host_overhead_pct:.6g}, expected {row.expected_host_overhead_pct:.6g}"))
            if row.exit_class == "ok" and row.host_overhead_pct is not None and row.host_overhead_pct > args.max_ok_host_overhead_pct:
                findings.append(Finding(row.source, row.row, "error", "ok_host_overhead_too_high", "host_overhead_pct", f"{label}: OK row host overhead is {row.host_overhead_pct:.6g}%"))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


def write_json(path: Path, rows: list[OverheadRow], findings: list[Finding]) -> None:
    overhead_values = [row.host_overhead_pct for row in rows if row.host_overhead_pct is not None]
    sorted_overheads = sorted(overhead_values)
    groups = build_groups(rows)
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "rows": len(rows),
            "ok_rows": sum(1 for row in rows if row.exit_class == "ok"),
            "groups": len(groups),
            "negative_host_overhead_rows": sum(1 for row in rows if row.host_overhead_us is not None and row.host_overhead_us < 0),
            "findings": len(findings),
            "max_host_overhead_pct": max(overhead_values, default=None),
            "median_host_overhead_pct": statistics.median(sorted_overheads) if sorted_overheads else None,
            "p95_host_overhead_pct": percentile(sorted_overheads, 95),
            "avg_host_overhead_pct": sum(overhead_values) / len(overhead_values) if overhead_values else None,
        },
        "rows": [asdict(row) for row in rows],
        "groups": [asdict(group) for group in groups],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def write_markdown(path: Path, rows: list[OverheadRow], findings: list[Finding]) -> None:
    overhead_values = sorted(row.host_overhead_pct for row in rows if row.host_overhead_pct is not None)
    groups = build_groups(rows)
    max_pct = max(overhead_values, default=None)
    median_pct = statistics.median(overhead_values) if overhead_values else None
    p95_pct = percentile(overhead_values, 95)
    lines = [
        "# QEMU Host Overhead Audit",
        "",
        f"Generated: {iso_now()}",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Groups: {len(groups)}",
        f"Negative host overhead rows: {sum(1 for row in rows if row.host_overhead_us is not None and row.host_overhead_us < 0)}",
        f"Findings: {len(findings)}",
        f"Max host overhead %: {max_pct if max_pct is not None else '-'}",
        f"Median host overhead %: {median_pct if median_pct is not None else '-'}",
        f"P95 host overhead %: {p95_pct if p95_pct is not None else '-'}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} ({finding.metric}) {finding.detail}" for finding in findings)
    else:
        lines.append("No host overhead findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[OverheadRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(OverheadRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fieldnames = list(Finding.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_group_csv(path: Path, rows: list[OverheadRow]) -> None:
    fieldnames = list(OverheadGroup.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for group in build_groups(rows):
            writer.writerow(asdict(group))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_host_overhead_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="host_overhead_accounting")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} host overhead finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Output directory")
    parser.add_argument("--output-stem", default="qemu_host_overhead_audit_latest", help="Output filename stem")
    parser.add_argument("--min-artifacts", type=int, default=1, help="Minimum input artifact files")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum loaded rows")
    parser.add_argument("--us-tolerance", type=float, default=2.0, help="Allowed host_overhead_us drift")
    parser.add_argument("--pct-tolerance", type=float, default=0.01, help="Allowed host_overhead_pct drift")
    parser.add_argument("--max-ok-host-overhead-pct", type=float, default=50.0, help="Maximum OK-row host overhead percent")
    parser.add_argument("--fail-negative-host-overhead", action="store_true", help="Fail when wall_elapsed_us is smaller than elapsed_us")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_group_csv(args.output_dir / f"{stem}_groups.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
