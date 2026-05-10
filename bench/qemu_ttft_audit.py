#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for TTFT telemetry.

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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class TtftRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    iteration: int | None
    exit_class: str
    tokens: int | None
    ttft_us: float | None
    elapsed_us: float | None
    wall_elapsed_us: float | None
    ttft_elapsed_pct: float | None
    ttft_wall_pct: float | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
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


def include_row(raw: dict[str, Any], include_failed: bool) -> bool:
    phase = str(raw.get("phase") or "measured").lower()
    exit_class = str(raw.get("exit_class") or "").lower()
    if phase != "measured":
        return False
    return include_failed or exit_class == "ok"


def ttft_row(source: Path, row_number: int, raw: dict[str, Any]) -> TtftRow:
    ttft_us = finite_float(raw.get("ttft_us"))
    elapsed_us = finite_float(raw.get("elapsed_us"))
    wall_elapsed_us = finite_float(raw.get("wall_elapsed_us"))
    return TtftRow(
        source=str(source),
        row=row_number,
        profile=row_text(raw, "profile"),
        model=row_text(raw, "model"),
        quantization=row_text(raw, "quantization"),
        prompt=row_text(raw, "prompt", "prompt_id"),
        phase=row_text(raw, "phase", default="measured").lower(),
        iteration=finite_int(raw.get("iteration")),
        exit_class=row_text(raw, "exit_class").lower(),
        tokens=finite_int(raw.get("tokens")),
        ttft_us=ttft_us,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        ttft_elapsed_pct=ttft_us * 100.0 / elapsed_us if ttft_us is not None and elapsed_us and elapsed_us > 0 else None,
        ttft_wall_pct=ttft_us * 100.0 / wall_elapsed_us if ttft_us is not None and wall_elapsed_us and wall_elapsed_us > 0 else None,
    )


def row_label(row: TtftRow) -> str:
    return f"{row.profile}/{row.model}/{row.quantization}/{row.prompt}/{row.iteration or '-'}"


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[TtftRow], list[Finding]]:
    rows: list[TtftRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", str(exc)))
            continue
        for row_number, raw in enumerate(loaded_rows, 1):
            if not include_row(raw, args.include_failed):
                continue
            row = ttft_row(path, row_number, raw)
            rows.append(row)
            label = row_label(row)
            if row.tokens is None or row.tokens <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_tokens", "tokens", f"{label}: positive tokens are required"))
            if row.elapsed_us is None or row.elapsed_us <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_elapsed_us", "elapsed_us", f"{label}: positive elapsed_us is required"))
            if row.wall_elapsed_us is None or row.wall_elapsed_us <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_wall_elapsed_us", "wall_elapsed_us", f"{label}: positive wall_elapsed_us is required"))
            if row.ttft_us is None:
                findings.append(Finding(row.source, row.row, "error", "missing_ttft_us", "ttft_us", f"{label}: ttft_us is required"))
                continue
            if row.ttft_us < 0:
                findings.append(Finding(row.source, row.row, "error", "negative_ttft_us", "ttft_us", f"{label}: ttft_us must be non-negative"))
            if row.elapsed_us is not None and row.elapsed_us > 0 and row.ttft_us > row.elapsed_us + args.us_tolerance:
                findings.append(Finding(row.source, row.row, "error", "ttft_after_guest_elapsed", "ttft_us", f"{label}: ttft_us {row.ttft_us:.6g} exceeds elapsed_us {row.elapsed_us:.6g}"))
            if row.wall_elapsed_us is not None and row.wall_elapsed_us > 0 and row.ttft_us > row.wall_elapsed_us + args.us_tolerance:
                findings.append(Finding(row.source, row.row, "error", "ttft_after_wall_elapsed", "ttft_us", f"{label}: ttft_us {row.ttft_us:.6g} exceeds wall_elapsed_us {row.wall_elapsed_us:.6g}"))
            if args.max_ttft_us is not None and row.ttft_us > args.max_ttft_us:
                findings.append(Finding(row.source, row.row, "error", "max_ttft_us", "ttft_us", f"{label}: ttft_us {row.ttft_us:.6g} exceeds {args.max_ttft_us:.6g}"))
            if row.ttft_elapsed_pct is not None and row.ttft_elapsed_pct > args.max_ttft_elapsed_pct:
                findings.append(Finding(row.source, row.row, "error", "max_ttft_elapsed_pct", "ttft_elapsed_pct", f"{label}: TTFT is {row.ttft_elapsed_pct:.6g}% of guest elapsed time"))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


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


def summary(rows: list[TtftRow], findings: list[Finding]) -> dict[str, Any]:
    ttft_values = sorted(row.ttft_us for row in rows if row.ttft_us is not None)
    elapsed_pct_values = sorted(row.ttft_elapsed_pct for row in rows if row.ttft_elapsed_pct is not None)
    return {
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.exit_class == "ok"),
        "findings": len(findings),
        "min_ttft_us": min(ttft_values, default=None),
        "median_ttft_us": statistics.median(ttft_values) if ttft_values else None,
        "p95_ttft_us": percentile(ttft_values, 95),
        "max_ttft_us": max(ttft_values, default=None),
        "max_ttft_elapsed_pct": max(elapsed_pct_values, default=None),
    }


def write_json(path: Path, rows: list[TtftRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[TtftRow], findings: list[Finding]) -> None:
    stats = summary(rows, findings)
    lines = [
        "# QEMU TTFT Audit",
        "",
        f"Generated: {iso_now()}",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {stats['rows']}",
        f"Findings: {stats['findings']}",
        f"Median TTFT us: {stats['median_ttft_us'] if stats['median_ttft_us'] is not None else '-'}",
        f"P95 TTFT us: {stats['p95_ttft_us'] if stats['p95_ttft_us'] is not None else '-'}",
        f"Max TTFT elapsed %: {stats['max_ttft_elapsed_pct'] if stats['max_ttft_elapsed_pct'] is not None else '-'}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} ({finding.metric}) {finding.detail}" for finding in findings)
    else:
        lines.append("No TTFT findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[TtftRow]) -> None:
    fieldnames = list(TtftRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fieldnames = list(Finding.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_ttft_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="ttft_telemetry")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} TTFT finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Output directory")
    parser.add_argument("--output-stem", default="qemu_ttft_audit_latest", help="Output filename stem")
    parser.add_argument("--min-artifacts", type=int, default=1, help="Minimum input artifact files")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum measured rows")
    parser.add_argument("--include-failed", action="store_true", help="Audit failed measured rows in addition to OK rows")
    parser.add_argument("--us-tolerance", type=float, default=2.0, help="Allowed comparison drift in microseconds")
    parser.add_argument("--max-ttft-us", type=float, help="Optional absolute TTFT ceiling")
    parser.add_argument("--max-ttft-elapsed-pct", type=float, default=100.0, help="Maximum TTFT share of guest elapsed time")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_artifacts < 0 or args.min_rows < 0 or args.us_tolerance < 0 or args.max_ttft_elapsed_pct < 0:
        build_parser().error("--min-artifacts, --min-rows, --us-tolerance, and --max-ttft-elapsed-pct must be >= 0")
    rows, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
