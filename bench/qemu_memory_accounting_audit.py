#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for memory telemetry consistency.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class MemoryRow:
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
    memory_bytes: int | None
    memory_bytes_per_token: float | None
    host_child_peak_rss_bytes: int | None
    host_rss_bytes_per_token: float | None
    host_rss_over_guest_ratio: float | None
    checks: int
    findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    metric: str
    stored: str
    expected: str
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


def format_expected(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.9g}"


def tolerance(expected: float, relative: float, absolute: float) -> float:
    return max(absolute, abs(expected) * relative)


def check_metric(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    metric: str,
    stored: float | None,
    expected: float | None,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> int:
    if expected is None:
        return 0
    if stored is None:
        findings.append(Finding(str(source), row_number, "error", "missing_metric", metric, "", format_expected(expected), "derived metric is absent"))
        return 1
    checks = 1
    allowed = tolerance(expected, relative_tolerance, absolute_tolerance)
    if abs(stored - expected) > allowed:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "metric_drift",
                metric,
                format_expected(stored),
                format_expected(expected),
                f"outside tolerance {allowed:.9g}",
            )
        )
    return checks


def require_positive(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    metric: str,
    value: int | float | None,
    required: bool,
) -> None:
    if value is None:
        if required:
            findings.append(Finding(str(source), row_number, "error", "missing_metric", metric, "", "positive number", "metric is required"))
        return
    if value <= 0:
        findings.append(Finding(str(source), row_number, "error", "invalid_metric", metric, format_expected(value), "positive number", "metric must be positive"))


def memory_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> MemoryRow:
    findings: list[Finding] = []
    checks = 0
    tokens = finite_int(row.get("tokens"))
    memory_bytes = finite_int(row.get("memory_bytes"))
    memory_bytes_per_token = finite_float(row.get("memory_bytes_per_token"))
    host_child_peak_rss_bytes = finite_int(row.get("host_child_peak_rss_bytes"))
    host_rss_bytes_per_token = finite_float(row.get("host_rss_bytes_per_token"))
    host_rss_over_guest_ratio = (
        host_child_peak_rss_bytes / memory_bytes
        if host_child_peak_rss_bytes is not None and memory_bytes is not None and memory_bytes > 0
        else None
    )

    require_positive(findings, source=source, row_number=row_number, metric="tokens", value=tokens, required=args.require_tokens)
    require_positive(findings, source=source, row_number=row_number, metric="memory_bytes", value=memory_bytes, required=args.require_memory_bytes)
    require_positive(
        findings,
        source=source,
        row_number=row_number,
        metric="host_child_peak_rss_bytes",
        value=host_child_peak_rss_bytes,
        required=args.require_host_rss,
    )

    usable_tokens = tokens if tokens is not None and tokens > 0 else None
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="memory_bytes_per_token",
        stored=memory_bytes_per_token,
        expected=(memory_bytes / usable_tokens if memory_bytes is not None and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_rss_bytes_per_token",
        stored=host_rss_bytes_per_token,
        expected=(host_child_peak_rss_bytes / usable_tokens if host_child_peak_rss_bytes is not None and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )

    if memory_bytes is not None and host_child_peak_rss_bytes is not None:
        checks += 1
        if args.require_guest_memory_within_host_rss and memory_bytes > host_child_peak_rss_bytes:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "memory_bound_violation",
                    "memory_bytes",
                    str(memory_bytes),
                    f"<= {host_child_peak_rss_bytes}",
                    "guest-reported memory exceeds host child peak RSS",
                )
            )

        if args.max_host_rss_over_guest_ratio is not None:
            checks += 1
            if host_rss_over_guest_ratio is not None and host_rss_over_guest_ratio > args.max_host_rss_over_guest_ratio:
                findings.append(
                    Finding(
                        str(source),
                        row_number,
                        "error",
                        "host_rss_over_guest_ratio",
                        "host_rss_over_guest_ratio",
                        format_expected(host_rss_over_guest_ratio),
                        f"<= {format_expected(args.max_host_rss_over_guest_ratio)}",
                        "host child peak RSS exceeds allowed multiple of guest-reported memory",
                    )
                )

    row_findings = len(findings)
    args._findings.extend(findings)
    return MemoryRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        commit=row_text(row, "commit"),
        phase=row_text(row, "phase", default="measured"),
        exit_class=row_text(row, "exit_class"),
        tokens=tokens,
        memory_bytes=memory_bytes,
        memory_bytes_per_token=memory_bytes_per_token,
        host_child_peak_rss_bytes=host_child_peak_rss_bytes,
        host_rss_bytes_per_token=host_rss_bytes_per_token,
        host_rss_over_guest_ratio=host_rss_over_guest_ratio,
        checks=checks,
        findings=row_findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[MemoryRow], list[Finding]]:
    rows: list[MemoryRow] = []
    findings: list[Finding] = []
    args._findings = findings
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", "", "", str(exc)))
            continue
        for row_number, row in enumerate(loaded_rows, 1):
            phase = row_text(row, "phase", default="measured")
            exit_class = row_text(row, "exit_class")
            if args.measured_only and phase == "warmup":
                continue
            if args.ok_only and exit_class not in {"ok", "-"}:
                continue
            rows.append(memory_row(path, row_number, row, args))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", str(seen_files), str(args.min_artifacts), "not enough artifacts found"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "not enough benchmark rows found"))
    return rows, findings


def summary(rows: list[MemoryRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "findings": len(findings),
        "checks": sum(row.checks for row in rows),
        "memory_bytes_max": max((row.memory_bytes or 0 for row in rows), default=0),
        "host_child_peak_rss_bytes_max": max((row.host_child_peak_rss_bytes or 0 for row in rows), default=0),
        "host_rss_over_guest_ratio_max": max((row.host_rss_over_guest_ratio or 0 for row in rows), default=0),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
        "prompts": sorted({row.prompt for row in rows if row.prompt != "-"}),
    }


def write_json(path: Path, rows: list[MemoryRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[MemoryRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Memory Accounting Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Checks: {sum(row.checks for row in rows)}",
        f"Findings: {len(findings)}",
        "",
        "## Rows",
        "",
        "| Source | Row | Prompt | Memory Bytes | Host Peak RSS Bytes | Checks | Findings |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.source} | {row.row} | {row.prompt} | {row.memory_bytes or ''} | "
            f"{row.host_child_peak_rss_bytes or ''} | {row.checks} | {row.findings} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Stored | Expected | Detail |", "| --- | ---: | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.stored} | {finding.expected} | {finding.detail} |"
            )
    else:
        lines.append("No memory accounting findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[MemoryRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MemoryRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


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
            "name": "holyc_qemu_memory_accounting_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "memory_accounting"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} memory accounting finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_memory_accounting_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=0.001)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--allow-missing-tokens", dest="require_tokens", action="store_false")
    parser.add_argument("--require-memory-bytes", action="store_true")
    parser.add_argument("--require-host-rss", action="store_true")
    parser.add_argument("--require-guest-memory-within-host-rss", action="store_true")
    parser.add_argument(
        "--max-host-rss-over-guest-ratio",
        type=float,
        default=None,
        help="Fail if host_child_peak_rss_bytes / memory_bytes exceeds this ratio",
    )
    parser.add_argument("--all-phases", dest="measured_only", action="store_false", help="Include warmup rows")
    parser.add_argument("--all-exit-classes", dest="ok_only", action="store_false", help="Include non-ok rows")
    parser.set_defaults(require_tokens=True, measured_only=True, ok_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")
    if args.relative_tolerance < 0 or args.absolute_tolerance < 0:
        parser.error("--relative-tolerance and --absolute-tolerance must be >= 0")
    if args.max_host_rss_over_guest_ratio is not None and args.max_host_rss_over_guest_ratio <= 0:
        parser.error("--max-host-rss-over-guest-ratio must be > 0")

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
