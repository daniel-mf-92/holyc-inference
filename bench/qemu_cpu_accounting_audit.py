#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for host CPU accounting consistency.

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
class CpuRow:
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
    wall_elapsed_us: float | None
    host_child_user_cpu_us: float | None
    host_child_system_cpu_us: float | None
    host_child_cpu_us: float | None
    host_child_cpu_pct: float | None
    host_child_tok_per_cpu_s: float | None
    host_child_peak_rss_bytes: int | None
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
        findings.append(
            Finding(str(source), row_number, "error", "missing_metric", metric, "", format_expected(expected), "derived CPU metric is absent")
        )
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


def cpu_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> CpuRow:
    findings: list[Finding] = []
    checks = 0
    tokens = finite_int(row.get("tokens"))
    wall_elapsed_us = finite_float(row.get("wall_elapsed_us"))
    user_cpu_us = finite_float(row.get("host_child_user_cpu_us"))
    system_cpu_us = finite_float(row.get("host_child_system_cpu_us"))
    cpu_us = finite_float(row.get("host_child_cpu_us"))
    cpu_pct = finite_float(row.get("host_child_cpu_pct"))
    tok_per_cpu_s = finite_float(row.get("host_child_tok_per_cpu_s"))
    peak_rss_bytes = finite_int(row.get("host_child_peak_rss_bytes"))

    for metric, value in (
        ("host_child_user_cpu_us", user_cpu_us),
        ("host_child_system_cpu_us", system_cpu_us),
        ("host_child_cpu_us", cpu_us),
        ("wall_elapsed_us", wall_elapsed_us),
    ):
        if value is None:
            if args.require_cpu_metrics:
                findings.append(Finding(str(source), row_number, "error", "missing_metric", metric, "", "number", "CPU accounting input is required"))
        elif value < 0:
            findings.append(Finding(str(source), row_number, "error", "invalid_metric", metric, format_expected(value), "non-negative", "CPU accounting value must be non-negative"))

    if tokens is not None and tokens <= 0:
        findings.append(Finding(str(source), row_number, "error", "invalid_metric", "tokens", str(tokens), "positive integer", "token count must be positive"))

    expected_cpu_us = user_cpu_us + system_cpu_us if user_cpu_us is not None and system_cpu_us is not None else None
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_child_cpu_us",
        stored=cpu_us,
        expected=expected_cpu_us,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_child_cpu_pct",
        stored=cpu_pct,
        expected=(expected_cpu_us * 100.0 / wall_elapsed_us if expected_cpu_us is not None and wall_elapsed_us and wall_elapsed_us > 0 else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_child_tok_per_cpu_s",
        stored=tok_per_cpu_s,
        expected=(tokens * 1_000_000.0 / expected_cpu_us if tokens and tokens > 0 and expected_cpu_us and expected_cpu_us > 0 else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )

    if args.max_host_child_cpu_pct is not None:
        if cpu_pct is None:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "missing_metric",
                    "host_child_cpu_pct",
                    "",
                    str(args.max_host_child_cpu_pct),
                    "CPU percentage gate requires telemetry",
                )
            )
        elif cpu_pct > args.max_host_child_cpu_pct:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "max_host_child_cpu_pct",
                    "host_child_cpu_pct",
                    format_expected(cpu_pct),
                    format_expected(args.max_host_child_cpu_pct),
                    "host child CPU percentage exceeds gate",
                )
            )

    if args.min_host_child_tok_per_cpu_s is not None:
        if tok_per_cpu_s is None:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "missing_metric",
                    "host_child_tok_per_cpu_s",
                    "",
                    str(args.min_host_child_tok_per_cpu_s),
                    "CPU efficiency gate requires telemetry",
                )
            )
        elif tok_per_cpu_s < args.min_host_child_tok_per_cpu_s:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "min_host_child_tok_per_cpu_s",
                    "host_child_tok_per_cpu_s",
                    format_expected(tok_per_cpu_s),
                    format_expected(args.min_host_child_tok_per_cpu_s),
                    "host child token throughput per CPU second is below gate",
                )
            )

    row_findings = len(findings)
    args._findings.extend(findings)
    return CpuRow(
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
        wall_elapsed_us=wall_elapsed_us,
        host_child_user_cpu_us=user_cpu_us,
        host_child_system_cpu_us=system_cpu_us,
        host_child_cpu_us=cpu_us,
        host_child_cpu_pct=cpu_pct,
        host_child_tok_per_cpu_s=tok_per_cpu_s,
        host_child_peak_rss_bytes=peak_rss_bytes,
        checks=checks,
        findings=row_findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[CpuRow], list[Finding]]:
    rows: list[CpuRow] = []
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
            rows.append(cpu_row(path, row_number, row, args))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", str(seen_files), str(args.min_artifacts), "not enough artifacts found"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "not enough benchmark rows found"))
    return rows, findings


def summary(rows: list[CpuRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "findings": len(findings),
        "checks": sum(row.checks for row in rows),
        "cpu_us_total": sum(row.host_child_cpu_us or 0 for row in rows),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
        "prompts": sorted({row.prompt for row in rows if row.prompt != "-"}),
    }


def write_json(path: Path, rows: list[CpuRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[CpuRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU CPU Accounting Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Checks: {sum(row.checks for row in rows)}",
        f"Findings: {len(findings)}",
        "",
        "## Rows",
        "",
        "| Source | Row | Prompt | Child CPU us | CPU % | Tok/CPU s | Checks | Findings |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.source} | {row.row} | {row.prompt} | {row.host_child_cpu_us or ''} | {row.host_child_cpu_pct or ''} | {row.host_child_tok_per_cpu_s or ''} | {row.checks} | {row.findings} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Stored | Expected | Detail |", "| --- | ---: | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.stored} | {finding.expected} | {finding.detail} |"
            )
    else:
        lines.append("No CPU accounting findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[CpuRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CpuRow.__dataclass_fields__))
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
            "name": "holyc_qemu_cpu_accounting_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "cpu_accounting"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} CPU accounting findings"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind} {finding.metric}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("bench/results")], help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern when scanning directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_cpu_accounting_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    parser.add_argument("--require-cpu-metrics", action="store_true", help="Fail when CPU accounting inputs are missing")
    parser.add_argument("--max-host-child-cpu-pct", type=float, help="Fail rows whose host child CPU percentage exceeds this value")
    parser.add_argument("--min-host-child-tok-per-cpu-s", type=float, help="Fail rows whose tokens per host child CPU second fall below this value")
    parser.add_argument("--include-warmup", dest="measured_only", action="store_false", help="Include warmup rows")
    parser.add_argument("--include-failures", dest="ok_only", action="store_false", help="Include non-ok benchmark rows")
    parser.set_defaults(measured_only=True, ok_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
