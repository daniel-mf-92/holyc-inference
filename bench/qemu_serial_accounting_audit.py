#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for serial output accounting.

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
class SerialRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    phase: str
    exit_class: str
    stdout_bytes: int | None
    stderr_bytes: int | None
    serial_output_bytes: int | None
    expected_serial_output_bytes: int | None
    stdout_lines: int | None
    stderr_lines: int | None
    serial_output_lines: int | None
    expected_serial_output_lines: int | None
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


def text_bytes(value: Any) -> int:
    if value in (None, ""):
        return 0
    return len(str(value).encode("utf-8"))


def text_lines(value: Any) -> int:
    if value in (None, ""):
        return 0
    return len(str(value).splitlines())


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


def format_value(value: int | None) -> str:
    return "" if value is None else str(value)


def require_nonnegative(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    metric: str,
    value: int | None,
    required: bool,
) -> None:
    if value is None:
        if required:
            findings.append(Finding(str(source), row_number, "error", "missing_metric", metric, "", "non-negative integer", "metric is required"))
        return
    if value < 0:
        findings.append(Finding(str(source), row_number, "error", "invalid_metric", metric, str(value), "non-negative integer", "metric must be non-negative"))


def check_sum(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    metric: str,
    stored: int | None,
    expected: int | None,
    required: bool,
) -> int:
    if expected is None:
        return 0
    if stored is None:
        if required:
            findings.append(Finding(str(source), row_number, "error", "missing_metric", metric, "", str(expected), "derived serial metric is absent"))
        return 0
    if stored != expected:
        findings.append(Finding(str(source), row_number, "error", "metric_drift", metric, str(stored), str(expected), "stored serial metric does not match stdout plus stderr"))
    return 1


def serial_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> SerialRow:
    findings: list[Finding] = []
    stdout_bytes = finite_int(row.get("stdout_bytes"))
    stderr_bytes = finite_int(row.get("stderr_bytes"))
    serial_output_bytes = finite_int(row.get("serial_output_bytes"))
    stdout_lines = finite_int(row.get("stdout_lines"))
    stderr_lines = finite_int(row.get("stderr_lines"))
    serial_output_lines = finite_int(row.get("serial_output_lines"))

    if stdout_bytes is None and "stdout_tail" in row and args.derive_min_tail_metrics:
        stdout_bytes = text_bytes(row.get("stdout_tail"))
    if stderr_bytes is None and "stderr_tail" in row and args.derive_min_tail_metrics:
        stderr_bytes = text_bytes(row.get("stderr_tail"))
    if stdout_lines is None and "stdout_tail" in row and args.derive_min_tail_metrics:
        stdout_lines = text_lines(row.get("stdout_tail"))
    if stderr_lines is None and "stderr_tail" in row and args.derive_min_tail_metrics:
        stderr_lines = text_lines(row.get("stderr_tail"))

    for metric, value, required in (
        ("stdout_bytes", stdout_bytes, args.require_metrics),
        ("stderr_bytes", stderr_bytes, args.require_metrics),
        ("serial_output_bytes", serial_output_bytes, args.require_metrics),
        ("serial_output_lines", serial_output_lines, args.require_metrics),
        ("stdout_lines", stdout_lines, args.require_line_components),
        ("stderr_lines", stderr_lines, args.require_line_components),
    ):
        require_nonnegative(findings, source=source, row_number=row_number, metric=metric, value=value, required=required)

    expected_bytes = stdout_bytes + stderr_bytes if stdout_bytes is not None and stderr_bytes is not None else None
    expected_lines = stdout_lines + stderr_lines if stdout_lines is not None and stderr_lines is not None else None
    checks = 0
    checks += check_sum(
        findings,
        source=source,
        row_number=row_number,
        metric="serial_output_bytes",
        stored=serial_output_bytes,
        expected=expected_bytes,
        required=args.require_metrics,
    )
    checks += check_sum(
        findings,
        source=source,
        row_number=row_number,
        metric="serial_output_lines",
        stored=serial_output_lines,
        expected=expected_lines,
        required=args.require_metrics,
    )

    return SerialRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        commit=row_text(row, "commit"),
        phase=row_text(row, "phase", default="measured"),
        exit_class=row_text(row, "exit_class"),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        serial_output_bytes=serial_output_bytes,
        expected_serial_output_bytes=expected_bytes,
        stdout_lines=stdout_lines,
        stderr_lines=stderr_lines,
        serial_output_lines=serial_output_lines,
        expected_serial_output_lines=expected_lines,
        checks=checks,
        findings=len(findings),
    ), findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SerialRow], list[Finding]]:
    rows: list[SerialRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "-", "", "", str(exc)))
            continue
        for row_number, raw_row in enumerate(loaded_rows, 1):
            phase = row_text(raw_row, "phase", default="measured")
            exit_class = row_text(raw_row, "exit_class")
            if args.measured_only and phase == "warmup":
                continue
            if args.ok_only and exit_class not in {"ok", "-"}:
                continue
            row, row_findings = serial_row(path, row_number, raw_row, args)
            rows.append(row)
            findings.extend(row_findings)

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", str(seen_files), str(args.min_artifacts), "too few artifacts matched inputs"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "too few rows matched filters"))
    return rows, findings


def summary(rows: list[SerialRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "findings": len(findings),
        "checks": sum(row.checks for row in rows),
        "serial_output_bytes_total": sum(row.serial_output_bytes or 0 for row in rows),
        "serial_output_lines_total": sum(row.serial_output_lines or 0 for row in rows),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
    }


def write_json(path: Path, rows: list[SerialRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[SerialRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Serial Accounting Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Checks: {sum(row.checks for row in rows)}",
        f"Findings: {len(findings)}",
        "",
        "## Rows",
        "",
        "| Source | Row | Prompt | Serial bytes | Expected bytes | Serial lines | Expected lines | Findings |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.source} | {row.row} | {row.prompt} | {format_value(row.serial_output_bytes)} | "
            f"{format_value(row.expected_serial_output_bytes)} | {format_value(row.serial_output_lines)} | "
            f"{format_value(row.expected_serial_output_lines)} | {row.findings} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Stored | Expected | Detail |", "| --- | ---: | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.stored} | {finding.expected} | {finding.detail} |")
    else:
        lines.append("No serial accounting findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[SerialRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SerialRow.__dataclass_fields__))
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
            "name": "holyc_qemu_serial_accounting_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "serial_accounting"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} serial accounting finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_serial_accounting_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-metrics", action="store_true", help="Require stdout/stderr byte counters plus serial byte/line counters")
    parser.add_argument("--require-line-components", action="store_true", help="Require stdout_lines/stderr_lines and enforce their sum against serial_output_lines")
    parser.add_argument("--derive-min-tail-metrics", action="store_true", help="Use stdout_tail/stderr_tail as lower-bound counters when explicit counters are absent")
    parser.add_argument("--all-phases", dest="measured_only", action="store_false", help="Include warmup rows")
    parser.add_argument("--all-exit-classes", dest="ok_only", action="store_false", help="Include non-ok rows")
    parser.set_defaults(measured_only=True, ok_only=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")

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
