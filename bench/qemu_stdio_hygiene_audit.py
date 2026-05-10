#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for stdio hygiene.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class StdioRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    stdout_bytes: int | None
    stderr_bytes: int | None
    stdout_tail_bytes: int
    stderr_tail_bytes: int
    failure_reason_bytes: int
    has_failure_signal: bool


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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def text_bytes(value: Any) -> int:
    if value in (None, ""):
        return 0
    return len(str(value).encode("utf-8"))


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


def add_counter_findings(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    metric: str,
    stored: int | None,
    tail_bytes: int,
) -> None:
    if stored is None:
        findings.append(
            Finding(str(source), row_number, "error", "missing_byte_counter", metric, "byte counter is absent")
        )
        return
    if stored < 0:
        findings.append(
            Finding(str(source), row_number, "error", "negative_byte_counter", metric, "byte counter is negative")
        )
    if stored < tail_bytes:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "tail_exceeds_counter",
                metric,
                f"{metric}={stored} is smaller than captured tail bytes {tail_bytes}",
            )
        )


def stdio_row(source: Path, row_number: int, raw: dict[str, Any], args: argparse.Namespace) -> tuple[StdioRow, list[Finding]]:
    findings: list[Finding] = []
    stdout_bytes = finite_int(raw.get("stdout_bytes"))
    stderr_bytes = finite_int(raw.get("stderr_bytes"))
    stdout_tail_bytes = text_bytes(raw.get("stdout_tail"))
    stderr_tail_bytes = text_bytes(raw.get("stderr_tail"))
    failure_reason_bytes = text_bytes(raw.get("failure_reason"))
    exit_class = row_text(raw, "exit_class", default="ok").lower()
    timed_out = parse_bool(raw.get("timed_out"))

    add_counter_findings(
        findings,
        source=source,
        row_number=row_number,
        metric="stdout_bytes",
        stored=stdout_bytes,
        tail_bytes=stdout_tail_bytes,
    )
    add_counter_findings(
        findings,
        source=source,
        row_number=row_number,
        metric="stderr_bytes",
        stored=stderr_bytes,
        tail_bytes=stderr_tail_bytes,
    )

    ok_row = exit_class == "ok" and timed_out is not True
    if ok_row and not args.allow_ok_stderr and ((stderr_bytes or 0) > 0 or stderr_tail_bytes > 0):
        findings.append(
            Finding(str(source), row_number, "error", "ok_stderr_noise", "stderr", "OK row emitted stderr output")
        )

    budgets = (
        ("stdout_bytes", stdout_bytes, args.max_stdout_bytes),
        ("stderr_bytes", stderr_bytes, args.max_stderr_bytes),
        ("stdout_tail_bytes", stdout_tail_bytes, args.max_stdout_tail_bytes),
        ("stderr_tail_bytes", stderr_tail_bytes, args.max_stderr_tail_bytes),
        ("failure_reason_bytes", failure_reason_bytes, args.max_failure_reason_bytes),
    )
    for metric, value, budget in budgets:
        if value is not None and value > budget:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "stdio_budget_exceeded",
                    metric,
                    f"{metric}={value} exceeds budget {budget}",
                )
            )

    has_failure_signal = bool(failure_reason_bytes or stderr_tail_bytes or stdout_tail_bytes)
    if not ok_row and args.require_failure_signal and not has_failure_signal:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "silent_failure",
                "failure_signal",
                "non-OK row has no failure_reason, stderr_tail, or stdout_tail",
            )
        )

    row = StdioRow(
        source=str(source),
        row=row_number,
        profile=row_text(raw, "profile"),
        model=row_text(raw, "model"),
        quantization=row_text(raw, "quantization"),
        prompt=row_text(raw, "prompt", "prompt_id"),
        phase=row_text(raw, "phase", default="measured"),
        exit_class=exit_class,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_tail_bytes=stdout_tail_bytes,
        stderr_tail_bytes=stderr_tail_bytes,
        failure_reason_bytes=failure_reason_bytes,
        has_failure_signal=has_failure_signal,
    )
    return row, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[StdioRow], list[Finding]]:
    rows: list[StdioRow] = []
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
            row, row_findings = stdio_row(path, row_number, raw_row, args)
            rows.append(row)
            findings.extend(row_findings)

    if seen_files < args.min_artifacts:
        findings.append(
            Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}")
        )
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


def summary(rows: list[StdioRow], findings: list[Finding]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.exit_class == "ok"]
    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "failure_rows": len(rows) - len(ok_rows),
        "rows_with_stderr": sum(1 for row in rows if (row.stderr_bytes or 0) > 0 or row.stderr_tail_bytes > 0),
        "silent_failure_rows": sum(1 for row in rows if row.exit_class != "ok" and not row.has_failure_signal),
        "max_stdout_tail_bytes": max((row.stdout_tail_bytes for row in rows), default=0),
        "max_stderr_tail_bytes": max((row.stderr_tail_bytes for row in rows), default=0),
        "findings": len(findings),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
    }


def write_json(path: Path, rows: list[StdioRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[StdioRow], findings: list[Finding]) -> None:
    report = summary(rows, findings)
    lines = [
        "# QEMU Stdio Hygiene Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {report['rows']}",
        f"OK rows: {report['ok_rows']}",
        f"Failure rows: {report['failure_rows']}",
        f"Rows with stderr: {report['rows_with_stderr']}",
        f"Silent failure rows: {report['silent_failure_rows']}",
        f"Findings: {len(findings)}",
        "",
        "## Findings",
        "",
    ]
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.detail} |")
    else:
        lines.append("No stdio hygiene findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows_csv(path: Path, rows: list[StdioRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StdioRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_summary_csv(path: Path, rows: list[StdioRow], findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["metric", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, value in summary(rows, findings).items():
            if isinstance(value, list):
                value = ";".join(value)
            writer.writerow({"metric": metric, "value": value})


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_stdio_hygiene_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="qemu_stdio_hygiene")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} stdio hygiene finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.metric}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_stdio_hygiene_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--allow-ok-stderr", action="store_true")
    parser.add_argument("--max-stdout-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--max-stderr-bytes", type=int, default=64 * 1024)
    parser.add_argument("--max-stdout-tail-bytes", type=int, default=8192)
    parser.add_argument("--max-stderr-tail-bytes", type=int, default=8192)
    parser.add_argument("--max-failure-reason-bytes", type=int, default=1024)
    parser.add_argument("--no-require-failure-signal", dest="require_failure_signal", action="store_false")
    parser.set_defaults(require_failure_signal=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "max_stdout_bytes",
        "max_stderr_bytes",
        "max_stdout_tail_bytes",
        "max_stderr_tail_bytes",
        "max_failure_reason_bytes",
    ):
        if getattr(args, name) < 0:
            print(f"error: --{name.replace('_', '-')} must be non-negative", file=sys.stderr)
            return 2
    rows, findings = audit(args.paths, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_summary_csv(args.output_dir / f"{stem}.csv", rows, findings)
    write_rows_csv(args.output_dir / f"{stem}_rows.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
