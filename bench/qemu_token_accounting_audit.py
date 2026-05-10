#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for token accounting consistency.

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
class AccountingRow:
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
    expected_tokens: int | None
    expected_tokens_match: bool | None
    prompt_bytes: int | None
    elapsed_us: float | None
    wall_elapsed_us: float | None
    timeout_seconds: float | None
    wall_timeout_pct: float | None
    host_overhead_us: float | None
    host_overhead_pct: float | None
    tok_per_s: float | None
    wall_tok_per_s: float | None
    prompt_bytes_per_s: float | None
    wall_prompt_bytes_per_s: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    tokens_per_prompt_byte: float | None
    memory_bytes: int | None
    memory_bytes_per_token: float | None
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


def bool_value(value: Any) -> bool | None:
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


def tolerance(expected: float, relative: float, absolute: float) -> float:
    return max(absolute, abs(expected) * relative)


def format_expected(value: float | int | bool | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{value:.9g}"


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
            Finding(str(source), row_number, "error", "missing_metric", metric, "", format_expected(expected), "derived metric is absent")
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


def accounting_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> AccountingRow:
    findings: list[Finding] = []
    checks = 0
    tokens = finite_int(row.get("tokens"))
    expected_tokens = finite_int(row.get("expected_tokens"))
    expected_tokens_match = bool_value(row.get("expected_tokens_match"))
    prompt_bytes = finite_int(row.get("prompt_bytes"))
    elapsed_us = finite_float(row.get("elapsed_us"))
    wall_elapsed_us = finite_float(row.get("wall_elapsed_us"))
    timeout_seconds = finite_float(row.get("timeout_seconds"))
    wall_timeout_pct = finite_float(row.get("wall_timeout_pct"))
    host_overhead_us = finite_float(row.get("host_overhead_us"))
    host_overhead_pct = finite_float(row.get("host_overhead_pct"))
    tok_per_s = finite_float(row.get("tok_per_s"))
    wall_tok_per_s = finite_float(row.get("wall_tok_per_s"))
    prompt_bytes_per_s = finite_float(row.get("prompt_bytes_per_s"))
    wall_prompt_bytes_per_s = finite_float(row.get("wall_prompt_bytes_per_s"))
    us_per_token = finite_float(row.get("us_per_token"))
    wall_us_per_token = finite_float(row.get("wall_us_per_token"))
    tokens_per_prompt_byte = finite_float(row.get("tokens_per_prompt_byte"))
    memory_bytes = finite_int(row.get("memory_bytes"))
    memory_bytes_per_token = finite_float(row.get("memory_bytes_per_token"))

    if tokens is None:
        if args.require_tokens:
            findings.append(Finding(str(source), row_number, "error", "missing_metric", "tokens", "", "positive integer", "token count is required"))
    elif tokens <= 0:
        findings.append(Finding(str(source), row_number, "error", "invalid_metric", "tokens", str(tokens), "positive integer", "token count must be positive"))

    if prompt_bytes is not None and prompt_bytes <= 0:
        findings.append(
            Finding(str(source), row_number, "error", "invalid_metric", "prompt_bytes", str(prompt_bytes), "positive integer", "prompt byte count must be positive")
        )
    usable_tokens = tokens if tokens is not None and tokens > 0 else None
    expected_host_overhead_us = (
        wall_elapsed_us - elapsed_us if elapsed_us is not None and wall_elapsed_us is not None else None
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_overhead_us",
        stored=host_overhead_us,
        expected=expected_host_overhead_us,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="host_overhead_pct",
        stored=host_overhead_pct,
        expected=(
            expected_host_overhead_us * 100.0 / elapsed_us
            if expected_host_overhead_us is not None and elapsed_us is not None and elapsed_us > 0
            else None
        ),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="wall_timeout_pct",
        stored=wall_timeout_pct,
        expected=(
            wall_elapsed_us * 100.0 / (timeout_seconds * 1_000_000.0)
            if wall_elapsed_us is not None and timeout_seconds is not None and timeout_seconds > 0
            else None
        ),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="us_per_token",
        stored=us_per_token,
        expected=(elapsed_us / usable_tokens if elapsed_us is not None and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="wall_us_per_token",
        stored=wall_us_per_token,
        expected=(wall_elapsed_us / usable_tokens if wall_elapsed_us is not None and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="tok_per_s",
        stored=tok_per_s,
        expected=(usable_tokens * 1_000_000.0 / elapsed_us if elapsed_us is not None and elapsed_us > 0 and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="wall_tok_per_s",
        stored=wall_tok_per_s,
        expected=(usable_tokens * 1_000_000.0 / wall_elapsed_us if wall_elapsed_us is not None and wall_elapsed_us > 0 and usable_tokens else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="prompt_bytes_per_s",
        stored=prompt_bytes_per_s,
        expected=(
            prompt_bytes * 1_000_000.0 / elapsed_us
            if prompt_bytes is not None and prompt_bytes > 0 and elapsed_us is not None and elapsed_us > 0
            else None
        ),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="wall_prompt_bytes_per_s",
        stored=wall_prompt_bytes_per_s,
        expected=(
            prompt_bytes * 1_000_000.0 / wall_elapsed_us
            if prompt_bytes is not None and prompt_bytes > 0 and wall_elapsed_us is not None and wall_elapsed_us > 0
            else None
        ),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="tokens_per_prompt_byte",
        stored=tokens_per_prompt_byte,
        expected=(usable_tokens / prompt_bytes if usable_tokens and prompt_bytes and prompt_bytes > 0 else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    checks += check_metric(
        findings,
        source=source,
        row_number=row_number,
        metric="memory_bytes_per_token",
        stored=memory_bytes_per_token,
        expected=(memory_bytes / usable_tokens if usable_tokens and memory_bytes is not None else None),
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )

    if args.require_expected_tokens and expected_tokens is None:
        findings.append(
            Finding(str(source), row_number, "error", "missing_metric", "expected_tokens", "", "integer", "expected token count is required")
        )
    if expected_tokens is not None and usable_tokens is not None:
        checks += 1
        expected_match = expected_tokens == usable_tokens
        if args.require_expected_tokens_match and expected_tokens_match is None:
            findings.append(
                Finding(str(source), row_number, "error", "missing_metric", "expected_tokens_match", "", format_expected(expected_match), "expected token match flag is required")
            )
        elif expected_tokens_match is not None and expected_tokens_match != expected_match:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "expected_tokens_match_drift",
                    "expected_tokens_match",
                    format_expected(expected_tokens_match),
                    format_expected(expected_match),
                    "flag must equal expected_tokens == tokens",
                )
            )
        if args.require_expected_tokens_match and not expected_match:
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "expected_tokens_mismatch",
                    "expected_tokens",
                    format_expected(tokens),
                    format_expected(expected_tokens),
                    "actual token count must match expected_tokens",
                )
            )

    row_findings = len(findings)
    args._findings.extend(findings)
    return AccountingRow(
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
        expected_tokens=expected_tokens,
        expected_tokens_match=expected_tokens_match,
        prompt_bytes=prompt_bytes,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        timeout_seconds=timeout_seconds,
        wall_timeout_pct=wall_timeout_pct,
        host_overhead_us=host_overhead_us,
        host_overhead_pct=host_overhead_pct,
        tok_per_s=tok_per_s,
        wall_tok_per_s=wall_tok_per_s,
        prompt_bytes_per_s=prompt_bytes_per_s,
        wall_prompt_bytes_per_s=wall_prompt_bytes_per_s,
        us_per_token=us_per_token,
        wall_us_per_token=wall_us_per_token,
        tokens_per_prompt_byte=tokens_per_prompt_byte,
        memory_bytes=memory_bytes,
        memory_bytes_per_token=memory_bytes_per_token,
        checks=checks,
        findings=row_findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[AccountingRow], list[Finding]]:
    rows: list[AccountingRow] = []
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
            rows.append(accounting_row(path, row_number, row, args))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", str(seen_files), str(args.min_artifacts), "not enough artifacts found"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "not enough benchmark rows found"))
    return rows, findings


def summary(rows: list[AccountingRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "findings": len(findings),
        "checks": sum(row.checks for row in rows),
        "tokens_total": sum(row.tokens or 0 for row in rows),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
        "prompts": sorted({row.prompt for row in rows if row.prompt != "-"}),
    }


def write_json(path: Path, rows: list[AccountingRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[AccountingRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Token Accounting Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Checks: {sum(row.checks for row in rows)}",
        f"Findings: {len(findings)}",
        "",
        "## Rows",
        "",
        "| Source | Row | Prompt | Tokens | Checks | Findings |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row.source} | {row.row} | {row.prompt} | {row.tokens or ''} | {row.checks} | {row.findings} |")
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Stored | Expected | Detail |", "| --- | ---: | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.stored} | {finding.expected} | {finding.detail} |"
            )
    else:
        lines.append("No token accounting findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[AccountingRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AccountingRow.__dataclass_fields__))
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
            "name": "holyc_qemu_token_accounting_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "token_accounting"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} token accounting finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_token_accounting_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=0.001)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--allow-missing-tokens", dest="require_tokens", action="store_false")
    parser.add_argument("--require-expected-tokens", action="store_true")
    parser.add_argument("--require-expected-tokens-match", action="store_true")
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
