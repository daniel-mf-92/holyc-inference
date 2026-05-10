#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for token latency consistency.

This host-side tool reads benchmark JSON/JSONL/CSV artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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
class TokenLatencyRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    tokens: int | None
    elapsed_us: float | None
    wall_elapsed_us: float | None
    us_per_token: float | None
    wall_us_per_token: float | None
    expected_us_per_token: float | None
    expected_wall_us_per_token: float | None


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


def close_enough(actual: float, expected: float, tolerance_pct: float) -> bool:
    tolerance = max(0.01, abs(expected) * tolerance_pct / 100.0)
    return abs(actual - expected) <= tolerance


def include_row(raw: dict[str, Any], include_failed: bool) -> bool:
    phase = str(raw.get("phase") or "measured").lower()
    exit_class = str(raw.get("exit_class") or "").lower()
    if phase != "measured":
        return False
    return include_failed or exit_class == "ok"


def token_latency_row(source: Path, row_number: int, raw: dict[str, Any], args: argparse.Namespace) -> tuple[TokenLatencyRow, list[Finding]]:
    findings: list[Finding] = []
    tokens = finite_int(raw.get("tokens"))
    elapsed_us = finite_float(raw.get("elapsed_us"))
    wall_elapsed_us = finite_float(raw.get("wall_elapsed_us"))
    us_per_token = finite_float(raw.get("us_per_token"))
    wall_us_per_token = finite_float(raw.get("wall_us_per_token"))
    expected_us_per_token = elapsed_us / tokens if tokens and tokens > 0 and elapsed_us is not None else None
    expected_wall_us_per_token = wall_elapsed_us / tokens if tokens and tokens > 0 and wall_elapsed_us is not None else None

    label = "/".join(
        [
            row_text(raw, "profile"),
            row_text(raw, "model"),
            row_text(raw, "quantization"),
            row_text(raw, "prompt", "prompt_id"),
        ]
    )
    if tokens is None or tokens <= 0:
        findings.append(Finding(str(source), row_number, "error", "missing_tokens", "tokens", f"{label}: positive token count is required"))
    if elapsed_us is None or elapsed_us <= 0:
        findings.append(Finding(str(source), row_number, "error", "missing_elapsed_us", "elapsed_us", f"{label}: positive guest elapsed_us is required"))
    if wall_elapsed_us is None or wall_elapsed_us <= 0:
        findings.append(Finding(str(source), row_number, "error", "missing_wall_elapsed_us", "wall_elapsed_us", f"{label}: positive wall_elapsed_us is required"))

    if expected_us_per_token is not None:
        if us_per_token is None:
            findings.append(Finding(str(source), row_number, "error", "missing_us_per_token", "us_per_token", f"{label}: guest token latency is required"))
        elif not close_enough(us_per_token, expected_us_per_token, args.tolerance_pct):
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "us_per_token_drift",
                    "us_per_token",
                    f"{label}: stored {us_per_token:.6g}, expected {expected_us_per_token:.6g}",
                )
            )
        elif args.max_us_per_token is not None and us_per_token > args.max_us_per_token:
            findings.append(Finding(str(source), row_number, "error", "max_us_per_token", "us_per_token", f"{label}: {us_per_token:.6g} > {args.max_us_per_token:.6g}"))

    if expected_wall_us_per_token is not None:
        if wall_us_per_token is None:
            findings.append(Finding(str(source), row_number, "error", "missing_wall_us_per_token", "wall_us_per_token", f"{label}: wall token latency is required"))
        elif not close_enough(wall_us_per_token, expected_wall_us_per_token, args.tolerance_pct):
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "wall_us_per_token_drift",
                    "wall_us_per_token",
                    f"{label}: stored {wall_us_per_token:.6g}, expected {expected_wall_us_per_token:.6g}",
                )
            )
        elif args.max_wall_us_per_token is not None and wall_us_per_token > args.max_wall_us_per_token:
            findings.append(
                Finding(str(source), row_number, "error", "max_wall_us_per_token", "wall_us_per_token", f"{label}: {wall_us_per_token:.6g} > {args.max_wall_us_per_token:.6g}")
            )

    row = TokenLatencyRow(
        source=str(source),
        row=row_number,
        profile=row_text(raw, "profile"),
        model=row_text(raw, "model"),
        quantization=row_text(raw, "quantization"),
        prompt=row_text(raw, "prompt", "prompt_id"),
        phase=row_text(raw, "phase", default="measured").lower(),
        exit_class=row_text(raw, "exit_class").lower(),
        tokens=tokens,
        elapsed_us=elapsed_us,
        wall_elapsed_us=wall_elapsed_us,
        us_per_token=us_per_token,
        wall_us_per_token=wall_us_per_token,
        expected_us_per_token=expected_us_per_token,
        expected_wall_us_per_token=expected_wall_us_per_token,
    )
    return row, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[TokenLatencyRow], list[Finding]]:
    rows: list[TokenLatencyRow] = []
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
            row, row_findings = token_latency_row(path, row_number, raw, args)
            rows.append(row)
            findings.extend(row_findings)
    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


def summary(rows: list[TokenLatencyRow], findings: list[Finding]) -> dict[str, Any]:
    guest_latencies = [row.us_per_token for row in rows if row.us_per_token is not None]
    wall_latencies = [row.wall_us_per_token for row in rows if row.wall_us_per_token is not None]
    return {
        "rows": len(rows),
        "findings": len(findings),
        "max_us_per_token": max(guest_latencies, default=None),
        "max_wall_us_per_token": max(wall_latencies, default=None),
        "min_us_per_token": min(guest_latencies, default=None),
        "min_wall_us_per_token": min(wall_latencies, default=None),
    }


def write_json(path: Path, rows: list[TokenLatencyRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[TokenLatencyRow]) -> None:
    fields = list(TokenLatencyRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fields = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def format_number(value: float | None) -> str:
    return f"{value:.6g}" if value is not None else "n/a"


def write_markdown(path: Path, rows: list[TokenLatencyRow], findings: list[Finding]) -> None:
    stats = summary(rows, findings)
    lines = [
        "# QEMU Token Latency Audit",
        "",
        f"- Rows: {stats['rows']}",
        f"- Findings: {stats['findings']}",
        f"- Max guest us/token: {format_number(stats['max_us_per_token'])}",
        f"- Max wall us/token: {format_number(stats['max_wall_us_per_token'])}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} {finding.metric} {finding.detail}" for finding in findings)
    else:
        lines.append("No token latency findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_token_latency_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "token_latency"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} token latency finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for report artifacts")
    parser.add_argument("--output-stem", default="qemu_token_latency_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--include-failed", action="store_true", help="Audit failed measured rows in addition to OK rows")
    parser.add_argument("--tolerance-pct", type=float, default=0.001, help="Allowed formula drift percentage")
    parser.add_argument("--max-us-per-token", type=float, help="Optional guest latency gate")
    parser.add_argument("--max-wall-us-per-token", type=float, help="Optional wall-clock latency gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0 or args.min_rows < 0 or args.tolerance_pct < 0:
        parser.error("--min-artifacts, --min-rows, and --tolerance-pct must be >= 0")
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
