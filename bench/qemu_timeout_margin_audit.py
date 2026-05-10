#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for timeout headroom.

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
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class TimeoutRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    iteration: int | None
    exit_class: str
    timed_out: bool | None
    timeout_seconds: float | None
    wall_elapsed_us: float | None
    wall_timeout_pct: float | None
    timeout_budget_us: float | None
    timeout_margin_us: float | None


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


def timeout_row(source: Path, row_number: int, row: dict[str, Any]) -> TimeoutRow:
    timeout_seconds = finite_float(row.get("timeout_seconds"))
    wall_elapsed_us = finite_float(row.get("wall_elapsed_us"))
    wall_timeout_pct = finite_float(row.get("wall_timeout_pct"))
    timeout_budget_us = timeout_seconds * 1_000_000.0 if timeout_seconds is not None else None
    timeout_margin_us = timeout_budget_us - wall_elapsed_us if timeout_budget_us is not None and wall_elapsed_us is not None else None
    return TimeoutRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        phase=row_text(row, "phase", default="measured").lower(),
        iteration=finite_int(row.get("iteration")),
        exit_class=row_text(row, "exit_class").lower(),
        timed_out=parse_bool(row.get("timed_out")),
        timeout_seconds=timeout_seconds,
        wall_elapsed_us=wall_elapsed_us,
        wall_timeout_pct=wall_timeout_pct,
        timeout_budget_us=timeout_budget_us,
        timeout_margin_us=timeout_margin_us,
    )


def row_label(row: TimeoutRow) -> str:
    return f"{row.profile}/{row.model}/{row.quantization}/{row.prompt}/{row.phase}/{row.iteration or '-'}"


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[TimeoutRow], list[Finding]]:
    rows: list[TimeoutRow] = []
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
            row = timeout_row(path, row_number, raw_row)
            rows.append(row)
            label = row_label(row)
            if row.timeout_seconds is None or row.timeout_seconds <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_timeout_seconds", "timeout_seconds", f"{label}: timeout_seconds must be positive"))
            if row.wall_elapsed_us is None or row.wall_elapsed_us <= 0:
                findings.append(Finding(row.source, row.row, "error", "missing_wall_elapsed_us", "wall_elapsed_us", f"{label}: wall_elapsed_us must be positive"))
            if row.timeout_budget_us is not None and row.wall_elapsed_us is not None and row.wall_elapsed_us > row.timeout_budget_us:
                findings.append(Finding(row.source, row.row, "error", "timeout_budget_exceeded", "wall_elapsed_us", f"{label}: wall_elapsed_us exceeds timeout budget"))
            if row.timeout_budget_us and row.wall_elapsed_us is not None:
                expected_pct = row.wall_elapsed_us * 100.0 / row.timeout_budget_us
                if row.wall_timeout_pct is None:
                    findings.append(Finding(row.source, row.row, "error", "missing_wall_timeout_pct", "wall_timeout_pct", f"{label}: wall_timeout_pct is required"))
                elif abs(row.wall_timeout_pct - expected_pct) > args.pct_tolerance:
                    findings.append(Finding(row.source, row.row, "error", "wall_timeout_pct_drift", "wall_timeout_pct", f"{label}: stored {row.wall_timeout_pct:.6g}, expected {expected_pct:.6g}"))
            if row.exit_class == "ok" and row.wall_timeout_pct is not None and row.wall_timeout_pct > args.max_ok_timeout_pct:
                findings.append(Finding(row.source, row.row, "error", "ok_timeout_margin_too_small", "wall_timeout_pct", f"{label}: OK row used {row.wall_timeout_pct:.6g}% of timeout budget"))
            if (
                args.min_timeout_timeout_pct is not None
                and (row.exit_class == "timeout" or row.timed_out is True)
                and row.wall_timeout_pct is not None
                and row.wall_timeout_pct < args.min_timeout_timeout_pct
            ):
                findings.append(
                    Finding(
                        row.source,
                        row.row,
                        "error",
                        "timeout_budget_underused",
                        "wall_timeout_pct",
                        f"{label}: timeout row used {row.wall_timeout_pct:.6g}% of timeout budget, below {args.min_timeout_timeout_pct:.6g}%",
                    )
                )

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


def write_json(path: Path, rows: list[TimeoutRow], findings: list[Finding]) -> None:
    ok_rows = sum(1 for row in rows if row.exit_class == "ok")
    timeout_rows = sum(1 for row in rows if row.exit_class == "timeout" or row.timed_out is True)
    max_pct = max((row.wall_timeout_pct for row in rows if row.wall_timeout_pct is not None), default=None)
    min_margin = min((row.timeout_margin_us for row in rows if row.timeout_margin_us is not None), default=None)
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "rows": len(rows),
            "ok_rows": ok_rows,
            "timeout_rows": timeout_rows,
            "findings": len(findings),
            "max_wall_timeout_pct": max_pct,
            "min_timeout_margin_us": min_margin,
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[TimeoutRow]) -> None:
    fields = list(TimeoutRow.__dataclass_fields__)
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


def write_markdown(path: Path, rows: list[TimeoutRow], findings: list[Finding]) -> None:
    timeout_rows = sum(1 for row in rows if row.exit_class == "timeout" or row.timed_out is True)
    max_pct = max((row.wall_timeout_pct for row in rows if row.wall_timeout_pct is not None), default=None)
    min_margin = min((row.timeout_margin_us for row in rows if row.timeout_margin_us is not None), default=None)
    lines = [
        "# QEMU Timeout Margin Audit",
        "",
        f"- Rows: {len(rows)}",
        f"- Timeout rows: {timeout_rows}",
        f"- Findings: {len(findings)}",
        f"- Max wall timeout pct: {max_pct:.6g}" if max_pct is not None else "- Max wall timeout pct: n/a",
        f"- Min timeout margin us: {min_margin:.6g}" if min_margin is not None else "- Min timeout margin us: n/a",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} {finding.metric} {finding.detail}" for finding in findings)
    else:
        lines.append("No timeout margin findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_timeout_margin_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "timeout_margin"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} timeout margin findings"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for report artifacts")
    parser.add_argument("--output-stem", default="qemu_timeout_margin_audit_latest", help="Report filename stem")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-ok-timeout-pct", type=float, default=90.0, help="Maximum timeout budget percentage allowed for OK rows")
    parser.add_argument("--min-timeout-timeout-pct", type=float, default=90.0, help="Minimum timeout budget percentage expected for timeout rows")
    parser.add_argument("--pct-tolerance", type=float, default=0.001, help="Absolute tolerance for stored wall_timeout_pct")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
