#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for prompt/token budget drift.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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
class BudgetRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    prompt_bytes: int | None
    guest_prompt_bytes: int | None
    expected_tokens: int | None
    tokens: int | None
    expected_tokens_match: bool | None
    prompt_bytes_over_budget: int | None
    tokens_over_budget: int | None


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


def bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    return None


def budget_row(source: Path, row_number: int, raw: dict[str, Any], args: argparse.Namespace) -> tuple[BudgetRow, list[Finding]]:
    findings: list[Finding] = []
    prompt = row_text(raw, "prompt", "prompt_id")
    prompt_bytes = finite_int(raw.get("prompt_bytes"))
    guest_prompt_bytes = finite_int(raw.get("guest_prompt_bytes"))
    expected_tokens = finite_int(raw.get("expected_tokens"))
    tokens = finite_int(raw.get("tokens"))
    expected_tokens_match = bool_or_none(raw.get("expected_tokens_match"))

    if prompt_bytes is None:
        findings.append(Finding(str(source), row_number, "error", "missing_prompt_bytes", "prompt_bytes", f"{prompt}: prompt byte count is required"))
    elif prompt_bytes <= 0:
        findings.append(Finding(str(source), row_number, "error", "nonpositive_prompt_bytes", "prompt_bytes", f"{prompt}: prompt byte count must be positive"))
    elif args.max_prompt_bytes is not None and prompt_bytes > args.max_prompt_bytes:
        over = prompt_bytes - args.max_prompt_bytes
        findings.append(Finding(str(source), row_number, "error", "prompt_bytes_over_budget", "prompt_bytes", f"{prompt}: {prompt_bytes} exceeds max {args.max_prompt_bytes} by {over}"))

    if args.require_guest_prompt_bytes:
        if guest_prompt_bytes is None:
            findings.append(Finding(str(source), row_number, "error", "missing_guest_prompt_bytes", "guest_prompt_bytes", f"{prompt}: guest prompt byte count is required"))
        elif prompt_bytes is not None and guest_prompt_bytes != prompt_bytes:
            findings.append(
                Finding(str(source), row_number, "error", "guest_prompt_bytes_mismatch", "guest_prompt_bytes", f"{prompt}: guest {guest_prompt_bytes} != host {prompt_bytes}")
            )

    if tokens is None:
        findings.append(Finding(str(source), row_number, "error", "missing_tokens", "tokens", f"{prompt}: token count is required"))
    elif tokens < 0:
        findings.append(Finding(str(source), row_number, "error", "negative_tokens", "tokens", f"{prompt}: token count must be nonnegative"))
    elif args.max_tokens is not None and tokens > args.max_tokens:
        over = tokens - args.max_tokens
        findings.append(Finding(str(source), row_number, "error", "tokens_over_budget", "tokens", f"{prompt}: {tokens} exceeds max {args.max_tokens} by {over}"))

    if args.require_expected_tokens:
        if expected_tokens is None:
            findings.append(Finding(str(source), row_number, "error", "missing_expected_tokens", "expected_tokens", f"{prompt}: expected token budget is required"))
        elif tokens is not None and tokens != expected_tokens:
            findings.append(Finding(str(source), row_number, "error", "expected_tokens_mismatch", "tokens", f"{prompt}: tokens {tokens} != expected {expected_tokens}"))
        if expected_tokens_match is False:
            findings.append(Finding(str(source), row_number, "error", "expected_tokens_match_false", "expected_tokens_match", f"{prompt}: stored expected_tokens_match is false"))

    row = BudgetRow(
        source=str(source),
        row=row_number,
        profile=row_text(raw, "profile"),
        model=row_text(raw, "model"),
        quantization=row_text(raw, "quantization"),
        prompt=prompt,
        phase=row_text(raw, "phase", default="measured").lower(),
        exit_class=row_text(raw, "exit_class").lower(),
        prompt_bytes=prompt_bytes,
        guest_prompt_bytes=guest_prompt_bytes,
        expected_tokens=expected_tokens,
        tokens=tokens,
        expected_tokens_match=expected_tokens_match,
        prompt_bytes_over_budget=prompt_bytes - args.max_prompt_bytes if prompt_bytes is not None and args.max_prompt_bytes is not None and prompt_bytes > args.max_prompt_bytes else None,
        tokens_over_budget=tokens - args.max_tokens if tokens is not None and args.max_tokens is not None and tokens > args.max_tokens else None,
    )
    return row, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[BudgetRow], list[Finding]]:
    rows: list[BudgetRow] = []
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
            row, row_findings = budget_row(path, row_number, raw, args)
            rows.append(row)
            findings.extend(row_findings)
    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))
    return rows, findings


def summary(rows: list[BudgetRow], findings: list[Finding]) -> dict[str, Any]:
    prompt_bytes = [row.prompt_bytes for row in rows if row.prompt_bytes is not None]
    tokens = [row.tokens for row in rows if row.tokens is not None]
    mismatches = sum(1 for row in rows if row.expected_tokens_match is False)
    return {
        "rows": len(rows),
        "findings": len(findings),
        "max_prompt_bytes": max(prompt_bytes, default=None),
        "max_tokens": max(tokens, default=None),
        "expected_token_mismatch_rows": mismatches,
        "prompt_byte_budget_failures": sum(1 for row in rows if row.prompt_bytes_over_budget is not None),
        "token_budget_failures": sum(1 for row in rows if row.tokens_over_budget is not None),
    }


def write_json(path: Path, rows: list[BudgetRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[BudgetRow]) -> None:
    fields = list(BudgetRow.__dataclass_fields__)
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


def write_markdown(path: Path, rows: list[BudgetRow], findings: list[Finding]) -> None:
    stats = summary(rows, findings)
    lines = [
        "# QEMU Prompt Budget Audit",
        "",
        f"- Rows: {stats['rows']}",
        f"- Findings: {stats['findings']}",
        f"- Max prompt bytes: {stats['max_prompt_bytes'] if stats['max_prompt_bytes'] is not None else 'n/a'}",
        f"- Max tokens: {stats['max_tokens'] if stats['max_tokens'] is not None else 'n/a'}",
        f"- Expected-token mismatch rows: {stats['expected_token_mismatch_rows']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} {finding.metric} {finding.detail}" for finding in findings)
    else:
        lines.append("No prompt budget findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_budget_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "prompt_budget"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} prompt budget finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for report artifacts")
    parser.add_argument("--output-stem", default="qemu_prompt_budget_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--include-failed", action="store_true", help="Audit failed measured rows in addition to OK rows")
    parser.add_argument("--max-prompt-bytes", type=int, help="Optional prompt byte budget")
    parser.add_argument("--max-tokens", type=int, help="Optional emitted token budget")
    parser.add_argument("--require-expected-tokens", action="store_true", help="Require expected_tokens and matching token counts")
    parser.add_argument("--require-guest-prompt-bytes", action="store_true", help="Require guest prompt byte echo telemetry to match host bytes")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")
    if args.max_prompt_bytes is not None and args.max_prompt_bytes < 0:
        parser.error("--max-prompt-bytes must be >= 0")
    if args.max_tokens is not None and args.max_tokens < 0:
        parser.error("--max-tokens must be >= 0")
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
