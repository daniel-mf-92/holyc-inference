#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark rows for output determinism.

This host-side tool reads benchmark artifacts only. It never launches QEMU and
never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
OUTPUT_HASH_KEYS = ("output_sha256", "response_sha256", "completion_sha256", "generated_sha256", "generated_text_sha256")
OUTPUT_TEXT_KEYS = ("output", "response", "completion", "generated_text")
TOKEN_KEYS = ("tokens", "generated_tokens", "output_tokens", "completion_tokens")


@dataclass(frozen=True)
class OutputRow:
    source: str
    row: int
    key: str
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    seed: str
    phase: str
    exit_class: str
    tokens: int | None
    output_sha256: str
    output_source: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    key: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def finite_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def first_int(row: dict[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        value = finite_int(row.get(key))
        if value is not None:
            return value
    return None


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
                if stripped:
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


def output_identity(row: dict[str, Any]) -> tuple[str, str]:
    for key in OUTPUT_HASH_KEYS:
        value = text_value(row.get(key)).strip().lower()
        if value:
            return value, key
    for key in OUTPUT_TEXT_KEYS:
        value = row.get(key)
        if value not in (None, ""):
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            return digest, f"{key}:derived"
    return "", ""


def include_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    phase = row_text(row, "phase", default="measured").lower()
    exit_class = row_text(row, "exit_class").lower()
    if args.only_measured and phase != "measured":
        return False
    return args.include_failed or exit_class == "ok"


def row_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    values: list[str] = []
    for field in args.key_fields:
        values.append(row_text(row, field, default="-"))
    return "|".join(values)


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[OutputRow], list[Finding]]:
    rows: list[OutputRow] = []
    findings: list[Finding] = []
    grouped: dict[str, list[OutputRow]] = defaultdict(list)

    for path in iter_input_files(paths, args.pattern):
        for row_number, raw in enumerate(load_rows(path), 1):
            if not include_row(raw, args):
                continue
            key = row_key(raw, args)
            output_sha, output_source = output_identity(raw)
            tokens = first_int(raw, TOKEN_KEYS)
            row = OutputRow(
                source=str(path),
                row=row_number,
                key=key,
                profile=row_text(raw, "profile"),
                model=row_text(raw, "model"),
                quantization=row_text(raw, "quantization"),
                prompt=row_text(raw, "prompt", "prompt_id"),
                commit=row_text(raw, "commit", "git_commit"),
                seed=row_text(raw, "seed", "rng_seed"),
                phase=row_text(raw, "phase", default="measured").lower(),
                exit_class=row_text(raw, "exit_class").lower(),
                tokens=tokens,
                output_sha256=output_sha,
                output_source=output_source,
            )
            rows.append(row)
            grouped[key].append(row)
            if args.require_output_hash and not output_sha:
                findings.append(Finding(str(path), row_number, "error", "missing_output_identity", key, "output_sha256", "row has no output hash or output text field"))
            if args.require_tokens and tokens is None:
                findings.append(Finding(str(path), row_number, "error", "missing_tokens", key, "tokens", "row has no generated token count"))

    for key, group in sorted(grouped.items()):
        if len(group) < args.min_repeats:
            findings.append(
                Finding(group[0].source, group[0].row, "error", "min_repeats", key, "rows", f"key has {len(group)} row(s), expected at least {args.min_repeats}")
            )
            continue
        hashes = {row.output_sha256 for row in group if row.output_sha256}
        if len(hashes) > 1:
            findings.append(
                Finding(group[0].source, group[0].row, "error", "output_hash_drift", key, "output_sha256", f"key has {len(hashes)} distinct output hashes")
            )
        token_values = {row.tokens for row in group if row.tokens is not None}
        if len(token_values) > 1:
            findings.append(Finding(group[0].source, group[0].row, "error", "token_count_drift", key, "tokens", f"key has token counts {sorted(token_values)}"))

    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "", "rows", f"matched {len(rows)} row(s), expected at least {args.min_rows}"))
    return rows, findings


def report_payload(rows: list[OutputRow], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    groups = {row.key for row in rows}
    return {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "rows": len(rows),
            "groups": len(groups),
            "findings": len(findings),
            "min_rows": args.min_rows,
            "min_repeats": args.min_repeats,
            "key_fields": args.key_fields,
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not records:
            return
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Output Determinism Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        "",
        "| Rows | Groups | Findings | Min repeats |",
        "| ---: | ---: | ---: | ---: |",
        f"| {summary['rows']} | {summary['groups']} | {summary['findings']} | {summary['min_repeats']} |",
        "",
    ]
    if report["findings"]:
        lines += ["## Findings", "", "| Severity | Kind | Key | Field | Detail |", "| --- | --- | --- | --- | --- |"]
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['key']} | {finding['field']} | {finding['detail']} |")
    else:
        lines.append("No output determinism findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_output_determinism_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="output_determinism")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} output determinism finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.key}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_output_determinism_audit_latest")
    parser.add_argument("--key-field", dest="key_fields", action="append", default=["profile", "model", "quantization", "prompt", "commit", "seed"])
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-repeats", type=int, default=2)
    parser.add_argument("--require-output-hash", action="store_true")
    parser.add_argument("--require-tokens", action="store_true")
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--only-measured", action="store_true", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = report_payload(rows, findings, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    (stem.with_suffix(".json")).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(stem.with_suffix(".md"), report)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
