#!/usr/bin/env python3
"""Audit saved QEMU benchmark rows against serial BENCH_RESULT payloads.

This host-side tool reads existing qemu_prompt_bench JSON artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
RESULT_LINE_RE = re.compile(r"(?:BENCH_RESULT|bench_result)\s*[:=]\s*(\{.*\})")


@dataclass(frozen=True)
class PayloadRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    exit_class: str
    payload_count: int
    tokens: int | None
    payload_tokens: int | None
    elapsed_us: int | None
    payload_elapsed_us: int | None
    ttft_us: int | None
    payload_ttft_us: int | None
    memory_bytes: int | None
    payload_memory_bytes: int | None
    prompt_bytes: int | None
    payload_prompt_bytes: int | None
    prompt_sha256: str
    payload_prompt_sha256: str
    checks: int
    findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    field: str
    stored: str
    payload: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    yield from flatten_json_payload(payload)


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


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def result_payloads(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    text = "\n".join(
        value for value in (text_value(row.get("stdout_tail")), text_value(row.get("stderr_tail"))) if value
    )
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for match in RESULT_LINE_RE.finditer(text):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid BENCH_RESULT JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append("BENCH_RESULT payload must be a JSON object")
            continue
        payloads.append(payload)
    return payloads, errors


def value_text(value: Any) -> str:
    return "" if value is None else str(value)


def compare_int(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    field: str,
    stored: int | None,
    payload: int | None,
) -> int:
    if stored is None and payload is None:
        return 0
    if stored != payload:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "payload_mismatch",
                field,
                value_text(stored),
                value_text(payload),
                f"normalized {field} differs from serial BENCH_RESULT payload",
            )
        )
    return 1


def compare_text(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    field: str,
    stored: str,
    payload: str,
) -> int:
    if not stored and not payload:
        return 0
    if stored != payload:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "payload_mismatch",
                field,
                stored,
                payload,
                f"normalized {field} differs from serial BENCH_RESULT payload",
            )
        )
    return 1


def audit_row(source: Path, row_number: int, raw: dict[str, Any], args: argparse.Namespace) -> tuple[PayloadRow, list[Finding]]:
    findings: list[Finding] = []
    payloads, parse_errors = result_payloads(raw)
    for error in parse_errors:
        findings.append(Finding(str(source), row_number, "error", "invalid_payload", "stdout_tail", "", "", error))

    exit_class = row_text(raw, "exit_class", default="ok").lower()
    ok_row = exit_class == "ok"
    if ok_row and args.require_ok_payload and not payloads:
        findings.append(
            Finding(str(source), row_number, "error", "missing_payload", "stdout_tail", "", "", "OK row has no BENCH_RESULT payload in captured serial tails")
        )
    if len(payloads) > 1 and not args.allow_multiple_payloads:
        findings.append(
            Finding(str(source), row_number, "error", "multiple_payloads", "stdout_tail", "", str(len(payloads)), "row has multiple BENCH_RESULT payloads")
        )

    payload = payloads[-1] if payloads else {}
    tokens = finite_int(raw.get("tokens"))
    payload_tokens = finite_int(payload.get("tokens"))
    elapsed_us = finite_int(raw.get("elapsed_us"))
    payload_elapsed_us = finite_int(payload.get("elapsed_us"))
    ttft_us = finite_int(raw.get("ttft_us"))
    payload_ttft_us = finite_int(payload.get("time_to_first_token_us"))
    memory_bytes = finite_int(raw.get("memory_bytes"))
    payload_memory_bytes = finite_int(payload.get("memory_bytes"))
    prompt_bytes = finite_int(raw.get("guest_prompt_bytes"))
    if prompt_bytes is None:
        prompt_bytes = finite_int(raw.get("prompt_bytes"))
    payload_prompt_bytes = finite_int(payload.get("prompt_bytes"))
    prompt_sha256 = text_value(raw.get("guest_prompt_sha256")) or text_value(raw.get("prompt_sha256"))
    payload_prompt_sha256 = text_value(payload.get("prompt_sha256"))

    checks = 0
    if payloads:
        checks += compare_int(findings, source=source, row_number=row_number, field="tokens", stored=tokens, payload=payload_tokens)
        checks += compare_int(findings, source=source, row_number=row_number, field="elapsed_us", stored=elapsed_us, payload=payload_elapsed_us)
        checks += compare_int(findings, source=source, row_number=row_number, field="ttft_us", stored=ttft_us, payload=payload_ttft_us)
        checks += compare_int(findings, source=source, row_number=row_number, field="memory_bytes", stored=memory_bytes, payload=payload_memory_bytes)
        checks += compare_int(findings, source=source, row_number=row_number, field="prompt_bytes", stored=prompt_bytes, payload=payload_prompt_bytes)
        checks += compare_text(
            findings,
            source=source,
            row_number=row_number,
            field="prompt_sha256",
            stored=prompt_sha256,
            payload=payload_prompt_sha256,
        )

    return (
        PayloadRow(
            source=str(source),
            row=row_number,
            profile=row_text(raw, "profile"),
            model=row_text(raw, "model"),
            quantization=row_text(raw, "quantization"),
            prompt=row_text(raw, "prompt", "prompt_id"),
            phase=row_text(raw, "phase", default="measured"),
            exit_class=exit_class,
            payload_count=len(payloads),
            tokens=tokens,
            payload_tokens=payload_tokens,
            elapsed_us=elapsed_us,
            payload_elapsed_us=payload_elapsed_us,
            ttft_us=ttft_us,
            payload_ttft_us=payload_ttft_us,
            memory_bytes=memory_bytes,
            payload_memory_bytes=payload_memory_bytes,
            prompt_bytes=prompt_bytes,
            payload_prompt_bytes=payload_prompt_bytes,
            prompt_sha256=prompt_sha256,
            payload_prompt_sha256=payload_prompt_sha256,
            checks=checks,
            findings=len(findings),
        ),
        findings,
    )


def audit(inputs: Iterable[Path], args: argparse.Namespace) -> tuple[list[PayloadRow], list[Finding]]:
    rows: list[PayloadRow] = []
    findings: list[Finding] = []
    patterns = args.pattern or list(DEFAULT_PATTERNS)
    files = list(iter_input_files(inputs, patterns))
    if not files:
        findings.append(Finding("-", 0, "error", "no_artifacts", "inputs", "", "", "no matching benchmark artifacts found"))
        return rows, findings

    for path in files:
        try:
            raw_rows = list(load_rows(path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", "", "", str(exc)))
            continue
        for row_number, raw in enumerate(raw_rows, 1):
            row, row_findings = audit_row(path, row_number, raw, args)
            rows.append(row)
            findings.extend(row_findings)

    if args.min_rows and len(rows) < args.min_rows:
        findings.append(
            Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "not enough rows checked")
        )
    return rows, findings


def write_json(path: Path, rows: list[PayloadRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "findings": len(findings),
            "rows_with_payload": sum(1 for row in rows if row.payload_count > 0),
            "checks": sum(row.checks for row in rows),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[PayloadRow]) -> None:
    fields = list(PayloadRow.__dataclass_fields__)
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


def write_markdown(path: Path, rows: list[PayloadRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Serial Payload Audit",
        "",
        f"- Rows: {len(rows)}",
        f"- Rows with payload: {sum(1 for row in rows if row.payload_count > 0)}",
        f"- Checks: {sum(row.checks for row in rows)}",
        f"- Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding.severity}: {finding.kind} row={finding.row} field={finding.field} - {finding.detail}")
    else:
        lines.append("No serial payload findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_serial_payload_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "serial_payload"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} serial payload findings"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON artifacts or directories to audit")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern when input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_serial_payload_audit_latest")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-ok-payload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-multiple-payloads", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
