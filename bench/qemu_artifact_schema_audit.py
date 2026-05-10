#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for schema completeness.

This host-side tool reads existing benchmark JSON artifacts only. It never
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
KNOWN_ARTIFACT_SCHEMA_VERSIONS = {"qemu-prompt-bench/v1"}
ROW_KEYS = ("warmups", "benchmarks")
TOP_LEVEL_REQUIRED = (
    "generated_at",
    "status",
    "profile",
    "model",
    "quantization",
    "command",
    "command_sha256",
    "command_airgap",
    "prompt_suite",
    "suite_summary",
)
MEASURED_REQUIRED_TEXT = (
    "benchmark",
    "profile",
    "model",
    "quantization",
    "phase",
    "prompt",
    "prompt_sha256",
    "command_sha256",
    "exit_class",
    "timestamp",
)
MEASURED_REQUIRED_NUMERIC = (
    "tokens",
    "elapsed_us",
    "wall_elapsed_us",
    "tok_per_s",
    "wall_tok_per_s",
    "us_per_token",
    "wall_us_per_token",
    "returncode",
    "timeout_seconds",
)


@dataclass(frozen=True)
class SchemaRecord:
    source: str
    row: int
    scope: str
    phase: str
    prompt: str
    required_fields: int
    present_fields: int
    missing_fields: int
    invalid_fields: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


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


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "artifact root must be a JSON object"
    return payload, ""


def present(value: Any) -> bool:
    return value not in (None, "")


def finite_number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rows_from_payload(payload: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    rows: list[tuple[str, int, dict[str, Any]]] = []
    for key in ROW_KEYS:
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for index, row in enumerate(value, 1):
            if isinstance(row, dict):
                rows.append((key, index, row))
    return rows


def add_missing(findings: list[Finding], source: Path, row: int, field: str, scope: str) -> None:
    findings.append(Finding(str(source), row, "error", "missing_field", field, f"{scope} is missing required field {field!r}"))


def add_invalid(findings: list[Finding], source: Path, row: int, field: str, scope: str, detail: str) -> None:
    findings.append(Finding(str(source), row, "error", "invalid_field", field, f"{scope}.{field}: {detail}"))


def audit_top_level(source: Path, payload: dict[str, Any]) -> tuple[SchemaRecord, list[Finding]]:
    findings: list[Finding] = []
    present_fields = 0
    invalid_fields = 0
    for field in TOP_LEVEL_REQUIRED:
        if present(payload.get(field)):
            present_fields += 1
        else:
            add_missing(findings, source, 0, field, "artifact")
    status = payload.get("status")
    if present(status) and str(status) not in {"pass", "fail"}:
        invalid_fields += 1
        add_invalid(findings, source, 0, "status", "artifact", f"expected 'pass' or 'fail', got {status!r}")
    generated_at = payload.get("generated_at")
    if present(generated_at) and parse_iso(str(generated_at)) is None:
        invalid_fields += 1
        add_invalid(findings, source, 0, "generated_at", "artifact", "expected timezone-aware ISO timestamp")
    artifact_schema_version = payload.get("artifact_schema_version")
    if present(artifact_schema_version) and str(artifact_schema_version) not in KNOWN_ARTIFACT_SCHEMA_VERSIONS:
        invalid_fields += 1
        add_invalid(
            findings,
            source,
            0,
            "artifact_schema_version",
            "artifact",
            f"expected one of {sorted(KNOWN_ARTIFACT_SCHEMA_VERSIONS)}, got {artifact_schema_version!r}",
        )
    command_airgap = payload.get("command_airgap")
    if present(command_airgap) and not isinstance(command_airgap, dict):
        invalid_fields += 1
        add_invalid(findings, source, 0, "command_airgap", "artifact", "expected object")
    prompt_suite = payload.get("prompt_suite")
    if present(prompt_suite) and not isinstance(prompt_suite, dict):
        invalid_fields += 1
        add_invalid(findings, source, 0, "prompt_suite", "artifact", "expected object")
    suite_summary = payload.get("suite_summary")
    if present(suite_summary) and not isinstance(suite_summary, dict):
        invalid_fields += 1
        add_invalid(findings, source, 0, "suite_summary", "artifact", "expected object")
    record = SchemaRecord(
        source=str(source),
        row=0,
        scope="artifact",
        phase="",
        prompt="",
        required_fields=len(TOP_LEVEL_REQUIRED),
        present_fields=present_fields,
        missing_fields=len(TOP_LEVEL_REQUIRED) - present_fields,
        invalid_fields=invalid_fields,
    )
    return record, findings


def audit_row(source: Path, scope: str, index: int, row: dict[str, Any]) -> tuple[SchemaRecord, list[Finding]]:
    findings: list[Finding] = []
    phase = str(row.get("phase") or "")
    prompt = str(row.get("prompt") or row.get("prompt_id") or "")
    required = list(MEASURED_REQUIRED_TEXT) + list(MEASURED_REQUIRED_NUMERIC)
    present_fields = 0
    invalid_fields = 0
    location = f"{scope}[{index}]"

    for field in MEASURED_REQUIRED_TEXT:
        if present(row.get(field)):
            present_fields += 1
        else:
            add_missing(findings, source, index, field, location)
    for field in MEASURED_REQUIRED_NUMERIC:
        number = finite_number(row.get(field))
        if number is None:
            add_missing(findings, source, index, field, location)
            continue
        present_fields += 1
        if field != "returncode" and number <= 0:
            invalid_fields += 1
            add_invalid(findings, source, index, field, location, f"expected positive finite number, got {row.get(field)!r}")
        if field == "tokens" and not number.is_integer():
            invalid_fields += 1
            add_invalid(findings, source, index, field, location, f"expected integer token count, got {row.get(field)!r}")
    if phase and phase not in {"warmup", "measured"}:
        invalid_fields += 1
        add_invalid(findings, source, index, "phase", location, f"expected warmup or measured, got {phase!r}")
    timestamp = row.get("timestamp")
    if present(timestamp) and parse_iso(str(timestamp)) is None:
        invalid_fields += 1
        add_invalid(findings, source, index, "timestamp", location, "expected timezone-aware ISO timestamp")

    record = SchemaRecord(
        source=str(source),
        row=index,
        scope=scope,
        phase=phase,
        prompt=prompt,
        required_fields=len(required),
        present_fields=present_fields,
        missing_fields=len(required) - present_fields,
        invalid_fields=invalid_fields,
    )
    return record, findings


def audit_artifact(path: Path) -> tuple[list[SchemaRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [Finding(str(path), 0, "error", "load_error", "artifact", error)]
    records: list[SchemaRecord] = []
    findings: list[Finding] = []
    top_record, top_findings = audit_top_level(path, payload)
    records.append(top_record)
    findings.extend(top_findings)
    rows = rows_from_payload(payload)
    if not rows:
        findings.append(Finding(str(path), 0, "error", "missing_rows", "benchmarks", "artifact has no warmup or benchmark rows"))
    for scope, index, raw in rows:
        row_record, row_findings = audit_row(path, scope, index, raw)
        records.append(row_record)
        findings.extend(row_findings)
    return records, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SchemaRecord], list[Finding]]:
    records: list[SchemaRecord] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        artifact_records, artifact_findings = audit_artifact(path)
        records.extend(artifact_records)
        findings.extend(artifact_findings)
    if seen_files < args.min_artifacts:
        findings.append(Finding("", 0, "error", "min_artifacts", "inputs", f"found {seen_files}, expected at least {args.min_artifacts}"))
    row_records = [record for record in records if record.scope in ROW_KEYS]
    if len(row_records) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "benchmarks", f"found {len(row_records)}, expected at least {args.min_rows}"))
    return records, findings


def summary(records: list[SchemaRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "records": len(records),
        "artifacts": sum(1 for record in records if record.scope == "artifact"),
        "rows": sum(1 for record in records if record.scope in ROW_KEYS),
        "missing_fields": sum(record.missing_fields for record in records),
        "invalid_fields": sum(record.invalid_fields for record in records),
        "findings": len(findings),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(records: list[SchemaRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "pass" if not findings else "fail"
    stats = summary(records, findings)
    payload = {
        "tool": "qemu_artifact_schema_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": stats,
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(record) for record in records], list(SchemaRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    lines = [
        "# QEMU Artifact Schema Audit",
        "",
        f"- Status: {status}",
        f"- Artifacts: {stats['artifacts']}",
        f"- Rows: {stats['rows']}",
        f"- Missing fields: {stats['missing_fields']}",
        f"- Invalid fields: {stats['invalid_fields']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding.severity}: {finding.kind} `{finding.field}` at {finding.source}:{finding.row}: {finding.detail}" for finding in findings)
    else:
        lines.append("No QEMU artifact schema findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_artifact_schema_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "schema_completeness"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} schema finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for report artifacts")
    parser.add_argument("--output-stem", default="qemu_artifact_schema_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
