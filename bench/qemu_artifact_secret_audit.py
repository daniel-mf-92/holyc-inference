#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for secret-like text leakage.

This host-side tool reads existing benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest. It scans captured command arguments,
stdio tails, failure reasons, and environment maps for high-confidence
credential patterns before results are retained or uploaded as CI artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_benchmark_matrix*.json")
DEFAULT_TEXT_PATTERNS = (
    "qemu_prompt_bench*.csv",
    "qemu_prompt_bench*.md",
    "qemu_prompt_bench*.xml",
    "qemu_benchmark_matrix*.csv",
    "qemu_benchmark_matrix*.md",
    "qemu_benchmark_matrix*.xml",
)
ROW_KEYS = ("benchmarks", "results", "runs", "rows", "cells", "warmups")
SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_\-.])("
    r"api[-_]?key|auth|authorization|bearer|client[-_]?secret|cookie|credential|password|private[-_]?key|secret|"
    r"(?:access|api|id|refresh|session)[-_]?token|set[-_]?cookie|x[-_]?api[-_]?key"
    r")(?:$|[_\-.])",
    re.IGNORECASE,
)
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("authorization_header", re.compile(r"\bauthorization\s*:\s*(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE)),
    ("cookie_header", re.compile(r"\b(?:set-cookie|cookie)\s*:\s*[A-Za-z0-9_.$!%*+`'~()<>:@,;=/?#\[\]{}|-]{12,}", re.IGNORECASE)),
    ("openai_api_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("stripe_secret_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
)
URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s'\"<>]+", re.IGNORECASE)


@dataclass(frozen=True)
class SecretRecord:
    source: str
    row: int
    text_fields_checked: int
    sensitive_fields_checked: int
    secret_findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    json_path: str
    detail: str
    redacted_sample: str


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
        elif path.is_file() and path.suffix.lower() == ".json":
            yield path


def iter_text_sidecars(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            seen: set[Path] = set()
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file() and path.suffix.lower() != ".json":
            yield path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    inherited = {key: value for key, value in payload.items() if key not in ROW_KEYS}
    rows: list[dict[str, Any]] = []
    for key in ROW_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    rows.append(merged)
    return rows or [payload]


def is_text_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def iter_text_fields(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from iter_text_fields(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_text_fields(item, f"{path}[{index}]")
        return
    if value is None:
        return
    if is_text_scalar(value):
        yield path, str(value)


def redacted(value: str, start: int, end: int, context: int = 18) -> str:
    prefix_start = max(0, start - context)
    suffix_end = min(len(value), end + context)
    prefix = value[prefix_start:start]
    suffix = value[end:suffix_end]
    if prefix_start > 0:
        prefix = "..." + prefix
    if suffix_end < len(value):
        suffix = suffix + "..."
    return f"{prefix}<redacted:{end - start} chars>{suffix}"


def url_has_credentials(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return bool(parsed.username or parsed.password)


def sensitive_field(path: str) -> bool:
    leaf = path.rsplit(".", 1)[-1]
    return bool(SENSITIVE_KEY_RE.search(leaf))


def blank_like_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in {"", "-", "none", "null", "redacted", "<redacted>", "***", "0", "false"}


def scan_text(source: Path, row_number: int, json_path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    kind,
                    json_path,
                    "high-confidence secret-like token found in benchmark artifact text",
                    redacted(text, match.start(), match.end()),
                )
            )
    for match in URL_RE.finditer(text):
        candidate = match.group(0)
        if url_has_credentials(candidate):
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    "error",
                    "url_embedded_credentials",
                    json_path,
                    "URL contains username or password material",
                    redacted(text, match.start(), match.end()),
                )
            )
    if sensitive_field(json_path) and not blank_like_secret(text):
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "sensitive_field_populated",
                json_path,
                "sensitive field name is populated in a benchmark artifact",
                redacted(text, 0, len(text)),
            )
        )
    return findings


def audit_row(source: Path, row_number: int, row: dict[str, Any]) -> tuple[SecretRecord, list[Finding]]:
    checked = 0
    sensitive_checked = 0
    findings: list[Finding] = []
    for json_path, text in iter_text_fields(row):
        checked += 1
        if sensitive_field(json_path):
            sensitive_checked += 1
        findings.extend(scan_text(source, row_number, json_path, text))
    return SecretRecord(str(source), row_number, checked, sensitive_checked, len(findings)), findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SecretRecord], list[Finding]]:
    records: list[SecretRecord] = []
    findings: list[Finding] = []
    seen_files = 0
    input_paths = list(paths)
    for path in iter_input_files(input_paths, args.pattern):
        seen_files += 1
        try:
            rows = flatten_rows(load_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "$", str(exc), ""))
            continue
        for row_number, row in enumerate(rows, 1):
            record, row_findings = audit_row(path, row_number, row)
            records.append(record)
            findings.extend(row_findings)

    if args.scan_text_sidecars:
        for path in iter_text_sidecars(input_paths, args.text_pattern):
            seen_files += 1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                findings.append(Finding(str(path), 0, "error", "decode_error", "$text", str(exc), ""))
                continue
            except OSError as exc:
                findings.append(Finding(str(path), 0, "error", "read_error", "$text", str(exc), ""))
                continue
            row_findings = scan_text(path, 1, "$text", text)
            records.append(SecretRecord(str(path), 1, 1, 0, len(row_findings)))
            findings.extend(row_findings)

    if seen_files < args.min_artifacts:
        findings.append(
            Finding("-", 0, "error", "min_artifacts", "$", f"found {seen_files}, expected at least {args.min_artifacts}", "")
        )
    if len(records) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "$", f"found {len(records)}, expected at least {args.min_rows}", ""))
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary(records: list[SecretRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "text_fields_checked": sum(record.text_fields_checked for record in records),
        "sensitive_fields_checked": sum(record.sensitive_fields_checked for record in records),
        "findings": len(findings),
        "sources": len({record.source for record in records}),
    }


def write_outputs(records: list[SecretRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if findings else "pass"
    payload = {
        "tool": "qemu_artifact_secret_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary(records, findings),
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(record) for record in records], list(SecretRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))

    report = summary(records, findings)
    lines = [
        "# QEMU Artifact Secret Audit",
        "",
        f"Status: {status}",
        f"Rows: {report['rows']}",
        f"Text fields checked: {report['text_fields_checked']}",
        f"Sensitive fields checked: {report['sensitive_fields_checked']}",
        f"Findings: {len(findings)}",
        "",
        "## Findings",
        "",
    ]
    if findings:
        lines.extend(["| Source | Row | Kind | JSON path | Detail | Sample |", "| --- | ---: | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.source} | {finding.row} | {finding.kind} | `{finding.json_path}` | {finding.detail} | `{finding.redacted_sample}` |"
            )
    else:
        lines.append("No secret-like artifact text findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element("testsuite", name="holyc_qemu_artifact_secret_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="qemu_artifact_secret_audit")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} secret-like artifact finding(s)")
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.json_path}" for finding in findings)
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern when an input is a directory")
    parser.add_argument(
        "--text-pattern",
        action="append",
        default=list(DEFAULT_TEXT_PATTERNS),
        help="text sidecar glob pattern when an input is a directory",
    )
    parser.add_argument(
        "--no-text-sidecars",
        action="store_false",
        dest="scan_text_sidecars",
        help="scan only JSON benchmark artifacts",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_artifact_secret_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.set_defaults(scan_text_sidecars=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0:
        parser.error("--min-artifacts must be >= 0")
    if args.min_rows < 0:
        parser.error("--min-rows must be >= 0")
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
