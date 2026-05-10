#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for network endpoint text.

This host-side tool reads existing benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest. It scans captured commands, stdio
tails, failure reasons, and metadata for URL-like or endpoint-like network text
that would indicate air-gap drift in retained benchmark artifacts. It can also
scan retained CSV, Markdown, and XML sidecars so endpoint text cannot bypass the
JSON artifact gate.
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
NETWORK_SCHEME_RE = re.compile(r"\b(?:https?|ftp|ssh|sftp|ws|wss|tcp|udp|tls)://[^\s'\"<>]+", re.IGNORECASE)
QEMU_ENDPOINT_RE = re.compile(r"\b(?:tcp|udp|unix|telnet|websocket):[^\s'\"<>]+", re.IGNORECASE)
IP_ENDPOINT_RE = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}:\d{1,5}\b"
)
NETWORK_KEYWORD_RE = re.compile(
    r"\b(?:dhcp|dns|ethernet|hostfwd|guestfwd|http|https|listen|socket|tcp|tls|udp|websocket)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NetworkTextRecord:
    source: str
    row: int
    artifact_kind: str
    text_fields_checked: int
    endpoint_findings: int
    keyword_findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    json_path: str
    sample: str
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


def redacted_sample(text: str, start: int, end: int, context: int = 28) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    prefix = text[left:start]
    suffix = text[end:right]
    if left > 0:
        prefix = "..." + prefix
    if right < len(text):
        suffix += "..."
    return f"{prefix}{text[start:end]}{suffix}"


def scan_text(source: Path, row_number: int, json_path: str, text: str, fail_on_keywords: bool) -> list[Finding]:
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    endpoint_patterns = (
        ("network_url", NETWORK_SCHEME_RE, "network URL-like text found in benchmark artifact"),
        ("qemu_endpoint", QEMU_ENDPOINT_RE, "QEMU endpoint-like text found in benchmark artifact"),
        ("ip_endpoint", IP_ENDPOINT_RE, "IP:port endpoint found in benchmark artifact"),
    )
    for kind, pattern, detail in endpoint_patterns:
        for match in pattern.finditer(text):
            sample = redacted_sample(text, match.start(), match.end())
            key = (kind, sample)
            if key in seen:
                continue
            seen.add(key)
            findings.append(Finding(str(source), row_number, "error", kind, json_path, sample, detail))

    for match in NETWORK_KEYWORD_RE.finditer(text):
        keyword = match.group(0).lower()
        key = ("network_keyword", keyword)
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            Finding(
                str(source),
                row_number,
                "error" if fail_on_keywords else "warning",
                "network_keyword",
                json_path,
                redacted_sample(text, match.start(), match.end()),
                "network-related keyword found in benchmark artifact text",
            )
        )
    return findings


def audit_row(
    source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace
) -> tuple[NetworkTextRecord, list[Finding]]:
    checked = 0
    findings: list[Finding] = []
    for json_path, text in iter_text_fields(row):
        checked += 1
        findings.extend(scan_text(source, row_number, json_path, text, args.fail_on_keywords))
    endpoint_findings = sum(1 for finding in findings if finding.kind != "network_keyword")
    keyword_findings = sum(1 for finding in findings if finding.kind == "network_keyword")
    return NetworkTextRecord(str(source), row_number, "json", checked, endpoint_findings, keyword_findings), findings


def audit_text_sidecar(source: Path, args: argparse.Namespace) -> tuple[NetworkTextRecord, list[Finding]]:
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        finding = Finding(str(source), 0, "error", "load_error", "$", "", str(exc))
        return NetworkTextRecord(str(source), 0, "text_sidecar", 0, 0, 0), [finding]

    findings = scan_text(source, 0, "$", text, args.fail_on_keywords)
    endpoint_findings = sum(1 for finding in findings if finding.kind != "network_keyword")
    keyword_findings = sum(1 for finding in findings if finding.kind == "network_keyword")
    return NetworkTextRecord(str(source), 0, "text_sidecar", 1, endpoint_findings, keyword_findings), findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[NetworkTextRecord], list[Finding]]:
    records: list[NetworkTextRecord] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            rows = flatten_rows(load_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "$", "", str(exc)))
            continue
        for row_number, row in enumerate(rows, 1):
            record, row_findings = audit_row(path, row_number, row, args)
            records.append(record)
            findings.extend(row_findings)
    if args.scan_text_sidecars:
        for path in iter_text_sidecars(paths, args.text_pattern):
            seen_files += 1
            record, sidecar_findings = audit_text_sidecar(path, args)
            records.append(record)
            findings.extend(sidecar_findings)

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "$", "", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(records) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "$", "", f"found {len(records)}, expected at least {args.min_rows}"))
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary(records: list[NetworkTextRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "json_rows": sum(1 for record in records if record.artifact_kind == "json"),
        "text_sidecars": sum(1 for record in records if record.artifact_kind == "text_sidecar"),
        "text_fields_checked": sum(record.text_fields_checked for record in records),
        "endpoint_findings": sum(record.endpoint_findings for record in records),
        "keyword_findings": sum(record.keyword_findings for record in records),
        "findings": len(findings),
        "sources": len({record.source for record in records}),
    }


def write_outputs(records: list[NetworkTextRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    payload = {
        "tool": "qemu_artifact_network_text_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary(records, findings),
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", [asdict(record) for record in records], list(NetworkTextRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))

    report = payload["summary"]
    lines = [
        "# QEMU Artifact Network Text Audit",
        "",
        f"Status: {status}",
        f"Rows: {report['rows']}",
        f"JSON rows: {report['json_rows']}",
        f"Text sidecars: {report['text_sidecars']}",
        f"Text fields checked: {report['text_fields_checked']}",
        f"Endpoint findings: {report['endpoint_findings']}",
        f"Keyword findings: {report['keyword_findings']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", "", "| Source | Row | Severity | Kind | JSON path | Sample |", "| --- | ---: | --- | --- | --- | --- |"])
        for finding in findings[:100]:
            lines.append(
                f"| {finding.source} | {finding.row} | {finding.severity} | {finding.kind} | `{finding.json_path}` | `{finding.sample}` |"
            )
        if len(findings) > 100:
            lines.append(f"| ... |  |  |  |  | {len(findings) - 100} more finding(s); see CSV. |")
    else:
        lines.append("No network endpoint text findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    failures = sum(1 for finding in findings if finding.severity == "error")
    suite = ET.Element("testsuite", name="holyc_qemu_artifact_network_text_audit", tests="1", failures=str(1 if failures else 0))
    case = ET.SubElement(suite, "testcase", name="qemu_artifact_network_text_audit")
    if failures:
        failure = ET.SubElement(case, "failure", message=f"{failures} network endpoint artifact finding(s)")
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.json_path}" for finding in findings if finding.severity == "error")
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern when an input is a directory")
    parser.add_argument("--text-pattern", action="append", default=list(DEFAULT_TEXT_PATTERNS), help="text sidecar glob pattern when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_artifact_network_text_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--no-text-sidecars", dest="scan_text_sidecars", action="store_false", help="scan JSON artifacts only")
    parser.add_argument("--fail-on-keywords", action="store_true", help="promote standalone network keyword matches from warnings to errors")
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
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
