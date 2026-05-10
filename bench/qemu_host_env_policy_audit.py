#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for host environment air-gap policy.

This host-side tool reads existing artifacts only. It never launches QEMU and
never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_benchmark_matrix*.json")
ROW_KEYS = ("benchmarks", "results", "runs", "rows", "cells")
ENV_KEYS = ("environment", "host_environment", "env", "qemu_environment")
DISALLOWED_ENV_NAMES = {
    "all_proxy",
    "ftp_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "rsync_proxy",
    "socks_proxy",
}
DISALLOWED_ENV_SUBSTRINGS = ("proxy",)
URL_SCHEMES = ("http://", "https://", "ftp://", "socks://", "socks5://")


@dataclass(frozen=True)
class EnvRecord:
    source: str
    row: int
    scope: str
    env_count: int
    network_env_count: int
    url_value_count: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    variable: str
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


def normalize_env(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for item in value:
            if isinstance(item, str) and "=" in item:
                key, raw_value = item.split("=", 1)
                result[key] = raw_value
            elif isinstance(item, dict) and "name" in item:
                result[str(item["name"])] = str(item.get("value", ""))
        return result
    return {}


def env_maps(raw: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    maps: list[tuple[str, dict[str, str]]] = []
    for key in ENV_KEYS:
        env = normalize_env(raw.get(key))
        if env:
            maps.append((key, env))
    return maps


def is_network_env_name(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in DISALLOWED_ENV_NAMES or any(part in lowered for part in DISALLOWED_ENV_SUBSTRINGS)


def value_contains_url(value: str) -> bool:
    lowered = value.strip().lower()
    return any(scheme in lowered for scheme in URL_SCHEMES)


def audit_env(source: Path, row_number: int, scope: str, env: dict[str, str], fail_on_url_values: bool) -> tuple[EnvRecord, list[Finding]]:
    findings: list[Finding] = []
    network_env_count = 0
    url_value_count = 0
    for name, value in sorted(env.items()):
        if is_network_env_name(name):
            network_env_count += 1
            findings.append(Finding(str(source), row_number, "error", "network_env_var", name, f"{scope}.{name} is not allowed in air-gapped QEMU artifacts"))
        if value_contains_url(value):
            url_value_count += 1
            severity = "error" if fail_on_url_values else "warning"
            findings.append(Finding(str(source), row_number, severity, "url_env_value", name, f"{scope}.{name} contains a URL-like value"))
    return EnvRecord(str(source), row_number, scope, len(env), network_env_count, url_value_count), findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[EnvRecord], list[Finding]]:
    records: list[EnvRecord] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            rows = flatten_rows(load_json(path))
        except (OSError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "read_error", "", str(exc)))
            continue
        for row_number, raw in enumerate(rows, 1):
            maps = env_maps(raw)
            if not maps and args.require_environment:
                findings.append(Finding(str(path), row_number, "error", "missing_environment", "", "no captured host environment map found"))
            for scope, env in maps:
                record, env_findings = audit_env(path, row_number, scope, env, args.fail_on_url_values)
                records.append(record)
                findings.extend(env_findings)
    if seen_files == 0:
        findings.append(Finding("", 0, "error", "no_inputs", "", "no input artifacts matched"))
    if len(records) < args.min_records:
        findings.append(Finding("", 0, "error", "min_records", "", f"{len(records)} environment record(s), need at least {args.min_records}"))
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(records: list[EnvRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    summary = {
        "records": len(records),
        "findings": len(findings),
        "network_env_vars": sum(record.network_env_count for record in records),
        "url_values": sum(record.url_value_count for record in records),
        "status": status,
    }
    payload = {
        "tool": "qemu_host_env_policy_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary,
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_dir / f"{stem}.csv", [asdict(record) for record in records], ["source", "row", "scope", "env_count", "network_env_count", "url_value_count"])
    write_csv(output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], ["source", "row", "severity", "kind", "variable", "detail"])
    lines = [
        "# QEMU Host Environment Policy Audit",
        "",
        f"- Status: {status}",
        f"- Environment records: {len(records)}",
        f"- Network environment variables: {summary['network_env_vars']}",
        f"- URL-like environment values: {summary['url_values']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            location = f"{finding.source}:{finding.row}" if finding.source else "inputs"
            variable = f" `{finding.variable}`" if finding.variable else ""
            lines.append(f"- {finding.severity}: {finding.kind}{variable} at {location}: {finding.detail}")
    else:
        lines.append("No host environment policy findings.")
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_host_env_policy_audit",
            "tests": "1",
            "failures": "1" if status == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "host_environment_policy"})
    if status == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} host environment policy finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_host_env_policy_audit_latest")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--require-environment", action="store_true")
    parser.add_argument("--fail-on-url-values", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_records < 0 or not math.isfinite(args.min_records):
        raise SystemExit("--min-records must be non-negative")
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
