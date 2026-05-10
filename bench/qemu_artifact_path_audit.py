#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for host-path portability.

This host-side tool reads existing benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest. It scans captured commands, stdio
tails, failure reasons, and environment maps for absolute host paths that make
artifacts machine-specific. Findings are warnings by default so historical
artifacts can be indexed, and can be promoted to errors in CI with
--fail-on-host-paths.
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
ROW_KEYS = ("benchmarks", "results", "runs", "rows", "cells", "warmups")
ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:Users|Volumes|home|mnt|opt|private|tmp|var)/[^\s'\"<>),;]+")
DEFAULT_ALLOWED_PREFIXES = (
    "/tmp/",
    "/private/tmp/",
    "/var/folders/",
    "/opt/homebrew/bin/",
    "/opt/homebrew/Cellar/qemu/",
    "/usr/bin/",
    "/bin/",
)


@dataclass(frozen=True)
class PathRecord:
    source: str
    row: int
    text_fields_checked: int
    absolute_paths_seen: int
    disallowed_paths_seen: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    json_path: str
    path: str
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


def normalize_path(candidate: str) -> str:
    return candidate.rstrip(".,:;]")


def is_allowed_path(path: str, allowed_prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in allowed_prefixes)


def scan_text(
    source: Path,
    row_number: int,
    json_path: str,
    text: str,
    allowed_prefixes: tuple[str, ...],
    severity: str,
) -> tuple[int, list[Finding]]:
    seen: set[str] = set()
    findings: list[Finding] = []
    for match in ABSOLUTE_PATH_RE.finditer(text):
        candidate = normalize_path(match.group(0))
        if candidate in seen:
            continue
        seen.add(candidate)
        if not is_allowed_path(candidate, allowed_prefixes):
            findings.append(
                Finding(
                    str(source),
                    row_number,
                    severity,
                    "host_path_leak",
                    json_path,
                    candidate,
                    "benchmark artifact contains a machine-specific absolute host path",
                )
            )
    return len(seen), findings


def audit_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[PathRecord, list[Finding]]:
    checked = 0
    absolute_paths = 0
    findings: list[Finding] = []
    severity = "error" if args.fail_on_host_paths else "warning"
    allowed_prefixes = tuple(args.allowed_prefix)
    for json_path, text in iter_text_fields(row):
        checked += 1
        count, text_findings = scan_text(source, row_number, json_path, text, allowed_prefixes, severity)
        absolute_paths += count
        findings.extend(text_findings)
    return PathRecord(str(source), row_number, checked, absolute_paths, len(findings)), findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PathRecord], list[Finding]]:
    records: list[PathRecord] = []
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


def summary(records: list[PathRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(records),
        "text_fields_checked": sum(record.text_fields_checked for record in records),
        "absolute_paths_seen": sum(record.absolute_paths_seen for record in records),
        "disallowed_paths_seen": sum(record.disallowed_paths_seen for record in records),
        "findings": len(findings),
        "sources": len({record.source for record in records}),
    }


def write_outputs(records: list[PathRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if any(finding.severity == "error" for finding in findings) else "pass"
    payload = {
        "tool": "qemu_artifact_path_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary(records, findings),
        "allowed_prefixes": list(args.allowed_prefix),
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / f"{stem}.csv",
        [asdict(record) for record in records],
        ["source", "row", "text_fields_checked", "absolute_paths_seen", "disallowed_paths_seen"],
    )
    write_csv(
        args.output_dir / f"{stem}_findings.csv",
        [asdict(finding) for finding in findings],
        ["source", "row", "severity", "kind", "json_path", "path", "detail"],
    )

    lines = [
        "# QEMU Artifact Path Audit",
        "",
        f"- Status: {status}",
        f"- Rows: {len(records)}",
        f"- Absolute paths seen: {payload['summary']['absolute_paths_seen']}",
        f"- Disallowed path findings: {payload['summary']['disallowed_paths_seen']}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings[:50]:
            location = f"{finding.source}:{finding.row}" if finding.source else "inputs"
            lines.append(f"- {finding.severity}: {finding.kind} at {location} `{finding.json_path}`: `{finding.path}`")
        if len(findings) > 50:
            lines.append(f"- ... {len(findings) - 50} more finding(s); see CSV.")
    else:
        lines.append("No host-path portability findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element(
        "testsuite",
        name="qemu_artifact_path_audit",
        tests=str(max(1, len(records))),
        failures=str(sum(1 for finding in findings if finding.severity == "error")),
    )
    if findings:
        case = ET.SubElement(suite, "testcase", name="host_path_policy")
        for finding in findings:
            if finding.severity == "error":
                failure = ET.SubElement(case, "failure", message=f"{finding.kind}: {finding.json_path}")
                failure.text = f"{finding.source}:{finding.row}: {finding.path} - {finding.detail}"
    else:
        ET.SubElement(suite, "testcase", name="host_path_policy")
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob for benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_artifact_path_audit_latest")
    parser.add_argument("--allowed-prefix", action="append", default=list(DEFAULT_ALLOWED_PREFIXES), help="Absolute path prefix allowed in retained artifacts")
    parser.add_argument("--fail-on-host-paths", action="store_true", help="Promote host path findings from warnings to errors")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if any(finding.severity == "error" for finding in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
