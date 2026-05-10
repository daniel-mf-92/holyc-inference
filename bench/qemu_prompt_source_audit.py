#!/usr/bin/env python3
"""Audit QEMU benchmark rows against their declared prompt-suite source.

This host-side tool reads saved benchmark JSON artifacts and local prompt
suite files only. It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class PromptSourceRow:
    source: str
    row: int
    list_name: str
    phase: str
    prompt: str
    prompt_sha256: str
    expected_prompt_sha256: str
    prompt_bytes: int | None
    expected_prompt_bytes: int
    expected_tokens: int | None
    suite_expected_tokens: int | None
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
    expected: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
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


def artifact_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def data_rows(payload: dict[str, Any], findings: list[Finding], path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for list_name in ("warmups", "benchmarks"):
        raw = payload.get(list_name)
        if raw is None:
            continue
        if not isinstance(raw, list):
            findings.append(Finding(str(path), 0, "error", f"invalid_{list_name}", list_name, "", "list", f"{list_name} must be a list"))
            continue
        for item in raw:
            if isinstance(item, dict):
                rows.append((list_name, item))
            else:
                findings.append(Finding(str(path), len(rows) + 1, "error", "invalid_row", list_name, "", "object", "benchmark row must be an object"))
    return rows


def load_prompt_map(path: Path, findings: list[Finding], artifact: Path) -> dict[str, qemu_prompt_bench.PromptCase]:
    try:
        cases = qemu_prompt_bench.load_prompt_cases(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding(str(artifact), 0, "error", "prompt_source_unreadable", "prompt_suite.source", str(path), "readable prompt suite", str(exc)))
        return {}

    prompt_map: dict[str, qemu_prompt_bench.PromptCase] = {}
    for case in cases:
        if case.prompt_id in prompt_map:
            findings.append(Finding(str(artifact), 0, "error", "duplicate_prompt_id", "prompt_suite.source", case.prompt_id, "unique prompt_id", "prompt suite has duplicate prompt IDs"))
        prompt_map[case.prompt_id] = case
    return prompt_map


def add_mismatch(
    findings: list[Finding],
    path: Path,
    row_number: int,
    kind: str,
    field: str,
    stored: object,
    expected: object,
) -> None:
    findings.append(Finding(str(path), row_number, "error", kind, field, "" if stored is None else str(stored), "" if expected is None else str(expected), f"{field} does not match prompt suite source"))


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[list[PromptSourceRow], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [Finding(str(path), 0, "error", "load_error", "artifact", "", "JSON object", error)]

    findings: list[Finding] = []
    suite = payload.get("prompt_suite")
    if not isinstance(suite, dict):
        findings.append(Finding(str(path), 0, "error", "missing_prompt_suite", "prompt_suite", "", "object", "prompt_suite object is absent"))
        return [], findings

    source = text(suite.get("source"))
    if not source:
        findings.append(Finding(str(path), 0, "error", "missing_prompt_source", "prompt_suite.source", "", "path", "prompt suite source is absent"))
        return [], findings

    prompt_source = artifact_path(Path.cwd(), source)
    if not prompt_source.exists():
        findings.append(Finding(str(path), 0, "error", "prompt_source_missing", "prompt_suite.source", source, "existing local file", "prompt suite source does not exist"))
        return [], findings

    prompt_map = load_prompt_map(prompt_source, findings, path)
    rows: list[PromptSourceRow] = []
    for row_number, (list_name, row) in enumerate(data_rows(payload, findings, path), 1):
        prompt_id = text(row.get("prompt") or row.get("prompt_id"))
        phase = text(row.get("phase")) or ("warmup" if list_name == "warmups" else "measured")
        prompt_case = prompt_map.get(prompt_id)
        checks = 0
        before = len(findings)
        if prompt_case is None:
            findings.append(Finding(str(path), row_number, "error", "unknown_prompt_id", "prompt", prompt_id, "prompt_id from prompt suite", "benchmark row prompt is absent from prompt suite source"))
            rows.append(PromptSourceRow(str(path), row_number, list_name, phase, prompt_id, text(row.get("prompt_sha256")), "", int_or_none(row.get("prompt_bytes")), 0, int_or_none(row.get("expected_tokens")), None, checks, len(findings) - before))
            continue

        expected_hash = qemu_prompt_bench.prompt_hash(prompt_case.prompt)
        expected_bytes = qemu_prompt_bench.prompt_bytes(prompt_case.prompt)
        stored_hash = text(row.get("prompt_sha256"))
        stored_bytes = int_or_none(row.get("prompt_bytes"))
        stored_tokens = int_or_none(row.get("expected_tokens"))
        checks += 1
        if stored_hash != expected_hash:
            add_mismatch(findings, path, row_number, "prompt_sha256_mismatch", "prompt_sha256", stored_hash, expected_hash)
        checks += 1
        if stored_bytes != expected_bytes:
            add_mismatch(findings, path, row_number, "prompt_bytes_mismatch", "prompt_bytes", stored_bytes, expected_bytes)
        if args.require_expected_tokens:
            checks += 1
            if stored_tokens != prompt_case.expected_tokens:
                add_mismatch(findings, path, row_number, "expected_tokens_mismatch", "expected_tokens", stored_tokens, prompt_case.expected_tokens)

        rows.append(
            PromptSourceRow(
                str(path),
                row_number,
                list_name,
                phase,
                prompt_id,
                stored_hash,
                expected_hash,
                stored_bytes,
                expected_bytes,
                stored_tokens,
                prompt_case.expected_tokens,
                checks,
                len(findings) - before,
            )
        )
    return rows, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PromptSourceRow], list[Finding]]:
    rows: list[PromptSourceRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        artifact_rows, artifact_findings = audit_artifact(path, args)
        rows.extend(artifact_rows)
        findings.extend(artifact_findings)
    if seen_files == 0:
        findings.append(Finding("", 0, "error", "no_input_files", "inputs", "", "benchmark artifact", "no benchmark artifacts matched input paths/patterns"))
    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "not enough benchmark rows checked"))
    return rows, findings


def build_report(rows: list[PromptSourceRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "checks": sum(row.checks for row in rows),
            "row_findings": sum(row.findings for row in rows),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[PromptSourceRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PromptSourceRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Prompt Source Audit",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- status: {report['status']}",
        f"- rows: {summary['rows']}",
        f"- checks: {summary['checks']}",
        f"- findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", "", "| source | row | kind | field | detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['source']} | {finding['row']} | {finding['kind']} | {finding['field']} | {finding['detail']} |")
    else:
        lines.append("No prompt source findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element("testsuite", {"name": "holyc_qemu_prompt_source_audit", "tests": "1", "failures": "1" if findings else "0"})
    testcase = ET.SubElement(testsuite, "testcase", {"name": "prompt_source"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for audit outputs")
    parser.add_argument("--output-stem", default="qemu_prompt_source_audit_latest", help="Output filename stem")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum rows required")
    parser.add_argument("--require-expected-tokens", action="store_true", help="Require row expected_tokens to match the prompt suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = build_report(rows, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
