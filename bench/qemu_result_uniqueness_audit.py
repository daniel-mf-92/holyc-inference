#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for duplicate result identities.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
class ResultRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    prompt_sha256: str
    phase: str
    iteration: int | None
    launch_index: int | None
    commit: str
    command_sha256: str
    tokens: int | None
    elapsed_us: int | None
    exit_class: str
    identity: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    identity: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def identity_hash(parts: tuple[str, ...]) -> str:
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def result_row(source: Path, row_number: int, row: dict[str, Any]) -> ResultRow:
    prompt = row_text(row, "prompt", "prompt_id")
    prompt_sha256 = row_text(row, "prompt_sha256", "guest_prompt_sha256")
    phase = row_text(row, "phase", default="measured").lower()
    iteration = finite_int(row.get("iteration"))
    launch_index = finite_int(row.get("launch_index"))
    parts = (
        str(source),
        row_text(row, "profile"),
        row_text(row, "model"),
        row_text(row, "quantization"),
        prompt_sha256 if prompt_sha256 != "-" else prompt,
        phase,
        str(iteration) if iteration is not None else "-",
        str(launch_index) if launch_index is not None else "-",
        row_text(row, "commit"),
    )
    return ResultRow(
        source=str(source),
        row=row_number,
        profile=parts[1],
        model=parts[2],
        quantization=parts[3],
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        phase=phase,
        iteration=iteration,
        launch_index=launch_index,
        commit=parts[8],
        command_sha256=row_text(row, "command_sha256"),
        tokens=finite_int(row.get("tokens")),
        elapsed_us=finite_int(row.get("elapsed_us")),
        exit_class=row_text(row, "exit_class"),
        identity=identity_hash(parts),
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ResultRow], list[Finding]]:
    rows: list[ResultRow] = []
    findings: list[Finding] = []
    seen_files = 0
    seen_identities: dict[str, ResultRow] = {}
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            payload_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "-", str(exc)))
            continue
        for row_number, raw in enumerate(payload_rows, 1):
            row = result_row(path, row_number, raw)
            rows.append(row)
            if args.require_launch_index and row.launch_index is None:
                findings.append(Finding(row.source, row.row, "error", "missing_launch_index", row.identity, "launch_index must be an integer"))
            if args.require_iteration and row.iteration is None:
                findings.append(Finding(row.source, row.row, "error", "missing_iteration", row.identity, "iteration must be an integer"))
            if row.identity in seen_identities:
                first = seen_identities[row.identity]
                findings.append(
                    Finding(
                        row.source,
                        row.row,
                        "error",
                        "duplicate_result_identity",
                        row.identity,
                        f"duplicates {first.source}:{first.row}",
                    )
                )
            else:
                seen_identities[row.identity] = row

    if seen_files == 0:
        findings.append(Finding("", 0, "error", "no_input_files", "-", "no benchmark artifacts matched input paths/patterns"))
    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "-", f"found {len(rows)} row(s), below minimum {args.min_rows}"))
    return rows, findings


def build_report(rows: list[ResultRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "unique_identities": len({row.identity for row in rows}),
            "duplicates": sum(1 for finding in findings if finding.kind == "duplicate_result_identity"),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    fieldnames = list(ResultRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Result Uniqueness Audit",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- status: {report['status']}",
        f"- rows: {summary['rows']}",
        f"- unique_identities: {summary['unique_identities']}",
        f"- duplicates: {summary['duplicates']}",
        f"- findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", "", "| source | row | kind | detail |", "| --- | ---: | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['source']} | {finding['row']} | {finding['kind']} | {finding['detail']} |")
    else:
        lines.append("No result uniqueness findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_result_uniqueness_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "result_uniqueness"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for audit outputs")
    parser.add_argument("--output-stem", default="qemu_result_uniqueness_audit_latest", help="Output filename stem")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum benchmark rows required")
    parser.add_argument("--require-launch-index", action="store_true", help="Require every row to carry an integer launch_index")
    parser.add_argument("--require-iteration", action="store_true", help="Require every row to carry an integer iteration")
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
