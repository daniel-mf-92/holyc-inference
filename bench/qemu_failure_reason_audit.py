#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for failure-diagnosis consistency.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
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
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")
ALLOWED_EXIT_CLASSES = {"ok", "timeout", "launch_error", "nonzero_exit"}


@dataclass(frozen=True)
class FailureReasonRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    phase: str
    exit_class: str
    returncode: int | None
    timed_out: bool | None
    failure_reason: str
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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


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


def fmt(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def add_finding(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    kind: str,
    field: str,
    stored: object,
    expected: str,
    detail: str,
) -> None:
    findings.append(Finding(str(source), row_number, "error", kind, field, fmt(stored), expected, detail))


def audit_row(source: Path, row_number: int, row: dict[str, Any]) -> tuple[FailureReasonRow, list[Finding]]:
    findings: list[Finding] = []
    exit_class = row_text(row, "exit_class", default="")
    returncode = finite_int(row.get("returncode"))
    timed_out = parse_bool(row.get("timed_out"))
    failure_reason = "" if row.get("failure_reason") in (None, "") else str(row.get("failure_reason"))
    checks = 0

    checks += 1
    if exit_class not in ALLOWED_EXIT_CLASSES:
        add_finding(
            findings,
            source=source,
            row_number=row_number,
            kind="invalid_exit_class",
            field="exit_class",
            stored=exit_class,
            expected="one of: ok, timeout, launch_error, nonzero_exit",
            detail="exit_class must use the normalized QEMU benchmark vocabulary",
        )

    checks += 1
    if timed_out is None:
        add_finding(
            findings,
            source=source,
            row_number=row_number,
            kind="missing_timed_out",
            field="timed_out",
            stored=row.get("timed_out"),
            expected="boolean",
            detail="timed_out must be present so timeout failures can be separated from process exits",
        )

    if exit_class == "ok":
        checks += 3
        if timed_out is True:
            add_finding(findings, source=source, row_number=row_number, kind="ok_timed_out", field="timed_out", stored=timed_out, expected="false", detail="ok rows must not be timed out")
        if returncode not in (None, 0):
            add_finding(findings, source=source, row_number=row_number, kind="ok_nonzero_returncode", field="returncode", stored=returncode, expected="0 or absent", detail="ok rows must not carry a non-zero process exit code")
        if failure_reason:
            add_finding(findings, source=source, row_number=row_number, kind="ok_with_failure_reason", field="failure_reason", stored=failure_reason, expected="empty", detail="ok rows must not retain stale failure text")
    elif exit_class == "timeout":
        checks += 3
        if timed_out is not True:
            add_finding(findings, source=source, row_number=row_number, kind="timeout_without_timed_out", field="timed_out", stored=timed_out, expected="true", detail="timeout rows must set timed_out")
        if not failure_reason:
            add_finding(findings, source=source, row_number=row_number, kind="missing_failure_reason", field="failure_reason", stored=failure_reason, expected="non-empty", detail="timeout rows must explain the timeout")
        if returncode == 0:
            add_finding(findings, source=source, row_number=row_number, kind="timeout_zero_returncode", field="returncode", stored=returncode, expected="non-zero or absent", detail="timeout rows must not look like a clean process exit")
    elif exit_class == "nonzero_exit":
        checks += 3
        if timed_out is True:
            add_finding(findings, source=source, row_number=row_number, kind="nonzero_exit_timed_out", field="timed_out", stored=timed_out, expected="false", detail="nonzero_exit rows must not also be classified as timeout")
        if returncode in (None, 0):
            add_finding(findings, source=source, row_number=row_number, kind="missing_nonzero_returncode", field="returncode", stored=returncode, expected="non-zero integer", detail="nonzero_exit rows must carry the process exit code")
        if not failure_reason:
            add_finding(findings, source=source, row_number=row_number, kind="missing_failure_reason", field="failure_reason", stored=failure_reason, expected="non-empty", detail="nonzero_exit rows must explain the failed process exit")
    elif exit_class == "launch_error":
        checks += 2
        if timed_out is True:
            add_finding(findings, source=source, row_number=row_number, kind="launch_error_timed_out", field="timed_out", stored=timed_out, expected="false", detail="launch_error rows must not also be classified as timeout")
        if not failure_reason:
            add_finding(findings, source=source, row_number=row_number, kind="missing_failure_reason", field="failure_reason", stored=failure_reason, expected="non-empty", detail="launch_error rows must explain why QEMU did not launch")

    return (
        FailureReasonRow(
            source=str(source),
            row=row_number,
            profile=row_text(row, "profile"),
            model=row_text(row, "model"),
            quantization=row_text(row, "quantization"),
            prompt=row_text(row, "prompt", "prompt_id"),
            commit=row_text(row, "commit"),
            phase=row_text(row, "phase", default="measured"),
            exit_class=exit_class or "-",
            returncode=returncode,
            timed_out=timed_out,
            failure_reason=failure_reason,
            checks=checks,
            findings=len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[FailureReasonRow], list[Finding]]:
    rows: list[FailureReasonRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", "", "readable benchmark artifact", str(exc)))
            continue
        for row_number, raw_row in enumerate(loaded_rows, 1):
            row, row_findings = audit_row(path, row_number, raw_row)
            rows.append(row)
            findings.extend(row_findings)

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", str(seen_files), str(args.min_artifacts), "too few artifacts matched inputs"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "too few rows matched inputs"))
    return rows, findings


def summary(rows: list[FailureReasonRow], findings: list[Finding]) -> dict[str, Any]:
    by_exit_class: dict[str, int] = {}
    for row in rows:
        by_exit_class[row.exit_class] = by_exit_class.get(row.exit_class, 0) + 1
    return {
        "rows": len(rows),
        "findings": len(findings),
        "checks": sum(row.checks for row in rows),
        "exit_classes": dict(sorted(by_exit_class.items())),
        "failure_rows": sum(1 for row in rows if row.exit_class != "ok"),
        "missing_failure_reason_rows": sum(1 for row in rows if row.exit_class != "ok" and not row.failure_reason),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
    }


def write_json(path: Path, rows: list[FailureReasonRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[FailureReasonRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Failure Reason Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(rows)}",
        f"Checks: {sum(row.checks for row in rows)}",
        f"Findings: {len(findings)}",
        "",
        "## Exit Classes",
        "",
        "| Exit class | Rows |",
        "| --- | ---: |",
    ]
    counts = summary(rows, findings)["exit_classes"]
    for exit_class, count in counts.items():
        lines.append(f"| {exit_class} | {count} |")
    lines.extend(["", "## Rows", "", "| Source | Row | Prompt | Phase | Exit class | Returncode | Timed out | Findings |", "| --- | ---: | --- | --- | --- | ---: | --- | ---: |"])
    for row in rows:
        lines.append(f"| {row.source} | {row.row} | {row.prompt} | {row.phase} | {row.exit_class} | {fmt(row.returncode)} | {fmt(row.timed_out)} | {row.findings} |")
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Field | Stored | Expected | Detail |", "| --- | ---: | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.field} | {finding.stored} | {finding.expected} | {finding.detail} |")
    else:
        lines.append("No failure-reason findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[FailureReasonRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FailureReasonRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_failure_reason_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "failure_reason"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} failure-reason finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_failure_reason_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if args.min_artifacts < 0 or args.min_rows < 0:
        parser.error("--min-artifacts and --min-rows must be >= 0")

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
