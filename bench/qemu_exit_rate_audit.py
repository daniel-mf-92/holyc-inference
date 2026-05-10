#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for exit-class failure rates.

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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
ALLOWED_EXIT_CLASSES = {"ok", "timeout", "launch_error", "nonzero_exit"}


@dataclass(frozen=True)
class ExitRateRow:
    source: str
    profile: str
    model: str
    quantization: str
    phase: str
    rows: int
    ok_rows: int
    failed_rows: int
    timeout_rows: int
    nonzero_exit_rows: int
    launch_error_rows: int
    failure_pct: float
    timeout_pct: float
    nonzero_exit_pct: float
    launch_error_pct: float


@dataclass(frozen=True)
class Finding:
    source: str
    group: str
    severity: str
    kind: str
    metric: str
    value: float | int | str
    threshold: float | int | str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def pct(count: int, total: int) -> float:
    return 0.0 if total <= 0 else (count * 100.0) / total


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


def flatten_json_payload(payload: Any, *, include_warmup: bool) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS and key != "warmups"}
    keys = list(RESULT_KEYS)
    if include_warmup:
        keys.insert(0, "warmups")
    yielded = False
    for key in keys:
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


def load_rows(path: Path, *, include_warmup: bool) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        yield from flatten_json_payload(json.loads(path.read_text(encoding="utf-8")), include_warmup=include_warmup)
        return
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if stripped:
                    try:
                        yield from flatten_json_payload(json.loads(stripped), include_warmup=include_warmup)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        return
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def normalized_exit_class(row: dict[str, Any]) -> str:
    exit_class = str(row.get("exit_class") or "").strip().lower()
    if exit_class:
        return exit_class
    timed_out = parse_bool(row.get("timed_out"))
    returncode = finite_int(row.get("returncode"))
    if timed_out is True:
        return "timeout"
    if returncode not in (None, 0):
        return "nonzero_exit"
    return "ok"


def group_key(source: Path, row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(source),
        row_text(row, "profile"),
        row_text(row, "model"),
        row_text(row, "quantization"),
        row_text(row, "phase", default="measured"),
    )


def group_label(row: ExitRateRow) -> str:
    return "/".join((row.profile, row.model, row.quantization, row.phase))


def summarize_group(key: tuple[str, str, str, str, str], rows: list[dict[str, Any]]) -> tuple[ExitRateRow, list[Finding]]:
    source, profile, model, quantization, phase = key
    exit_classes = [normalized_exit_class(row) for row in rows]
    ok_rows = sum(1 for item in exit_classes if item == "ok")
    timeout_rows = sum(1 for item in exit_classes if item == "timeout")
    nonzero_rows = sum(1 for item in exit_classes if item == "nonzero_exit")
    launch_errors = sum(1 for item in exit_classes if item == "launch_error")
    failed_rows = len(rows) - ok_rows
    summary = ExitRateRow(
        source=source,
        profile=profile,
        model=model,
        quantization=quantization,
        phase=phase,
        rows=len(rows),
        ok_rows=ok_rows,
        failed_rows=failed_rows,
        timeout_rows=timeout_rows,
        nonzero_exit_rows=nonzero_rows,
        launch_error_rows=launch_errors,
        failure_pct=pct(failed_rows, len(rows)),
        timeout_pct=pct(timeout_rows, len(rows)),
        nonzero_exit_pct=pct(nonzero_rows, len(rows)),
        launch_error_pct=pct(launch_errors, len(rows)),
    )
    findings = [
        Finding(source, f"{profile}/{model}/{quantization}/{phase}", "error", "invalid_exit_class", "exit_class", item, "ok|timeout|launch_error|nonzero_exit", "exit_class must use the normalized benchmark vocabulary")
        for item in sorted(set(exit_classes) - ALLOWED_EXIT_CLASSES)
    ]
    return summary, findings


def evaluate(rows: list[ExitRateRow], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    if len(rows) < args.min_groups:
        findings.append(Finding("", "coverage", "error", "min_groups", "groups", len(rows), args.min_groups, "too few exit-rate groups found"))
    for row in rows:
        label = group_label(row)
        checks = (
            ("min_rows", "rows", row.rows, args.min_rows, row.rows < args.min_rows, "group has too few measured rows"),
            ("max_failure_pct", "failure_pct", row.failure_pct, args.max_failure_pct, row.failure_pct > args.max_failure_pct, "failure rate exceeds configured gate"),
            ("max_timeout_pct", "timeout_pct", row.timeout_pct, args.max_timeout_pct, row.timeout_pct > args.max_timeout_pct, "timeout rate exceeds configured gate"),
            ("max_nonzero_exit_pct", "nonzero_exit_pct", row.nonzero_exit_pct, args.max_nonzero_exit_pct, row.nonzero_exit_pct > args.max_nonzero_exit_pct, "nonzero-exit rate exceeds configured gate"),
            ("max_launch_error_pct", "launch_error_pct", row.launch_error_pct, args.max_launch_error_pct, row.launch_error_pct > args.max_launch_error_pct, "launch-error rate exceeds configured gate"),
        )
        for kind, metric, value, threshold, failed, detail in checks:
            if failed:
                findings.append(Finding(row.source, label, "error", kind, metric, value, threshold, detail))
    return findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ExitRateRow], list[Finding]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        try:
            for raw in load_rows(path, include_warmup=args.include_warmup):
                if not isinstance(raw, dict):
                    continue
                if not args.include_warmup and row_text(raw, "phase", default="measured") == "warmup":
                    continue
                grouped.setdefault(group_key(path, raw), []).append(raw)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), "artifact", "error", "load_error", "artifact", str(exc), "valid input", "could not load benchmark artifact"))

    summaries: list[ExitRateRow] = []
    for key, rows in sorted(grouped.items()):
        summary, group_findings = summarize_group(key, rows)
        summaries.append(summary)
        findings.extend(group_findings)
    findings.extend(evaluate(summaries, args))
    return summaries, findings


def build_report(rows: list[ExitRateRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "groups": len(rows),
            "rows": sum(row.rows for row in rows),
            "failed_rows": sum(row.failed_rows for row in rows),
            "timeout_rows": sum(row.timeout_rows for row in rows),
            "findings": len(findings),
        },
        "groups": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[ExitRateRow], findings: list[Finding]) -> None:
    lines = ["# QEMU Exit Rate Audit", "", f"Groups: {len(rows)}", f"Findings: {len(findings)}", ""]
    if findings:
        lines.extend(["| Group | Kind | Metric | Value | Threshold | Detail |", "| --- | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.group} | {finding.kind} | {finding.metric} | {finding.value} | {finding.threshold} | {finding.detail.replace('|', '\\|')} |")
    else:
        lines.append("No exit-rate findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[ExitRateRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExitRateRow.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_qemu_exit_rate_audit", "tests": str(max(1, len(findings) or 1)), "failures": str(len(findings)), "errors": "0"},
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_exit_rate_audit", "name": "exit_rates"})
    for index, finding in enumerate(findings, 1):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_exit_rate_audit", "name": f"{finding.kind}_{index}"})
        failure = ET.SubElement(case, "failure", {"type": finding.kind, "message": finding.detail})
        failure.text = f"{finding.group} {finding.metric}={finding.value} threshold={finding.threshold}"
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="benchmark artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob for directory inputs")
    parser.add_argument("--include-warmup", action="store_true", help="include warmup rows in the audit")
    parser.add_argument("--min-groups", type=int, default=1, help="minimum grouped summaries required")
    parser.add_argument("--min-rows", type=int, default=1, help="minimum rows required per group")
    parser.add_argument("--max-failure-pct", type=float, default=0.0, help="maximum allowed failure percentage")
    parser.add_argument("--max-timeout-pct", type=float, default=0.0, help="maximum allowed timeout percentage")
    parser.add_argument("--max-nonzero-exit-pct", type=float, default=0.0, help="maximum allowed nonzero-exit percentage")
    parser.add_argument("--max-launch-error-pct", type=float, default=0.0, help="maximum allowed launch-error percentage")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="output directory")
    parser.add_argument("--output-stem", default="qemu_exit_rate_audit_latest", help="output filename stem")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = build_report(rows, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
