#!/usr/bin/env python3
"""Audit saved QEMU prompt benchmark artifacts for actual runtime budgets.

This host-side tool reads benchmark JSON/JSONL/CSV artifacts only. It never
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class RuntimeRow:
    source: str
    group: str
    profile: str
    model: str
    quantization: str
    total_rows: int
    measured_rows: int
    warmup_rows: int
    timeout_rows: int
    failed_rows: int
    missing_wall_elapsed_rows: int
    total_wall_elapsed_us: int
    measured_wall_elapsed_us: int
    warmup_wall_elapsed_us: int
    total_wall_seconds: float
    measured_wall_seconds: float
    warmup_wall_seconds: float
    warmup_wall_pct: float | None
    max_row_wall_seconds: float


@dataclass(frozen=True)
class Finding:
    source: str
    group: str
    severity: str
    kind: str
    metric: str
    value: int | float
    limit: int | float
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


def text_value(row: dict[str, Any], key: str, default: str = "-") -> str:
    value = row.get(key)
    if value in (None, ""):
        return default
    return str(value)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def group_label(profile: str, model: str, quantization: str) -> str:
    return f"{profile}/{model}/{quantization}"


def load_grouped_rows(paths: list[Path], patterns: list[str]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for source in iter_input_files(paths, patterns):
        for row_number, row in enumerate(load_rows(source), 1):
            profile = text_value(row, "profile")
            model = text_value(row, "model")
            quantization = text_value(row, "quantization")
            key = (str(source), profile, model, quantization)
            enriched = dict(row)
            enriched["_source"] = str(source)
            enriched["_row_number"] = row_number
            grouped.setdefault(key, []).append(enriched)
    return grouped


def is_failed_row(row: dict[str, Any]) -> bool:
    return text_value(row, "exit_class", "ok") not in {"ok", "-"} or truthy(row.get("timed_out")) or finite_int(row.get("returncode")) not in (None, 0)


def summarize_group(key: tuple[str, str, str, str], rows: list[dict[str, Any]]) -> RuntimeRow:
    source, profile, model, quantization = key
    total_wall_elapsed_us = 0
    measured_wall_elapsed_us = 0
    warmup_wall_elapsed_us = 0
    max_row_wall_elapsed_us = 0
    missing_wall_elapsed_rows = 0
    measured_rows = 0
    warmup_rows = 0
    timeout_rows = 0
    failed_rows = 0

    for row in rows:
        phase = text_value(row, "phase", "measured")
        if phase == "warmup":
            warmup_rows += 1
        elif phase == "measured":
            measured_rows += 1
        if truthy(row.get("timed_out")):
            timeout_rows += 1
        if is_failed_row(row):
            failed_rows += 1

        wall_elapsed_us = finite_int(row.get("wall_elapsed_us"))
        if wall_elapsed_us is None:
            missing_wall_elapsed_rows += 1
            continue
        total_wall_elapsed_us += wall_elapsed_us
        max_row_wall_elapsed_us = max(max_row_wall_elapsed_us, wall_elapsed_us)
        if phase == "warmup":
            warmup_wall_elapsed_us += wall_elapsed_us
        elif phase == "measured":
            measured_wall_elapsed_us += wall_elapsed_us

    return RuntimeRow(
        source=source,
        group=group_label(profile, model, quantization),
        profile=profile,
        model=model,
        quantization=quantization,
        total_rows=len(rows),
        measured_rows=measured_rows,
        warmup_rows=warmup_rows,
        timeout_rows=timeout_rows,
        failed_rows=failed_rows,
        missing_wall_elapsed_rows=missing_wall_elapsed_rows,
        total_wall_elapsed_us=total_wall_elapsed_us,
        measured_wall_elapsed_us=measured_wall_elapsed_us,
        warmup_wall_elapsed_us=warmup_wall_elapsed_us,
        total_wall_seconds=total_wall_elapsed_us / 1_000_000.0,
        measured_wall_seconds=measured_wall_elapsed_us / 1_000_000.0,
        warmup_wall_seconds=warmup_wall_elapsed_us / 1_000_000.0,
        warmup_wall_pct=(warmup_wall_elapsed_us * 100.0 / total_wall_elapsed_us) if total_wall_elapsed_us > 0 else None,
        max_row_wall_seconds=max_row_wall_elapsed_us / 1_000_000.0,
    )


def add_limit_finding(
    findings: list[Finding],
    row: RuntimeRow,
    *,
    kind: str,
    metric: str,
    value: int | float,
    limit: int | float | None,
    detail: str,
) -> None:
    if limit is not None and value > limit:
        findings.append(Finding(row.source, row.group, "error", kind, metric, value, limit, detail))


def evaluate(rows: list[RuntimeRow], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    total_rows = sum(row.total_rows for row in rows)
    total_wall_seconds = sum(row.total_wall_seconds for row in rows)
    measured_wall_seconds = sum(row.measured_wall_seconds for row in rows)
    warmup_wall_seconds = sum(row.warmup_wall_seconds for row in rows)

    if total_rows < args.min_rows:
        findings.append(Finding("", "all", "error", "min_rows", "total_rows", total_rows, args.min_rows, "too few benchmark rows found"))

    if args.max_total_wall_seconds is not None and total_wall_seconds > args.max_total_wall_seconds:
        findings.append(
            Finding(
                "",
                "all",
                "error",
                "max_total_wall_seconds",
                "total_wall_seconds",
                total_wall_seconds,
                args.max_total_wall_seconds,
                "aggregate benchmark wall time exceeds CI budget",
            )
        )
    if args.max_measured_wall_seconds is not None and measured_wall_seconds > args.max_measured_wall_seconds:
        findings.append(
            Finding(
                "",
                "all",
                "error",
                "max_measured_wall_seconds",
                "measured_wall_seconds",
                measured_wall_seconds,
                args.max_measured_wall_seconds,
                "aggregate measured wall time exceeds CI budget",
            )
        )
    if args.max_warmup_wall_seconds is not None and warmup_wall_seconds > args.max_warmup_wall_seconds:
        findings.append(
            Finding(
                "",
                "all",
                "error",
                "max_warmup_wall_seconds",
                "warmup_wall_seconds",
                warmup_wall_seconds,
                args.max_warmup_wall_seconds,
                "aggregate warmup wall time exceeds CI budget",
            )
        )
    if args.max_warmup_wall_pct is not None and total_wall_seconds > 0:
        warmup_wall_pct = warmup_wall_seconds * 100.0 / total_wall_seconds
        if warmup_wall_pct > args.max_warmup_wall_pct:
            findings.append(
                Finding(
                    "",
                    "all",
                    "error",
                    "max_warmup_wall_pct",
                    "warmup_wall_pct",
                    warmup_wall_pct,
                    args.max_warmup_wall_pct,
                    "aggregate warmup wall time share exceeds CI budget",
                )
            )

    for row in rows:
        add_limit_finding(
            findings,
            row,
            kind="max_group_wall_seconds",
            metric="total_wall_seconds",
            value=row.total_wall_seconds,
            limit=args.max_group_wall_seconds,
            detail=f"{row.group} total wall time exceeds per-group budget",
        )
        add_limit_finding(
            findings,
            row,
            kind="max_group_measured_wall_seconds",
            metric="measured_wall_seconds",
            value=row.measured_wall_seconds,
            limit=args.max_group_measured_wall_seconds,
            detail=f"{row.group} measured wall time exceeds per-group budget",
        )
        if (
            args.max_group_warmup_wall_pct is not None
            and row.warmup_wall_pct is not None
            and row.warmup_wall_pct > args.max_group_warmup_wall_pct
        ):
            findings.append(
                Finding(
                    row.source,
                    row.group,
                    "error",
                    "max_group_warmup_wall_pct",
                    "warmup_wall_pct",
                    row.warmup_wall_pct,
                    args.max_group_warmup_wall_pct,
                    f"{row.group} warmup wall time share exceeds per-group budget",
                )
            )
        add_limit_finding(
            findings,
            row,
            kind="max_row_wall_seconds",
            metric="max_row_wall_seconds",
            value=row.max_row_wall_seconds,
            limit=args.max_row_wall_seconds,
            detail=f"{row.group} contains a benchmark row above the per-row wall-time budget",
        )
        if args.require_wall_elapsed_us and row.missing_wall_elapsed_rows:
            findings.append(
                Finding(
                    row.source,
                    row.group,
                    "error",
                    "missing_wall_elapsed_us",
                    "missing_wall_elapsed_rows",
                    row.missing_wall_elapsed_rows,
                    0,
                    f"{row.group} has rows without wall_elapsed_us telemetry",
                )
            )
        if args.fail_on_failures and row.failed_rows:
            findings.append(
                Finding(
                    row.source,
                    row.group,
                    "error",
                    "failed_rows",
                    "failed_rows",
                    row.failed_rows,
                    0,
                    f"{row.group} has failed, timed-out, or nonzero-returncode rows",
                )
            )
    return findings


def write_csv(path: Path, rows: list[RuntimeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RuntimeRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# QEMU Runtime Budget Audit",
        "",
        f"- Status: {report['status']}",
        f"- Rows: {report['summary']['rows']}",
        f"- Groups: {report['summary']['groups']}",
        f"- Total wall seconds: {report['summary']['total_wall_seconds']:.6f}",
        f"- Measured wall seconds: {report['summary']['measured_wall_seconds']:.6f}",
        f"- Warmup wall seconds: {report['summary']['warmup_wall_seconds']:.6f}",
        f"- Warmup wall pct: {format_optional_float(report['summary']['warmup_wall_pct'])}",
        f"- Findings: {len(report['findings'])}",
        "",
        "## Groups",
        "",
        "| Group | Rows | Measured | Warmup | Failed | Missing wall | Total s | Measured s | Warmup s | Warmup % | Max row s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["groups"]:
        lines.append(
            "| {group} | {total_rows} | {measured_rows} | {warmup_rows} | {failed_rows} | {missing_wall_elapsed_rows} | "
            "{total_wall_seconds:.6f} | {measured_wall_seconds:.6f} | {warmup_wall_seconds:.6f} | {warmup_wall_pct} | {max_row_wall_seconds:.6f} |".format(
                **{**row, "warmup_wall_pct": format_optional_float(row.get("warmup_wall_pct"))}
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", "", "| Severity | Kind | Group | Metric | Value | Limit | Detail |", "| --- | --- | --- | --- | ---: | ---: | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {group} | {metric} | {value} | {limit} | {detail} |".format(**finding)
            )
    else:
        lines.extend(["", "No runtime budget findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element("testsuite", {"name": "holyc_qemu_runtime_budget_audit", "tests": "1", "failures": str(int(bool(findings)))})
    case = ET.SubElement(suite, "testcase", {"name": "runtime_budget"})
    if findings:
        failure = ET.SubElement(case, "failure", {"type": "runtime_budget", "message": f"{len(findings)} runtime budget finding(s)"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def format_optional_float(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{float(value):.6f}"
    return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories to audit")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Glob pattern for directory inputs; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_runtime_budget_audit_latest")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-total-wall-seconds", type=float)
    parser.add_argument("--max-measured-wall-seconds", type=float)
    parser.add_argument("--max-warmup-wall-seconds", type=float)
    parser.add_argument("--max-warmup-wall-pct", type=float, help="Fail when aggregate warmup wall time exceeds this percent of total wall time")
    parser.add_argument("--max-group-wall-seconds", type=float)
    parser.add_argument("--max-group-measured-wall-seconds", type=float)
    parser.add_argument("--max-group-warmup-wall-pct", type=float, help="Fail when any profile/model/quantization group spends this percent of wall time in warmup")
    parser.add_argument("--max-row-wall-seconds", type=float)
    parser.add_argument("--allow-missing-wall-elapsed-us", dest="require_wall_elapsed_us", action="store_false")
    parser.add_argument("--fail-on-failures", action="store_true", help="Fail when rows record timeout, non-ok exit_class, or nonzero returncode")
    parser.set_defaults(require_wall_elapsed_us=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.patterns or list(DEFAULT_PATTERNS)
    grouped = load_grouped_rows(args.inputs, patterns)
    rows = [summarize_group(key, grouped[key]) for key in sorted(grouped)]
    findings = evaluate(rows, args)
    total_wall_seconds = sum(row.total_wall_seconds for row in rows)
    warmup_wall_seconds = sum(row.warmup_wall_seconds for row in rows)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "groups": len(rows),
            "rows": sum(row.total_rows for row in rows),
            "measured_rows": sum(row.measured_rows for row in rows),
            "warmup_rows": sum(row.warmup_rows for row in rows),
            "failed_rows": sum(row.failed_rows for row in rows),
            "missing_wall_elapsed_rows": sum(row.missing_wall_elapsed_rows for row in rows),
            "total_wall_seconds": total_wall_seconds,
            "measured_wall_seconds": sum(row.measured_wall_seconds for row in rows),
            "warmup_wall_seconds": warmup_wall_seconds,
            "warmup_wall_pct": (warmup_wall_seconds * 100.0 / total_wall_seconds) if total_wall_seconds > 0 else None,
        },
        "groups": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.output_stem}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{args.output_stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{args.output_stem}.md", report)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", report)
    print(f"{report['status']} rows={report['summary']['rows']} groups={report['summary']['groups']} findings={len(findings)} output={output}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
