#!/usr/bin/env python3
"""Audit saved QEMU benchmark rows for exit-class consistency.

This host-side tool reads existing qemu_prompt_bench artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class ExitClassRow:
    source: str
    row: int
    list_name: str
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    returncode: int | None
    timed_out: bool | None
    exit_class: str
    computed_exit_class: str
    failure_reason: str
    computed_failure_reason: str
    tokens: int | None
    elapsed_us: int | None
    status: str


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
    text = str(value).strip().casefold()
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


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def flatten_json_payload(payload: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield "rows", item
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
                    yield key, merged
    if not yielded:
        yield "artifact", payload


def load_rows(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
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
            for row in csv.DictReader(handle):
                yield "csv", row
        return
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def computed_exit(returncode: int | None, timed_out: bool | None) -> str:
    if returncode is None or timed_out is None:
        return ""
    return qemu_prompt_bench.classify_exit(returncode, timed_out)


def computed_failure(returncode: int | None, timed_out: bool | None) -> str:
    if returncode is None or timed_out is None:
        return ""
    return qemu_prompt_bench.failure_reason(returncode, timed_out) or ""


def audit_row(source: Path, row_number: int, list_name: str, raw: dict[str, Any], args: argparse.Namespace) -> tuple[ExitClassRow, list[Finding]]:
    findings: list[Finding] = []
    returncode = finite_int(raw.get("returncode"))
    timed_out = parse_bool(raw.get("timed_out"))
    exit_class = row_text(raw, "exit_class", default="")
    failure_reason = text_value(raw.get("failure_reason"))
    tokens = finite_int(raw.get("tokens"))
    elapsed_us = finite_int(raw.get("elapsed_us"))
    expected_exit = computed_exit(returncode, timed_out)
    expected_failure = computed_failure(returncode, timed_out)

    if returncode is None:
        findings.append(Finding(str(source), row_number, "error", "missing_returncode", "returncode", "returncode must be recorded"))
    if timed_out is None:
        findings.append(Finding(str(source), row_number, "error", "missing_timed_out", "timed_out", "timed_out must be recorded"))
    if exit_class not in qemu_prompt_bench.EXIT_CLASSES:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "invalid_exit_class",
                "exit_class",
                f"exit_class must be one of {', '.join(qemu_prompt_bench.EXIT_CLASSES)}",
            )
        )
    elif expected_exit and exit_class != expected_exit:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "exit_class_mismatch",
                "exit_class",
                f"stored {exit_class!r} does not match returncode/timed_out-derived {expected_exit!r}",
            )
        )

    if expected_failure and failure_reason != expected_failure:
        findings.append(
            Finding(
                str(source),
                row_number,
                "error",
                "failure_reason_mismatch",
                "failure_reason",
                f"stored {failure_reason!r} does not match derived {expected_failure!r}",
            )
        )
    if expected_failure == "" and failure_reason:
        findings.append(
            Finding(str(source), row_number, "error", "ok_row_has_failure_reason", "failure_reason", "ok rows must not carry a failure reason")
        )

    if args.require_success_telemetry and expected_exit == "ok":
        if tokens is None or tokens <= 0:
            findings.append(Finding(str(source), row_number, "error", "ok_row_missing_tokens", "tokens", "ok rows must record positive tokens"))
        if elapsed_us is None or elapsed_us <= 0:
            findings.append(Finding(str(source), row_number, "error", "ok_row_missing_elapsed_us", "elapsed_us", "ok rows must record positive elapsed_us"))

    if args.require_failure_reason and expected_exit not in ("", "ok") and not failure_reason:
        findings.append(Finding(str(source), row_number, "error", "failure_row_missing_reason", "failure_reason", "failed rows must record a failure reason"))

    status = "fail" if findings else "pass"
    row = ExitClassRow(
        source=str(source),
        row=row_number,
        list_name=list_name,
        profile=row_text(raw, "profile"),
        model=row_text(raw, "model"),
        quantization=row_text(raw, "quantization"),
        prompt=row_text(raw, "prompt", "prompt_id"),
        phase=row_text(raw, "phase"),
        returncode=returncode,
        timed_out=timed_out,
        exit_class=exit_class,
        computed_exit_class=expected_exit,
        failure_reason=failure_reason,
        computed_failure_reason=expected_failure,
        tokens=tokens,
        elapsed_us=elapsed_us,
        status=status,
    )
    return row, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ExitClassRow], list[Finding]]:
    rows: list[ExitClassRow] = []
    findings: list[Finding] = []
    files = list(iter_input_files(paths, args.pattern))
    if not files:
        findings.append(Finding("", 0, "error", "no_artifacts", "input", "no matching benchmark artifacts found"))
        return rows, findings

    for path in files:
        try:
            for row_number, (list_name, raw) in enumerate(load_rows(path), 1):
                row, row_findings = audit_row(path, row_number, list_name, raw, args)
                rows.append(row)
                findings.extend(row_findings)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", str(exc)))

    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "rows", f"found {len(rows)} rows, expected at least {args.min_rows}"))
    return rows, findings


def summary(rows: list[ExitClassRow], findings: list[Finding]) -> dict[str, Any]:
    by_class = {name: 0 for name in qemu_prompt_bench.EXIT_CLASSES}
    for row in rows:
        if row.exit_class in by_class:
            by_class[row.exit_class] += 1
    return {
        "rows": len(rows),
        "findings": len(findings),
        "pass_rows": sum(1 for row in rows if row.status == "pass"),
        "fail_rows": sum(1 for row in rows if row.status == "fail"),
        **{f"exit_class_{key}_rows": value for key, value in by_class.items()},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# QEMU Exit Class Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Rows: {report['summary']['rows']}",
        f"- Findings: {report['summary']['findings']}",
        "",
    ]
    findings = report["findings"]
    if findings:
        lines.extend(["## Findings", "", "| Source | Row | Kind | Field | Detail |", "|---|---:|---|---|---|"])
        for finding in findings[:50]:
            lines.append(
                f"| `{finding['source']}` | {finding['row']} | `{finding['kind']}` | `{finding['field']}` | {finding['detail']} |"
            )
    else:
        lines.append("No exit-class findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_exit_class_audit",
            "tests": "1",
            "failures": str(1 if findings else 0),
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "exit_class_consistency"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} exit-class findings"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings[:100])
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Report output directory")
    parser.add_argument("--output-stem", default="qemu_exit_class_audit_latest", help="Report filename stem")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum rows required")
    parser.add_argument("--require-success-telemetry", action="store_true", help="Require ok rows to include positive tokens and elapsed_us")
    parser.add_argument("--require-failure-reason", action="store_true", help="Require failed rows to include failure_reason")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": summary(rows, findings),
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / f"{stem}.md", report)
    write_csv(output_dir / f"{stem}.csv", [asdict(row) for row in rows], list(ExitClassRow.__dataclass_fields__))
    write_csv(output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    write_junit(output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
