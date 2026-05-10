#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for failure taxonomy consistency.

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
KNOWN_EXIT_CLASSES = {"ok", "timeout", "launch_error", "nonzero_exit"}


@dataclass(frozen=True)
class FailureRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    phase: str
    iteration: int | None
    exit_class: str
    timed_out: bool | None
    returncode: int | None
    failure_reason: str
    tokens: int | None
    wall_elapsed_us: float | None
    wall_tok_per_s: float | None


@dataclass(frozen=True)
class ExitClassSummary:
    exit_class: str
    rows: int
    timed_out: int
    with_failure_reason: int
    ok_metric_rows: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    metric: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def finite_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


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


def failure_row(source: Path, row_number: int, row: dict[str, Any]) -> FailureRow:
    return FailureRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        phase=row_text(row, "phase", default="measured").lower(),
        iteration=finite_int(row.get("iteration")),
        exit_class=row_text(row, "exit_class").lower(),
        timed_out=parse_bool(row.get("timed_out")),
        returncode=finite_int(row.get("returncode")),
        failure_reason=row_text(row, "failure_reason", default=""),
        tokens=finite_int(row.get("tokens")),
        wall_elapsed_us=finite_float(row.get("wall_elapsed_us")),
        wall_tok_per_s=finite_float(row.get("wall_tok_per_s")),
    )


def row_label(row: FailureRow) -> str:
    return f"{row.profile}/{row.model}/{row.quantization}/{row.prompt}/{row.phase}/{row.iteration or '-'}"


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[FailureRow], list[ExitClassSummary], list[Finding]]:
    rows: list[FailureRow] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            loaded_rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "artifact", str(exc)))
            continue
        for row_number, raw_row in enumerate(loaded_rows, 1):
            row = failure_row(path, row_number, raw_row)
            rows.append(row)
            label = row_label(row)
            if row.exit_class not in KNOWN_EXIT_CLASSES:
                findings.append(Finding(row.source, row.row, "error", "unknown_exit_class", "exit_class", f"{label}: {row.exit_class!r} is not a known exit class"))
            if row.timed_out is None:
                findings.append(Finding(row.source, row.row, "error", "missing_timed_out", "timed_out", f"{label}: timed_out must be boolean"))
            elif row.timed_out and row.exit_class != "timeout":
                findings.append(Finding(row.source, row.row, "error", "timeout_exit_class_mismatch", "exit_class", f"{label}: timed_out=true but exit_class={row.exit_class!r}"))
            elif row.exit_class == "timeout" and row.timed_out is not True:
                findings.append(Finding(row.source, row.row, "error", "timeout_flag_mismatch", "timed_out", f"{label}: exit_class=timeout but timed_out is not true"))
            if row.exit_class == "ok":
                if row.failure_reason:
                    findings.append(Finding(row.source, row.row, "error", "ok_has_failure_reason", "failure_reason", f"{label}: OK row has failure_reason={row.failure_reason!r}"))
                if row.returncode not in (0, None):
                    findings.append(Finding(row.source, row.row, "error", "ok_returncode_mismatch", "returncode", f"{label}: OK row returncode={row.returncode}"))
                if args.require_ok_metrics:
                    if row.tokens is None or row.tokens <= 0:
                        findings.append(Finding(row.source, row.row, "error", "ok_missing_tokens", "tokens", f"{label}: OK row must report positive tokens"))
                    if row.wall_elapsed_us is None or row.wall_elapsed_us <= 0:
                        findings.append(Finding(row.source, row.row, "error", "ok_missing_wall_elapsed_us", "wall_elapsed_us", f"{label}: OK row must report positive wall_elapsed_us"))
                    if row.wall_tok_per_s is None or row.wall_tok_per_s <= 0:
                        findings.append(Finding(row.source, row.row, "error", "ok_missing_wall_tok_per_s", "wall_tok_per_s", f"{label}: OK row must report positive wall_tok_per_s"))
            elif row.exit_class in KNOWN_EXIT_CLASSES and not row.failure_reason:
                findings.append(Finding(row.source, row.row, "error", "failure_missing_reason", "failure_reason", f"{label}: non-OK row must include failure_reason"))
            if row.exit_class == "nonzero_exit" and (row.returncode is None or row.returncode == 0):
                findings.append(Finding(row.source, row.row, "error", "nonzero_returncode_mismatch", "returncode", f"{label}: nonzero_exit row returncode must be nonzero"))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(rows) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(rows)}, expected at least {args.min_rows}"))

    failures = [row for row in rows if row.exit_class != "ok"]
    timeouts = [row for row in rows if row.exit_class == "timeout"]
    if rows:
        failure_pct = 100.0 * len(failures) / len(rows)
        timeout_pct = 100.0 * len(timeouts) / len(rows)
        if failure_pct > args.max_failure_pct:
            findings.append(Finding("-", 0, "error", "max_failure_pct", "exit_class", f"failure_pct={failure_pct:.3f} exceeds {args.max_failure_pct:.3f}"))
        if timeout_pct > args.max_timeout_pct:
            findings.append(Finding("-", 0, "error", "max_timeout_pct", "exit_class", f"timeout_pct={timeout_pct:.3f} exceeds {args.max_timeout_pct:.3f}"))

    summaries: list[ExitClassSummary] = []
    for exit_class in sorted({row.exit_class for row in rows} | KNOWN_EXIT_CLASSES):
        class_rows = [row for row in rows if row.exit_class == exit_class]
        summaries.append(
            ExitClassSummary(
                exit_class=exit_class,
                rows=len(class_rows),
                timed_out=sum(1 for row in class_rows if row.timed_out is True),
                with_failure_reason=sum(1 for row in class_rows if bool(row.failure_reason)),
                ok_metric_rows=sum(1 for row in class_rows if row.tokens and row.wall_elapsed_us and row.wall_tok_per_s),
            )
        )
    return rows, summaries, findings


def summary(rows: list[FailureRow], findings: list[Finding]) -> dict[str, Any]:
    failures = [row for row in rows if row.exit_class != "ok"]
    timeouts = [row for row in rows if row.exit_class == "timeout"]
    return {
        "rows": len(rows),
        "ok_rows": sum(1 for row in rows if row.exit_class == "ok"),
        "failure_rows": len(failures),
        "timeout_rows": len(timeouts),
        "failure_pct": 100.0 * len(failures) / len(rows) if rows else 0.0,
        "timeout_pct": 100.0 * len(timeouts) / len(rows) if rows else 0.0,
        "findings": len(findings),
        "profiles": sorted({row.profile for row in rows if row.profile != "-"}),
        "models": sorted({row.model for row in rows if row.model != "-"}),
        "quantizations": sorted({row.quantization for row in rows if row.quantization != "-"}),
        "prompts": sorted({row.prompt for row in rows if row.prompt != "-"}),
    }


def write_json(path: Path, rows: list[FailureRow], summaries: list[ExitClassSummary], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(rows, findings),
        "exit_classes": [asdict(item) for item in summaries],
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[FailureRow], summaries: list[ExitClassSummary], findings: list[Finding]) -> None:
    report = summary(rows, findings)
    lines = [
        "# QEMU Failure Taxonomy Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {report['rows']}",
        f"Failure rows: {report['failure_rows']} ({report['failure_pct']:.3f}%)",
        f"Timeout rows: {report['timeout_rows']} ({report['timeout_pct']:.3f}%)",
        f"Findings: {len(findings)}",
        "",
        "## Exit Classes",
        "",
        "| Exit Class | Rows | Timed Out | With Failure Reason | OK Metric Rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(f"| {item.exit_class} | {item.rows} | {item.timed_out} | {item.with_failure_reason} | {item.ok_metric_rows} |")
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Metric | Detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.metric} | {finding.detail} |")
    else:
        lines.append("No failure taxonomy findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, summaries: list[ExitClassSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExitClassSummary.__dataclass_fields__))
        writer.writeheader()
        for item in summaries:
            writer.writerow(asdict(item))


def write_rows_csv(path: Path, rows: list[FailureRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FailureRow.__dataclass_fields__))
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
    suite = ET.Element("testsuite", name="holyc_qemu_failure_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="qemu_failure_taxonomy")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} failure taxonomy finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.metric}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_failure_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--max-failure-pct", type=float, default=100.0)
    parser.add_argument("--max-timeout-pct", type=float, default=100.0)
    parser.add_argument("--no-require-ok-metrics", dest="require_ok_metrics", action="store_false")
    parser.set_defaults(require_ok_metrics=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, summaries, findings = audit(args.paths, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", rows, summaries, findings)
    write_markdown(args.output_dir / f"{stem}.md", rows, summaries, findings)
    write_csv(args.output_dir / f"{stem}.csv", summaries)
    write_rows_csv(args.output_dir / f"{stem}_rows.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
