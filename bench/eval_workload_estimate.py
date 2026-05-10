#!/usr/bin/env python3
"""Estimate offline multiple-choice eval workload before QEMU runs.

This host-side tool reads local JSONL datasets accepted by dataset_pack.py,
projects prompt/choice scoring tokens, and emits budget-friendly reports. It
never launches QEMU, downloads data, or touches the TempleOS guest.
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

import dataset_pack


@dataclass(frozen=True)
class EvalWorkloadRow:
    source: str
    row: int
    dataset: str
    split: str
    record_id: str
    choices: int
    prompt_bytes: int
    choice_bytes: int
    prompt_tokens_est: int
    choice_tokens_est: int
    scored_tokens_est: int
    launches_est: int
    wall_seconds_est: float | None


@dataclass(frozen=True)
class ScopeEstimate:
    scope: str
    dataset: str
    split: str
    records: int
    choices: int
    prompt_bytes: int
    choice_bytes: int
    prompt_tokens_est: int
    choice_tokens_est: int
    scored_tokens_est: int
    launches_est: int
    wall_seconds_est: float | None


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    field: str
    value: str
    limit: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def estimate_tokens(byte_count: int, bytes_per_token: float) -> int:
    if byte_count <= 0:
        return 0
    return max(1, math.ceil(byte_count / bytes_per_token))


def launches_for_record(choice_count: int, launch_mode: str) -> int:
    if launch_mode == "single":
        return 0
    if launch_mode == "per-choice":
        return choice_count
    return 1


def wall_seconds(tokens: int, launches: int, args: argparse.Namespace) -> float | None:
    if args.tok_per_s is None:
        return None
    return (tokens / args.tok_per_s) + (launches * args.qemu_launch_overhead_s)


def load_records(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[EvalWorkloadRow], list[Finding]]:
    rows: list[EvalWorkloadRow] = []
    findings: list[Finding] = []
    for path in paths:
        try:
            raw_rows = dataset_pack.read_jsonl(path)
        except (OSError, ValueError) as exc:
            findings.append(Finding(str(path), "error", "read_error", "input", "", "", str(exc)))
            continue
        for index, raw in enumerate(raw_rows):
            try:
                record = dataset_pack.normalize_row(raw, index, args.default_dataset, args.default_split)
            except ValueError as exc:
                findings.append(Finding(str(path), "error", "schema_error", f"row {index + 1}", "", "", str(exc)))
                continue
            prompt_bytes = len(record.prompt.encode("utf-8"))
            choice_byte_values = [len(choice.encode("utf-8")) for choice in record.choices]
            prompt_tokens = estimate_tokens(prompt_bytes, args.bytes_per_token)
            choice_tokens = sum(estimate_tokens(value, args.bytes_per_token) for value in choice_byte_values)
            scored_tokens = (prompt_tokens * len(record.choices)) + choice_tokens
            launches = launches_for_record(len(record.choices), args.launch_mode)
            rows.append(
                EvalWorkloadRow(
                    source=str(path),
                    row=index + 1,
                    dataset=record.dataset,
                    split=record.split,
                    record_id=record.record_id,
                    choices=len(record.choices),
                    prompt_bytes=prompt_bytes,
                    choice_bytes=sum(choice_byte_values),
                    prompt_tokens_est=prompt_tokens,
                    choice_tokens_est=choice_tokens,
                    scored_tokens_est=scored_tokens,
                    launches_est=launches,
                    wall_seconds_est=wall_seconds(scored_tokens, launches, args),
                )
            )
    if args.launch_mode == "single" and rows:
        first = rows[0]
        total_tokens = sum(row.scored_tokens_est for row in rows)
        total_wall = wall_seconds(total_tokens, 1, args)
        rows[0] = EvalWorkloadRow(
            first.source,
            first.row,
            first.dataset,
            first.split,
            first.record_id,
            first.choices,
            first.prompt_bytes,
            first.choice_bytes,
            first.prompt_tokens_est,
            first.choice_tokens_est,
            first.scored_tokens_est,
            1,
            total_wall,
        )
    return rows, findings


def scope_key(row: EvalWorkloadRow) -> str:
    return f"{row.dataset}:{row.split}"


def summarize(rows: list[EvalWorkloadRow], args: argparse.Namespace) -> list[ScopeEstimate]:
    grouped: dict[str, list[EvalWorkloadRow]] = {}
    for row in rows:
        grouped.setdefault(scope_key(row), []).append(row)
    estimates: list[ScopeEstimate] = []
    for scope, scope_rows in sorted(grouped.items()):
        dataset, split = scope.split(":", 1)
        tokens = sum(row.scored_tokens_est for row in scope_rows)
        launches = sum(row.launches_est for row in scope_rows)
        if args.launch_mode == "single":
            launches = 1
        estimates.append(
            ScopeEstimate(
                scope=scope,
                dataset=dataset,
                split=split,
                records=len(scope_rows),
                choices=sum(row.choices for row in scope_rows),
                prompt_bytes=sum(row.prompt_bytes for row in scope_rows),
                choice_bytes=sum(row.choice_bytes for row in scope_rows),
                prompt_tokens_est=sum(row.prompt_tokens_est for row in scope_rows),
                choice_tokens_est=sum(row.choice_tokens_est for row in scope_rows),
                scored_tokens_est=tokens,
                launches_est=launches,
                wall_seconds_est=wall_seconds(tokens, launches, args),
            )
        )
    return estimates


def gate(rows: list[EvalWorkloadRow], estimates: list[ScopeEstimate], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    total_records = sum(estimate.records for estimate in estimates)
    if total_records < args.min_records:
        findings.append(Finding("-", "error", "min_records", "records", str(total_records), str(args.min_records), "not enough normalized records"))
    for row in rows:
        label = f"{row.dataset}:{row.split}:{row.record_id}"
        if args.max_choices_per_record is not None and row.choices > args.max_choices_per_record:
            findings.append(
                Finding(
                    row.source,
                    "error",
                    "choices_per_record_budget",
                    label,
                    str(row.choices),
                    str(args.max_choices_per_record),
                    "record choice count exceeds budget",
                )
            )
        if args.max_record_scored_tokens is not None and row.scored_tokens_est > args.max_record_scored_tokens:
            findings.append(
                Finding(
                    row.source,
                    "error",
                    "record_scored_tokens_budget",
                    label,
                    str(row.scored_tokens_est),
                    str(args.max_record_scored_tokens),
                    "record estimated scored tokens exceed budget",
                )
            )
        if args.max_record_launches is not None and row.launches_est > args.max_record_launches:
            findings.append(
                Finding(
                    row.source,
                    "error",
                    "record_launch_budget",
                    label,
                    str(row.launches_est),
                    str(args.max_record_launches),
                    "record estimated QEMU launches exceed budget",
                )
            )
        if args.max_record_wall_seconds is not None and row.wall_seconds_est is not None and row.wall_seconds_est > args.max_record_wall_seconds:
            findings.append(
                Finding(
                    row.source,
                    "error",
                    "record_wall_time_budget",
                    label,
                    f"{row.wall_seconds_est:.3f}",
                    str(args.max_record_wall_seconds),
                    "record estimated wall time exceeds budget",
                )
            )
    for estimate in estimates:
        if estimate.records < args.min_records_per_scope:
            findings.append(
                Finding("-", "error", "min_records_per_scope", estimate.scope, str(estimate.records), str(args.min_records_per_scope), "not enough records")
            )
        if args.max_scored_tokens is not None and estimate.scored_tokens_est > args.max_scored_tokens:
            findings.append(
                Finding("-", "error", "scored_tokens_budget", estimate.scope, str(estimate.scored_tokens_est), str(args.max_scored_tokens), "estimated scored tokens exceed budget")
            )
        if args.max_launches is not None and estimate.launches_est > args.max_launches:
            findings.append(Finding("-", "error", "launch_budget", estimate.scope, str(estimate.launches_est), str(args.max_launches), "estimated QEMU launches exceed budget"))
        if args.max_wall_seconds is not None and estimate.wall_seconds_est is not None and estimate.wall_seconds_est > args.max_wall_seconds:
            findings.append(
                Finding("-", "error", "wall_time_budget", estimate.scope, f"{estimate.wall_seconds_est:.3f}", str(args.max_wall_seconds), "estimated wall time exceeds budget")
            )
    return findings


def build_report(rows: list[EvalWorkloadRow], estimates: list[ScopeEstimate], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "config": {
            "bytes_per_token": args.bytes_per_token,
            "tok_per_s": args.tok_per_s,
            "qemu_launch_overhead_s": args.qemu_launch_overhead_s,
            "launch_mode": args.launch_mode,
        },
        "summary": {
            "records": len(rows),
            "scopes": len(estimates),
            "choices": sum(row.choices for row in rows),
            "scored_tokens_est": sum(row.scored_tokens_est for row in rows),
            "launches_est": sum(estimate.launches_est for estimate in estimates),
            "wall_seconds_est": None if args.tok_per_s is None else sum(estimate.wall_seconds_est or 0.0 for estimate in estimates),
            "findings": len(findings),
        },
        "scopes": [asdict(estimate) for estimate in estimates],
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Eval Workload Estimate",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Records: {summary['records']}",
        f"Choices: {summary['choices']}",
        f"Estimated scored tokens: {summary['scored_tokens_est']}",
        f"Estimated launches: {summary['launches_est']}",
        f"Estimated wall seconds: {summary['wall_seconds_est']}",
        f"Findings: {summary['findings']}",
        "",
        "## Scopes",
        "",
    ]
    scopes = report["scopes"]
    assert isinstance(scopes, list)
    for scope in scopes:
        lines.append(f"- {scope['scope']}: records={scope['records']} tokens={scope['scored_tokens_est']} launches={scope['launches_est']}")
    findings = report["findings"]
    assert isinstance(findings, list)
    if findings:
        lines.extend(["", "## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding['kind']} {finding['field']}: {finding['detail']}")
    else:
        lines.extend(["", "No eval workload budget findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_eval_workload_estimate", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="workload_budget")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} workload finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Local eval JSONL inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_workload_estimate_latest")
    parser.add_argument("--default-dataset", default="local")
    parser.add_argument("--default-split", default="eval")
    parser.add_argument("--bytes-per-token", type=float, default=4.0)
    parser.add_argument("--tok-per-s", type=float)
    parser.add_argument("--qemu-launch-overhead-s", type=float, default=0.0)
    parser.add_argument("--launch-mode", choices=("per-record", "per-choice", "single"), default="per-record")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-records-per-scope", type=int, default=1)
    parser.add_argument("--max-choices-per-record", type=int)
    parser.add_argument("--max-record-scored-tokens", type=int)
    parser.add_argument("--max-record-launches", type=int)
    parser.add_argument("--max-record-wall-seconds", type=float)
    parser.add_argument("--max-scored-tokens", type=int)
    parser.add_argument("--max-launches", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bytes_per_token <= 0:
        parser.error("--bytes-per-token must be > 0")
    if args.tok_per_s is not None and args.tok_per_s <= 0:
        parser.error("--tok-per-s must be > 0")
    if args.qemu_launch_overhead_s < 0:
        parser.error("--qemu-launch-overhead-s must be >= 0")
    if args.min_records < 0 or args.min_records_per_scope < 0:
        parser.error("--min-records and --min-records-per-scope must be >= 0")
    for name in ("max_choices_per_record", "max_record_scored_tokens", "max_record_launches", "max_record_wall_seconds"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")

    rows, findings = load_records(args.inputs, args)
    estimates = summarize(rows, args)
    findings.extend(gate(rows, estimates, args))
    report = build_report(rows, estimates, findings, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), estimates, list(ScopeEstimate.__dataclass_fields__))
    write_csv(stem.with_name(stem.name + "_rows.csv"), rows, list(EvalWorkloadRow.__dataclass_fields__))
    write_csv(stem.with_name(stem.name + "_findings.csv"), findings, list(Finding.__dataclass_fields__))
    write_markdown(stem.with_suffix(".md"), report)
    write_junit(stem.with_name(stem.name + "_junit.xml"), findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
