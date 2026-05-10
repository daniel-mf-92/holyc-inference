#!/usr/bin/env python3
"""Join eval quality reports with QEMU prompt benchmark throughput.

This host-side tool consumes existing artifacts only. It never launches QEMU,
never touches the TempleOS guest, and expects benchmark commands to have been
captured with the repository air-gap policy (`-nic none`).
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


@dataclass(frozen=True)
class EvalReport:
    source: str
    status: str
    dataset: str
    split: str
    model: str
    quantization: str
    records: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    agreement: float | None
    regressions: int
    error: str = ""


@dataclass(frozen=True)
class PerfSummary:
    source: str
    model: str
    quantization: str
    profile: str
    prompt_count: int
    run_count: int
    ok_runs: int
    min_tok_per_s: float | None
    median_tok_per_s: float | None
    min_wall_tok_per_s: float | None
    median_wall_tok_per_s: float | None
    max_memory_bytes: int | None
    airgap_violations: int


@dataclass(frozen=True)
class ScorecardRow:
    model: str
    quantization: str
    dataset: str
    split: str
    eval_source: str
    perf_source: str
    status: str
    records: int
    holyc_accuracy: float | None
    llama_accuracy: float | None
    agreement: float | None
    regressions: int
    prompt_count: int
    ok_runs: int
    min_tok_per_s: float | None
    median_tok_per_s: float | None
    min_wall_tok_per_s: float | None
    median_wall_tok_per_s: float | None
    max_memory_bytes: int | None
    airgap_violations: int


@dataclass(frozen=True)
class Finding:
    gate: str
    model: str
    quantization: str
    dataset: str
    split: str
    value: float | int | str | None
    threshold: float | int | str | None
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: Any) -> int:
    number = as_float(value)
    return int(number) if number is not None else 0


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def load_json(path: Path) -> tuple[Any | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"


def iter_json_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.json") if child.is_file()))
        elif path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
    return sorted(files)


def load_eval_report(path: Path) -> EvalReport:
    payload, error = load_json(path)
    if not isinstance(payload, dict):
        return EvalReport(str(path), "invalid", "", "", "", "", 0, None, None, None, 0, error or "root must be object")
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return EvalReport(
            str(path),
            "invalid",
            str(payload.get("dataset") or ""),
            str(payload.get("split") or ""),
            str(payload.get("model") or ""),
            str(payload.get("quantization") or ""),
            0,
            None,
            None,
            None,
            len(payload.get("regressions") or []),
            "missing summary object",
        )
    return EvalReport(
        source=str(path),
        status=str(payload.get("status") or "pass").lower(),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        records=as_int(summary.get("record_count")),
        holyc_accuracy=as_float(summary.get("holyc_accuracy")),
        llama_accuracy=as_float(summary.get("llama_accuracy")),
        agreement=as_float(summary.get("agreement")),
        regressions=len(payload.get("regressions") or []),
    )


def flatten_bench_rows(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for key in ("benchmarks", "results", "runs", "rows"):
        rows = payload.get(key)
        if isinstance(rows, list):
            inherited = {name: value for name, value in payload.items() if name != key}
            for row in rows:
                if isinstance(row, dict):
                    merged = dict(inherited)
                    merged.update(row)
                    yield merged
            return
    yield payload


def command_airgap_violations(row: dict[str, Any]) -> int:
    if row.get("command_airgap_ok") is False:
        return 1
    violations = row.get("command_airgap_violations")
    if isinstance(violations, list):
        return len(violations)
    command = row.get("command")
    if isinstance(command, list):
        tokens = [str(token) for token in command]
        if "-nic" in tokens:
            nic_index = tokens.index("-nic")
            if nic_index + 1 < len(tokens) and tokens[nic_index + 1] == "none":
                return 0
        return 1
    return 0


def load_perf_summaries(path: Path) -> list[PerfSummary]:
    payload, error = load_json(path)
    if error or payload is None:
        return []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in flatten_bench_rows(payload):
        if str(row.get("phase") or "measured") == "warmup":
            continue
        model = str(row.get("model") or "")
        quantization = str(row.get("quantization") or "")
        profile = str(row.get("profile") or "")
        if not model or not quantization:
            continue
        groups.setdefault((model, quantization, profile), []).append(row)

    summaries: list[PerfSummary] = []
    for (model, quantization, profile), rows in sorted(groups.items()):
        ok_rows = [
            row
            for row in rows
            if str(row.get("exit_class") or "") == "ok"
            and not bool(row.get("timed_out"))
            and row.get("failure_reason") in (None, "")
        ]
        tok_values = [value for row in ok_rows if (value := as_float(row.get("tok_per_s"))) is not None]
        wall_values = [value for row in ok_rows if (value := as_float(row.get("wall_tok_per_s"))) is not None]
        memory_values = [as_int(row.get("memory_bytes")) for row in ok_rows if as_int(row.get("memory_bytes")) > 0]
        prompts = {str(row.get("prompt") or row.get("prompt_id") or "") for row in rows}
        summaries.append(
            PerfSummary(
                source=str(path),
                model=model,
                quantization=quantization,
                profile=profile,
                prompt_count=len({prompt for prompt in prompts if prompt}),
                run_count=len(rows),
                ok_runs=len(ok_rows),
                min_tok_per_s=min(tok_values) if tok_values else None,
                median_tok_per_s=median(tok_values),
                min_wall_tok_per_s=min(wall_values) if wall_values else None,
                median_wall_tok_per_s=median(wall_values),
                max_memory_bytes=max(memory_values) if memory_values else None,
                airgap_violations=sum(command_airgap_violations(row) for row in rows),
            )
        )
    return summaries


def pick_perf(summary: list[PerfSummary], model: str, quantization: str) -> PerfSummary | None:
    matches = [item for item in summary if item.model == model and item.quantization == quantization]
    if not matches:
        return None
    return sorted(matches, key=lambda item: (item.ok_runs, item.prompt_count, item.median_tok_per_s or 0.0), reverse=True)[0]


def build_scorecard(eval_reports: list[EvalReport], perf_summaries: list[PerfSummary]) -> list[ScorecardRow]:
    rows: list[ScorecardRow] = []
    for report in eval_reports:
        perf = pick_perf(perf_summaries, report.model, report.quantization)
        rows.append(
            ScorecardRow(
                model=report.model,
                quantization=report.quantization,
                dataset=report.dataset,
                split=report.split,
                eval_source=report.source,
                perf_source=perf.source if perf else "",
                status="pass",
                records=report.records,
                holyc_accuracy=report.holyc_accuracy,
                llama_accuracy=report.llama_accuracy,
                agreement=report.agreement,
                regressions=report.regressions,
                prompt_count=perf.prompt_count if perf else 0,
                ok_runs=perf.ok_runs if perf else 0,
                min_tok_per_s=perf.min_tok_per_s if perf else None,
                median_tok_per_s=perf.median_tok_per_s if perf else None,
                min_wall_tok_per_s=perf.min_wall_tok_per_s if perf else None,
                median_wall_tok_per_s=perf.median_wall_tok_per_s if perf else None,
                max_memory_bytes=perf.max_memory_bytes if perf else None,
                airgap_violations=perf.airgap_violations if perf else 0,
            )
        )
    return rows


def evaluate(rows: list[ScorecardRow], eval_reports: list[EvalReport], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    if len(rows) < args.min_eval_reports:
        findings.append(Finding("min_eval_reports", "", "", "", "", len(rows), args.min_eval_reports, "too few eval reports"))
    if sum(row.records for row in rows) < args.min_records:
        findings.append(
            Finding("min_records", "", "", "", "", sum(row.records for row in rows), args.min_records, "too few eval records")
        )
    for report in eval_reports:
        if report.status == "invalid":
            findings.append(
                Finding("invalid_eval_report", report.model, report.quantization, report.dataset, report.split, report.error, "valid", report.error)
            )
    for row in rows:
        identity = (row.model, row.quantization, row.dataset, row.split)
        if args.fail_on_failed_eval and row.eval_source and any(
            report.source == row.eval_source and report.status != "pass" for report in eval_reports
        ):
            findings.append(Finding("eval_status", *identity, "fail", "pass", f"{row.eval_source} did not pass"))
        if args.fail_on_regressions and row.regressions:
            findings.append(Finding("eval_regressions", *identity, row.regressions, 0, "eval report has regressions"))
        if args.require_perf_match and not row.perf_source:
            findings.append(Finding("perf_match", *identity, "missing", "present", "no benchmark artifact matched model and quantization"))
        if args.min_holyc_accuracy is not None and (row.holyc_accuracy is None or row.holyc_accuracy < args.min_holyc_accuracy):
            findings.append(Finding("min_holyc_accuracy", *identity, row.holyc_accuracy, args.min_holyc_accuracy, "HolyC accuracy below gate"))
        if args.min_agreement is not None and (row.agreement is None or row.agreement < args.min_agreement):
            findings.append(Finding("min_agreement", *identity, row.agreement, args.min_agreement, "HolyC/llama agreement below gate"))
        if args.min_ok_runs is not None and row.ok_runs < args.min_ok_runs:
            findings.append(Finding("min_ok_runs", *identity, row.ok_runs, args.min_ok_runs, "too few successful benchmark runs"))
        if args.min_prompts is not None and row.prompt_count < args.min_prompts:
            findings.append(Finding("min_prompts", *identity, row.prompt_count, args.min_prompts, "too few benchmark prompts"))
        if args.min_tok_per_s is not None and (row.min_tok_per_s is None or row.min_tok_per_s < args.min_tok_per_s):
            findings.append(Finding("min_tok_per_s", *identity, row.min_tok_per_s, args.min_tok_per_s, "guest tok/s below gate"))
        if args.min_wall_tok_per_s is not None and (
            row.min_wall_tok_per_s is None or row.min_wall_tok_per_s < args.min_wall_tok_per_s
        ):
            findings.append(Finding("min_wall_tok_per_s", *identity, row.min_wall_tok_per_s, args.min_wall_tok_per_s, "wall tok/s below gate"))
        if args.max_memory_bytes is not None and (row.max_memory_bytes is None or row.max_memory_bytes > args.max_memory_bytes):
            findings.append(Finding("max_memory_bytes", *identity, row.max_memory_bytes, args.max_memory_bytes, "memory above gate"))
        if row.airgap_violations:
            findings.append(Finding("airgap", *identity, row.airgap_violations, 0, "benchmark command artifact has air-gap violations"))
    return findings


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    fieldnames = list(dictionaries[0]) if dictionaries else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dictionaries)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Eval Perf Scorecard",
        "",
        f"- Status: {payload['status']}",
        f"- Eval reports: {payload['summary']['eval_reports']}",
        f"- Total records: {payload['summary']['records']}",
        f"- Scorecard rows: {payload['summary']['scorecard_rows']}",
        f"- Findings: {payload['summary']['findings']}",
        "",
        "| model | quantization | dataset | records | accuracy | agreement | min tok/s | min wall tok/s | ok runs |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["scorecard"]:
        lines.append(
            "| {model} | {quantization} | {dataset}/{split} | {records} | {holyc_accuracy} | {agreement} | {min_tok_per_s} | {min_wall_tok_per_s} | {ok_runs} |".format(
                **{key: ("" if value is None else value) for key, value in row.items()}
            )
        )
    if payload["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in payload["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_eval_perf_scorecard",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "scorecard_gates"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} scorecard finding(s)"})
        failure.text = "\n".join(f"{finding.gate}: {finding.message}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", nargs="+", type=Path, required=True, help="eval_compare JSON files or directories")
    parser.add_argument("--bench", nargs="+", type=Path, required=True, help="qemu_prompt_bench JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="eval_perf_scorecard_latest")
    parser.add_argument("--min-eval-reports", type=int, default=1)
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--min-holyc-accuracy", type=float)
    parser.add_argument("--min-agreement", type=float)
    parser.add_argument("--min-ok-runs", type=int)
    parser.add_argument("--min-prompts", type=int)
    parser.add_argument("--min-tok-per-s", type=float)
    parser.add_argument("--min-wall-tok-per-s", type=float)
    parser.add_argument("--max-memory-bytes", type=int)
    parser.add_argument("--require-perf-match", action="store_true")
    parser.add_argument("--fail-on-failed-eval", action="store_true")
    parser.add_argument("--fail-on-regressions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_reports = [load_eval_report(path) for path in iter_json_files(args.eval)]
    perf_summaries: list[PerfSummary] = []
    for path in iter_json_files(args.bench):
        perf_summaries.extend(load_perf_summaries(path))
    rows = build_scorecard(eval_reports, perf_summaries)
    findings = evaluate(rows, eval_reports, args)
    rows = [ScorecardRow(**{**asdict(row), "status": "fail" if findings else "pass"}) for row in rows]
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "eval_reports": len(eval_reports),
            "perf_summaries": len(perf_summaries),
            "scorecard_rows": len(rows),
            "records": sum(row.records for row in rows),
            "findings": len(findings),
        },
        "scorecard": [asdict(row) for row in rows],
        "perf_summaries": [asdict(summary) for summary in perf_summaries],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.output_stem
    (stem.with_suffix(".json")).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(stem.with_suffix(".csv"), rows)
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_markdown(stem.with_suffix(".md"), payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
