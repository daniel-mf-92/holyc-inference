#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for per-prompt quantization pairing.

This host-side tool reads saved benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
DEFAULT_REQUIRED_QUANTS = ("Q4_0", "Q8_0")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class PairRow:
    profile: str
    model: str
    prompt: str
    phase: str
    iteration: str
    commit: str
    required_quantizations: str
    present_quantizations: str
    missing_quantizations: str
    rows: int
    successful_rows: int
    sources: str


@dataclass(frozen=True)
class Finding:
    profile: str
    model: str
    prompt: str
    phase: str
    iteration: str
    commit: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def is_success(row: dict[str, Any]) -> bool:
    exit_class = row_text(row, "exit_class", default="").lower()
    if exit_class:
        exit_ok = exit_class == "ok"
    else:
        exit_ok = str(row.get("returncode", "0")) == "0"
    return exit_ok and not bool_value(row.get("timed_out")) and row.get("failure_reason") in (None, "")


def pairing_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        row_text(row, "profile"),
        row_text(row, "model"),
        row_text(row, "prompt", "prompt_id"),
        row_text(row, "phase", default="measured"),
        row_text(row, "iteration"),
        row_text(row, "commit"),
    )


def collect_rows(paths: Iterable[Path], patterns: list[str], only_measured: bool) -> tuple[list[tuple[Path, dict[str, Any]]], list[Finding]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, patterns):
        seen_files += 1
        try:
            loaded = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding("-", "-", "-", "-", "-", "-", "error", "load_error", f"{path}: {exc}"))
            continue
        for row in loaded:
            if only_measured and row_text(row, "phase", default="measured") != "measured":
                continue
            rows.append((path, row))
    if seen_files == 0:
        findings.append(Finding("-", "-", "-", "-", "-", "-", "error", "no_inputs", "no benchmark artifacts matched"))
    return rows, findings


def build_report(rows: list[tuple[Path, dict[str, Any]]], initial_findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    required = tuple(args.required_quantization)
    required_set = set(required)
    grouped: dict[tuple[str, str, str, str, str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for path, row in rows:
        quant = row_text(row, "quantization", default="")
        if quant:
            grouped.setdefault(pairing_key(row), []).append((path, row))

    pair_rows: list[PairRow] = []
    findings = list(initial_findings)
    complete_pairs = 0
    for key in sorted(grouped):
        group = grouped[key]
        present = sorted({row_text(row, "quantization", default="") for _, row in group if row_text(row, "quantization", default="")})
        present_set = set(present)
        missing = [quant for quant in required if quant not in present_set]
        successful = sum(1 for _, row in group if row_text(row, "quantization", default="") in required_set and is_success(row))
        if not missing:
            complete_pairs += 1
        profile, model, prompt, phase, iteration, commit = key
        if missing:
            findings.append(
                Finding(
                    profile,
                    model,
                    prompt,
                    phase,
                    iteration,
                    commit,
                    "error",
                    "missing_quant_pair",
                    f"missing required quantization(s): {','.join(missing)}",
                )
            )
        if args.require_success and successful < len(required):
            findings.append(
                Finding(
                    profile,
                    model,
                    prompt,
                    phase,
                    iteration,
                    commit,
                    "error",
                    "incomplete_success_pair",
                    f"found {successful} successful required quantization rows, expected {len(required)}",
                )
            )
        pair_rows.append(
            PairRow(
                profile=profile,
                model=model,
                prompt=prompt,
                phase=phase,
                iteration=iteration,
                commit=commit,
                required_quantizations=",".join(required),
                present_quantizations=",".join(present),
                missing_quantizations=",".join(missing),
                rows=len(group),
                successful_rows=successful,
                sources=";".join(sorted({str(path) for path, _ in group})),
            )
        )

    if complete_pairs < args.min_pairs:
        findings.append(
            Finding("-", "-", "-", "-", "-", "-", "error", "min_pairs", f"found {complete_pairs} complete quant pairs, expected at least {args.min_pairs}")
        )

    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "required_quantizations": list(required),
        "summary": {
            "rows": len(rows),
            "pair_keys": len(pair_rows),
            "complete_pairs": complete_pairs,
            "findings": len(findings),
        },
        "pairs": [asdict(row) for row in pair_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Quant Pairing Audit",
        "",
        f"- Status: {report['status']}",
        f"- Required quantizations: {','.join(report['required_quantizations'])}",
        f"- Pair keys: {summary['pair_keys']}",
        f"- Complete pairs: {summary['complete_pairs']}",
        f"- Findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines += ["## Findings", ""]
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['profile']}/{finding['model']}/{finding['prompt']}/{finding['iteration']} - {finding['detail']}")
    else:
        lines.append("No quant pairing findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    failures = report["findings"]
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_quant_pairing_audit",
            "tests": str(max(1, report["summary"]["pair_keys"])),
            "failures": str(len(failures)),
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "quant_pairing"})
    if failures:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(failures)} quant pairing finding(s)"})
        failure.text = "\n".join(f"{item['kind']}: {item['detail']}" for item in failures)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> None:
    stem = args.output_stem
    output_dir = args.output_dir
    write_json(output_dir / f"{stem}.json", report)
    write_markdown(output_dir / f"{stem}.md", report)
    write_csv(
        output_dir / f"{stem}.csv",
        report["pairs"],
        [
            "profile",
            "model",
            "prompt",
            "phase",
            "iteration",
            "commit",
            "required_quantizations",
            "present_quantizations",
            "missing_quantizations",
            "rows",
            "successful_rows",
            "sources",
        ],
    )
    write_csv(
        output_dir / f"{stem}_findings.csv",
        report["findings"],
        ["profile", "model", "prompt", "phase", "iteration", "commit", "severity", "kind", "detail"],
    )
    write_junit(output_dir / f"{stem}_junit.xml", report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern for directory inputs")
    parser.add_argument("--required-quantization", action="append", default=list(DEFAULT_REQUIRED_QUANTS), help="Required quantization label")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_quant_pairing_audit_latest")
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--require-success", action="store_true", help="Require every required quantization row in a pair to be successful")
    parser.add_argument("--include-warmups", action="store_true", help="Include warmup rows in pairing keys")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows, findings = collect_rows(args.inputs, args.pattern, only_measured=not args.include_warmups)
    report = build_report(rows, findings, args)
    write_outputs(report, args)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
