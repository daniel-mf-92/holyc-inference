#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for warmup/measured phase sequencing.

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
KNOWN_PHASES = {"warmup", "measured"}


@dataclass(frozen=True)
class PhaseRun:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    phase: str
    iteration: int | None
    exit_class: str


@dataclass(frozen=True)
class PhaseGroup:
    source: str
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    warmups: int
    measured: int
    measured_ok: int
    total: int
    min_row: int
    max_row: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    group: str
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


def group_key(run: PhaseRun) -> tuple[str, str, str, str, str, str]:
    return (run.source, run.profile, run.model, run.quantization, run.prompt, run.commit)


def group_label(key: tuple[str, str, str, str, str, str]) -> str:
    return "/".join(key[1:])


def phase_run(source: Path, row_number: int, row: dict[str, Any]) -> PhaseRun:
    return PhaseRun(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt", "prompt_id"),
        commit=row_text(row, "commit"),
        phase=row_text(row, "phase", default="measured").lower(),
        iteration=finite_int(row.get("iteration")),
        exit_class=row_text(row, "exit_class"),
    )


def phase_group(key: tuple[str, str, str, str, str, str], runs: list[PhaseRun]) -> PhaseGroup:
    warmups = [run for run in runs if run.phase == "warmup"]
    measured = [run for run in runs if run.phase == "measured"]
    measured_ok = [run for run in measured if run.exit_class in {"ok", "-"}]
    return PhaseGroup(
        source=key[0],
        profile=key[1],
        model=key[2],
        quantization=key[3],
        prompt=key[4],
        commit=key[5],
        warmups=len(warmups),
        measured=len(measured),
        measured_ok=len(measured_ok),
        total=len(runs),
        min_row=min(run.row for run in runs),
        max_row=max(run.row for run in runs),
    )


def duplicate_iterations(runs: list[PhaseRun]) -> list[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    duplicates: list[tuple[str, int]] = []
    for run in runs:
        if run.iteration is None:
            continue
        key = (run.phase, run.iteration)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PhaseRun], list[PhaseGroup], list[Finding]]:
    runs: list[PhaseRun] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        try:
            rows = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "-", "-", str(exc)))
            continue
        for row_number, row in enumerate(rows, 1):
            run = phase_run(path, row_number, row)
            label = group_label(group_key(run))
            runs.append(run)
            if run.phase not in KNOWN_PHASES:
                findings.append(Finding(str(path), row_number, "error", "unknown_phase", label, "phase", f"phase {run.phase!r} is not one of warmup/measured"))
            if args.require_iteration and run.iteration is None:
                findings.append(Finding(str(path), row_number, "error", "missing_iteration", label, "iteration", "iteration must be an integer"))
            elif run.iteration is not None and run.iteration < 1:
                findings.append(Finding(str(path), row_number, "error", "invalid_iteration", label, "iteration", "iteration must be >= 1"))

    grouped: dict[tuple[str, str, str, str, str, str], list[PhaseRun]] = {}
    for run in runs:
        grouped.setdefault(group_key(run), []).append(run)

    groups: list[PhaseGroup] = []
    for key in sorted(grouped):
        group_runs = sorted(grouped[key], key=lambda run: run.row)
        groups.append(phase_group(key, group_runs))
        label = group_label(key)
        seen_measured = False
        for run in group_runs:
            if run.phase == "measured":
                seen_measured = True
            elif run.phase == "warmup" and seen_measured:
                findings.append(Finding(run.source, run.row, "error", "warmup_after_measured", label, "phase", "warmup row appears after measured rows"))
        for phase, iteration in duplicate_iterations(group_runs):
            findings.append(Finding(key[0], 0, "error", "duplicate_iteration", label, f"{phase}.iteration", f"iteration {iteration} appears more than once"))

    if seen_files < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", "-", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(runs) < args.min_rows:
        findings.append(Finding("-", 0, "error", "min_rows", "-", "rows", f"found {len(runs)}, expected at least {args.min_rows}"))
    for group in groups:
        label = group_label((group.source, group.profile, group.model, group.quantization, group.prompt, group.commit))
        if group.warmups < args.min_warmups_per_group:
            findings.append(Finding(group.source, 0, "error", "min_warmups_per_group", label, "warmups", f"found {group.warmups}, expected at least {args.min_warmups_per_group}"))
        if group.measured < args.min_measured_per_group:
            findings.append(Finding(group.source, 0, "error", "min_measured_per_group", label, "measured", f"found {group.measured}, expected at least {args.min_measured_per_group}"))
        if args.require_measured_ok and group.measured_ok < group.measured:
            findings.append(Finding(group.source, 0, "error", "measured_not_ok", label, "measured_ok", f"{group.measured_ok}/{group.measured} measured rows have exit_class ok"))

    return runs, groups, findings


def summary(runs: list[PhaseRun], groups: list[PhaseGroup], findings: list[Finding]) -> dict[str, Any]:
    return {
        "rows": len(runs),
        "groups": len(groups),
        "warmups": sum(1 for run in runs if run.phase == "warmup"),
        "measured": sum(1 for run in runs if run.phase == "measured"),
        "measured_ok": sum(1 for run in runs if run.phase == "measured" and run.exit_class in {"ok", "-"}),
        "findings": len(findings),
        "profiles": sorted({run.profile for run in runs if run.profile != "-"}),
        "models": sorted({run.model for run in runs if run.model != "-"}),
        "quantizations": sorted({run.quantization for run in runs if run.quantization != "-"}),
        "prompts": sorted({run.prompt for run in runs if run.prompt != "-"}),
    }


def write_json(path: Path, runs: list[PhaseRun], groups: list[PhaseGroup], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(runs, groups, findings),
        "groups": [asdict(group) for group in groups],
        "runs": [asdict(run) for run in runs],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, runs: list[PhaseRun], groups: list[PhaseGroup], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Phase Sequence Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Rows: {len(runs)}",
        f"Groups: {len(groups)}",
        f"Findings: {len(findings)}",
        "",
        "## Groups",
        "",
        "| Profile | Model | Quantization | Prompt | Commit | Warmups | Measured | Measured OK |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group.profile} | {group.model} | {group.quantization} | {group.prompt} | {group.commit} | "
            f"{group.warmups} | {group.measured} | {group.measured_ok} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Group | Metric | Detail |", "| --- | ---: | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.group} | {finding.metric} | {finding.detail} |")
    else:
        lines.append("No phase sequence findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, groups: list[PhaseGroup]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PhaseGroup.__dataclass_fields__))
        writer.writeheader()
        for group in groups:
            writer.writerow(asdict(group))


def write_runs_csv(path: Path, runs: list[PhaseRun]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PhaseRun.__dataclass_fields__))
        writer.writeheader()
        for run in runs:
            writer.writerow(asdict(run))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_phase_sequence_audit", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="qemu_phase_sequence")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} phase sequence finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.group}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern for directory inputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_phase_sequence_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-warmups-per-group", type=int, default=0)
    parser.add_argument("--min-measured-per-group", type=int, default=1)
    parser.add_argument("--no-require-iteration", dest="require_iteration", action="store_false")
    parser.add_argument("--require-measured-ok", action="store_true")
    parser.set_defaults(require_iteration=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs, groups, findings = audit(args.paths, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", runs, groups, findings)
    write_markdown(args.output_dir / f"{stem}.md", runs, groups, findings)
    write_csv(args.output_dir / f"{stem}.csv", groups)
    write_runs_csv(args.output_dir / f"{stem}_runs.csv", runs)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
