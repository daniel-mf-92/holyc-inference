#!/usr/bin/env python3
"""Audit saved QEMU benchmark rows for deterministic seed metadata.

This host-side tool reads existing benchmark JSON/JSONL/CSV artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("warmups", "benchmarks", "results", "runs", "rows")
SEED_KEYS = ("seed", "rng_seed", "sampler_seed")


@dataclass(frozen=True)
class SeedRow:
    source: str
    row: int
    key: str
    profile: str
    model: str
    quantization: str
    phase: str
    prompt: str
    iteration: str
    commit: str
    exit_class: str
    seed: int | None
    seed_source: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    key: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text_value(value: Any, default: str = "") -> str:
    if value in (None, ""):
        return default
    return str(value)


def finite_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value.is_integer() else None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(parsed) or not parsed.is_integer():
        return None
    return int(parsed)


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


def seed_value(row: dict[str, Any]) -> tuple[int | None, str, bool]:
    for key in SEED_KEYS:
        if key in row and row.get(key) not in (None, ""):
            return finite_int(row.get(key)), key, True
    return None, "", False


def include_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    phase = text_value(row.get("phase"), "measured").lower()
    exit_class = text_value(row.get("exit_class")).lower()
    if args.only_measured and phase != "measured":
        return False
    return args.include_failed or exit_class in ("", "ok")


def row_key(row: dict[str, Any], args: argparse.Namespace) -> str:
    return "|".join(text_value(row.get(field), "-") for field in args.key_fields)


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SeedRow], list[Finding]]:
    rows: list[SeedRow] = []
    findings: list[Finding] = []
    grouped: dict[str, list[SeedRow]] = defaultdict(list)
    files = 0

    for path in iter_input_files(paths, args.pattern):
        files += 1
        try:
            loaded = list(load_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(Finding(str(path), 0, "error", "load_error", "", str(exc)))
            continue

        for row_number, raw in enumerate(loaded, 1):
            if not include_row(raw, args):
                continue
            seed, seed_source, present = seed_value(raw)
            key = row_key(raw, args)
            record = SeedRow(
                source=str(path),
                row=row_number,
                key=key,
                profile=text_value(raw.get("profile")),
                model=text_value(raw.get("model")),
                quantization=text_value(raw.get("quantization")),
                phase=text_value(raw.get("phase"), "measured").lower(),
                prompt=text_value(raw.get("prompt") or raw.get("prompt_id")),
                iteration=text_value(raw.get("iteration")),
                commit=text_value(raw.get("commit") or raw.get("git_commit")),
                exit_class=text_value(raw.get("exit_class")).lower(),
                seed=seed,
                seed_source=seed_source,
            )
            rows.append(record)
            grouped[key].append(record)
            if args.require_seed and not present:
                findings.append(Finding(str(path), row_number, "error", "missing_seed", key, "row has no seed/rng_seed/sampler_seed field"))
            elif present and seed is None:
                findings.append(Finding(str(path), row_number, "error", "invalid_seed", key, f"{seed_source} must be an integer"))
            elif seed is not None and seed < 0:
                findings.append(Finding(str(path), row_number, "error", "negative_seed", key, f"{seed_source}={seed}"))

    for key, group in sorted(grouped.items()):
        seeds = {row.seed for row in group if row.seed is not None}
        if args.require_stable_seed and len(seeds) > 1:
            first = group[0]
            findings.append(Finding(first.source, first.row, "error", "seed_drift", key, f"observed seeds: {','.join(map(str, sorted(seeds)))}"))

    if files == 0:
        findings.append(Finding("", 0, "error", "no_input_files", "", "no benchmark artifacts matched input paths/patterns"))
    if len(rows) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "", f"found {len(rows)} row(s), below minimum {args.min_rows}"))
    return rows, findings


def build_report(rows: list[SeedRow], findings: list[Finding]) -> dict[str, Any]:
    seeded_rows = sum(1 for row in rows if row.seed is not None)
    unique_seeds = {row.seed for row in rows if row.seed is not None}
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "seeded_rows": seeded_rows,
            "unique_seeds": len(unique_seeds),
            "missing_seed_rows": sum(1 for finding in findings if finding.kind == "missing_seed"),
            "invalid_seed_rows": sum(1 for finding in findings if finding.kind == "invalid_seed"),
            "negative_seed_rows": sum(1 for finding in findings if finding.kind == "negative_seed"),
            "seed_drift_groups": sum(1 for finding in findings if finding.kind == "seed_drift"),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[SeedRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SeedRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Seed Audit",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- status: {report['status']}",
        f"- rows: {summary['rows']}",
        f"- seeded_rows: {summary['seeded_rows']}",
        f"- unique_seeds: {summary['unique_seeds']}",
        f"- seed_drift_groups: {summary['seed_drift_groups']}",
        f"- findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", "", "| source | row | kind | key | detail |", "| --- | ---: | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['source']} | {finding['row']} | {finding['kind']} | {finding['key']} | {finding['detail']} |")
    else:
        lines.append("No seed metadata findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_seed_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "qemu_seed_metadata"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob pattern used when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Directory for audit outputs")
    parser.add_argument("--output-stem", default="qemu_seed_audit_latest", help="Output filename stem")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum included rows required")
    parser.add_argument("--key-fields", nargs="+", default=["profile", "model", "quantization", "prompt", "iteration", "commit"], help="Fields that define a deterministic seed group")
    parser.add_argument("--include-failed", action="store_true", help="Include failed rows instead of only ok/unspecified exit_class rows")
    parser.add_argument("--only-measured", action=argparse.BooleanOptionalAction, default=True, help="Only audit measured rows")
    parser.add_argument("--require-seed", action=argparse.BooleanOptionalAction, default=True, help="Require seed/rng_seed/sampler_seed on included rows")
    parser.add_argument("--require-stable-seed", action=argparse.BooleanOptionalAction, default=True, help="Require one seed per deterministic key group")
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
