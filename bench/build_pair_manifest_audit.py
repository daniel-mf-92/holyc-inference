#!/usr/bin/env python3
"""Audit build-pair selection manifests before perf regression CI consumes them.

This host-side tool reads outputs from build_pair_select.py only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PairRecord:
    source: str
    row: int
    key: str
    baseline_source: str
    candidate_source: str
    baseline_commit: str
    candidate_commit: str
    baseline_generated_at: str
    candidate_generated_at: str
    baseline_measured_runs: int | None
    candidate_measured_runs: int | None
    build_compare_args: str


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


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def text_value(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return str(value) if value not in (None, "") else ""


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    pairs = payload.get("pairs")
    if isinstance(pairs, list):
        for item in pairs:
            if isinstance(item, dict):
                yield item
        return
    yield payload


def normalize_args(value: Any) -> str:
    if isinstance(value, list):
        return shlex.join(str(item) for item in value)
    return str(value or "")


def load_pair_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return list(flatten_json_payload(json.loads(path.read_text(encoding="utf-8"))))
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.extend(flatten_json_payload(json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        return rows
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"{path}: unsupported input suffix {path.suffix}")


def parse_pair(source: Path, row_number: int, row: dict[str, Any]) -> PairRecord:
    return PairRecord(
        source=str(source),
        row=row_number,
        key=text_value(row, "key"),
        baseline_source=text_value(row, "baseline_source"),
        candidate_source=text_value(row, "candidate_source"),
        baseline_commit=text_value(row, "baseline_commit"),
        candidate_commit=text_value(row, "candidate_commit"),
        baseline_generated_at=text_value(row, "baseline_generated_at"),
        candidate_generated_at=text_value(row, "candidate_generated_at"),
        baseline_measured_runs=parse_int(row.get("baseline_measured_runs")),
        candidate_measured_runs=parse_int(row.get("candidate_measured_runs")),
        build_compare_args=normalize_args(row.get("build_compare_args")),
    )


def add_finding(findings: list[Finding], pair: PairRecord, kind: str, field: str, detail: str) -> None:
    findings.append(Finding(pair.source, pair.row, "error", kind, field, detail))


def audit_pair(pair: PairRecord, args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    required_text = (
        "key",
        "baseline_source",
        "candidate_source",
        "baseline_commit",
        "candidate_commit",
        "baseline_generated_at",
        "candidate_generated_at",
        "build_compare_args",
    )
    for field in required_text:
        if not getattr(pair, field):
            add_finding(findings, pair, "missing_field", field, f"{field} is required")

    if pair.baseline_source and pair.candidate_source and pair.baseline_source == pair.candidate_source:
        add_finding(findings, pair, "same_source", "candidate_source", "baseline and candidate sources must differ")
    if (
        not args.allow_same_commit
        and pair.baseline_commit
        and pair.candidate_commit
        and pair.baseline_commit == pair.candidate_commit
    ):
        add_finding(findings, pair, "same_commit", "candidate_commit", "baseline and candidate commits must differ")

    baseline_time = parse_iso(pair.baseline_generated_at)
    candidate_time = parse_iso(pair.candidate_generated_at)
    if pair.baseline_generated_at and baseline_time is None:
        add_finding(findings, pair, "invalid_timestamp", "baseline_generated_at", "expected timezone-aware ISO timestamp")
    if pair.candidate_generated_at and candidate_time is None:
        add_finding(findings, pair, "invalid_timestamp", "candidate_generated_at", "expected timezone-aware ISO timestamp")
    if baseline_time and candidate_time and candidate_time < baseline_time:
        add_finding(findings, pair, "candidate_older_than_baseline", "candidate_generated_at", "candidate is older than baseline")

    for field in ("baseline_measured_runs", "candidate_measured_runs"):
        value = getattr(pair, field)
        if value is None or value < args.min_measured_runs:
            add_finding(findings, pair, "insufficient_runs", field, f"{field} must be at least {args.min_measured_runs}")

    if pair.build_compare_args:
        if pair.baseline_source and pair.baseline_source not in pair.build_compare_args:
            add_finding(findings, pair, "missing_build_compare_source", "build_compare_args", "baseline source absent from build_compare_args")
        if pair.candidate_source and pair.candidate_source not in pair.build_compare_args:
            add_finding(findings, pair, "missing_build_compare_source", "build_compare_args", "candidate source absent from build_compare_args")
    return findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PairRecord], list[Finding]]:
    pairs: list[PairRecord] = []
    findings: list[Finding] = []
    for path in paths:
        for row_number, row in enumerate(load_pair_rows(path), 1):
            pair = parse_pair(path, row_number, row)
            pairs.append(pair)
            findings.extend(audit_pair(pair, args))
    if len(pairs) < args.min_pairs:
        findings.append(Finding("", 0, "error", "min_pairs", "pairs", f"found {len(pairs)} pairs below required {args.min_pairs}"))
    return pairs, findings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Build Pair Manifest Audit",
        "",
        f"Status: {payload['status']}",
        f"Pairs: {payload['summary']['pairs']}",
        f"Findings: {payload['summary']['findings']}",
        "",
    ]
    if payload["findings"]:
        lines.extend(["| severity | kind | source | row | field | detail |", "| --- | --- | --- | ---: | --- | --- |"])
        for finding in payload["findings"]:
            lines.append(
                f"| {finding['severity']} | {finding['kind']} | {finding['source']} | {finding['row']} | {finding['field']} | {finding['detail']} |"
            )
    else:
        lines.append("No build-pair manifest findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, payload: dict[str, Any]) -> None:
    findings = payload["findings"]
    suite = ET.Element("testsuite", name="holyc_build_pair_manifest_audit", tests=str(max(1, len(payload["pairs"]))), failures=str(len(findings)))
    if not payload["pairs"]:
        ET.SubElement(suite, "testcase", name="manifest_nonempty", classname="build_pair_manifest_audit")
    for pair in payload["pairs"]:
        case = ET.SubElement(suite, "testcase", name=f"{pair['source']}:{pair['row']}", classname="build_pair_manifest_audit")
        pair_findings = [finding for finding in findings if finding["source"] == pair["source"] and finding["row"] == pair["row"]]
        for finding in pair_findings:
            failure = ET.SubElement(case, "failure", type=finding["kind"], message=finding["detail"])
            failure.text = json.dumps(finding, sort_keys=True)
    for finding in [item for item in findings if not item["source"]]:
        case = ET.SubElement(suite, "testcase", name=finding["kind"], classname="build_pair_manifest_audit")
        failure = ET.SubElement(case, "failure", type=finding["kind"], message=finding["detail"])
        failure.text = json.dumps(finding, sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="build_pair_select JSON/JSONL/CSV outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="build_pair_manifest_audit_latest")
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--min-measured-runs", type=int, default=1)
    parser.add_argument("--allow-same-commit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs, findings = audit(args.inputs, args)
    status = "fail" if findings else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {"inputs": len(args.inputs), "pairs": len(pairs), "findings": len(findings)},
        "pairs": [asdict(pair) for pair in pairs],
        "findings": [asdict(finding) for finding in findings],
        "tool": "build_pair_manifest_audit",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.output_stem}.json", payload)
    write_csv(args.output_dir / f"{args.output_stem}.csv", pairs, list(PairRecord.__dataclass_fields__))
    write_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings, list(Finding.__dataclass_fields__))
    write_markdown(args.output_dir / f"{args.output_stem}.md", payload)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", payload)
    print(json.dumps({"status": status, "pairs": len(pairs), "findings": len(findings)}, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
