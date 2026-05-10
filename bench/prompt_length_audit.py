#!/usr/bin/env python3
"""Audit benchmark prompt suites for length-bucket coverage.

This host-side tool validates local prompt suites before QEMU benchmark runs.
It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
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


@dataclass(frozen=True)
class Bucket:
    name: str
    min_bytes: int
    max_bytes: int | None


@dataclass(frozen=True)
class PromptRow:
    source: str
    prompt_id: str
    prompt_sha256: str
    prompt_bytes: int
    expected_tokens: int | None
    bucket: str


@dataclass(frozen=True)
class SourceAudit:
    source: str
    status: str
    prompts: int
    prompt_bytes_min: int | None
    prompt_bytes_max: int | None
    prompt_bytes_total: int
    prompt_bytes_avg: float | None
    duplicate_prompt_sha256: int
    bucket_counts: dict[str, int]
    suite_sha256: str
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    kind: str
    detail: str


DEFAULT_BUCKETS = (
    Bucket("short", 0, 128),
    Bucket("medium", 129, 512),
    Bucket("long", 513, None),
)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_bucket(value: str) -> Bucket:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bucket must be NAME:MIN_BYTES:MAX_BYTES, with empty MAX_BYTES allowed")
    name, min_text, max_text = (part.strip() for part in parts)
    if not name:
        raise argparse.ArgumentTypeError("bucket name cannot be empty")
    try:
        min_bytes = int(min_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bucket minimum: {min_text!r}") from exc
    if min_bytes < 0:
        raise argparse.ArgumentTypeError("bucket minimum cannot be negative")
    max_bytes: int | None
    if max_text == "":
        max_bytes = None
    else:
        try:
            max_bytes = int(max_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid bucket maximum: {max_text!r}") from exc
        if max_bytes < min_bytes:
            raise argparse.ArgumentTypeError("bucket maximum cannot be below minimum")
    return Bucket(name=name, min_bytes=min_bytes, max_bytes=max_bytes)


def parse_min_bucket(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("min bucket gate must be NAME=COUNT")
    name, count_text = (part.strip() for part in value.split("=", 1))
    if not name:
        raise argparse.ArgumentTypeError("bucket name cannot be empty")
    try:
        count = int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid bucket count: {count_text!r}") from exc
    if count < 0:
        raise argparse.ArgumentTypeError("bucket count cannot be negative")
    return name, count


def bucket_for_prompt(prompt_bytes: int, buckets: list[Bucket]) -> str:
    for bucket in buckets:
        if prompt_bytes < bucket.min_bytes:
            continue
        if bucket.max_bytes is not None and prompt_bytes > bucket.max_bytes:
            continue
        return bucket.name
    return "unbucketed"


def audit_source(path: Path, buckets: list[Bucket], args: argparse.Namespace) -> tuple[SourceAudit, list[PromptRow], list[Finding]]:
    findings: list[Finding] = []
    try:
        cases = qemu_prompt_bench.load_prompt_cases(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return (
            SourceAudit(
                source=str(path),
                status="fail",
                prompts=0,
                prompt_bytes_min=None,
                prompt_bytes_max=None,
                prompt_bytes_total=0,
                prompt_bytes_avg=None,
                duplicate_prompt_sha256=0,
                bucket_counts={bucket.name: 0 for bucket in buckets},
                suite_sha256="",
                error=str(exc),
            ),
            [],
            [Finding(str(path), "load_error", str(exc))],
        )

    rows: list[PromptRow] = []
    lengths: list[int] = []
    hashes: list[str] = []
    bucket_counts: dict[str, int] = {bucket.name: 0 for bucket in buckets}
    bucket_counts["unbucketed"] = 0
    for case in cases:
        size = qemu_prompt_bench.prompt_bytes(case.prompt)
        digest = qemu_prompt_bench.prompt_hash(case.prompt)
        bucket_name = bucket_for_prompt(size, buckets)
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
        rows.append(
            PromptRow(
                source=str(path),
                prompt_id=case.prompt_id,
                prompt_sha256=digest,
                prompt_bytes=size,
                expected_tokens=case.expected_tokens,
                bucket=bucket_name,
            )
        )
        lengths.append(size)
        hashes.append(digest)

    duplicate_count = sum(count - 1 for count in collections.Counter(hashes).values() if count > 1)
    if len(cases) < args.min_total_prompts:
        findings.append(
            Finding(str(path), "min_total_prompts", f"{len(cases)} prompts below required {args.min_total_prompts}")
        )
    if duplicate_count and args.fail_on_duplicate_prompts:
        findings.append(Finding(str(path), "duplicate_prompts", f"{duplicate_count} duplicate prompt payloads"))
    for name, minimum in args.min_bucket_prompts:
        count = bucket_counts.get(name, 0)
        if count < minimum:
            findings.append(Finding(str(path), "min_bucket_prompts", f"{name} has {count} prompts below required {minimum}"))
    if bucket_counts.get("unbucketed", 0):
        findings.append(Finding(str(path), "unbucketed_prompts", f"{bucket_counts['unbucketed']} prompts matched no bucket"))

    status = "fail" if findings else "pass"
    total_bytes = sum(lengths)
    audit = SourceAudit(
        source=str(path),
        status=status,
        prompts=len(cases),
        prompt_bytes_min=min(lengths) if lengths else None,
        prompt_bytes_max=max(lengths) if lengths else None,
        prompt_bytes_total=total_bytes,
        prompt_bytes_avg=(total_bytes / len(lengths) if lengths else None),
        duplicate_prompt_sha256=duplicate_count,
        bucket_counts={key: bucket_counts[key] for key in sorted(bucket_counts)},
        suite_sha256=qemu_prompt_bench.prompt_suite_hash(cases),
    )
    return audit, rows, findings


def iter_sources(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            for pattern in ("*.jsonl", "*.json", "*.txt"):
                yield from sorted(child for child in path.rglob(pattern) if child.is_file())
        elif path.is_file():
            yield path


def status_for(audits: list[SourceAudit], findings: list[Finding], min_sources: int) -> str:
    if len(audits) < min_sources:
        return "fail"
    if findings:
        return "fail"
    if any(audit.status == "fail" for audit in audits):
        return "fail"
    return "pass"


def build_report(audits: list[SourceAudit], rows: list[PromptRow], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    all_lengths = [row.prompt_bytes for row in rows]
    status = status_for(audits, findings, args.min_sources)
    if len(audits) < args.min_sources:
        findings = findings + [Finding("", "min_sources", f"{len(audits)} sources below required {args.min_sources}")]
    bucket_totals: dict[str, int] = collections.Counter()
    for audit in audits:
        bucket_totals.update(audit.bucket_counts)
    return {
        "tool": "prompt_length_audit",
        "timestamp": iso_now(),
        "status": status,
        "buckets": [asdict(bucket) for bucket in args.buckets],
        "gates": {
            "min_sources": args.min_sources,
            "min_total_prompts": args.min_total_prompts,
            "min_bucket_prompts": dict(args.min_bucket_prompts),
            "fail_on_duplicate_prompts": args.fail_on_duplicate_prompts,
        },
        "summary": {
            "sources": len(audits),
            "prompts": len(rows),
            "prompt_bytes_min": min(all_lengths) if all_lengths else None,
            "prompt_bytes_max": max(all_lengths) if all_lengths else None,
            "prompt_bytes_total": sum(all_lengths),
            "bucket_counts": {key: bucket_totals[key] for key in sorted(bucket_totals)},
            "findings": len(findings),
        },
        "sources": [asdict(audit) for audit in audits],
        "prompts": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# Prompt Length Audit",
        "",
        f"Status: {report['status']}",
        f"Sources: {summary['sources']}",
        f"Prompts: {summary['prompts']}",
        f"Prompt bytes: {summary['prompt_bytes_min']}..{summary['prompt_bytes_max']}",
        "",
        "## Buckets",
        "",
        "| Bucket | Prompts |",
        "| --- | ---: |",
    ]
    for name, count in summary["bucket_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Sources", "", "| Source | Status | Prompts | Duplicates | Suite SHA256 |", "| --- | --- | ---: | ---: | --- |"])
    for source in report["sources"]:
        lines.append(
            f"| {source['source']} | {source['status']} | {source['prompts']} | "
            f"{source['duplicate_prompt_sha256']} | {source['suite_sha256']} |"
        )
    if report["findings"]:
        lines.extend(["", "## Findings", "", "| Source | Kind | Detail |", "| --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['source']} | {finding['kind']} | {finding['detail']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "source",
        "status",
        "prompts",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "prompt_bytes_total",
        "prompt_bytes_avg",
        "duplicate_prompt_sha256",
        "suite_sha256",
        "bucket_counts_json",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for source in report["sources"]:
            row = {key: source.get(key) for key in fields if key != "bucket_counts_json"}
            row["bucket_counts_json"] = json.dumps(source["bucket_counts"], sort_keys=True)
            writer.writerow(row)


def write_prompt_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["source", "prompt_id", "prompt_sha256", "prompt_bytes", "expected_tokens", "bucket"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["prompts"])


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    fields = ["source", "kind", "detail"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["findings"])


def write_junit(report: dict[str, Any], path: Path) -> None:
    tests = max(1, len(report["sources"]))
    failures = len(report["findings"]) if report["status"] == "fail" else 0
    suite = ET.Element("testsuite", name="holyc_prompt_length_audit", tests=str(tests), failures=str(failures))
    if report["sources"]:
        for source in report["sources"]:
            case = ET.SubElement(suite, "testcase", name=source["source"], classname="prompt_length_audit")
            source_findings = [finding for finding in report["findings"] if finding["source"] == source["source"]]
            for finding in source_findings:
                failure = ET.SubElement(case, "failure", message=finding["kind"], type=finding["kind"])
                failure.text = finding["detail"]
    else:
        case = ET.SubElement(suite, "testcase", name="sources", classname="prompt_length_audit")
        for finding in report["findings"]:
            failure = ET.SubElement(case, "failure", message=finding["kind"], type=finding["kind"])
            failure.text = finding["detail"]
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Prompt suite files or directories")
    parser.add_argument("--bucket", action="append", type=parse_bucket, dest="buckets", help="Length bucket NAME:MIN:MAX")
    parser.add_argument("--min-sources", type=int, default=1)
    parser.add_argument("--min-total-prompts", type=int, default=1)
    parser.add_argument("--min-bucket-prompts", action="append", type=parse_min_bucket, default=[])
    parser.add_argument("--fail-on-duplicate-prompts", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="prompt_length_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.buckets = args.buckets or list(DEFAULT_BUCKETS)
    if args.min_sources < 0 or args.min_total_prompts < 0:
        parser.error("--min-sources and --min-total-prompts cannot be negative")

    audits: list[SourceAudit] = []
    rows: list[PromptRow] = []
    findings: list[Finding] = []
    for source in iter_sources(args.inputs):
        audit, source_rows, source_findings = audit_source(source, args.buckets, args)
        audits.append(audit)
        rows.extend(source_rows)
        findings.extend(source_findings)

    report = build_report(audits, rows, findings, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(report, args.output_dir / f"{stem}.json")
    write_markdown(report, args.output_dir / f"{stem}.md")
    write_csv(report, args.output_dir / f"{stem}.csv")
    write_prompt_csv(report, args.output_dir / f"{stem}_prompts.csv")
    write_findings_csv(report, args.output_dir / f"{stem}_findings.csv")
    write_junit(report, args.output_dir / f"{stem}_junit.xml")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
