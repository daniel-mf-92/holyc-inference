#!/usr/bin/env python3
"""Audit QEMU benchmark latest JSONs for generated_at freshness.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*_latest.json",)


@dataclass(frozen=True)
class FreshnessRecord:
    source: str
    status: str
    generated_at: str
    age_seconds: float | None
    max_age_seconds: float | None
    within_max_age: bool | None
    artifact_status: str
    rows: int
    error: str = ""


@dataclass(frozen=True)
class FreshnessFinding:
    source: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_now(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = parse_timestamp(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("--now must be an ISO-8601 timestamp with timezone")
    return parsed


def iter_latest_jsons(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
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


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "artifact root must be a JSON object"
    return payload, ""


def row_count(payload: dict[str, Any]) -> int:
    count = 0
    for key in ("warmups", "benchmarks"):
        value = payload.get(key)
        if isinstance(value, list):
            count += sum(1 for row in value if isinstance(row, dict))
    return count


def audit_latest(path: Path, *, now: datetime, max_age_seconds: float | None) -> FreshnessRecord:
    payload, error = load_json_object(path)
    if payload is None:
        return FreshnessRecord(str(path), "fail", "", None, max_age_seconds, None, "", 0, error)

    generated_at = str(payload.get("generated_at") or "")
    generated = parse_timestamp(generated_at)
    if generated is None:
        return FreshnessRecord(
            str(path),
            "fail",
            generated_at,
            None,
            max_age_seconds,
            None,
            str(payload.get("status") or ""),
            row_count(payload),
            "missing or invalid generated_at timestamp",
        )

    age_seconds = (now - generated).total_seconds()
    within_max_age = None if max_age_seconds is None else age_seconds <= max_age_seconds
    status = "pass" if within_max_age is not False else "fail"
    return FreshnessRecord(
        source=str(path),
        status=status,
        generated_at=generated_at,
        age_seconds=age_seconds,
        max_age_seconds=max_age_seconds,
        within_max_age=within_max_age,
        artifact_status=str(payload.get("status") or ""),
        rows=row_count(payload),
    )


def evaluate(records: list[FreshnessRecord], min_latest: int) -> list[FreshnessFinding]:
    findings: list[FreshnessFinding] = []
    if len(records) < min_latest:
        findings.append(
            FreshnessFinding(
                "",
                "min_latest",
                f"found {len(records)} latest QEMU benchmark artifact(s), below minimum {min_latest}",
            )
        )
    for record in records:
        if record.error:
            findings.append(FreshnessFinding(record.source, "invalid_generated_at", record.error))
        if record.age_seconds is not None and record.age_seconds < 0:
            findings.append(
                FreshnessFinding(
                    record.source,
                    "future_generated_at",
                    f"generated_at is {-record.age_seconds:.3f}s after audit time",
                )
            )
        if record.within_max_age is False:
            findings.append(
                FreshnessFinding(
                    record.source,
                    "stale_artifact",
                    f"age {record.age_seconds:.3f}s exceeds max {record.max_age_seconds:.3f}s",
                )
            )
    return findings


def build_report(records: list[FreshnessRecord], findings: list[FreshnessFinding], args: argparse.Namespace) -> dict[str, Any]:
    ages = [record.age_seconds for record in records if record.age_seconds is not None]
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "latest_artifacts": len(records),
            "findings": len(findings),
            "fresh_artifacts": sum(1 for record in records if record.within_max_age is True),
            "stale_artifacts": sum(1 for record in records if record.within_max_age is False),
            "invalid_generated_at": sum(1 for record in records if record.error),
            "max_age_seconds": args.max_age_seconds,
            "oldest_age_seconds": max(ages) if ages else None,
            "newest_age_seconds": min(ages) if ages else None,
        },
        "config": {
            "patterns": args.pattern,
            "min_latest": args.min_latest,
            "now": args.now,
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[FreshnessRecord]) -> None:
    fieldnames = list(FreshnessRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[FreshnessFinding]) -> None:
    fieldnames = list(FreshnessFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Result Freshness Audit",
        "",
        f"- Status: {report['status']}",
        f"- Latest artifacts: {summary['latest_artifacts']}",
        f"- Findings: {summary['findings']}",
        f"- Stale artifacts: {summary['stale_artifacts']}",
        f"- Oldest age seconds: {summary['oldest_age_seconds']}",
        "",
        "| Source | Status | Generated at | Age seconds | Rows |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for record in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {generated_at} | {age_seconds} | {rows} |".format(**record)
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['detail']} ({finding['source']})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    failures = len(report["findings"])
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_result_freshness_audit",
            "tests": "1",
            "failures": str(1 if failures else 0),
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "freshness"})
    if failures:
        failure = ET.SubElement(case, "failure", {"message": f"{failures} freshness finding(s)"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in report["findings"])
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("bench/results")])
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_result_freshness_audit_latest")
    parser.add_argument("--min-latest", type=int, default=1)
    parser.add_argument("--max-age-hours", type=float)
    parser.add_argument("--now", help="Override current time for deterministic tests")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.max_age_seconds = None if args.max_age_hours is None else args.max_age_hours * 3600.0
    now = parse_now(args.now)
    args.now = now.isoformat(timespec="seconds").replace("+00:00", "Z")

    records = [
        audit_latest(path, now=now, max_age_seconds=args.max_age_seconds)
        for path in iter_latest_jsons(args.paths, args.pattern)
    ]
    findings = evaluate(records, args.min_latest)
    report = build_report(records, findings, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", report)
    print(f"status={report['status']}")
    print(f"latest_artifacts={report['summary']['latest_artifacts']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
