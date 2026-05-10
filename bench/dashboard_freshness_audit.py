#!/usr/bin/env python3
"""Audit saved dashboard JSON artifacts for timestamp freshness.

This host-side tool reads existing dashboard artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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


@dataclass(frozen=True)
class FreshnessRecord:
    source: str
    generated_at: str
    age_hours: float | None
    status: str
    error: str = ""


@dataclass(frozen=True)
class FreshnessFinding:
    source: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def iter_json_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*_latest.json") if child.is_file())
        elif path.is_file() and path.suffix.lower() == ".json":
            yield path


def load_generated_at(path: Path) -> tuple[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return "", str(exc)
    except json.JSONDecodeError as exc:
        return "", f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return "", "dashboard root must be a JSON object"
    value = payload.get("generated_at") or payload.get("created_at") or payload.get("timestamp") or ""
    return str(value), ""


def audit_dashboard(path: Path, now: datetime, max_age_hours: float, future_skew_minutes: float) -> tuple[FreshnessRecord, list[FreshnessFinding]]:
    generated_at, error = load_generated_at(path)
    findings: list[FreshnessFinding] = []
    if error:
        record = FreshnessRecord(str(path), generated_at, None, "fail", error)
        findings.append(FreshnessFinding(str(path), "error", "load_error", error))
        return record, findings

    parsed = parse_timestamp(generated_at)
    if parsed is None:
        record = FreshnessRecord(str(path), generated_at, None, "fail", "missing or invalid generated_at")
        findings.append(FreshnessFinding(str(path), "error", "invalid_timestamp", "missing or invalid generated_at"))
        return record, findings

    age_hours = (now - parsed).total_seconds() / 3600.0
    status = "pass"
    if age_hours < -(future_skew_minutes / 60.0):
        status = "fail"
        findings.append(
            FreshnessFinding(
                str(path),
                "error",
                "future_timestamp",
                f"generated_at is {-age_hours:.2f} hour(s) in the future",
            )
        )
    elif age_hours > max_age_hours:
        status = "fail"
        findings.append(
            FreshnessFinding(
                str(path),
                "error",
                "stale_dashboard",
                f"generated_at age {age_hours:.2f} hour(s) exceeds limit {max_age_hours:.2f}",
            )
        )

    return FreshnessRecord(str(path), generated_at, round(age_hours, 3), status), findings


def build_report(records: list[FreshnessRecord], findings: list[FreshnessFinding], max_age_hours: float) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "dashboards": len(records),
            "findings": len(findings),
            "stale_dashboards": sum(1 for record in records if record.status != "pass"),
            "max_age_hours": max_age_hours,
        },
        "dashboards": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[FreshnessRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FreshnessRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[FreshnessFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FreshnessFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Dashboard Freshness Audit",
        "",
        f"- Status: {report['status']}",
        f"- Dashboards: {summary['dashboards']}",
        f"- Findings: {summary['findings']}",
        f"- Max age hours: {summary['max_age_hours']}",
        "",
        "| Dashboard | Status | Generated at | Age hours |",
        "| --- | --- | --- | --- |",
    ]
    for record in report["dashboards"]:
        lines.append(
            "| {source} | {status} | {generated_at} | {age_hours} |".format(**record)
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} {finding['detail']}".strip())
    else:
        lines.extend(["", "No dashboard freshness findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[FreshnessFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dashboard_freshness_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.source}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "dashboard_freshness"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Dashboard JSON files or directories to audit")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-stem", default="dashboard_freshness_audit_latest")
    parser.add_argument("--max-age-hours", type=float, default=96.0)
    parser.add_argument("--future-skew-minutes", type=float, default=5.0)
    parser.add_argument("--min-dashboards", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    now = datetime.now(timezone.utc)
    records: list[FreshnessRecord] = []
    findings: list[FreshnessFinding] = []
    for path in iter_json_files(args.inputs):
        record, record_findings = audit_dashboard(path, now, args.max_age_hours, args.future_skew_minutes)
        records.append(record)
        findings.extend(record_findings)

    if len(records) < args.min_dashboards:
        findings.append(
            FreshnessFinding(
                "",
                "error",
                "min_dashboards",
                f"found {len(records)} dashboard artifact(s), below minimum {args.min_dashboards}",
            )
        )

    report = build_report(records, findings, args.max_age_hours)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_csv(output_base.with_suffix(".csv"), records)
    write_findings_csv(output_base.with_name(f"{output_base.name}_findings.csv"), findings)
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
