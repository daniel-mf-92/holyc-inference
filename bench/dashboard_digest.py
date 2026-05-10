#!/usr/bin/env python3
"""Summarize host-side dashboard artifacts for CI.

This tool consumes existing dashboard JSON artifacts and writes a compact
status digest. It is host-side only and never launches QEMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DashboardRecord:
    source: str
    name: str
    status: str
    generated_at: str
    findings: int
    summary_keys: str
    error: str = ""


@dataclass(frozen=True)
class DigestFinding:
    gate: str
    source: str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return 0
    return 0


def dashboard_name(path: Path, payload: dict[str, Any] | None = None) -> str:
    if payload:
        raw_name = payload.get("name") or payload.get("tool") or payload.get("report")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    name = path.name
    for suffix in ("_latest.json", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def summary_findings(payload: dict[str, Any]) -> int:
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in ("findings", "failures", "violations", "regressions"):
            if key in summary:
                return as_int(summary[key])
    findings = payload.get("findings")
    if isinstance(findings, list):
        return len(findings)
    if isinstance(findings, int):
        return findings
    return 0


def summary_keys(payload: dict[str, Any]) -> str:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return ""
    return ",".join(sorted(str(key) for key in summary))


def normalize_status(payload: dict[str, Any], findings: int) -> str:
    raw_status = payload.get("status")
    if isinstance(raw_status, str) and raw_status.strip():
        status = raw_status.strip().lower()
        if status in {"pass", "ok", "success"}:
            return "pass"
        if status in {"fail", "failed", "error"}:
            return "fail"
        return status
    return "fail" if findings else "pass"


def load_dashboard(path: Path) -> DashboardRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return DashboardRecord(
            source=str(path),
            name=dashboard_name(path),
            status="missing",
            generated_at="",
            findings=0,
            summary_keys="",
            error=str(exc),
        )
    except json.JSONDecodeError as exc:
        return DashboardRecord(
            source=str(path),
            name=dashboard_name(path),
            status="invalid",
            generated_at="",
            findings=0,
            summary_keys="",
            error=f"invalid json: {exc}",
        )
    if not isinstance(payload, dict):
        return DashboardRecord(
            source=str(path),
            name=dashboard_name(path),
            status="invalid",
            generated_at="",
            findings=0,
            summary_keys="",
            error="dashboard root must be a JSON object",
        )
    findings = summary_findings(payload)
    generated_at = payload.get("generated_at") or payload.get("created_at") or ""
    return DashboardRecord(
        source=str(path),
        name=dashboard_name(path, payload),
        status=normalize_status(payload, findings),
        generated_at=str(generated_at),
        findings=findings,
        summary_keys=summary_keys(payload),
    )


def evaluate(
    records: list[DashboardRecord],
    *,
    fail_on_missing: bool,
    fail_on_fail_status: bool,
    min_dashboards: int,
) -> list[DigestFinding]:
    findings: list[DigestFinding] = []
    if len(records) < min_dashboards:
        findings.append(
            DigestFinding(
                gate="min_dashboards",
                source="",
                message=f"found {len(records)} dashboard(s), below minimum {min_dashboards}",
            )
        )
    for record in records:
        if record.status in {"missing", "invalid"}:
            if fail_on_missing:
                findings.append(
                    DigestFinding(
                        gate=record.status,
                        source=record.source,
                        message=record.error or f"{record.source} is {record.status}",
                    )
                )
            continue
        if fail_on_fail_status and record.status != "pass":
            findings.append(
                DigestFinding(
                    gate="dashboard_status",
                    source=record.source,
                    message=f"{record.name} status is {record.status}",
                )
            )
    return findings


def build_report(records: list[DashboardRecord], findings: list[DigestFinding]) -> dict[str, Any]:
    status = "fail" if findings else "pass"
    status_counts: dict[str, int] = {}
    for record in records:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    return {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "dashboards": len(records),
            "findings": len(findings),
            "status_counts": status_counts,
            "total_dashboard_findings": sum(record.findings for record in records),
        },
        "dashboards": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[DashboardRecord]) -> None:
    fieldnames = list(DashboardRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Dashboard Digest",
        "",
        f"- Status: {report['status']}",
        f"- Dashboards: {summary['dashboards']}",
        f"- Findings: {summary['findings']}",
        f"- Total dashboard findings: {summary['total_dashboard_findings']}",
        "",
        "| Dashboard | Status | Findings | Source |",
        "| --- | --- | ---: | --- |",
    ]
    for record in report["dashboards"]:
        lines.append(
            f"| {record['name']} | {record['status']} | {record['findings']} | {record['source']} |"
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['gate']}: {finding['message']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[DigestFinding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dashboard_digest",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {"name": f"{finding.gate}:{finding.source or 'digest'}"},
            )
            failure = ET.SubElement(testcase, "failure", {"message": finding.message})
            failure.text = finding.message
    else:
        ET.SubElement(testsuite, "testcase", {"name": "dashboard_digest"})
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dashboards", nargs="+", type=Path, help="Dashboard JSON artifacts to summarize.")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-stem", default="dashboard_digest_latest")
    parser.add_argument("--min-dashboards", type=int, default=1)
    parser.add_argument("--fail-on-missing", action="store_true")
    parser.add_argument("--fail-on-fail-status", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [load_dashboard(path) for path in args.dashboards]
    findings = evaluate(
        records,
        fail_on_missing=args.fail_on_missing,
        fail_on_fail_status=args.fail_on_fail_status,
        min_dashboards=args.min_dashboards,
    )
    report = build_report(records, findings)

    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
