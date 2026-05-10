#!/usr/bin/env python3
"""Audit dashboard JSON artifacts for expected CI sidecars.

This host-side tool reads existing dashboard artifacts only. It never launches
QEMU or touches the TempleOS guest.
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
from typing import Any, Iterable


SIDECARS = ("csv", "markdown", "junit")


@dataclass(frozen=True)
class DashboardSidecarRecord:
    source: str
    json_valid: bool
    csv_path: str
    csv_present: bool
    markdown_path: str
    markdown_present: bool
    junit_path: str
    junit_present: bool
    status: str
    error: str = ""


@dataclass(frozen=True)
class SidecarFinding:
    source: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_json_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*.json") if child.is_file())
        elif path.suffix.lower() == ".json":
            yield path


def junit_sidecar_candidates(path: Path) -> list[Path]:
    candidates = [path.with_name(f"{path.stem}_junit.xml")]
    if path.stem.endswith("_latest"):
        prefix = path.stem[: -len("_latest")]
        candidates.append(path.with_name(f"{prefix}_junit_latest.xml"))
    return candidates


def csv_sidecar_candidates(path: Path) -> list[Path]:
    candidates = [path.with_suffix(".csv")]
    if path.stem.endswith("_latest"):
        prefix = path.stem[: -len("_latest")]
        candidates.extend(sorted(path.parent.glob(f"{prefix}_*_latest.csv")))
    return candidates


def first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_json_status(path: Path) -> tuple[bool, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return False, str(exc)
    except json.JSONDecodeError as exc:
        return False, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return False, "dashboard root must be a JSON object"
    return True, ""


def audit_dashboard(path: Path) -> DashboardSidecarRecord:
    json_valid, error = load_json_status(path)
    csv_path = first_existing(csv_sidecar_candidates(path))
    markdown_path = path.with_suffix(".md")
    junit_path = first_existing(junit_sidecar_candidates(path))
    csv_present = csv_path.exists()
    markdown_present = markdown_path.exists()
    junit_present = junit_path.exists()
    status = "pass" if json_valid and csv_present and markdown_present and junit_present else "fail"
    return DashboardSidecarRecord(
        source=str(path),
        json_valid=json_valid,
        csv_path=str(csv_path),
        csv_present=csv_present,
        markdown_path=str(markdown_path),
        markdown_present=markdown_present,
        junit_path=str(junit_path),
        junit_present=junit_present,
        status=status,
        error=error,
    )


def evaluate(records: list[DashboardSidecarRecord], min_dashboards: int) -> list[SidecarFinding]:
    findings: list[SidecarFinding] = []
    if len(records) < min_dashboards:
        findings.append(
            SidecarFinding(
                source="",
                kind="min_dashboards",
                detail=f"found {len(records)} dashboard JSON artifact(s), below minimum {min_dashboards}",
            )
        )
    for record in records:
        if not record.json_valid:
            findings.append(SidecarFinding(record.source, "invalid_json", record.error))
        missing = []
        if not record.csv_present:
            missing.append(record.csv_path)
        if not record.markdown_present:
            missing.append(record.markdown_path)
        if not record.junit_present:
            missing.append(record.junit_path)
        if missing:
            findings.append(
                SidecarFinding(record.source, "missing_sidecars", "missing: " + ", ".join(missing))
            )
    return findings


def build_report(records: list[DashboardSidecarRecord], findings: list[SidecarFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "dashboards": len(records),
            "findings": len(findings),
            "missing_sidecar_dashboards": sum(1 for record in records if record.status != "pass"),
            "invalid_json": sum(1 for record in records if not record.json_valid),
        },
        "dashboards": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[DashboardSidecarRecord]) -> None:
    fieldnames = list(DashboardSidecarRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Dashboard Sidecar Audit",
        "",
        f"- Status: {report['status']}",
        f"- Dashboards: {summary['dashboards']}",
        f"- Findings: {summary['findings']}",
        f"- Missing sidecar dashboards: {summary['missing_sidecar_dashboards']}",
        "",
        "| Dashboard | Status | CSV | Markdown | JUnit |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in report["dashboards"]:
        lines.append(
            "| {source} | {status} | {csv_present} | {markdown_present} | {junit_present} |".format(
                **record
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} {finding['detail']}".strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[SidecarFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dashboard_sidecar_audit",
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
        ET.SubElement(suite, "testcase", {"name": "dashboard_sidecars"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Dashboard JSON files or directories to audit")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-stem", default="dashboard_sidecar_audit_latest")
    parser.add_argument("--min-dashboards", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [audit_dashboard(path) for path in sorted(iter_json_files(args.inputs))]
    findings = evaluate(records, args.min_dashboards)
    report = build_report(records, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
