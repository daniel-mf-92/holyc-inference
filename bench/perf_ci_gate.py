#!/usr/bin/env python3
"""CI gate for saved performance dashboard artifacts.

This host-side tool reads existing dashboard JSON/Markdown/CSV/JUnit artifacts
only. It never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DASHBOARDS = (
    "perf_regression_latest.json",
    "perf_slo_audit_latest.json",
)


@dataclass(frozen=True)
class GateRecord:
    source: str
    name: str
    status: str
    generated_at: str
    summary_rows: int | None
    summary_findings: int | None
    sidecars_checked: int
    sidecars_missing: int


@dataclass(frozen=True)
class GateFinding:
    source: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def dashboard_name(path: Path, payload: dict[str, Any] | None = None) -> str:
    if payload:
        raw = payload.get("name") or payload.get("tool") or payload.get("report")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    name = path.name
    if name.endswith("_latest.json"):
        return name[: -len("_latest.json")]
    return path.stem


def first_int(*values: Any) -> int | None:
    for value in values:
        result = as_int(value)
        if result is not None:
            return result
    return None


def sidecar_candidates(path: Path) -> list[Path]:
    stem = path.with_suffix("")
    candidates = [stem.with_suffix(".md")]
    csv_candidates = [stem.with_suffix(".csv")]
    if stem.name.endswith("_latest"):
        base = stem.name[: -len("_latest")]
        csv_candidates.extend(sorted(stem.parent.glob(f"{base}_*_latest.csv")))
    candidates.append(next((candidate for candidate in csv_candidates if candidate.exists()), csv_candidates[0]))
    junit_candidates = [stem.with_name(f"{stem.name}_junit.xml")]
    if stem.name.endswith("_latest"):
        base = stem.name[: -len("_latest")]
        junit_candidates.append(stem.with_name(f"{base}_junit_latest.xml"))
    candidates.append(next((candidate for candidate in junit_candidates if candidate.exists()), junit_candidates[0]))
    return candidates


def load_dashboard(path: Path, require_sidecars: bool) -> tuple[GateRecord, list[GateFinding]]:
    findings: list[GateFinding] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return (
            GateRecord(str(path), dashboard_name(path), "missing", "", None, None, 0, 0),
            [GateFinding(str(path), "error", "missing_json", str(exc))],
        )
    except json.JSONDecodeError as exc:
        return (
            GateRecord(str(path), dashboard_name(path), "invalid", "", None, None, 0, 0),
            [GateFinding(str(path), "error", "invalid_json", str(exc))],
        )

    if not isinstance(payload, dict):
        return (
            GateRecord(str(path), dashboard_name(path), "invalid", "", None, None, 0, 0),
            [GateFinding(str(path), "error", "invalid_json", "dashboard root must be an object")],
        )

    raw_status = str(payload.get("status") or "").strip().lower()
    status = raw_status or "unknown"
    if status != "pass":
        findings.append(GateFinding(str(path), "error", "dashboard_status", f"status is {status}"))

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = first_int(summary.get("rows"), summary.get("dashboards"))
    findings_count = as_int(summary.get("findings"))
    if rows is not None and rows <= 0:
        findings.append(GateFinding(str(path), "error", "empty_dashboard", "summary row count is zero"))
    if findings_count is not None and findings_count > 0:
        findings.append(
            GateFinding(str(path), "error", "dashboard_findings", f"summary reports {findings_count} finding(s)")
        )

    sidecars = sidecar_candidates(path) if require_sidecars else []
    missing_sidecars = [candidate for candidate in sidecars if not candidate.exists()]
    for missing in missing_sidecars:
        findings.append(GateFinding(str(path), "error", "missing_sidecar", str(missing)))

    generated_at = str(payload.get("generated_at") or payload.get("created_at") or "")
    return (
        GateRecord(
            source=str(path),
            name=dashboard_name(path, payload),
            status=status,
            generated_at=generated_at,
            summary_rows=rows,
            summary_findings=findings_count,
            sidecars_checked=len(sidecars),
            sidecars_missing=len(missing_sidecars),
        ),
        findings,
    )


def build_report(records: list[GateRecord], findings: list[GateFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "dashboards": len(records),
            "findings": len(findings),
            "sidecars_checked": sum(record.sidecars_checked for record in records),
            "sidecars_missing": sum(record.sidecars_missing for record in records),
        },
        "dashboards": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[GateRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GateRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[GateFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GateFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Perf CI Gate",
        "",
        f"- Status: {report['status']}",
        f"- Dashboards: {summary['dashboards']}",
        f"- Findings: {summary['findings']}",
        f"- Sidecars missing: {summary['sidecars_missing']}/{summary['sidecars_checked']}",
        "",
        "## Dashboards",
        "",
        "| Dashboard | Status | Rows | Findings | Missing sidecars |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for record in report["dashboards"]:
        lines.append(
            "| {name} | {status} | {rows} | {findings} | {missing} |".format(
                name=record["name"],
                status=record["status"],
                rows="" if record["summary_rows"] is None else record["summary_rows"],
                findings="" if record["summary_findings"] is None else record["summary_findings"],
                missing=record["sidecars_missing"],
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} - {finding['detail']}")
    else:
        lines.extend(["", "No perf CI gate findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, records: list[GateRecord], findings: list[GateFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_perf_ci_gate",
            "tests": str(max(1, len(records))),
            "failures": str(len(findings)),
        },
    )
    if records:
        for record in records:
            case = ET.SubElement(suite, "testcase", {"classname": "perf_ci_gate", "name": record.name})
            for finding in findings:
                if finding.source == record.source:
                    failure = ET.SubElement(case, "failure", {"message": finding.kind})
                    failure.text = finding.detail
    else:
        case = ET.SubElement(suite, "testcase", {"classname": "perf_ci_gate", "name": "dashboards"})
        for finding in findings:
            failure = ET.SubElement(case, "failure", {"message": finding.kind})
            failure.text = finding.detail
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dashboard",
        nargs="*",
        type=Path,
        help="Dashboard JSON files to gate. Defaults to perf regression and perf SLO latest dashboards.",
    )
    parser.add_argument("--dashboard-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/dashboards"))
    parser.add_argument("--output-stem", default="perf_ci_gate_latest")
    parser.add_argument("--no-sidecars", action="store_true", help="Do not require Markdown/CSV/JUnit sidecars.")
    return parser


def default_dashboards(dashboard_dir: Path) -> list[Path]:
    return [dashboard_dir / name for name in DEFAULT_DASHBOARDS]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dashboards = args.dashboard or default_dashboards(args.dashboard_dir)
    records: list[GateRecord] = []
    findings: list[GateFinding] = []
    for path in dashboards:
        record, record_findings = load_dashboard(path, require_sidecars=not args.no_sidecars)
        records.append(record)
        findings.extend(record_findings)

    report = build_report(records, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_stem
    write_json(output.with_suffix(".json"), report)
    write_markdown(output.with_suffix(".md"), report)
    write_csv(output.with_suffix(".csv"), records)
    write_findings_csv(output.with_name(f"{output.name}_findings.csv"), findings)
    write_junit(output.with_name(f"{output.name}_junit.xml"), records, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
