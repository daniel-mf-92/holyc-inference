#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for top-level status consistency.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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
ROW_KEYS = ("warmups", "benchmarks")
SUMMARY_KEYS = {
    "runs": "rows",
    "ok_runs": "ok_rows",
    "failed_runs": "failed_rows",
    "timed_out_runs": "timed_out_rows",
    "nonzero_exit_runs": "nonzero_exit_rows",
}


@dataclass(frozen=True)
class StatusRecord:
    source: str
    status: str
    expected_status: str
    rows: int
    ok_rows: int
    failed_rows: int
    timed_out_rows: int
    nonzero_exit_rows: int
    telemetry_findings: int
    variability_findings: int
    command_airgap_ok: bool | None
    suite_runs: int | None
    suite_ok_runs: int | None
    suite_failed_runs: int | None
    suite_timed_out_runs: int | None
    suite_nonzero_exit_runs: int | None


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def benchmark_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ROW_KEYS:
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        rows.extend(row for row in value if isinstance(row, dict))
    return rows


def row_is_ok(row: dict[str, Any]) -> bool:
    exit_class = str(row.get("exit_class") or "").lower()
    timed_out = parse_bool(row.get("timed_out"))
    returncode = finite_int(row.get("returncode"))
    if exit_class and exit_class != "ok":
        return False
    if timed_out is True:
        return False
    if returncode not in (None, 0):
        return False
    return True


def command_airgap_ok(payload: dict[str, Any]) -> bool | None:
    metadata = payload.get("command_airgap")
    if not isinstance(metadata, dict) or "ok" not in metadata:
        return None
    return metadata.get("ok") is True


def suite_value(payload: dict[str, Any], key: str) -> int | None:
    summary = payload.get("suite_summary")
    if not isinstance(summary, dict):
        return None
    return finite_int(summary.get(key))


def status_signal(record: StatusRecord, args: argparse.Namespace) -> bool:
    if record.failed_rows > 0:
        return True
    if record.telemetry_findings > 0 or record.variability_findings > 0:
        return True
    if args.require_command_airgap and record.command_airgap_ok is not True:
        return True
    if args.require_rows and record.rows == 0:
        return True
    return False


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[StatusRecord | None, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return None, [Finding(str(path), "error", "load_error", "artifact", error)]

    findings: list[Finding] = []
    status = str(payload.get("status") or "")
    rows = benchmark_rows(payload)
    ok_rows = sum(1 for row in rows if row_is_ok(row))
    failed_rows = len(rows) - ok_rows
    timed_out_rows = sum(1 for row in rows if parse_bool(row.get("timed_out")) is True)
    nonzero_exit_rows = sum(1 for row in rows if (finite_int(row.get("returncode")) or 0) != 0)
    telemetry_findings = list_count(payload.get("telemetry_findings"))
    variability_findings = list_count(payload.get("variability_findings"))
    airgap_ok = command_airgap_ok(payload)
    expected_status = "fail" if (
        failed_rows
        or telemetry_findings
        or variability_findings
        or (args.require_command_airgap and airgap_ok is not True)
        or (args.require_rows and not rows)
    ) else "pass"

    record = StatusRecord(
        source=str(path),
        status=status,
        expected_status=expected_status,
        rows=len(rows),
        ok_rows=ok_rows,
        failed_rows=failed_rows,
        timed_out_rows=timed_out_rows,
        nonzero_exit_rows=nonzero_exit_rows,
        telemetry_findings=telemetry_findings,
        variability_findings=variability_findings,
        command_airgap_ok=airgap_ok,
        suite_runs=suite_value(payload, "runs"),
        suite_ok_runs=suite_value(payload, "ok_runs"),
        suite_failed_runs=suite_value(payload, "failed_runs"),
        suite_timed_out_runs=suite_value(payload, "timed_out_runs"),
        suite_nonzero_exit_runs=suite_value(payload, "nonzero_exit_runs"),
    )

    allowed_statuses = {"pass", "fail"}
    if args.allow_planned:
        allowed_statuses.add("planned")
    if status not in allowed_statuses:
        findings.append(Finding(str(path), "error", "invalid_status", "status", f"unexpected status {status!r}"))
    if status == "planned" and (rows or not args.allow_planned):
        findings.append(Finding(str(path), "error", "planned_with_rows", "status", "planned artifacts must not contain benchmark rows"))

    if args.require_rows and not rows:
        findings.append(Finding(str(path), "error", "missing_rows", "benchmarks", "no warmup or benchmark rows found"))
    if status == "pass" and failed_rows:
        findings.append(Finding(str(path), "error", "pass_with_failed_rows", "status", f"{failed_rows} row(s) are not OK"))
    if status == "pass" and telemetry_findings:
        findings.append(Finding(str(path), "error", "pass_with_telemetry_findings", "telemetry_findings", f"{telemetry_findings} finding(s) present"))
    if status == "pass" and variability_findings:
        findings.append(Finding(str(path), "error", "pass_with_variability_findings", "variability_findings", f"{variability_findings} finding(s) present"))
    if args.require_command_airgap and status == "pass" and airgap_ok is not True:
        findings.append(Finding(str(path), "error", "pass_without_airgap_ok", "command_airgap.ok", "passing artifact lacks explicit air-gap OK metadata"))
    if status == "fail" and not status_signal(record, args):
        findings.append(Finding(str(path), "error", "fail_without_failure_signal", "status", "failing artifact has no row, telemetry, variability, or air-gap failure signal"))
    if status in {"pass", "fail"} and status != expected_status:
        findings.append(Finding(str(path), "error", "status_mismatch", "status", f"status={status!r}, expected {expected_status!r}"))

    measured_rows = [row for row in payload.get("benchmarks", []) if isinstance(row, dict)] if isinstance(payload.get("benchmarks"), list) else []
    measured_ok = sum(1 for row in measured_rows if row_is_ok(row))
    measured_failed = len(measured_rows) - measured_ok
    measured_timed_out = sum(1 for row in measured_rows if parse_bool(row.get("timed_out")) is True)
    measured_nonzero = sum(1 for row in measured_rows if (finite_int(row.get("returncode")) or 0) != 0)
    expected_summary = {
        "runs": len(measured_rows),
        "ok_runs": measured_ok,
        "failed_runs": measured_failed,
        "timed_out_runs": measured_timed_out,
        "nonzero_exit_runs": measured_nonzero,
    }
    for suite_key, record_key in SUMMARY_KEYS.items():
        stored = getattr(record, f"suite_{suite_key}")
        expected = expected_summary[suite_key]
        if stored is None:
            findings.append(Finding(str(path), "error", f"missing_suite_{suite_key}", f"suite_summary.{suite_key}", "suite counter is absent"))
        elif stored != expected:
            findings.append(
                Finding(
                    str(path),
                    "error",
                    f"suite_{suite_key}_mismatch",
                    f"suite_summary.{suite_key}",
                    f"stored {stored}, expected {expected}",
                )
            )

    return record, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[StatusRecord], list[Finding]]:
    records: list[StatusRecord] = []
    findings: list[Finding] = []
    for path in sorted(iter_input_files(paths, args.pattern)):
        record, artifact_findings = audit_artifact(path, args)
        if record is not None:
            records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(Finding("-", "error", "min_artifacts", "artifacts", f"found {len(records)}, expected at least {args.min_artifacts}"))
    return records, findings


def build_report(records: list[StatusRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "findings": len(findings),
            "rows": sum(record.rows for record in records),
            "failed_rows": sum(record.failed_rows for record in records),
            "telemetry_findings": sum(record.telemetry_findings for record in records),
            "variability_findings": sum(record.variability_findings for record in records),
            "airgap_missing_or_failed": sum(1 for record in records if record.command_airgap_ok is not True),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[StatusRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(StatusRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# QEMU Status Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Artifacts: {report['summary']['artifacts']}",
        f"Rows: {report['summary']['rows']}",
        f"Findings: {report['summary']['findings']}",
        "",
        "## Artifacts",
        "",
        "| Source | Status | Expected | Rows | OK | Failed | Telemetry findings | Variability findings | Air-gap OK |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {expected_status} | {rows} | {ok_rows} | {failed_rows} | {telemetry_findings} | {variability_findings} | {command_airgap_ok} |".format(
                **record
            )
        )
    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(["| Source | Kind | Field | Detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append("| {source} | {kind} | {field} | {detail} |".format(**finding))
    else:
        lines.append("No status consistency findings.")
    return "\n".join(lines) + "\n"


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_status_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "status_consistency"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} status finding(s)"})
        failure.text = "\n".join(f"{finding.source}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON artifact files or directories")
    parser.add_argument("--pattern", action="append", default=None, help="Glob pattern when scanning directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_status_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--no-require-rows", dest="require_rows", action="store_false", default=True)
    parser.add_argument("--no-require-command-airgap", dest="require_command_airgap", action="store_false", default=True)
    parser.add_argument("--allow-planned", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.pattern = args.pattern or list(DEFAULT_PATTERNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, findings = audit(args.inputs, args)
    report = build_report(records, findings)

    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    (args.output_dir / f"{stem}.md").write_text(markdown_report(report), encoding="utf-8")
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
