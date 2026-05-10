#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for bounded output sizes.

This host-side tool reads saved benchmark artifacts only. It never launches QEMU
and never touches the TempleOS guest.
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


RESULT_KEYS = ("benchmarks", "results", "runs", "rows")
DEFAULT_PATTERNS = ("qemu_prompt_bench*.json", "qemu_prompt_bench*.jsonl", "qemu_prompt_bench*.csv")


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    suffix: str
    size_bytes: int
    row_count: int
    max_serial_output_bytes: int | None
    max_stdout_tail_bytes: int
    max_stderr_tail_bytes: int
    max_failure_reason_bytes: int


@dataclass(frozen=True)
class BudgetFinding:
    source: str
    row: int
    severity: str
    kind: str
    metric: str
    value: int
    limit: int
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text_bytes(value: Any) -> int:
    if value is None:
        return 0
    return len(str(value).encode("utf-8"))


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

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS and key != "warmups"}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            yielded = True
            for item in rows:
                if isinstance(item, dict):
                    merged = dict(inherited)
                    merged.update(item)
                    yield merged
    if not yielded:
        yield payload


def load_rows(path: Path) -> list[dict[str, Any]]:
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


def add_limit_finding(
    findings: list[BudgetFinding],
    source: Path,
    row: int,
    kind: str,
    metric: str,
    value: int,
    limit: int,
    detail: str,
) -> None:
    if value > limit:
        findings.append(BudgetFinding(str(source), row, "error", kind, metric, value, limit, detail))


def audit_file(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord, list[BudgetFinding]]:
    findings: list[BudgetFinding] = []
    size_bytes = path.stat().st_size
    add_limit_finding(
        findings,
        path,
        0,
        "file_size_exceeded",
        "size_bytes",
        size_bytes,
        args.max_file_bytes,
        "benchmark artifact file exceeds configured byte budget",
    )

    try:
        rows = load_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(BudgetFinding(str(path), 0, "error", "load_error", "rows", 0, 0, str(exc)))
        return ArtifactRecord(str(path), "fail", path.suffix.lower(), size_bytes, 0, None, 0, 0, 0), findings

    max_serial: int | None = None
    max_stdout_tail = 0
    max_stderr_tail = 0
    max_failure_reason = 0
    for index, row in enumerate(rows, 1):
        serial_bytes = finite_int(row.get("serial_output_bytes"))
        if serial_bytes is not None:
            max_serial = serial_bytes if max_serial is None else max(max_serial, serial_bytes)
            add_limit_finding(
                findings,
                path,
                index,
                "serial_output_budget_exceeded",
                "serial_output_bytes",
                serial_bytes,
                args.max_serial_output_bytes,
                "captured serial output exceeds configured byte budget",
            )

        stdout_tail_bytes = text_bytes(row.get("stdout_tail"))
        stderr_tail_bytes = text_bytes(row.get("stderr_tail"))
        failure_reason_bytes = text_bytes(row.get("failure_reason"))
        max_stdout_tail = max(max_stdout_tail, stdout_tail_bytes)
        max_stderr_tail = max(max_stderr_tail, stderr_tail_bytes)
        max_failure_reason = max(max_failure_reason, failure_reason_bytes)
        add_limit_finding(
            findings,
            path,
            index,
            "stdout_tail_budget_exceeded",
            "stdout_tail_bytes",
            stdout_tail_bytes,
            args.max_stdout_tail_bytes,
            "stdout tail exceeds configured byte budget",
        )
        add_limit_finding(
            findings,
            path,
            index,
            "stderr_tail_budget_exceeded",
            "stderr_tail_bytes",
            stderr_tail_bytes,
            args.max_stderr_tail_bytes,
            "stderr tail exceeds configured byte budget",
        )
        add_limit_finding(
            findings,
            path,
            index,
            "failure_reason_budget_exceeded",
            "failure_reason_bytes",
            failure_reason_bytes,
            args.max_failure_reason_bytes,
            "failure reason exceeds configured byte budget",
        )

    return (
        ArtifactRecord(
            source=str(path),
            status="fail" if findings else "pass",
            suffix=path.suffix.lower(),
            size_bytes=size_bytes,
            row_count=len(rows),
            max_serial_output_bytes=max_serial,
            max_stdout_tail_bytes=max_stdout_tail,
            max_stderr_tail_bytes=max_stderr_tail,
            max_failure_reason_bytes=max_failure_reason,
        ),
        findings,
    )


def build_report(records: list[ArtifactRecord], findings: list[BudgetFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "rows": sum(record.row_count for record in records),
            "findings": len(findings),
            "failed_artifacts": sum(1 for record in records if record.status != "pass"),
            "total_size_bytes": sum(record.size_bytes for record in records),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[ArtifactRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArtifactRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[BudgetFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BudgetFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Artifact Budget Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Rows: {summary['rows']}",
        f"- Total size bytes: {summary['total_size_bytes']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Size bytes | Rows | Max serial bytes | Max stdout tail bytes | Max stderr tail bytes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {size_bytes} | {row_count} | {max_serial_output_bytes} | {max_stdout_tail_bytes} | {max_stderr_tail_bytes} |".format(
                **record
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} row {finding['row']} {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[BudgetFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_artifact_budget_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.metric}:{finding.row}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = f"{finding.value} > {finding.limit}: {finding.detail}"
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_artifact_budget_audit"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark artifacts or directories to audit")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_artifact_budget_audit_latest")
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-serial-output-bytes", type=int, default=131_072)
    parser.add_argument("--max-stdout-tail-bytes", type=int, default=4096)
    parser.add_argument("--max-stderr-tail-bytes", type=int, default=4096)
    parser.add_argument("--max-failure-reason-bytes", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records: list[ArtifactRecord] = []
    findings: list[BudgetFinding] = []
    for path in sorted(iter_input_files(args.inputs, args.pattern)):
        record, file_findings = audit_file(path, args)
        records.append(record)
        findings.extend(file_findings)

    report = build_report(records, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
