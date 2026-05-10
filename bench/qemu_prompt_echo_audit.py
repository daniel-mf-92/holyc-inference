#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for host/guest prompt echo parity.

This host-side tool reads saved QEMU benchmark artifacts only. It never
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
RESULT_KEYS = ("benchmarks", "results", "runs", "rows")


@dataclass(frozen=True)
class EchoRow:
    source: str
    row: int
    profile: str
    model: str
    quantization: str
    prompt: str
    commit: str
    phase: str
    exit_class: str
    prompt_bytes: int | None
    guest_prompt_bytes: int | None
    guest_prompt_bytes_match: bool | None
    prompt_sha256: str
    guest_prompt_sha256: str
    guest_prompt_sha256_match: bool | None
    checks: int
    findings: int


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    field: str
    host: str
    guest: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def int_value(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def bool_value(value: Any) -> bool | None:
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


def text_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(value)


def row_text(row: dict[str, Any], *keys: str, default: str = "-") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


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


def check_required(
    findings: list[Finding],
    *,
    source: Path,
    row_number: int,
    field: str,
    host: str,
    guest: str,
    required: bool,
) -> int:
    if host and guest:
        return 1
    if required:
        findings.append(Finding(str(source), row_number, "error", "missing_prompt_echo", field, host, guest, "host or guest prompt field is absent"))
    return 0


def audit_row(source: Path, row_number: int, row: dict[str, Any], args: argparse.Namespace) -> tuple[EchoRow | None, list[Finding]]:
    phase = row_text(row, "phase")
    exit_class = row_text(row, "exit_class")
    if args.only_ok and exit_class != "ok":
        return None, []
    if args.only_measured and phase != "measured":
        return None, []

    findings: list[Finding] = []
    checks = 0
    prompt_bytes = int_value(row.get("prompt_bytes"))
    guest_prompt_bytes = int_value(row.get("guest_prompt_bytes"))
    bytes_match = bool_value(row.get("guest_prompt_bytes_match"))
    prompt_sha = text_value(row.get("prompt_sha256"))
    guest_prompt_sha = text_value(row.get("guest_prompt_sha256"))
    sha_match = bool_value(row.get("guest_prompt_sha256_match"))

    checks += check_required(
        findings,
        source=source,
        row_number=row_number,
        field="prompt_bytes",
        host="" if prompt_bytes is None else str(prompt_bytes),
        guest="" if guest_prompt_bytes is None else str(guest_prompt_bytes),
        required=args.require_guest_echo,
    )
    if prompt_bytes is not None and guest_prompt_bytes is not None and prompt_bytes != guest_prompt_bytes:
        findings.append(
            Finding(str(source), row_number, "error", "prompt_bytes_drift", "prompt_bytes", str(prompt_bytes), str(guest_prompt_bytes), "guest prompt byte count differs from host prompt")
        )
    if bytes_match is not None:
        checks += 1
        expected = prompt_bytes is not None and guest_prompt_bytes is not None and prompt_bytes == guest_prompt_bytes
        if bytes_match != expected:
            findings.append(
                Finding(str(source), row_number, "error", "prompt_bytes_match_flag_drift", "guest_prompt_bytes_match", str(expected).lower(), str(bytes_match).lower(), "guest prompt byte match flag disagrees with derived parity")
            )
    elif args.require_guest_echo:
        findings.append(
            Finding(str(source), row_number, "error", "missing_prompt_echo_flag", "guest_prompt_bytes_match", "", "", "guest prompt byte match flag is absent")
        )

    checks += check_required(
        findings,
        source=source,
        row_number=row_number,
        field="prompt_sha256",
        host=prompt_sha,
        guest=guest_prompt_sha,
        required=args.require_guest_echo,
    )
    if prompt_sha and guest_prompt_sha and prompt_sha != guest_prompt_sha:
        findings.append(
            Finding(str(source), row_number, "error", "prompt_sha256_drift", "prompt_sha256", prompt_sha, guest_prompt_sha, "guest prompt SHA differs from host prompt SHA")
        )
    if sha_match is not None:
        checks += 1
        expected = bool(prompt_sha and guest_prompt_sha and prompt_sha == guest_prompt_sha)
        if sha_match != expected:
            findings.append(
                Finding(str(source), row_number, "error", "prompt_sha256_match_flag_drift", "guest_prompt_sha256_match", str(expected).lower(), str(sha_match).lower(), "guest prompt SHA match flag disagrees with derived parity")
            )
    elif args.require_guest_echo:
        findings.append(
            Finding(str(source), row_number, "error", "missing_prompt_echo_flag", "guest_prompt_sha256_match", "", "", "guest prompt SHA match flag is absent")
        )

    echo_row = EchoRow(
        source=str(source),
        row=row_number,
        profile=row_text(row, "profile"),
        model=row_text(row, "model"),
        quantization=row_text(row, "quantization"),
        prompt=row_text(row, "prompt"),
        commit=row_text(row, "commit"),
        phase=phase,
        exit_class=exit_class,
        prompt_bytes=prompt_bytes,
        guest_prompt_bytes=guest_prompt_bytes,
        guest_prompt_bytes_match=bytes_match,
        prompt_sha256=prompt_sha,
        guest_prompt_sha256=guest_prompt_sha,
        guest_prompt_sha256_match=sha_match,
        checks=checks,
        findings=len(findings),
    )
    return echo_row, findings


def audit(paths: list[Path], args: argparse.Namespace) -> tuple[list[EchoRow], list[Finding]]:
    rows: list[EchoRow] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        for row_number, row in enumerate(load_rows(path), 1):
            echo_row, row_findings = audit_row(path, row_number, row, args)
            if echo_row is not None:
                rows.append(echo_row)
                findings.extend(row_findings)
    if len(rows) < args.min_rows:
        findings.append(
            Finding("", 0, "error", "min_rows", "rows", str(len(rows)), str(args.min_rows), "too few rows were audited")
        )
    return rows, findings


def build_report(rows: list[EchoRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "rows": len(rows),
            "checks": sum(row.checks for row in rows),
            "findings": len(findings),
        },
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Prompt Echo Audit",
        "",
        f"- Status: {report['status']}",
        f"- Rows: {summary['rows']}",
        f"- Checks: {summary['checks']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Source | Row | Prompt | Phase | Exit | Byte match | SHA match | Findings |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {source} | {row} | {prompt} | {phase} | {exit_class} | {guest_prompt_bytes_match} | {guest_prompt_sha256_match} | {findings} |".format(
                **row
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            where = f"{finding['source']}:{finding['row']}" if finding["source"] else "inputs"
            lines.append(f"- {finding['kind']}: {where} {finding['field']} {finding['detail']}")
    else:
        lines.extend(["", "No prompt echo findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_echo_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            name = f"{finding.kind}:{finding.source}:{finding.row}:{finding.field}".rstrip(":")
            case = ET.SubElement(suite, "testcase", {"name": name})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_prompt_echo"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob for benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_echo_audit_latest")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--only-ok", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--only-measured", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-guest-echo", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, findings = audit(args.inputs, args)
    report = build_report(rows, findings)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    findings_path = args.output_dir / f"{args.output_stem}_findings.csv"
    junit_path = args.output_dir / f"{args.output_stem}_junit.xml"
    write_json(json_path, report)
    write_markdown(md_path, report)
    write_csv(csv_path, rows, list(EchoRow.__dataclass_fields__))
    write_csv(findings_path, findings, list(Finding.__dataclass_fields__))
    write_junit(junit_path, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
