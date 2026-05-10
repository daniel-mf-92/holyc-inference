#!/usr/bin/env python3
"""Audit saved benchmark QEMU commands for explicit air-gap enforcement.

This host-side tool reads existing JSON/JSONL/CSV benchmark artifacts. It does
not launch QEMU or touch the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_prompt_bench


COMMAND_KEYS = ("command", "qemu_command", "launch_command", "qemu_launch_command")
RESULT_KEYS = ("benchmarks", "results", "runs", "rows", "warmups")


@dataclass(frozen=True)
class CommandRecord:
    source: str
    row: int
    prompt: str
    phase: str
    command: list[str]
    recorded_airgap_ok: bool | None
    recorded_explicit_nic_none: bool | None
    recorded_legacy_net_none: bool | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    prompt: str
    phase: str
    detail: str
    command: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def as_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "pass", "ok"}:
            return True
        if lowered in {"0", "false", "no", "n", "fail"}:
            return False
    return None


def parse_command(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if isinstance(value, str) and value.strip():
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
                return list(decoded)
        try:
            return shlex.split(stripped)
        except ValueError:
            return None
    return None


def normalize_command(value: Any) -> list[str] | None:
    return parse_command(value)


def qemu_like(command: list[str]) -> bool:
    executable = Path(command[0]).name.lower() if command else ""
    return "qemu-system" in executable or any(arg.startswith("-drive") for arg in command)


def is_network_device_arg(value: str) -> bool:
    return qemu_prompt_bench.is_network_device_arg(value)


def command_violations(command: list[str]) -> list[str]:
    return qemu_prompt_bench.command_airgap_violations(command)


def row_command(raw: dict[str, Any]) -> list[str] | None:
    for key in COMMAND_KEYS:
        command = parse_command(raw.get(key))
        if command is not None:
            return command
    return None


def flatten_json_payload(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return

    inherited = {key: value for key, value in payload.items() if key not in RESULT_KEYS}
    yielded = False
    for key in RESULT_KEYS:
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        yielded = True
        for row in rows:
            if isinstance(row, dict):
                merged = dict(inherited)
                merged.update(row)
                yield merged
    if not yielded:
        yield payload


def load_json_records(path: Path) -> Iterable[dict[str, Any]]:
    yield from flatten_json_payload(json.loads(path.read_text(encoding="utf-8")))


def load_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            yield from flatten_json_payload(payload)


def load_csv_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def iter_input_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".csv"}
            )
        elif path.suffix.lower() in {".json", ".jsonl", ".csv"}:
            yield path


def load_records(paths: Iterable[Path]) -> list[CommandRecord]:
    records: list[CommandRecord] = []
    for path in sorted(iter_input_files(paths)):
        suffix = path.suffix.lower()
        if suffix == ".json":
            loader = load_json_records
        elif suffix == ".jsonl":
            loader = load_jsonl_records
        else:
            loader = load_csv_records

        for row_number, raw in enumerate(loader(path), 1):
            command = row_command(raw)
            if command is None or not qemu_like(command):
                continue
            records.append(
                CommandRecord(
                    source=str(path),
                    row=row_number,
                    prompt=str(raw.get("prompt") or raw.get("prompt_id") or "-"),
                    phase=str(raw.get("phase") or "-"),
                    command=command,
                    recorded_airgap_ok=as_bool(raw.get("command_airgap_ok")),
                    recorded_explicit_nic_none=as_bool(raw.get("command_has_explicit_nic_none")),
                    recorded_legacy_net_none=as_bool(raw.get("command_has_legacy_net_none")),
                )
            )
    return records


def command_string(command: list[str]) -> str:
    return shlex.join(command)


def evaluate(records: list[CommandRecord], min_commands: int) -> list[Finding]:
    findings: list[Finding] = []
    for record in records:
        violations = qemu_prompt_bench.command_airgap_violations(record.command)
        actual_ok = not violations
        actual_explicit = qemu_prompt_bench.has_explicit_nic_none(record.command)
        actual_legacy = qemu_prompt_bench.has_legacy_net_none(record.command)
        rendered_command = command_string(record.command)

        for violation in violations:
            findings.append(
                Finding(
                    source=record.source,
                    row=record.row,
                    severity="error",
                    kind="airgap_violation",
                    prompt=record.prompt,
                    phase=record.phase,
                    detail=violation,
                    command=rendered_command,
                )
            )

        recorded_checks = (
            ("command_airgap_ok", record.recorded_airgap_ok, actual_ok),
            ("command_has_explicit_nic_none", record.recorded_explicit_nic_none, actual_explicit),
            ("command_has_legacy_net_none", record.recorded_legacy_net_none, actual_legacy),
        )
        for field, recorded, actual in recorded_checks:
            if recorded is not None and recorded != actual:
                findings.append(
                    Finding(
                        source=record.source,
                        row=record.row,
                        severity="error",
                        kind="recorded_airgap_drift",
                        prompt=record.prompt,
                        phase=record.phase,
                        detail=f"{field} recorded={recorded} actual={actual}",
                        command=rendered_command,
                    )
                )

    if len(records) < min_commands:
        findings.append(
            Finding(
                source="suite",
                row=0,
                severity="error",
                kind="coverage",
                prompt="-",
                phase="-",
                detail=f"commands {len(records)} < {min_commands}",
                command="",
            )
        )
    return findings


def summarize(records: list[CommandRecord], findings: list[Finding]) -> dict[str, Any]:
    commands_with_nic_none = sum(1 for record in records if qemu_prompt_bench.has_explicit_nic_none(record.command))
    commands_with_legacy_net_none = sum(1 for record in records if qemu_prompt_bench.has_legacy_net_none(record.command))
    violation_rows = {
        (finding.source, finding.row)
        for finding in findings
        if finding.kind == "airgap_violation"
    }
    return {
        "commands": len(records),
        "commands_with_explicit_nic_none": commands_with_nic_none,
        "commands_with_legacy_net_none": commands_with_legacy_net_none,
        "airgap_violation_rows": len(violation_rows),
        "findings": len(findings),
        "sources": sorted({record.source for record in records}),
    }


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Airgap Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Commands checked: {summary['commands']}",
        f"Commands with `-nic none`: {summary['commands_with_explicit_nic_none']}",
        f"Commands with legacy `-net none`: {summary['commands_with_legacy_net_none']}",
        f"Findings: {summary['findings']}",
        "",
        "## Findings",
        "",
    ]
    if not report["findings"]:
        lines.append("No air-gap findings.")
    else:
        lines.extend(
            [
                "| Source | Row | Kind | Prompt | Phase | Detail |",
                "| --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for finding in report["findings"]:
            lines.append(
                "| {source} | {row} | {kind} | {prompt} | {phase} | {detail} |".format(
                    **{key: markdown_escape(value) for key, value in finding.items()}
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "row", "severity", "kind", "prompt", "phase", "detail", "command"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_airgap_audit",
            "tests": "1",
            "failures": str(len(findings)),
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "airgap_audit"})
    for finding in findings:
        failure = ET.SubElement(
            case,
            "failure",
            {"type": finding.kind, "message": finding.detail},
        )
        failure.text = json.dumps(asdict(finding), sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(report: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    md_path = args.output_dir / f"{args.output_stem}.md"
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    junit_path = args.output_dir / f"{args.output_stem}_junit.xml"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(csv_path, [Finding(**finding) for finding in report["findings"]])
    write_junit(junit_path, [Finding(**finding) for finding in report["findings"]])
    return json_path, md_path, csv_path, junit_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON/JSONL/CSV files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="airgap_audit_latest")
    parser.add_argument("--min-commands", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.inputs)
    findings = evaluate(records, args.min_commands)
    report = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "inputs": [str(path) for path in args.inputs],
        "thresholds": {"min_commands": args.min_commands},
        "summary": summarize(records, findings),
        "findings": [asdict(finding) for finding in findings],
    }
    json_path, md_path, csv_path, junit_path = write_outputs(report, args)
    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_junit={junit_path}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
