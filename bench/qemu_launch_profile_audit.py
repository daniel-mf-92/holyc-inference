#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for launch profile drift.

This host-side tool reads existing qemu_prompt_bench JSON artifacts only. It
never launches QEMU and never touches the TempleOS guest. It verifies that
recorded command arrays keep a stable executable, machine, CPU, accelerator,
and memory profile across top-level, warmup, and measured benchmark records.
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
COMMAND_ROW_KEYS = ("warmups", "benchmarks")
VALUE_OPTIONS = {
    "-accel": "accelerator",
    "-cpu": "cpu",
    "-m": "memory",
    "-machine": "machine",
    "-M": "machine",
}


@dataclass(frozen=True)
class LaunchProfile:
    executable: str
    machine: str
    cpu: str
    accelerator: str
    memory: str


@dataclass(frozen=True)
class ProfileRecord:
    source: str
    json_path: str
    executable: str
    machine: str
    cpu: str
    accelerator: str
    memory: str
    violation_count: int


@dataclass(frozen=True)
class Finding:
    source: str
    json_path: str
    kind: str
    field: str
    value: str
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


def option_value(command: list[str], option: str) -> str:
    for index, arg in enumerate(command):
        if arg == option and index + 1 < len(command):
            return command[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return ""


def command_profile(command: list[str]) -> LaunchProfile:
    values = {field: "" for field in VALUE_OPTIONS.values()}
    for option, field in VALUE_OPTIONS.items():
        value = option_value(command, option)
        if value and not values[field]:
            values[field] = value
    return LaunchProfile(
        executable=Path(command[0]).name if command else "",
        machine=values["machine"],
        cpu=values["cpu"],
        accelerator=values["accelerator"],
        memory=values["memory"],
    )


def command_arrays(payload: dict[str, Any]) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    root_command = payload.get("command")
    if isinstance(root_command, list) and all(isinstance(item, str) for item in root_command):
        commands.append(("$.command", list(root_command)))
    for key in COMMAND_ROW_KEYS:
        rows = payload.get(key)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            command = row.get("command")
            if isinstance(command, list) and all(isinstance(item, str) for item in command):
                commands.append((f"$.{key}[{index}].command", list(command)))
    return commands


def field_value(profile: LaunchProfile, field: str) -> str:
    return str(getattr(profile, field))


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[list[ProfileRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), "$", "invalid_artifact", "artifact", "", error)
        return [], [finding]

    commands = command_arrays(payload)
    findings: list[Finding] = []
    records: list[ProfileRecord] = []
    if not commands:
        return [], [Finding(str(path), "$", "missing_command", "command", "", "artifact has no recorded command arrays")]

    baseline_path, baseline_command = commands[0]
    baseline = command_profile(baseline_command)
    for json_path, command in commands:
        profile = command_profile(command)
        row_findings: list[Finding] = []
        if args.require_memory and not profile.memory:
            row_findings.append(Finding(str(path), json_path, "missing_memory", "memory", "", "command lacks -m memory setting"))
        if args.require_machine and not profile.machine:
            row_findings.append(Finding(str(path), json_path, "missing_machine", "machine", "", "command lacks -machine/-M setting"))
        if args.require_cpu and not profile.cpu:
            row_findings.append(Finding(str(path), json_path, "missing_cpu", "cpu", "", "command lacks -cpu setting"))
        if args.require_accelerator and not profile.accelerator:
            row_findings.append(Finding(str(path), json_path, "missing_accelerator", "accelerator", "", "command lacks -accel setting"))
        for field in ("executable", "machine", "cpu", "accelerator", "memory"):
            expected = field_value(baseline, field)
            actual = field_value(profile, field)
            if actual != expected:
                row_findings.append(
                    Finding(
                        str(path),
                        json_path,
                        "profile_drift",
                        field,
                        actual,
                        f"{field} differs from {baseline_path}: expected {expected!r}",
                    )
                )
        findings.extend(row_findings)
        records.append(
            ProfileRecord(
                source=str(path),
                json_path=json_path,
                executable=profile.executable,
                machine=profile.machine,
                cpu=profile.cpu,
                accelerator=profile.accelerator,
                memory=profile.memory,
                violation_count=len(row_findings),
            )
        )
    return records, findings


def profile_signature(record: ProfileRecord) -> tuple[str, str, str, str, str]:
    return (record.executable, record.machine, record.cpu, record.accelerator, record.memory)


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ProfileRecord], list[Finding]]:
    records: list[ProfileRecord] = []
    findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        artifact_records, artifact_findings = audit_artifact(path, args)
        records.extend(artifact_records)
        findings.extend(artifact_findings)
    if seen_files < args.min_artifacts:
        findings.append(Finding("", "$", "min_artifacts", "inputs", str(seen_files), f"expected at least {args.min_artifacts} artifact(s)"))
    if len(records) < args.min_commands:
        findings.append(Finding("", "$", "min_commands", "command", str(len(records)), f"expected at least {args.min_commands} command(s)"))
    if args.fail_on_cross_artifact_drift:
        signatures_by_source: dict[str, set[tuple[str, str, str, str, str]]] = {}
        for record in records:
            signatures_by_source.setdefault(record.source, set()).add(profile_signature(record))
        artifact_signatures = {
            source: next(iter(signatures))
            for source, signatures in signatures_by_source.items()
            if len(signatures) == 1
        }
        distinct_signatures = set(artifact_signatures.values())
        if len(distinct_signatures) > 1:
            findings.append(
                Finding(
                    "",
                    "$",
                    "cross_artifact_profile_drift",
                    "profile",
                    str(len(distinct_signatures)),
                    "benchmark artifacts use different executable/machine/cpu/accelerator/memory profiles",
                )
            )
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(records: list[ProfileRecord], findings: list[Finding], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = "fail" if findings else "pass"
    summary = {
        "artifacts": len({record.source for record in records}),
        "commands": len(records),
        "findings": len(findings),
        "commands_with_violations": sum(1 for record in records if record.violation_count),
        "status": status,
    }
    payload = {
        "tool": "qemu_launch_profile_audit",
        "generated_at": iso_now(),
        "status": status,
        "summary": summary,
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        args.output_dir / f"{stem}.csv",
        [asdict(record) for record in records],
        ["source", "json_path", "executable", "machine", "cpu", "accelerator", "memory", "violation_count"],
    )
    write_csv(
        args.output_dir / f"{stem}_findings.csv",
        [asdict(finding) for finding in findings],
        ["source", "json_path", "kind", "field", "value", "detail"],
    )
    lines = [
        "# QEMU Launch Profile Audit",
        "",
        f"- Status: {status}",
        f"- Commands checked: {len(records)}",
        f"- Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            location = f"{finding.source}:{finding.json_path}" if finding.source else "inputs"
            lines.append(f"- {finding.kind} `{finding.field}` at {location}: {finding.detail}")
    else:
        lines.append("No launch profile drift findings.")
    (args.output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite = ET.Element(
        "testsuite",
        name="qemu_launch_profile_audit",
        tests=str(max(1, len(records))),
        failures=str(len(findings)),
    )
    if not records:
        testcase = ET.SubElement(suite, "testcase", name="inputs")
        for finding in findings:
            ET.SubElement(testcase, "failure", message=finding.kind).text = finding.detail
    for record in records:
        testcase = ET.SubElement(suite, "testcase", name=f"{record.source}:{record.json_path}")
        for finding in findings:
            if finding.source == record.source and finding.json_path == record.json_path:
                ET.SubElement(testcase, "failure", message=finding.kind).text = finding.detail
    ET.ElementTree(suite).write(args.output_dir / f"{stem}_junit.xml", encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=None, help="glob pattern when scanning directories")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_launch_profile_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-commands", type=int, default=1)
    parser.add_argument("--require-memory", action="store_true")
    parser.add_argument("--require-machine", action="store_true")
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--require-accelerator", action="store_true")
    parser.add_argument("--fail-on-cross-artifact-drift", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.pattern is None:
        args.pattern = list(DEFAULT_PATTERNS)
    records, findings = audit(args.inputs, args)
    write_outputs(records, findings, args)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
