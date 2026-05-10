#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for effective NIC cardinality.

This host-side tool reads benchmark JSON artifacts only. It never launches QEMU.
Every saved top-level, warmup, and measured command must include exactly one
explicit `-nic none` disablement and no networking violations. Real QEMU
executable enforcement is opt-in so synthetic CI harness artifacts can still be
audited for the same effective air-gap flag contract.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class CommandRecord:
    source: str
    row: int
    list_name: str
    phase: str
    prompt: str
    launch_index: int | None
    command_argc: int
    executable: str
    qemu_system_executable: bool
    nic_none_count: int
    explicit_nic_none: bool
    legacy_net_none: bool
    airgap_ok: bool
    violation_count: int


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    command_rows: int
    qemu_system_rows: int
    exact_one_nic_none_rows: int
    airgap_ok_rows: int
    findings: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
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


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number.is_integer():
        return None
    return int(number)


def command_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def is_qemu_system_executable(command: list[str]) -> bool:
    return bool(command) and Path(command[0].strip("'\"`")).name.lower().startswith("qemu-system")


def canonical_qemu_option(arg: str) -> str:
    if arg.startswith("--") and len(arg) > 2:
        return "-" + arg[2:]
    return arg


def nic_none_count(command: list[str]) -> int:
    count = 0
    index = 0
    while index < len(command):
        option = canonical_qemu_option(command[index])
        if option == "-nic" and index + 1 < len(command) and command[index + 1] == "none":
            count += 1
            index += 2
            continue
        if option == "-nic=none":
            count += 1
        index += 1
    return count


def data_rows(payload: dict[str, Any], findings: list[Finding], path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for list_name in ("warmups", "benchmarks"):
        raw = payload.get(list_name)
        if raw is None:
            continue
        if not isinstance(raw, list):
            findings.append(Finding(str(path), 0, "error", f"invalid_{list_name}", list_name, f"{list_name} must be a list"))
            continue
        for row in raw:
            if isinstance(row, dict):
                rows.append((list_name, row))
            else:
                findings.append(Finding(str(path), len(rows) + 1, "error", "invalid_row", list_name, "benchmark row must be an object"))
    return rows


def audit_command(
    path: Path,
    row_number: int,
    list_name: str,
    phase: str,
    prompt: str,
    launch_index: int | None,
    command: list[str],
    findings: list[Finding],
    *,
    require_qemu_system_executable: bool,
) -> CommandRecord:
    executable = command[0] if command else ""
    qemu_system = is_qemu_system_executable(command)
    if require_qemu_system_executable and not qemu_system:
        findings.append(
            Finding(
                str(path),
                row_number,
                "error",
                "non_qemu_system_executable",
                "command",
                f"command must start with qemu-system-* executable, got {executable or '<empty>'}",
            )
        )

    nic_count = nic_none_count(command)
    if nic_count != 1:
        findings.append(
            Finding(
                str(path),
                row_number,
                "error",
                "nic_none_cardinality",
                "command",
                f"command must include exactly one explicit `-nic none`; found {nic_count}",
            )
        )

    airgap = qemu_prompt_bench.command_airgap_metadata(command)
    for violation in airgap["violations"]:
        findings.append(Finding(str(path), row_number, "error", "airgap_violation", "command", violation))

    return CommandRecord(
        source=str(path),
        row=row_number,
        list_name=list_name,
        phase=phase,
        prompt=prompt,
        launch_index=launch_index,
        command_argc=len(command),
        executable=executable,
        qemu_system_executable=qemu_system,
        nic_none_count=nic_count,
        explicit_nic_none=bool(airgap["explicit_nic_none"]),
        legacy_net_none=bool(airgap["legacy_net_none"]),
        airgap_ok=bool(airgap["ok"]),
        violation_count=len(airgap["violations"]),
    )


def audit_artifact(
    path: Path, *, require_top_command: bool, require_qemu_system_executable: bool
) -> tuple[ArtifactRecord, list[CommandRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), 0, "error", "load_error", "artifact", error)
        return ArtifactRecord(str(path), "fail", 0, 0, 0, 0, 1, error), [], [finding]

    findings: list[Finding] = []
    command_records: list[CommandRecord] = []
    top_command = command_list(payload.get("command"))
    if top_command is None:
        if require_top_command:
            findings.append(Finding(str(path), 0, "error", "missing_top_command", "command", "top-level command must be a list of strings"))
    else:
        command_records.append(
            audit_command(
                path,
                0,
                "artifact",
                "artifact",
                "-",
                None,
                top_command,
                findings,
                require_qemu_system_executable=require_qemu_system_executable,
            )
        )

    for row_number, (list_name, row) in enumerate(data_rows(payload, findings, path), 1):
        command = command_list(row.get("command"))
        if command is None:
            findings.append(Finding(str(path), row_number, "error", "missing_row_command", "command", "row command must be a list of strings"))
            continue
        phase = text(row.get("phase")) or ("warmup" if list_name == "warmups" else "measured")
        prompt = text(row.get("prompt")) or text(row.get("prompt_id")) or "-"
        launch_index = int_or_none(row.get("launch_index"))
        command_records.append(
            audit_command(
                path,
                row_number,
                list_name,
                phase,
                prompt,
                launch_index,
                command,
                findings,
                require_qemu_system_executable=require_qemu_system_executable,
            )
        )

    status = "pass" if not findings else "fail"
    return (
        ArtifactRecord(
            source=str(path),
            status=status,
            command_rows=len(command_records),
            qemu_system_rows=sum(1 for record in command_records if record.qemu_system_executable),
            exact_one_nic_none_rows=sum(1 for record in command_records if record.nic_none_count == 1),
            airgap_ok_rows=sum(1 for record in command_records if record.airgap_ok),
            findings=len(findings),
            error="",
        ),
        command_records,
        findings,
    )


def audit(paths: list[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[CommandRecord], list[Finding]]:
    artifacts: list[ArtifactRecord] = []
    commands: list[CommandRecord] = []
    findings: list[Finding] = []
    files = list(iter_input_files(paths, args.pattern))
    if not files:
        finding = Finding("", 0, "error", "no_artifacts", "inputs", "no matching QEMU benchmark JSON artifacts found")
        return [ArtifactRecord("", "fail", 0, 0, 0, 0, 1, "no matching artifacts")], [], [finding]
    for path in files:
        artifact, rows, artifact_findings = audit_artifact(
            path,
            require_top_command=args.require_top_command,
            require_qemu_system_executable=args.require_qemu_system_executable,
        )
        artifacts.append(artifact)
        commands.extend(rows)
        findings.extend(artifact_findings)
    if args.min_artifacts and len(files) < args.min_artifacts:
        findings.append(
            Finding("", 0, "error", "min_artifacts", "inputs", f"expected at least {args.min_artifacts} artifacts, found {len(files)}")
        )
    if args.min_command_rows and len(commands) < args.min_command_rows:
        findings.append(
            Finding("", 0, "error", "min_command_rows", "commands", f"expected at least {args.min_command_rows} command rows, found {len(commands)}")
        )
    return artifacts, commands, findings


def write_csv(path: Path, rows: list[Any]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else ["source"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU NIC Cardinality Audit",
        "",
        f"Status: **{report['status']}**",
        f"Artifacts checked: {summary['artifacts']}",
        f"Command rows checked: {summary['command_rows']}",
        f"Rows with qemu-system executable: {summary['qemu_system_rows']}",
        f"Rows with exactly one `-nic none`: {summary['exact_one_nic_none_rows']}",
        f"Air-gap OK rows: {summary['airgap_ok_rows']}",
        f"Findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", ""])
        for finding in report["findings"]:
            source = finding["source"] or "<inputs>"
            lines.append(f"- {finding['severity']} {finding['kind']} in {source} row {finding['row']}: {finding['detail']}")
    else:
        lines.append("No NIC cardinality findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_nic_cardinality_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "qemu_nic_cardinality"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} NIC cardinality findings"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_nic_cardinality_audit_latest")
    parser.add_argument("--require-top-command", action="store_true")
    parser.add_argument("--require-qemu-system-executable", action="store_true")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-command-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts, commands, findings = audit(args.inputs, args)
    status = "pass" if not findings else "fail"
    report = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "artifacts": len(artifacts),
            "command_rows": len(commands),
            "qemu_system_rows": sum(1 for command in commands if command.qemu_system_executable),
            "exact_one_nic_none_rows": sum(1 for command in commands if command.nic_none_count == 1),
            "airgap_ok_rows": sum(1 for command in commands if command.airgap_ok),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "commands": [asdict(command) for command in commands],
        "findings": [asdict(finding) for finding in findings],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", commands)
    write_csv(args.output_dir / f"{stem}_artifacts.csv", artifacts)
    write_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", report)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
