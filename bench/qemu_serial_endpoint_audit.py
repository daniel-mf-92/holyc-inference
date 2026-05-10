#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for serial and monitor endpoints.

This host-side tool reads benchmark JSON artifacts only. It never launches QEMU.
Benchmark commands must keep guest I/O local to stdio and must not expose serial,
monitor, QMP, gdb, or chardev endpoints over sockets.
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
ENDPOINT_OPTIONS = {"-serial", "-monitor", "-qmp", "-qmp-pretty", "-gdb", "-chardev"}
SOCKET_MARKERS = ("tcp:", "udp:", "telnet:", "socket,", "websocket", "hostfwd=", "guestfwd=", "tftp=", "unix:")
STDIO_SERIAL_VALUES = {"stdio", "mon:stdio"}


@dataclass(frozen=True)
class CommandRecord:
    source: str
    row: int
    list_name: str
    phase: str
    prompt: str
    launch_index: int | None
    command_argc: int
    serial_stdio: bool
    nographic: bool
    socket_endpoint_count: int
    endpoint_options: str


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    command_rows: int
    serial_stdio_rows: int
    socket_endpoint_rows: int
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


def option_value(command: list[str], index: int) -> str:
    arg = command[index]
    if "=" in arg:
        return arg.split("=", 1)[1]
    if index + 1 < len(command):
        return command[index + 1]
    return ""


def is_endpoint_option(arg: str) -> bool:
    return arg in ENDPOINT_OPTIONS or any(arg.startswith(f"{option}=") for option in ENDPOINT_OPTIONS)


def has_socket_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in SOCKET_MARKERS)


def has_serial_stdio(command: list[str]) -> bool:
    if "-nographic" in command:
        return True
    for index, arg in enumerate(command):
        if arg == "-serial" or arg.startswith("-serial="):
            if option_value(command, index).lower() in STDIO_SERIAL_VALUES:
                return True
    return False


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
    require_serial_stdio: bool,
) -> CommandRecord:
    endpoint_values: list[str] = []
    socket_endpoint_count = 0
    index = 0
    while index < len(command):
        arg = command[index]
        if is_endpoint_option(arg):
            value = option_value(command, index)
            endpoint_values.append(f"{arg} {value}".strip())
            if has_socket_marker(value):
                socket_endpoint_count += 1
                findings.append(
                    Finding(
                        str(path),
                        row_number,
                        "error",
                        "socket_endpoint",
                        "command",
                        f"`{arg} {value}` violates stdio-only QEMU endpoint policy",
                    )
                )
            if "=" not in arg:
                index += 2
                continue
        index += 1

    serial_stdio = has_serial_stdio(command)
    if require_serial_stdio and not serial_stdio:
        findings.append(
            Finding(str(path), row_number, "error", "missing_serial_stdio", "command", "command must include `-serial stdio`, `-serial mon:stdio`, or `-nographic`")
        )

    return CommandRecord(
        source=str(path),
        row=row_number,
        list_name=list_name,
        phase=phase,
        prompt=prompt,
        launch_index=launch_index,
        command_argc=len(command),
        serial_stdio=serial_stdio,
        nographic="-nographic" in command,
        socket_endpoint_count=socket_endpoint_count,
        endpoint_options=" | ".join(endpoint_values),
    )


def audit_artifact(path: Path, *, require_top_command: bool, require_serial_stdio: bool) -> tuple[ArtifactRecord, list[CommandRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), 0, "error", "load_error", "artifact", error)
        return ArtifactRecord(str(path), "fail", 0, 0, 0, 1, error), [], [finding]

    findings: list[Finding] = []
    commands: list[CommandRecord] = []
    top_command = command_list(payload.get("command"))
    if top_command is None:
        if require_top_command:
            findings.append(Finding(str(path), 0, "error", "missing_top_command", "command", "top-level command must be a list of strings"))
    else:
        commands.append(audit_command(path, 0, "artifact", "artifact", "-", None, top_command, findings, require_serial_stdio=require_serial_stdio))

    for row_number, (list_name, row) in enumerate(data_rows(payload, findings, path), 1):
        command = command_list(row.get("command"))
        if command is None:
            findings.append(Finding(str(path), row_number, "error", "missing_row_command", "command", "row command must be a list of strings"))
            continue
        phase = text(row.get("phase")) or ("warmup" if list_name == "warmups" else "measured")
        prompt = text(row.get("prompt")) or text(row.get("prompt_id")) or "-"
        commands.append(
            audit_command(
                path,
                row_number,
                list_name,
                phase,
                prompt,
                int_or_none(row.get("launch_index")),
                command,
                findings,
                require_serial_stdio=require_serial_stdio,
            )
        )

    return (
        ArtifactRecord(
            source=str(path),
            status="pass" if not findings else "fail",
            command_rows=len(commands),
            serial_stdio_rows=sum(1 for command in commands if command.serial_stdio),
            socket_endpoint_rows=sum(1 for command in commands if command.socket_endpoint_count),
            findings=len(findings),
        ),
        commands,
        findings,
    )


def audit(paths: list[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[CommandRecord], list[Finding]]:
    artifacts: list[ArtifactRecord] = []
    commands: list[CommandRecord] = []
    findings: list[Finding] = []
    files = list(iter_input_files(paths, args.pattern))
    if not files:
        finding = Finding("", 0, "error", "no_artifacts", "inputs", "no matching QEMU benchmark JSON artifacts found")
        return [ArtifactRecord("", "fail", 0, 0, 0, 1, "no matching artifacts")], [], [finding]
    for path in files:
        artifact, rows, artifact_findings = audit_artifact(path, require_top_command=args.require_top_command, require_serial_stdio=args.require_serial_stdio)
        artifacts.append(artifact)
        commands.extend(rows)
        findings.extend(artifact_findings)
    if args.min_artifacts and len(files) < args.min_artifacts:
        findings.append(Finding("", 0, "error", "min_artifacts", "inputs", f"expected at least {args.min_artifacts} artifacts, found {len(files)}"))
    if args.min_command_rows and len(commands) < args.min_command_rows:
        findings.append(Finding("", 0, "error", "min_command_rows", "commands", f"expected at least {args.min_command_rows} command rows, found {len(commands)}"))
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
        "# QEMU Serial Endpoint Audit",
        "",
        f"Status: **{report['status']}**",
        f"Artifacts checked: {summary['artifacts']}",
        f"Command rows checked: {summary['command_rows']}",
        f"Rows with stdio serial: {summary['serial_stdio_rows']}",
        f"Rows with socket endpoints: {summary['socket_endpoint_rows']}",
        f"Findings: {summary['findings']}",
        "",
    ]
    if report["findings"]:
        lines.extend(["## Findings", ""])
        for finding in report["findings"]:
            source = finding["source"] or "<inputs>"
            lines.append(f"- {finding['severity']} {finding['kind']} in {source} row {finding['row']}: {finding['detail']}")
    else:
        lines.append("No QEMU serial endpoint findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element("testsuite", {"name": "holyc_qemu_serial_endpoint_audit", "tests": "1", "failures": "1" if findings else "0"})
    case = ET.SubElement(suite, "testcase", {"name": "qemu_serial_endpoint"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} QEMU serial endpoint findings"})
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob pattern when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_serial_endpoint_audit_latest")
    parser.add_argument("--require-top-command", action="store_true")
    parser.add_argument("--require-serial-stdio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-command-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts, commands, findings = audit(args.inputs, args)
    report = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "artifacts": len(artifacts),
            "command_rows": len(commands),
            "serial_stdio_rows": sum(1 for command in commands if command.serial_stdio),
            "socket_endpoint_rows": sum(1 for command in commands if command.socket_endpoint_count),
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
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
