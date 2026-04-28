#!/usr/bin/env python3
"""Audit documented/source QEMU commands for explicit air-gap settings.

This host-side tool scans shell-like docs and config files for literal
`qemu-system*` launch commands. Any discovered QEMU command must include
`-nic none` and must not include network backends or virtual NIC devices.
It does not launch QEMU.
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

import airgap_audit


TEXT_SUFFIXES = {
    ".args",
    ".bash",
    ".cfg",
    ".conf",
    ".json",
    ".md",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    "bench/results",
    "bench/dashboards",
}


@dataclass(frozen=True)
class SourceFinding:
    source: str
    line: int
    reason: str
    command: list[str]
    text: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def has_excluded_part(path: Path, root: Path, excludes: set[str]) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    rel_posix = rel.as_posix()
    parts = set(rel.parts)
    return any(item in parts or rel_posix.startswith(f"{item}/") for item in excludes)


def iter_source_files(paths: Iterable[Path], excludes: set[str]) -> Iterable[Path]:
    root = Path.cwd()
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if (
                    child.is_file()
                    and child.suffix.lower() in TEXT_SUFFIXES
                    and not has_excluded_part(child, root, excludes)
                ):
                    yield child
        elif path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if not has_excluded_part(path, root, excludes):
                yield path


def logical_lines(path: Path) -> Iterable[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    buffer: list[str] = []
    start_line = 1
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not buffer:
            start_line = line_number
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].rstrip())
            continue
        if buffer:
            buffer.append(stripped)
            yield start_line, " ".join(part for part in buffer if part)
            buffer = []
        else:
            yield line_number, stripped
    if buffer:
        yield start_line, " ".join(part for part in buffer if part)


def strip_shell_prefixes(text: str) -> str:
    stripped = text.strip()
    for prefix in ("$", ">", "-"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].lstrip()
    for prefix in ("run:", "command:", "cmd:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].lstrip()
    return stripped.strip("`")


def command_from_text(text: str) -> list[str] | None:
    if "qemu-system" not in text:
        return None
    candidate = strip_shell_prefixes(text)
    try:
        tokens = shlex.split(candidate)
    except ValueError:
        tokens = candidate.split()
    for index, token in enumerate(tokens):
        cleaned = token.strip("`'\"")
        executable = Path(cleaned).name.lower()
        if executable.startswith("qemu-system") and "*" not in executable:
            tokens[index] = cleaned
            return tokens[index:]
    return None


def fragment_violations(args: list[str]) -> list[str]:
    """Return network violations for QEMU argument fragments.

    Matrix/profile argument fragments normally do not contain the qemu-system
    executable, and they do not need to include `-nic none` because the host
    launcher injects it. They still must not add any networking back in.
    """
    violations: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        next_arg = args[index + 1] if index + 1 < len(args) else ""

        if arg == "-nic":
            if next_arg != "none":
                violations.append(f"non-air-gapped `-nic {next_arg}`")
            index += 2
            continue
        if arg.startswith("-nic=") and arg != "-nic=none":
            violations.append(f"non-air-gapped `{arg}`")

        if arg == "-net":
            if next_arg != "none":
                violations.append(f"networking `-net {next_arg}`")
            index += 2
            continue
        if arg.startswith("-net=") and arg != "-net=none":
            violations.append(f"networking `{arg}`")

        if arg == "-netdev" or arg.startswith("-netdev"):
            violations.append(f"network backend `{arg}`")
        if arg == "-device" and airgap_audit.is_network_device_arg(next_arg):
            violations.append(f"network device `{next_arg}`")
        if arg.startswith("-device=") and airgap_audit.is_network_device_arg(arg):
            violations.append(f"network device `{arg}`")

        index += 1
    return violations


def iter_json_arg_fragments(payload: Any, path: str = "$") -> Iterable[tuple[str, list[str]]]:
    if path == "$" and isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        yield path, list(payload)
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in {"qemu_args", "qemu_extra_args", "qemu_flags"} and isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    yield child_path, list(value)
                    continue
            yield from iter_json_arg_fragments(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from iter_json_arg_fragments(value, f"{path}[{index}]")


def audit_json_arg_fragments(path: Path) -> tuple[int, list[SourceFinding]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, []
    fragments_checked = 0
    findings: list[SourceFinding] = []
    for json_path, args in iter_json_arg_fragments(payload):
        fragments_checked += 1
        for violation in fragment_violations(args):
            findings.append(
                SourceFinding(
                    source=str(path),
                    line=1,
                    reason=f"{json_path}: {violation}",
                    command=args,
                    text=json.dumps(args, separators=(",", ":")),
                )
            )
    return fragments_checked, findings


def audit_args_file_fragments(path: Path) -> tuple[int, list[SourceFinding]]:
    fragments_checked = 0
    findings: list[SourceFinding] = []
    for line_number, text in logical_lines(path):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            args = shlex.split(stripped)
        except ValueError:
            args = stripped.split()
        fragments_checked += 1
        for violation in fragment_violations(args):
            findings.append(
                SourceFinding(
                    source=str(path),
                    line=line_number,
                    reason=violation,
                    command=args,
                    text=text,
                )
            )
    return fragments_checked, findings


def audit(paths: Iterable[Path], excludes: set[str]) -> tuple[int, list[SourceFinding]]:
    commands_checked = 0
    findings: list[SourceFinding] = []
    for path in iter_source_files(paths, excludes):
        if path.suffix.lower() == ".json":
            fragments_checked, json_findings = audit_json_arg_fragments(path)
            commands_checked += fragments_checked
            findings.extend(json_findings)
        elif path.suffix.lower() == ".args":
            fragments_checked, args_findings = audit_args_file_fragments(path)
            commands_checked += fragments_checked
            findings.extend(args_findings)

        for line_number, text in logical_lines(path):
            command = command_from_text(text)
            if command is None:
                continue
            commands_checked += 1
            for violation in airgap_audit.command_violations(command):
                findings.append(
                    SourceFinding(
                        source=str(path),
                        line=line_number,
                        reason=violation,
                        command=command,
                        text=text,
                    )
                )
    return commands_checked, findings


def markdown_report(report: dict[str, object]) -> str:
    findings = report["findings"]
    assert isinstance(findings, list)
    lines = [
        "# QEMU Source Air-Gap Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Source commands checked: {report['commands_checked']}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(
            [
                "| Source | Line | Reason | Command |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for finding in findings:
            command = shlex.join(finding["command"])
            lines.append(
                "| {source} | {line} | {reason} | `{command}` |".format(
                    source=finding["source"],
                    line=finding["line"],
                    reason=finding["reason"].replace("|", "\\|"),
                    command=command.replace("`", "\\`"),
                )
            )
    else:
        lines.append("All documented/source QEMU commands explicitly disable networking with `-nic none`.")
    return "\n".join(lines) + "\n"


def write_csv(findings: list[SourceFinding], path: Path) -> None:
    fields = ["source", "line", "reason", "command", "text"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(
                {
                    "source": finding.source,
                    "line": finding.line,
                    "reason": finding.reason,
                    "command": shlex.join(finding.command),
                    "text": finding.text,
                }
            )


def write_junit(report: dict[str, object], path: Path) -> None:
    findings = [row for row in report["findings"] if isinstance(row, dict)]
    failing_commands: dict[tuple[str, int, tuple[str, ...]], list[str]] = {}
    for finding in findings:
        key = (
            str(finding["source"]),
            int(finding["line"]),
            tuple(str(item) for item in finding["command"]),
        )
        failing_commands.setdefault(key, []).append(str(finding["reason"]))

    passing_commands = max(0, int(report["commands_checked"]) - len(failing_commands))
    testcase_count = len(failing_commands) + (1 if passing_commands else 0)
    if testcase_count == 0:
        testcase_count = 1

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_source_airgap_audit",
            "tests": str(testcase_count),
            "failures": str(len(failing_commands)),
            "errors": "0",
        },
    )

    if passing_commands:
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_source_audit.commands",
                "name": f"air_gapped_source_qemu_commands:{passing_commands}",
            },
        )
    elif not failing_commands:
        ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_source_audit.commands",
                "name": "no_source_qemu_commands_checked",
            },
        )

    for index, ((source, line, command), reasons) in enumerate(sorted(failing_commands.items()), 1):
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "qemu_source_audit.commands",
                "name": f"{Path(source).name}:{line}:{index}",
            },
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_source_airgap_violation",
                "message": "; ".join(reasons),
            },
        )
        failure.text = "\n".join(
            [
                f"source={source}",
                f"line={line}",
                f"command={shlex.join(list(command))}",
                "reasons:",
                *[f"- {reason}" for reason in reasons],
            ]
        )

    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Source file or directory; defaults to bench, .github, and LOOP_PROMPT_GPT55.md",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Relative path or path part to exclude; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/qemu_source_audit_latest.json"),
    )
    parser.add_argument("--markdown", type=Path, help="Optional Markdown audit report path")
    parser.add_argument("--csv", type=Path, help="Optional CSV findings report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML audit report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or [Path("bench"), Path(".github"), Path("LOOP_PROMPT_GPT55.md")]
    excludes = set(DEFAULT_EXCLUDE_PARTS)
    excludes.update(args.exclude)
    commands_checked, findings = audit(inputs, excludes)
    report = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "commands_checked": commands_checked,
        "findings": [asdict(finding) for finding in findings],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(findings, args.csv)
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        write_junit(report, args.junit)
    print(f"wrote_json={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")
    print(f"commands_checked={commands_checked}")
    if findings:
        for finding in findings[:20]:
            print(f"{finding.source}:{finding.line}: {finding.reason}", file=sys.stderr)
        if len(findings) > 20:
            print(f"... {len(findings) - 20} more findings", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
