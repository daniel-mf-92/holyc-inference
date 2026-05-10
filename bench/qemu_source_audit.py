#!/usr/bin/env python3
"""Audit documented/source QEMU commands for explicit air-gap settings.

This host-side tool scans shell-like docs and config files for literal
`qemu-system*` launch commands. Any discovered QEMU command must include
`-nic none` and must not include network backends or virtual NIC devices.
It does not launch QEMU.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shlex
import sys
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import airgap_audit
import qemu_prompt_bench


TEXT_SUFFIXES = {
    ".args",
    ".bash",
    ".cfg",
    ".conf",
    ".json",
    ".md",
    ".sh",
    ".toml",
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
QEMU_ARG_FRAGMENT_KEYS = {"qemu_args", "qemu_extra_args", "qemu_flags"}
QEMU_ARG_FILE_KEYS = {"qemu_args_file", "qemu_args_files"}


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
    if args and Path(args[0].strip("'\"`")).name.lower().startswith("qemu-system"):
        violations.append("embedded qemu-system executable; args fragments must not contain launch commands")

    index = 0
    while index < len(args):
        arg = args[index]
        next_arg = args[index + 1] if index + 1 < len(args) else ""
        previous_arg = args[index - 1] if index else ""

        if arg in qemu_prompt_bench.TLS_VALUE_OPTIONS and qemu_prompt_bench.is_tls_arg(next_arg):
            violations.append(f"tls option `{arg} {next_arg}`")
        if previous_arg not in qemu_prompt_bench.TLS_VALUE_OPTIONS and qemu_prompt_bench.is_tls_arg(arg):
            violations.append(f"tls option `{arg}`")
        if qemu_prompt_bench.is_remote_resource_arg(arg):
            violations.append(f"remote resource `{arg}`")
        if (
            arg in {"-blockdev", "-cdrom", "-drive", "-hda", "-hdb", "-hdc", "-hdd", "-initrd", "-kernel"}
            and qemu_prompt_bench.is_remote_resource_arg(next_arg)
        ):
            violations.append(f"remote resource `{arg} {next_arg}`")

        if arg.startswith("@") and len(arg) > 1:
            violations.append(f"nested qemu args include `{arg}`")

        if arg == "-readconfig":
            detail = (
                f"qemu config include `-readconfig {next_arg}`"
                if next_arg
                else "qemu config include `-readconfig`"
            )
            violations.append(detail)
            index += 2
            continue
        if arg.startswith("-readconfig="):
            violations.append(f"qemu config include `{arg}`")

        if arg == "-nic":
            if next_arg != "none":
                violations.append(f"non-air-gapped `-nic {next_arg}`")
            index += 2
            continue
        if arg.startswith("-nic=") and arg != "-nic=none":
            violations.append(f"non-air-gapped `{arg}`")

        if arg == "-net":
            if next_arg == "none":
                violations.append("legacy `-net none` present; benchmark fragments must use injected `-nic none`")
            else:
                violations.append(f"networking `-net {next_arg}`")
            index += 2
            continue
        if arg.startswith("-net="):
            if arg == "-net=none":
                violations.append("legacy `-net=none` present; benchmark fragments must use injected `-nic none`")
            else:
                violations.append(f"networking `{arg}`")

        if arg == "-netdev" or arg.startswith("-netdev"):
            violations.append(f"network backend `{arg}`")
        if arg in qemu_prompt_bench.USER_NET_SERVICE_OPTIONS or any(
            arg.startswith(f"{option}=") for option in qemu_prompt_bench.USER_NET_SERVICE_OPTIONS
        ):
            violations.append(f"user-mode network service `{arg}`")
        if arg in qemu_prompt_bench.HOST_FILESYSTEM_SHARE_OPTIONS:
            violations.append(f"host filesystem share `{arg} {next_arg}`")
            index += 2
            continue
        if any(arg.startswith(f"{option}=") for option in qemu_prompt_bench.HOST_FILESYSTEM_SHARE_OPTIONS):
            violations.append(f"host filesystem share `{arg}`")
        if (
            arg in qemu_prompt_bench.HOST_FILESYSTEM_SHARE_VALUE_OPTIONS
            and qemu_prompt_bench.is_host_filesystem_share_marker_arg(next_arg)
        ):
            violations.append(f"host filesystem share marker `{arg} {next_arg}`")
        if arg == "-device" and airgap_audit.is_network_device_arg(next_arg):
            violations.append(f"network device `{next_arg}`")
        if arg.startswith("-device=") and airgap_audit.is_network_device_arg(arg):
            violations.append(f"network device `{arg}`")
        if arg == "-vnc" and next_arg != "none":
            violations.append(f"remote display socket `-vnc {next_arg}`")
            index += 2
            continue
        if arg.startswith("-vnc=") and arg != "-vnc=none":
            violations.append(f"remote display socket `{arg}`")
        if arg == "-display" and qemu_prompt_bench.is_remote_display_arg(next_arg):
            violations.append(f"remote display socket `-display {next_arg}`")
            index += 2
            continue
        if arg.startswith("-display=") and qemu_prompt_bench.is_remote_display_arg(arg):
            violations.append(f"remote display socket `{arg}`")
        if arg == "-spice":
            violations.append(f"remote display socket `-spice {next_arg}`")
            index += 2
            continue
        if arg.startswith("-spice"):
            violations.append(f"remote display socket `{arg}`")
        if (
            arg in qemu_prompt_bench.SOCKET_TRANSPORT_OPTIONS
            and qemu_prompt_bench.is_socket_endpoint_arg(next_arg)
        ):
            violations.append(f"socket endpoint `{arg} {next_arg}`")
            index += 2
            continue
        if (
            any(arg.startswith(f"{option}=") for option in qemu_prompt_bench.SOCKET_TRANSPORT_OPTIONS)
            and qemu_prompt_bench.is_socket_endpoint_arg(arg)
        ):
            violations.append(f"socket endpoint `{arg}`")
        if qemu_prompt_bench.is_socket_endpoint_arg(arg) and ("hostfwd=" in arg.lower() or "guestfwd=" in arg.lower()):
            violations.append(f"forwarded socket endpoint `{arg}`")

        index += 1
    return violations


def iter_json_arg_fragments(payload: Any, path: str = "$") -> Iterable[tuple[str, list[str]]]:
    if path == "$" and isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        yield path, list(payload)
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in QEMU_ARG_FRAGMENT_KEYS and isinstance(value, list):
                if all(isinstance(item, str) for item in value):
                    yield child_path, list(value)
                    continue
            yield from iter_json_arg_fragments(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from iter_json_arg_fragments(value, f"{path}[{index}]")


def json_string_or_string_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def iter_json_arg_file_refs(payload: Any, path: str = "$") -> Iterable[tuple[str, list[str]]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            if key in QEMU_ARG_FILE_KEYS:
                refs = json_string_or_string_list(value)
                if refs is not None:
                    yield child_path, refs
                    continue
            yield from iter_json_arg_file_refs(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from iter_json_arg_file_refs(value, f"{path}[{index}]")


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


def audit_json_arg_file_refs(
    path: Path,
    audited_args_files: set[Path],
) -> tuple[int, list[SourceFinding]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0, []
    return audit_arg_file_refs(path, iter_json_arg_file_refs(payload), audited_args_files)


def load_toml_payload(path: Path) -> Any | None:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        return None


def audit_toml_arg_fragments(path: Path) -> tuple[int, list[SourceFinding]]:
    payload = load_toml_payload(path)
    if payload is None:
        return 0, []

    fragments_checked = 0
    findings: list[SourceFinding] = []
    for toml_path, args in iter_json_arg_fragments(payload):
        fragments_checked += 1
        for violation in fragment_violations(args):
            findings.append(
                SourceFinding(
                    source=str(path),
                    line=1,
                    reason=f"{toml_path}: {violation}",
                    command=args,
                    text=json.dumps(args, separators=(",", ":")),
                )
            )
    return fragments_checked, findings


def audit_toml_arg_file_refs(
    path: Path,
    audited_args_files: set[Path],
) -> tuple[int, list[SourceFinding]]:
    payload = load_toml_payload(path)
    if payload is None:
        return 0, []
    return audit_arg_file_refs(path, iter_json_arg_file_refs(payload), audited_args_files)


def strip_unquoted_comment(text: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
            continue
        if char == "#" and not quote:
            return text[:index].rstrip()
    return text.rstrip()


def parse_yaml_scalar(value: str) -> str:
    stripped = strip_unquoted_comment(value).strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped[1:-1]
        return str(parsed)
    return stripped


def split_inline_yaml_list(value: str) -> list[str] | None:
    stripped = strip_unquoted_comment(value).strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return list(parsed)

    inner = stripped[1:-1].strip()
    if not inner:
        return []
    reader = csv.reader([inner], skipinitialspace=True)
    try:
        return [parse_yaml_scalar(item) for item in next(reader)]
    except csv.Error:
        return [parse_yaml_scalar(item) for item in inner.split(",")]


def yaml_key_line(text: str, keys: set[str]) -> tuple[int, str, str] | None:
    stripped = text.rstrip()
    if not stripped:
        return None
    indent = len(stripped) - len(stripped.lstrip(" "))
    body = stripped.lstrip()
    if body.startswith("- "):
        body = body[2:].lstrip()
    if ":" not in body:
        return None
    key, value = body.split(":", 1)
    key = key.strip().strip("'\"")
    if key not in keys:
        return None
    return indent, key, value.strip()


def yaml_arg_key_line(text: str) -> tuple[int, str, str] | None:
    return yaml_key_line(text, QEMU_ARG_FRAGMENT_KEYS)


def yaml_arg_file_key_line(text: str) -> tuple[int, str, str] | None:
    return yaml_key_line(text, QEMU_ARG_FILE_KEYS)


def yaml_list_item(text: str) -> tuple[int, str] | None:
    stripped = text.rstrip()
    if not stripped:
        return None
    indent = len(stripped) - len(stripped.lstrip(" "))
    body = stripped.lstrip()
    if not body.startswith("- "):
        return None
    return indent, body[2:].strip()


def audit_yaml_arg_fragments(path: Path) -> tuple[int, list[SourceFinding]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    fragments_checked = 0
    findings: list[SourceFinding] = []
    index = 0
    while index < len(lines):
        line_number = index + 1
        parsed_key = yaml_arg_key_line(lines[index])
        if parsed_key is None:
            index += 1
            continue

        key_indent, key, raw_value = parsed_key
        args = split_inline_yaml_list(raw_value)
        if args is None and raw_value:
            args = shlex.split(raw_value)
        if args is None:
            args = []
            scan = index + 1
            while scan < len(lines):
                item = yaml_list_item(lines[scan])
                if item is None:
                    if strip_unquoted_comment(lines[scan]).strip():
                        break
                    scan += 1
                    continue
                item_indent, item_value = item
                if item_indent <= key_indent:
                    break
                args.append(parse_yaml_scalar(item_value))
                scan += 1
            index = scan - 1

        fragments_checked += 1
        for violation in fragment_violations(args):
            findings.append(
                SourceFinding(
                    source=str(path),
                    line=line_number,
                    reason=f"$.{key}: {violation}",
                    command=args,
                    text=strip_unquoted_comment(lines[line_number - 1]).strip(),
                )
            )
        index += 1
    return fragments_checked, findings


def parse_yaml_string_or_list(raw_value: str, lines: list[str], index: int, key_indent: int) -> tuple[list[str], int]:
    refs = split_inline_yaml_list(raw_value)
    if refs is not None:
        return refs, index
    if raw_value:
        return [parse_yaml_scalar(raw_value)], index

    refs = []
    scan = index + 1
    while scan < len(lines):
        item = yaml_list_item(lines[scan])
        if item is None:
            if strip_unquoted_comment(lines[scan]).strip():
                break
            scan += 1
            continue
        item_indent, item_value = item
        if item_indent <= key_indent:
            break
        refs.append(parse_yaml_scalar(item_value))
        scan += 1
    return refs, scan - 1


def iter_yaml_arg_file_refs(path: Path) -> Iterable[tuple[str, list[str]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    index = 0
    while index < len(lines):
        parsed_key = yaml_arg_file_key_line(lines[index])
        if parsed_key is None:
            index += 1
            continue
        key_indent, key, raw_value = parsed_key
        refs, next_index = parse_yaml_string_or_list(raw_value, lines, index, key_indent)
        yield f"$.{key}", refs
        index = next_index + 1


def audit_yaml_arg_file_refs(
    path: Path,
    audited_args_files: set[Path],
) -> tuple[int, list[SourceFinding]]:
    return audit_arg_file_refs(path, iter_yaml_arg_file_refs(path), audited_args_files)


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


def resolve_arg_file_ref(config_path: Path, ref: str) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else config_path.parent / path


def audit_args_file_once(path: Path, audited_args_files: set[Path]) -> tuple[int, list[SourceFinding]]:
    key = path.resolve(strict=False)
    if key in audited_args_files:
        return 0, []
    audited_args_files.add(key)
    return audit_args_file_fragments(path)


def audit_arg_file_refs(
    config_path: Path,
    refs_by_path: Iterable[tuple[str, list[str]]],
    audited_args_files: set[Path],
) -> tuple[int, list[SourceFinding]]:
    fragments_checked = 0
    findings: list[SourceFinding] = []
    for json_path, refs in refs_by_path:
        for ref in refs:
            ref_path = resolve_arg_file_ref(config_path, ref)
            if not ref_path.is_file():
                findings.append(
                    SourceFinding(
                        source=str(config_path),
                        line=1,
                        reason=f"{json_path}: referenced qemu args file not found: {ref}",
                        command=[],
                        text=ref,
                    )
                )
                continue
            checked, ref_findings = audit_args_file_once(ref_path, audited_args_files)
            fragments_checked += checked
            findings.extend(ref_findings)
    return fragments_checked, findings


def audit(paths: Iterable[Path], excludes: set[str]) -> tuple[int, list[SourceFinding]]:
    commands_checked = 0
    findings: list[SourceFinding] = []
    audited_args_files: set[Path] = set()
    for path in iter_source_files(paths, excludes):
        if path.suffix.lower() == ".json":
            fragments_checked, json_findings = audit_json_arg_fragments(path)
            commands_checked += fragments_checked
            findings.extend(json_findings)
            ref_fragments_checked, ref_findings = audit_json_arg_file_refs(path, audited_args_files)
            commands_checked += ref_fragments_checked
            findings.extend(ref_findings)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            fragments_checked, yaml_findings = audit_yaml_arg_fragments(path)
            commands_checked += fragments_checked
            findings.extend(yaml_findings)
            ref_fragments_checked, ref_findings = audit_yaml_arg_file_refs(path, audited_args_files)
            commands_checked += ref_fragments_checked
            findings.extend(ref_findings)
        elif path.suffix.lower() == ".toml":
            fragments_checked, toml_findings = audit_toml_arg_fragments(path)
            commands_checked += fragments_checked
            findings.extend(toml_findings)
            ref_fragments_checked, ref_findings = audit_toml_arg_file_refs(path, audited_args_files)
            commands_checked += ref_fragments_checked
            findings.extend(ref_findings)
        elif path.suffix.lower() == ".args":
            fragments_checked, args_findings = audit_args_file_once(path, audited_args_files)
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
