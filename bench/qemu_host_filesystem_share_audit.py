#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for host filesystem sharing.

This host-side tool reads benchmark JSON artifacts only. It never launches QEMU.
TempleOS benchmark guests must remain air-gapped and isolated from host
filesystems, so recorded QEMU commands must not include 9p, virtio-fs, SMB, or
other host path passthrough devices.
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
SHARE_OPTIONS = {"-virtfs", "-fsdev", "-smb"}
DEVICE_OPTIONS = {"-device"}
SHARE_DEVICE_MARKERS = (
    "virtio-9p",
    "virtio-fs",
    "vhost-user-fs",
    "9p-pci",
    "9p-device",
)
SHARE_VALUE_MARKERS = (
    "mount_tag=",
    "security_model=",
    "multidevs=",
    "smb=",
    "smbserver=",
)


@dataclass(frozen=True)
class ShareRecord:
    source: str
    status: str
    command_arrays_checked: int
    share_options: int
    share_devices: int
    host_path_markers: int
    violation_count: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    json_path: str
    kind: str
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


def walk_command_arrays(payload: Any, json_path: str = "$") -> Iterable[tuple[str, list[str]]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{json_path}.{key}"
            if key == "command" and isinstance(value, list) and all(isinstance(item, str) for item in value):
                yield child_path, list(value)
            yield from walk_command_arrays(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from walk_command_arrays(value, f"{json_path}[{index}]")


def option_value(command: list[str], index: int) -> str:
    arg = command[index]
    if "=" in arg:
        return arg.split("=", 1)[1]
    if index + 1 < len(command):
        return command[index + 1]
    return ""


def option_matches(arg: str, options: set[str]) -> bool:
    return arg in options or any(arg.startswith(f"{option}=") for option in options)


def contains_marker(value: str, markers: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in markers)


def audit_command(path: Path, json_path: str, command: list[str], findings: list[Finding]) -> tuple[int, int, int]:
    share_options = 0
    share_devices = 0
    host_path_markers = 0
    index = 0
    while index < len(command):
        arg = command[index]
        value = option_value(command, index)
        if option_matches(arg, SHARE_OPTIONS):
            share_options += 1
            findings.append(
                Finding(
                    str(path),
                    json_path,
                    "host_filesystem_share_option",
                    f"{arg} {value}".strip(),
                    "QEMU host filesystem passthrough is forbidden for air-gapped benchmarks",
                )
            )
            if contains_marker(value, SHARE_VALUE_MARKERS):
                host_path_markers += 1
            if "=" not in arg:
                index += 2
                continue
        elif option_matches(arg, DEVICE_OPTIONS):
            if contains_marker(value, SHARE_DEVICE_MARKERS):
                share_devices += 1
                findings.append(
                    Finding(
                        str(path),
                        json_path,
                        "host_filesystem_share_device",
                        f"{arg} {value}".strip(),
                        "virtio/9p filesystem sharing devices are forbidden for air-gapped benchmarks",
                    )
                )
            if contains_marker(value, SHARE_VALUE_MARKERS):
                host_path_markers += 1
            if "=" not in arg:
                index += 2
                continue
        elif contains_marker(value, SHARE_VALUE_MARKERS) or contains_marker(arg, SHARE_VALUE_MARKERS):
            host_path_markers += 1
            marker_value = value if contains_marker(value, SHARE_VALUE_MARKERS) else arg
            findings.append(
                Finding(
                    str(path),
                    json_path,
                    "host_filesystem_share_marker",
                    marker_value,
                    "filesystem-sharing marker found in QEMU command",
                )
            )
            if "=" not in arg:
                index += 2
                continue
        index += 1
    return share_options, share_devices, host_path_markers


def audit_artifact(path: Path) -> tuple[ShareRecord, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), "$", "invalid_artifact", "", error)
        return ShareRecord(str(path), "fail", 0, 0, 0, 0, 1, error), [finding]

    findings: list[Finding] = []
    command_arrays_checked = 0
    share_options = 0
    share_devices = 0
    host_path_markers = 0
    for json_path, command in walk_command_arrays(payload):
        command_arrays_checked += 1
        option_count, device_count, marker_count = audit_command(path, json_path, command, findings)
        share_options += option_count
        share_devices += device_count
        host_path_markers += marker_count

    if command_arrays_checked == 0:
        findings.append(Finding(str(path), "$", "missing_command", "", "artifact contains no QEMU command arrays"))

    return (
        ShareRecord(
            source=str(path),
            status="fail" if findings else "pass",
            command_arrays_checked=command_arrays_checked,
            share_options=share_options,
            share_devices=share_devices,
            host_path_markers=host_path_markers,
            violation_count=len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ShareRecord], list[Finding]]:
    records: list[ShareRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        record, artifact_findings = audit_artifact(path)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(Finding("", "$", "min_artifacts", "", f"checked {len(records)} artifacts, required at least {args.min_artifacts}"))
    return records, findings


def write_json(path: Path, records: list[ShareRecord], findings: list[Finding]) -> None:
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "command_arrays_checked": sum(record.command_arrays_checked for record in records),
            "share_options": sum(record.share_options for record in records),
            "share_devices": sum(record.share_devices for record in records),
            "host_path_markers": sum(record.host_path_markers for record in records),
            "findings": len(findings),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ShareRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Host Filesystem Share Audit",
        "",
        f"Artifacts checked: {len(records)}",
        f"Command arrays checked: {sum(record.command_arrays_checked for record in records)}",
        f"Share options: {sum(record.share_options for record in records)}",
        f"Share devices: {sum(record.share_devices for record in records)}",
        f"Host path markers: {sum(record.host_path_markers for record in records)}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["| Artifact | JSON path | Kind | Value | Detail |", "| --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {json_path} | {kind} | {value} | {detail} |".format(
                    source=finding.source,
                    json_path=finding.json_path.replace("|", "\\|"),
                    kind=finding.kind,
                    value=finding.value.replace("|", "\\|"),
                    detail=finding.detail.replace("|", "\\|"),
                )
            )
    else:
        lines.append("No QEMU host filesystem sharing references were found.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[ShareRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ShareRecord.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    failures_by_source: dict[str, list[Finding]] = {}
    for finding in findings:
        failures_by_source.setdefault(finding.source or "coverage", []).append(finding)
    test_count = max(1, len(failures_by_source) + (0 if findings else 1))
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_host_filesystem_share_audit",
            "tests": str(test_count),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_host_filesystem_share_audit", "name": "all_artifacts"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_host_filesystem_share_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_host_filesystem_share_violation",
                "message": "; ".join(finding.kind for finding in source_findings),
            },
        )
        failure.text = "\n".join(f"{finding.json_path}: {finding.detail}" for finding in source_findings)
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob to use when an input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_host_filesystem_share_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0:
        parser.error("--min-artifacts must be >= 0")

    records, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", records, findings)
    write_markdown(args.output_dir / f"{stem}.md", records, findings)
    write_records_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
