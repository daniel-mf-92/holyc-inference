#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for block-device policy.

This host-side tool reads benchmark JSON artifacts only. It never launches QEMU.
Benchmark commands should use the canonical local raw IDE image drive and must
not add remote block transports, `-blockdev` graphs, or extra legacy disk media
that can make replay use different storage than the benchmark metadata records.
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
DISK_VALUE_OPTIONS = {"-cdrom", "-fda", "-fdb", "-hda", "-hdb", "-hdc", "-hdd"}
REMOTE_BLOCK_MARKERS = (
    "curl:",
    "ftp://",
    "ftps://",
    "gluster",
    "http://",
    "https://",
    "iscsi://",
    "nbd:",
    "rbd:",
    "ssh://",
    "tcp:",
    "udp:",
)


@dataclass(frozen=True)
class BlockRecord:
    source: str
    status: str
    command_arrays_checked: int
    drive_options: int
    blockdev_options: int
    legacy_disk_options: int
    raw_ide_drives: int
    remote_block_references: int
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


def split_drive_options(value: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for part in value.split(","):
        if "=" not in part:
            continue
        key, item = part.split("=", 1)
        options[key.strip().lower()] = item.strip()
    return options


def option_value(command: list[str], index: int) -> str:
    arg = command[index]
    if "=" in arg:
        return arg.split("=", 1)[1]
    if index + 1 < len(command):
        return command[index + 1]
    return ""


def has_remote_block_marker(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in REMOTE_BLOCK_MARKERS)


def is_raw_ide_drive(value: str) -> bool:
    options = split_drive_options(value)
    return bool(options.get("file")) and options.get("format", "").lower() == "raw" and options.get("if", "").lower() == "ide"


def audit_command(path: Path, json_path: str, command: list[str], findings: list[Finding]) -> tuple[int, int, int, int, int]:
    drive_options = 0
    blockdev_options = 0
    legacy_disk_options = 0
    raw_ide_drives = 0
    remote_block_references = 0

    index = 0
    while index < len(command):
        arg = command[index]
        value = option_value(command, index)
        if arg == "-drive" or arg.startswith("-drive="):
            drive_options += 1
            if is_raw_ide_drive(value):
                raw_ide_drives += 1
            else:
                findings.append(
                    Finding(
                        str(path),
                        json_path,
                        "non_canonical_drive",
                        value,
                        "QEMU benchmark drives must include file=...,format=raw,if=ide",
                    )
                )
            if has_remote_block_marker(value):
                remote_block_references += 1
                findings.append(
                    Finding(str(path), json_path, "remote_block_transport", value, "remote block transports violate the air-gap policy")
                )
            if "=" not in arg:
                index += 2
                continue
        elif arg == "-blockdev" or arg.startswith("-blockdev="):
            blockdev_options += 1
            findings.append(
                Finding(str(path), json_path, "blockdev_graph", value, "`-blockdev` is forbidden for repeatable benchmark replay")
            )
            if has_remote_block_marker(value):
                remote_block_references += 1
                findings.append(
                    Finding(str(path), json_path, "remote_block_transport", value, "remote block transports violate the air-gap policy")
                )
            if "=" not in arg:
                index += 2
                continue
        elif arg in DISK_VALUE_OPTIONS or any(arg.startswith(option + "=") for option in DISK_VALUE_OPTIONS):
            legacy_disk_options += 1
            findings.append(
                Finding(str(path), json_path, "extra_disk_media", f"{arg} {value}".strip(), "legacy disk media options are forbidden in benchmark commands")
            )
            if has_remote_block_marker(value):
                remote_block_references += 1
                findings.append(
                    Finding(str(path), json_path, "remote_block_transport", value, "remote block transports violate the air-gap policy")
                )
            if "=" not in arg:
                index += 2
                continue
        index += 1

    return drive_options, blockdev_options, legacy_disk_options, raw_ide_drives, remote_block_references


def audit_artifact(path: Path) -> tuple[BlockRecord, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), "$", "invalid_artifact", "", error)
        return BlockRecord(str(path), "fail", 0, 0, 0, 0, 0, 0, 1, error), [finding]

    findings: list[Finding] = []
    command_arrays_checked = 0
    drive_options = 0
    blockdev_options = 0
    legacy_disk_options = 0
    raw_ide_drives = 0
    remote_block_references = 0
    for json_path, command in walk_command_arrays(payload):
        command_arrays_checked += 1
        drive_count, blockdev_count, legacy_count, raw_count, remote_count = audit_command(path, json_path, command, findings)
        drive_options += drive_count
        blockdev_options += blockdev_count
        legacy_disk_options += legacy_count
        raw_ide_drives += raw_count
        remote_block_references += remote_count
        if drive_count == 0:
            findings.append(Finding(str(path), json_path, "missing_image_drive", "", "recorded QEMU command has no `-drive file=...,format=raw,if=ide` argument"))

    return (
        BlockRecord(
            source=str(path),
            status="fail" if findings else "pass",
            command_arrays_checked=command_arrays_checked,
            drive_options=drive_options,
            blockdev_options=blockdev_options,
            legacy_disk_options=legacy_disk_options,
            raw_ide_drives=raw_ide_drives,
            remote_block_references=remote_block_references,
            violation_count=len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[BlockRecord], list[Finding]]:
    records: list[BlockRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        record, artifact_findings = audit_artifact(path)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(
            Finding("", "$", "min_artifacts", "", f"checked {len(records)} artifacts, required at least {args.min_artifacts}")
        )
    return records, findings


def write_json(path: Path, records: list[BlockRecord], findings: list[Finding]) -> None:
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "command_arrays_checked": sum(record.command_arrays_checked for record in records),
            "drive_options": sum(record.drive_options for record in records),
            "remote_block_references": sum(record.remote_block_references for record in records),
            "findings": len(findings),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[BlockRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Block Device Policy Audit",
        "",
        f"Artifacts checked: {len(records)}",
        f"Command arrays checked: {sum(record.command_arrays_checked for record in records)}",
        f"Drive options: {sum(record.drive_options for record in records)}",
        f"Remote block references: {sum(record.remote_block_references for record in records)}",
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
        lines.append("All audited QEMU block-device references satisfy the benchmark replay policy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[BlockRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BlockRecord.__dataclass_fields__), lineterminator="\n")
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
            "name": "holyc_qemu_block_device_policy_audit",
            "tests": str(test_count),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_block_device_policy_audit", "name": "all_artifacts"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_block_device_policy_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_block_device_policy_violation",
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
    parser.add_argument("--output-stem", default="qemu_block_device_policy_audit_latest")
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
