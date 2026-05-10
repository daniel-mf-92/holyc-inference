#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for image/drive reference drift.

This host-side tool reads existing qemu_prompt_bench JSON artifacts only. It
does not launch QEMU. It verifies that the artifact's declared `image.path`
matches disk image references embedded in recorded QEMU command arrays, so
benchmark replay cannot silently use a different TempleOS image than the
metadata describes.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
DISK_VALUE_OPTIONS = {"-cdrom", "-fda", "-fdb", "-hda", "-hdb", "-hdc", "-hdd"}


@dataclass(frozen=True)
class ImageRecord:
    source: str
    status: str
    image_path: str
    image_exists: bool | None
    image_size_bytes: int | None
    image_sha256: str | None
    command_arrays_checked: int
    drive_references: int
    distinct_drive_paths: int
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


def drive_paths_from_command(command: list[str]) -> list[str]:
    paths: list[str] = []
    index = 0
    while index < len(command):
        arg = command[index]
        next_arg = command[index + 1] if index + 1 < len(command) else ""
        if arg == "-drive" and next_arg:
            drive_file = split_drive_options(next_arg).get("file")
            if drive_file:
                paths.append(drive_file)
            index += 2
            continue
        if arg.startswith("-drive="):
            drive_file = split_drive_options(arg.split("=", 1)[1]).get("file")
            if drive_file:
                paths.append(drive_file)
        elif arg in DISK_VALUE_OPTIONS and next_arg:
            paths.append(next_arg)
            index += 2
            continue
        elif any(arg.startswith(option + "=") for option in DISK_VALUE_OPTIONS):
            paths.append(arg.split("=", 1)[1])
        index += 1
    return paths


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def bool_value(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def image_metadata(payload: dict[str, Any]) -> tuple[str, bool | None, int | None, str | None]:
    image = payload.get("image")
    if not isinstance(image, dict):
        return "", None, None, None
    return (
        text_value(image.get("path")),
        bool_value(image.get("exists")),
        int_value(image.get("size_bytes")),
        text_value(image.get("sha256")) or None,
    )


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ImageRecord, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), "$", "invalid_artifact", "", error)
        return ImageRecord(str(path), "fail", "", None, None, None, 0, 0, 0, 1, error), [finding]

    image_path, image_exists, image_size_bytes, image_sha256 = image_metadata(payload)
    findings: list[Finding] = []
    if not image_path:
        findings.append(Finding(str(path), "$.image.path", "missing_image_path", "", "artifact must declare image.path"))
    if args.require_existing_image and image_exists is not True:
        findings.append(
            Finding(str(path), "$.image.exists", "image_not_marked_existing", str(image_exists), "image.exists must be true")
        )
    if args.require_image_hash and not image_sha256:
        findings.append(Finding(str(path), "$.image.sha256", "missing_image_hash", "", "image.sha256 is required"))
    if args.require_image_size and image_size_bytes is None:
        findings.append(Finding(str(path), "$.image.size_bytes", "missing_image_size", "", "image.size_bytes is required"))

    command_arrays_checked = 0
    drive_refs: list[tuple[str, str]] = []
    for json_path, command in walk_command_arrays(payload):
        command_arrays_checked += 1
        command_drive_paths = drive_paths_from_command(command)
        if not command_drive_paths and args.require_drive_reference:
            findings.append(
                Finding(str(path), json_path, "missing_drive_reference", "", "recorded QEMU command has no disk image argument")
            )
        for drive_path in command_drive_paths:
            drive_refs.append((json_path, drive_path))
            if image_path and drive_path != image_path:
                findings.append(
                    Finding(
                        str(path),
                        json_path,
                        "drive_image_mismatch",
                        drive_path,
                        f"drive image does not match artifact image.path {image_path!r}",
                    )
                )

    distinct_drive_paths = {drive_path for _, drive_path in drive_refs}
    if args.require_single_drive_path and len(distinct_drive_paths) > 1:
        findings.append(
            Finding(
                str(path),
                "$",
                "multiple_drive_paths",
                ";".join(sorted(distinct_drive_paths)),
                "recorded commands reference more than one distinct disk image path",
            )
        )

    return (
        ImageRecord(
            source=str(path),
            status="fail" if findings else "pass",
            image_path=image_path,
            image_exists=image_exists,
            image_size_bytes=image_size_bytes,
            image_sha256=image_sha256,
            command_arrays_checked=command_arrays_checked,
            drive_references=len(drive_refs),
            distinct_drive_paths=len(distinct_drive_paths),
            violation_count=len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ImageRecord], list[Finding]]:
    records: list[ImageRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        record, artifact_findings = audit_artifact(path, args)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(
            Finding("", "$", "min_artifacts", "", f"checked {len(records)} artifacts, required at least {args.min_artifacts}")
        )
    return records, findings


def write_json(path: Path, records: list[ImageRecord], findings: list[Finding]) -> None:
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "command_arrays_checked": sum(record.command_arrays_checked for record in records),
            "drive_references": sum(record.drive_references for record in records),
            "findings": len(findings),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ImageRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Image Reference Audit",
        "",
        f"Artifacts checked: {len(records)}",
        f"Command arrays checked: {sum(record.command_arrays_checked for record in records)}",
        f"Drive references: {sum(record.drive_references for record in records)}",
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
        lines.append("All audited QEMU command image references match artifact image metadata.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[ImageRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ImageRecord.__dataclass_fields__), lineterminator="\n")
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
            "name": "holyc_qemu_image_reference_audit",
            "tests": str(test_count),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_image_reference_audit", "name": "all_artifacts"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_image_reference_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_image_reference_violation",
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
    parser.add_argument("--output-stem", default="qemu_image_reference_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--require-drive-reference", action="store_true", help="fail commands that do not record a disk image argument")
    parser.add_argument("--require-existing-image", action="store_true", help="fail unless image.exists is true")
    parser.add_argument("--require-image-hash", action="store_true", help="fail unless image.sha256 is populated")
    parser.add_argument("--require-image-size", action="store_true", help="fail unless image.size_bytes is populated")
    parser.add_argument("--require-single-drive-path", action="store_true", help="fail artifacts whose command arrays reference multiple disk paths")
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
