#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for air-gapped local references.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU. It rejects remote URLs, network share references, scp-like
remote paths, and networking command arguments in fields that describe QEMU
commands or benchmark input/output resources.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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
REMOTE_SCHEMES = ("ftp://", "git://", "gs://", "http://", "https://", "s3://", "ssh://", "tcp://", "udp://")
KEY_MARKERS = (
    "arg",
    "bin",
    "bundle",
    "checkpoint",
    "command",
    "dataset",
    "dir",
    "file",
    "image",
    "input",
    "model",
    "output",
    "path",
    "prompt",
    "source",
    "vocab",
    "weight",
)
SKIP_KEYS = {"stdout_tail", "stderr_tail", "serial_tail", "stdout", "stderr", "serial_output"}
SCP_REMOTE_RE = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:.+")


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    checked_strings: int
    command_arrays_checked: int
    remote_reference_count: int
    command_airgap_violation_count: int
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


def should_scan_string(json_path: str, *, scan_all_strings: bool) -> bool:
    if scan_all_strings:
        return True
    parts = [part.strip("[]").lower() for part in json_path.split(".")]
    if any(part in SKIP_KEYS for part in parts):
        return False
    return any(marker in part for part in parts for marker in KEY_MARKERS)


def remote_reference_kind(value: str) -> str | None:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered.startswith(REMOTE_SCHEMES):
        return "remote_uri"
    if lowered.startswith("\\\\") or lowered.startswith("//"):
        return "network_share"
    if SCP_REMOTE_RE.match(stripped):
        return "scp_remote_path"
    return None


def stringify(value: str, *, limit: int = 240) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def walk_strings(payload: Any, json_path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield from walk_strings(value, f"{json_path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from walk_strings(value, f"{json_path}[{index}]")
    elif isinstance(payload, str):
        yield json_path, payload


def walk_command_arrays(payload: Any, json_path: str = "$") -> Iterable[tuple[str, list[str]]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{json_path}.{key}"
            if key == "command" and isinstance(value, list) and all(isinstance(item, str) for item in value):
                yield child_path, value
            yield from walk_command_arrays(value, child_path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from walk_command_arrays(value, f"{json_path}[{index}]")


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), "$", "invalid_artifact", "", error)
        return ArtifactRecord(str(path), "fail", 0, 0, 0, 0, error), [finding]

    findings: list[Finding] = []
    checked_strings = 0
    command_arrays_checked = 0
    command_airgap_violations = 0

    for json_path, value in walk_strings(payload):
        if not should_scan_string(json_path, scan_all_strings=args.scan_all_strings):
            continue
        checked_strings += 1
        kind = remote_reference_kind(value)
        if kind:
            findings.append(
                Finding(
                    str(path),
                    json_path,
                    kind,
                    stringify(value),
                    "benchmark artifacts must reference local host files and offline metadata only",
                )
            )

    for json_path, command in walk_command_arrays(payload):
        command_arrays_checked += 1
        metadata = qemu_prompt_bench.command_airgap_metadata(command)
        if not metadata["explicit_nic_none"]:
            command_airgap_violations += 1
            findings.append(Finding(str(path), json_path, "missing_nic_none", "", "QEMU command must include `-nic none`"))
        if metadata["legacy_net_none"]:
            command_airgap_violations += 1
            findings.append(Finding(str(path), json_path, "legacy_net_none", "", "legacy `-net none` is disallowed drift"))
        for violation in metadata["violations"]:
            command_airgap_violations += 1
            findings.append(Finding(str(path), json_path, "command_airgap_violation", "", str(violation)))

    return (
        ArtifactRecord(
            source=str(path),
            status="fail" if findings else "pass",
            checked_strings=checked_strings,
            command_arrays_checked=command_arrays_checked,
            remote_reference_count=sum(1 for finding in findings if finding.kind in {"remote_uri", "network_share", "scp_remote_path"}),
            command_airgap_violation_count=command_airgap_violations,
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[Finding]]:
    records: list[ArtifactRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, args.pattern):
        record, artifact_findings = audit_artifact(path, args)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(
            Finding(
                "",
                "$",
                "min_artifacts",
                "",
                f"checked {len(records)} artifacts, required at least {args.min_artifacts}",
            )
        )
    return records, findings


def write_json(path: Path, records: list[ArtifactRecord], findings: list[Finding]) -> None:
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "checked_strings": sum(record.checked_strings for record in records),
            "command_arrays_checked": sum(record.command_arrays_checked for record in records),
            "remote_references": sum(record.remote_reference_count for record in records),
            "command_airgap_violations": sum(record.command_airgap_violation_count for record in records),
            "findings": len(findings),
        },
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ArtifactRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Artifact Reference Audit",
        "",
        f"Artifacts checked: {len(records)}",
        f"Strings checked: {sum(record.checked_strings for record in records)}",
        f"Command arrays checked: {sum(record.command_arrays_checked for record in records)}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["| Artifact | JSON path | Kind | Detail |", "| --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {json_path} | {kind} | {detail} |".format(
                    source=finding.source,
                    json_path=finding.json_path.replace("|", "\\|"),
                    kind=finding.kind,
                    detail=finding.detail.replace("|", "\\|"),
                )
            )
    else:
        lines.append("All audited artifact references are local/offline and all command arrays preserve the QEMU air-gap policy.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[ArtifactRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArtifactRecord.__dataclass_fields__), lineterminator="\n")
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
            "name": "holyc_qemu_artifact_reference_audit",
            "tests": str(test_count),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_artifact_reference_audit", "name": "all_artifacts"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_artifact_reference_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_artifact_reference_violation",
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
    parser.add_argument("--output-stem", default="qemu_artifact_reference_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--scan-all-strings", action="store_true", help="also scan non-resource text fields such as output tails")
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
