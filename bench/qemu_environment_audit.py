#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for host environment provenance.

This host-side tool reads saved benchmark JSON artifacts only. It never launches
QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
REQUIRED_ENV_FIELDS = ("platform", "machine", "python", "cpu_count", "qemu_bin")


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    commit: str
    profile: str
    model: str
    quantization: str
    qemu_bin: str
    qemu_path: str
    qemu_version: str
    platform: str
    machine: str
    python: str
    cpu_count: int | None
    command_sha256: str
    command_airgap_ok: bool
    rows: int
    row_missing_command_provenance: int
    row_commit_mismatches: int
    row_command_hash_mismatches: int
    row_command_airgap_mismatches: int


@dataclass(frozen=True)
class Finding:
    source: str
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


def text_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def benchmark_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("benchmarks")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord | None, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return None, [Finding(str(path), "error", "load_error", "", error)]

    findings: list[Finding] = []
    environment = payload.get("environment")
    if not isinstance(environment, dict):
        environment = {}
        findings.append(Finding(str(path), "error", "missing_environment", "environment", "environment object is absent"))

    for field in REQUIRED_ENV_FIELDS:
        value = environment.get(field)
        if value in (None, ""):
            findings.append(Finding(str(path), "error", "missing_environment_field", field, f"{field} is absent"))

    cpu_count = int_value(environment.get("cpu_count"))
    if cpu_count is None or cpu_count <= 0:
        findings.append(Finding(str(path), "error", "invalid_cpu_count", "cpu_count", "cpu_count must be a positive integer"))

    if args.require_qemu_path and not text_value(environment.get("qemu_path")):
        findings.append(Finding(str(path), "error", "missing_qemu_path", "qemu_path", "qemu_path is required"))
    if args.require_qemu_version and not text_value(environment.get("qemu_version")):
        findings.append(Finding(str(path), "error", "missing_qemu_version", "qemu_version", "qemu_version is required"))

    command = payload.get("command")
    command_sha256 = text_value(payload.get("command_sha256"))
    command_airgap = payload.get("command_airgap") if isinstance(payload.get("command_airgap"), dict) else {}
    command_airgap_ok = command_airgap.get("ok") is True
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        findings.append(Finding(str(path), "error", "command_type", "command", "command must be a list of strings"))
    else:
        expected_hash = qemu_prompt_bench.command_hash(command)
        if command_sha256 != expected_hash:
            findings.append(Finding(str(path), "error", "command_hash", "command_sha256", "top-level command hash mismatch"))
        metadata = qemu_prompt_bench.command_airgap_metadata(command)
        if not metadata["ok"]:
            findings.append(
                Finding(str(path), "error", "command_airgap", "command", "; ".join(metadata["violations"]))
            )
        if command_airgap_ok != metadata["ok"]:
            findings.append(
                Finding(str(path), "error", "command_airgap_drift", "command_airgap", "recorded air-gap metadata drift")
            )

    commit = text_value(payload.get("commit"))
    if not commit:
        findings.append(Finding(str(path), "error", "missing_commit", "commit", "top-level commit is absent"))

    rows = benchmark_rows(payload)
    row_missing_command_provenance = 0
    row_commit_mismatches = 0
    row_command_hash_mismatches = 0
    row_command_airgap_mismatches = 0
    for index, row in enumerate(rows, 1):
        if commit and text_value(row.get("commit")) and row.get("commit") != commit:
            row_commit_mismatches += 1
            findings.append(
                Finding(str(path), "error", "row_commit_drift", f"benchmarks[{index}].commit", "row commit differs from artifact commit")
            )
        row_command = row.get("command")
        if not isinstance(row_command, list) or not all(isinstance(item, str) for item in row_command):
            if args.require_row_command_provenance:
                row_missing_command_provenance += 1
                findings.append(
                    Finding(
                        str(path),
                        "error",
                        "row_command_missing",
                        f"benchmarks[{index}].command",
                        "row command list is required for command provenance gating",
                    )
                )
            continue

        expected_hash = qemu_prompt_bench.command_hash(row_command)
        if row.get("command_sha256") != expected_hash:
            row_command_hash_mismatches += 1
            findings.append(
                Finding(
                    str(path),
                    "error",
                    "row_command_hash",
                    f"benchmarks[{index}].command_sha256",
                    "row command hash mismatch",
                )
            )

        row_airgap = qemu_prompt_bench.command_airgap_metadata(row_command)
        if args.require_row_command_provenance:
            required_fields = (
                "command_sha256",
                "command_airgap_ok",
                "command_has_explicit_nic_none",
                "command_has_legacy_net_none",
                "command_airgap_violations",
            )
            for field in required_fields:
                if field not in row:
                    row_missing_command_provenance += 1
                    findings.append(
                        Finding(
                            str(path),
                            "error",
                            "row_command_provenance_missing",
                            f"benchmarks[{index}].{field}",
                            f"{field} is required for command provenance gating",
                        )
                    )

        row_expected_values = {
            "command_airgap_ok": row_airgap["ok"],
            "command_has_explicit_nic_none": row_airgap["explicit_nic_none"],
            "command_has_legacy_net_none": row_airgap["legacy_net_none"],
            "command_airgap_violations": row_airgap["violations"],
        }
        for field, expected in row_expected_values.items():
            if field in row and row.get(field) != expected:
                row_command_airgap_mismatches += 1
                findings.append(
                    Finding(
                        str(path),
                        "error",
                        "row_command_airgap_drift",
                        f"benchmarks[{index}].{field}",
                        f"row {field} drifted from command-derived air-gap metadata",
                    )
                )

    record = ArtifactRecord(
        source=str(path),
        status=text_value(payload.get("status")) or "unknown",
        commit=commit,
        profile=text_value(payload.get("profile")),
        model=text_value(payload.get("model")),
        quantization=text_value(payload.get("quantization")),
        qemu_bin=text_value(environment.get("qemu_bin")),
        qemu_path=text_value(environment.get("qemu_path")),
        qemu_version=text_value(environment.get("qemu_version")),
        platform=text_value(environment.get("platform")),
        machine=text_value(environment.get("machine")),
        python=text_value(environment.get("python")),
        cpu_count=cpu_count,
        command_sha256=command_sha256,
        command_airgap_ok=command_airgap_ok,
        rows=len(rows),
        row_missing_command_provenance=row_missing_command_provenance,
        row_commit_mismatches=row_commit_mismatches,
        row_command_hash_mismatches=row_command_hash_mismatches,
        row_command_airgap_mismatches=row_command_airgap_mismatches,
    )
    return record, findings


def environment_signature(record: ArtifactRecord) -> tuple[str, str, str, int | None, str, str, str]:
    return (
        record.platform,
        record.machine,
        record.python,
        record.cpu_count,
        record.qemu_bin,
        record.qemu_path,
        record.qemu_version,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[Finding]]:
    records: list[ArtifactRecord] = []
    findings: list[Finding] = []
    for path in sorted(iter_input_files(paths, args.pattern)):
        record, artifact_findings = audit_artifact(path, args)
        if record is not None:
            records.append(record)
        findings.extend(artifact_findings)

    if len(records) < args.min_artifacts:
        findings.append(
            Finding(
                "-",
                "error",
                "min_artifacts",
                "artifacts",
                f"found {len(records)} artifact(s), expected at least {args.min_artifacts}",
            )
        )

    if args.fail_on_environment_drift and records:
        signatures = {environment_signature(record) for record in records}
        if len(signatures) > 1:
            findings.append(
                Finding(
                    "-",
                    "error",
                    "environment_drift",
                    "environment",
                    f"found {len(signatures)} distinct host environment signatures",
                )
            )

    return records, findings


def summary(records: list[ArtifactRecord], findings: list[Finding]) -> dict[str, Any]:
    return {
        "artifacts": len(records),
        "findings": len(findings),
        "statuses": sorted({record.status for record in records}),
        "commits": sorted({record.commit for record in records if record.commit}),
        "environment_signatures": len({environment_signature(record) for record in records}),
        "total_rows": sum(record.rows for record in records),
        "artifacts_with_airgap_ok": sum(1 for record in records if record.command_airgap_ok),
        "row_missing_command_provenance": sum(record.row_missing_command_provenance for record in records),
        "row_commit_mismatches": sum(record.row_commit_mismatches for record in records),
        "row_command_hash_mismatches": sum(record.row_command_hash_mismatches for record in records),
        "row_command_airgap_mismatches": sum(record.row_command_airgap_mismatches for record in records),
    }


def write_json(path: Path, records: list[ArtifactRecord], findings: list[Finding]) -> None:
    status = "pass" if not findings else "fail"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": summary(records, findings),
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[ArtifactRecord], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Environment Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Artifacts: {len(records)}",
        f"Findings: {len(findings)}",
        "",
        "## Artifacts",
        "",
        "| Source | Commit | Profile | Model | Quantization | QEMU | Rows | Air-gap OK |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for record in records:
        qemu = record.qemu_version or record.qemu_path or record.qemu_bin
        lines.append(
            f"| {record.source} | {record.commit or '-'} | {record.profile or '-'} | "
            f"{record.model or '-'} | {record.quantization or '-'} | {qemu or '-'} | "
            f"{record.rows} | {record.command_airgap_ok} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Severity | Kind | Field | Detail |", "| --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                f"| {finding.source} | {finding.severity} | {finding.kind} | {finding.field or '-'} | {finding.detail} |"
            )
    else:
        lines.append("No environment provenance findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[ArtifactRecord]) -> None:
    fieldnames = list(ArtifactRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_environment_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "environment_provenance"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} environment provenance finding(s)"})
        failure.text = "\n".join(f"{finding.source}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_environment_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--require-qemu-path", action="store_true")
    parser.add_argument("--require-qemu-version", action="store_true")
    parser.add_argument("--require-row-command-provenance", action="store_true")
    parser.add_argument("--fail-on-environment-drift", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0:
        parser.error("--min-artifacts must be >= 0")
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)

    records, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", records, findings)
    write_markdown(args.output_dir / f"{stem}.md", records, findings)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
