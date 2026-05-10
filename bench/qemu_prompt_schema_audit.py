#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for schema and telemetry integrity.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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


TOP_LEVEL_TYPES: dict[str, type | tuple[type, ...]] = {
    "generated_at": str,
    "status": str,
    "command": list,
    "command_sha256": str,
    "command_airgap": dict,
    "prompt_suite": dict,
    "launch_plan": list,
    "launch_sequence_integrity": dict,
    "suite_summary": dict,
    "phase_summaries": list,
    "benchmarks": list,
}
ROW_TYPES: dict[str, type | tuple[type, ...]] = {
    "benchmark": str,
    "profile": str,
    "model": str,
    "quantization": str,
    "phase": str,
    "launch_index": int,
    "prompt": str,
    "prompt_sha256": str,
    "prompt_bytes": int,
    "iteration": int,
    "timestamp": str,
    "elapsed_us": int,
    "wall_elapsed_us": int,
    "timeout_seconds": (int, float),
    "host_overhead_us": int,
    "returncode": int,
    "timed_out": bool,
    "exit_class": str,
    "command": list,
    "command_sha256": str,
    "command_airgap_ok": bool,
    "command_has_explicit_nic_none": bool,
    "command_has_legacy_net_none": bool,
}
NUMERIC_ROW_FIELDS = (
    "tokens",
    "expected_tokens",
    "elapsed_us",
    "wall_elapsed_us",
    "timeout_seconds",
    "host_overhead_us",
    "tok_per_s",
    "wall_tok_per_s",
    "ttft_us",
    "memory_bytes",
    "prompt_bytes_per_s",
    "wall_prompt_bytes_per_s",
    "us_per_token",
    "wall_us_per_token",
    "serial_output_bytes",
    "serial_output_lines",
)
EXIT_CLASSES = {"ok", "timeout", "launch_error", "nonzero_exit"}


@dataclass(frozen=True)
class ArtifactSchema:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    measured_rows: int
    warmup_rows: int
    planned_measured_launches: int | None
    planned_warmup_launches: int | None
    planned_total_launches: int | None
    ok_rows: int
    failed_rows: int
    command_airgap_ok: bool
    schema_fields_checked: int
    telemetry_fields_checked: int
    error: str = ""


@dataclass(frozen=True)
class SchemaFinding:
    source: str
    row: str
    field: str
    kind: str
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


def type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return "|".join(item.__name__ for item in expected)
    return expected.__name__


def has_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == (int, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def check_typed_fields(
    source: Path,
    row_name: str,
    payload: dict[str, Any],
    fields: dict[str, type | tuple[type, ...]],
) -> list[SchemaFinding]:
    findings: list[SchemaFinding] = []
    for field, expected in fields.items():
        if field not in payload:
            findings.append(SchemaFinding(str(source), row_name, field, "missing_field", "required field is absent"))
            continue
        if not has_type(payload[field], expected):
            findings.append(
                SchemaFinding(
                    str(source),
                    row_name,
                    field,
                    "field_type",
                    f"expected {type_name(expected)}, got {type(payload[field]).__name__}",
                )
            )
    return findings


def check_command_airgap(source: Path, row_name: str, command: Any) -> list[SchemaFinding]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return [SchemaFinding(str(source), row_name, "command", "command_type", "command must be a list of strings")]
    metadata = qemu_prompt_bench.command_airgap_metadata(command)
    findings: list[SchemaFinding] = []
    if metadata["legacy_net_none"]:
        findings.append(SchemaFinding(str(source), row_name, "command", "legacy_net_none", "legacy -net none is disallowed drift"))
    if not metadata["explicit_nic_none"]:
        findings.append(SchemaFinding(str(source), row_name, "command", "missing_nic_none", "QEMU command must include -nic none"))
    for violation in metadata["violations"]:
        findings.append(SchemaFinding(str(source), row_name, "command", "network_arg", str(violation)))
    return findings


def check_row(source: Path, row_name: str, row: Any, top: dict[str, Any], args: argparse.Namespace) -> list[SchemaFinding]:
    if not isinstance(row, dict):
        return [SchemaFinding(str(source), row_name, "", "row_type", "benchmark row must be an object")]
    findings = check_typed_fields(source, row_name, row, ROW_TYPES)
    command = row.get("command")
    findings.extend(check_command_airgap(source, row_name, command))
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        expected_hash = qemu_prompt_bench.command_hash(command)
        if row.get("command_sha256") != expected_hash:
            findings.append(SchemaFinding(str(source), row_name, "command_sha256", "command_hash", "row command_sha256 does not match command"))
    if row.get("command_airgap_ok") is not True:
        findings.append(SchemaFinding(str(source), row_name, "command_airgap_ok", "airgap_telemetry", "row command_airgap_ok must be true"))
    if row.get("command_has_explicit_nic_none") is not True:
        findings.append(SchemaFinding(str(source), row_name, "command_has_explicit_nic_none", "airgap_telemetry", "row must record explicit -nic none"))
    if row.get("command_has_legacy_net_none") is not False:
        findings.append(SchemaFinding(str(source), row_name, "command_has_legacy_net_none", "airgap_telemetry", "row must reject legacy -net none"))
    if row.get("exit_class") not in EXIT_CLASSES:
        findings.append(SchemaFinding(str(source), row_name, "exit_class", "exit_class", f"unexpected exit_class {row.get('exit_class')!r}"))
    if row.get("phase") not in {"warmup", "measured"}:
        findings.append(SchemaFinding(str(source), row_name, "phase", "phase", f"unexpected phase {row.get('phase')!r}"))
    for field in ("profile", "model", "quantization"):
        if top.get(field) and row.get(field) != top.get(field):
            findings.append(SchemaFinding(str(source), row_name, field, "top_level_mismatch", f"row {field} does not match artifact {field}"))
    for field in NUMERIC_ROW_FIELDS:
        value = row.get(field)
        if value is not None and not finite_number(value):
            findings.append(SchemaFinding(str(source), row_name, field, "nonfinite_number", "numeric telemetry must be finite"))
    for field in ("elapsed_us", "wall_elapsed_us", "prompt_bytes"):
        value = row.get(field)
        if isinstance(value, int) and value < 0:
            findings.append(SchemaFinding(str(source), row_name, field, "negative_value", "value must be non-negative"))
    if finite_number(row.get("timeout_seconds")) and float(row["timeout_seconds"]) <= 0:
        findings.append(SchemaFinding(str(source), row_name, "timeout_seconds", "nonpositive_timeout", "timeout_seconds must be positive"))
    if row.get("exit_class") == "ok":
        for field in args.require_ok_telemetry:
            if row.get(field) is None:
                findings.append(SchemaFinding(str(source), row_name, field, "missing_ok_telemetry", "ok row is missing required telemetry"))
    return findings


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactSchema, list[SchemaFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        artifact = ArtifactSchema(str(path), "fail", "", "", "", 0, 0, None, None, None, 0, 0, False, 0, 0, error)
        return artifact, [SchemaFinding(str(path), "", "", "invalid_artifact", error)]

    findings = check_typed_fields(path, "artifact", payload, TOP_LEVEL_TYPES)
    measured = payload.get("benchmarks") if isinstance(payload.get("benchmarks"), list) else []
    warmups = payload.get("warmups") if isinstance(payload.get("warmups"), list) else []
    top_command = payload.get("command")
    command_airgap_ok = False
    findings.extend(check_command_airgap(path, "artifact", top_command))
    if isinstance(top_command, list) and all(isinstance(item, str) for item in top_command):
        command_airgap_ok = qemu_prompt_bench.command_airgap_metadata(top_command)["ok"]
        if payload.get("command_sha256") != qemu_prompt_bench.command_hash(top_command):
            findings.append(SchemaFinding(str(path), "artifact", "command_sha256", "command_hash", "artifact command_sha256 does not match command"))
    command_airgap = payload.get("command_airgap")
    if isinstance(command_airgap, dict):
        if command_airgap.get("ok") is not True:
            findings.append(SchemaFinding(str(path), "artifact", "command_airgap.ok", "airgap_telemetry", "artifact command_airgap.ok must be true"))
        if command_airgap.get("legacy_net_none") is not False:
            findings.append(SchemaFinding(str(path), "artifact", "command_airgap.has_legacy_net_none", "airgap_telemetry", "artifact must reject legacy -net none"))
        if command_airgap.get("explicit_nic_none") is not True:
            findings.append(SchemaFinding(str(path), "artifact", "command_airgap.has_explicit_nic_none", "airgap_telemetry", "artifact must record explicit -nic none"))
    for index, row in enumerate(warmups):
        findings.extend(check_row(path, f"warmups[{index}]", row, payload, args))
    for index, row in enumerate(measured):
        findings.extend(check_row(path, f"benchmarks[{index}]", row, payload, args))

    planned_measured = int_or_none(payload.get("planned_measured_launches"))
    planned_warmup = int_or_none(payload.get("planned_warmup_launches"))
    planned_total = int_or_none(payload.get("planned_total_launches"))
    if planned_measured is not None and planned_measured != len(measured):
        findings.append(SchemaFinding(str(path), "artifact", "planned_measured_launches", "launch_count", f"planned {planned_measured} does not match {len(measured)} measured rows"))
    if planned_warmup is not None and planned_warmup != len(warmups):
        findings.append(SchemaFinding(str(path), "artifact", "planned_warmup_launches", "launch_count", f"planned {planned_warmup} does not match {len(warmups)} warmup rows"))
    if planned_total is not None and planned_total != len(measured) + len(warmups):
        findings.append(SchemaFinding(str(path), "artifact", "planned_total_launches", "launch_count", f"planned {planned_total} does not match emitted rows"))
    suite_summary = payload.get("suite_summary")
    if isinstance(suite_summary, dict) and int_or_none(suite_summary.get("runs")) not in (None, len(measured)):
        findings.append(SchemaFinding(str(path), "artifact", "suite_summary.runs", "summary_count", "suite_summary.runs does not match measured rows"))
    if len(measured) < args.min_measured_rows:
        findings.append(SchemaFinding(str(path), "artifact", "benchmarks", "min_measured_rows", f"measured rows {len(measured)} below minimum {args.min_measured_rows}"))

    ok_rows = sum(1 for row in measured if isinstance(row, dict) and row.get("exit_class") == "ok")
    failed_rows = len(measured) - ok_rows
    if args.require_success and failed_rows:
        findings.append(SchemaFinding(str(path), "artifact", "benchmarks", "failed_rows", f"{failed_rows} measured row(s) are not ok"))
    artifact = ArtifactSchema(
        source=str(path),
        status="fail" if findings else "pass",
        profile=str(payload.get("profile") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        measured_rows=len(measured),
        warmup_rows=len(warmups),
        planned_measured_launches=planned_measured,
        planned_warmup_launches=planned_warmup,
        planned_total_launches=planned_total,
        ok_rows=ok_rows,
        failed_rows=failed_rows,
        command_airgap_ok=command_airgap_ok,
        schema_fields_checked=len(TOP_LEVEL_TYPES) + (len(measured) + len(warmups)) * len(ROW_TYPES),
        telemetry_fields_checked=(len(measured) + len(warmups)) * len(NUMERIC_ROW_FIELDS),
    )
    return artifact, findings


def build_report(artifacts: list[ArtifactSchema], findings: list[SchemaFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "measured_rows": sum(artifact.measured_rows for artifact in artifacts),
            "warmup_rows": sum(artifact.warmup_rows for artifact in artifacts),
            "ok_rows": sum(artifact.ok_rows for artifact in artifacts),
            "failed_rows": sum(artifact.failed_rows for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Any], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Prompt Schema Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Measured rows: {summary['measured_rows']}",
        f"- Warmup rows: {summary['warmup_rows']}",
        f"- OK rows: {summary['ok_rows']}",
        f"- Failed rows: {summary['failed_rows']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Profile | Model | Quantization | Measured rows | Warmup rows | OK rows | Air-gap OK |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {profile} | {model} | {quantization} | {measured_rows} | {warmup_rows} | {ok_rows} | {command_airgap_ok} |".format(
                **artifact
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            prefix = f"{finding['source']} {finding['row']} {finding['field']}".strip()
            lines.append(f"- {finding['kind']}: {prefix} {finding['detail']}".strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[SchemaFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_schema_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            name = f"{finding.kind}:{finding.source}:{finding.row}:{finding.field}".rstrip(":")
            case = ET.SubElement(suite, "testcase", {"name": name})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_prompt_schema"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*_latest.json"], help="Directory glob for benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_schema_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-measured-rows", type=int, default=1)
    parser.add_argument("--require-success", action="store_true", help="Fail when any measured row is not exit_class=ok")
    parser.add_argument(
        "--require-ok-telemetry",
        action="append",
        default=["tokens", "tok_per_s", "wall_tok_per_s"],
        help="Telemetry field required on each ok row; repeatable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: list[ArtifactSchema] = []
    findings: list[SchemaFinding] = []
    for path in sorted(iter_input_files(args.inputs, args.pattern)):
        artifact, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    if len(artifacts) < args.min_artifacts:
        findings.append(
            SchemaFinding(
                "",
                "",
                "",
                "min_artifacts",
                f"benchmark artifact count {len(artifacts)} is below minimum {args.min_artifacts}",
            )
        )
    report = build_report(artifacts, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", artifacts, list(ArtifactSchema.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", findings, list(SchemaFinding.__dataclass_fields__))
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
