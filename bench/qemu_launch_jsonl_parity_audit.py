#!/usr/bin/env python3
"""Audit QEMU prompt benchmark JSON artifacts against launch JSONL sidecars.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench_latest.json",)
DEFAULT_COMPARE_FIELDS = (
    "benchmark",
    "profile",
    "model",
    "quantization",
    "phase",
    "launch_index",
    "iteration",
    "prompt",
    "prompt_sha256",
    "tokens",
    "elapsed_us",
    "wall_elapsed_us",
    "returncode",
    "timed_out",
    "exit_class",
    "command_sha256",
    "command_airgap_ok",
    "command_has_explicit_nic_none",
    "command_has_legacy_net_none",
    "launch_plan_sha256",
    "expected_launch_sequence_sha256",
    "observed_launch_sequence_sha256",
)


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    jsonl_path: str
    json_rows: int
    jsonl_rows: int
    compared_fields: int
    compared_values: int
    findings: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    jsonl_path: str
    row: str
    field: str
    kind: str
    json_value: str
    jsonl_value: str
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


def load_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return None, str(exc)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"line {line_number}: invalid json: {exc}"
        if not isinstance(row, dict):
            return None, f"line {line_number}: row must be a JSON object"
        rows.append(row)
    return rows, ""


def launch_jsonl_path(source: Path) -> Path:
    return source.with_name(f"{source.stem.replace('_latest', '')}_launches_latest.jsonl")


def launch_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top_level = {
        "generated_at": payload.get("generated_at"),
        "status": payload.get("status"),
        "command_sha256": payload.get("command_sha256"),
        "command_airgap_ok": (payload.get("command_airgap") or {}).get("ok"),
        "command_has_explicit_nic_none": (payload.get("command_airgap") or {}).get("explicit_nic_none"),
        "command_has_legacy_net_none": (payload.get("command_airgap") or {}).get("legacy_net_none"),
        "command_airgap_violations": (payload.get("command_airgap") or {}).get("violations") or [],
        "launch_plan_sha256": payload.get("launch_plan_sha256"),
        "expected_launch_sequence_sha256": payload.get("expected_launch_sequence_sha256"),
        "observed_launch_sequence_sha256": payload.get("observed_launch_sequence_sha256"),
        "prompt_suite_sha256": (payload.get("prompt_suite") or {}).get("suite_sha256"),
        "profile": payload.get("profile"),
        "model": payload.get("model"),
        "quantization": payload.get("quantization"),
        "commit": payload.get("commit"),
    }
    for key in ("warmups", "benchmarks"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            if isinstance(row, dict):
                merged = dict(top_level)
                merged.update(row)
                rows.append(merged)
    return sorted(rows, key=lambda item: int(item.get("launch_index") or 0))


def normalize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def compare_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord, list[Finding]]:
    payload, error = load_json_object(path)
    sidecar = launch_jsonl_path(path)
    if payload is None:
        finding = Finding(str(path), str(sidecar), "artifact", "", "load_error", "", "", error)
        return ArtifactRecord(str(path), "fail", str(sidecar), 0, 0, 0, 0, 1, error), [finding]

    json_rows = launch_rows(payload)
    if not sidecar.exists():
        detail = "launch JSONL sidecar is missing"
        finding = Finding(str(path), str(sidecar), "jsonl", "", "missing_jsonl", "", "", detail)
        return ArtifactRecord(str(path), "fail", str(sidecar), len(json_rows), 0, 0, 0, 1, detail), [finding]

    jsonl_rows, jsonl_error = load_jsonl_rows(sidecar)
    if jsonl_rows is None:
        finding = Finding(str(path), str(sidecar), "jsonl", "", "load_error", "", "", jsonl_error)
        return ArtifactRecord(str(path), "fail", str(sidecar), len(json_rows), 0, 0, 0, 1, jsonl_error), [finding]

    findings: list[Finding] = []
    if len(json_rows) != len(jsonl_rows):
        findings.append(
            Finding(
                str(path),
                str(sidecar),
                "artifact",
                "row_count",
                "row_count_mismatch",
                str(len(json_rows)),
                str(len(jsonl_rows)),
                "JSON warmup+benchmark launch row count differs from launch JSONL row count",
            )
        )

    compared_values = 0
    fields = list(args.compare_field)
    for index, (json_row, jsonl_row) in enumerate(zip(json_rows, jsonl_rows), 1):
        for field in fields:
            compared_values += 1
            json_value = normalize(json_row.get(field))
            jsonl_value = normalize(jsonl_row.get(field))
            if json_value != jsonl_value:
                findings.append(
                    Finding(
                        str(path),
                        str(sidecar),
                        str(index),
                        field,
                        "value_mismatch",
                        json_value,
                        jsonl_value,
                        "launch JSONL value differs from canonical JSON launch row",
                    )
                )

    if len(json_rows) < args.min_rows:
        findings.append(
            Finding(
                str(path),
                str(sidecar),
                "artifact",
                "json_rows",
                "min_rows",
                str(len(json_rows)),
                str(args.min_rows),
                "artifact has fewer warmup+benchmark rows than required",
            )
        )

    status = "fail" if findings else "pass"
    return (
        ArtifactRecord(
            str(path),
            status,
            str(sidecar),
            len(json_rows),
            len(jsonl_rows),
            len(fields),
            compared_values,
            len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[Finding]]:
    records: list[ArtifactRecord] = []
    findings: list[Finding] = []
    for path in iter_input_files(paths, list(args.pattern)):
        record, artifact_findings = compare_artifact(path, args)
        records.append(record)
        findings.extend(artifact_findings)
    if len(records) < args.min_artifacts:
        findings.append(
            Finding(
                "",
                "",
                "artifact",
                "artifacts",
                "min_artifacts",
                str(len(records)),
                str(args.min_artifacts),
                "fewer benchmark artifacts found than required",
            )
        )
    return records, findings


def write_json_report(records: list[ArtifactRecord], findings: list[Finding], args: argparse.Namespace, path: Path) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(records),
            "passed": sum(1 for record in records if record.status == "pass"),
            "failed": sum(1 for record in records if record.status == "fail"),
            "json_rows": sum(record.json_rows for record in records),
            "jsonl_rows": sum(record.jsonl_rows for record in records),
            "findings": len(findings),
        },
        "compare_fields": list(args.compare_field),
        "artifacts": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(records: list[ArtifactRecord], path: Path) -> None:
    fieldnames = list(ArtifactRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(findings: list[Finding], path: Path) -> None:
    fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(records: list[ArtifactRecord], findings: list[Finding], path: Path) -> None:
    lines = [
        "# QEMU Launch JSONL Parity Audit",
        "",
        f"- Status: {'FAIL' if findings else 'PASS'}",
        f"- Artifacts: {len(records)}",
        f"- Findings: {len(findings)}",
        "",
        "| Source | JSON rows | JSONL rows | Findings |",
        "| --- | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(f"| {record.source} | {record.json_rows} | {record.jsonl_rows} | {record.findings} |")
    lines.extend(["", "## Findings"])
    if findings:
        lines.extend(f"- {finding.kind}: {finding.source} row {finding.row} {finding.field} {finding.detail}" for finding in findings[:50])
    else:
        lines.append("No launch JSONL parity findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(records: list[ArtifactRecord], findings: list[Finding], path: Path) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "qemu_launch_jsonl_parity_audit",
            "tests": str(max(1, len(records))),
            "failures": str(len(findings)),
            "errors": "0",
        },
    )
    if records:
        for record in records:
            case = ET.SubElement(suite, "testcase", {"classname": "qemu_launch_jsonl_parity_audit", "name": record.source})
            for finding in findings:
                if finding.source == record.source:
                    failure = ET.SubElement(case, "failure", {"message": finding.kind})
                    failure.text = finding.detail
    else:
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_launch_jsonl_parity_audit", "name": "artifacts"})
        for finding in findings:
            failure = ET.SubElement(case, "failure", {"message": finding.kind})
            failure.text = finding.detail
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path, help="Benchmark JSON artifact or directory")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Glob used when input is a directory")
    parser.add_argument("--compare-field", action="append", default=list(DEFAULT_COMPARE_FIELDS), help="Field to compare")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_launch_jsonl_parity_audit_latest")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, findings = audit(args.input, args)
    write_json_report(records, findings, args, args.output_dir / f"{args.output_stem}.json")
    write_csv(records, args.output_dir / f"{args.output_stem}.csv")
    write_findings_csv(findings, args.output_dir / f"{args.output_stem}_findings.csv")
    write_markdown(records, findings, args.output_dir / f"{args.output_stem}.md")
    write_junit(records, findings, args.output_dir / f"{args.output_stem}_junit.xml")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
