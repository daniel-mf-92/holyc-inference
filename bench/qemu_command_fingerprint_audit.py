#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for command fingerprint integrity.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
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

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import qemu_prompt_bench


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class CommandRow:
    source: str
    row: int
    list_name: str
    phase: str
    prompt: str
    iteration: int | None
    launch_index: int | None
    command_sha256: str
    computed_command_sha256: str
    command_airgap_ok: bool
    explicit_nic_none: bool
    legacy_net_none: bool
    command_argc: int


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    top_command_sha256: str
    computed_top_command_sha256: str
    row_count: int
    unique_row_command_hashes: int
    airgap_ok_rows: int
    explicit_nic_none_rows: int
    legacy_net_none_rows: int
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


def optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def optional_string_list(value: Any) -> list[str] | None:
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
            findings.append(
                Finding(str(path), 0, "error", f"invalid_{list_name}", list_name, f"{list_name} must be a list")
            )
            continue
        for row in raw:
            if isinstance(row, dict):
                rows.append((list_name, row))
            else:
                findings.append(
                    Finding(str(path), len(rows) + 1, "error", "invalid_row", list_name, "benchmark row must be an object")
                )
    return rows


def row_label(row: dict[str, Any], list_name: str) -> str:
    phase = text(row.get("phase")) or ("warmup" if list_name == "warmups" else "measured")
    prompt = text(row.get("prompt")) or text(row.get("prompt_id")) or "-"
    iteration = row.get("iteration", "-")
    launch_index = row.get("launch_index", "-")
    return f"{list_name} launch_index={launch_index} phase={phase} prompt={prompt} iteration={iteration}"


def audit_command(
    path: Path,
    row_number: int,
    field: str,
    command: list[str],
    stored_hash: str,
    findings: list[Finding],
) -> tuple[str, dict[str, Any]]:
    computed_hash = qemu_prompt_bench.command_hash(command)
    if stored_hash != computed_hash:
        findings.append(
            Finding(
                str(path),
                row_number,
                "error",
                "command_sha256_mismatch",
                field,
                f"stored {stored_hash or '<missing>'} does not match computed {computed_hash}",
            )
        )
    airgap = qemu_prompt_bench.command_airgap_metadata(command)
    if airgap["legacy_net_none"]:
        findings.append(
            Finding(str(path), row_number, "error", "legacy_net_none", field, "legacy -net none is not allowed in benchmark artifacts")
        )
    if not airgap["explicit_nic_none"]:
        findings.append(
            Finding(str(path), row_number, "error", "missing_explicit_nic_none", field, "command must include explicit -nic none")
        )
    if not airgap["ok"]:
        findings.append(
            Finding(str(path), row_number, "error", "command_airgap_violation", field, "; ".join(airgap["violations"]))
        )
    return computed_hash, airgap


def audit_stored_airgap_metadata(
    path: Path,
    row_number: int,
    row: dict[str, Any],
    airgap: dict[str, Any],
    findings: list[Finding],
    require_metadata: bool,
) -> None:
    expected_fields = {
        "command_airgap_ok": airgap["ok"],
        "command_has_explicit_nic_none": airgap["explicit_nic_none"],
        "command_has_legacy_net_none": airgap["legacy_net_none"],
    }
    for field, expected in expected_fields.items():
        stored = optional_bool(row.get(field))
        if stored is None:
            if require_metadata:
                findings.append(
                    Finding(str(path), row_number, "error", f"missing_{field}", field, "row must carry computed command air-gap metadata")
                )
            continue
        if stored != expected:
            findings.append(
                Finding(
                    str(path),
                    row_number,
                    "error",
                    f"{field}_mismatch",
                    field,
                    f"stored {stored} does not match computed {expected}",
                )
            )

    stored_violations = optional_string_list(row.get("command_airgap_violations"))
    expected_violations = list(airgap["violations"])
    if stored_violations is None:
        if require_metadata:
            findings.append(
                Finding(
                    str(path),
                    row_number,
                    "error",
                    "missing_command_airgap_violations",
                    "command_airgap_violations",
                    "row must carry computed command air-gap violations",
                )
            )
        return
    if stored_violations != expected_violations:
        findings.append(
            Finding(
                str(path),
                row_number,
                "error",
                "command_airgap_violations_mismatch",
                "command_airgap_violations",
                f"stored {stored_violations!r} does not match computed {expected_violations!r}",
            )
        )


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactRecord, list[CommandRow], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = Finding(str(path), 0, "error", "load_error", "artifact", error)
        return ArtifactRecord(str(path), "fail", "", "", 0, 0, 0, 0, 0, 1, error), [], [finding]

    findings: list[Finding] = []
    row_records: list[CommandRow] = []

    top_command = command_list(payload.get("command"))
    top_hash = text(payload.get("command_sha256"))
    computed_top_hash = ""
    if top_command is None:
        if args.require_top_command:
            findings.append(Finding(str(path), 0, "error", "missing_top_command", "command", "top-level command must be a list of strings"))
    else:
        computed_top_hash, _ = audit_command(path, 0, "command", top_command, top_hash, findings)

    for row_number, (list_name, row) in enumerate(data_rows(payload, findings, path), 1):
        command = command_list(row.get("command"))
        if command is None:
            findings.append(
                Finding(str(path), row_number, "error", "missing_row_command", "command", f"{row_label(row, list_name)}: command must be a list of strings")
            )
            continue
        stored_hash = text(row.get("command_sha256"))
        computed_hash, airgap = audit_command(path, row_number, "command_sha256", command, stored_hash, findings)
        audit_stored_airgap_metadata(path, row_number, row, airgap, findings, args.require_row_airgap_metadata)
        effective_hash = stored_hash or computed_hash
        if top_hash and effective_hash != top_hash and not args.allow_row_command_drift:
            findings.append(
                Finding(
                    str(path),
                    row_number,
                    "error",
                    "row_command_hash_drift",
                    "command_sha256",
                    f"{row_label(row, list_name)}: row command hash differs from top-level command hash",
                )
            )

        row_records.append(
            CommandRow(
                source=str(path),
                row=row_number,
                list_name=list_name,
                phase=text(row.get("phase")) or ("warmup" if list_name == "warmups" else "measured"),
                prompt=text(row.get("prompt")) or text(row.get("prompt_id")) or "-",
                iteration=int_or_none(row.get("iteration")),
                launch_index=int_or_none(row.get("launch_index")),
                command_sha256=stored_hash,
                computed_command_sha256=computed_hash,
                command_airgap_ok=airgap["ok"],
                explicit_nic_none=airgap["explicit_nic_none"],
                legacy_net_none=airgap["legacy_net_none"],
                command_argc=len(command),
            )
        )

    if args.min_rows and len(row_records) < args.min_rows:
        findings.append(
            Finding(str(path), 0, "error", "min_rows", "rows", f"found {len(row_records)}, expected at least {args.min_rows}")
        )

    unique_hashes = {row.computed_command_sha256 for row in row_records}
    if args.require_single_command_hash and len(unique_hashes) > 1:
        findings.append(
            Finding(
                str(path),
                0,
                "error",
                "multiple_row_command_hashes",
                "command_sha256",
                f"found {len(unique_hashes)} row command hashes, expected exactly 1",
            )
        )

    artifact = ArtifactRecord(
        source=str(path),
        status="fail" if findings else "pass",
        top_command_sha256=top_hash,
        computed_top_command_sha256=computed_top_hash,
        row_count=len(row_records),
        unique_row_command_hashes=len(unique_hashes),
        airgap_ok_rows=sum(1 for row in row_records if row.command_airgap_ok),
        explicit_nic_none_rows=sum(1 for row in row_records if row.explicit_nic_none),
        legacy_net_none_rows=sum(1 for row in row_records if row.legacy_net_none),
        findings=len(findings),
    )
    return artifact, row_records, findings


def write_json(path: Path, artifacts: list[ArtifactRecord], rows: list[CommandRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "rows": len(rows),
            "findings": len(findings),
            "failed_artifacts": sum(1 for artifact in artifacts if artifact.status != "pass"),
            "unique_row_command_hashes": len({row.computed_command_sha256 for row in rows}),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[CommandRow]) -> None:
    fields = list(CommandRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fields = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, artifacts: list[ArtifactRecord], rows: list[CommandRow], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Command Fingerprint Audit",
        "",
        f"- Artifacts: {len(artifacts)}",
        f"- Rows: {len(rows)}",
        f"- Findings: {len(findings)}",
        f"- Unique row command hashes: {len({row.computed_command_sha256 for row in rows})}",
        "",
    ]
    if findings:
        lines.extend(["## Findings", ""])
        for finding in findings:
            lines.append(f"- {finding.severity}: {finding.kind} row={finding.row} field={finding.field} - {finding.detail}")
    else:
        lines.append("No command fingerprint findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_command_fingerprint_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    testcase = ET.SubElement(testsuite, "testcase", {"name": "command_fingerprints"})
    if findings:
        failure = ET.SubElement(testcase, "failure", {"message": f"{len(findings)} command fingerprint findings"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON artifacts or directories to audit")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern when input is a directory")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_command_fingerprint_audit_latest")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--allow-row-command-drift", action="store_true")
    parser.add_argument("--require-single-command-hash", action="store_true")
    parser.add_argument("--require-top-command", action="store_true")
    parser.add_argument("--require-row-airgap-metadata", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.pattern or list(DEFAULT_PATTERNS)
    artifacts: list[ArtifactRecord] = []
    rows: list[CommandRow] = []
    findings: list[Finding] = []

    for path in iter_input_files(args.inputs, patterns):
        artifact, artifact_rows, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        rows.extend(artifact_rows)
        findings.extend(artifact_findings)

    if not artifacts:
        findings.append(Finding("-", 0, "error", "no_artifacts", "inputs", "no matching benchmark artifacts found"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", artifacts, rows, findings)
    write_csv(args.output_dir / f"{stem}.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", artifacts, rows, findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
