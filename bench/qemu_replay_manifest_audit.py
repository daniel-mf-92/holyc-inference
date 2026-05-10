#!/usr/bin/env python3
"""Audit exported QEMU replay manifests and argv sidecars.

This host-side tool reads qemu_replay_manifest JSON outputs only. It never
launches QEMU and keeps the TempleOS guest air-gapped by validating that every
recorded replay argv still contains explicit `-nic none` and no networking
arguments.
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


DEFAULT_PATTERNS = ("qemu_replay_manifest*.json",)
REQUIRED_ENTRY_FIELDS = {
    "key",
    "source",
    "source_size_bytes",
    "source_mtime_ns",
    "source_sha256",
    "status",
    "profile",
    "model",
    "quantization",
    "prompt_suite_sha256",
    "launch_plan_sha256",
    "expected_launch_sequence_sha256",
    "command_sha256",
    "command_argc",
    "explicit_nic_none",
    "legacy_net_none",
    "airgap_ok",
    "launch_plan_entries",
    "measured_rows",
    "argv",
}


@dataclass(frozen=True)
class ManifestAuditRow:
    source: str
    status: str
    entries: int
    unique_replay_keys: int
    argv_sidecar: str
    argv_sidecar_rows: int
    airgap_ok_entries: int
    measured_rows: int
    launch_plan_entries: int
    findings: int


@dataclass(frozen=True)
class ManifestFinding:
    source: str
    entry: str
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
                    if (
                        child.is_file()
                        and child not in seen
                        and "_audit" not in child.stem
                        and not child.name.endswith("_junit.xml")
                    ):
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
        return None, "manifest root must be a JSON object"
    return payload, ""


def load_argv_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return rows, str(exc)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            return rows, f"line {line_number}: invalid json: {exc}"
        if not isinstance(row, dict):
            return rows, f"line {line_number}: row must be a JSON object"
        rows.append(row)
    return rows, ""


def string_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def sibling_argv_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}_argv.jsonl")


def resolve_source_path(source: str, manifest_path: Path) -> Path | None:
    if not source:
        return None
    path = Path(source)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_summary(
    source: Path,
    payload: dict[str, Any],
    entries: list[dict[str, Any]],
    findings: list[ManifestFinding],
) -> None:
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        findings.append(ManifestFinding(str(source), "", "error", "summary_type", "summary", "summary must be an object"))
        return

    expected = {
        "artifacts": len(entries),
        "unique_replay_keys": len({entry.get("key") for entry in entries if isinstance(entry.get("key"), str)}),
        "airgap_ok_entries": sum(1 for entry in entries if entry.get("airgap_ok") is True),
        "measured_rows": sum(int_value(entry.get("measured_rows")) or 0 for entry in entries),
        "launch_plan_entries": sum(int_value(entry.get("launch_plan_entries")) or 0 for entry in entries),
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            findings.append(
                ManifestFinding(
                    str(source),
                    "",
                    "error",
                    "summary_drift",
                    f"summary.{field}",
                    f"expected {expected_value}, got {summary.get(field)!r}",
                )
            )
    if payload.get("status") == "pass" and summary.get("findings", 0) != 0:
        findings.append(
            ManifestFinding(str(source), "", "error", "status_drift", "summary.findings", "pass manifest has findings")
        )


def check_duplicate_entry_keys(source: Path, entries: list[dict[str, Any]], findings: list[ManifestFinding]) -> None:
    seen: dict[str, int] = {}
    for index, entry in enumerate(entries, 1):
        key = entry.get("key")
        if not isinstance(key, str):
            continue
        previous = seen.get(key)
        if previous is not None:
            findings.append(
                ManifestFinding(
                    str(source),
                    f"entries[{index}]",
                    "error",
                    "duplicate_replay_key",
                    "key",
                    f"duplicates entries[{previous}] key {key!r}",
                )
            )
        else:
            seen[key] = index


def check_entry(
    source: Path,
    manifest_path: Path,
    entry: Any,
    index: int,
    args: argparse.Namespace,
) -> list[ManifestFinding]:
    row_name = f"entries[{index}]"
    if not isinstance(entry, dict):
        return [ManifestFinding(str(source), row_name, "error", "entry_type", "", "entry must be an object")]

    findings: list[ManifestFinding] = []
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
    for field in missing:
        findings.append(ManifestFinding(str(source), row_name, "error", "missing_field", field, "required field is absent"))

    argv = string_list(entry.get("argv"))
    if argv is None:
        findings.append(ManifestFinding(str(source), row_name, "error", "argv_type", "argv", "argv must be a string array"))
        argv = []

    if argv:
        expected_hash = qemu_prompt_bench.command_hash(argv)
        if entry.get("command_sha256") != expected_hash:
            findings.append(
                ManifestFinding(
                    str(source),
                    row_name,
                    "error",
                    "command_hash",
                    "command_sha256",
                    f"stored command hash {entry.get('command_sha256')!r} does not match argv hash {expected_hash}",
                )
            )
        if entry.get("command_argc") != len(argv):
            findings.append(
                ManifestFinding(str(source), row_name, "error", "command_argc", "command_argc", "argc differs from argv")
            )

    airgap = qemu_prompt_bench.command_airgap_metadata(argv)
    if entry.get("airgap_ok") is not True or not airgap["ok"]:
        findings.append(
            ManifestFinding(
                str(source),
                row_name,
                "error",
                "command_airgap",
                "argv",
                "; ".join(airgap["violations"]) or "entry airgap_ok is not true",
            )
        )
    if entry.get("explicit_nic_none") is not True or airgap["explicit_nic_none"] is not True:
        findings.append(
            ManifestFinding(str(source), row_name, "error", "missing_nic_none", "argv", "replay argv must include -nic none")
        )
    if entry.get("legacy_net_none") is not False or airgap["legacy_net_none"]:
        findings.append(
            ManifestFinding(str(source), row_name, "error", "legacy_net_none", "argv", "legacy -net none is disallowed")
        )

    key = entry.get("key")
    if isinstance(key, str):
        expected_suffix = f"{entry.get('prompt_suite_sha256')}/{entry.get('command_sha256')}"
        if not key.endswith(expected_suffix):
            findings.append(
                ManifestFinding(str(source), row_name, "error", "key_drift", "key", "key does not end with suite/hash pair")
            )
    else:
        findings.append(ManifestFinding(str(source), row_name, "error", "key_type", "key", "key must be a string"))

    for field in ("source", "profile", "model", "quantization", "prompt_suite_sha256", "launch_plan_sha256"):
        if not isinstance(entry.get(field), str) or not entry.get(field):
            findings.append(ManifestFinding(str(source), row_name, "error", "blank_field", field, "field must be non-empty"))

    source_path = resolve_source_path(str(entry.get("source") or ""), manifest_path)
    if args.require_existing_source and source_path is None:
        findings.append(
            ManifestFinding(str(source), row_name, "error", "missing_source_artifact", "source", "source artifact is absent")
        )
    if source_path is not None:
        stat = source_path.stat()
        if int_value(entry.get("source_size_bytes")) != stat.st_size:
            findings.append(
                ManifestFinding(
                    str(source),
                    row_name,
                    "error",
                    "source_size_drift",
                    "source_size_bytes",
                    f"stored={entry.get('source_size_bytes')!r} expected={stat.st_size}",
                )
            )
        if int_value(entry.get("source_mtime_ns")) is None:
            findings.append(
                ManifestFinding(
                    str(source),
                    row_name,
                    "error",
                    "missing_source_mtime",
                    "source_mtime_ns",
                    "source artifact mtime metadata is absent",
                )
            )
        expected_source_sha256 = qemu_prompt_bench.file_sha256(source_path)
        if entry.get("source_sha256") != expected_source_sha256:
            findings.append(
                ManifestFinding(
                    str(source),
                    row_name,
                    "error",
                    "source_sha256_drift",
                    "source_sha256",
                    f"stored={entry.get('source_sha256')!r} expected={expected_source_sha256}",
                )
            )

    return findings


def check_argv_sidecar(
    source: Path,
    entries: list[dict[str, Any]],
    sidecar_path: Path,
    args: argparse.Namespace,
) -> tuple[int, list[ManifestFinding]]:
    findings: list[ManifestFinding] = []
    if not sidecar_path.exists():
        if args.require_argv_jsonl:
            findings.append(
                ManifestFinding(str(source), "", "error", "missing_argv_sidecar", str(sidecar_path), "argv sidecar is absent")
            )
        return 0, findings

    rows, error = load_argv_jsonl(sidecar_path)
    if error:
        return 0, [ManifestFinding(str(source), "", "error", "argv_sidecar_load", str(sidecar_path), error)]

    expected_by_key = {entry.get("key"): entry for entry in entries if isinstance(entry.get("key"), str)}
    observed_keys: set[str] = set()
    for index, row in enumerate(rows, 1):
        row_name = f"argv[{index}]"
        key = row.get("key")
        if not isinstance(key, str):
            findings.append(ManifestFinding(str(source), row_name, "error", "argv_key_type", "key", "key must be a string"))
            continue
        if key in observed_keys:
            findings.append(
                ManifestFinding(
                    str(source),
                    row_name,
                    "error",
                    "duplicate_argv_key",
                    "key",
                    f"argv sidecar repeats key {key!r}",
                )
            )
        observed_keys.add(key)
        entry = expected_by_key.get(key)
        if entry is None:
            findings.append(ManifestFinding(str(source), row_name, "error", "argv_extra_key", "key", "key is not in manifest"))
            continue
        argv = string_list(row.get("argv"))
        if argv is None:
            findings.append(ManifestFinding(str(source), row_name, "error", "argv_type", "argv", "argv must be a string array"))
            argv = []
        if argv:
            row_hash = qemu_prompt_bench.command_hash(argv)
            if row.get("command_sha256") != row_hash:
                findings.append(
                    ManifestFinding(
                        str(source),
                        row_name,
                        "error",
                        "argv_command_hash",
                        "command_sha256",
                        "sidecar command hash does not match sidecar argv",
                    )
                )
            airgap = qemu_prompt_bench.command_airgap_metadata(argv)
            if not airgap["ok"]:
                findings.append(
                    ManifestFinding(
                        str(source),
                        row_name,
                        "error",
                        "argv_command_airgap",
                        "argv",
                        "; ".join(airgap["violations"]),
                    )
                )
        if argv != string_list(entry.get("argv")):
            findings.append(ManifestFinding(str(source), row_name, "error", "argv_drift", "argv", "sidecar argv differs"))
        if row.get("command_sha256") != entry.get("command_sha256"):
            findings.append(
                ManifestFinding(str(source), row_name, "error", "argv_hash_drift", "command_sha256", "sidecar hash differs")
            )

    for key in sorted(set(expected_by_key) - observed_keys):
        findings.append(ManifestFinding(str(source), key, "error", "argv_missing_key", "key", "manifest key missing from sidecar"))

    return len(rows), findings


def audit_manifest(path: Path, args: argparse.Namespace) -> tuple[ManifestAuditRow, list[ManifestFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = ManifestFinding(str(path), "", "error", "load_error", "", error)
        return ManifestAuditRow(str(path), "fail", 0, 0, "", 0, 0, 0, 0, 1), [finding]

    findings: list[ManifestFinding] = []
    entries_payload = payload.get("entries")
    if not isinstance(entries_payload, list):
        entries: list[dict[str, Any]] = []
        findings.append(ManifestFinding(str(path), "", "error", "entries_type", "entries", "entries must be an array"))
    else:
        entries = [entry for entry in entries_payload if isinstance(entry, dict)]
        for index, entry in enumerate(entries_payload, 1):
            findings.extend(check_entry(path, path, entry, index, args))
        check_duplicate_entry_keys(path, entries, findings)

    check_summary(path, payload, entries, findings)
    sidecar_path = args.argv_jsonl or sibling_argv_path(path)
    sidecar_rows, sidecar_findings = check_argv_sidecar(path, entries, sidecar_path, args)
    findings.extend(sidecar_findings)

    row = ManifestAuditRow(
        source=str(path),
        status="fail" if findings else "pass",
        entries=len(entries),
        unique_replay_keys=len({entry.get("key") for entry in entries if isinstance(entry.get("key"), str)}),
        argv_sidecar=str(sidecar_path),
        argv_sidecar_rows=sidecar_rows,
        airgap_ok_entries=sum(1 for entry in entries if entry.get("airgap_ok") is True),
        measured_rows=sum(int_value(entry.get("measured_rows")) or 0 for entry in entries),
        launch_plan_entries=sum(int_value(entry.get("launch_plan_entries")) or 0 for entry in entries),
        findings=len(findings),
    )
    return row, findings


def build_report(rows: list[ManifestAuditRow], findings: list[ManifestFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "manifests": len(rows),
            "entries": sum(row.entries for row in rows),
            "unique_replay_keys": sum(row.unique_replay_keys for row in rows),
            "argv_sidecar_rows": sum(row.argv_sidecar_rows for row in rows),
            "airgap_ok_entries": sum(row.airgap_ok_entries for row in rows),
            "measured_rows": sum(row.measured_rows for row in rows),
            "launch_plan_entries": sum(row.launch_plan_entries for row in rows),
            "findings": len(findings),
        },
        "manifests": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[ManifestAuditRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestAuditRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_findings_csv(path: Path, findings: list[ManifestFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Replay Manifest Audit",
        "",
        f"- Status: {report['status']}",
        f"- Manifests: {summary['manifests']}",
        f"- Entries: {summary['entries']}",
        f"- Argv sidecar rows: {summary['argv_sidecar_rows']}",
        f"- Air-gap OK entries: {summary['airgap_ok_entries']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Source | Status | Entries | Argv rows | Air-gap OK | Findings |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["manifests"]:
        lines.append(
            "| {source} | {status} | {entries} | {argv_sidecar_rows} | {airgap_ok_entries} | {findings} |".format(
                **row
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"][:50]:
            lines.append("- {kind}: {entry} {field} in {source} ({detail})".format(**finding))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[ManifestFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_replay_manifest_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.field}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = f"{finding.source}: {finding.detail}"
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_replay_manifest_audit"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Replay manifest JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--argv-jsonl", type=Path, help="Optional argv sidecar path for a single manifest")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_replay_manifest_audit_latest")
    parser.add_argument("--min-manifests", type=int, default=1)
    parser.add_argument("--require-argv-jsonl", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-existing-source", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.argv_jsonl and len(args.inputs) != 1:
        print("--argv-jsonl may only be used with one manifest input", file=sys.stderr)
        return 2
    if args.min_manifests < 0:
        print("--min-manifests must be non-negative", file=sys.stderr)
        return 2

    rows: list[ManifestAuditRow] = []
    findings: list[ManifestFinding] = []
    for path in iter_input_files(args.inputs, args.pattern):
        row, row_findings = audit_manifest(path, args)
        rows.append(row)
        findings.extend(row_findings)

    if len(rows) < args.min_manifests:
        findings.append(
            ManifestFinding(
                "",
                "",
                "error",
                "min_manifests",
                "inputs",
                f"found {len(rows)} manifests, expected at least {args.min_manifests}",
            )
        )

    report = build_report(rows, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_csv(output_base.with_suffix(".csv"), rows)
    write_findings_csv(output_base.with_name(f"{output_base.name}_findings.csv"), findings)
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
