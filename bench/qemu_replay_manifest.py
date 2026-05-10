#!/usr/bin/env python3
"""Build an air-gapped replay manifest from saved QEMU benchmark artifacts.

This host-side tool reads existing qemu_prompt_bench JSON artifacts only. It
does not launch QEMU. The manifest captures the exact argv, prompt suite, launch
plan, and provenance needed to replay a benchmark run while auditing that every
recorded command remains explicitly air-gapped with `-nic none`.
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
class ReplayEntry:
    key: str
    source: str
    source_size_bytes: int | None
    source_mtime_ns: int | None
    source_sha256: str | None
    generated_at: str
    status: str
    commit: str
    profile: str
    model: str
    quantization: str
    prompt_suite_source: str
    prompt_suite_sha256: str
    launch_plan_sha256: str
    expected_launch_sequence_sha256: str
    command_sha256: str
    command_argc: int
    qemu_bin: str
    explicit_nic_none: bool
    legacy_net_none: bool
    airgap_ok: bool
    prompt_count: int | None
    launch_plan_entries: int
    measured_rows: int
    argv: list[str]


@dataclass(frozen=True)
class ReplayFinding:
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
    return value if isinstance(value, int) else None


def string_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def replay_key(entry: ReplayEntry) -> str:
    return "/".join(
        (
            entry.profile or "-",
            entry.model or "-",
            entry.quantization or "-",
            entry.prompt_suite_sha256 or "no-suite",
            entry.command_sha256 or "no-command",
        )
    )


def audit_artifact(path: Path) -> tuple[ReplayEntry | None, list[ReplayFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return None, [ReplayFinding(str(path), "error", "load_error", "", error)]

    source_metadata = qemu_prompt_bench.input_file_metadata(path, include_sha256=True)
    findings: list[ReplayFinding] = []
    command = string_list(payload.get("command"))
    if command is None:
        findings.append(ReplayFinding(str(path), "error", "command_type", "command", "command must be a string array"))
        command = []

    command_sha256 = text_value(payload.get("command_sha256"))
    if command:
        expected_command_sha256 = qemu_prompt_bench.command_hash(command)
        if command_sha256 != expected_command_sha256:
            findings.append(
                ReplayFinding(
                    str(path),
                    "error",
                    "command_hash",
                    "command_sha256",
                    f"stored command hash {command_sha256!r} does not match argv hash {expected_command_sha256}",
                )
            )
            command_sha256 = expected_command_sha256
    else:
        expected_command_sha256 = ""

    airgap = qemu_prompt_bench.command_airgap_metadata(command)
    if not airgap["ok"]:
        findings.append(
            ReplayFinding(str(path), "error", "command_airgap", "command", "; ".join(airgap["violations"]))
        )

    recorded_airgap = payload.get("command_airgap")
    if isinstance(recorded_airgap, dict):
        for field, observed in (
            ("ok", airgap["ok"]),
            ("explicit_nic_none", airgap["explicit_nic_none"]),
            ("legacy_net_none", airgap["legacy_net_none"]),
        ):
            if recorded_airgap.get(field) != observed:
                findings.append(
                    ReplayFinding(
                        str(path),
                        "error",
                        "command_airgap_drift",
                        f"command_airgap.{field}",
                        "recorded air-gap metadata differs from argv-derived metadata",
                    )
                )
    else:
        findings.append(
            ReplayFinding(str(path), "error", "missing_command_airgap", "command_airgap", "object is absent")
        )

    prompt_suite = payload.get("prompt_suite") if isinstance(payload.get("prompt_suite"), dict) else {}
    prompt_count = int_value(prompt_suite.get("prompt_count"))
    if not text_value(prompt_suite.get("suite_sha256")):
        findings.append(
            ReplayFinding(str(path), "error", "missing_prompt_suite_hash", "prompt_suite.suite_sha256", "hash is absent")
        )

    launch_plan = object_list(payload.get("launch_plan"))
    if not launch_plan:
        findings.append(ReplayFinding(str(path), "error", "missing_launch_plan", "launch_plan", "no launch rows found"))
    if not text_value(payload.get("launch_plan_sha256")):
        findings.append(
            ReplayFinding(str(path), "error", "missing_launch_plan_hash", "launch_plan_sha256", "hash is absent")
        )

    benchmarks = object_list(payload.get("benchmarks"))
    measured_rows = sum(1 for row in benchmarks if str(row.get("phase") or "measured") == "measured")
    for index, row in enumerate(benchmarks, 1):
        row_command = string_list(row.get("command"))
        if row_command is None:
            findings.append(
                ReplayFinding(str(path), "error", "row_command_type", f"benchmarks[{index}].command", "must be a string array")
            )
            continue
        row_hash = qemu_prompt_bench.command_hash(row_command)
        if row_hash != command_sha256:
            findings.append(
                ReplayFinding(
                    str(path),
                    "error",
                    "row_command_drift",
                    f"benchmarks[{index}].command",
                    "row command differs from artifact command",
                )
            )
        if row.get("command_sha256") != row_hash:
            findings.append(
                ReplayFinding(
                    str(path),
                    "error",
                    "row_command_hash",
                    f"benchmarks[{index}].command_sha256",
                    "row command hash does not match row argv",
                )
            )

    entry = ReplayEntry(
        key="",
        source=str(path),
        source_size_bytes=int_value(source_metadata.get("size_bytes")),
        source_mtime_ns=int_value(source_metadata.get("mtime_ns")),
        source_sha256=text_value(source_metadata.get("sha256")) or None,
        generated_at=text_value(payload.get("generated_at")),
        status="fail" if findings else (text_value(payload.get("status")) or "unknown"),
        commit=text_value(payload.get("commit")),
        profile=text_value(payload.get("profile")),
        model=text_value(payload.get("model")),
        quantization=text_value(payload.get("quantization")),
        prompt_suite_source=text_value(prompt_suite.get("source")),
        prompt_suite_sha256=text_value(prompt_suite.get("suite_sha256")),
        launch_plan_sha256=text_value(payload.get("launch_plan_sha256")),
        expected_launch_sequence_sha256=text_value(payload.get("expected_launch_sequence_sha256")),
        command_sha256=command_sha256 or expected_command_sha256,
        command_argc=len(command),
        qemu_bin=command[0] if command else "",
        explicit_nic_none=bool(airgap["explicit_nic_none"]),
        legacy_net_none=bool(airgap["legacy_net_none"]),
        airgap_ok=bool(airgap["ok"]),
        prompt_count=prompt_count,
        launch_plan_entries=len(launch_plan),
        measured_rows=measured_rows,
        argv=command,
    )
    return ReplayEntry(**{**asdict(entry), "key": replay_key(entry)}), findings


def build_report(entries: list[ReplayEntry], findings: list[ReplayFinding]) -> dict[str, Any]:
    entries = sorted(entries, key=lambda item: (item.key, item.generated_at, item.source))
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(entries),
            "unique_replay_keys": len({entry.key for entry in entries}),
            "airgap_ok_entries": sum(1 for entry in entries if entry.airgap_ok),
            "measured_rows": sum(entry.measured_rows for entry in entries),
            "launch_plan_entries": sum(entry.launch_plan_entries for entry in entries),
            "findings": len(findings),
        },
        "entries": [asdict(entry) for entry in entries],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_entries_csv(path: Path, entries: list[ReplayEntry]) -> None:
    fieldnames = [field for field in ReplayEntry.__dataclass_fields__ if field != "argv"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row.pop("argv", None)
            writer.writerow(row)


def write_argv_jsonl(path: Path, entries: list[ReplayEntry]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "key": entry.key,
                        "source": entry.source,
                        "command_sha256": entry.command_sha256,
                        "argv": entry.argv,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def write_findings_csv(path: Path, findings: list[ReplayFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReplayFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Replay Manifest",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Unique replay keys: {summary['unique_replay_keys']}",
        f"- Air-gap OK entries: {summary['airgap_ok_entries']}",
        f"- Measured rows: {summary['measured_rows']}",
        f"- Launch plan entries: {summary['launch_plan_entries']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Key | Status | Prompt suite | Command | Rows | Air-gap | Source |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for entry in report["entries"]:
        lines.append(
            "| {key} | {status} | {prompt_suite_sha256} | {command_sha256} | {measured_rows} | {airgap_ok} | {source} |".format(
                **entry
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"][:50]:
            lines.append("- {kind}: {field} in {source} ({detail})".format(**finding))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[ReplayFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_replay_manifest",
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
        ET.SubElement(suite, "testcase", {"name": "qemu_replay_manifest"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_replay_manifest_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_artifacts < 0:
        print("--min-artifacts must be non-negative", file=sys.stderr)
        return 2

    entries: list[ReplayEntry] = []
    findings: list[ReplayFinding] = []
    for path in iter_input_files(args.inputs, args.pattern):
        entry, entry_findings = audit_artifact(path)
        if entry is not None:
            entries.append(entry)
        findings.extend(entry_findings)

    if len(entries) < args.min_artifacts:
        findings.append(
            ReplayFinding(
                "",
                "error",
                "min_artifacts",
                "inputs",
                f"found {len(entries)} artifacts, expected at least {args.min_artifacts}",
            )
        )

    report = build_report(entries, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_entries_csv(output_base.with_suffix(".csv"), entries)
    write_argv_jsonl(output_base.with_name(f"{output_base.name}_argv.jsonl"), entries)
    write_findings_csv(output_base.with_name(f"{output_base.name}_findings.csv"), findings)
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
