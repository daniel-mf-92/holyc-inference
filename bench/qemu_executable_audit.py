#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for executable provenance.

This host-side tool reads existing qemu_prompt_bench JSON artifacts only. It
never launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)
WRAPPER_BASENAMES = {
    "bash",
    "env",
    "fish",
    "python",
    "python3",
    "sh",
    "sudo",
    "zsh",
}


@dataclass(frozen=True)
class ExecutableRecord:
    source: str
    row: int
    scope: str
    phase: str
    prompt: str
    argv0: str
    argv0_basename: str
    environment_qemu_bin: str
    environment_qemu_path: str
    absolute_argv0: bool


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


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def command_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    return None


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


def qemu_basename(argv0: str) -> str:
    return Path(argv0.strip()).name


def is_qemu_system(argv0: str) -> bool:
    return qemu_basename(argv0).startswith("qemu-system-")


def rows(payload: dict[str, Any], findings: list[Finding], source: Path) -> list[tuple[str, int, dict[str, Any]]]:
    loaded: list[tuple[str, int, dict[str, Any]]] = []
    for scope in ("warmups", "benchmarks"):
        value = payload.get(scope)
        if value is None:
            continue
        if not isinstance(value, list):
            findings.append(Finding(str(source), 0, "error", f"invalid_{scope}", scope, f"{scope} must be a list"))
            continue
        for index, item in enumerate(value, 1):
            if isinstance(item, dict):
                loaded.append((scope, index, item))
            else:
                findings.append(Finding(str(source), index, "error", "invalid_row", scope, "row must be an object"))
    return loaded


def add_command_findings(
    findings: list[Finding],
    *,
    source: Path,
    row: int,
    scope: str,
    command: list[str] | None,
    env_qemu_bin: str,
    env_qemu_path: str,
    top_argv0: str,
    args: argparse.Namespace,
) -> ExecutableRecord | None:
    if command is None:
        findings.append(Finding(str(source), row, "error", "missing_command", "command", f"{scope} command must be a non-empty string list"))
        return None

    argv0 = command[0]
    basename = qemu_basename(argv0)
    if basename in WRAPPER_BASENAMES:
        findings.append(Finding(str(source), row, "error", "wrapped_qemu_command", "command[0]", f"argv0 is wrapper {basename!r}; record QEMU executable directly"))
    if not is_qemu_system(argv0):
        findings.append(Finding(str(source), row, "error", "non_qemu_executable", "command[0]", f"argv0={argv0!r} does not start with qemu-system-"))
    if args.require_absolute and not os.path.isabs(argv0):
        findings.append(Finding(str(source), row, "error", "relative_qemu_executable", "command[0]", f"argv0={argv0!r} is not absolute"))
    if env_qemu_bin and basename != env_qemu_bin:
        findings.append(Finding(str(source), row, "error", "qemu_bin_mismatch", "environment.qemu_bin", f"command basename {basename!r} differs from qemu_bin {env_qemu_bin!r}"))
    if env_qemu_path and Path(env_qemu_path).name != basename:
        findings.append(Finding(str(source), row, "error", "qemu_path_mismatch", "environment.qemu_path", f"command basename {basename!r} differs from qemu_path {env_qemu_path!r}"))
    if top_argv0 and argv0 != top_argv0 and not args.allow_row_executable_drift:
        findings.append(Finding(str(source), row, "error", "row_executable_drift", "command[0]", f"row argv0 {argv0!r} differs from top-level argv0 {top_argv0!r}"))

    return ExecutableRecord(
        source=str(source),
        row=row,
        scope=scope,
        phase="top" if row == 0 else text(command[0] and ""),
        prompt="",
        argv0=argv0,
        argv0_basename=basename,
        environment_qemu_bin=env_qemu_bin,
        environment_qemu_path=env_qemu_path,
        absolute_argv0=os.path.isabs(argv0),
    )


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[list[ExecutableRecord], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [Finding(str(path), 0, "error", "load_error", "artifact", error)]

    findings: list[Finding] = []
    records: list[ExecutableRecord] = []
    environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    env_qemu_bin = text(environment.get("qemu_bin"))
    env_qemu_path = text(environment.get("qemu_path"))
    top_command = command_list(payload.get("command"))
    top_argv0 = top_command[0] if top_command else ""

    top_record = add_command_findings(
        findings,
        source=path,
        row=0,
        scope="artifact",
        command=top_command,
        env_qemu_bin=env_qemu_bin,
        env_qemu_path=env_qemu_path,
        top_argv0="",
        args=args,
    )
    if top_record is not None:
        records.append(top_record)

    for scope, index, row in rows(payload, findings, path):
        command = command_list(row.get("command"))
        record = add_command_findings(
            findings,
            source=path,
            row=index,
            scope=scope,
            command=command,
            env_qemu_bin=env_qemu_bin,
            env_qemu_path=env_qemu_path,
            top_argv0=top_argv0,
            args=args,
        )
        if record is not None:
            records.append(
                ExecutableRecord(
                    source=record.source,
                    row=record.row,
                    scope=record.scope,
                    phase=text(row.get("phase")) or ("warmup" if scope == "warmups" else "measured"),
                    prompt=text(row.get("prompt") or row.get("prompt_id")),
                    argv0=record.argv0,
                    argv0_basename=record.argv0_basename,
                    environment_qemu_bin=record.environment_qemu_bin,
                    environment_qemu_path=record.environment_qemu_path,
                    absolute_argv0=record.absolute_argv0,
                )
            )

    return records, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ExecutableRecord], list[Finding]]:
    records: list[ExecutableRecord] = []
    findings: list[Finding] = []
    files = 0
    for path in iter_input_files(paths, args.pattern):
        files += 1
        artifact_records, artifact_findings = audit_artifact(path, args)
        records.extend(artifact_records)
        findings.extend(artifact_findings)
    if files < args.min_artifacts:
        findings.append(Finding("", 0, "error", "min_artifacts", "input", f"found {files}, expected at least {args.min_artifacts}"))
    row_records = [record for record in records if record.row > 0]
    if len(row_records) < args.min_rows:
        findings.append(Finding("", 0, "error", "min_rows", "rows", f"found {len(row_records)}, expected at least {args.min_rows}"))
    return records, findings


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_qemu_executable_audit", tests="1", failures=str(len(findings)))
    case = ET.SubElement(suite, "testcase", name="qemu_executable_audit")
    for finding in findings:
        failure = ET.SubElement(case, "failure", type=finding.kind, message=finding.detail)
        failure.text = json.dumps(asdict(finding), sort_keys=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# QEMU Executable Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {report['summary']['artifacts']}",
        f"- Row commands: {report['summary']['row_commands']}",
        f"- Findings: {report['summary']['findings']}",
        "",
    ]
    findings = report["findings"]
    if findings:
        lines.append("## Findings")
        lines.append("")
        for finding in findings[:50]:
            lines.append(f"- {finding['kind']}: {finding['detail']}")
    else:
        lines.append("No QEMU executable provenance findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(records: list[ExecutableRecord], findings: list[Finding]) -> dict[str, Any]:
    artifact_sources = {record.source for record in records}
    row_records = [record for record in records if record.row > 0]
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifact_sources),
            "commands": len(records),
            "row_commands": len(row_records),
            "absolute_commands": sum(1 for record in records if record.absolute_argv0),
            "unique_qemu_basenames": sorted({record.argv0_basename for record in records if record.argv0_basename}),
            "findings": len(findings),
        },
        "records": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=None, help="Glob pattern when an input path is a directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_executable_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-absolute", action="store_true")
    parser.add_argument("--allow-row-executable-drift", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.pattern = args.pattern or list(DEFAULT_PATTERNS)

    records, findings = audit(args.paths, args)
    report = build_report(records, findings)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem

    (output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / f"{stem}.md", report)
    write_csv(output_dir / f"{stem}.csv", [asdict(record) for record in records], list(ExecutableRecord.__dataclass_fields__))
    write_csv(output_dir / f"{stem}_findings.csv", [asdict(finding) for finding in findings], list(Finding.__dataclass_fields__))
    write_junit(output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
