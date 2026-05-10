#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for identity drift.

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


IDENTITY_FIELDS = ("profile", "model", "quantization")


@dataclass(frozen=True)
class IdentityArtifact:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    commit: str
    command_sha256: str
    rows: int
    warmup_rows: int
    measured_rows: int
    identity_fields_checked: int
    command_hashes_checked: int
    commit_fields_checked: int
    error: str = ""


@dataclass(frozen=True)
class IdentityFinding:
    source: str
    row: str
    field: str
    severity: str
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


def row_name(index: int, row: dict[str, Any]) -> str:
    phase = str(row.get("phase") or "measured")
    prompt = str(row.get("prompt") or "")
    launch = row.get("launch_index")
    if prompt:
        return f"{phase}[{index}]:{prompt}:launch={launch}"
    return f"{phase}[{index}]:launch={launch}"


def benchmark_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for key in ("warmups", "benchmarks"):
        value = payload.get(key, [])
        if value is None:
            continue
        if not isinstance(value, list):
            return [], f"{key} must be a list"
        for index, row in enumerate(value):
            if not isinstance(row, dict):
                return [], f"{key}[{index}] must be an object"
            rows.append(row)
    return rows, ""


def empty_identity_fields(payload: dict[str, Any], source: Path) -> list[IdentityFinding]:
    findings: list[IdentityFinding] = []
    for field in IDENTITY_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            findings.append(
                IdentityFinding(str(source), "artifact", field, "error", "empty_identity", f"top-level {field} is empty")
            )
    return findings


def command_hash_findings(source: Path, row_label: str, command: Any, expected_hash: Any) -> list[IdentityFinding]:
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return [
            IdentityFinding(
                str(source),
                row_label,
                "command",
                "error",
                "command_type",
                "command must be a list of strings",
            )
        ]
    actual_hash = qemu_prompt_bench.command_hash(command)
    if expected_hash != actual_hash:
        return [
            IdentityFinding(
                str(source),
                row_label,
                "command_sha256",
                "error",
                "command_hash",
                f"stored command_sha256 {expected_hash!r} does not match computed {actual_hash}",
            )
        ]
    return []


def audit_artifact(path: Path) -> tuple[IdentityArtifact, list[IdentityFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return (
            IdentityArtifact(str(path), "fail", "", "", "", "", "", 0, 0, 0, 0, 0, 0, error),
            [IdentityFinding(str(path), "artifact", "", "error", "load_error", error)],
        )

    findings = empty_identity_fields(payload, path)
    rows, row_error = benchmark_rows(payload)
    if row_error:
        findings.append(IdentityFinding(str(path), "artifact", "rows", "error", "row_error", row_error))
        rows = []

    artifact_command = payload.get("command")
    artifact_command_sha256 = payload.get("command_sha256")
    if artifact_command is not None or artifact_command_sha256 is not None:
        findings.extend(command_hash_findings(path, "artifact", artifact_command, artifact_command_sha256))

    identity_checks = 0
    command_checks = 1 if artifact_command is not None or artifact_command_sha256 is not None else 0
    commit_checks = 0
    artifact_commit = payload.get("commit")

    for index, row in enumerate(rows):
        label = row_name(index, row)
        for field in IDENTITY_FIELDS:
            identity_checks += 1
            if row.get(field) != payload.get(field):
                findings.append(
                    IdentityFinding(
                        str(path),
                        label,
                        field,
                        "error",
                        "identity_drift",
                        f"row {field} {row.get(field)!r} does not match artifact {payload.get(field)!r}",
                    )
                )

        row_command = row.get("command")
        row_command_sha256 = row.get("command_sha256")
        if row_command is not None or row_command_sha256 is not None:
            command_checks += 1
            findings.extend(command_hash_findings(path, label, row_command, row_command_sha256))
            if artifact_command_sha256 and row_command_sha256 != artifact_command_sha256:
                findings.append(
                    IdentityFinding(
                        str(path),
                        label,
                        "command_sha256",
                        "error",
                        "command_identity_drift",
                        "row command_sha256 does not match artifact command_sha256",
                    )
                )

        if artifact_commit not in (None, ""):
            commit_checks += 1
            if row.get("commit") != artifact_commit:
                findings.append(
                    IdentityFinding(
                        str(path),
                        label,
                        "commit",
                        "error",
                        "commit_drift",
                        f"row commit {row.get('commit')!r} does not match artifact {artifact_commit!r}",
                    )
                )

    warmup_rows = sum(1 for row in rows if str(row.get("phase") or "") == "warmup")
    artifact = IdentityArtifact(
        source=str(path),
        status="fail" if findings else "pass",
        profile=str(payload.get("profile") or ""),
        model=str(payload.get("model") or ""),
        quantization=str(payload.get("quantization") or ""),
        commit=str(artifact_commit or ""),
        command_sha256=str(artifact_command_sha256 or ""),
        rows=len(rows),
        warmup_rows=warmup_rows,
        measured_rows=len(rows) - warmup_rows,
        identity_fields_checked=identity_checks,
        command_hashes_checked=command_checks,
        commit_fields_checked=commit_checks,
        error=row_error,
    )
    return artifact, findings


def write_csv(path: Path, rows: list[Any], fallback_fields: list[str]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# QEMU Identity Audit",
        "",
        f"- Status: {payload['status']}",
        f"- Artifacts: {payload['summary']['artifacts']}",
        f"- Rows: {payload['summary']['rows']}",
        f"- Findings: {len(payload['findings'])}",
        "",
        "## Artifacts",
        "",
        "| Source | Status | Profile | Model | Quantization | Rows | Command hashes checked |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for artifact in payload["artifacts"]:
        lines.append(
            "| {source} | {status} | {profile} | {model} | {quantization} | {rows} | {command_hashes_checked} |".format(
                **artifact
            )
        )
    lines.extend(["", "## Findings", ""])
    if payload["findings"]:
        for finding in payload["findings"]:
            lines.append(
                f"- {finding['severity']} {finding['kind']} {finding['source']}:{finding['row']} {finding['field']}: {finding['detail']}"
            )
    else:
        lines.append("No identity drift findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[IdentityFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {"name": "holyc_qemu_identity_audit", "tests": "1", "failures": "1" if findings else "0"},
    )
    case = ET.SubElement(suite, "testcase", {"name": "artifact_identity"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} identity finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.source}:{finding.row} {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark artifact JSON files or directories")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*.json"], help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_identity_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[IdentityArtifact] = []
    findings: list[IdentityFinding] = []

    for path in iter_input_files(args.inputs, args.pattern):
        artifact, artifact_findings = audit_artifact(path)
        artifacts.append(artifact)
        findings.extend(artifact_findings)

    rows = sum(artifact.rows for artifact in artifacts)
    if len(artifacts) < args.min_artifacts:
        findings.append(
            IdentityFinding("", "summary", "artifacts", "error", "min_artifacts", f"found {len(artifacts)}, expected at least {args.min_artifacts}")
        )
    if rows < args.min_rows:
        findings.append(
            IdentityFinding("", "summary", "rows", "error", "min_rows", f"found {rows}, expected at least {args.min_rows}")
        )

    status = "fail" if findings else "pass"
    payload = {
        "generated_at": iso_now(),
        "status": status,
        "summary": {
            "artifacts": len(artifacts),
            "rows": rows,
            "warmup_rows": sum(artifact.warmup_rows for artifact in artifacts),
            "measured_rows": sum(artifact.measured_rows for artifact in artifacts),
            "identity_fields_checked": sum(artifact.identity_fields_checked for artifact in artifacts),
            "command_hashes_checked": sum(artifact.command_hashes_checked for artifact in artifacts),
            "commit_fields_checked": sum(artifact.commit_fields_checked for artifact in artifacts),
            "findings": len(findings),
        },
        "gates": {"min_artifacts": args.min_artifacts, "min_rows": args.min_rows},
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }

    stem = args.output_stem
    (args.output_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / f"{stem}.csv", artifacts, list(IdentityArtifact.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", findings, list(IdentityFinding.__dataclass_fields__))
    write_markdown(args.output_dir / f"{stem}.md", payload)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
