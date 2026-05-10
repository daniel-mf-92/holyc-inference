#!/usr/bin/env python3
"""Audit saved QEMU benchmark matrix artifacts.

This host-side tool validates artifacts produced by qemu_benchmark_matrix.py.
It never launches QEMU; it only checks that recorded commands remain air-gapped,
hashes still match argv, and launch rows match the per-build warmup/repeat plan.
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


DEFAULT_PATTERNS = ("qemu_benchmark_matrix*.json",)


@dataclass(frozen=True)
class MatrixArtifact:
    source: str
    status: str
    builds: int
    launches: int
    prompts: int | None
    airgap_ok_builds: int
    finding_count: int


@dataclass(frozen=True)
class MatrixFinding:
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


def object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def string_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def add_finding(findings: list[MatrixFinding], source: Path, kind: str, field: str, detail: str) -> None:
    findings.append(MatrixFinding(str(source), "error", kind, field, detail))


def audit_artifact(path: Path) -> tuple[MatrixArtifact, list[MatrixFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        finding = MatrixFinding(str(path), "error", "load_error", "", error)
        return MatrixArtifact(str(path), "fail", 0, 0, None, 0, 1), [finding]

    findings: list[MatrixFinding] = []
    builds = object_list(payload.get("builds"))
    launches = object_list(payload.get("launches"))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    prompts = int_value(summary.get("prompts"))

    if not builds:
        add_finding(findings, path, "missing_builds", "builds", "matrix artifact must include build rows")
    if not launches:
        add_finding(findings, path, "missing_launches", "launches", "matrix artifact must include launch rows")

    launch_counts_by_build: dict[str, int] = {}
    phase_counts_by_build: dict[str, dict[str, int]] = {}
    command_hashes_by_build: dict[str, set[str]] = {}
    launch_indexes_by_build: dict[str, list[int]] = {}
    for index, row in enumerate(launches, 1):
        build = str(row.get("build") or "")
        if not build:
            add_finding(findings, path, "launch_missing_build", f"launches[{index}].build", "launch row has no build")
            continue
        launch_counts_by_build[build] = launch_counts_by_build.get(build, 0) + 1
        phase = str(row.get("phase") or "")
        phase_counts_by_build.setdefault(build, {})[phase] = phase_counts_by_build.setdefault(build, {}).get(phase, 0) + 1
        command_hashes_by_build.setdefault(build, set()).add(str(row.get("command_sha256") or ""))
        launch_index = int_value(row.get("launch_index"))
        if launch_index is None or launch_index <= 0:
            add_finding(findings, path, "invalid_launch_index", f"launches[{index}].launch_index", "must be a positive integer")
        else:
            launch_indexes_by_build.setdefault(build, []).append(launch_index)
        if not row.get("prompt_id"):
            add_finding(findings, path, "missing_prompt_id", f"launches[{index}].prompt_id", "launch row has no prompt_id")
        if not row.get("prompt_sha256"):
            add_finding(findings, path, "missing_prompt_hash", f"launches[{index}].prompt_sha256", "launch row has no prompt hash")

    seen_builds: set[str] = set()
    airgap_ok_builds = 0
    for index, build in enumerate(builds, 1):
        build_name = str(build.get("build") or "")
        if not build_name:
            add_finding(findings, path, "build_missing_name", f"builds[{index}].build", "build row has no name")
            continue
        if build_name in seen_builds:
            add_finding(findings, path, "duplicate_build", f"builds[{index}].build", f"duplicate build {build_name}")
        seen_builds.add(build_name)

        command = string_list(build.get("command"))
        if command is None:
            add_finding(findings, path, "command_type", f"builds[{index}].command", "command must be a string array")
            command = []
        command_hash = str(build.get("command_sha256") or "")
        if command:
            expected_hash = qemu_prompt_bench.command_hash(command)
            if command_hash != expected_hash:
                add_finding(findings, path, "command_hash", f"builds[{index}].command_sha256", f"expected {expected_hash}, found {command_hash}")
            airgap = qemu_prompt_bench.command_airgap_metadata(command)
            if not airgap["ok"]:
                add_finding(findings, path, "command_airgap", f"builds[{index}].command", "; ".join(airgap["violations"]))
            else:
                airgap_ok_builds += 1
            if build.get("command_airgap_ok") != airgap["ok"]:
                add_finding(findings, path, "recorded_airgap_drift", f"builds[{index}].command_airgap_ok", "recorded value differs from argv audit")

        prompt_count = int_value(build.get("prompt_count"))
        warmup = int_value(build.get("warmup"))
        repeat = int_value(build.get("repeat"))
        launch_count = int_value(build.get("launch_count"))
        if None not in (prompt_count, warmup, repeat):
            expected_launches = prompt_count * (warmup + repeat)  # type: ignore[operator]
            if launch_count != expected_launches:
                add_finding(findings, path, "launch_count_formula", f"builds[{index}].launch_count", f"expected {expected_launches}, found {launch_count}")
            if launch_counts_by_build.get(build_name, 0) != expected_launches:
                add_finding(findings, path, "launch_row_count", "launches", f"{build_name} expected {expected_launches} rows, found {launch_counts_by_build.get(build_name, 0)}")
            phases = phase_counts_by_build.get(build_name, {})
            expected_warmups = prompt_count * warmup  # type: ignore[operator]
            expected_measured = prompt_count * repeat  # type: ignore[operator]
            if phases.get("warmup", 0) != expected_warmups:
                add_finding(findings, path, "warmup_count", "launches.phase", f"{build_name} expected {expected_warmups} warmups, found {phases.get('warmup', 0)}")
            if phases.get("measured", 0) != expected_measured:
                add_finding(findings, path, "measured_count", "launches.phase", f"{build_name} expected {expected_measured} measured rows, found {phases.get('measured', 0)}")
        else:
            add_finding(findings, path, "missing_plan_counts", f"builds[{index}]", "prompt_count, warmup, and repeat must be integers")

        row_hashes = command_hashes_by_build.get(build_name, set())
        if row_hashes and row_hashes != {command_hash}:
            add_finding(findings, path, "launch_command_hash_drift", "launches.command_sha256", f"{build_name} launch hashes {sorted(row_hashes)} differ from build hash {command_hash}")
        indexes = sorted(launch_indexes_by_build.get(build_name, []))
        if indexes and indexes != list(range(1, len(indexes) + 1)):
            add_finding(findings, path, "launch_index_sequence", "launches.launch_index", f"{build_name} launch indexes are not contiguous from 1")

    for build_name in sorted(set(launch_counts_by_build) - seen_builds):
        add_finding(findings, path, "unknown_launch_build", "launches.build", f"launches reference unknown build {build_name}")

    if int_value(summary.get("builds")) is not None and summary["builds"] != len(builds):
        add_finding(findings, path, "summary_build_count", "summary.builds", f"expected {len(builds)}, found {summary['builds']}")
    if int_value(summary.get("launches")) is not None and summary["launches"] != len(launches):
        add_finding(findings, path, "summary_launch_count", "summary.launches", f"expected {len(launches)}, found {summary['launches']}")
    if int_value(summary.get("airgap_ok_builds")) is not None and summary["airgap_ok_builds"] != airgap_ok_builds:
        add_finding(findings, path, "summary_airgap_count", "summary.airgap_ok_builds", f"expected {airgap_ok_builds}, found {summary['airgap_ok_builds']}")

    artifact = MatrixArtifact(
        source=str(path),
        status="fail" if findings else "pass",
        builds=len(builds),
        launches=len(launches),
        prompts=prompts,
        airgap_ok_builds=airgap_ok_builds,
        finding_count=len(findings),
    )
    return artifact, findings


def build_report(paths: list[Path], patterns: list[str], min_artifacts: int) -> dict[str, Any]:
    artifacts: list[MatrixArtifact] = []
    findings: list[MatrixFinding] = []
    for path in iter_input_files(paths, patterns):
        artifact, artifact_findings = audit_artifact(path)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    if len(artifacts) < min_artifacts:
        findings.append(MatrixFinding("", "error", "min_artifacts", "inputs", f"expected at least {min_artifacts} artifacts, found {len(artifacts)}"))
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "passing_artifacts": sum(1 for artifact in artifacts if artifact.status == "pass"),
            "builds": sum(artifact.builds for artifact in artifacts),
            "launches": sum(artifact.launches for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / output_stem
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with base.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MatrixArtifact.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(report["artifacts"])
    with base.with_name(f"{base.name}_findings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MatrixFinding.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(report["findings"])
    lines = [
        "# QEMU Benchmark Matrix Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {report['summary']['artifacts']}",
        f"- Builds: {report['summary']['builds']}",
        f"- Launches: {report['summary']['launches']}",
        f"- Findings: {report['summary']['findings']}",
    ]
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append("- {source}: {kind} {field} ({detail})".format(**finding))
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_benchmark_matrix_audit",
            "tests": str(max(1, len(report["findings"]))),
            "failures": str(len(report["findings"])),
        },
    )
    if report["findings"]:
        for finding in report["findings"]:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding['kind']}:{finding['field']}"})
            failure = ET.SubElement(case, "failure", {"message": finding["detail"]})
            failure.text = finding["detail"]
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_benchmark_matrix_audit"})
    ET.ElementTree(suite).write(base.with_name(f"{base.name}_junit.xml"), encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Matrix artifact files or directories")
    parser.add_argument("--pattern", action="append", dest="patterns", help="Glob to use when an input is a directory")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_benchmark_matrix_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args.inputs, args.patterns or list(DEFAULT_PATTERNS), args.min_artifacts)
    write_report(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
