#!/usr/bin/env python3
"""Audit QEMU prompt benchmark phase summaries against raw rows.

This host-side tool reads saved benchmark JSON artifacts only. It never
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


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class PhaseArtifact:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    warmup_rows: int
    measured_rows: int
    stored_phase_summaries: int
    expected_phase_summaries: int
    checked_fields: int
    mismatched_fields: int
    error: str = ""


@dataclass(frozen=True)
class PhaseFinding:
    source: str
    phase: str
    field: str
    kind: str
    stored: str
    expected: str


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


def bench_run_from_row(row: dict[str, Any]) -> qemu_prompt_bench.BenchRun:
    data = dict(row)
    data["command_airgap_violations"] = tuple(data.get("command_airgap_violations") or ())
    return qemu_prompt_bench.BenchRun(**data)


def runs_from_payload(payload: dict[str, Any], key: str) -> tuple[list[qemu_prompt_bench.BenchRun], str]:
    raw_rows = payload.get(key)
    if not isinstance(raw_rows, list):
        return [], f"missing {key} list"
    runs: list[qemu_prompt_bench.BenchRun] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            return [], f"{key} row {index} must be an object"
        try:
            runs.append(bench_run_from_row(row))
        except TypeError as exc:
            return [], f"{key} row {index} cannot rebuild BenchRun: {exc}"
    return runs, ""


def value_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def values_match(stored: Any, expected: Any, tolerance: float) -> bool:
    if stored is None or expected is None:
        return stored is expected
    if isinstance(stored, bool) or isinstance(expected, bool):
        return stored == expected
    if isinstance(stored, (int, float)) and isinstance(expected, (int, float)):
        if not math.isfinite(float(stored)) or not math.isfinite(float(expected)):
            return stored == expected
        return abs(float(stored) - float(expected)) <= tolerance
    return stored == expected


def phase_map(rows: list[dict[str, Any]], source: Path) -> tuple[dict[str, dict[str, Any]], list[PhaseFinding]]:
    mapped: dict[str, dict[str, Any]] = {}
    findings: list[PhaseFinding] = []
    for index, row in enumerate(rows, 1):
        phase = str(row.get("phase") or "")
        if not phase:
            findings.append(PhaseFinding(str(source), f"row:{index}", "phase", "missing_phase", "", "phase name"))
            continue
        if phase in mapped:
            findings.append(PhaseFinding(str(source), phase, "phase", "duplicate_phase_summary", phase, "unique phase"))
            continue
        mapped[phase] = row
    return mapped, findings


def compare_phase(
    source: Path,
    phase: str,
    stored: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float,
) -> tuple[int, list[PhaseFinding]]:
    findings: list[PhaseFinding] = []
    checked = 0
    for field in sorted(expected):
        checked += 1
        if field not in stored:
            findings.append(PhaseFinding(str(source), phase, field, "missing_field", "", value_text(expected[field])))
            continue
        if not values_match(stored[field], expected[field], tolerance):
            findings.append(
                PhaseFinding(
                    str(source),
                    phase,
                    field,
                    "value_mismatch",
                    value_text(stored[field]),
                    value_text(expected[field]),
                )
            )
    for field in sorted(set(stored) - set(expected)):
        if field.startswith("_"):
            continue
        findings.append(PhaseFinding(str(source), phase, field, "unexpected_field", value_text(stored[field]), ""))
    return checked, findings


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[PhaseArtifact, list[PhaseFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return (
            PhaseArtifact(str(path), "fail", "", "", "", 0, 0, 0, 0, 0, 1, error),
            [PhaseFinding(str(path), "artifact", "", "load_error", "", error)],
        )

    warmups, warmup_error = runs_from_payload(payload, "warmups")
    measured, measured_error = runs_from_payload(payload, "benchmarks")
    if warmup_error or measured_error:
        error = "; ".join(part for part in (warmup_error, measured_error) if part)
        return (
            PhaseArtifact(
                str(path),
                "fail",
                str(payload.get("profile") or ""),
                str(payload.get("model") or ""),
                str(payload.get("quantization") or ""),
                len(warmups),
                len(measured),
                0,
                0,
                0,
                1,
                error,
            ),
            [PhaseFinding(str(path), "artifact", "", "row_error", "", error)],
        )

    stored_rows = payload.get("phase_summaries")
    if not isinstance(stored_rows, list) or not all(isinstance(row, dict) for row in stored_rows):
        stored_rows = []
        findings = [PhaseFinding(str(path), "artifact", "phase_summaries", "missing_phase_summaries", "", "list[object]")]
    else:
        findings = []

    expected_rows = qemu_prompt_bench.phase_summaries(warmups, measured)
    stored_by_phase, phase_findings = phase_map(stored_rows, path)
    expected_by_phase, _ = phase_map(expected_rows, path)
    findings.extend(phase_findings)

    checked_fields = 0
    for phase in sorted(set(stored_by_phase) | set(expected_by_phase)):
        stored = stored_by_phase.get(phase)
        expected = expected_by_phase.get(phase)
        if stored is None:
            findings.append(PhaseFinding(str(path), phase, "phase_summaries", "missing_phase_summary", "", "object"))
            continue
        if expected is None:
            findings.append(PhaseFinding(str(path), phase, "phase_summaries", "unexpected_phase_summary", "object", ""))
            continue
        checked, phase_compare_findings = compare_phase(path, phase, stored, expected, args.float_tolerance)
        checked_fields += checked
        findings.extend(phase_compare_findings)

    return (
        PhaseArtifact(
            str(path),
            "pass" if not findings else "fail",
            str(payload.get("profile") or ""),
            str(payload.get("model") or ""),
            str(payload.get("quantization") or ""),
            len(warmups),
            len(measured),
            len(stored_rows),
            len(expected_rows),
            checked_fields,
            len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[PhaseArtifact], list[PhaseFinding]]:
    artifacts: list[PhaseArtifact] = []
    findings: list[PhaseFinding] = []
    for path in iter_input_files(paths, args.pattern):
        artifact, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    if len(artifacts) < args.min_artifacts:
        findings.append(
            PhaseFinding("-", "inputs", "artifacts", "min_artifacts", str(len(artifacts)), str(args.min_artifacts))
        )
    return artifacts, findings


def write_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [asdict(row) for row in rows]
    fieldnames = list(dicts[0]) if dicts else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(dicts)


def write_json(path: Path, artifacts: list[PhaseArtifact], findings: list[PhaseFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "artifacts": [asdict(row) for row in artifacts],
        "findings": [asdict(row) for row in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, artifacts: list[PhaseArtifact], findings: list[PhaseFinding]) -> None:
    lines = [
        "# QEMU Phase Summary Consistency Audit",
        "",
        f"Generated: {iso_now()}",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Artifacts: {len(artifacts)}",
        f"Findings: {len(findings)}",
        "",
        "| Source | Status | Warmups | Measured | Stored phases | Expected phases | Checked fields | Mismatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for artifact in artifacts:
        lines.append(
            "| {source} | {status} | {warmup_rows} | {measured_rows} | {stored_phase_summaries} | "
            "{expected_phase_summaries} | {checked_fields} | {mismatched_fields} |".format(**asdict(artifact))
        )
    if findings:
        lines.extend(["", "## Findings", "", "| Source | Phase | Field | Kind | Stored | Expected |", "| --- | --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {phase} | {field} | {kind} | {stored} | {expected} |".format(**asdict(finding))
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, artifacts: list[PhaseArtifact], findings: list[PhaseFinding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "qemu_phase_summary_consistency_audit",
            "tests": str(max(1, len(artifacts))),
            "failures": str(len(findings)),
        },
    )
    if artifacts:
        for artifact in artifacts:
            case = ET.SubElement(suite, "testcase", {"name": Path(artifact.source).name})
            artifact_findings = [finding for finding in findings if finding.source == artifact.source]
            if artifact_findings:
                failure = ET.SubElement(case, "failure", {"message": f"{len(artifact_findings)} phase summary finding(s)"})
                failure.text = "\n".join(
                    f"{finding.phase}.{finding.field}: {finding.kind} stored={finding.stored} expected={finding.expected}"
                    for finding in artifact_findings
                )
    else:
        case = ET.SubElement(suite, "testcase", {"name": "inputs"})
        failure = ET.SubElement(case, "failure", {"message": "no artifacts audited"})
        failure.text = "\n".join(f"{finding.kind}: {finding.stored} expected {finding.expected}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path("bench/results")])
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_phase_summary_consistency_audit_latest")
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    parser.add_argument("--min-artifacts", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts, findings = audit(args.paths, args)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", artifacts, findings)
    write_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", artifacts, findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", artifacts, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
