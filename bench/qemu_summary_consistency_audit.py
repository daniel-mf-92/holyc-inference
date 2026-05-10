#!/usr/bin/env python3
"""Audit QEMU prompt benchmark summaries against raw benchmark rows.

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


@dataclass(frozen=True)
class ConsistencyArtifact:
    source: str
    status: str
    profile: str
    model: str
    quantization: str
    measured_rows: int
    stored_prompt_summaries: int
    expected_prompt_summaries: int
    checked_fields: int
    mismatched_fields: int
    error: str = ""


@dataclass(frozen=True)
class ConsistencyFinding:
    source: str
    scope: str
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


def measured_runs(payload: dict[str, Any]) -> tuple[list[qemu_prompt_bench.BenchRun], str]:
    raw_rows = payload.get("benchmarks")
    if not isinstance(raw_rows, list):
        return [], "missing benchmarks list"
    runs: list[qemu_prompt_bench.BenchRun] = []
    for index, row in enumerate(raw_rows):
        if not isinstance(row, dict):
            return [], f"benchmark row {index} must be an object"
        if str(row.get("phase") or "measured") != "measured":
            continue
        try:
            runs.append(bench_run_from_row(row))
        except TypeError as exc:
            return [], f"benchmark row {index} cannot rebuild BenchRun: {exc}"
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


def compare_dicts(
    source: Path,
    scope: str,
    stored: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float,
) -> tuple[int, list[ConsistencyFinding]]:
    findings: list[ConsistencyFinding] = []
    checked = 0
    for field in sorted(expected):
        checked += 1
        if field not in stored:
            findings.append(
                ConsistencyFinding(str(source), scope, field, "missing_field", "", value_text(expected[field]))
            )
            continue
        if not values_match(stored[field], expected[field], tolerance):
            findings.append(
                ConsistencyFinding(
                    str(source),
                    scope,
                    field,
                    "value_mismatch",
                    value_text(stored[field]),
                    value_text(expected[field]),
                )
            )
    for field in sorted(set(stored) - set(expected)):
        if field.startswith("_"):
            continue
        findings.append(
            ConsistencyFinding(str(source), scope, field, "unexpected_field", value_text(stored[field]), "")
        )
    return checked, findings


def prompt_summary_map(summaries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        prompt = str(summary.get("prompt") or "")
        if prompt:
            mapped[prompt] = summary
    return mapped


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ConsistencyArtifact, list[ConsistencyFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return (
            ConsistencyArtifact(str(path), "fail", "", "", "", 0, 0, 0, 0, 1, error),
            [ConsistencyFinding(str(path), "artifact", "", "load_error", "", error)],
        )

    runs, run_error = measured_runs(payload)
    if run_error:
        return (
            ConsistencyArtifact(
                str(path),
                "fail",
                str(payload.get("profile") or ""),
                str(payload.get("model") or ""),
                str(payload.get("quantization") or ""),
                0,
                0,
                0,
                0,
                1,
                run_error,
            ),
            [ConsistencyFinding(str(path), "benchmarks", "", "row_error", "", run_error)],
        )

    stored_suite = payload.get("suite_summary")
    stored_summaries = payload.get("summaries")
    findings: list[ConsistencyFinding] = []
    checked_fields = 0
    if not isinstance(stored_suite, dict):
        findings.append(ConsistencyFinding(str(path), "suite", "suite_summary", "missing_summary", "", "object"))
        stored_suite = {}
    if not isinstance(stored_summaries, list) or not all(isinstance(row, dict) for row in stored_summaries):
        findings.append(ConsistencyFinding(str(path), "prompt", "summaries", "missing_summary", "", "list[object]"))
        stored_summaries = []

    expected_suite = qemu_prompt_bench.suite_summary(runs)
    expected_summaries = qemu_prompt_bench.summarize_runs(runs)
    checked, suite_findings = compare_dicts(path, "suite", stored_suite, expected_suite, args.float_tolerance)
    checked_fields += checked
    findings.extend(suite_findings)

    stored_by_prompt = prompt_summary_map(stored_summaries)
    expected_by_prompt = prompt_summary_map(expected_summaries)
    for prompt in sorted(set(stored_by_prompt) | set(expected_by_prompt)):
        stored = stored_by_prompt.get(prompt)
        expected = expected_by_prompt.get(prompt)
        if stored is None:
            findings.append(
                ConsistencyFinding(str(path), f"prompt:{prompt}", "summaries", "missing_prompt_summary", "", "object")
            )
            continue
        if expected is None:
            findings.append(
                ConsistencyFinding(
                    str(path), f"prompt:{prompt}", "summaries", "unexpected_prompt_summary", "object", ""
                )
            )
            continue
        checked, prompt_findings = compare_dicts(
            path,
            f"prompt:{prompt}",
            stored,
            expected,
            args.float_tolerance,
        )
        checked_fields += checked
        findings.extend(prompt_findings)

    if len(runs) < args.min_measured_rows:
        findings.append(
            ConsistencyFinding(
                str(path),
                "artifact",
                "measured_rows",
                "min_measured_rows",
                str(len(runs)),
                str(args.min_measured_rows),
            )
        )

    status = "fail" if findings else "pass"
    return (
        ConsistencyArtifact(
            source=str(path),
            status=status,
            profile=str(payload.get("profile") or ""),
            model=str(payload.get("model") or ""),
            quantization=str(payload.get("quantization") or ""),
            measured_rows=len(runs),
            stored_prompt_summaries=len(stored_by_prompt),
            expected_prompt_summaries=len(expected_by_prompt),
            checked_fields=checked_fields,
            mismatched_fields=len(findings),
        ),
        findings,
    )


def build_report(artifacts: list[ConsistencyArtifact], findings: list[ConsistencyFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "passing_artifacts": sum(1 for artifact in artifacts if artifact.status == "pass"),
            "failing_artifacts": sum(1 for artifact in artifacts if artifact.status != "pass"),
            "measured_rows": sum(artifact.measured_rows for artifact in artifacts),
            "checked_fields": sum(artifact.checked_fields for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifact_csv(path: Path, artifacts: list[ConsistencyArtifact]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ConsistencyArtifact.__dataclass_fields__))
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[ConsistencyFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ConsistencyFinding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Summary Consistency Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Measured rows: {summary['measured_rows']}",
        f"- Checked fields: {summary['checked_fields']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Measured rows | Prompt summaries | Checked fields | Findings |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {measured_rows} | {stored_prompt_summaries}/{expected_prompt_summaries} | {checked_fields} | {mismatched_fields} |".format(
                **artifact
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"][:50]:
            lines.append(
                "- {kind}: {scope} {field} stored={stored!r} expected={expected!r}".format(**finding)
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[ConsistencyFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_summary_consistency_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.scope}:{finding.field}"})
            message = f"stored={finding.stored!r} expected={finding.expected!r}"
            failure = ET.SubElement(case, "failure", {"message": message})
            failure.text = message
    else:
        ET.SubElement(suite, "testcase", {"name": "summary_consistency"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON files or directories to audit")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench*.json"])
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_summary_consistency_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-measured-rows", type=int, default=1)
    parser.add_argument("--float-tolerance", type=float, default=1e-9)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_artifacts < 0 or args.min_measured_rows < 0 or args.float_tolerance < 0:
        print("minimums and --float-tolerance must be non-negative", file=sys.stderr)
        return 2

    artifacts: list[ConsistencyArtifact] = []
    findings: list[ConsistencyFinding] = []
    for path in iter_input_files(args.inputs, args.pattern):
        artifact, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)

    if len(artifacts) < args.min_artifacts:
        findings.append(
            ConsistencyFinding(
                "",
                "artifact",
                "min_artifacts",
                "min_artifacts",
                str(len(artifacts)),
                str(args.min_artifacts),
            )
        )
    report = build_report(artifacts, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.output_dir / args.output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_artifact_csv(output_base.with_suffix(".csv"), artifacts)
    write_findings_csv(output_base.with_name(f"{output_base.name}_findings.csv"), findings)
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), findings)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
