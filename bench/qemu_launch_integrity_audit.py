#!/usr/bin/env python3
"""Audit QEMU benchmark launch plan and observed sequence integrity.

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


@dataclass(frozen=True)
class LaunchArtifact:
    source: str
    status: str
    has_launch_plan: bool
    launch_plan_rows: int
    observed_rows: int
    matched_launches: int
    mismatched_launches: int
    missing_launches: int
    extra_launches: int
    launch_sequence_match: bool
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    field: str
    kind: str
    stored: str
    expected: str
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


def observed_sequence_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "launch_index": row.get("launch_index"),
            "phase": row.get("phase"),
            "prompt_id": row.get("prompt"),
            "prompt_sha256": row.get("prompt_sha256"),
            "prompt_bytes": row.get("prompt_bytes"),
            "expected_tokens": row.get("expected_tokens"),
            "iteration": row.get("iteration"),
        }
        for row in sorted(rows, key=lambda item: int(item.get("launch_index") or 0))
    ]


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def compare_field(
    findings: list[Finding],
    path: Path,
    field: str,
    stored: Any,
    expected: Any,
    detail: str,
) -> None:
    if stored != expected:
        findings.append(Finding(str(path), field, "value_mismatch", text(stored), text(expected), detail))


def rows_from_payload(payload: dict[str, Any], key: str) -> tuple[list[dict[str, Any]], str]:
    raw = payload.get(key)
    if raw is None:
        return [], ""
    if not isinstance(raw, list):
        return [], f"{key} must be a list"
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            return [], f"{key}[{index}] must be an object"
        rows.append(row)
    return rows, ""


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[LaunchArtifact, list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        artifact = LaunchArtifact(str(path), "fail", False, 0, 0, 0, 0, 0, 0, False, error)
        return artifact, [Finding(str(path), "", "load_error", "", error, "could not load benchmark artifact")]

    findings: list[Finding] = []
    raw_plan = payload.get("launch_plan")
    has_launch_plan = isinstance(raw_plan, list)
    if not isinstance(raw_plan, list):
        raw_plan = []
        if args.require_launch_plan:
            findings.append(Finding(str(path), "launch_plan", "missing_launch_plan", text(payload.get("launch_plan")), "list", "launch_plan must be present"))
    plan = [row for row in raw_plan if isinstance(row, dict)]
    if len(plan) != len(raw_plan):
        findings.append(Finding(str(path), "launch_plan", "invalid_plan_row", "", "object rows", "launch_plan contains non-object rows"))

    warmups, warmup_error = rows_from_payload(payload, "warmups")
    benchmarks, bench_error = rows_from_payload(payload, "benchmarks")
    for field, row_error in (("warmups", warmup_error), ("benchmarks", bench_error)):
        if row_error:
            findings.append(Finding(str(path), field, "invalid_rows", "", "list of objects", row_error))

    expected_sequence = qemu_prompt_bench.launch_sequence_from_plan(plan)
    observed_sequence = observed_sequence_from_rows(warmups + benchmarks)
    expected_integrity = qemu_prompt_bench.launch_sequence_integrity(expected_sequence, observed_sequence)

    if has_launch_plan:
        if args.require_sequence_metadata or "launch_plan_sha256" in payload:
            compare_field(
                findings,
                path,
                "launch_plan_sha256",
                payload.get("launch_plan_sha256"),
                qemu_prompt_bench.launch_plan_hash(plan),
                "stored launch_plan_sha256 must match launch_plan",
            )
        if args.require_sequence_metadata or "expected_launch_sequence_sha256" in payload:
            compare_field(
                findings,
                path,
                "expected_launch_sequence_sha256",
                payload.get("expected_launch_sequence_sha256"),
                expected_integrity["expected_launch_sequence_sha256"],
                "stored expected sequence hash must match launch_plan",
            )
        if args.require_sequence_metadata or "observed_launch_sequence_sha256" in payload:
            compare_field(
                findings,
                path,
                "observed_launch_sequence_sha256",
                payload.get("observed_launch_sequence_sha256"),
                expected_integrity["observed_launch_sequence_sha256"],
                "stored observed sequence hash must match warmups plus benchmarks",
            )

        stored_integrity = payload.get("launch_sequence_integrity")
        if not isinstance(stored_integrity, dict):
            if args.require_sequence_metadata:
                findings.append(Finding(str(path), "launch_sequence_integrity", "missing_integrity", text(stored_integrity), "object", "launch_sequence_integrity must be present"))
            stored_integrity = {}
        else:
            for field, expected in expected_integrity.items():
                compare_field(findings, path, f"launch_sequence_integrity.{field}", stored_integrity.get(field), expected, "stored launch_sequence_integrity is stale")

    if has_launch_plan and args.require_match and expected_integrity["launch_sequence_match"] is not True:
        findings.append(
            Finding(
                str(path),
                "launch_sequence_integrity.launch_sequence_match",
                "launch_sequence_mismatch",
                text(expected_integrity["launch_sequence_match"]),
                "true",
                "observed warmup/measured rows do not match launch_plan",
            )
        )
    if len(benchmarks) < args.min_measured_rows:
        findings.append(Finding(str(path), "benchmarks", "min_measured_rows", str(len(benchmarks)), str(args.min_measured_rows), "not enough measured rows"))

    artifact = LaunchArtifact(
        source=str(path),
        status="fail" if findings else "pass",
        has_launch_plan=has_launch_plan,
        launch_plan_rows=len(plan),
        observed_rows=len(warmups) + len(benchmarks),
        matched_launches=int(expected_integrity["matched_launches"]),
        mismatched_launches=int(expected_integrity["mismatched_launches"]),
        missing_launches=int(expected_integrity["missing_launches"]),
        extra_launches=int(expected_integrity["extra_launches"]),
        launch_sequence_match=bool(expected_integrity["launch_sequence_match"]),
    )
    return artifact, findings


def build_report(artifacts: list[LaunchArtifact], findings: list[Finding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "artifacts_with_launch_plan": sum(1 for artifact in artifacts if artifact.has_launch_plan),
            "launch_plan_rows": sum(artifact.launch_plan_rows for artifact in artifacts),
            "observed_rows": sum(artifact.observed_rows for artifact in artifacts),
            "matched_launches": sum(artifact.matched_launches for artifact in artifacts),
            "mismatched_launches": sum(artifact.mismatched_launches for artifact in artifacts),
            "missing_launches": sum(artifact.missing_launches for artifact in artifacts),
            "extra_launches": sum(artifact.extra_launches for artifact in artifacts),
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
        "# QEMU Launch Integrity Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Artifacts with launch plan: {summary['artifacts_with_launch_plan']}",
        f"- Launch plan rows: {summary['launch_plan_rows']}",
        f"- Observed rows: {summary['observed_rows']}",
        f"- Matched launches: {summary['matched_launches']}",
        f"- Mismatched launches: {summary['mismatched_launches']}",
        f"- Missing launches: {summary['missing_launches']}",
        f"- Extra launches: {summary['extra_launches']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Has plan | Plan rows | Observed rows | Match | Missing | Extra |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {has_launch_plan} | {launch_plan_rows} | {observed_rows} | {launch_sequence_match} | {missing_launches} | {extra_launches} |".format(
                **artifact
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['source']} {finding['field']} {finding['detail']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_launch_integrity_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.field}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = f"{finding.source}: {finding.field}: {finding.detail}"
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_launch_integrity"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=["qemu_prompt_bench_latest.json"], help="Directory glob for completed benchmark artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_launch_integrity_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-measured-rows", type=int, default=1)
    parser.add_argument("--require-launch-plan", action="store_true", help="Fail artifacts that predate launch_plan telemetry")
    parser.add_argument("--require-sequence-metadata", action="store_true", help="Fail artifacts missing stored launch sequence hashes/integrity metadata")
    parser.add_argument("--require-match", action="store_true", help="Fail when observed launches do not exactly match launch_plan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: list[LaunchArtifact] = []
    findings: list[Finding] = []
    for path in sorted(iter_input_files(args.inputs, args.pattern)):
        artifact, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    if len(artifacts) < args.min_artifacts:
        findings.append(Finding("", "inputs", "min_artifacts", str(len(artifacts)), str(args.min_artifacts), "not enough benchmark artifacts matched"))
    report = build_report(artifacts, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", artifacts, list(LaunchArtifact.__dataclass_fields__))
    write_csv(args.output_dir / f"{stem}_findings.csv", findings, list(Finding.__dataclass_fields__))
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
