#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for warmup/measured isolation.

This host-side tool reads saved benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class ArtifactAudit:
    source: str
    status: str
    warmup_rows: int
    measured_rows: int
    planned_warmup_launches: int | None
    planned_measured_launches: int | None
    measured_tokens_total: int
    warmup_tokens_total: int
    suite_summary_total_tokens: int | None
    phase_summary_warmup_rows: int | None
    phase_summary_measured_rows: int | None


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


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


def load_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be a JSON object")
    return payload


def row_phase(row: dict[str, Any]) -> str:
    value = row.get("phase")
    return str(value) if value not in (None, "") else "-"


def row_launch_index(row: dict[str, Any]) -> int | None:
    return finite_int(row.get("launch_index"))


def row_tokens_total(rows: list[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        tokens = finite_int(row.get("tokens"))
        if tokens is not None:
            total += tokens
    return total


def phase_summary_rows(payload: dict[str, Any], phase: str) -> int | None:
    summaries = payload.get("phase_summaries")
    if isinstance(summaries, list):
        summary = next((item for item in summaries if isinstance(item, dict) and item.get("phase") == phase), None)
    elif isinstance(summaries, dict):
        summary = summaries.get(phase)
    else:
        return None
    if not isinstance(summary, dict):
        return None
    for key in ("runs", "rows", "count"):
        value = finite_int(summary.get(key))
        if value is not None:
            return value
    return None


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactAudit | None, list[Finding]]:
    findings: list[Finding] = []
    try:
        payload = load_artifact(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, [Finding(str(path), 0, "error", "load_error", str(exc))]

    warmups = payload.get("warmups")
    measured = payload.get("benchmarks")
    if not isinstance(warmups, list):
        findings.append(Finding(str(path), 0, "error", "missing_warmups", "artifact must contain a warmups list"))
        warmups = []
    if not isinstance(measured, list):
        findings.append(Finding(str(path), 0, "error", "missing_benchmarks", "artifact must contain a benchmarks list"))
        measured = []

    warmup_rows = [row for row in warmups if isinstance(row, dict)]
    measured_rows = [row for row in measured if isinstance(row, dict)]
    if len(warmup_rows) != len(warmups):
        findings.append(Finding(str(path), 0, "error", "invalid_warmup_row", "all warmups entries must be objects"))
    if len(measured_rows) != len(measured):
        findings.append(Finding(str(path), 0, "error", "invalid_measured_row", "all benchmarks entries must be objects"))

    for index, row in enumerate(warmup_rows, 1):
        if row_phase(row) != "warmup":
            findings.append(Finding(str(path), index, "error", "warmup_phase_drift", f"warmup row phase is {row_phase(row)!r}"))
    for index, row in enumerate(measured_rows, 1):
        if row_phase(row) != "measured":
            findings.append(Finding(str(path), index, "error", "measured_phase_drift", f"benchmark row phase is {row_phase(row)!r}"))

    planned_warmups = finite_int(payload.get("planned_warmup_launches"))
    planned_measured = finite_int(payload.get("planned_measured_launches"))
    if planned_warmups is not None and planned_warmups != len(warmup_rows):
        findings.append(
            Finding(str(path), 0, "error", "planned_warmup_count_drift", f"planned {planned_warmups}, found {len(warmup_rows)} warmup rows")
        )
    if planned_measured is not None and planned_measured != len(measured_rows):
        findings.append(
            Finding(str(path), 0, "error", "planned_measured_count_drift", f"planned {planned_measured}, found {len(measured_rows)} measured rows")
        )

    warmup_launches = {launch for row in warmup_rows if (launch := row_launch_index(row)) is not None}
    measured_launches = {launch for row in measured_rows if (launch := row_launch_index(row)) is not None}
    overlap = sorted(warmup_launches & measured_launches)
    if overlap:
        findings.append(Finding(str(path), 0, "error", "launch_index_overlap", f"warmup/measured launch_index overlap: {overlap}"))

    measured_tokens = row_tokens_total(measured_rows)
    warmup_tokens = row_tokens_total(warmup_rows)
    suite_summary = payload.get("suite_summary")
    suite_tokens = None
    if isinstance(suite_summary, dict):
        suite_tokens = finite_int(suite_summary.get("total_tokens"))
        if suite_tokens is not None and suite_tokens != measured_tokens:
            findings.append(
                Finding(
                    str(path),
                    0,
                    "error",
                    "suite_summary_warmup_leak",
                    f"suite_summary total_tokens={suite_tokens}, measured tokens={measured_tokens}, warmup tokens={warmup_tokens}",
                )
            )

    phase_warmup_rows = phase_summary_rows(payload, "warmup")
    phase_measured_rows = phase_summary_rows(payload, "measured")
    if args.require_phase_summaries:
        if phase_warmup_rows is None:
            findings.append(Finding(str(path), 0, "error", "missing_warmup_phase_summary", "phase_summaries.warmup row count is absent"))
        if phase_measured_rows is None:
            findings.append(Finding(str(path), 0, "error", "missing_measured_phase_summary", "phase_summaries.measured row count is absent"))
    if phase_warmup_rows is not None and phase_warmup_rows != len(warmup_rows):
        findings.append(Finding(str(path), 0, "error", "warmup_phase_summary_drift", f"phase summary {phase_warmup_rows}, rows {len(warmup_rows)}"))
    if phase_measured_rows is not None and phase_measured_rows != len(measured_rows):
        findings.append(Finding(str(path), 0, "error", "measured_phase_summary_drift", f"phase summary {phase_measured_rows}, rows {len(measured_rows)}"))

    if len(measured_rows) < args.min_measured_rows:
        findings.append(Finding(str(path), 0, "error", "min_measured_rows", f"found {len(measured_rows)}, expected at least {args.min_measured_rows}"))
    if len(warmup_rows) < args.min_warmup_rows:
        findings.append(Finding(str(path), 0, "error", "min_warmup_rows", f"found {len(warmup_rows)}, expected at least {args.min_warmup_rows}"))

    audit = ArtifactAudit(
        source=str(path),
        status="pass" if not findings else "fail",
        warmup_rows=len(warmup_rows),
        measured_rows=len(measured_rows),
        planned_warmup_launches=planned_warmups,
        planned_measured_launches=planned_measured,
        measured_tokens_total=measured_tokens,
        warmup_tokens_total=warmup_tokens,
        suite_summary_total_tokens=suite_tokens,
        phase_summary_warmup_rows=phase_warmup_rows,
        phase_summary_measured_rows=phase_measured_rows,
    )
    return audit, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactAudit], list[Finding]]:
    artifacts: list[ArtifactAudit] = []
    findings: list[Finding] = []
    seen = 0
    for path in iter_input_files(paths, args.pattern):
        seen += 1
        artifact, artifact_findings = audit_artifact(path, args)
        if artifact is not None:
            artifacts.append(artifact)
        findings.extend(artifact_findings)
    if seen < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", f"found {seen}, expected at least {args.min_artifacts}"))
    return artifacts, findings


def summary(artifacts: list[ArtifactAudit], findings: list[Finding]) -> dict[str, Any]:
    return {
        "artifacts": len(artifacts),
        "findings": len(findings),
        "warmup_rows": sum(artifact.warmup_rows for artifact in artifacts),
        "measured_rows": sum(artifact.measured_rows for artifact in artifacts),
        "warmup_tokens_total": sum(artifact.warmup_tokens_total for artifact in artifacts),
        "measured_tokens_total": sum(artifact.measured_tokens_total for artifact in artifacts),
    }


def write_json(path: Path, artifacts: list[ArtifactAudit], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(artifacts, findings),
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, artifacts: list[ArtifactAudit], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Warmup Isolation Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Artifacts: {len(artifacts)}",
        f"Findings: {len(findings)}",
        "",
        "## Artifacts",
        "",
        "| Source | Status | Warmups | Measured | Warmup tokens | Measured tokens | Suite tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for artifact in artifacts:
        lines.append(
            f"| {artifact.source} | {artifact.status} | {artifact.warmup_rows} | {artifact.measured_rows} | "
            f"{artifact.warmup_tokens_total} | {artifact.measured_tokens_total} | {artifact.suite_summary_total_tokens if artifact.suite_summary_total_tokens is not None else '-'} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Detail |", "| --- | ---: | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.detail} |")
    else:
        lines.append("No warmup isolation findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, artifacts: list[ArtifactAudit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArtifactAudit.__dataclass_fields__))
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_warmup_isolation_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "warmup_isolation"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} warmup isolation finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_warmup_isolation_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-warmup-rows", type=int, default=0)
    parser.add_argument("--min-measured-rows", type=int, default=1)
    parser.add_argument("--require-phase-summaries", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if args.min_artifacts < 0 or args.min_warmup_rows < 0 or args.min_measured_rows < 0:
        parser.error("--min-artifacts, --min-warmup-rows, and --min-measured-rows must be >= 0")

    artifacts, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", artifacts, findings)
    write_markdown(args.output_dir / f"{stem}.md", artifacts, findings)
    write_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
