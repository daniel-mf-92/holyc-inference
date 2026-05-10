#!/usr/bin/env python3
"""Audit QEMU benchmark artifacts for launch order integrity.

This host-side tool reads saved QEMU benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class LaunchRow:
    source: str
    row: int
    list_name: str
    phase: str
    launch_index: int | None
    prompt: str
    iteration: int | None
    timestamp: str
    wall_elapsed_us: int | None
    start_timestamp: str
    end_timestamp: str


@dataclass(frozen=True)
class ArtifactAudit:
    source: str
    status: str
    launch_rows: int
    warmup_rows: int
    measured_rows: int
    planned_total_launches: int | None
    planned_warmup_launches: int | None
    planned_measured_launches: int | None
    min_launch_index: int | None
    max_launch_index: int | None
    timestamp_rows: int
    error: str = ""


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


def parse_iso_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def fmt_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def row_text(row: dict[str, Any], key: str, default: str = "-") -> str:
    value = row.get(key)
    return str(value) if value not in (None, "") else default


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


def collect_rows(path: Path, payload: dict[str, Any]) -> tuple[list[LaunchRow], list[Finding]]:
    rows: list[LaunchRow] = []
    findings: list[Finding] = []
    ordinal = 0
    for list_name in ("warmups", "benchmarks"):
        raw_rows = payload.get(list_name)
        if not isinstance(raw_rows, list):
            findings.append(Finding(str(path), 0, "error", f"missing_{list_name}", f"{list_name} must be a list"))
            continue
        for item in raw_rows:
            ordinal += 1
            if not isinstance(item, dict):
                findings.append(Finding(str(path), ordinal, "error", "row_type", f"{list_name} row must be an object"))
                continue
            launch_index = finite_int(item.get("launch_index"))
            wall_elapsed_us = finite_int(item.get("wall_elapsed_us"))
            end = parse_iso_utc(item.get("timestamp"))
            start = end - timedelta(microseconds=wall_elapsed_us) if end and wall_elapsed_us is not None else None
            rows.append(
                LaunchRow(
                    source=str(path),
                    row=ordinal,
                    list_name=list_name,
                    phase=row_text(item, "phase", "warmup" if list_name == "warmups" else "measured").lower(),
                    launch_index=launch_index,
                    prompt=row_text(item, "prompt"),
                    iteration=finite_int(item.get("iteration")),
                    timestamp=row_text(item, "timestamp", ""),
                    wall_elapsed_us=wall_elapsed_us,
                    start_timestamp=fmt_dt(start),
                    end_timestamp=fmt_dt(end),
                )
            )
    return rows, findings


def launch_label(row: LaunchRow) -> str:
    return f"{row.list_name}[{row.row}] phase={row.phase} prompt={row.prompt} iteration={row.iteration or '-'}"


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[ArtifactAudit, list[LaunchRow], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        artifact = ArtifactAudit(str(path), "fail", 0, 0, 0, None, None, None, None, None, 0, error)
        return artifact, [], [Finding(str(path), 0, "error", "load_error", error)]

    rows, findings = collect_rows(path, payload)
    launch_rows = [row for row in rows if row.launch_index is not None]
    launch_indexes = [row.launch_index for row in launch_rows if row.launch_index is not None]
    by_launch: dict[int, list[LaunchRow]] = {}
    for row in launch_rows:
        assert row.launch_index is not None
        by_launch.setdefault(row.launch_index, []).append(row)
        if row.launch_index <= 0:
            findings.append(Finding(str(path), row.row, "error", "invalid_launch_index", f"{launch_label(row)} has non-positive launch_index"))
    for row in rows:
        if row.launch_index is None:
            findings.append(Finding(str(path), row.row, "error", "missing_launch_index", f"{launch_label(row)} is missing integer launch_index"))
        if row.wall_elapsed_us is None or row.wall_elapsed_us <= 0:
            findings.append(Finding(str(path), row.row, "error", "invalid_wall_elapsed_us", f"{launch_label(row)} has invalid wall_elapsed_us"))
        if not row.end_timestamp:
            findings.append(Finding(str(path), row.row, "error", "invalid_timestamp", f"{launch_label(row)} has invalid timestamp"))

    for launch_index, grouped in sorted(by_launch.items()):
        if len(grouped) > 1:
            labels = ", ".join(launch_label(row) for row in grouped)
            findings.append(Finding(str(path), grouped[0].row, "error", "duplicate_launch_index", f"launch_index {launch_index} appears in {labels}"))

    if launch_indexes:
        expected = list(range(1, max(launch_indexes) + 1))
        missing = sorted(set(expected) - set(launch_indexes))
        if missing:
            findings.append(Finding(str(path), 0, "error", "launch_index_gap", f"missing launch_index values: {missing}"))
        if min(launch_indexes) != 1:
            findings.append(Finding(str(path), 0, "error", "launch_index_start", f"first launch_index is {min(launch_indexes)}, expected 1"))

    planned_total = finite_int(payload.get("planned_total_launches"))
    planned_warmups = finite_int(payload.get("planned_warmup_launches"))
    planned_measured = finite_int(payload.get("planned_measured_launches"))
    warmup_rows = [row for row in rows if row.list_name == "warmups"]
    measured_rows = [row for row in rows if row.list_name == "benchmarks"]
    if planned_total is not None and planned_total != len(rows):
        findings.append(Finding(str(path), 0, "error", "planned_total_drift", f"planned {planned_total}, found {len(rows)} launch rows"))
    if planned_warmups is not None and planned_warmups != len(warmup_rows):
        findings.append(Finding(str(path), 0, "error", "planned_warmup_drift", f"planned {planned_warmups}, found {len(warmup_rows)} warmup rows"))
    if planned_measured is not None and planned_measured != len(measured_rows):
        findings.append(Finding(str(path), 0, "error", "planned_measured_drift", f"planned {planned_measured}, found {len(measured_rows)} measured rows"))

    max_warmup = max((row.launch_index for row in warmup_rows if row.launch_index is not None), default=None)
    min_measured = min((row.launch_index for row in measured_rows if row.launch_index is not None), default=None)
    if max_warmup is not None and min_measured is not None and max_warmup >= min_measured:
        findings.append(Finding(str(path), 0, "error", "warmup_after_measured", f"max warmup launch_index {max_warmup} >= min measured launch_index {min_measured}"))

    sorted_rows = sorted((row for row in rows if row.launch_index is not None), key=lambda row: row.launch_index or 0)
    previous_end: datetime | None = None
    previous_start: datetime | None = None
    previous_row: LaunchRow | None = None
    tolerance = timedelta(microseconds=args.timestamp_tolerance_us)
    overlap_tolerance = timedelta(microseconds=args.overlap_tolerance_us)
    for row in sorted_rows:
        end = parse_iso_utc(row.timestamp)
        if end is None:
            continue
        start = end - timedelta(microseconds=row.wall_elapsed_us or 0)
        if previous_end and end + tolerance < previous_end:
            findings.append(Finding(str(path), row.row, "error", "timestamp_regressed", f"{launch_label(row)} ended before previous launch_index {previous_row.launch_index if previous_row else '-'}"))
        if args.check_interval_overlap and previous_end and start + overlap_tolerance < previous_end:
            findings.append(Finding(str(path), row.row, "error", "launch_interval_overlap", f"{launch_label(row)} starts before previous launch completed"))
        if previous_start and start + tolerance < previous_start:
            findings.append(Finding(str(path), row.row, "error", "start_timestamp_regressed", f"{launch_label(row)} starts before previous launch start"))
        previous_end = end
        previous_start = start
        previous_row = row

    timestamp_rows = sum(1 for row in rows if row.end_timestamp)
    artifact = ArtifactAudit(
        source=str(path),
        status="pass" if not findings else "fail",
        launch_rows=len(rows),
        warmup_rows=len(warmup_rows),
        measured_rows=len(measured_rows),
        planned_total_launches=planned_total,
        planned_warmup_launches=planned_warmups,
        planned_measured_launches=planned_measured,
        min_launch_index=min(launch_indexes) if launch_indexes else None,
        max_launch_index=max(launch_indexes) if launch_indexes else None,
        timestamp_rows=timestamp_rows,
    )
    return artifact, rows, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactAudit], list[LaunchRow], list[Finding]]:
    artifacts: list[ArtifactAudit] = []
    rows: list[LaunchRow] = []
    findings: list[Finding] = []
    seen = 0
    for path in iter_input_files(paths, args.pattern):
        seen += 1
        artifact, artifact_rows, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        rows.extend(artifact_rows)
        findings.extend(artifact_findings)
    if seen < args.min_artifacts:
        findings.append(Finding("-", 0, "error", "min_artifacts", f"found {seen}, expected at least {args.min_artifacts}"))
    return artifacts, rows, findings


def summary(artifacts: list[ArtifactAudit], rows: list[LaunchRow], findings: list[Finding]) -> dict[str, Any]:
    return {
        "artifacts": len(artifacts),
        "failed_artifacts": sum(1 for artifact in artifacts if artifact.status != "pass"),
        "launch_rows": len(rows),
        "warmup_rows": sum(artifact.warmup_rows for artifact in artifacts),
        "measured_rows": sum(artifact.measured_rows for artifact in artifacts),
        "timestamp_rows": sum(artifact.timestamp_rows for artifact in artifacts),
        "findings": len(findings),
    }


def write_json(path: Path, artifacts: list[ArtifactAudit], rows: list[LaunchRow], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": summary(artifacts, rows, findings),
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "rows": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, artifacts: list[ArtifactAudit], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Launch Order Audit",
        "",
        f"Status: {'pass' if not findings else 'fail'}",
        f"Artifacts: {len(artifacts)}",
        f"Findings: {len(findings)}",
        "",
        "## Artifacts",
        "",
        "| Source | Status | Launches | Warmups | Measured | Planned | Timestamp rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for artifact in artifacts:
        lines.append(
            f"| {artifact.source} | {artifact.status} | {artifact.launch_rows} | {artifact.warmup_rows} | "
            f"{artifact.measured_rows} | {artifact.planned_total_launches if artifact.planned_total_launches is not None else '-'} | {artifact.timestamp_rows} |"
        )
    lines.extend(["", "## Findings", ""])
    if findings:
        lines.extend(["| Source | Row | Kind | Detail |", "| --- | ---: | --- | --- |"])
        for finding in findings:
            lines.append(f"| {finding.source} | {finding.row} | {finding.kind} | {finding.detail} |")
    else:
        lines.append("No launch order findings.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, artifacts: list[ArtifactAudit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ArtifactAudit.__dataclass_fields__))
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_rows_csv(path: Path, rows: list[LaunchRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LaunchRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


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
            "name": "holyc_qemu_launch_order_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "launch_order"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} launch order finding(s)"})
        failure.text = "\n".join(f"{finding.source}:{finding.row}: {finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Benchmark JSON artifacts or directories")
    parser.add_argument("--pattern", action="append", default=[], help="Directory glob pattern; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_launch_order_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--timestamp-tolerance-us", type=int, default=1_000_000, help="Allowed timestamp ordering skew")
    parser.add_argument("--overlap-tolerance-us", type=int, default=1_000_000, help="Allowed interval overlap when timestamps are second-resolution")
    parser.add_argument("--check-interval-overlap", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.pattern:
        args.pattern = list(DEFAULT_PATTERNS)
    if args.min_artifacts < 0 or args.timestamp_tolerance_us < 0 or args.overlap_tolerance_us < 0:
        parser.error("--min-artifacts, --timestamp-tolerance-us, and --overlap-tolerance-us must be >= 0")

    artifacts, rows, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", artifacts, rows, findings)
    write_markdown(args.output_dir / f"{stem}.md", artifacts, findings)
    write_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_rows_csv(args.output_dir / f"{stem}_rows.csv", rows)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
