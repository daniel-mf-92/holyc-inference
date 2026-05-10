#!/usr/bin/env python3
"""Audit QEMU benchmark artifact timestamps and row chronology.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)\.json$")


@dataclass(frozen=True)
class TimestampArtifact:
    source: str
    status: str
    generated_at: str
    filename_stamp: str
    row_count: int
    parsed_row_timestamps: int
    earliest_row_timestamp: str
    latest_row_timestamp: str
    max_row_skew_seconds: float
    min_row_skew_seconds: float
    error: str = ""


@dataclass(frozen=True)
class TimestampFinding:
    source: str
    scope: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso_utc(value: Any) -> tuple[datetime | None, str]:
    if not isinstance(value, str) or not value:
        return None, "missing timestamp"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        return None, str(exc)
    if parsed.tzinfo is None:
        return None, "timestamp must include timezone"
    return parsed.astimezone(timezone.utc), ""


def stamp_from_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def filename_stamp(path: Path) -> str:
    match = STAMP_RE.search(path.name)
    return match.group(1) if match else ""


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


def row_identity(index: int, row: dict[str, Any]) -> str:
    launch = row.get("launch_index")
    phase = row.get("phase")
    prompt = row.get("prompt")
    parts = [f"row={index}"]
    if launch is not None:
        parts.append(f"launch_index={launch}")
    if phase:
        parts.append(f"phase={phase}")
    if prompt:
        parts.append(f"prompt={prompt}")
    return " ".join(parts)


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[TimestampArtifact, list[TimestampFinding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return (
            TimestampArtifact(str(path), "fail", "", filename_stamp(path), 0, 0, "", "", 0.0, 0.0, error),
            [TimestampFinding(str(path), "artifact", "load_error", error)],
        )

    findings: list[TimestampFinding] = []
    generated_at_text = str(payload.get("generated_at") or "")
    generated_at, generated_error = parse_iso_utc(generated_at_text)
    if generated_error:
        findings.append(TimestampFinding(str(path), "artifact", "invalid_generated_at", generated_error))

    now = datetime.now(timezone.utc)
    if generated_at and (generated_at - now).total_seconds() > args.max_future_seconds:
        findings.append(
            TimestampFinding(
                str(path),
                "artifact",
                "generated_at_in_future",
                f"generated_at is more than {args.max_future_seconds}s in the future",
            )
        )

    stamp = filename_stamp(path)
    if stamp and generated_at and stamp != stamp_from_datetime(generated_at):
        findings.append(
            TimestampFinding(
                str(path),
                "artifact",
                "filename_stamp_mismatch",
                f"filename stamp {stamp} does not match generated_at {stamp_from_datetime(generated_at)}",
            )
        )

    rows = payload.get("benchmarks")
    if not isinstance(rows, list):
        findings.append(TimestampFinding(str(path), "benchmarks", "missing_benchmarks", "benchmarks must be a list"))
        rows = []

    parsed_rows: list[datetime] = []
    previous: datetime | None = None
    max_skew = 0.0
    min_skew = 0.0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            findings.append(TimestampFinding(str(path), f"row:{index}", "invalid_row", "benchmark row must be an object"))
            continue
        timestamp, timestamp_error = parse_iso_utc(row.get("timestamp"))
        scope = row_identity(index, row)
        if timestamp_error:
            findings.append(TimestampFinding(str(path), scope, "invalid_row_timestamp", timestamp_error))
            continue
        assert timestamp is not None
        parsed_rows.append(timestamp)
        if previous and timestamp < previous:
            findings.append(
                TimestampFinding(
                    str(path),
                    scope,
                    "row_timestamp_regressed",
                    f"{timestamp.isoformat()} is before previous row {previous.isoformat()}",
                )
            )
        previous = timestamp
        if generated_at:
            skew = (timestamp - generated_at).total_seconds()
            max_skew = max(max_skew, skew)
            min_skew = min(min_skew, skew)
            if skew > args.max_row_after_generated_at_seconds:
                findings.append(
                    TimestampFinding(
                        str(path),
                        scope,
                        "row_after_generated_at",
                        f"row timestamp is {skew:.3f}s after generated_at",
                    )
                )
            if -skew > args.max_row_before_generated_at_seconds:
                findings.append(
                    TimestampFinding(
                        str(path),
                        scope,
                        "row_before_generated_at",
                        f"row timestamp is {-skew:.3f}s before generated_at",
                    )
                )

    if args.require_rows and not parsed_rows:
        findings.append(TimestampFinding(str(path), "benchmarks", "no_row_timestamps", "no parseable row timestamps found"))

    status = "fail" if findings else "pass"
    return (
        TimestampArtifact(
            source=str(path),
            status=status,
            generated_at=generated_at_text,
            filename_stamp=stamp,
            row_count=len(rows),
            parsed_row_timestamps=len(parsed_rows),
            earliest_row_timestamp=parsed_rows[0].isoformat().replace("+00:00", "Z") if parsed_rows else "",
            latest_row_timestamp=parsed_rows[-1].isoformat().replace("+00:00", "Z") if parsed_rows else "",
            max_row_skew_seconds=round(max_skew, 6),
            min_row_skew_seconds=round(min_skew, 6),
        ),
        findings,
    )


def build_report(artifacts: list[TimestampArtifact], findings: list[TimestampFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "findings": len(findings),
            "rows": sum(artifact.row_count for artifact in artifacts),
            "parsed_row_timestamps": sum(artifact.parsed_row_timestamps for artifact in artifacts),
            "failed_artifacts": sum(1 for artifact in artifacts if artifact.status != "pass"),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, artifacts: list[TimestampArtifact]) -> None:
    fieldnames = list(TimestampArtifact.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[TimestampFinding]) -> None:
    fieldnames = list(TimestampFinding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Timestamp Audit",
        "",
        f"- Status: {report['status']}",
        f"- Artifacts: {summary['artifacts']}",
        f"- Rows: {summary['rows']}",
        f"- Parsed row timestamps: {summary['parsed_row_timestamps']}",
        f"- Findings: {summary['findings']}",
        "",
        "| Artifact | Status | Generated at | Rows | Latest row timestamp | Min row skew seconds | Max row skew seconds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            "| {source} | {status} | {generated_at} | {row_count} | {latest_row_timestamp} | {min_row_skew_seconds} | {max_row_skew_seconds} |".format(
                **artifact
            )
        )
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append(f"- {finding['kind']}: {finding['scope']} {finding['detail']}".strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[TimestampFinding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_timestamp_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding.kind}:{finding.scope}"})
            failure = ET.SubElement(case, "failure", {"message": finding.detail})
            failure.text = finding.detail
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_timestamp_audit"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU benchmark JSON files or directories to audit")
    parser.add_argument(
        "--pattern",
        action="append",
        default=["qemu_prompt_bench_latest.json", "qemu_prompt_bench_????????T??????Z.json"],
        help="Directory glob",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_timestamp_audit_latest")
    parser.add_argument("--max-future-seconds", type=float, default=300.0)
    parser.add_argument("--max-row-after-generated-at-seconds", type=float, default=5.0)
    parser.add_argument("--max-row-before-generated-at-seconds", type=float, default=3600.0)
    parser.add_argument("--require-rows", action="store_true", help="Fail artifacts without parseable row timestamps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts: list[TimestampArtifact] = []
    findings: list[TimestampFinding] = []
    for path in sorted(iter_input_files(args.inputs, args.pattern)):
        artifact, artifact_findings = audit_artifact(path, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    report = build_report(artifacts, findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
