#!/usr/bin/env python3
"""Audit QEMU prompt benchmark JSON artifacts against their primary CSV sidecars.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
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


DEFAULT_PATTERNS = ("qemu_prompt_bench_latest.json",)
DEFAULT_COMPARE_FIELDS = (
    "timestamp",
    "commit",
    "benchmark",
    "profile",
    "model",
    "quantization",
    "prompt",
    "prompt_sha256",
    "phase",
    "launch_index",
    "iteration",
    "tokens",
    "expected_tokens",
    "elapsed_us",
    "wall_elapsed_us",
    "returncode",
    "timed_out",
    "exit_class",
    "command_sha256",
    "command_airgap_ok",
    "command_has_explicit_nic_none",
    "command_has_legacy_net_none",
)


@dataclass(frozen=True)
class ParityArtifact:
    source: str
    status: str
    csv_path: str
    json_rows: int
    csv_rows: int
    compared_fields: int
    compared_values: int
    findings: int
    error: str = ""


@dataclass(frozen=True)
class ParityFinding:
    source: str
    csv_path: str
    row: str
    field: str
    kind: str
    json_value: str
    csv_value: str
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


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]] | None, list[str], str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return rows, list(reader.fieldnames or []), ""
    except OSError as exc:
        return None, [], str(exc)
    except csv.Error as exc:
        return None, [], f"invalid csv: {exc}"


def primary_csv_path(source: Path) -> Path:
    return source.with_suffix(".csv")


def normalize_json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def compare_artifact(path: Path, args: argparse.Namespace) -> tuple[ParityArtifact, list[ParityFinding]]:
    payload, error = load_json_object(path)
    csv_path = primary_csv_path(path)
    if payload is None:
        finding = ParityFinding(str(path), str(csv_path), "artifact", "", "load_error", "", "", error)
        return ParityArtifact(str(path), "fail", str(csv_path), 0, 0, 0, 0, 1, error), [finding]

    raw_rows = payload.get("benchmarks")
    if not isinstance(raw_rows, list) or not all(isinstance(row, dict) for row in raw_rows):
        detail = "benchmarks must be a list of objects"
        finding = ParityFinding(str(path), str(csv_path), "benchmarks", "", "json_schema", "", "", detail)
        return ParityArtifact(str(path), "fail", str(csv_path), 0, 0, 0, 0, 1, detail), [finding]

    if not csv_path.exists():
        detail = "primary CSV sidecar is missing"
        finding = ParityFinding(str(path), str(csv_path), "csv", "", "missing_csv", "", "", detail)
        return ParityArtifact(str(path), "fail", str(csv_path), len(raw_rows), 0, 0, 0, 1, detail), [finding]

    csv_rows, csv_fields, csv_error = load_csv_rows(csv_path)
    if csv_rows is None:
        finding = ParityFinding(str(path), str(csv_path), "csv", "", "load_error", "", "", csv_error)
        return ParityArtifact(str(path), "fail", str(csv_path), len(raw_rows), 0, 0, 0, 1, csv_error), [finding]

    findings: list[ParityFinding] = []
    compare_fields = list(args.compare_field)
    missing_fields = [field for field in compare_fields if field not in csv_fields]
    if args.require_all_columns:
        for field in missing_fields:
            findings.append(
                ParityFinding(
                    str(path),
                    str(csv_path),
                    "header",
                    field,
                    "missing_csv_column",
                    "",
                    "",
                    "CSV column is absent",
                )
            )

    if len(raw_rows) != len(csv_rows):
        findings.append(
            ParityFinding(
                str(path),
                str(csv_path),
                "artifact",
                "row_count",
                "row_count_mismatch",
                str(len(raw_rows)),
                str(len(csv_rows)),
                "JSON benchmark row count differs from CSV row count",
            )
        )

    comparable_fields = [field for field in compare_fields if field not in missing_fields]
    compared_values = 0
    for index, (json_row, csv_row) in enumerate(zip(raw_rows, csv_rows), start=1):
        for field in comparable_fields:
            compared_values += 1
            json_value = normalize_json_value(json_row.get(field))
            csv_value = csv_row.get(field, "")
            if json_value != csv_value:
                findings.append(
                    ParityFinding(
                        str(path),
                        str(csv_path),
                        str(index),
                        field,
                        "value_mismatch",
                        json_value,
                        csv_value,
                        "CSV value differs from JSON benchmark row",
                    )
                )

    if len(raw_rows) < args.min_rows:
        findings.append(
            ParityFinding(
                str(path),
                str(csv_path),
                "artifact",
                "json_rows",
                "min_rows",
                str(len(raw_rows)),
                str(args.min_rows),
                "artifact has fewer JSON benchmark rows than required",
            )
        )

    status = "fail" if findings else "pass"
    return (
        ParityArtifact(
            str(path),
            status,
            str(csv_path),
            len(raw_rows),
            len(csv_rows),
            len(comparable_fields),
            compared_values,
            len(findings),
        ),
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ParityArtifact], list[ParityFinding]]:
    artifacts: list[ParityArtifact] = []
    findings: list[ParityFinding] = []
    sources = list(iter_input_files(paths, args.pattern))
    if len(sources) < args.min_artifacts:
        findings.append(
            ParityFinding(
                "",
                "",
                "artifact",
                "",
                "min_artifacts",
                str(len(sources)),
                str(args.min_artifacts),
                "fewer artifacts matched than required",
            )
        )
    for source in sources:
        artifact, artifact_findings = compare_artifact(source, args)
        artifacts.append(artifact)
        findings.extend(artifact_findings)
    return artifacts, findings


def build_report(artifacts: list[ParityArtifact], findings: list[ParityFinding]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "json_rows": sum(artifact.json_rows for artifact in artifacts),
            "csv_rows": sum(artifact.csv_rows for artifact in artifacts),
            "compared_values": sum(artifact.compared_values for artifact in artifacts),
            "findings": len(findings),
        },
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, artifacts: list[ParityArtifact], findings: list[ParityFinding]) -> None:
    lines = [
        "# QEMU CSV Parity Audit",
        "",
        f"Artifacts: {len(artifacts)}",
        f"JSON rows: {sum(artifact.json_rows for artifact in artifacts)}",
        f"CSV rows: {sum(artifact.csv_rows for artifact in artifacts)}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["| Artifact | Row | Field | Kind | Detail |", "| --- | --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {row} | {field} | {kind} | {detail} |".format(
                    source=finding.source,
                    row=finding.row,
                    field=finding.field,
                    kind=finding.kind,
                    detail=finding.detail.replace("|", "\\|"),
                )
            )
    else:
        lines.append("All audited QEMU prompt benchmark CSV sidecars match their JSON benchmark rows.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifacts_csv(path: Path, artifacts: list[ParityArtifact]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ParityArtifact.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(asdict(artifact))


def write_findings_csv(path: Path, findings: list[ParityFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ParityFinding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, artifacts: list[ParityArtifact], findings: list[ParityFinding]) -> None:
    tests = max(1, len(artifacts))
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_csv_parity_audit",
            "tests": str(tests),
            "failures": str(len(findings)),
        },
    )
    if artifacts:
        for artifact in artifacts:
            case = ET.SubElement(suite, "testcase", {"name": artifact.source})
            artifact_findings = [finding for finding in findings if finding.source == artifact.source]
            for finding in artifact_findings:
                failure = ET.SubElement(case, "failure", {"type": finding.kind, "message": finding.detail})
                failure.text = f"{finding.row}:{finding.field} json={finding.json_value!r} csv={finding.csv_value!r}"
    else:
        case = ET.SubElement(suite, "testcase", {"name": "artifact_discovery"})
        for finding in findings:
            failure = ET.SubElement(case, "failure", {"type": finding.kind, "message": finding.detail})
            failure.text = f"json={finding.json_value!r} csv={finding.csv_value!r}"
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[Path("bench/results")])
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="artifact glob to audit")
    parser.add_argument("--compare-field", action="append", default=list(DEFAULT_COMPARE_FIELDS))
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--require-all-columns", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_csv_parity_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts, findings = audit(args.inputs, args)
    report = build_report(artifacts, findings)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", artifacts, findings)
    write_artifacts_csv(args.output_dir / f"{stem}.csv", artifacts)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", artifacts, findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
