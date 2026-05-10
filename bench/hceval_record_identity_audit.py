#!/usr/bin/env python3
"""Audit packed HCEval records for stable identities and duplicate payloads.

This host-side tool reads existing `.hceval` binaries only. It never launches
QEMU, never downloads data, and never touches the TempleOS guest.
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

import dataset_pack
import hceval_inspect


DEFAULT_PATTERNS = ("*.hceval",)


@dataclass(frozen=True)
class IdentityRecord:
    source: str
    row: int
    dataset: str
    split: str
    record_id: str
    provenance: str
    choice_count: int
    answer_index: int
    prompt_sha256: str
    choices_sha256: str
    input_sha256: str
    answer_payload_sha256: str
    full_payload_sha256: str


@dataclass(frozen=True)
class ArtifactRecord:
    source: str
    status: str
    dataset: str
    split: str
    record_count: int
    unique_record_ids: int
    unique_input_payloads: int
    unique_answer_payloads: int
    finding_count: int
    error: str = ""


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    field: str
    value: str
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


def pack_records(dataset: hceval_inspect.HCEvalDataset) -> list[dataset_pack.EvalRecord]:
    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    return [
        dataset_pack.EvalRecord(
            record_id=record.record_id,
            dataset=dataset_name,
            split=split,
            prompt=record.prompt,
            choices=record.choices,
            answer_index=record.answer_index,
            provenance=record.provenance,
        )
        for record in dataset.records
    ]


def identity_records(path: Path, dataset: hceval_inspect.HCEvalDataset) -> list[IdentityRecord]:
    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    fingerprints = dataset_pack.record_fingerprints(pack_records(dataset))
    rows: list[IdentityRecord] = []
    for index, (record, fingerprint) in enumerate(zip(dataset.records, fingerprints, strict=True), 1):
        rows.append(
            IdentityRecord(
                source=str(path),
                row=index,
                dataset=dataset_name,
                split=split,
                record_id=record.record_id,
                provenance=record.provenance,
                choice_count=len(record.choices),
                answer_index=record.answer_index,
                prompt_sha256=str(fingerprint["prompt_sha256"]),
                choices_sha256=str(fingerprint["choices_sha256"]),
                input_sha256=str(fingerprint["input_sha256"]),
                answer_payload_sha256=str(fingerprint["answer_payload_sha256"]),
                full_payload_sha256=str(fingerprint["full_payload_sha256"]),
            )
        )
    return rows


def add_duplicate_findings(
    rows: list[IdentityRecord],
    findings: list[Finding],
    *,
    field: str,
    kind: str,
    detail: str,
) -> None:
    seen: dict[str, IdentityRecord] = {}
    for row in rows:
        value = str(getattr(row, field))
        first = seen.get(value)
        if first is None:
            seen[value] = row
            continue
        findings.append(
            Finding(
                row.source,
                row.row,
                "error",
                kind,
                field,
                value,
                f"{detail}; first seen at {first.source}:{first.row}",
            )
        )


def audit_artifact(path: Path) -> tuple[ArtifactRecord, list[IdentityRecord], list[Finding]]:
    findings: list[Finding] = []
    try:
        dataset = hceval_inspect.parse_hceval(path)
        rows = identity_records(path, dataset)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        finding = Finding(str(path), 0, "error", "load_error", "input", "", str(exc))
        return ArtifactRecord(str(path), "fail", "", "", 0, 0, 0, 0, 1, str(exc)), [], [finding]

    for row in rows:
        if not row.record_id.strip():
            findings.append(Finding(row.source, row.row, "error", "blank_record_id", "record_id", row.record_id, "record_id must be non-empty"))
        if row.record_id != row.record_id.strip():
            findings.append(
                Finding(row.source, row.row, "error", "record_id_whitespace", "record_id", row.record_id, "record_id must be trimmed")
            )

    add_duplicate_findings(
        rows,
        findings,
        field="record_id",
        kind="duplicate_record_id",
        detail="packed records must have stable one-to-one record IDs",
    )
    add_duplicate_findings(
        rows,
        findings,
        field="input_sha256",
        kind="duplicate_input_payload",
        detail="prompt and choices duplicate another packed record",
    )
    add_duplicate_findings(
        rows,
        findings,
        field="answer_payload_sha256",
        kind="duplicate_answer_payload",
        detail="prompt, choices, and answer duplicate another packed record",
    )

    dataset_name = str(dataset.metadata.get("dataset", ""))
    split = str(dataset.metadata.get("split", ""))
    return (
        ArtifactRecord(
            source=str(path),
            status="fail" if findings else "pass",
            dataset=dataset_name,
            split=split,
            record_count=len(rows),
            unique_record_ids=len({row.record_id for row in rows}),
            unique_input_payloads=len({row.input_sha256 for row in rows}),
            unique_answer_payloads=len({row.answer_payload_sha256 for row in rows}),
            finding_count=len(findings),
        ),
        rows,
        findings,
    )


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[ArtifactRecord], list[IdentityRecord], list[Finding]]:
    sources = list(iter_input_files(paths, args.pattern))
    artifacts: list[ArtifactRecord] = []
    rows: list[IdentityRecord] = []
    findings: list[Finding] = []
    if len(sources) < args.min_artifacts:
        findings.append(
            Finding(
                "",
                0,
                "error",
                "min_artifacts",
                "artifacts",
                str(len(sources)),
                f"found {len(sources)} HCEval artifact(s), below minimum {args.min_artifacts}",
            )
        )
    for source in sources:
        artifact, artifact_rows, artifact_findings = audit_artifact(source)
        artifacts.append(artifact)
        rows.extend(artifact_rows)
        findings.extend(artifact_findings)
    return artifacts, rows, findings


def build_report(artifacts: list[ArtifactRecord], rows: list[IdentityRecord], findings: list[Finding], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": "hceval-record-identity-audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(artifacts),
            "passing_artifacts": sum(1 for artifact in artifacts if artifact.status == "pass"),
            "failing_artifacts": sum(1 for artifact in artifacts if artifact.status == "fail"),
            "records": len(rows),
            "unique_record_ids": len({row.record_id for row in rows}),
            "unique_input_payloads": len({row.input_sha256 for row in rows}),
            "unique_answer_payloads": len({row.answer_payload_sha256 for row in rows}),
            "findings": len(findings),
        },
        "settings": {"patterns": args.pattern, "min_artifacts": args.min_artifacts},
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "records": [asdict(row) for row in rows],
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# HCEval Record Identity Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Artifacts: {summary['artifacts']}",
        f"Records: {summary['records']}",
        f"Findings: {summary['findings']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(
            f"- {finding['kind']} `{finding['source']}` row {finding['row']}: {finding['detail']}"
            for finding in report["findings"][:50]
        )
        if len(report["findings"]) > 50:
            lines.append(f"- ... {len(report['findings']) - 50} more finding(s)")
    else:
        lines.append("No findings.")

    lines.extend(["", "## Artifacts", "", "| Artifact | Status | Dataset | Split | Records | Unique IDs | Unique inputs |"])
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for artifact in report["artifacts"]:
        lines.append(
            f"| {artifact['source']} | {artifact['status']} | {artifact['dataset'] or '-'} | {artifact['split'] or '-'} | "
            f"{artifact['record_count']} | {artifact['unique_record_ids']} | {artifact['unique_input_payloads']} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_junit(report: dict[str, Any], path: Path) -> None:
    testcase = ET.Element("testcase", classname="hceval_record_identity_audit", name="record_identity")
    testsuite = ET.Element(
        "testsuite",
        name="hceval_record_identity_audit",
        tests="1",
        failures="1" if report["findings"] else "0",
        errors="0",
    )
    if report["findings"]:
        failure = ET.SubElement(testcase, "failure", message=f"{len(report['findings'])} finding(s)")
        failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in report["findings"][:50])
    testsuite.append(testcase)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Input .hceval files or directories")
    parser.add_argument("--input", dest="input_paths", action="append", type=Path, default=[], help="Additional .hceval file or directory")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern used when an input is a directory")
    parser.add_argument("--min-artifacts", type=int, default=1, help="Fail if fewer artifacts are discovered")
    parser.add_argument("--output", type=Path, default=Path("bench/results/datasets/hceval_record_identity_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional per-record CSV path")
    parser.add_argument("--artifacts-csv", type=Path, help="Optional per-artifact CSV path")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when findings are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.pattern = args.pattern or list(DEFAULT_PATTERNS)
    inputs = [*args.paths, *args.input_paths]
    artifacts, rows, findings = audit(inputs, args)
    report = build_report(artifacts, rows, findings, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        write_csv(args.csv, list(IdentityRecord.__dataclass_fields__.keys()), report["records"])
    if args.artifacts_csv:
        write_csv(args.artifacts_csv, list(ArtifactRecord.__dataclass_fields__.keys()), report["artifacts"])
    if args.findings_csv:
        write_csv(args.findings_csv, list(Finding.__dataclass_fields__.keys()), report["findings"])
    if args.junit:
        write_junit(report, args.junit)

    print(f"status={report['status']}")
    print(f"artifacts={report['summary']['artifacts']}")
    print(f"records={report['summary']['records']}")
    print(f"findings={report['summary']['findings']}")
    return 1 if args.fail_on_findings and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
