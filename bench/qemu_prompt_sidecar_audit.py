#!/usr/bin/env python3
"""Audit QEMU prompt benchmark artifacts for required report sidecars.

This host-side tool reads saved benchmark artifacts only. It never launches
QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*_latest.json",)
DEFAULT_REQUIRED = ("csv", "markdown", "junit")
OPTIONAL_SIDECARS = ("launches_csv", "summary_csv", "phases_csv")


@dataclass(frozen=True)
class SidecarRecord:
    source: str
    status: str
    sidecar_kind: str
    sidecar_path: str
    exists: bool
    size_bytes: int
    error: str = ""


@dataclass(frozen=True)
class SidecarFinding:
    source: str
    sidecar_kind: str
    sidecar_path: str
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


def qemu_prompt_base(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_latest")] if stem.endswith("_latest") else stem


def sidecar_path(source: Path, kind: str) -> Path:
    base = qemu_prompt_base(source)
    if kind == "csv":
        return source.with_suffix(".csv")
    if kind == "markdown":
        return source.with_suffix(".md")
    if kind == "junit":
        return source.with_name(f"{base}_junit_latest.xml")
    if kind == "launches_csv":
        return source.with_name(f"{base}_launches_latest.csv")
    if kind == "summary_csv":
        return source.with_name(f"{base}_summary_latest.csv")
    if kind == "phases_csv":
        return source.with_name(f"{base}_phases_latest.csv")
    raise ValueError(f"unknown sidecar kind: {kind}")


def audit_source(source: Path, sidecar_kinds: Iterable[str]) -> tuple[list[SidecarRecord], list[SidecarFinding]]:
    records: list[SidecarRecord] = []
    findings: list[SidecarFinding] = []
    for kind in sidecar_kinds:
        path = sidecar_path(source, kind)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        error = ""
        if not exists:
            error = "required sidecar is missing"
            findings.append(SidecarFinding(str(source), kind, str(path), "missing_sidecar", error))
        elif size <= 0:
            error = "required sidecar is empty"
            findings.append(SidecarFinding(str(source), kind, str(path), "empty_sidecar", error))
        records.append(SidecarRecord(str(source), "fail" if error else "pass", kind, str(path), exists, size, error))
    return records, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[SidecarRecord], list[SidecarFinding]]:
    sources = list(iter_input_files(paths, args.pattern))
    records: list[SidecarRecord] = []
    findings: list[SidecarFinding] = []
    if len(sources) < args.min_artifacts:
        findings.append(
            SidecarFinding(
                "",
                "artifact",
                "",
                "min_artifacts",
                f"found {len(sources)} artifact(s), below minimum {args.min_artifacts}",
            )
        )
    sidecar_kinds = list(args.require_sidecar)
    for source in sources:
        source_records, source_findings = audit_source(source, sidecar_kinds)
        records.extend(source_records)
        findings.extend(source_findings)
    return records, findings


def build_report(records: list[SidecarRecord], findings: list[SidecarFinding]) -> dict[str, Any]:
    sources = {record.source for record in records}
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "artifacts": len(sources),
            "sidecars_checked": len(records),
            "missing_sidecars": sum(1 for finding in findings if finding.kind == "missing_sidecar"),
            "empty_sidecars": sum(1 for finding in findings if finding.kind == "empty_sidecar"),
            "findings": len(findings),
        },
        "sidecars": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, records: list[SidecarRecord], findings: list[SidecarFinding]) -> None:
    lines = [
        "# QEMU Prompt Sidecar Audit",
        "",
        f"Sidecars checked: {len(records)}",
        f"Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.extend(["| Artifact | Sidecar | Kind | Detail |", "| --- | --- | --- | --- |"])
        for finding in findings:
            lines.append(
                "| {source} | {sidecar} | {kind} | {detail} |".format(
                    source=finding.source,
                    sidecar=finding.sidecar_path,
                    kind=finding.kind,
                    detail=finding.detail.replace("|", "\\|"),
                )
            )
    else:
        lines.append("All audited QEMU prompt benchmark artifacts have the required non-empty sidecars.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records_csv(path: Path, records: list[SidecarRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SidecarRecord.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[SidecarFinding]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SidecarFinding.__dataclass_fields__), lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_junit(path: Path, findings: list[SidecarFinding]) -> None:
    failures_by_source: dict[str, list[SidecarFinding]] = {}
    for finding in findings:
        failures_by_source.setdefault(finding.source or "coverage", []).append(finding)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_prompt_sidecar_audit",
            "tests": str(max(1, len(failures_by_source) or 1)),
            "failures": str(len(failures_by_source)),
            "errors": "0",
        },
    )
    if not findings:
        ET.SubElement(suite, "testcase", {"classname": "qemu_prompt_sidecar_audit", "name": "all_sidecars"})
    for source, source_findings in sorted(failures_by_source.items()):
        case = ET.SubElement(suite, "testcase", {"classname": "qemu_prompt_sidecar_audit", "name": Path(source).name})
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "qemu_prompt_sidecar_violation",
                "message": "; ".join(finding.kind for finding in source_findings),
            },
        )
        failure.text = "\n".join(f"{finding.sidecar_path}: {finding.detail}" for finding in source_findings)
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="QEMU prompt benchmark JSON files or directories")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="glob for directory inputs")
    parser.add_argument("--require-sidecar", action="append", choices=DEFAULT_REQUIRED + OPTIONAL_SIDECARS, default=list(DEFAULT_REQUIRED))
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_prompt_sidecar_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.min_artifacts < 0:
        parser.error("--min-artifacts must be >= 0")

    records, findings = audit(args.inputs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(records, findings)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_markdown(args.output_dir / f"{stem}.md", records, findings)
    write_records_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
