#!/usr/bin/env python3
"""Audit packed HCEval dataset bundles for shard-level integrity.

This host-side tool scans one or more `.hceval` binaries, optionally validates
their companion pack manifests, and reports duplicate record IDs or payload
fingerprints across shards. It performs no network access and never launches
QEMU.
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
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import hceval_inspect


@dataclass(frozen=True)
class ShardSummary:
    path: str
    status: str
    dataset: str
    split: str
    record_count: int
    payload_sha256: str
    source_sha256: str
    manifest: str
    findings: list[str]


@dataclass(frozen=True)
class DuplicateFinding:
    kind: str
    value: str
    occurrences: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_inputs(paths: list[Path]) -> list[Path]:
    inputs: list[Path] = []
    for path in paths:
        if path.is_dir():
            inputs.extend(sorted(path.rglob("*.hceval")))
        else:
            inputs.append(path)
    return inputs


def manifest_for(path: Path) -> Path | None:
    candidate = path.with_suffix(path.suffix + ".manifest.json")
    if candidate.exists():
        return candidate
    alternate = path.with_suffix(".manifest.json")
    if alternate.exists():
        return alternate
    return None


def shard_summary(path: Path, require_manifest: bool) -> tuple[ShardSummary, hceval_inspect.HCEvalDataset | None]:
    findings: list[str] = []
    manifest = manifest_for(path)
    dataset: hceval_inspect.HCEvalDataset | None = None
    try:
        dataset = hceval_inspect.parse_hceval(path)
        findings.extend(hceval_inspect.validate_dataset(dataset, manifest))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(str(exc))

    if require_manifest and manifest is None:
        findings.append("missing companion manifest")

    return (
        ShardSummary(
            path=str(path),
            status="pass" if not findings else "fail",
            dataset=str(dataset.metadata.get("dataset", "")) if dataset else "",
            split=str(dataset.metadata.get("split", "")) if dataset else "",
            record_count=len(dataset.records) if dataset else 0,
            payload_sha256=dataset.payload_sha256 if dataset else "",
            source_sha256=dataset.source_digest if dataset else "",
            manifest=str(manifest) if manifest else "",
            findings=findings,
        ),
        dataset,
    )


def duplicate_findings(shards: list[tuple[Path, hceval_inspect.HCEvalDataset]]) -> list[DuplicateFinding]:
    by_record_id: dict[str, list[str]] = {}
    by_full_payload: dict[str, list[str]] = {}

    for path, dataset in shards:
        fingerprints = hceval_inspect.record_fingerprints(dataset)
        for record, fingerprint in zip(dataset.records, fingerprints, strict=True):
            label = f"{path}:{record.record_id}"
            by_record_id.setdefault(record.record_id, []).append(label)
            by_full_payload.setdefault(str(fingerprint["full_payload_sha256"]), []).append(label)

    duplicates: list[DuplicateFinding] = []
    for value, occurrences in sorted(by_record_id.items()):
        if len(occurrences) > 1:
            duplicates.append(DuplicateFinding("record_id", value, occurrences))
    for value, occurrences in sorted(by_full_payload.items()):
        if len(occurrences) > 1:
            duplicates.append(DuplicateFinding("full_payload_sha256", value, occurrences))
    return duplicates


def build_report(inputs: list[Path], require_manifest: bool) -> dict[str, Any]:
    summaries: list[ShardSummary] = []
    parsed: list[tuple[Path, hceval_inspect.HCEvalDataset]] = []
    for path in inputs:
        summary, dataset = shard_summary(path, require_manifest=require_manifest)
        summaries.append(summary)
        if dataset is not None:
            parsed.append((path, dataset))

    duplicates = duplicate_findings(parsed)
    findings = [
        f"{summary.path}: {finding}"
        for summary in summaries
        for finding in summary.findings
    ]
    findings.extend(
        f"duplicate {duplicate.kind} {duplicate.value}: {', '.join(duplicate.occurrences)}"
        for duplicate in duplicates
    )

    datasets = sorted({summary.dataset for summary in summaries if summary.dataset})
    splits = sorted({summary.split for summary in summaries if summary.split})
    return {
        "datasets": datasets,
        "duplicates": [asdict(duplicate) for duplicate in duplicates],
        "findings": findings,
        "generated_at": iso_now(),
        "record_count": sum(summary.record_count for summary in summaries),
        "shard_count": len(summaries),
        "shards": [asdict(summary) for summary in summaries],
        "splits": splits,
        "status": "pass" if not findings else "fail",
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Bundle Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Shards: {report['shard_count']}",
        f"Records: {report['record_count']}",
        f"Datasets: {', '.join(report['datasets']) if report['datasets'] else '-'}",
        f"Splits: {', '.join(report['splits']) if report['splits'] else '-'}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(f"- {finding}" for finding in report["findings"])
    else:
        lines.append("No findings.")

    lines.extend(["", "## Shards", "", "| Path | Status | Dataset | Split | Records |", "| --- | --- | --- | --- | ---: |"])
    for shard in report["shards"]:
        lines.append(
            f"| {shard['path']} | {shard['status']} | {shard['dataset'] or '-'} | "
            f"{shard['split'] or '-'} | {shard['record_count']} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "path",
        "status",
        "dataset",
        "split",
        "record_count",
        "payload_sha256",
        "source_sha256",
        "manifest",
        "findings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shard in report["shards"]:
            row = {field: shard.get(field, "") for field in fieldnames}
            row["findings"] = "; ".join(shard.get("findings", []))
            writer.writerow(row)


def junit_report(report: dict[str, Any]) -> str:
    findings = [str(finding) for finding in report.get("findings", [])]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_bundle_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "hceval_bundle_audit", "name": "bundle"})
    if findings:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "hceval_bundle_audit_failure",
                "message": f"{len(findings)} bundle finding(s)",
            },
        )
        failure.text = "\n".join(findings)
    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Input .hceval file(s) or directories")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional shard CSV report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML report path")
    parser.add_argument("--require-manifest", action="store_true", help="Fail shards without companion manifests")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = resolve_inputs(args.input)
    if not inputs:
        print("error: no .hceval inputs found", file=sys.stderr)
        return 2

    report = build_report(inputs, require_manifest=args.require_manifest)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        write_csv(report, args.csv)
        print(f"wrote_csv={args.csv}")
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        args.junit.write_text(junit_report(report), encoding="utf-8")
        print(f"wrote_junit={args.junit}")

    print(f"status={report['status']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
