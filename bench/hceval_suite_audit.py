#!/usr/bin/env python3
"""Audit one or more HolyC-loadable `.hceval` dataset artifacts offline.

This is a host-side suite gate around `hceval_inspect.py`: it scans explicit
files or directories, validates packed binary structure and optional companion
manifests, applies byte budgets, and emits deterministic JSON/CSV/Markdown/JUnit
reports for CI and curation review.
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
class SuiteFinding:
    severity: str
    kind: str
    scope: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def discover_inputs(inputs: list[Path], glob_pattern: str) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            candidates = sorted(path for path in input_path.glob(glob_pattern) if path.is_file())
        else:
            candidates = [input_path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(candidate)
                seen.add(resolved)
    return discovered


def manifest_for(path: Path, suffix: str) -> Path:
    return Path(str(path) + suffix)


def inspect_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    findings: list[SuiteFinding] = []
    manifest_path = manifest_for(path, args.manifest_suffix)
    manifest_arg = manifest_path if manifest_path.exists() else None
    if args.require_manifest and manifest_arg is None:
        findings.append(
            SuiteFinding("error", "missing_manifest", str(path), f"missing companion manifest {manifest_path}")
        )

    try:
        dataset = hceval_inspect.parse_hceval(path)
        inspector_findings = hceval_inspect.validate_dataset(
            dataset,
            manifest_arg,
            max_prompt_bytes=args.max_prompt_bytes,
            max_choice_bytes=args.max_choice_bytes,
            max_record_payload_bytes=args.max_record_payload_bytes,
        )
        report = hceval_inspect.build_report(path, dataset, inspector_findings)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(SuiteFinding("error", "read_error", str(path), str(exc)))
        report = {
            "binary_layout": {},
            "byte_stats": {},
            "dataset": "",
            "payload_sha256": "",
            "record_count": 0,
            "split": "",
            "source_sha256": "",
            "status": "fail",
        }

    for detail in report.get("findings", []):
        findings.append(SuiteFinding("error", "inspection_finding", str(path), str(detail)))

    return {
        "binary_bytes": report.get("binary_layout", {}).get("binary_bytes", 0),
        "dataset": report.get("dataset", ""),
        "findings": [asdict(finding) for finding in findings],
        "input": str(path),
        "manifest": str(manifest_arg) if manifest_arg else "",
        "max_choice_bytes": report.get("byte_stats", {}).get("max_choice_bytes", 0),
        "max_prompt_bytes": report.get("byte_stats", {}).get("max_prompt_bytes", 0),
        "max_record_payload_bytes": report.get("byte_stats", {}).get("max_record_payload_bytes", 0),
        "payload_sha256": report.get("payload_sha256", ""),
        "record_count": report.get("record_count", 0),
        "source_sha256": report.get("source_sha256", ""),
        "split": report.get("split", ""),
        "status": "fail" if findings else "pass",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = discover_inputs(args.input, args.glob)
    rows = [inspect_one(path, args) for path in inputs]
    findings = [
        finding
        for row in rows
        for finding in row["findings"]
    ]
    if not inputs:
        findings.append(
            {
                "severity": "error",
                "kind": "no_inputs",
                "scope": ",".join(str(path) for path in args.input) or "input",
                "detail": "no .hceval inputs were discovered",
            }
        )

    error_count = sum(1 for finding in findings if finding["severity"] == "error")
    total_records = sum(int(row.get("record_count") or 0) for row in rows)
    total_binary_bytes = sum(int(row.get("binary_bytes") or 0) for row in rows)
    dataset_counts: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + int(row.get("record_count") or 0)

    return {
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "error_count": error_count,
        "findings": findings,
        "format": "hceval-suite-audit",
        "generated_at": iso_now(),
        "input_count": len(inputs),
        "inputs": [str(path) for path in inputs],
        "record_count": total_records,
        "rows": rows,
        "status": "fail" if error_count else "pass",
        "total_binary_bytes": total_binary_bytes,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Suite Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {report['input_count']}",
        f"Records: {report['record_count']}",
        f"Binary bytes: {report['total_binary_bytes']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(
            f"- {finding['kind']} `{finding['scope']}`: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No findings.")

    lines.extend(["", "## Inputs", ""])
    if report["rows"]:
        lines.append("| Input | Dataset | Split | Records | Binary bytes | Status |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for row in report["rows"]:
            lines.append(
                f"| {row['input']} | {row['dataset'] or '-'} | {row['split'] or '-'} | "
                f"{row['record_count']} | {row['binary_bytes']} | {row['status']} |"
            )
    else:
        lines.append("No inputs.")
    return "\n".join(lines) + "\n"


def junit_report(report: dict[str, Any]) -> str:
    rows = report.get("rows", [])
    failures = [row for row in rows if row.get("status") != "pass"]
    if not rows and report.get("findings"):
        failures = [{"input": "input", "findings": report["findings"]}]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_suite_audit",
            "tests": str(max(len(rows), 1)),
            "failures": str(len(failures)),
            "errors": "0",
            "timestamp": str(report.get("generated_at", "")),
        },
    )
    for row in rows or [{"input": "input", "findings": report.get("findings", []), "status": report["status"]}]:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "hceval_suite_audit",
                "name": str(row.get("input", "input")),
            },
        )
        if row.get("status") != "pass":
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "hceval_suite_failure",
                    "message": f"{len(row.get('findings', []))} finding(s)",
                },
            )
            failure.text = "\n".join(
                f"{finding['kind']}: {finding['detail']}" for finding in row.get("findings", [])
            )

    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "input",
        "manifest",
        "dataset",
        "split",
        "record_count",
        "binary_bytes",
        "max_prompt_bytes",
        "max_choice_bytes",
        "max_record_payload_bytes",
        "payload_sha256",
        "source_sha256",
        "status",
        "finding_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["rows"]:
            output_row = {field: row.get(field, "") for field in fieldnames}
            output_row["finding_count"] = len(row.get("findings", []))
            writer.writerow(output_row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input .hceval file or directory")
    parser.add_argument("--glob", default="**/*.hceval", help="Directory glob used for --input directories")
    parser.add_argument(
        "--manifest-suffix",
        default=".manifest.json",
        help="Companion manifest suffix appended to each .hceval input path",
    )
    parser.add_argument("--require-manifest", action="store_true", help="Fail inputs without companion manifests")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail if any prompt exceeds this UTF-8 byte limit")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail if any choice exceeds this UTF-8 byte limit")
    parser.add_argument(
        "--max-record-payload-bytes",
        type=int,
        help="Fail if any record payload excluding the fixed record header exceeds this byte limit",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional CSV report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit report path")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit non-zero when findings are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(args)

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
    return 1 if args.fail_on_findings and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
