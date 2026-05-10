#!/usr/bin/env python3
"""Audit packed HCEval choice semantics without launching the guest.

This host-side tool parses `.hceval` binaries and flags choice-level issues that
can survive structural packing: duplicate normalized choices, answer aliases,
and prompt text that already contains a candidate choice. It emits CI-friendly
JSON/CSV/Markdown/JUnit artifacts for offline dataset curation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


LABEL_PREFIX_RE = re.compile(r"^\s*(?:[A-Z]|\d+)[\).:\-]\s+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    input: str
    record_id: str
    choice_index: int
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_text(text: str, strip_choice_labels: bool) -> str:
    value = text.casefold().strip()
    if strip_choice_labels:
        value = LABEL_PREFIX_RE.sub("", value)
    return SPACE_RE.sub(" ", value).strip()


def discover_inputs(inputs: list[Path], glob_pattern: str) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for input_path in inputs:
        candidates = sorted(input_path.glob(glob_pattern)) if input_path.is_dir() else [input_path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            discovered.append(candidate)
            seen.add(resolved)
    return discovered


def audit_dataset(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    rows: list[dict[str, Any]] = []
    try:
        dataset = hceval_inspect.parse_hceval(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        findings.append(Finding("error", "read_error", str(path), "", -1, str(exc)))
        return {
            "input": str(path),
            "dataset": "",
            "split": "",
            "record_count": 0,
            "rows": rows,
            "findings": [asdict(finding) for finding in findings],
            "status": "fail",
        }

    for record in dataset.records:
        finding_count_before = len(findings)
        normalized_prompt = normalize_text(record.prompt, args.strip_choice_labels)
        normalized_choices = [
            normalize_text(choice, args.strip_choice_labels)
            for choice in record.choices
        ]
        duplicate_indexes: list[int] = []
        prompt_overlap_indexes: list[int] = []
        seen: dict[str, int] = {}

        for index, normalized_choice in enumerate(normalized_choices):
            if normalized_choice in seen:
                duplicate_indexes.extend([seen[normalized_choice], index])
                findings.append(
                    Finding(
                        "error",
                        "duplicate_normalized_choice",
                        str(path),
                        record.record_id,
                        index,
                        f"choice {index} duplicates normalized choice {seen[normalized_choice]}",
                    )
                )
            else:
                seen[normalized_choice] = index

            if (
                args.min_overlap_chars > 0
                and len(normalized_choice) >= args.min_overlap_chars
                and normalized_choice in normalized_prompt
            ):
                prompt_overlap_indexes.append(index)
                findings.append(
                    Finding(
                        "error",
                        "choice_text_in_prompt",
                        str(path),
                        record.record_id,
                        index,
                        f"choice {index} text appears in prompt after normalization",
                    )
                )

        if record.answer_index < len(normalized_choices):
            answer_alias_count = normalized_choices.count(normalized_choices[record.answer_index])
        else:
            answer_alias_count = 0
            findings.append(
                Finding(
                    "error",
                    "answer_index_out_of_range",
                    str(path),
                    record.record_id,
                    record.answer_index,
                    f"answer index {record.answer_index} is outside {len(normalized_choices)} choices",
                )
            )
        if answer_alias_count > 1:
            findings.append(
                Finding(
                    "error",
                    "answer_choice_alias",
                    str(path),
                    record.record_id,
                    record.answer_index,
                    f"answer choice has {answer_alias_count} normalized aliases",
                )
            )
        record_has_findings = len(findings) > finding_count_before

        rows.append(
            {
                "input": str(path),
                "record_id": record.record_id,
                "choice_count": len(record.choices),
                "unique_normalized_choices": len(set(normalized_choices)),
                "duplicate_choice_indexes": ",".join(str(index) for index in sorted(set(duplicate_indexes))),
                "prompt_overlap_choice_indexes": ",".join(str(index) for index in sorted(set(prompt_overlap_indexes))),
                "answer_index": record.answer_index,
                "answer_alias_count": answer_alias_count,
                "status": "fail" if record_has_findings else "pass",
            }
        )

    return {
        "input": str(path),
        "dataset": str(dataset.metadata.get("dataset", "")),
        "split": str(dataset.metadata.get("split", "")),
        "record_count": len(dataset.records),
        "rows": rows,
        "findings": [asdict(finding) for finding in findings],
        "status": "fail" if findings else "pass",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = discover_inputs(args.input, args.glob)
    input_reports = [audit_dataset(path, args) for path in inputs]
    findings = [finding for input_report in input_reports for finding in input_report["findings"]]
    rows = [row for input_report in input_reports for row in input_report["rows"]]
    if not inputs:
        findings.append(
            {
                "severity": "error",
                "kind": "no_inputs",
                "input": ",".join(str(path) for path in args.input) or "input",
                "record_id": "",
                "choice_index": -1,
                "detail": "no .hceval inputs were discovered",
            }
        )

    return {
        "format": "hceval-choice-semantics-audit",
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "inputs": [str(path) for path in inputs],
        "input_count": len(inputs),
        "record_count": len(rows),
        "finding_count": len(findings),
        "settings": {
            "glob": args.glob,
            "min_overlap_chars": args.min_overlap_chars,
            "strip_choice_labels": args.strip_choice_labels,
        },
        "input_reports": input_reports,
        "rows": rows,
        "findings": findings,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# HCEval Choice Semantics Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Inputs: {report['input_count']}",
        f"Records: {report['record_count']}",
        f"Findings: {report['finding_count']}",
        "",
        "## Findings",
        "",
    ]
    if report["findings"]:
        lines.extend(
            f"- {finding['kind']} `{finding['record_id'] or finding['input']}`: {finding['detail']}"
            for finding in report["findings"]
        )
    else:
        lines.append("No findings.")

    lines.extend(["", "## Records", ""])
    if report["rows"]:
        lines.append("| Input | Record | Choices | Unique normalized | Answer aliases | Status |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in report["rows"]:
            lines.append(
                f"| {row['input']} | {row['record_id']} | {row['choice_count']} | "
                f"{row['unique_normalized_choices']} | {row['answer_alias_count']} | {row['status']} |"
            )
    else:
        lines.append("No records.")
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "input",
        "record_id",
        "choice_count",
        "unique_normalized_choices",
        "duplicate_choice_indexes",
        "prompt_overlap_choice_indexes",
        "answer_index",
        "answer_alias_count",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = ["severity", "kind", "input", "record_id", "choice_index", "detail"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["findings"])


def junit_report(report: dict[str, Any]) -> str:
    failures = [finding for finding in report["findings"] if finding["severity"] == "error"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_hceval_choice_semantics_audit",
            "tests": str(max(report["record_count"], 1)),
            "failures": str(len(failures)),
            "errors": "0",
            "timestamp": str(report["generated_at"]),
        },
    )
    rows = report["rows"] or [{"record_id": "input", "input": ",".join(report["inputs"])}]
    findings_by_record: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for finding in report["findings"]:
        key = (str(finding["input"]), str(finding["record_id"] or "input"))
        findings_by_record.setdefault(key, []).append(finding)

    for row in rows:
        key = (str(row["input"]), str(row["record_id"]))
        case = ET.SubElement(
            suite,
            "testcase",
            {"classname": "hceval_choice_semantics_audit", "name": f"{key[0]}::{key[1]}"},
        )
        row_findings = findings_by_record.get(key, [])
        if row_findings:
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "type": "hceval_choice_semantics_failure",
                    "message": f"{len(row_findings)} finding(s)",
                },
            )
            failure.text = "\n".join(f"{finding['kind']}: {finding['detail']}" for finding in row_findings)

    ET.indent(suite, space="  ")
    return ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input .hceval file or directory")
    parser.add_argument("--glob", default="**/*.hceval", help="Directory glob used for --input directories")
    parser.add_argument(
        "--min-overlap-chars",
        type=int,
        default=12,
        help="Minimum normalized choice length before prompt-overlap findings are emitted",
    )
    parser.add_argument(
        "--strip-choice-labels",
        action="store_true",
        help="Ignore leading labels such as 'A)' or '1.' when normalizing choices",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional per-record CSV report path")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML report path")
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
    if args.findings_csv:
        write_findings_csv(report, args.findings_csv)
        print(f"wrote_findings_csv={args.findings_csv}")
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        args.junit.write_text(junit_report(report), encoding="utf-8")
        print(f"wrote_junit={args.junit}")

    print(f"status={report['status']}")
    return 1 if args.fail_on_findings and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
