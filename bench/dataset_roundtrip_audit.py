#!/usr/bin/env python3
"""Audit JSONL eval rows through an in-memory HCEval pack/inspect roundtrip.

The audit is offline-only. It normalizes local JSONL rows, packs them with the
same binary writer used for HolyC-loadable `.hceval` files, parses the binary
back through the host inspector, and verifies that record fingerprints, source
digests, and binary layout stay stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
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


@dataclass(frozen=True)
class RoundtripFinding:
    severity: str
    kind: str
    scope: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[dataset_pack.EvalRecord], list[dict[str, Any]], list[RoundtripFinding]]:
    records: list[dataset_pack.EvalRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[RoundtripFinding] = []

    for path in paths:
        try:
            rows = dataset_pack.read_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(RoundtripFinding("error", "read_error", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                records.append(dataset_pack.normalize_row(row, index, default_dataset, default_split))
            except ValueError as exc:
                findings.append(
                    RoundtripFinding(
                        "error",
                        "schema_error",
                        f"{path}:{index + 1}",
                        str(exc),
                    )
                )

    return records, inputs, findings


def record_identity(record: dataset_pack.EvalRecord) -> dict[str, Any]:
    return {
        "answer_index": record.answer_index,
        "choices": record.choices,
        "prompt": record.prompt,
        "provenance": record.provenance,
        "record_id": record.record_id,
    }


def roundtrip_dataset(records: list[dataset_pack.EvalRecord], dataset: str, split: str) -> hceval_inspect.HCEvalDataset:
    payload = dataset_pack.pack_records(records, dataset, split)
    with tempfile.TemporaryDirectory(prefix="hceval-roundtrip-") as tmp:
        path = Path(tmp) / "roundtrip.hceval"
        path.write_bytes(payload)
        return hceval_inspect.parse_hceval(path)


def append_mismatch_findings(
    findings: list[RoundtripFinding],
    expected: Any,
    actual: Any,
    *,
    kind: str,
    scope: str,
    detail: str,
) -> None:
    if expected != actual:
        findings.append(
            RoundtripFinding(
                "error",
                kind,
                scope,
                f"{detail}: expected {json.dumps(expected, sort_keys=True)}, got {json.dumps(actual, sort_keys=True)}",
            )
        )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    records, inputs, findings = read_records(args.input, args.dataset, args.split)
    inspected: hceval_inspect.HCEvalDataset | None = None
    inspector_findings: list[str] = []

    if records:
        try:
            inspected = roundtrip_dataset(records, args.dataset, args.split)
            inspector_findings = hceval_inspect.validate_dataset(
                inspected,
                max_prompt_bytes=args.max_prompt_bytes,
                max_choice_bytes=args.max_choice_bytes,
                max_record_payload_bytes=args.max_record_payload_bytes,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            findings.append(RoundtripFinding("error", "roundtrip_error", "binary", str(exc)))

    expected_fingerprints = dataset_pack.record_fingerprints(records)
    expected_layout = dataset_pack.binary_layout(records, args.dataset, args.split)
    expected_source_sha256 = dataset_pack.sha256_json([asdict(record) for record in records])
    actual_records = hceval_inspect.as_pack_records(inspected) if inspected is not None else []
    actual_identities = [record_identity(record) for record in actual_records]
    expected_identities = [record_identity(record) for record in records]
    actual_fingerprints = hceval_inspect.record_fingerprints(inspected) if inspected is not None else []
    actual_layout = hceval_inspect.binary_layout(inspected) if inspected is not None else {}
    actual_source_sha256 = inspected.source_digest if inspected is not None else ""

    for detail in inspector_findings:
        if (
            detail == "header source digest does not match reconstructed records"
            and expected_source_sha256 == actual_source_sha256
        ):
            continue
        findings.append(RoundtripFinding("error", "inspector_finding", "binary", detail))

    if records:
        append_mismatch_findings(
            findings,
            expected_source_sha256,
            actual_source_sha256,
            kind="source_digest_mismatch",
            scope="binary",
            detail="header source digest changed after roundtrip",
        )
        append_mismatch_findings(
            findings,
            expected_layout,
            actual_layout,
            kind="binary_layout_mismatch",
            scope="binary",
            detail="parsed binary layout differs from packer layout",
        )
        append_mismatch_findings(
            findings,
            expected_fingerprints,
            actual_fingerprints,
            kind="record_fingerprint_mismatch",
            scope="records",
            detail="parsed record fingerprints differ from normalized source records",
        )
        append_mismatch_findings(
            findings,
            expected_identities,
            actual_identities,
            kind="record_payload_mismatch",
            scope="records",
            detail="parsed record payloads differ from normalized source records",
        )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "actual_binary_layout": actual_layout,
        "actual_record_fingerprints": actual_fingerprints,
        "actual_source_sha256": actual_source_sha256,
        "byte_stats": dataset_pack.byte_stats(records),
        "choice_count_histogram": dataset_pack.choice_count_histogram(records),
        "dataset": args.dataset,
        "error_count": error_count,
        "expected_binary_layout": expected_layout,
        "expected_record_fingerprints": expected_fingerprints,
        "expected_source_sha256": expected_source_sha256,
        "findings": [asdict(finding) for finding in findings],
        "format": "hceval-roundtrip-audit",
        "generated_at": iso_now(),
        "inputs": inputs,
        "record_count": len(records),
        "split": args.split,
        "status": "fail" if error_count else "pass",
        "warning_count": warning_count,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Roundtrip Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Dataset metadata: {report['dataset']}",
        f"- Split metadata: {report['split']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Binary Layout",
        "",
        "| field | expected | actual |",
        "| --- | ---: | ---: |",
    ]
    keys = sorted(set(report["expected_binary_layout"]) | set(report["actual_binary_layout"]))
    for key in keys:
        lines.append(
            f"| {key} | {report['expected_binary_layout'].get(key, '')} | "
            f"{report['actual_binary_layout'].get(key, '')} |"
        )

    lines.extend(["", "## Findings", ""])
    if report["findings"]:
        lines.extend(["| severity | kind | scope | detail |", "| --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(f"| {finding['severity']} | {finding['kind']} | {finding['scope']} | {finding['detail']} |")
    else:
        lines.append("No roundtrip findings.")
    return "\n".join(lines) + "\n"


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "scope", "detail"])
        writer.writeheader()
        writer.writerows(findings)


def write_fingerprint_csv(path: Path, fingerprints: list[dict[str, Any]]) -> None:
    fieldnames = [
        "record_index",
        "record_id",
        "choice_count",
        "answer_index",
        "prompt_sha256",
        "choices_sha256",
        "input_sha256",
        "answer_payload_sha256",
        "full_payload_sha256",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in fingerprints:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_roundtrip_audit",
            "tests": "1",
            "failures": "1" if report["error_count"] else "0",
            "errors": "0",
            "timestamp": report["generated_at"],
        },
    )
    case = ET.SubElement(suite, "testcase", {"classname": "dataset_roundtrip_audit", "name": "roundtrip"})
    if report["error_count"]:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "dataset_roundtrip_failure",
                "message": f"{report['error_count']} roundtrip finding(s)",
            },
        )
        failure.text = "\n".join(
            f"{item['kind']} {item['scope']}: {item['detail']}" for item in report["findings"]
        )
    ET.indent(suite, space="  ")
    path.write_text(ET.tostring(suite, encoding="unicode", xml_declaration=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="Local JSONL eval input.")
    parser.add_argument("--output", type=Path, required=True, help="JSON audit report path.")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path.")
    parser.add_argument("--findings-csv", type=Path, help="Optional findings CSV path.")
    parser.add_argument("--fingerprints-csv", type=Path, help="Optional expected record fingerprint CSV path.")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML path.")
    parser.add_argument("--dataset", default="eval", help="Dataset metadata to write into the packed binary.")
    parser.add_argument("--split", default="validation", help="Split metadata to write into the packed binary.")
    parser.add_argument("--max-prompt-bytes", type=int, help="Fail if any prompt exceeds this UTF-8 byte limit.")
    parser.add_argument("--max-choice-bytes", type=int, help="Fail if any choice exceeds this UTF-8 byte limit.")
    parser.add_argument(
        "--max-record-payload-bytes",
        type=int,
        help="Fail if any record payload excluding the fixed record header exceeds this byte limit.",
    )
    parser.add_argument("--fail-on-findings", action="store_true", help="Return nonzero if findings are present.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    report = build_report(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.findings_csv:
        write_findings_csv(args.findings_csv, report["findings"])
    if args.fingerprints_csv:
        write_fingerprint_csv(args.fingerprints_csv, report["expected_record_fingerprints"])
    if args.junit:
        write_junit(args.junit, report)

    if args.fail_on_findings and report["findings"]:
        return 1
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
