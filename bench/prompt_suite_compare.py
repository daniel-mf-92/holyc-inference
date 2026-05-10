#!/usr/bin/env python3
"""Compare local benchmark prompt suites for apples-to-apples runs.

This host-side tool verifies that two JSONL prompt suites contain the same
prompt IDs, order, prompt text hashes, byte sizes, and expected-token metadata.
It never launches QEMU and never fetches remote data.
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

import qemu_prompt_bench


@dataclass(frozen=True)
class PromptEntry:
    index: int
    prompt_id: str
    sha256: str
    bytes: int
    expected_tokens: int | None


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    prompt_id: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_suite(path: Path) -> list[PromptEntry]:
    cases = qemu_prompt_bench.load_prompt_cases(path)
    return [
        PromptEntry(
            index=index,
            prompt_id=case.prompt_id,
            sha256=qemu_prompt_bench.prompt_hash(case.prompt),
            bytes=qemu_prompt_bench.prompt_bytes(case.prompt),
            expected_tokens=case.expected_tokens,
        )
        for index, case in enumerate(cases)
    ]


def suite_sha256(entries: list[PromptEntry]) -> str:
    return qemu_prompt_bench.prompt_suite_hash_parts(
        [
            {"prompt_id": entry.prompt_id, "sha256": entry.sha256, "bytes": entry.bytes}
            for entry in entries
        ]
    )


def duplicate_ids(entries: list[PromptEntry]) -> dict[str, list[int]]:
    seen: dict[str, list[int]] = {}
    for entry in entries:
        seen.setdefault(entry.prompt_id, []).append(entry.index)
    return {prompt_id: indexes for prompt_id, indexes in sorted(seen.items()) if len(indexes) > 1}


def append_finding(
    findings: list[Finding],
    kind: str,
    prompt_id: str,
    detail: str,
    severity: str = "error",
) -> None:
    findings.append(Finding(severity=severity, kind=kind, prompt_id=prompt_id, detail=detail))


def compare_suites(
    baseline: Path,
    candidate: Path,
    *,
    ignore_order: bool = False,
    ignore_expected_tokens: bool = False,
) -> dict[str, Any]:
    baseline_entries = load_suite(baseline)
    candidate_entries = load_suite(candidate)
    findings: list[Finding] = []

    baseline_dupes = duplicate_ids(baseline_entries)
    candidate_dupes = duplicate_ids(candidate_entries)
    for prompt_id, indexes in baseline_dupes.items():
        append_finding(findings, "duplicate_baseline_id", prompt_id, f"baseline indexes={indexes}")
    for prompt_id, indexes in candidate_dupes.items():
        append_finding(findings, "duplicate_candidate_id", prompt_id, f"candidate indexes={indexes}")

    baseline_by_id = {entry.prompt_id: entry for entry in baseline_entries}
    candidate_by_id = {entry.prompt_id: entry for entry in candidate_entries}
    baseline_ids = [entry.prompt_id for entry in baseline_entries]
    candidate_ids = [entry.prompt_id for entry in candidate_entries]

    missing_candidate = sorted(set(baseline_by_id) - set(candidate_by_id))
    extra_candidate = sorted(set(candidate_by_id) - set(baseline_by_id))
    for prompt_id in missing_candidate:
        append_finding(findings, "missing_candidate_prompt", prompt_id, "present in baseline only")
    for prompt_id in extra_candidate:
        append_finding(findings, "extra_candidate_prompt", prompt_id, "present in candidate only")

    if len(baseline_entries) != len(candidate_entries):
        append_finding(
            findings,
            "prompt_count_mismatch",
            "",
            f"baseline={len(baseline_entries)} candidate={len(candidate_entries)}",
        )

    if not ignore_order and baseline_ids != candidate_ids:
        append_finding(findings, "prompt_order_mismatch", "", "prompt ID order differs")

    compared_rows: list[dict[str, Any]] = []
    for prompt_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        left = baseline_by_id[prompt_id]
        right = candidate_by_id[prompt_id]
        row_findings: list[str] = []
        if left.sha256 != right.sha256:
            row_findings.append("prompt_sha256")
            append_finding(
                findings,
                "prompt_text_mismatch",
                prompt_id,
                f"baseline_sha256={left.sha256} candidate_sha256={right.sha256}",
            )
        if left.bytes != right.bytes:
            row_findings.append("prompt_bytes")
            append_finding(
                findings,
                "prompt_byte_mismatch",
                prompt_id,
                f"baseline_bytes={left.bytes} candidate_bytes={right.bytes}",
            )
        if not ignore_expected_tokens and left.expected_tokens != right.expected_tokens:
            row_findings.append("expected_tokens")
            append_finding(
                findings,
                "expected_tokens_mismatch",
                prompt_id,
                f"baseline={left.expected_tokens} candidate={right.expected_tokens}",
            )
        if not ignore_order and left.index != right.index:
            row_findings.append("index")
            append_finding(
                findings,
                "prompt_index_mismatch",
                prompt_id,
                f"baseline_index={left.index} candidate_index={right.index}",
            )
        compared_rows.append(
            {
                "prompt_id": prompt_id,
                "baseline_index": left.index,
                "candidate_index": right.index,
                "baseline_sha256": left.sha256,
                "candidate_sha256": right.sha256,
                "baseline_bytes": left.bytes,
                "candidate_bytes": right.bytes,
                "baseline_expected_tokens": left.expected_tokens,
                "candidate_expected_tokens": right.expected_tokens,
                "status": "fail" if row_findings else "pass",
                "findings": ";".join(row_findings),
            }
        )

    error_count = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "baseline": str(baseline),
        "candidate": str(candidate),
        "ignore_order": ignore_order,
        "ignore_expected_tokens": ignore_expected_tokens,
        "baseline_prompt_count": len(baseline_entries),
        "candidate_prompt_count": len(candidate_entries),
        "baseline_suite_sha256": suite_sha256(baseline_entries),
        "candidate_suite_sha256": suite_sha256(candidate_entries),
        "compared_prompt_count": len(compared_rows),
        "error_count": error_count,
        "findings": [asdict(finding) for finding in findings],
        "prompts": compared_rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Prompt Suite Compare",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Baseline prompts: {report['baseline_prompt_count']}",
        f"Candidate prompts: {report['candidate_prompt_count']}",
        f"Compared prompts: {report['compared_prompt_count']}",
        f"Baseline suite SHA256: `{report['baseline_suite_sha256']}`",
        f"Candidate suite SHA256: `{report['candidate_suite_sha256']}`",
        "",
    ]
    if report["findings"]:
        lines.extend(["| Kind | Prompt ID | Detail |", "| --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {kind} | {prompt_id} | {detail} |".format(
                    kind=finding["kind"],
                    prompt_id=finding["prompt_id"],
                    detail=str(finding["detail"]).replace("|", "\\|"),
                )
            )
    else:
        lines.append("Prompt suites match for all enabled parity checks.")
    return "\n".join(lines) + "\n"


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown_report(report), encoding="utf-8")


def write_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row_type",
        "status",
        "kind",
        "prompt_id",
        "detail",
        "baseline_index",
        "candidate_index",
        "baseline_sha256",
        "candidate_sha256",
        "baseline_bytes",
        "candidate_bytes",
        "baseline_expected_tokens",
        "candidate_expected_tokens",
        "findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(
                {
                    "row_type": "finding",
                    "status": finding["severity"],
                    "kind": finding["kind"],
                    "prompt_id": finding["prompt_id"],
                    "detail": finding["detail"],
                }
            )
        for row in report["prompts"]:
            writer.writerow({"row_type": "prompt", **row})


def write_junit(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_prompt_suite_compare",
            "tests": "1",
            "failures": "1" if report["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "prompt_suite_parity"})
    if report["status"] == "fail":
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "prompt_suite_mismatch",
                "message": f"{report['error_count']} prompt-suite mismatch finding(s)",
            },
        )
        failure.text = "\n".join(
            f"{finding['kind']} {finding['prompt_id']}: {finding['detail']}"
            for finding in report["findings"]
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Reference prompt JSONL")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate prompt JSONL")
    parser.add_argument("--output", type=Path, default=Path("bench/results/prompt_suite_compare_latest.json"))
    parser.add_argument("--markdown", type=Path, default=Path("bench/results/prompt_suite_compare_latest.md"))
    parser.add_argument("--csv", type=Path, default=Path("bench/results/prompt_suite_compare_latest.csv"))
    parser.add_argument("--junit", type=Path, default=Path("bench/results/prompt_suite_compare_junit_latest.xml"))
    parser.add_argument("--ignore-order", action="store_true", help="Allow matching prompt IDs in different order")
    parser.add_argument(
        "--ignore-expected-tokens",
        action="store_true",
        help="Do not compare expected_tokens metadata",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = compare_suites(
            args.baseline,
            args.candidate,
            ignore_order=args.ignore_order,
            ignore_expected_tokens=args.ignore_expected_tokens,
        )
    except (OSError, ValueError) as exc:
        print(f"prompt_suite_compare: {exc}", file=sys.stderr)
        return 2

    write_json(args.output, report)
    write_markdown(args.markdown, report)
    write_csv(args.csv, report)
    write_junit(args.junit, report)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
