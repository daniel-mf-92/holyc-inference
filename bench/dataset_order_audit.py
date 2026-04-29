#!/usr/bin/env python3
"""Audit local multiple-choice eval row order for answer-position bias.

The audit is offline-only. It normalizes the same JSONL row shapes accepted by
dataset_pack.py, records answer-index run/transition telemetry in source order,
and can fail when a curated subset has long same-answer blocks or too few answer
position switches. This catches row-order bias before HCEval binaries are loaded
by the air-gapped TempleOS guest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
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


@dataclass(frozen=True)
class LoadedRecord:
    source: str
    row_number: int
    record: dataset_pack.EvalRecord


@dataclass(frozen=True)
class OrderFinding:
    severity: str
    kind: str
    scope: str
    source: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ref(record: LoadedRecord) -> str:
    return f"{record.source}:{record.row_number}"


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return dataset_pack.read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_records(
    paths: Iterable[Path],
    default_dataset: str,
    default_split: str,
) -> tuple[list[LoadedRecord], list[dict[str, Any]], list[OrderFinding]]:
    records: list[LoadedRecord] = []
    inputs: list[dict[str, Any]] = []
    findings: list[OrderFinding] = []

    for path in paths:
        try:
            rows = read_rows(path)
        except ValueError as exc:
            findings.append(OrderFinding("error", "read_error", "input", str(path), str(exc)))
            rows = []

        input_info: dict[str, Any] = {"path": str(path), "rows": len(rows)}
        if path.exists():
            input_info["sha256"] = file_sha256(path)
        inputs.append(input_info)

        for index, row in enumerate(rows):
            try:
                record = dataset_pack.normalize_row(row, index, default_dataset, default_split)
            except ValueError as exc:
                findings.append(
                    OrderFinding(
                        "error",
                        "schema_error",
                        "input",
                        f"{path}:{index + 1}",
                        str(exc),
                    )
                )
                continue
            records.append(LoadedRecord(str(path), index + 1, record))

    return records, inputs, findings


def answer_histogram(records: list[LoadedRecord]) -> dict[str, int]:
    counter = collections.Counter(str(loaded.record.answer_index) for loaded in records)
    return {key: counter[key] for key in sorted(counter, key=int)}


def answer_runs(records: list[LoadedRecord]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not records:
        return runs

    run_answer = records[0].record.answer_index
    run_start = 0
    for index, loaded in enumerate(records[1:], 1):
        if loaded.record.answer_index == run_answer:
            continue
        run_records = records[run_start:index]
        runs.append(run_payload(run_answer, run_start, run_records))
        run_answer = loaded.record.answer_index
        run_start = index

    runs.append(run_payload(run_answer, run_start, records[run_start:]))
    return runs


def run_payload(answer_index: int, start_index: int, records: list[LoadedRecord]) -> dict[str, Any]:
    return {
        "answer_index": answer_index,
        "length": len(records),
        "start_index": start_index,
        "end_index": start_index + len(records) - 1,
        "sources": [source_ref(record) for record in records],
    }


def transition_count(records: list[LoadedRecord]) -> int:
    return sum(
        1
        for left, right in zip(records, records[1:])
        if left.record.answer_index != right.record.answer_index
    )


def group_records(records: list[LoadedRecord], group_by: str) -> dict[str, list[LoadedRecord]]:
    if group_by == "overall":
        return {"overall": records}
    if group_by == "dataset":
        groups: dict[str, list[LoadedRecord]] = {}
        for loaded in records:
            groups.setdefault(loaded.record.dataset, []).append(loaded)
        return dict(sorted(groups.items()))
    if group_by == "dataset_split":
        groups = {}
        for loaded in records:
            key = f"{loaded.record.dataset}:{loaded.record.split}"
            groups.setdefault(key, []).append(loaded)
        return dict(sorted(groups.items()))
    raise ValueError(f"unsupported group_by {group_by!r}")


def order_stats_for(scope: str, records: list[LoadedRecord]) -> dict[str, Any]:
    runs = answer_runs(records)
    longest = max(runs, key=lambda run: (run["length"], -run["start_index"])) if runs else None
    switches = transition_count(records)
    possible_switches = max(len(records) - 1, 0)
    return {
        "scope": scope,
        "record_count": len(records),
        "answer_histogram": answer_histogram(records),
        "answer_sequence": [loaded.record.answer_index for loaded in records],
        "transition_count": switches,
        "possible_transition_count": possible_switches,
        "switch_rate": switches / possible_switches if possible_switches else None,
        "run_count": len(runs),
        "longest_run": longest,
        "longest_run_pct": (longest["length"] / len(records) * 100.0) if longest and records else None,
        "leading_run": runs[0] if runs else None,
        "trailing_run": runs[-1] if runs else None,
        "runs": runs,
    }


def build_order_stats(records: list[LoadedRecord], group_by: str) -> list[dict[str, Any]]:
    return [order_stats_for(scope, group) for scope, group in group_records(records, group_by).items()]


def add_gate_findings(
    stats: list[dict[str, Any]],
    findings: list[OrderFinding],
    max_longest_answer_run: int | None,
    max_longest_answer_run_pct: float | None,
    max_edge_answer_run: int | None,
    min_answer_switches: int | None,
) -> None:
    for item in stats:
        scope = item["scope"]
        longest = item["longest_run"]
        if longest and max_longest_answer_run is not None and longest["length"] > max_longest_answer_run:
            findings.append(
                OrderFinding(
                    "error",
                    "long_answer_run",
                    scope,
                    ",".join(longest["sources"]),
                    (
                        f"answer index {longest['answer_index']} appears in {longest['length']} consecutive rows, "
                        f"above {max_longest_answer_run}"
                    ),
                )
            )
        longest_pct = item["longest_run_pct"]
        if (
            longest
            and max_longest_answer_run_pct is not None
            and longest_pct is not None
            and longest_pct > max_longest_answer_run_pct
        ):
            findings.append(
                OrderFinding(
                    "error",
                    "long_answer_run_pct",
                    scope,
                    ",".join(longest["sources"]),
                    (
                        f"answer index {longest['answer_index']} longest run covers {longest_pct:.2f}% "
                        f"of rows, above {max_longest_answer_run_pct:.2f}%"
                    ),
                )
            )
        for edge_name in ("leading_run", "trailing_run"):
            run = item[edge_name]
            if run and max_edge_answer_run is not None and run["length"] > max_edge_answer_run:
                findings.append(
                    OrderFinding(
                        "error",
                        edge_name,
                        scope,
                        ",".join(run["sources"]),
                        (
                            f"answer index {run['answer_index']} {edge_name.replace('_', ' ')} has "
                            f"{run['length']} rows, above {max_edge_answer_run}"
                        ),
                    )
                )
        if min_answer_switches is not None and item["transition_count"] < min_answer_switches:
            findings.append(
                OrderFinding(
                    "error",
                    "too_few_answer_switches",
                    scope,
                    scope,
                    (
                        f"{item['transition_count']} answer-index switches, below "
                        f"{min_answer_switches}"
                    ),
                )
            )


def build_report(
    inputs: list[dict[str, Any]],
    records: list[LoadedRecord],
    group_by: str,
    findings: list[OrderFinding],
) -> dict[str, Any]:
    stats = build_order_stats(records, group_by)
    error_count = sum(1 for finding in findings if finding.severity == "error")
    warning_count = sum(1 for finding in findings if finding.severity == "warning")
    return {
        "generated_at": iso_now(),
        "format": "hceval-order-audit",
        "status": "fail" if error_count else "pass",
        "inputs": inputs,
        "group_by": group_by,
        "record_count": len(records),
        "error_count": error_count,
        "warning_count": warning_count,
        "order_stats": stats,
        "findings": [asdict(finding) for finding in findings],
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Dataset Order Audit",
        "",
        f"- Status: {report['status']}",
        f"- Records: {report['record_count']}",
        f"- Group by: {report['group_by']}",
        f"- Errors: {report['error_count']}",
        f"- Warnings: {report['warning_count']}",
        "",
        "## Order Stats",
        "",
        "| scope | records | switches | switch_rate | longest_run_answer | longest_run_len | longest_run_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["order_stats"]:
        longest = item["longest_run"] or {}
        switch_rate = item["switch_rate"]
        longest_pct = item["longest_run_pct"]
        lines.append(
            "| {scope} | {records} | {switches} | {switch_rate} | {answer} | {length} | {longest_pct} |".format(
                scope=item["scope"],
                records=item["record_count"],
                switches=item["transition_count"],
                switch_rate="" if switch_rate is None else f"{switch_rate:.4f}",
                answer=longest.get("answer_index", ""),
                length=longest.get("length", ""),
                longest_pct="" if longest_pct is None else f"{longest_pct:.2f}",
            )
        )
    lines.extend(["", "## Findings", ""])
    if not report["findings"]:
        lines.append("No order-bias findings.")
    else:
        lines.extend(["| severity | kind | scope | source | detail |", "| --- | --- | --- | --- | --- |"])
        for finding in report["findings"]:
            lines.append(
                "| {severity} | {kind} | {scope} | {source} | {detail} |".format(**finding)
            )
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scope",
                "record_count",
                "transition_count",
                "possible_transition_count",
                "switch_rate",
                "run_count",
                "longest_run_answer",
                "longest_run_length",
                "longest_run_pct",
                "answer_histogram",
            ],
        )
        writer.writeheader()
        for item in report["order_stats"]:
            longest = item["longest_run"] or {}
            writer.writerow(
                {
                    "scope": item["scope"],
                    "record_count": item["record_count"],
                    "transition_count": item["transition_count"],
                    "possible_transition_count": item["possible_transition_count"],
                    "switch_rate": item["switch_rate"],
                    "run_count": item["run_count"],
                    "longest_run_answer": longest.get("answer_index", ""),
                    "longest_run_length": longest.get("length", ""),
                    "longest_run_pct": item["longest_run_pct"],
                    "answer_histogram": json.dumps(item["answer_histogram"], sort_keys=True),
                }
            )


def write_findings_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["severity", "kind", "scope", "source", "detail"])
        writer.writeheader()
        for finding in report["findings"]:
            writer.writerow(finding)


def write_junit(report: dict[str, Any], path: Path) -> None:
    error_findings = [finding for finding in report["findings"] if finding["severity"] == "error"]
    testcase_count = len(error_findings) + (1 if not error_findings else 0)
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "holyc_dataset_order_audit",
            "tests": str(testcase_count),
            "failures": str(len(error_findings)),
            "errors": "0",
        },
    )
    if error_findings:
        for finding in error_findings:
            testcase = ET.SubElement(
                testsuite,
                "testcase",
                {
                    "classname": "dataset_order_audit",
                    "name": f"{finding['kind']}:{finding['scope']}",
                },
            )
            ET.SubElement(testcase, "failure", {"type": finding["kind"], "message": finding["detail"]})
    else:
        ET.SubElement(testsuite, "testcase", {"classname": "dataset_order_audit", "name": "order_audit_pass"})
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(testsuite).write(path, encoding="utf-8", xml_declaration=True)


def write_outputs(
    report: dict[str, Any],
    output: Path,
    markdown: Path | None,
    csv_path: Path | None,
    findings_csv: Path | None,
    junit: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(report), encoding="utf-8")
    if csv_path:
        write_csv(report, csv_path)
    if findings_csv:
        write_findings_csv(report, findings_csv)
    if junit:
        write_junit(report, junit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input eval JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report")
    parser.add_argument("--csv", type=Path, help="Optional CSV order-stats output")
    parser.add_argument("--findings-csv", type=Path, help="Optional CSV findings output")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML output")
    parser.add_argument("--default-dataset", default="eval", help="Dataset name for rows missing dataset")
    parser.add_argument("--default-split", default="validation", help="Split name for rows missing split")
    parser.add_argument(
        "--group-by",
        choices=["overall", "dataset", "dataset_split"],
        default="overall",
        help="Scope used for run and transition gates",
    )
    parser.add_argument("--max-longest-answer-run", type=int, help="Fail if a scope has a longer same-answer run")
    parser.add_argument(
        "--max-longest-answer-run-pct",
        type=float,
        help="Fail if a scope's longest same-answer run covers more than this percentage",
    )
    parser.add_argument("--max-edge-answer-run", type=int, help="Fail if leading/trailing same-answer runs exceed this")
    parser.add_argument("--min-answer-switches", type=int, help="Fail if a scope has fewer answer-index switches")
    parser.add_argument("--fail-on-findings", action="store_true", help="Exit nonzero when errors are present")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("max_longest_answer_run", "max_edge_answer_run", "min_answer_switches"):
        value = getattr(args, name)
        if value is not None and value < 0:
            print(f"error: --{name.replace('_', '-')} must be non-negative", file=sys.stderr)
            return 2
    if args.max_longest_answer_run_pct is not None and not 0.0 <= args.max_longest_answer_run_pct <= 100.0:
        print("error: --max-longest-answer-run-pct must be between 0 and 100", file=sys.stderr)
        return 2

    records, inputs, findings = load_records(args.input, args.default_dataset, args.default_split)
    stats = build_order_stats(records, args.group_by)
    add_gate_findings(
        stats,
        findings,
        args.max_longest_answer_run,
        args.max_longest_answer_run_pct,
        args.max_edge_answer_run,
        args.min_answer_switches,
    )
    report = build_report(inputs, records, args.group_by, findings)
    write_outputs(report, args.output, args.markdown, args.csv, args.findings_csv, args.junit)

    print(f"wrote_report={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.findings_csv:
        print(f"wrote_findings_csv={args.findings_csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")

    if args.fail_on_findings and report["error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
