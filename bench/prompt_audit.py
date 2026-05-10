#!/usr/bin/env python3
"""Audit local benchmark prompt files before QEMU runs.

The audit records prompt counts, byte/line statistics, duplicate IDs, duplicate
prompt text hashes, and an order-sensitive suite hash. It is host-side only and
does not launch QEMU.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
class Issue:
    severity: str
    prompt_id: str
    message: str


@dataclass(frozen=True)
class PromptStat:
    prompt_id: str
    sha256: str
    bytes: int
    chars: int
    lines: int
    expected_tokens: int | None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def prompt_bytes(prompt: str) -> int:
    return qemu_prompt_bench.prompt_bytes(prompt)


def prompt_lines(prompt: str) -> int:
    return prompt.count("\n") + 1 if prompt else 0


def suite_hash(stats: list[PromptStat]) -> str:
    return qemu_prompt_bench.prompt_suite_hash_parts(
        [
            {"prompt_id": stat.prompt_id, "sha256": stat.sha256, "bytes": stat.bytes}
            for stat in stats
        ]
    )


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(stats: list[PromptStat]) -> dict[str, Any]:
    byte_values = [stat.bytes for stat in stats]
    line_values = [stat.lines for stat in stats]
    expected_token_values = [
        stat.expected_tokens for stat in stats if stat.expected_tokens is not None
    ]
    return {
        "prompt_count": len(stats),
        "suite_sha256": suite_hash(stats),
        "bytes_min": min(byte_values) if byte_values else None,
        "bytes_median": statistics.median(byte_values) if byte_values else None,
        "bytes_p95": percentile(byte_values, 95.0),
        "bytes_max": max(byte_values) if byte_values else None,
        "lines_max": max(line_values) if line_values else None,
        "expected_token_prompts": len(expected_token_values),
        "expected_tokens_total": sum(expected_token_values),
        "expected_tokens_min": min(expected_token_values) if expected_token_values else None,
        "expected_tokens_max": max(expected_token_values) if expected_token_values else None,
    }


def append_issue(issues: list[Issue], severity: str, prompt_id: str, message: str) -> None:
    issues.append(Issue(severity=severity, prompt_id=prompt_id, message=message))


def build_report(
    prompts: Path,
    *,
    min_prompts: int,
    max_prompt_bytes: int | None,
    expect_suite_sha256: str | None = None,
    fail_on_duplicate_text: bool,
    require_expected_tokens: bool = False,
    min_expected_token_prompts: int = 0,
    min_expected_tokens: int | None = None,
    max_expected_tokens: int | None = None,
) -> dict[str, Any]:
    cases = qemu_prompt_bench.load_prompt_cases(prompts)
    stats = [
        PromptStat(
            prompt_id=case.prompt_id,
            sha256=qemu_prompt_bench.prompt_hash(case.prompt),
            bytes=prompt_bytes(case.prompt),
            chars=len(case.prompt),
            lines=prompt_lines(case.prompt),
            expected_tokens=case.expected_tokens,
        )
        for case in cases
    ]
    issues: list[Issue] = []

    if len(stats) < min_prompts:
        append_issue(
            issues,
            "error",
            "",
            f"prompt count {len(stats)} is below required minimum {min_prompts}",
        )

    actual_suite_sha256 = suite_hash(stats)
    if expect_suite_sha256 is not None and actual_suite_sha256 != expect_suite_sha256:
        append_issue(
            issues,
            "error",
            "",
            f"prompt suite sha256 {actual_suite_sha256} does not match expected {expect_suite_sha256}",
        )

    seen_ids: dict[str, int] = {}
    for index, stat in enumerate(stats, 1):
        previous = seen_ids.get(stat.prompt_id)
        if previous is not None:
            append_issue(
                issues,
                "error",
                stat.prompt_id,
                f"duplicate prompt id also seen at row {previous}",
            )
        else:
            seen_ids[stat.prompt_id] = index

        if max_prompt_bytes is not None and stat.bytes > max_prompt_bytes:
            append_issue(
                issues,
                "error",
                stat.prompt_id,
                f"prompt is {stat.bytes} bytes, above max {max_prompt_bytes}",
            )
        if require_expected_tokens and stat.expected_tokens is None:
            append_issue(
                issues,
                "error",
                stat.prompt_id,
                "prompt is missing expected_tokens metadata",
            )
        if (
            stat.expected_tokens is not None
            and min_expected_tokens is not None
            and stat.expected_tokens < min_expected_tokens
        ):
            append_issue(
                issues,
                "error",
                stat.prompt_id,
                f"expected_tokens {stat.expected_tokens} is below min {min_expected_tokens}",
            )
        if (
            stat.expected_tokens is not None
            and max_expected_tokens is not None
            and stat.expected_tokens > max_expected_tokens
        ):
            append_issue(
                issues,
                "error",
                stat.prompt_id,
                f"expected_tokens {stat.expected_tokens} is above max {max_expected_tokens}",
            )

    expected_token_prompts = sum(1 for stat in stats if stat.expected_tokens is not None)
    if expected_token_prompts < min_expected_token_prompts:
        append_issue(
            issues,
            "error",
            "",
            "expected-token prompt count "
            f"{expected_token_prompts} is below required minimum {min_expected_token_prompts}",
        )

    seen_hashes: dict[str, str] = {}
    duplicate_text_severity = "error" if fail_on_duplicate_text else "warning"
    for stat in stats:
        previous = seen_hashes.get(stat.sha256)
        if previous is not None:
            append_issue(
                issues,
                duplicate_text_severity,
                stat.prompt_id,
                f"prompt text duplicates {previous}",
            )
        else:
            seen_hashes[stat.sha256] = stat.prompt_id

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return {
        "generated_at": iso_now(),
        "status": "fail" if error_count else "pass",
        "source": str(prompts),
        "summary": summarize(stats),
        "prompts": [asdict(stat) for stat in stats],
        "issues": [asdict(issue) for issue in issues],
        "error_count": error_count,
        "warning_count": warning_count,
        "limits": {
            "min_prompts": min_prompts,
            "max_prompt_bytes": max_prompt_bytes,
            "expect_suite_sha256": expect_suite_sha256,
            "fail_on_duplicate_text": fail_on_duplicate_text,
            "require_expected_tokens": require_expected_tokens,
            "min_expected_token_prompts": min_expected_token_prompts,
            "min_expected_tokens": min_expected_tokens,
            "max_expected_tokens": max_expected_tokens,
        },
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Prompt Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Source: {report['source']}",
        f"Errors: {report['error_count']}",
        f"Warnings: {report['warning_count']}",
        "",
        "## Summary",
        "",
        "| Prompts | Suite sha256 | Min bytes | Median bytes | P95 bytes | Max bytes | Max lines | Expected-token prompts | Expected tokens total |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| {prompt_count} | {suite_sha256} | {bytes_min} | {bytes_median} | {bytes_p95} | {bytes_max} | {lines_max} | {expected_token_prompts} | {expected_tokens_total} |".format(
            **{key: format_value(value) for key, value in summary.items()}
        ),
        "",
        "## Prompts",
        "",
        "| Prompt | Bytes | Chars | Lines | Expected tokens | SHA256 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for stat in report["prompts"]:
        lines.append(
            "| {prompt_id} | {bytes} | {chars} | {lines} | {expected_tokens} | {sha256} |".format(
                **{key: format_value(value) for key, value in stat.items()}
            )
        )

    if report["issues"]:
        lines.extend(["", "## Issues", "", "| Severity | Prompt | Message |", "| --- | --- | --- |"])
        for issue in report["issues"]:
            lines.append("| {severity} | {prompt_id} | {message} |".format(**issue))
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "row_type",
        "prompt_id",
        "sha256",
        "bytes",
        "chars",
        "lines",
        "expected_tokens",
        "severity",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for stat in report["prompts"]:
            writer.writerow(
                {
                    "row_type": "prompt",
                    "prompt_id": stat["prompt_id"],
                    "sha256": stat["sha256"],
                    "bytes": stat["bytes"],
                    "chars": stat["chars"],
                    "lines": stat["lines"],
                    "expected_tokens": stat["expected_tokens"],
                    "severity": "",
                    "message": "",
                }
            )
        for issue in report["issues"]:
            writer.writerow(
                {
                    "row_type": "issue",
                    "prompt_id": issue["prompt_id"],
                    "sha256": "",
                    "bytes": "",
                    "chars": "",
                    "lines": "",
                    "expected_tokens": "",
                    "severity": issue["severity"],
                    "message": issue["message"],
                }
            )


def write_junit(report: dict[str, Any], path: Path) -> None:
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    warnings = [issue for issue in report["issues"] if issue["severity"] == "warning"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_prompt_audit",
            "tests": "1",
            "failures": "1" if errors else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "prompt_audit",
            "name": f"prompt_suite:{report['summary']['suite_sha256']}",
        },
    )
    if errors:
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "prompt_audit_error",
                "message": f"{len(errors)} prompt audit error(s), {len(warnings)} warning(s)",
            },
        )
        failure.text = "\n".join(
            f"{issue['prompt_id'] or report['source']}: {issue['message']}" for issue in errors
        )
    elif warnings:
        system_out = ET.SubElement(case, "system-out")
        system_out.text = "\n".join(
            f"{issue['prompt_id'] or report['source']}: {issue['message']}" for issue in warnings
        )
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="Prompt JSON, JSONL, or text file")
    parser.add_argument("--output", type=Path, default=Path("bench/results/prompt_audit_latest.json"))
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    parser.add_argument("--csv", type=Path, help="Optional CSV prompt/issue report path")
    parser.add_argument("--junit", type=Path, help="Optional JUnit XML audit report path")
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument("--max-prompt-bytes", type=int)
    parser.add_argument(
        "--expect-suite-sha256",
        help="Fail when the order-sensitive prompt suite SHA256 differs from this pinned value",
    )
    parser.add_argument(
        "--fail-on-duplicate-text",
        action="store_true",
        help="Treat duplicate prompt text as an error instead of a warning",
    )
    parser.add_argument(
        "--require-expected-tokens",
        action="store_true",
        help="Fail if any prompt omits expected_tokens metadata",
    )
    parser.add_argument(
        "--min-expected-token-prompts",
        type=int,
        default=0,
        help="Fail unless at least this many prompts declare expected_tokens",
    )
    parser.add_argument(
        "--min-expected-tokens",
        type=int,
        help="Fail when a declared expected_tokens value is below this bound",
    )
    parser.add_argument(
        "--max-expected-tokens",
        type=int,
        help="Fail when a declared expected_tokens value is above this bound",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_prompts < 0:
        print("error: --min-prompts must be non-negative", file=sys.stderr)
        return 2
    if args.max_prompt_bytes is not None and args.max_prompt_bytes < 1:
        print("error: --max-prompt-bytes must be positive", file=sys.stderr)
        return 2
    if args.min_expected_token_prompts < 0:
        print("error: --min-expected-token-prompts must be non-negative", file=sys.stderr)
        return 2
    if args.min_expected_tokens is not None and args.min_expected_tokens < 0:
        print("error: --min-expected-tokens must be non-negative", file=sys.stderr)
        return 2
    if args.max_expected_tokens is not None and args.max_expected_tokens < 0:
        print("error: --max-expected-tokens must be non-negative", file=sys.stderr)
        return 2
    if (
        args.min_expected_tokens is not None
        and args.max_expected_tokens is not None
        and args.min_expected_tokens > args.max_expected_tokens
    ):
        print("error: --min-expected-tokens cannot exceed --max-expected-tokens", file=sys.stderr)
        return 2

    try:
        report = build_report(
            args.prompts,
            min_prompts=args.min_prompts,
            max_prompt_bytes=args.max_prompt_bytes,
            expect_suite_sha256=args.expect_suite_sha256,
            fail_on_duplicate_text=args.fail_on_duplicate_text,
            require_expected_tokens=args.require_expected_tokens,
            min_expected_token_prompts=args.min_expected_token_prompts,
            min_expected_tokens=args.min_expected_tokens,
            max_expected_tokens=args.max_expected_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(report, args.csv)
    if args.junit:
        args.junit.parent.mkdir(parents=True, exist_ok=True)
        write_junit(report, args.junit)

    print(f"wrote_json={args.output}")
    if args.markdown:
        print(f"wrote_markdown={args.markdown}")
    if args.csv:
        print(f"wrote_csv={args.csv}")
    if args.junit:
        print(f"wrote_junit={args.junit}")
    print(f"status={report['status']}")
    print(f"prompt_count={report['summary']['prompt_count']}")
    print(f"expected_token_prompts={report['summary']['expected_token_prompts']}")
    print(f"suite_sha256={report['summary']['suite_sha256']}")
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
