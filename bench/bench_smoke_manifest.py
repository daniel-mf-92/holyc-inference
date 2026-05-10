#!/usr/bin/env python3
"""Build a manifest of host-side bench CI smoke scripts.

This tool scans for ``*_ci_smoke.py`` files, records their paired host tool
coverage, and writes dashboard-friendly JSON/CSV/Markdown/JUnit outputs. It is
host-side only: it never runs smoke scripts, never launches QEMU, and never
touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_PATTERNS = ("*_ci_smoke.py",)
DEFAULT_ALLOW_UNPAIRED = ("dataset_ci_smoke.py", "perf_ci_smoke.py")


@dataclass(frozen=True)
class SmokeRecord:
    source: str
    name: str
    area: str
    paired_tool: str
    paired_tool_exists: bool
    bytes: int
    lines: int
    has_shebang: bool
    has_main_guard: bool
    docstring: str


@dataclass(frozen=True)
class Finding:
    source: str
    severity: str
    kind: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def iter_smoke_files(paths: Iterable[Path], patterns: list[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        if path.is_dir():
            for pattern in patterns:
                for child in sorted(path.rglob(pattern)):
                    if child.is_file() and child not in seen:
                        seen.add(child)
                        yield child
        elif path.is_file() and path not in seen:
            seen.add(path)
            yield path


def smoke_area(name: str) -> str:
    if name.startswith("dataset_"):
        return "dataset"
    if name.startswith("eval_") or name.startswith("perplexity_"):
        return "eval"
    if name.startswith("qemu_"):
        return "qemu"
    if name.startswith("quant_"):
        return "quant"
    if name.startswith("dashboard_"):
        return "dashboard"
    if name.startswith("hceval_"):
        return "hceval"
    if name.startswith("perf_"):
        return "perf"
    if name.startswith("build_"):
        return "build"
    if name.startswith("prompt_"):
        return "prompt"
    if name.startswith("bench_"):
        return "bench"
    if name.startswith("airgap_"):
        return "airgap"
    return "other"


def paired_tool_path(path: Path) -> Path:
    suffix = "_ci_smoke.py"
    if path.name.endswith(suffix):
        return path.with_name(path.name[: -len(suffix)] + ".py")
    return path


def parse_docstring(text: str) -> str:
    try:
        module = ast.parse(text)
    except SyntaxError:
        return ""
    return ast.get_docstring(module) or ""


def has_main_guard(text: str) -> bool:
    return 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text


def load_record(path: Path) -> SmokeRecord:
    text = path.read_text(encoding="utf-8")
    paired = paired_tool_path(path)
    return SmokeRecord(
        source=str(path),
        name=path.name,
        area=smoke_area(path.name),
        paired_tool=str(paired),
        paired_tool_exists=paired.exists(),
        bytes=path.stat().st_size,
        lines=len(text.splitlines()),
        has_shebang=text.startswith("#!"),
        has_main_guard=has_main_guard(text),
        docstring=parse_docstring(text).splitlines()[0] if parse_docstring(text) else "",
    )


def audit(records: list[SmokeRecord], args: argparse.Namespace) -> list[Finding]:
    findings: list[Finding] = []
    if len(records) < args.min_smokes:
        findings.append(
            Finding("", "error", "min_smokes", f"found {len(records)} smoke script(s), expected at least {args.min_smokes}")
        )
    allowed_unpaired = set(args.allow_unpaired)
    for record in records:
        if not record.has_main_guard:
            findings.append(Finding(record.source, "error", "missing_main_guard", "smoke script must be directly executable"))
        if record.lines < args.min_lines:
            findings.append(Finding(record.source, "error", "too_short", f"script has {record.lines} lines, expected at least {args.min_lines}"))
        if args.require_shebang and not record.has_shebang:
            findings.append(Finding(record.source, "error", "missing_shebang", "smoke script must start with a shebang"))
        if args.require_paired_tools and not record.paired_tool_exists and record.name not in allowed_unpaired:
            findings.append(Finding(record.source, "error", "missing_paired_tool", f"paired tool does not exist: {record.paired_tool}"))
    return findings


def build_report(records: list[SmokeRecord], findings: list[Finding], args: argparse.Namespace) -> dict[str, object]:
    area_counts: dict[str, int] = {}
    for record in records:
        area_counts[record.area] = area_counts.get(record.area, 0) + 1
    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "summary": {
            "smoke_scripts": len(records),
            "paired_tools": sum(1 for record in records if record.paired_tool_exists),
            "unpaired_tools": sum(1 for record in records if not record.paired_tool_exists),
            "findings": len(findings),
            "areas": area_counts,
        },
        "config": {
            "patterns": args.pattern,
            "min_smokes": args.min_smokes,
            "min_lines": args.min_lines,
            "require_shebang": args.require_shebang,
            "require_paired_tools": args.require_paired_tools,
            "allow_unpaired": args.allow_unpaired,
        },
        "smokes": [asdict(record) for record in records],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, records: list[SmokeRecord]) -> None:
    fieldnames = list(SmokeRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    fieldnames = list(Finding.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, report: dict[str, object]) -> None:
    summary = report["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Bench Smoke Manifest",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Smoke scripts: {summary['smoke_scripts']}",
        f"Paired tools: {summary['paired_tools']}",
        f"Unpaired tools: {summary['unpaired_tools']}",
        f"Findings: {summary['findings']}",
        "",
        "## Areas",
        "",
    ]
    areas = summary["areas"]
    assert isinstance(areas, dict)
    for area, count in sorted(areas.items()):
        lines.append(f"- {area}: {count}")
    findings = report["findings"]
    assert isinstance(findings, list)
    if findings:
        lines.extend(["", "## Findings", ""])
        for finding in findings:
            assert isinstance(finding, dict)
            lines.append(f"- {finding['severity']}: {finding['kind']} {finding['source']} {finding['detail']}")
    else:
        lines.extend(["", "No smoke manifest findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element("testsuite", name="holyc_bench_smoke_manifest", tests="1", failures=str(1 if findings else 0))
    case = ET.SubElement(suite, "testcase", name="smoke_manifest")
    if findings:
        failure = ET.SubElement(case, "failure", message=f"{len(findings)} smoke manifest finding(s)")
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Smoke script files or directories to scan")
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS), help="Directory glob pattern")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"), help="Output directory")
    parser.add_argument("--output-stem", default="bench_smoke_manifest_latest", help="Output filename stem")
    parser.add_argument("--min-smokes", type=int, default=1, help="Minimum smoke scripts required")
    parser.add_argument("--min-lines", type=int, default=5, help="Minimum nonempty smoke script size gate")
    parser.add_argument("--require-shebang", action="store_true", help="Require smoke scripts to start with #!")
    parser.add_argument("--require-paired-tools", action="store_true", help="Require each smoke to have a paired base tool")
    parser.add_argument("--allow-unpaired", action="append", default=list(DEFAULT_ALLOW_UNPAIRED), help="Smoke filename allowed without paired base tool")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = [load_record(path) for path in iter_smoke_files(args.inputs, args.pattern)]
    findings = audit(records, args)
    report = build_report(records, findings, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    write_json(args.output_dir / f"{stem}.json", report)
    write_csv(args.output_dir / f"{stem}.csv", records)
    write_findings_csv(args.output_dir / f"{stem}_findings.csv", findings)
    write_markdown(args.output_dir / f"{stem}.md", report)
    write_junit(args.output_dir / f"{stem}_junit.xml", findings)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
