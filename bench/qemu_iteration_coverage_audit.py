#!/usr/bin/env python3
"""Audit saved QEMU benchmark artifacts for prompt iteration coverage.

This host-side tool reads existing benchmark JSON artifacts only. It never
launches QEMU and never touches the TempleOS guest.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class IterationRow:
    source: str
    row: int
    list_name: str
    phase: str
    prompt: str
    iteration: int | None
    launch_index: int | None
    exit_class: str


@dataclass(frozen=True)
class IterationGroup:
    source: str
    list_name: str
    phase: str
    prompt: str
    rows: int
    ok_rows: int
    min_iteration: int | None
    max_iteration: int | None
    missing_iterations: str
    duplicate_iterations: str


@dataclass(frozen=True)
class Finding:
    source: str
    row: int
    severity: str
    kind: str
    group: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def finite_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def row_text(row: dict[str, Any], key: str, default: str = "-") -> str:
    value = row.get(key)
    return str(value) if value not in (None, "") else default


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


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(payload, dict):
        return None, "artifact root must be a JSON object"
    return payload, ""


def collect_rows(path: Path, payload: dict[str, Any]) -> tuple[list[IterationRow], list[Finding]]:
    rows: list[IterationRow] = []
    findings: list[Finding] = []
    ordinal = 0
    for list_name in ("warmups", "benchmarks"):
        raw_rows = payload.get(list_name)
        if raw_rows is None:
            raw_rows = []
        if not isinstance(raw_rows, list):
            findings.append(Finding(str(path), 0, "error", f"invalid_{list_name}", list_name, f"{list_name} must be a list"))
            continue
        for item in raw_rows:
            ordinal += 1
            if not isinstance(item, dict):
                findings.append(Finding(str(path), ordinal, "error", "row_type", list_name, f"{list_name} row must be an object"))
                continue
            default_phase = "warmup" if list_name == "warmups" else "measured"
            rows.append(
                IterationRow(
                    source=str(path),
                    row=ordinal,
                    list_name=list_name,
                    phase=row_text(item, "phase", default_phase).lower(),
                    prompt=row_text(item, "prompt"),
                    iteration=finite_int(item.get("iteration")),
                    launch_index=finite_int(item.get("launch_index")),
                    exit_class=row_text(item, "exit_class").lower(),
                )
            )
    return rows, findings


def group_label(row: IterationRow) -> str:
    return f"{row.list_name}/{row.phase}/{row.prompt}"


def audit_artifact(path: Path, args: argparse.Namespace) -> tuple[list[IterationRow], list[IterationGroup], list[Finding]]:
    payload, error = load_json_object(path)
    if payload is None:
        return [], [], [Finding(str(path), 0, "error", "load_error", "artifact", error)]

    rows, findings = collect_rows(path, payload)
    grouped: dict[tuple[str, str, str, str], list[IterationRow]] = collections.defaultdict(list)
    for row in rows:
        grouped[(row.source, row.list_name, row.phase, row.prompt)].append(row)
        if row.iteration is None:
            findings.append(Finding(row.source, row.row, "error", "missing_iteration", group_label(row), "iteration must be an integer"))
        elif row.iteration <= 0:
            findings.append(Finding(row.source, row.row, "error", "invalid_iteration", group_label(row), "iteration must be positive"))

    groups: list[IterationGroup] = []
    for (source, list_name, phase, prompt), group_rows in sorted(grouped.items()):
        counts: dict[int, int] = collections.Counter(row.iteration for row in group_rows if row.iteration is not None)
        iterations = sorted(counts)
        min_iteration = min(iterations) if iterations else None
        max_iteration = max(iterations) if iterations else None
        missing = []
        if max_iteration is not None:
            missing = [value for value in range(1, max_iteration + 1) if value not in counts]
        duplicates = [value for value, count in counts.items() if count > 1]
        label = f"{list_name}/{phase}/{prompt}"
        if args.require_contiguous_iterations and missing:
            findings.append(
                Finding(source, 0, "error", "iteration_gap", label, f"missing iteration(s): {','.join(map(str, missing))}")
            )
        for value in duplicates:
            findings.append(Finding(source, 0, "error", "duplicate_iteration", label, f"iteration {value} appears {counts[value]} times"))
        groups.append(
            IterationGroup(
                source=source,
                list_name=list_name,
                phase=phase,
                prompt=prompt,
                rows=len(group_rows),
                ok_rows=sum(1 for row in group_rows if row.exit_class == "ok"),
                min_iteration=min_iteration,
                max_iteration=max_iteration,
                missing_iterations=",".join(map(str, missing)),
                duplicate_iterations=",".join(map(str, duplicates)),
            )
        )

    measured = [group for group in groups if group.list_name == "benchmarks" and group.phase == "measured"]
    for group in measured:
        count = group.ok_rows if args.count_only_ok else group.rows
        if count < args.min_measured_iterations_per_prompt:
            findings.append(
                Finding(
                    group.source,
                    0,
                    "error",
                    "min_measured_iterations_per_prompt",
                    f"{group.list_name}/{group.phase}/{group.prompt}",
                    f"found {count}, expected at least {args.min_measured_iterations_per_prompt}",
                )
            )

    warmups = [group for group in groups if group.list_name == "warmups" and group.phase == "warmup"]
    for group in warmups:
        count = group.ok_rows if args.count_only_ok else group.rows
        if count < args.min_warmup_iterations_per_prompt:
            findings.append(
                Finding(
                    group.source,
                    0,
                    "error",
                    "min_warmup_iterations_per_prompt",
                    f"{group.list_name}/{group.phase}/{group.prompt}",
                    f"found {count}, expected at least {args.min_warmup_iterations_per_prompt}",
                )
            )

    return rows, groups, findings


def audit(paths: Iterable[Path], args: argparse.Namespace) -> tuple[list[IterationRow], list[IterationGroup], list[Finding]]:
    all_rows: list[IterationRow] = []
    all_groups: list[IterationGroup] = []
    all_findings: list[Finding] = []
    seen_files = 0
    for path in iter_input_files(paths, args.pattern):
        seen_files += 1
        rows, groups, findings = audit_artifact(path, args)
        all_rows.extend(rows)
        all_groups.extend(groups)
        all_findings.extend(findings)
    if seen_files < args.min_artifacts:
        all_findings.append(Finding("-", 0, "error", "min_artifacts", "artifacts", f"found {seen_files}, expected at least {args.min_artifacts}"))
    if len(all_rows) < args.min_rows:
        all_findings.append(Finding("-", 0, "error", "min_rows", "rows", f"found {len(all_rows)}, expected at least {args.min_rows}"))
    return all_rows, all_groups, all_findings


def write_json(path: Path, rows: list[IterationRow], groups: list[IterationGroup], findings: list[Finding]) -> None:
    payload = {
        "generated_at": iso_now(),
        "status": "pass" if not findings else "fail",
        "summary": {
            "rows": len(rows),
            "groups": len(groups),
            "prompts": len({(group.source, group.prompt) for group in groups if group.prompt != "-"}),
            "findings": len(findings),
        },
        "groups": [asdict(group) for group in groups],
        "findings": [asdict(finding) for finding in findings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, groups: list[IterationGroup]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(IterationGroup.__dataclass_fields__))
        writer.writeheader()
        for group in groups:
            writer.writerow(asdict(group))


def write_findings_csv(path: Path, findings: list[Finding]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        for finding in findings:
            writer.writerow(asdict(finding))


def write_markdown(path: Path, rows: list[IterationRow], groups: list[IterationGroup], findings: list[Finding]) -> None:
    lines = [
        "# QEMU Iteration Coverage Audit",
        "",
        f"- Generated: {iso_now()}",
        f"- Status: {'pass' if not findings else 'fail'}",
        f"- Rows: {len(rows)}",
        f"- Groups: {len(groups)}",
        f"- Findings: {len(findings)}",
        "",
    ]
    if findings:
        lines.append("## Findings")
        lines.append("")
        for finding in findings:
            lines.append(f"- {finding.severity}: {finding.kind} ({finding.group}) {finding.detail}")
    else:
        lines.append("No iteration coverage findings.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, findings: list[Finding]) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_iteration_coverage_audit",
            "tests": "1",
            "failures": "1" if findings else "0",
            "errors": "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "iteration_coverage"})
    if findings:
        failure = ET.SubElement(case, "failure", {"message": f"{len(findings)} iteration coverage finding(s)"})
        failure.text = "\n".join(f"{finding.kind}: {finding.detail}" for finding in findings)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--pattern", action="append", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_iteration_coverage_audit_latest")
    parser.add_argument("--min-artifacts", type=int, default=1)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-measured-iterations-per-prompt", type=int, default=1)
    parser.add_argument("--min-warmup-iterations-per-prompt", type=int, default=0)
    parser.add_argument("--count-only-ok", action="store_true")
    parser.add_argument("--allow-iteration-gaps", dest="require_contiguous_iterations", action="store_false")
    parser.set_defaults(require_contiguous_iterations=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows, groups, findings = audit(args.inputs, args)
    stem = args.output_dir / args.output_stem
    write_json(stem.with_suffix(".json"), rows, groups, findings)
    write_markdown(stem.with_suffix(".md"), rows, groups, findings)
    write_csv(stem.with_suffix(".csv"), groups)
    write_findings_csv(args.output_dir / f"{args.output_stem}_findings.csv", findings)
    write_junit(args.output_dir / f"{args.output_stem}_junit.xml", findings)
    print(f"{'pass' if not findings else 'fail'} rows={len(rows)} groups={len(groups)} findings={len(findings)}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
