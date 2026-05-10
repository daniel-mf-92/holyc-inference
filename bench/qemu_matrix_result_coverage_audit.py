#!/usr/bin/env python3
"""Audit QEMU benchmark matrix launch coverage against saved run artifacts.

This host-side tool reads qemu_benchmark_matrix.py and qemu_prompt_bench.py
artifacts only. It never launches QEMU and never touches the TempleOS guest.
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


DEFAULT_RESULT_PATTERNS = ("qemu_prompt_bench*.json",)


@dataclass(frozen=True)
class CoverageRow:
    command_sha256: str
    build: str
    profile: str
    model: str
    quantization: str
    phase: str
    prompt_id: str
    prompt_sha256: str
    iteration: int
    planned: int
    observed: int
    status: str


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    field: str
    detail: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: root must be a JSON object")
    return payload


def object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed
    return None


def row_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


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


def launch_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str, int] | None:
    iteration = int_value(row.get("iteration"))
    if iteration is None:
        return None
    command_sha256 = row_text(row, "command_sha256")
    profile = row_text(row, "profile")
    model = row_text(row, "model")
    quantization = row_text(row, "quantization")
    phase = row_text(row, "phase")
    prompt_id = row_text(row, "prompt_id", "prompt")
    prompt_sha256 = row_text(row, "prompt_sha256")
    if not all([command_sha256, profile, model, quantization, phase, prompt_id, prompt_sha256]):
        return None
    return (command_sha256, profile, model, quantization, phase, prompt_id, prompt_sha256, iteration)


def key_to_row(key: tuple[str, str, str, str, str, str, str, int], *, build: str, planned: int, observed: int) -> CoverageRow:
    status = "pass" if planned == observed else "fail"
    return CoverageRow(
        command_sha256=key[0],
        build=build,
        profile=key[1],
        model=key[2],
        quantization=key[3],
        phase=key[4],
        prompt_id=key[5],
        prompt_sha256=key[6],
        iteration=key[7],
        planned=planned,
        observed=observed,
        status=status,
    )


def result_rows(path: Path) -> Iterable[dict[str, Any]]:
    payload = load_json_object(path)
    for key in ("warmups", "benchmarks", "results", "runs", "rows"):
        yield from object_list(payload.get(key))


def audit(matrix_path: Path, result_paths: Iterable[Path], args: argparse.Namespace) -> dict[str, Any]:
    matrix = load_json_object(matrix_path)
    planned: dict[tuple[str, str, str, str, str, str, str, int], int] = {}
    builds_by_command: dict[str, str] = {}
    findings: list[Finding] = []

    for build in object_list(matrix.get("builds")):
        command_hash = row_text(build, "command_sha256")
        build_name = row_text(build, "build")
        if command_hash and build_name:
            builds_by_command[command_hash] = build_name

    for index, row in enumerate(object_list(matrix.get("launches")), 1):
        key = launch_key(row)
        if key is None:
            findings.append(Finding("error", "invalid_matrix_launch", f"launches[{index}]", "matrix launch row is missing identity fields"))
            continue
        planned[key] = planned.get(key, 0) + 1

    observed: dict[tuple[str, str, str, str, str, str, str, int], int] = {}
    scanned_result_files = 0
    scanned_result_rows = 0
    for path in result_paths:
        scanned_result_files += 1
        for index, row in enumerate(result_rows(path), 1):
            scanned_result_rows += 1
            command_hash = row_text(row, "command_sha256")
            if command_hash not in builds_by_command:
                continue
            key = launch_key(row)
            if key is None:
                findings.append(Finding("error", "invalid_result_launch", f"{path}:{index}", "result row for matrix command is missing identity fields"))
                continue
            observed[key] = observed.get(key, 0) + 1

    coverage_rows: list[CoverageRow] = []
    for key in sorted(set(planned) | set(observed)):
        row = key_to_row(
            key,
            build=builds_by_command.get(key[0], ""),
            planned=planned.get(key, 0),
            observed=observed.get(key, 0),
        )
        coverage_rows.append(row)
        if row.planned == 0 and row.observed and not args.allow_extra:
            findings.append(Finding("error", "unexpected_launch", row.prompt_id, f"{row.build}/{row.phase}/iter {row.iteration} was observed but not planned"))
        elif row.planned and row.observed == 0:
            findings.append(Finding("error", "missing_launch", row.prompt_id, f"{row.build}/{row.phase}/iter {row.iteration} was planned but not observed"))
        elif row.observed > row.planned:
            findings.append(Finding("error", "duplicate_launch", row.prompt_id, f"{row.build}/{row.phase}/iter {row.iteration} observed {row.observed}, planned {row.planned}"))

    if not planned:
        findings.append(Finding("error", "missing_matrix_launches", "launches", "matrix artifact has no launch rows"))
    if scanned_result_files == 0:
        findings.append(Finding("error", "missing_result_files", "results", "no result artifacts matched"))

    return {
        "generated_at": iso_now(),
        "status": "fail" if findings else "pass",
        "matrix": str(matrix_path),
        "summary": {
            "matrix_launch_keys": len(planned),
            "result_files": scanned_result_files,
            "result_rows_scanned": scanned_result_rows,
            "coverage_rows": len(coverage_rows),
            "covered_launch_keys": sum(1 for row in coverage_rows if row.planned and row.observed == row.planned),
            "findings": len(findings),
        },
        "coverage": [asdict(row) for row in coverage_rows],
        "findings": [asdict(finding) for finding in findings],
    }


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CoverageRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)


def write_findings_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(findings)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# QEMU Matrix Result Coverage Audit",
        "",
        f"- Status: {report['status']}",
        f"- Matrix launch keys: {summary['matrix_launch_keys']}",
        f"- Result files: {summary['result_files']}",
        f"- Covered launch keys: {summary['covered_launch_keys']}",
        f"- Findings: {summary['findings']}",
    ]
    if report["findings"]:
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            lines.append("- {kind}: {field} ({detail})".format(**finding))
    else:
        lines.extend(["", "No matrix result coverage findings."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_junit(path: Path, report: dict[str, Any]) -> None:
    findings = report["findings"]
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_qemu_matrix_result_coverage_audit",
            "tests": str(max(1, len(findings))),
            "failures": str(len(findings)),
        },
    )
    if findings:
        for finding in findings:
            case = ET.SubElement(suite, "testcase", {"name": f"{finding['kind']}:{finding['field']}"})
            failure = ET.SubElement(case, "failure", {"message": finding["detail"]})
            failure.text = finding["detail"]
    else:
        ET.SubElement(suite, "testcase", {"name": "qemu_matrix_result_coverage"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def write_report(report: dict[str, Any], output_dir: Path, output_stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / output_stem
    write_json(output_base.with_suffix(".json"), report)
    write_csv(output_base.with_suffix(".csv"), report["coverage"])
    write_findings_csv(output_base.with_name(f"{output_base.name}_findings.csv"), report["findings"])
    write_markdown(output_base.with_suffix(".md"), report)
    write_junit(output_base.with_name(f"{output_base.name}_junit.xml"), report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="qemu_benchmark_matrix JSON artifact")
    parser.add_argument("results", nargs="+", type=Path, help="qemu_prompt_bench JSON artifacts or directories")
    parser.add_argument("--result-pattern", action="append", default=[], help="glob pattern for result directories")
    parser.add_argument("--allow-extra", action="store_true", help="do not fail extra observed launches for matrix commands")
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="qemu_matrix_result_coverage_audit_latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.result_pattern or list(DEFAULT_RESULT_PATTERNS)
    try:
        result_files = list(iter_input_files(args.results, patterns))
        report = audit(args.matrix, result_files, args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 2
    write_report(report, args.output_dir, args.output_stem)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
