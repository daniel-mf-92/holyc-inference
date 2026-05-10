#!/usr/bin/env python3
"""Audit local benchmark matrix configs before QEMU execution.

The audit expands bench_matrix.py configs, verifies expected cell coverage,
checks for duplicate cell slugs, and rejects network-capable QEMU argument
fragments. It is host-side only and never launches QEMU.
"""

from __future__ import annotations

import argparse
import collections
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

import bench_matrix
import qemu_prompt_bench


@dataclass(frozen=True)
class Finding:
    severity: str
    matrix: str
    scope: str
    message: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def append_finding(findings: list[Finding], severity: str, matrix: Path, scope: str, message: str) -> None:
    findings.append(Finding(severity=severity, matrix=str(matrix), scope=scope, message=message))


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_arg_fragment(findings: list[Finding], matrix: Path, scope: str, args: list[str]) -> None:
    try:
        qemu_prompt_bench.reject_network_args(args)
    except ValueError as exc:
        append_finding(findings, "error", matrix, scope, str(exc))


def prompt_metadata(path: Path, findings: list[Finding], matrix: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "prompt_count": None,
        "suite_sha256": "",
    }
    if not path.exists():
        append_finding(findings, "error", matrix, "prompts", f"prompt suite does not exist: {path}")
        return metadata
    try:
        cases = qemu_prompt_bench.load_prompt_cases(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "error", matrix, "prompts", f"cannot load prompt suite: {exc}")
        return metadata
    metadata["prompt_count"] = len(cases)
    metadata["suite_sha256"] = qemu_prompt_bench.prompt_suite_hash(cases)
    if not cases:
        append_finding(findings, "error", matrix, "prompts", "prompt suite is empty")
    return metadata


def audit_matrix(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        payload = bench_matrix.read_matrix(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "error", path, "matrix", f"cannot read matrix: {exc}")
        return matrix_report(path, {}, [], [], findings, {}, args)

    root = Path(__file__).resolve().parents[1]
    base_dir = path.parent
    try:
        global_qemu_args = bench_matrix.matrix_qemu_args(payload, "matrix", base_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "error", path, "matrix.qemu_args", str(exc))
        global_qemu_args = []
    audit_arg_fragment(findings, path, "matrix.qemu_args", global_qemu_args)

    try:
        cells = bench_matrix.expand_cells(payload, base_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        append_finding(findings, "error", path, "axes", f"cannot expand cells: {exc}")
        cells = []

    slug_counts = collections.Counter(cell.slug for cell in cells)
    for slug, count in sorted(slug_counts.items()):
        if count > 1:
            append_finding(findings, "error", path, "cells", f"duplicate cell slug {slug!r} appears {count} times")

    for cell in cells:
        audit_arg_fragment(findings, path, f"cell:{cell.slug}:profile.qemu_args", cell.profile.qemu_args)
        audit_arg_fragment(findings, path, f"cell:{cell.slug}:model.qemu_args", cell.model.qemu_args)
        audit_arg_fragment(findings, path, f"cell:{cell.slug}:quantization.qemu_args", cell.quantization.qemu_args)

    prompt_info: dict[str, Any] = {}
    prompt_value = payload.get("prompts")
    if isinstance(prompt_value, str) and prompt_value:
        prompt_path = Path(prompt_value)
        prompt_info = prompt_metadata(prompt_path if prompt_path.is_absolute() else root / prompt_path, findings, path)
    else:
        append_finding(findings, "error", path, "prompts", "missing non-empty prompts path")

    expected_cells = args.expect_cells if args.expect_cells is not None else int_or_none(payload.get("expect_cells"))
    if expected_cells is not None and len(cells) != expected_cells:
        append_finding(
            findings,
            "error",
            path,
            "coverage",
            f"expanded {len(cells)} cells but expected {expected_cells}",
        )
    if args.min_cells is not None and len(cells) < args.min_cells:
        append_finding(findings, "error", path, "coverage", f"expanded {len(cells)} cells below --min-cells {args.min_cells}")

    axis_counts = {
        "profiles": len({cell.profile.name for cell in cells}),
        "models": len({cell.model.name for cell in cells}),
        "quantizations": len({cell.quantization.name for cell in cells}),
    }
    for key, minimum in (
        ("profiles", args.min_profiles),
        ("models", args.min_models),
        ("quantizations", args.min_quantizations),
    ):
        if minimum is not None and axis_counts[key] < minimum:
            append_finding(findings, "error", path, "coverage", f"{key}={axis_counts[key]} below minimum {minimum}")

    for quantization in args.require_quantization:
        if quantization not in {cell.quantization.name for cell in cells}:
            append_finding(findings, "error", path, "coverage", f"missing required quantization {quantization!r}")

    qemu_bin = str(payload.get("qemu_bin") or "qemu-system-x86_64")
    image_value = payload.get("image") if isinstance(payload.get("image"), str) else "TempleOS.img"
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = root / image_path

    cell_rows: list[dict[str, Any]] = []
    for cell in cells:
        try:
            command = bench_matrix.cell_command(qemu_bin, image_path, global_qemu_args, cell)
            airgap = qemu_prompt_bench.command_airgap_metadata(command)
            command_sha256 = qemu_prompt_bench.command_hash(command)
        except ValueError as exc:
            command = []
            airgap = {"ok": False, "explicit_nic_none": False, "legacy_net_none": False, "violations": [str(exc)]}
            command_sha256 = ""
        if not airgap["ok"]:
            append_finding(findings, "error", path, f"cell:{cell.slug}:command", "; ".join(airgap["violations"]))
        cell_rows.append(
            {
                "slug": cell.slug,
                "profile": cell.profile.name,
                "model": cell.model.name,
                "quantization": cell.quantization.name,
                "qemu_arg_count": len(global_qemu_args)
                + len(cell.profile.qemu_args)
                + len(cell.model.qemu_args)
                + len(cell.quantization.qemu_args),
                "command_sha256": command_sha256,
                "command_has_explicit_nic_none": airgap["explicit_nic_none"],
                "command_has_legacy_net_none": airgap["legacy_net_none"],
                "command_airgap_ok": airgap["ok"],
                "command_airgap_violations": list(airgap["violations"]),
            }
        )

    return matrix_report(path, payload, cells, cell_rows, findings, prompt_info, args)


def matrix_report(
    path: Path,
    payload: dict[str, Any],
    cells: list[bench_matrix.MatrixCell],
    cell_rows: list[dict[str, Any]],
    findings: list[Finding],
    prompt_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    errors = sum(1 for finding in findings if finding.severity == "error")
    return {
        "generated_at": iso_now(),
        "status": "fail" if errors else "pass",
        "matrix": str(path),
        "matrix_name": str(payload.get("name") or path.stem),
        "cell_count": len(cells),
        "axis_counts": {
            "profiles": len({cell.profile.name for cell in cells}),
            "models": len({cell.model.name for cell in cells}),
            "quantizations": len({cell.quantization.name for cell in cells}),
        },
        "coverage_gates": {
            "expect_cells": args.expect_cells if args.expect_cells is not None else int_or_none(payload.get("expect_cells")),
            "min_cells": args.min_cells,
            "min_profiles": args.min_profiles,
            "min_models": args.min_models,
            "min_quantizations": args.min_quantizations,
            "require_quantization": list(args.require_quantization),
        },
        "prompt_suite": prompt_info,
        "error_count": errors,
        "findings": [asdict(finding) for finding in findings],
        "cells": cell_rows,
    }


def combined_report(matrix_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generated_at": iso_now(),
        "status": "fail" if any(report["status"] == "fail" for report in matrix_reports) else "pass",
        "matrix_count": len(matrix_reports),
        "cell_count": sum(int(report["cell_count"]) for report in matrix_reports),
        "error_count": sum(int(report["error_count"]) for report in matrix_reports),
        "matrices": matrix_reports,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Matrix Audit",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Matrices: {report['matrix_count']}",
        f"Cells: {report['cell_count']}",
        f"Errors: {report['error_count']}",
        "",
    ]
    for matrix in report["matrices"]:
        lines.extend(
            [
                f"## {matrix['matrix_name']}",
                "",
                f"Matrix: `{matrix['matrix']}`",
                f"Status: {matrix['status']}",
                f"Cells: {matrix['cell_count']}",
                "Axis counts: profiles={profiles}, models={models}, quantizations={quantizations}".format(
                    **matrix["axis_counts"]
                ),
                "",
            ]
        )
        if matrix["findings"]:
            lines.extend(["| Severity | Scope | Message |", "| --- | --- | --- |"])
            for finding in matrix["findings"]:
                lines.append(f"| {finding['severity']} | {finding['scope']} | {finding['message']} |")
            lines.append("")
        if matrix["cells"]:
            lines.extend(["| Cell | Profile | Model | Quantization | Air-gap | Command SHA256 |", "| --- | --- | --- | --- | --- | --- |"])
            for cell in matrix["cells"]:
                lines.append(
                    f"| {cell['slug']} | {cell['profile']} | {cell['model']} | {cell['quantization']} | "
                    f"{cell['command_airgap_ok']} | {cell['command_sha256']} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def write_csv(report: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "row_type",
        "matrix",
        "status",
        "scope",
        "message",
        "slug",
        "profile",
        "model",
        "quantization",
        "command_airgap_ok",
        "command_sha256",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for matrix in report["matrices"]:
            for finding in matrix["findings"]:
                writer.writerow(
                    {
                        "row_type": "finding",
                        "matrix": finding["matrix"],
                        "status": finding["severity"],
                        "scope": finding["scope"],
                        "message": finding["message"],
                    }
                )
            for cell in matrix["cells"]:
                writer.writerow(
                    {
                        "row_type": "cell",
                        "matrix": matrix["matrix"],
                        "status": matrix["status"],
                        "slug": cell["slug"],
                        "profile": cell["profile"],
                        "model": cell["model"],
                        "quantization": cell["quantization"],
                        "command_airgap_ok": cell["command_airgap_ok"],
                        "command_sha256": cell["command_sha256"],
                    }
                )


def write_junit(report: dict[str, Any], path: Path) -> None:
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_matrix_audit",
            "tests": str(max(1, report["matrix_count"])),
            "failures": "1" if report["status"] == "fail" else "0",
        },
    )
    case = ET.SubElement(suite, "testcase", {"name": "bench_matrix_audit"})
    if report["status"] == "fail":
        failure = ET.SubElement(case, "failure", {"message": f"{report['error_count']} matrix audit errors"})
        failure.text = "\n".join(
            f"{finding['matrix']} {finding['scope']}: {finding['message']}"
            for matrix in report["matrices"]
            for finding in matrix["findings"]
            if finding["severity"] == "error"
        )
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrices", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--output-stem", default="bench_matrix_audit_latest")
    parser.add_argument("--expect-cells", type=int)
    parser.add_argument("--min-cells", type=int)
    parser.add_argument("--min-profiles", type=int)
    parser.add_argument("--min-models", type=int)
    parser.add_argument("--min-quantizations", type=int)
    parser.add_argument("--require-quantization", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = combined_report([audit_matrix(path, args) for path in args.matrices])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    csv_path = args.output_dir / f"{stem}.csv"
    junit_path = args.output_dir / f"{stem}_junit.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    write_csv(report, csv_path)
    write_junit(report, junit_path)
    print(json_path)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
