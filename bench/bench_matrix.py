#!/usr/bin/env python3
"""Run an air-gapped QEMU prompt benchmark matrix.

The matrix runner is host-side orchestration around qemu_prompt_bench.py. It
expands a local JSON matrix across profiles, models, and quantization formats,
runs each cell into an isolated results directory, then writes one summary under
bench/results. QEMU launches are delegated to qemu_prompt_bench.py, which always
injects `-nic none` and rejects networking arguments.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qemu_prompt_bench


DEFAULT_MATRIX_NAME = "bench-matrix"
SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class MatrixAxisItem:
    name: str
    qemu_args: list[str]


@dataclass(frozen=True)
class MatrixCell:
    profile: MatrixAxisItem
    model: MatrixAxisItem
    quantization: MatrixAxisItem

    @property
    def slug(self) -> str:
        return slugify(f"{self.profile.name}_{self.model.name}_{self.quantization.name}")


@dataclass(frozen=True)
class MatrixCellResult:
    profile: str
    model: str
    quantization: str
    commit: str
    status: str
    output_dir: str
    report: str
    command: list[str]
    command_sha256: str
    launch_plan_sha256: str
    prompts: int
    prompt_suite_sha256: str
    prompt_bytes_total: int
    prompt_bytes_min: int | None
    prompt_bytes_max: int | None
    total_tokens: int | None
    total_elapsed_us: int | None
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    wall_tok_per_s_median: float | None
    ttft_us_p95: float | None
    host_overhead_pct_median: float | None
    host_child_cpu_us_median: float | None
    host_child_cpu_pct_median: float | None
    host_child_tok_per_cpu_s_median: float | None
    host_child_peak_rss_bytes_max: int | None
    us_per_token_median: float | None
    wall_us_per_token_median: float | None
    max_memory_bytes: int | None
    variability_findings: int


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", value.strip()).strip("-._")
    return slug or "cell"


def read_matrix(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix file must contain a JSON object")
    return payload


def as_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def as_args(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return list(value)


def as_arg_files(value: Any, field: str, base_dir: Path) -> list[Path]:
    if value is None:
        return []
    raw_paths = [value] if isinstance(value, str) else value
    if not isinstance(raw_paths, list) or not all(isinstance(item, str) for item in raw_paths):
        raise ValueError(f"{field} must be a string or list of strings")
    paths = []
    for item in raw_paths:
        path = Path(item)
        paths.append(path if path.is_absolute() else base_dir / path)
    return paths


def matrix_qemu_args(row: dict[str, Any], field: str, base_dir: Path) -> list[str]:
    file_args = qemu_prompt_bench.load_qemu_args_files(
        as_arg_files(row.get("qemu_args_file") or row.get("qemu_args_files"), f"{field}.qemu_args_files", base_dir)
    )
    return file_args + as_args(row.get("qemu_args"), f"{field}.qemu_args")


def axis_items(
    payload: dict[str, Any],
    key: str,
    default_name: str = "default",
    base_dir: Path | None = None,
) -> list[MatrixAxisItem]:
    rows = payload.get(key)
    if rows is None:
        return [MatrixAxisItem(default_name, [])]
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{key} must be a non-empty list")

    items: list[MatrixAxisItem] = []
    for index, row in enumerate(rows, 1):
        label = f"{key}[{index}]"
        if isinstance(row, str):
            items.append(MatrixAxisItem(row, []))
            continue
        if not isinstance(row, dict):
            raise ValueError(f"{label} must be a string or object")
        items.append(
            MatrixAxisItem(
                name=as_string(row.get("name"), f"{label}.name"),
                qemu_args=matrix_qemu_args(row, label, base_dir or Path.cwd()),
            )
        )
    return items


def expand_cells(payload: dict[str, Any], base_dir: Path | None = None) -> list[MatrixCell]:
    base = base_dir or Path.cwd()
    profiles = axis_items(payload, "profiles", base_dir=base)
    models = axis_items(payload, "models", default_name="", base_dir=base)
    quantizations = axis_items(payload, "quantizations", default_name="", base_dir=base)
    return [
        MatrixCell(profile=profile, model=model, quantization=quantization)
        for profile in profiles
        for model in models
        for quantization in quantizations
    ]


def resolve_path(value: str | None, fallback: Path | None, root: Path) -> Path:
    if fallback is not None:
        return fallback
    if value is None:
        raise ValueError("missing required path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_executable(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute() or len(path.parts) == 1:
        return value
    return str(root / path)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def median_cell_tok_per_s(report: dict[str, Any]) -> float | None:
    values = [
        summary.get("tok_per_s_median")
        for summary in report.get("summaries", [])
        if isinstance(summary, dict) and summary.get("tok_per_s_median") is not None
    ]
    return statistics.median(float(value) for value in values) if values else None


def max_cell_memory_bytes(report: dict[str, Any]) -> int | None:
    values = [
        summary.get("memory_bytes_max")
        for summary in report.get("summaries", [])
        if isinstance(summary, dict) and summary.get("memory_bytes_max") is not None
    ]
    return max(int(value) for value in values) if values else None


def suite_float(report: dict[str, Any], key: str) -> float | None:
    suite_summary = report.get("suite_summary")
    if not isinstance(suite_summary, dict):
        return None
    value = suite_summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def suite_int(report: dict[str, Any], key: str) -> int | None:
    value = suite_float(report, key)
    return int(value) if value is not None else None


def report_commit(report: dict[str, Any]) -> str:
    commits = sorted(
        {
            str(row.get("commit"))
            for section in ("benchmarks", "warmups")
            for row in report.get(section, [])
            if isinstance(row, dict) and row.get("commit")
        }
    )
    return commits[0] if len(commits) == 1 else ""


def cell_command(
    qemu_bin: str,
    image: Path,
    global_qemu_args: list[str],
    cell: MatrixCell,
) -> list[str]:
    args = global_qemu_args + cell.profile.qemu_args + cell.model.qemu_args + cell.quantization.qemu_args
    return qemu_prompt_bench.build_command(qemu_bin, image, args)


def run_cell(
    *,
    cell: MatrixCell,
    image: Path,
    prompts: Path,
    qemu_bin: str,
    global_qemu_args: list[str],
    timeout: float,
    warmup: int,
    repeat: int,
    max_suite_cv_pct: float | None,
    max_prompt_cv_pct: float | None,
    max_suite_iqr_pct: float | None,
    max_prompt_iqr_pct: float | None,
    matrix_dir: Path,
    dry_run: bool,
) -> MatrixCellResult:
    command = cell_command(qemu_bin, image, global_qemu_args, cell)
    prompt_cases = qemu_prompt_bench.load_prompt_cases(prompts)
    prompt_count = len(prompt_cases)
    prompt_suite_sha256 = qemu_prompt_bench.prompt_suite_hash(prompt_cases)
    prompt_byte_counts = [qemu_prompt_bench.prompt_bytes(case.prompt) for case in prompt_cases]
    prompt_bytes_total = sum(prompt_byte_counts)
    prompt_bytes_min = min(prompt_byte_counts) if prompt_byte_counts else None
    prompt_bytes_max = max(prompt_byte_counts) if prompt_byte_counts else None
    cell_output_dir = matrix_dir / cell.slug
    report_path = cell_output_dir / "qemu_prompt_bench_latest.json"

    if dry_run:
        return MatrixCellResult(
            profile=cell.profile.name,
            model=cell.model.name,
            quantization=cell.quantization.name,
            commit="",
            status="planned",
            output_dir=str(cell_output_dir),
            report=str(report_path),
            command=command,
            command_sha256=qemu_prompt_bench.command_hash(command),
            launch_plan_sha256=qemu_prompt_bench.launch_plan_hash(
                qemu_prompt_bench.dry_run_launch_plan(prompt_cases, warmup=warmup, repeat=repeat)
            ),
            prompts=prompt_count,
            prompt_suite_sha256=prompt_suite_sha256,
            prompt_bytes_total=prompt_bytes_total,
            prompt_bytes_min=prompt_bytes_min,
            prompt_bytes_max=prompt_bytes_max,
            total_tokens=None,
            total_elapsed_us=None,
            measured_runs=0,
            warmup_runs=0,
            median_tok_per_s=None,
            wall_tok_per_s_median=None,
            ttft_us_p95=None,
            host_overhead_pct_median=None,
            host_child_cpu_us_median=None,
            host_child_cpu_pct_median=None,
            host_child_tok_per_cpu_s_median=None,
            host_child_peak_rss_bytes_max=None,
            us_per_token_median=None,
            wall_us_per_token_median=None,
            max_memory_bytes=None,
            variability_findings=0,
        )

    argv = [
        "--image",
        str(image),
        "--prompts",
        str(prompts),
        "--qemu-bin",
        qemu_bin,
        "--timeout",
        str(timeout),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--output-dir",
        str(cell_output_dir),
        "--profile",
        cell.profile.name,
        "--model",
        cell.model.name,
        "--quantization",
        cell.quantization.name,
    ]
    if max_suite_cv_pct is not None:
        argv.extend(["--max-suite-cv-pct", str(max_suite_cv_pct)])
    if max_prompt_cv_pct is not None:
        argv.extend(["--max-prompt-cv-pct", str(max_prompt_cv_pct)])
    if max_suite_iqr_pct is not None:
        argv.extend(["--max-suite-iqr-pct", str(max_suite_iqr_pct)])
    if max_prompt_iqr_pct is not None:
        argv.extend(["--max-prompt-iqr-pct", str(max_prompt_iqr_pct)])
    for arg in global_qemu_args + cell.profile.qemu_args + cell.model.qemu_args + cell.quantization.qemu_args:
        argv.append(f"--qemu-arg={arg}")

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status_code = qemu_prompt_bench.main(argv)
    if status_code not in {0, 1} or not report_path.exists():
        raise RuntimeError(f"benchmark cell {cell.slug} failed with status {status_code}: {stdout.getvalue()}")

    report = load_report(report_path)
    findings = report.get("variability_findings", [])
    return MatrixCellResult(
        profile=cell.profile.name,
        model=cell.model.name,
        quantization=cell.quantization.name,
        commit=report_commit(report),
        status=str(report.get("status", "unknown")),
        output_dir=str(cell_output_dir),
        report=str(report_path),
        command=command,
        command_sha256=str(report.get("command_sha256") or qemu_prompt_bench.command_hash(command)),
        launch_plan_sha256=str(report.get("launch_plan_sha256", "")),
        prompts=prompt_count,
        prompt_suite_sha256=str(
            (report.get("prompt_suite") or {}).get("suite_sha256") or prompt_suite_sha256
        ),
        prompt_bytes_total=prompt_bytes_total,
        prompt_bytes_min=prompt_bytes_min,
        prompt_bytes_max=prompt_bytes_max,
        total_tokens=suite_int(report, "total_tokens"),
        total_elapsed_us=suite_int(report, "total_elapsed_us"),
        measured_runs=len(report.get("benchmarks", [])),
        warmup_runs=len(report.get("warmups", [])),
        median_tok_per_s=median_cell_tok_per_s(report),
        wall_tok_per_s_median=suite_float(report, "wall_tok_per_s_median"),
        ttft_us_p95=suite_float(report, "ttft_us_p95"),
        host_overhead_pct_median=suite_float(report, "host_overhead_pct_median"),
        host_child_cpu_us_median=suite_float(report, "host_child_cpu_us_median"),
        host_child_cpu_pct_median=suite_float(report, "host_child_cpu_pct_median"),
        host_child_tok_per_cpu_s_median=suite_float(report, "host_child_tok_per_cpu_s_median"),
        host_child_peak_rss_bytes_max=(
            int(suite_float(report, "host_child_peak_rss_bytes_max"))
            if suite_float(report, "host_child_peak_rss_bytes_max") is not None
            else None
        ),
        us_per_token_median=suite_float(report, "us_per_token_median"),
        wall_us_per_token_median=suite_float(report, "wall_us_per_token_median"),
        max_memory_bytes=max_cell_memory_bytes(report),
        variability_findings=len(findings) if isinstance(findings, list) else 0,
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Matrix",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Matrix: {report['matrix_name']}",
        f"Cells: {len(report['cells'])}",
        "Variability gates: suite CV <= {suite}, prompt CV <= {prompt}, suite IQR <= {suite_iqr}, prompt IQR <= {prompt_iqr}".format(
            suite=format_gate(report["variability_gates"].get("max_suite_cv_pct")),
            prompt=format_gate(report["variability_gates"].get("max_prompt_cv_pct")),
            suite_iqr=format_gate(report["variability_gates"].get("max_suite_iqr_pct")),
            prompt_iqr=format_gate(report["variability_gates"].get("max_prompt_iqr_pct")),
        ),
        "",
        "## Cells",
        "",
    ]
    if report["cells"]:
        lines.append(
            "| Profile | Model | Quantization | Commit | Status | Prompts | Prompt bytes | Prompt byte range | Total tokens | Total elapsed us | Prompt suite | Command SHA256 | Launch plan SHA256 | Runs | Warmups | Guest tok/s | Wall tok/s | P95 TTFT us | Host overhead % | Host child CPU us | Host child CPU % | Host child tok/CPU s | Max host child RSS bytes | Guest us/token | Wall us/token | Max memory bytes | Variability findings |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for cell in report["cells"]:
            median = cell["median_tok_per_s"]
            memory = cell["max_memory_bytes"]
            median_cell = f"{median:.3f}" if median is not None else "-"
            wall_tok = cell["wall_tok_per_s_median"]
            wall_tok_cell = f"{wall_tok:.3f}" if wall_tok is not None else "-"
            ttft = cell["ttft_us_p95"]
            ttft_cell = f"{ttft:.3f}" if ttft is not None else "-"
            overhead = cell["host_overhead_pct_median"]
            overhead_cell = f"{overhead:.3f}" if overhead is not None else "-"
            child_cpu_us = cell["host_child_cpu_us_median"]
            child_cpu_us_cell = f"{child_cpu_us:.3f}" if child_cpu_us is not None else "-"
            child_cpu_pct = cell["host_child_cpu_pct_median"]
            child_cpu_pct_cell = f"{child_cpu_pct:.3f}" if child_cpu_pct is not None else "-"
            child_tok_per_cpu = cell["host_child_tok_per_cpu_s_median"]
            child_tok_per_cpu_cell = f"{child_tok_per_cpu:.3f}" if child_tok_per_cpu is not None else "-"
            child_rss = cell["host_child_peak_rss_bytes_max"]
            child_rss_cell = str(child_rss) if child_rss is not None else "-"
            us_per_token = cell["us_per_token_median"]
            us_per_token_cell = f"{us_per_token:.3f}" if us_per_token is not None else "-"
            wall_us_per_token = cell["wall_us_per_token_median"]
            wall_us_per_token_cell = f"{wall_us_per_token:.3f}" if wall_us_per_token is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            prompt_byte_range = (
                f"{cell['prompt_bytes_min']}-{cell['prompt_bytes_max']}"
                if cell["prompt_bytes_min"] is not None and cell["prompt_bytes_max"] is not None
                else "-"
            )
            lines.append(
                f"| {cell['profile']} | {cell['model']} | {cell['quantization']} | {cell['commit'] or '-'} | {cell['status']} | "
                f"{cell['prompts']} | {cell['prompt_bytes_total']} | {prompt_byte_range} | "
                f"{format_gate(cell['total_tokens'])} | {format_gate(cell['total_elapsed_us'])} | "
                f"{cell['prompt_suite_sha256']} | {cell['command_sha256']} | {cell['launch_plan_sha256']} | "
                f"{cell['measured_runs']} | {cell['warmup_runs']} | "
                f"{median_cell} | {wall_tok_cell} | {ttft_cell} | {overhead_cell} | "
                f"{child_cpu_us_cell} | {child_cpu_pct_cell} | {child_tok_per_cpu_cell} | {child_rss_cell} | {us_per_token_cell} | "
                f"{wall_us_per_token_cell} | {memory_cell} | {cell['variability_findings']} |"
            )
    else:
        lines.append("No matrix cells were expanded.")
    return "\n".join(lines) + "\n"


def format_gate(value: Any) -> str:
    return "-" if value is None else str(value)


def write_matrix_report(
    *,
    matrix_name: str,
    matrix_file: Path,
    output_dir: Path,
    matrix_dir: Path,
    cells: list[MatrixCellResult],
    dry_run: bool,
    max_suite_cv_pct: float | None,
    max_prompt_cv_pct: float | None,
    max_suite_iqr_pct: float | None,
    max_prompt_iqr_pct: float | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": iso_now(),
        "status": "planned" if dry_run else ("pass" if all(cell.status == "pass" for cell in cells) else "fail"),
        "matrix_name": matrix_name,
        "matrix_file": str(matrix_file),
        "matrix_dir": str(matrix_dir),
        "dry_run": dry_run,
        "variability_gates": {
            "max_suite_cv_pct": max_suite_cv_pct,
            "max_prompt_cv_pct": max_prompt_cv_pct,
            "max_suite_iqr_pct": max_suite_iqr_pct,
            "max_prompt_iqr_pct": max_prompt_iqr_pct,
        },
        "cells": [asdict(cell) for cell in cells],
    }
    json_path = output_dir / "bench_matrix_latest.json"
    md_path = output_dir / "bench_matrix_latest.md"
    csv_path = output_dir / "bench_matrix_latest.csv"
    summary_csv_path = output_dir / "bench_matrix_summary_latest.csv"
    junit_path = output_dir / "bench_matrix_junit_latest.xml"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_matrix_csv(cells, csv_path)
    write_matrix_summary_csv(cells, summary_csv_path)
    write_matrix_junit(cells, junit_path)
    return json_path


def write_matrix_csv(cells: list[MatrixCellResult], path: Path) -> None:
    fields = [
        "profile",
        "model",
        "quantization",
        "commit",
        "status",
        "output_dir",
        "report",
        "command_sha256",
        "launch_plan_sha256",
        "prompts",
        "prompt_suite_sha256",
        "prompt_bytes_total",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "total_tokens",
        "total_elapsed_us",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "wall_tok_per_s_median",
        "ttft_us_p95",
        "host_overhead_pct_median",
        "host_child_cpu_us_median",
        "host_child_cpu_pct_median",
        "host_child_tok_per_cpu_s_median",
        "host_child_peak_rss_bytes_max",
        "us_per_token_median",
        "wall_us_per_token_median",
        "max_memory_bytes",
        "variability_findings",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            row = asdict(cell)
            row["command"] = json.dumps(cell.command, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def finite_values(cells: list[MatrixCellResult], field: str) -> list[float]:
    values: list[float] = []
    for cell in cells:
        value = getattr(cell, field)
        if value is not None:
            values.append(float(value))
    return values


def median_field(cells: list[MatrixCellResult], field: str) -> float | None:
    values = finite_values(cells, field)
    return statistics.median(values) if values else None


def min_field(cells: list[MatrixCellResult], field: str) -> float | None:
    values = finite_values(cells, field)
    return min(values) if values else None


def max_field(cells: list[MatrixCellResult], field: str) -> float | None:
    values = finite_values(cells, field)
    return max(values) if values else None


def matrix_summary_row(cells: list[MatrixCellResult]) -> dict[str, Any]:
    statuses = {cell.status for cell in cells}
    if not cells:
        status = "empty"
    elif statuses == {"pass"}:
        status = "pass"
    elif statuses == {"planned"}:
        status = "planned"
    else:
        status = "fail"
    prompt_mins = [cell.prompt_bytes_min for cell in cells if cell.prompt_bytes_min is not None]
    prompt_maxes = [cell.prompt_bytes_max for cell in cells if cell.prompt_bytes_max is not None]
    return {
        "scope": "matrix",
        "profile": "",
        "model": "",
        "quantization": "",
        "status": status,
        "cells": len(cells),
        "passing_cells": sum(1 for cell in cells if cell.status == "pass"),
        "failing_cells": sum(1 for cell in cells if cell.status == "fail"),
        "planned_cells": sum(1 for cell in cells if cell.status == "planned"),
        "prompts": sum(cell.prompts for cell in cells),
        "prompt_bytes_total": sum(cell.prompt_bytes_total for cell in cells),
        "prompt_bytes_min": min(prompt_mins) if prompt_mins else None,
        "prompt_bytes_max": max(prompt_maxes) if prompt_maxes else None,
        "total_tokens": sum(
            cell.total_tokens for cell in cells if cell.total_tokens is not None
        )
        if any(cell.total_tokens is not None for cell in cells)
        else None,
        "total_elapsed_us": sum(
            cell.total_elapsed_us for cell in cells if cell.total_elapsed_us is not None
        )
        if any(cell.total_elapsed_us is not None for cell in cells)
        else None,
        "measured_runs": sum(cell.measured_runs for cell in cells),
        "warmup_runs": sum(cell.warmup_runs for cell in cells),
        "median_tok_per_s_min": min_field(cells, "median_tok_per_s"),
        "median_tok_per_s_median": median_field(cells, "median_tok_per_s"),
        "median_tok_per_s_max": max_field(cells, "median_tok_per_s"),
        "wall_tok_per_s_median": median_field(cells, "wall_tok_per_s_median"),
        "ttft_us_p95_max": max_field(cells, "ttft_us_p95"),
        "host_overhead_pct_median": median_field(cells, "host_overhead_pct_median"),
        "host_child_cpu_pct_median": median_field(cells, "host_child_cpu_pct_median"),
        "host_child_tok_per_cpu_s_median": median_field(cells, "host_child_tok_per_cpu_s_median"),
        "host_child_peak_rss_bytes_max": max(
            (cell.host_child_peak_rss_bytes_max for cell in cells if cell.host_child_peak_rss_bytes_max is not None),
            default=None,
        ),
        "us_per_token_median": median_field(cells, "us_per_token_median"),
        "wall_us_per_token_median": median_field(cells, "wall_us_per_token_median"),
        "max_memory_bytes": max(
            (cell.max_memory_bytes for cell in cells if cell.max_memory_bytes is not None),
            default=None,
        ),
        "variability_findings": sum(cell.variability_findings for cell in cells),
    }


def cell_summary_row(cell: MatrixCellResult) -> dict[str, Any]:
    return {
        "scope": "cell",
        "profile": cell.profile,
        "model": cell.model,
        "quantization": cell.quantization,
        "status": cell.status,
        "cells": 1,
        "passing_cells": 1 if cell.status == "pass" else 0,
        "failing_cells": 1 if cell.status == "fail" else 0,
        "planned_cells": 1 if cell.status == "planned" else 0,
        "prompts": cell.prompts,
        "prompt_bytes_total": cell.prompt_bytes_total,
        "prompt_bytes_min": cell.prompt_bytes_min,
        "prompt_bytes_max": cell.prompt_bytes_max,
        "total_tokens": cell.total_tokens,
        "total_elapsed_us": cell.total_elapsed_us,
        "measured_runs": cell.measured_runs,
        "warmup_runs": cell.warmup_runs,
        "median_tok_per_s_min": cell.median_tok_per_s,
        "median_tok_per_s_median": cell.median_tok_per_s,
        "median_tok_per_s_max": cell.median_tok_per_s,
        "wall_tok_per_s_median": cell.wall_tok_per_s_median,
        "ttft_us_p95_max": cell.ttft_us_p95,
        "host_overhead_pct_median": cell.host_overhead_pct_median,
        "host_child_cpu_pct_median": cell.host_child_cpu_pct_median,
        "host_child_tok_per_cpu_s_median": cell.host_child_tok_per_cpu_s_median,
        "host_child_peak_rss_bytes_max": cell.host_child_peak_rss_bytes_max,
        "us_per_token_median": cell.us_per_token_median,
        "wall_us_per_token_median": cell.wall_us_per_token_median,
        "max_memory_bytes": cell.max_memory_bytes,
        "variability_findings": cell.variability_findings,
    }


def write_matrix_summary_csv(cells: list[MatrixCellResult], path: Path) -> None:
    fields = [
        "scope",
        "profile",
        "model",
        "quantization",
        "status",
        "cells",
        "passing_cells",
        "failing_cells",
        "planned_cells",
        "prompts",
        "prompt_bytes_total",
        "prompt_bytes_min",
        "prompt_bytes_max",
        "total_tokens",
        "total_elapsed_us",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s_min",
        "median_tok_per_s_median",
        "median_tok_per_s_max",
        "wall_tok_per_s_median",
        "ttft_us_p95_max",
        "host_overhead_pct_median",
        "host_child_cpu_pct_median",
        "host_child_tok_per_cpu_s_median",
        "host_child_peak_rss_bytes_max",
        "us_per_token_median",
        "wall_us_per_token_median",
        "max_memory_bytes",
        "variability_findings",
    ]
    rows = [matrix_summary_row(cells)] + [cell_summary_row(cell) for cell in cells]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_matrix_junit(cells: list[MatrixCellResult], path: Path) -> None:
    failures = sum(1 for cell in cells if cell.status == "fail")
    suite = ET.Element(
        "testsuite",
        {
            "name": "holyc_bench_matrix",
            "tests": str(len(cells)),
            "failures": str(failures),
            "errors": "0",
        },
    )
    for cell in cells:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "bench_matrix.cell",
                "name": f"{cell.profile}:{cell.model}:{cell.quantization}",
            },
        )
        if cell.status != "fail":
            continue
        message = (
            f"status={cell.status} runs={cell.measured_runs} "
            f"warmups={cell.warmup_runs} variability_findings={cell.variability_findings}"
        )
        failure = ET.SubElement(
            case,
            "failure",
            {
                "type": "benchmark_matrix_cell_failure",
                "message": message,
            },
        )
        failure.text = "\n".join(
            [
                f"profile={cell.profile}",
                f"model={cell.model}",
                f"quantization={cell.quantization}",
                f"status={cell.status}",
                f"report={cell.report}",
                f"output_dir={cell.output_dir}",
                f"prompt_suite_sha256={cell.prompt_suite_sha256}",
                f"prompt_bytes_total={cell.prompt_bytes_total}",
                f"prompt_bytes_min={cell.prompt_bytes_min}",
                f"prompt_bytes_max={cell.prompt_bytes_max}",
                f"command_sha256={cell.command_sha256}",
                f"launch_plan_sha256={cell.launch_plan_sha256}",
                f"measured_runs={cell.measured_runs}",
                f"warmup_runs={cell.warmup_runs}",
                f"median_tok_per_s={cell.median_tok_per_s}",
                f"wall_tok_per_s_median={cell.wall_tok_per_s_median}",
                f"ttft_us_p95={cell.ttft_us_p95}",
                f"host_overhead_pct_median={cell.host_overhead_pct_median}",
                f"host_child_cpu_us_median={cell.host_child_cpu_us_median}",
                f"host_child_cpu_pct_median={cell.host_child_cpu_pct_median}",
                f"host_child_tok_per_cpu_s_median={cell.host_child_tok_per_cpu_s_median}",
                f"host_child_peak_rss_bytes_max={cell.host_child_peak_rss_bytes_max}",
                f"us_per_token_median={cell.us_per_token_median}",
                f"wall_us_per_token_median={cell.wall_us_per_token_median}",
                f"max_memory_bytes={cell.max_memory_bytes}",
                f"variability_findings={cell.variability_findings}",
                f"command={json.dumps(cell.command, separators=(',', ':'))}",
            ]
        )
    ET.indent(suite)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    with path.open("ab") as handle:
        handle.write(b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="Local JSON benchmark matrix")
    parser.add_argument("--image", type=Path, help="Override matrix image path")
    parser.add_argument("--prompts", type=Path, help="Override matrix prompts path")
    parser.add_argument("--qemu-bin", help="Override matrix QEMU binary")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--max-suite-cv-pct", type=float, help="Fail cells whose suite tok/s CV exceeds this percentage")
    parser.add_argument("--max-prompt-cv-pct", type=float, help="Fail cells whose per-prompt tok/s CV exceeds this percentage")
    parser.add_argument(
        "--max-suite-iqr-pct",
        type=float,
        help="Fail cells whose suite tok/s interquartile spread exceeds this percentage",
    )
    parser.add_argument(
        "--max-prompt-iqr-pct",
        type=float,
        help="Fail cells whose per-prompt tok/s interquartile spread exceeds this percentage",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("bench/results"))
    parser.add_argument("--dry-run", action="store_true", help="Validate and write planned commands without launching")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    try:
        payload = read_matrix(args.matrix)
        matrix_name = str(payload.get("name") or DEFAULT_MATRIX_NAME)
        image = resolve_path(payload.get("image"), args.image, root)
        prompts = resolve_path(payload.get("prompts"), args.prompts, root)
        qemu_bin = resolve_executable(
            args.qemu_bin or str(payload.get("qemu_bin") or "qemu-system-x86_64"),
            root,
        )
        timeout = (
            args.timeout
            if args.timeout is not None
            else float(payload.get("timeout", qemu_prompt_bench.DEFAULT_TIMEOUT_SECONDS))
        )
        warmup = args.warmup if args.warmup is not None else int(payload.get("warmup", 0))
        repeat = args.repeat if args.repeat is not None else int(payload.get("repeat", 1))
        if warmup < 0:
            raise ValueError("--warmup must be >= 0")
        if repeat < 1:
            raise ValueError("--repeat must be >= 1")
        max_suite_cv_pct = (
            args.max_suite_cv_pct
            if args.max_suite_cv_pct is not None
            else payload.get("max_suite_cv_pct")
        )
        max_prompt_cv_pct = (
            args.max_prompt_cv_pct
            if args.max_prompt_cv_pct is not None
            else payload.get("max_prompt_cv_pct")
        )
        max_suite_iqr_pct = (
            args.max_suite_iqr_pct
            if args.max_suite_iqr_pct is not None
            else payload.get("max_suite_iqr_pct")
        )
        max_prompt_iqr_pct = (
            args.max_prompt_iqr_pct
            if args.max_prompt_iqr_pct is not None
            else payload.get("max_prompt_iqr_pct")
        )
        max_suite_cv_pct = float(max_suite_cv_pct) if max_suite_cv_pct is not None else None
        max_prompt_cv_pct = float(max_prompt_cv_pct) if max_prompt_cv_pct is not None else None
        max_suite_iqr_pct = float(max_suite_iqr_pct) if max_suite_iqr_pct is not None else None
        max_prompt_iqr_pct = float(max_prompt_iqr_pct) if max_prompt_iqr_pct is not None else None
        if max_suite_cv_pct is not None and max_suite_cv_pct < 0:
            raise ValueError("--max-suite-cv-pct must be >= 0")
        if max_prompt_cv_pct is not None and max_prompt_cv_pct < 0:
            raise ValueError("--max-prompt-cv-pct must be >= 0")
        if max_suite_iqr_pct is not None and max_suite_iqr_pct < 0:
            raise ValueError("--max-suite-iqr-pct must be >= 0")
        if max_prompt_iqr_pct is not None and max_prompt_iqr_pct < 0:
            raise ValueError("--max-prompt-iqr-pct must be >= 0")

        matrix_base_dir = args.matrix.resolve().parent
        global_qemu_args = matrix_qemu_args(payload, "matrix", matrix_base_dir)
        cells = expand_cells(payload, base_dir=matrix_base_dir)
        matrix_dir = args.output_dir / f"bench_matrix_{iso_now().replace(':', '').replace('-', '')}"
        results = [
            run_cell(
                cell=cell,
                image=image,
                prompts=prompts,
                qemu_bin=qemu_bin,
                global_qemu_args=global_qemu_args,
                timeout=timeout,
                warmup=warmup,
                repeat=repeat,
                max_suite_cv_pct=max_suite_cv_pct,
                max_prompt_cv_pct=max_prompt_cv_pct,
                max_suite_iqr_pct=max_suite_iqr_pct,
                max_prompt_iqr_pct=max_prompt_iqr_pct,
                matrix_dir=matrix_dir,
                dry_run=args.dry_run,
            )
            for cell in cells
        ]
        output = write_matrix_report(
            matrix_name=matrix_name,
            matrix_file=args.matrix,
            output_dir=args.output_dir,
            matrix_dir=matrix_dir,
            cells=results,
            dry_run=args.dry_run,
            max_suite_cv_pct=max_suite_cv_pct,
            max_prompt_cv_pct=max_prompt_cv_pct,
            max_suite_iqr_pct=max_suite_iqr_pct,
            max_prompt_iqr_pct=max_prompt_iqr_pct,
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    status = "planned" if args.dry_run else ("pass" if all(cell.status == "pass" for cell in results) else "fail")
    print(f"wrote_json={output}")
    print(f"status={status}")
    print(f"cells={len(results)}")
    return 0 if status in {"pass", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
