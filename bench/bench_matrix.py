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
    status: str
    output_dir: str
    report: str
    command: list[str]
    prompts: int
    measured_runs: int
    warmup_runs: int
    median_tok_per_s: float | None
    max_memory_bytes: int | None


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


def axis_items(payload: dict[str, Any], key: str, default_name: str = "default") -> list[MatrixAxisItem]:
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
                qemu_args=as_args(row.get("qemu_args"), f"{label}.qemu_args"),
            )
        )
    return items


def expand_cells(payload: dict[str, Any]) -> list[MatrixCell]:
    profiles = axis_items(payload, "profiles")
    models = axis_items(payload, "models", default_name="")
    quantizations = axis_items(payload, "quantizations", default_name="")
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
    matrix_dir: Path,
    dry_run: bool,
) -> MatrixCellResult:
    command = cell_command(qemu_bin, image, global_qemu_args, cell)
    prompt_count = len(qemu_prompt_bench.load_prompt_cases(prompts))
    cell_output_dir = matrix_dir / cell.slug
    report_path = cell_output_dir / "qemu_prompt_bench_latest.json"

    if dry_run:
        return MatrixCellResult(
            profile=cell.profile.name,
            model=cell.model.name,
            quantization=cell.quantization.name,
            status="planned",
            output_dir=str(cell_output_dir),
            report=str(report_path),
            command=command,
            prompts=prompt_count,
            measured_runs=0,
            warmup_runs=0,
            median_tok_per_s=None,
            max_memory_bytes=None,
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
    for arg in global_qemu_args + cell.profile.qemu_args + cell.model.qemu_args + cell.quantization.qemu_args:
        argv.append(f"--qemu-arg={arg}")

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        status_code = qemu_prompt_bench.main(argv)
    if status_code != 0:
        raise RuntimeError(f"benchmark cell {cell.slug} failed with status {status_code}: {stdout.getvalue()}")

    report = load_report(report_path)
    return MatrixCellResult(
        profile=cell.profile.name,
        model=cell.model.name,
        quantization=cell.quantization.name,
        status=str(report.get("status", "unknown")),
        output_dir=str(cell_output_dir),
        report=str(report_path),
        command=command,
        prompts=prompt_count,
        measured_runs=len(report.get("benchmarks", [])),
        warmup_runs=len(report.get("warmups", [])),
        median_tok_per_s=median_cell_tok_per_s(report),
        max_memory_bytes=max_cell_memory_bytes(report),
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Matrix",
        "",
        f"Generated: {report['generated_at']}",
        f"Status: {report['status']}",
        f"Matrix: {report['matrix_name']}",
        f"Cells: {len(report['cells'])}",
        "",
        "## Cells",
        "",
    ]
    if report["cells"]:
        lines.append("| Profile | Model | Quantization | Status | Prompts | Runs | Warmups | Median tok/s | Max memory bytes |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for cell in report["cells"]:
            median = cell["median_tok_per_s"]
            memory = cell["max_memory_bytes"]
            median_cell = f"{median:.3f}" if median is not None else "-"
            memory_cell = str(memory) if memory is not None else "-"
            lines.append(
                f"| {cell['profile']} | {cell['model']} | {cell['quantization']} | {cell['status']} | "
                f"{cell['prompts']} | {cell['measured_runs']} | {cell['warmup_runs']} | "
                f"{median_cell} | {memory_cell} |"
            )
    else:
        lines.append("No matrix cells were expanded.")
    return "\n".join(lines) + "\n"


def write_matrix_report(
    *,
    matrix_name: str,
    matrix_file: Path,
    output_dir: Path,
    matrix_dir: Path,
    cells: list[MatrixCellResult],
    dry_run: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": iso_now(),
        "status": "planned" if dry_run else ("pass" if all(cell.status == "pass" for cell in cells) else "fail"),
        "matrix_name": matrix_name,
        "matrix_file": str(matrix_file),
        "matrix_dir": str(matrix_dir),
        "dry_run": dry_run,
        "cells": [asdict(cell) for cell in cells],
    }
    json_path = output_dir / "bench_matrix_latest.json"
    md_path = output_dir / "bench_matrix_latest.md"
    csv_path = output_dir / "bench_matrix_latest.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")
    write_matrix_csv(cells, csv_path)
    return json_path


def write_matrix_csv(cells: list[MatrixCellResult], path: Path) -> None:
    fields = [
        "profile",
        "model",
        "quantization",
        "status",
        "output_dir",
        "report",
        "prompts",
        "measured_runs",
        "warmup_runs",
        "median_tok_per_s",
        "max_memory_bytes",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for cell in cells:
            row = asdict(cell)
            row["command"] = json.dumps(cell.command, separators=(",", ":"))
            writer.writerow({field: row[field] for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="Local JSON benchmark matrix")
    parser.add_argument("--image", type=Path, help="Override matrix image path")
    parser.add_argument("--prompts", type=Path, help="Override matrix prompts path")
    parser.add_argument("--qemu-bin", help="Override matrix QEMU binary")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeat", type=int)
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

        global_qemu_args = as_args(payload.get("qemu_args"), "qemu_args")
        cells = expand_cells(payload)
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
