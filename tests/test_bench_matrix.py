#!/usr/bin/env python3
"""Tests for host-side benchmark matrix orchestration."""

from __future__ import annotations

import json
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench_matrix


def test_expand_cells_cross_product() -> None:
    cells = bench_matrix.expand_cells(
        {
            "profiles": ["dev", "secure"],
            "models": [{"name": "tiny"}],
            "quantizations": ["Q4_0", "Q8_0"],
        }
    )

    assert [cell.slug for cell in cells] == [
        "dev_tiny_Q4_0",
        "dev_tiny_Q8_0",
        "secure_tiny_Q4_0",
        "secure_tiny_Q8_0",
    ]


def test_dry_run_writes_planned_air_gapped_commands(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    prompts = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")
    matrix.write_text(
        json.dumps(
            {
                "name": "dry-run",
                "image": str(tmp_path / "TempleOS.img"),
                "prompts": str(prompts),
                "qemu_bin": "qemu-system-x86_64",
                "qemu_args": ["-m", "256M"],
                "profiles": [{"name": "secure", "qemu_args": ["-smp", "1"]}],
                "models": ["tiny"],
                "quantizations": ["Q4_0"],
            }
        ),
        encoding="utf-8",
    )

    status = bench_matrix.main(["--matrix", str(matrix), "--output-dir", str(output_dir), "--dry-run"])

    assert status == 0
    report = json.loads((output_dir / "bench_matrix_latest.json").read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((output_dir / "bench_matrix_latest.csv").open(newline="", encoding="utf-8")))
    cell = report["cells"][0]
    assert report["status"] == "planned"
    assert cell["status"] == "planned"
    assert csv_rows[0]["status"] == "planned"
    assert len(cell["prompt_suite_sha256"]) == 64
    assert csv_rows[0]["prompt_suite_sha256"] == cell["prompt_suite_sha256"]
    assert json.loads(csv_rows[0]["command"])[1:3] == ["-nic", "none"]
    assert cell["command"][1:3] == ["-nic", "none"]
    assert "-m" in cell["command"]
    assert "-smp" in cell["command"]


def test_cli_runs_synthetic_matrix(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    output_dir = tmp_path / "results"
    matrix.write_text(
        json.dumps(
            {
                "name": "synthetic",
                "image": str(tmp_path / "TempleOS.img"),
                "prompts": str(ROOT / "bench" / "prompts" / "smoke.jsonl"),
                "qemu_bin": str(ROOT / "bench" / "fixtures" / "qemu_synthetic_bench.py"),
                "repeat": 2,
                "warmup": 1,
                "profiles": [{"name": "ci-airgap-smoke"}],
                "models": ["synthetic-smoke"],
                "quantizations": ["Q4_0", "Q8_0"],
            }
        ),
        encoding="utf-8",
    )

    status = bench_matrix.main(["--matrix", str(matrix), "--output-dir", str(output_dir)])

    assert status == 0
    report = json.loads((output_dir / "bench_matrix_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "bench_matrix_latest.md").read_text(encoding="utf-8")
    csv_rows = list(csv.DictReader((output_dir / "bench_matrix_latest.csv").open(newline="", encoding="utf-8")))

    assert report["status"] == "pass"
    assert len(report["cells"]) == 2
    assert len(csv_rows) == 2
    assert {cell["quantization"] for cell in report["cells"]} == {"Q4_0", "Q8_0"}
    assert {row["quantization"] for row in csv_rows} == {"Q4_0", "Q8_0"}
    assert all(cell["measured_runs"] == 4 for cell in report["cells"])
    assert all(len(cell["prompt_suite_sha256"]) == 64 for cell in report["cells"])
    assert all(
        row["prompt_suite_sha256"] == report["cells"][index]["prompt_suite_sha256"]
        for index, row in enumerate(csv_rows)
    )
    assert all(row["measured_runs"] == "4" for row in csv_rows)
    assert all(cell["warmup_runs"] == 2 for cell in report["cells"])
    assert all(cell["median_tok_per_s"] == 160.0 for cell in report["cells"])
    assert all(cell["command"][1:3] == ["-nic", "none"] for cell in report["cells"])
    assert "Benchmark Matrix" in markdown
    assert "Prompt suite" in markdown
