#!/usr/bin/env python3
"""Tests for host-side benchmark matrix orchestration."""

from __future__ import annotations

import json
import sys
import csv
import xml.etree.ElementTree as ET
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
    qemu_args = tmp_path / "qemu.args"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")
    qemu_args.write_text("-m 256M\n", encoding="utf-8")
    matrix.write_text(
        json.dumps(
            {
                "name": "dry-run",
                "image": str(tmp_path / "TempleOS.img"),
                "prompts": str(prompts),
                "qemu_bin": "qemu-system-x86_64",
                "qemu_args_files": ["qemu.args"],
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
    junit_root = ET.parse(output_dir / "bench_matrix_junit_latest.xml").getroot()
    cell = report["cells"][0]
    assert report["status"] == "planned"
    assert cell["status"] == "planned"
    assert csv_rows[0]["status"] == "planned"
    assert junit_root.attrib["name"] == "holyc_bench_matrix"
    assert junit_root.attrib["tests"] == "1"
    assert junit_root.attrib["failures"] == "0"
    assert len(cell["prompt_suite_sha256"]) == 64
    assert csv_rows[0]["prompt_suite_sha256"] == cell["prompt_suite_sha256"]
    assert cell["command_sha256"] == bench_matrix.qemu_prompt_bench.command_hash(cell["command"])
    assert csv_rows[0]["command_sha256"] == cell["command_sha256"]
    assert json.loads(csv_rows[0]["command"])[1:3] == ["-nic", "none"]
    assert cell["command"][1:3] == ["-nic", "none"]
    assert "-m" in cell["command"]
    assert "-smp" in cell["command"]


def test_matrix_axis_qemu_args_files_are_air_gap_checked(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    prompts = tmp_path / "prompts.jsonl"
    output_dir = tmp_path / "results"
    prompts.write_text('{"prompt_id":"one","prompt":"A"}\n', encoding="utf-8")
    (tmp_path / "profile.args").write_text("-smp 2\n", encoding="utf-8")
    (tmp_path / "bad.args").write_text("-netdev user,id=n0\n", encoding="utf-8")
    matrix.write_text(
        json.dumps(
            {
                "name": "dry-run",
                "image": str(tmp_path / "TempleOS.img"),
                "prompts": str(prompts),
                "qemu_bin": "qemu-system-x86_64",
                "profiles": [{"name": "secure", "qemu_args_file": "profile.args"}],
                "models": ["tiny"],
                "quantizations": [{"name": "Q4_0", "qemu_args_file": "bad.args"}],
            }
        ),
        encoding="utf-8",
    )

    status = bench_matrix.main(["--matrix", str(matrix), "--output-dir", str(output_dir), "--dry-run"])

    assert status == 2


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
    summary_rows = list(
        csv.DictReader((output_dir / "bench_matrix_summary_latest.csv").open(newline="", encoding="utf-8"))
    )
    junit_root = ET.parse(output_dir / "bench_matrix_junit_latest.xml").getroot()

    assert report["status"] == "pass"
    assert len(report["cells"]) == 2
    assert len(csv_rows) == 2
    assert junit_root.attrib["tests"] == "2"
    assert junit_root.attrib["failures"] == "0"
    assert {cell["quantization"] for cell in report["cells"]} == {"Q4_0", "Q8_0"}
    assert {row["quantization"] for row in csv_rows} == {"Q4_0", "Q8_0"}
    assert all(cell["measured_runs"] == 4 for cell in report["cells"])
    assert all(len(cell["prompt_suite_sha256"]) == 64 for cell in report["cells"])
    assert all(len(cell["command_sha256"]) == 64 for cell in report["cells"])
    assert all(
        row["prompt_suite_sha256"] == report["cells"][index]["prompt_suite_sha256"]
        for index, row in enumerate(csv_rows)
    )
    assert all(
        row["command_sha256"] == report["cells"][index]["command_sha256"]
        for index, row in enumerate(csv_rows)
    )
    assert all(row["measured_runs"] == "4" for row in csv_rows)
    assert all(cell["warmup_runs"] == 2 for cell in report["cells"])
    assert all(cell["median_tok_per_s"] == 160.0 for cell in report["cells"])
    assert all(cell["host_child_cpu_us_median"] is not None for cell in report["cells"])
    assert all(cell["host_child_cpu_pct_median"] is not None for cell in report["cells"])
    assert all(cell["host_child_tok_per_cpu_s_median"] is not None for cell in report["cells"])
    assert all(row["host_child_cpu_us_median"] != "" for row in csv_rows)
    assert all(row["host_child_cpu_pct_median"] != "" for row in csv_rows)
    assert all(row["host_child_tok_per_cpu_s_median"] != "" for row in csv_rows)
    assert summary_rows[0]["scope"] == "matrix"
    assert summary_rows[0]["host_child_tok_per_cpu_s_median"] != ""
    assert all(cell["command"][1:3] == ["-nic", "none"] for cell in report["cells"])
    assert "Benchmark Matrix" in markdown
    assert "Host child CPU %" in markdown
    assert "Host child tok/CPU s" in markdown
    assert "Prompt suite" in markdown
    assert "Command SHA256" in markdown


def test_junit_marks_failed_matrix_cells(tmp_path: Path) -> None:
    output = tmp_path / "bench_matrix_junit_latest.xml"
    cells = [
        bench_matrix.MatrixCellResult(
            profile="secure",
            model="tiny",
            quantization="Q4_0",
            commit="a" * 12,
            status="pass",
            output_dir=str(tmp_path / "pass"),
            report=str(tmp_path / "pass" / "qemu_prompt_bench_latest.json"),
            command=["qemu-system-x86_64", "-nic", "none"],
            command_sha256="c" * 64,
            prompts=1,
            prompt_suite_sha256="a" * 64,
            prompt_bytes_total=16,
            prompt_bytes_min=16,
            prompt_bytes_max=16,
            measured_runs=3,
            warmup_runs=1,
            median_tok_per_s=120.0,
            wall_tok_per_s_median=None,
            ttft_us_p95=None,
            host_overhead_pct_median=None,
            host_child_cpu_us_median=None,
            host_child_cpu_pct_median=None,
            host_child_tok_per_cpu_s_median=None,
            host_child_peak_rss_bytes_max=None,
            us_per_token_median=None,
            wall_us_per_token_median=None,
            max_memory_bytes=4096,
            variability_findings=0,
        ),
        bench_matrix.MatrixCellResult(
            profile="secure",
            model="tiny",
            quantization="Q8_0",
            commit="a" * 12,
            status="fail",
            output_dir=str(tmp_path / "fail"),
            report=str(tmp_path / "fail" / "qemu_prompt_bench_latest.json"),
            command=["qemu-system-x86_64", "-nic", "none"],
            command_sha256="d" * 64,
            prompts=1,
            prompt_suite_sha256="b" * 64,
            prompt_bytes_total=16,
            prompt_bytes_min=16,
            prompt_bytes_max=16,
            measured_runs=3,
            warmup_runs=1,
            median_tok_per_s=90.0,
            wall_tok_per_s_median=None,
            ttft_us_p95=None,
            host_overhead_pct_median=None,
            host_child_cpu_us_median=12000.0,
            host_child_cpu_pct_median=85.0,
            host_child_tok_per_cpu_s_median=750.0,
            host_child_peak_rss_bytes_max=123456,
            us_per_token_median=None,
            wall_us_per_token_median=None,
            max_memory_bytes=8192,
            variability_findings=1,
        ),
    ]

    bench_matrix.write_matrix_junit(cells, output)

    root = ET.parse(output).getroot()
    failure = root.find("./testcase/failure")
    assert root.attrib["name"] == "holyc_bench_matrix"
    assert root.attrib["tests"] == "2"
    assert root.attrib["failures"] == "1"
    assert failure is not None
    assert failure.attrib["type"] == "benchmark_matrix_cell_failure"
    assert "variability_findings=1" in (failure.text or "")
    assert "command_sha256=" in (failure.text or "")
    assert "host_child_cpu_pct_median=85.0" in (failure.text or "")
    assert "host_child_tok_per_cpu_s_median=750.0" in (failure.text or "")
