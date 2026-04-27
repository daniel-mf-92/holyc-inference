#!/usr/bin/env python3
"""Tests for host-side benchmark result indexing."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import bench_result_index


def test_indexes_qemu_prompt_report_with_airgap_status(tmp_path: Path) -> None:
    report = tmp_path / "qemu_prompt_bench_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "a" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 123.0, "memory_bytes_max": 4096},
                "warmups": [],
                "benchmarks": [
                    {
                        "benchmark": "qemu_prompt",
                        "profile": "secure",
                        "model": "tiny",
                        "quantization": "Q4_0",
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([tmp_path])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.artifact_type == "qemu_prompt"
    assert summary.command_airgap_status == "pass"
    assert summary.profile == "secure"
    assert summary.quantization == "Q4_0"
    assert summary.median_tok_per_s == 123.0
    assert summary.max_memory_bytes == 4096


def test_indexes_matrix_cells_and_flags_network_devices(tmp_path: Path) -> None:
    report = tmp_path / "bench_matrix_latest.json"
    report.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "cells": [
                    {
                        "status": "pass",
                        "profile": "unsafe",
                        "model": "tiny",
                        "quantization": "Q8_0",
                        "prompt_suite_sha256": "b" * 64,
                        "prompts": 2,
                        "measured_runs": 4,
                        "warmup_runs": 1,
                        "median_tok_per_s": 99.5,
                        "max_memory_bytes": 8192,
                        "command": [
                            "qemu-system-x86_64",
                            "-device",
                            "virtio-net-pci",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summaries = bench_result_index.load_summaries([report])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.artifact_type == "bench_matrix_cell"
    assert summary.command_airgap_status == "fail"
    assert any("network device" in finding for finding in summary.command_findings)
    assert bench_result_index.index_status(summaries) == "fail"


def test_reports_prompt_suite_drift_for_comparable_artifacts(tmp_path: Path) -> None:
    for index, suite_hash in enumerate(("a" * 64, "b" * 64), 1):
        report_dir = tmp_path / f"run{index}"
        report_dir.mkdir()
        (report_dir / "qemu_prompt_bench_latest.json").write_text(
            json.dumps(
                {
                    "generated_at": f"2026-04-27T20:0{index}:00Z",
                    "status": "pass",
                    "prompt_suite": {"suite_sha256": suite_hash, "prompt_count": 1},
                    "suite_summary": {"tok_per_s_median": 100 + index},
                    "benchmarks": [
                        {
                            "profile": "secure",
                            "model": "tiny",
                            "quantization": "Q4_0",
                            "command": [
                                "qemu-system-x86_64",
                                "-nic",
                                "none",
                                "-drive",
                                "file=TempleOS.img,format=raw,if=ide",
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    drift = bench_result_index.prompt_suite_drift(bench_result_index.load_summaries([tmp_path]))

    assert len(drift) == 1
    assert drift[0].key == "secure/tiny/Q4_0"
    assert drift[0].hashes == ["a" * 64, "b" * 64]


def test_cli_writes_json_markdown_and_csv(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "qemu_prompt_bench_latest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-27T20:00:00Z",
                "status": "pass",
                "prompt_suite": {"suite_sha256": "c" * 64, "prompt_count": 1},
                "suite_summary": {"tok_per_s_median": 160.0},
                "benchmarks": [
                    {
                        "profile": "synthetic",
                        "model": "smoke",
                        "quantization": "Q4_0",
                        "command": [
                            "qemu-system-x86_64",
                            "-nic",
                            "none",
                            "-drive",
                            "file=TempleOS.img,format=raw,if=ide",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    status = bench_result_index.main(
        ["--input", str(input_dir), "--output-dir", str(output_dir), "--fail-on-airgap"]
    )

    assert status == 0
    payload = json.loads((output_dir / "bench_result_index_latest.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "bench_result_index_latest.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output_dir / "bench_result_index_latest.csv").open(encoding="utf-8")))

    assert payload["status"] == "pass"
    assert payload["artifacts"][0]["prompt_suite_sha256"] == "c" * 64
    assert payload["prompt_suite_drift"] == []
    assert "Benchmark Result Index" in markdown
    assert "Prompt suite drift: none detected." in markdown
    assert rows[0]["command_airgap_status"] == "pass"
